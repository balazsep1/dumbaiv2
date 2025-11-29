from typing import Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel



@dataclass
class HierarchicalReasoningModel_ACTV2InnerCarry:
    z_H: torch.Tensor  # [num_experts, B, d] - per-position state
    z_L: torch.Tensor
    z_mem: torch.Tensor
    evolution_mask: torch.Tensor  # [num_policy_slots, B] - per-batch mask
    steps: Optional[torch.Tensor] = None

@dataclass
class HierarchicalReasoningModel_ACTV2Carry:
    inner_carry: HierarchicalReasoningModel_ACTV2InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class MurderTreeV2Config(BaseModel):
    vocab_size: int
    seq_len: int
    hidden_size: int = 1024
    num_heads: int = 16
    rope_theta: float = 100000.0
    forward_dtype: str = "bfloat16"
    dropout: float = 0.1
    use_gradient_checkpointing: bool = False
    halt_bias_init: float = -6.0
    
    tier1_dim: int = 384
    tier1_depth: int = 2
    tier1_count: int = 32
    tier2_dim: int = 768
    tier2_depth: int = 4
    tier2_count: int = 8
    tier3_dim: int = 1024
    tier3_depth: int = 8
    tier3_count: int = 3

    verifier_depth: int = 32
    slow_verify_every: int = 8

    max_steps: int = 64
    ponder_cost_weight: float = 0.01
    evolution_trigger_threshold: float = 0.35
    num_policy_slots: int = 12
    evolution_candidates: int = 8
    rms_eps: float = 1e-5
    allow_inference_evolution: bool = False
    inference_evolution_threshold: float = 0.5
    inference_evolution_candidates: int = 4
    max_inference_evolutions: int = 10
    inference_noise_scale: float = 0.01
    policy_injection_weight: float = 0.3


class CosSin:
    def __init__(self, cos, sin):
        self.cos = cos
        self.sin = sin

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        if device is None:
            device = torch.device('cpu')
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        inv_freq.requires_grad_(False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device=None):
        if device is None:
            inv_freq_buf = self.get_buffer("inv_freq")
            device = inv_freq_buf.device if inv_freq_buf is not None else torch.device('cpu')
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        inv_freq = self.inv_freq.to(device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        cos_cached.requires_grad_(False)
        sin_cached.requires_grad_(False)
        
        if hasattr(self, 'cos_cached'):
            delattr(self, 'cos_cached')
        if hasattr(self, 'sin_cached'):
            delattr(self, 'sin_cached')
        
        self.register_buffer("cos_cached", cos_cached, persistent=True)
        self.register_buffer("sin_cached", sin_cached, persistent=True)

    def forward(self, seq_len):
        cos_cached = self.get_buffer("cos_cached")
        sin_cached = self.get_buffer("sin_cached")
        if cos_cached is None or sin_cached is None or seq_len > cos_cached.shape[2]:
            device = cos_cached.device if cos_cached is not None else torch.device('cpu')
            self._set_cos_sin_cache(seq_len, device=device)
            cos_cached = self.get_buffer("cos_cached")
            sin_cached = self.get_buffer("sin_cached")
        return CosSin(cos_cached[:, :, :seq_len, :], sin_cached[:, :, :seq_len, :])

def apply_rotary_emb(q, k, cos, sin):
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    
    q_dim = q.size(-1)
    cos_dim = cos.size(-1)
    
    if cos_dim != q_dim:
        raise ValueError(f"RoPE dimension mismatch: q/k have dim {q_dim} but cos/sin have dim {cos_dim}. "
                        f"This indicates a configuration error - head_dim must match rope_dim exactly.")
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def rotate_half(x):
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"rotate_half requires even dimension, got {x.shape[-1]}")
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class DTypeAwareLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, forward_dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias)
        self.forward_dtype = forward_dtype
        self._cached_weight = None
        self._cached_bias = None
        self._weight_device = None

    def forward(self, input):
        if self._cached_weight is None or self._weight_device != self.weight.device:
            self._cached_weight = self.weight.to(self.forward_dtype)
            self._cached_bias = self.bias.to(self.forward_dtype) if self.bias is not None else None
            self._weight_device = self.weight.device
        
        if input.dtype != self.forward_dtype:
            input = input.to(self.forward_dtype)
        return F.linear(input, self._cached_weight, self._cached_bias)

class DTypeAwareEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, forward_dtype=torch.bfloat16, padding_idx=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.forward_dtype = forward_dtype
        self._cached_weight = None
        self._weight_device = None
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight.data[self.padding_idx].zero_()

    def forward(self, input):
        if self._cached_weight is None or self._weight_device != self.weight.device:
            self._cached_weight = self.weight.to(self.forward_dtype)
            self._weight_device = self.weight.device
        return F.embedding(input, self._cached_weight, self.padding_idx)

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, causal=False, forward_dtype=torch.bfloat16, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        self.forward_dtype = forward_dtype

        self.q_proj = DTypeAwareLinear(hidden_size, hidden_size, bias=False, forward_dtype=forward_dtype)
        self.k_proj = DTypeAwareLinear(hidden_size, hidden_size, bias=False, forward_dtype=forward_dtype)
        self.v_proj = DTypeAwareLinear(hidden_size, hidden_size, bias=False, forward_dtype=forward_dtype)
        self.o_proj = DTypeAwareLinear(hidden_size, hidden_size, bias=False, forward_dtype=forward_dtype)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", None, persistent=True)

    def forward(self, hidden_states, cos_sin=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cos_sin is not None:
            q, k = apply_rotary_emb(q, k, cos_sin.cos, cos_sin.sin)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if self.causal:
            causal_mask = self.get_buffer("causal_mask")
            if causal_mask is None or causal_mask.shape[-1] < seq_len:
                mask = torch.full((1, 1, seq_len, seq_len), float('-1e4'), device=hidden_states.device, dtype=torch.float32)
                mask = torch.triu(mask, diagonal=1)
                mask.requires_grad_(False)
                self.register_buffer("causal_mask", mask, persistent=True)
                causal_mask = mask
            causal_mask = causal_mask[:, :, :seq_len, :seq_len].to(attn_weights.dtype)
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, input_size, output_size, expansion=2.0, forward_dtype=torch.bfloat16, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.expanded_size = int(input_size * expansion)
        self.forward_dtype = forward_dtype

        self.gate_proj = DTypeAwareLinear(input_size, self.expanded_size, forward_dtype=forward_dtype)
        self.up_proj = DTypeAwareLinear(input_size, self.expanded_size, forward_dtype=forward_dtype)
        self.down_proj = DTypeAwareLinear(self.expanded_size, output_size, forward_dtype=forward_dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps
        self.target_dtype = dtype
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normed = hidden_states_fp32 * torch.rsqrt(variance + self.eps)
        hidden_states_scaled = self.weight * hidden_states_normed
        return hidden_states_scaled.to(input_dtype)

def trunc_normal_init_(tensor, std=0.02, mean=0.0):
    original_dtype = tensor.dtype
    if tensor.dtype != torch.float32:
        tensor_fp32 = tensor.float()
        nn.init.trunc_normal_(tensor_fp32, mean=mean, std=std)
        tensor.copy_(tensor_fp32.to(original_dtype))
    else:
        nn.init.trunc_normal_(tensor, mean=mean, std=std)


class GriffinBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 16, forward_dtype=torch.bfloat16, dropout=0.0, causal=False, inject_dim=None):
        super().__init__()
        self.dim = dim
        self.forward_dtype = forward_dtype
        self.heads = heads
        self.head_dim = dim // heads
        
        self.norm1 = RMSNorm(dim, dtype=forward_dtype)
        self.attn = Attention(hidden_size=dim, num_heads=heads, forward_dtype=forward_dtype, dropout=dropout, causal=causal)
        self.norm2 = RMSNorm(dim, dtype=forward_dtype)
        self.mlp = SwiGLU(input_size=dim, output_size=dim, expansion=4.0, forward_dtype=forward_dtype, dropout=dropout)
        self.gate = nn.Parameter(torch.ones(dim, dtype=forward_dtype) * 0.5)
        self.inject_proj = DTypeAwareLinear(inject_dim, dim, forward_dtype=forward_dtype) if inject_dim is not None and inject_dim != dim else None

    def forward(self, h: torch.Tensor, inject: Optional[torch.Tensor] = None, cos_sin=None):
        x = self.norm1(h)
        if inject is not None:
            if inject.dim() == 2:
                inject = inject.unsqueeze(1)
            if self.inject_proj is not None:
                inject = self.inject_proj(inject)
            x = x + inject
        attn_out = self.attn(hidden_states=x, cos_sin=cos_sin)
        h = h + attn_out
        mlp_out = self.mlp(self.norm2(h))
        gate_value = torch.sigmoid(self.gate)
        gated_mlp = gate_value * mlp_out
        h = h + gated_mlp
        return h

def choose_heads(dim):
    for h in (16, 8, 12, 4, 32):
        if dim % h == 0:
            head_dim = dim // h
            if head_dim % 2 == 0 and head_dim >= 32:
                return h
    
    raise ValueError(f"Cannot find valid head count for dim={dim}. "
                    f"Dimension must be divisible by a reasonable head count (4-32) "
                    f"and result in head_dim >= 32 and even.")

class TieredProposerBank(nn.Module):
    def __init__(self, dim: int, depth: int, count: int, forward_dtype=torch.bfloat16, dropout=0.0, causal=True):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"TieredProposerBank dim must be even for RoPE, got {dim}")
        self.dim = dim
        self.depth = depth
        self.count = count
        self.forward_dtype = forward_dtype
        heads = choose_heads(dim)
        self.nets = nn.ModuleList([
            nn.ModuleList([GriffinBlock(dim, heads=heads, forward_dtype=forward_dtype, dropout=dropout, causal=causal) for _ in range(depth)])
            for _ in range(count)
        ])
        self.norm = RMSNorm(dim, dtype=forward_dtype)

    def forward(self, states: torch.Tensor, inject: torch.Tensor, cos_sin=None):
        if states.shape[0] != self.count:
            if states.shape[0] > self.count:
                states = states[:self.count]
            else:
                pad_count = self.count - states.shape[0]
                pad_shape = (pad_count,) + states.shape[1:]
                padding = torch.zeros(pad_shape, device=states.device, dtype=states.dtype)
                states = torch.cat([states, padding], dim=0)
        
        outs = []
        for i in range(self.count):
            state_input = states[i]
            if state_input.shape[-1] != self.dim:
                if state_input.shape[-1] < self.dim:
                    pad_size = self.dim - state_input.shape[-1]
                    state_input = F.pad(state_input, (0, pad_size), value=0.0)
                else:
                    state_input = state_input[..., :self.dim]
            
            h = state_input
            blocks_list = self.nets[i]
            for j in range(self.depth):
                h = blocks_list[j](h, inject=inject, cos_sin=cos_sin)  # type: ignore
            outs.append(self.norm(h))
        return torch.stack(outs, dim=0)

class MurderTreeV2Core(nn.Module):
    def __init__(self, cfg: MurderTreeV2Config):
        super().__init__()
        self.cfg = cfg
        self.dtype = getattr(torch, cfg.forward_dtype)
        self._device = None
        
        d = self.cfg.hidden_size
        
        if d % self.cfg.num_heads != 0:
            raise ValueError(f"hidden_size {d} must be divisible by num_heads {self.cfg.num_heads}")
        
        head_dim = d // self.cfg.num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim {head_dim} must be even for RoPE")
        
        if self.cfg.tier1_dim % 2 != 0 or self.cfg.tier2_dim % 2 != 0 or self.cfg.tier3_dim % 2 != 0:
            raise ValueError("All tier dimensions must be even for RoPE")
        
        self.embed = DTypeAwareEmbedding(self.cfg.vocab_size, d, forward_dtype=self.dtype, padding_idx=0)

        self.max_rope_dim = head_dim
        init_device = torch.device('cpu')
        self.rotary = RotaryEmbedding(
            dim=self.max_rope_dim,
            max_position_embeddings=self.cfg.seq_len,
            base=self.cfg.rope_theta,
            device=init_device
        )

        self.tier1 = TieredProposerBank(self.cfg.tier1_dim, self.cfg.tier1_depth, self.cfg.tier1_count, forward_dtype=self.dtype, dropout=self.cfg.dropout, causal=True)
        self.tier2 = TieredProposerBank(self.cfg.tier2_dim, self.cfg.tier2_depth, self.cfg.tier2_count, forward_dtype=self.dtype, dropout=self.cfg.dropout, causal=True)
        self.tier3 = TieredProposerBank(self.cfg.tier3_dim, self.cfg.tier3_depth, self.cfg.tier3_count, forward_dtype=self.dtype, dropout=self.cfg.dropout, causal=True)

        self.up1 = DTypeAwareLinear(d, self.cfg.tier1_dim, forward_dtype=self.dtype)
        self.up2 = DTypeAwareLinear(d, self.cfg.tier2_dim, forward_dtype=self.dtype)
        self.up3 = DTypeAwareLinear(d, self.cfg.tier3_dim, forward_dtype=self.dtype)

        self.down1 = DTypeAwareLinear(self.cfg.tier1_dim, d, forward_dtype=self.dtype)
        self.down2 = DTypeAwareLinear(self.cfg.tier2_dim, d, forward_dtype=self.dtype)
        self.down3 = DTypeAwareLinear(self.cfg.tier3_dim, d, forward_dtype=self.dtype)

        self.verifier = nn.ModuleList([GriffinBlock(d, heads=self.cfg.num_heads, forward_dtype=self.dtype, dropout=self.cfg.dropout, causal=True) for _ in range(self.cfg.verifier_depth)])
        self.fast_critic = DTypeAwareLinear(d, 1, bias=False, forward_dtype=self.dtype)
        self.slow_critic = DTypeAwareLinear(d, 1, bias=False, forward_dtype=self.dtype)

        self.lm_head = DTypeAwareLinear(d, self.cfg.vocab_size, bias=False, forward_dtype=self.dtype)
        self.halt_head = DTypeAwareLinear(d, 2, bias=True, forward_dtype=self.dtype)
        self.evolve_head = DTypeAwareLinear(d, 1, bias=False, forward_dtype=self.dtype)
        
        for m in self.modules():
            if isinstance(m, (DTypeAwareLinear, nn.Linear)):
                trunc_normal_init_(m.weight.data, std=0.02, mean=0.0)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (DTypeAwareEmbedding, nn.Embedding)):
                trunc_normal_init_(m.weight.data, std=0.02, mean=0.0)
        
        with torch.no_grad():
            self.halt_head.bias.data.fill_(self.cfg.halt_bias_init)
        
        # WARNING: lm_head and embed share weights - replacing embed will affect lm_head
        self.lm_head.weight = self.embed.weight

    def _get_safe_dtype(self, device):
        if self.dtype == torch.bfloat16 and device.type == 'cpu':
            return torch.float32
        return self.dtype

    @property
    def device(self):
        if self._device is None:
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = torch.device('cpu')
        return self._device
    
    def to(self, *args, **kwargs):
        self._device = None
        return super().to(*args, **kwargs)

    def _ensure_rotary(self, seq_len, device):
        model_device = self.device
        if seq_len > self.rotary.max_position_embeddings:
            self.rotary._set_cos_sin_cache(seq_len, device=model_device)
        
        return self.rotary(seq_len)

    def _get_evolution_conditions(self, q_evolve, evolution_mask):
        B = q_evolve.shape[0]
        device = q_evolve.device
        
        if self.training:
            current_evolutions = evolution_mask.sum(dim=0)
            can_evolve = current_evolutions < self.cfg.num_policy_slots
            evolve_now = (q_evolve.sigmoid() > self.cfg.evolution_trigger_threshold) & can_evolve
            num_candidates = self.cfg.evolution_candidates
            noise_scale = 0.02
            use_slow_critic = True
        else:
            if self.cfg.allow_inference_evolution:
                current_evolutions = evolution_mask.sum(dim=0)
                can_evolve = current_evolutions < self.cfg.max_inference_evolutions
                evolve_now = (q_evolve.sigmoid() > self.cfg.inference_evolution_threshold) & can_evolve
                num_candidates = self.cfg.inference_evolution_candidates
                noise_scale = self.cfg.inference_noise_scale
                use_slow_critic = False
            else:
                evolve_now = torch.zeros(B, dtype=torch.bool, device=device)
                num_candidates = 0
                noise_scale = 0.0
                use_slow_critic = False
        
        return evolve_now, num_candidates, noise_scale, use_slow_critic

    def _select_evolution_slot(self, evolution_mask_b, batch_idx, current_step):
        if evolution_mask_b.shape[0] != self.cfg.num_policy_slots:
            raise ValueError(f"evolution_mask_b has wrong shape: {evolution_mask_b.shape}")
        available_slots = torch.where(~evolution_mask_b)[0]
        if available_slots.numel() > 0:
            return available_slots[0].item()
        else:
            if self.training:
                return torch.randint(0, self.cfg.num_policy_slots, (1,), device=evolution_mask_b.device).item()
            else:
                return int((current_step + batch_idx) % self.cfg.num_policy_slots)

    def _vectorized_policy_evolution(self, carry, pooled_h, evolve_now, num_candidates, noise_scale, use_slow_critic, device):
        """Enhanced policy evolution with vectorized recombination and quality-aware slot management"""
        B = pooled_h.shape[0]
        d = self.cfg.hidden_size
        safe_dtype = self._get_safe_dtype(device)
        
        evolve_mask = evolve_now
        active_counts = carry.evolution_mask.sum(dim=0)
        
        base_noise = torch.randn(num_candidates, B, d, device=device, dtype=safe_dtype) * noise_scale
        
        cands = pooled_h.unsqueeze(0) + base_noise  # [num_candidates, B, d]
        
        multi_policy_mask = (active_counts >= 2) & evolve_mask  # [B]
        
        if multi_policy_mask.any():
            multi_batch_indices = torch.where(multi_policy_mask)[0]  # [M]
            M = multi_batch_indices.shape[0]
            
            active_mask_multi = carry.evolution_mask[:, multi_policy_mask]  # [slots, M]
            active_policies_multi = carry.z_mem[:, multi_policy_mask, :]  # [slots, M, d]
            
            active_counts_multi = active_mask_multi.sum(dim=0)  # [M]
            
            parent1_idx = torch.zeros(num_candidates, M, device=device, dtype=torch.long)
            parent2_idx = torch.zeros(num_candidates, M, device=device, dtype=torch.long)
            
            for m in range(M):
                active_count = active_counts_multi[m]
                parent1_idx[:, m] = torch.randint(0, active_count, (num_candidates,), device=device)
                
                if active_count > 1:
                    parent2_temp = torch.randint(0, active_count - 1, (num_candidates,), device=device)
                    parent2_idx[:, m] = torch.where(parent2_temp >= parent1_idx[:, m], 
                                                  parent2_temp + 1, parent2_temp)
                else:
                    parent2_idx[:, m] = parent1_idx[:, m]
            
            if self.training and M > 0:
                with torch.no_grad():
                    active_scores = torch.full((active_mask_multi.shape[0], M), -float('inf'), 
                                             device=device, dtype=torch.float32)
                    
                    for m in range(M):
                        active_indices_m = torch.where(active_mask_multi[:, m])[0]
                        if active_indices_m.numel() > 0:
                            active_policies_m = active_policies_multi[active_indices_m, m, :]
                            if use_slow_critic and self.training:
                                scores_m = self.slow_critic(active_policies_m.unsqueeze(1)).squeeze()
                            else:
                                scores_m = self.fast_critic(active_policies_m.unsqueeze(1)).squeeze()
                            active_scores[active_indices_m, m] = scores_m.float()
                    
                    for m in range(M):
                        active_count = active_counts_multi[m]
                        if active_count >= 2:
                            active_indices_m = torch.where(active_mask_multi[:, m])[0]
                            scores_m = active_scores[active_indices_m, m]
                            top_k = max(1, active_count // 2)
                            _, top_local_indices = torch.topk(scores_m, top_k)
                            
                            if top_k >= 2:
                                parent1_local_idx = torch.randint(0, top_k, (num_candidates,), device=device)
                                parent1_idx[:, m] = top_local_indices[parent1_local_idx]
                                
                                parent2_local_temp = torch.randint(0, top_k - 1, (num_candidates,), device=device)
                                parent2_local_idx = torch.where(parent2_local_temp >= parent1_local_idx,
                                                               parent2_local_temp + 1, parent2_local_temp)
                                parent2_idx[:, m] = top_local_indices[parent2_local_idx]
            
            parent1_policies = torch.zeros(num_candidates, M, d, device=device, dtype=safe_dtype)
            parent2_policies = torch.zeros(num_candidates, M, d, device=device, dtype=safe_dtype)
            
            for m in range(M):
                active_indices_m = torch.where(active_mask_multi[:, m])[0]
                for c in range(num_candidates):
                    p1_slot_idx = active_indices_m[parent1_idx[c, m]]
                    p2_slot_idx = active_indices_m[parent2_idx[c, m]]
                    parent1_policies[c, m] = active_policies_multi[p1_slot_idx, m]
                    parent2_policies[c, m] = active_policies_multi[p2_slot_idx, m]
            
            parent1 = parent1_policies
            parent2 = parent2_policies
            
            alphas = torch.rand(num_candidates, M, 1, device=device, dtype=parent1.dtype)
            recombined = alphas * parent1 + (1 - alphas) * parent2
            if recombined.shape != (num_candidates, M, d):
                raise ValueError(f"recombined shape mismatch: expected {(num_candidates, M, d)}, got {recombined.shape}")
            
            cands_multi = recombined + base_noise[:, multi_policy_mask, :]
            if cands_multi.shape != (num_candidates, M, d):
                raise ValueError(f"cands_multi shape mismatch: expected {(num_candidates, M, d)}, got {cands_multi.shape}")
            
            half = num_candidates // 2
            
            cands_multi = cands_multi.to(cands.dtype)
            cands[:half, multi_policy_mask, :] = cands_multi[:half]
        
        best_policy = pooled_h.unsqueeze(0)
        cands[0] = torch.where(evolve_mask.unsqueeze(-1), best_policy[0], cands[0])
        
        cands_flat = cands.reshape(-1, d)
        
        if use_slow_critic and self.training:
            values_flat = self.slow_critic(cands_flat.unsqueeze(1)).squeeze(-1)
        else:
            values_flat = self.fast_critic(cands_flat.unsqueeze(1)).squeeze(-1)
        
        values_flat = values_flat.squeeze(-1).float()
        values = values_flat.view(num_candidates, B)
        if values.shape != (num_candidates, B):
            raise ValueError(f"values shape mismatch: expected {(num_candidates, B)}, got {values.shape}")
        
        best_values, best_indices = values.max(dim=0)
        
        batch_indices = torch.arange(B, device=device)
        new_policy = cands[best_indices, batch_indices]
        if new_policy.shape != (B, d):
            raise ValueError(f"new_policy shape mismatch: expected {(B, d)}, got {new_policy.shape}")
        
        current_step = carry.steps[0] if carry.steps is not None and carry.steps.numel() > 0 else 0
        evolve_batch_indices = torch.where(evolve_mask)[0]
        
        if evolve_batch_indices.numel() > 0:
            slot_mask = ~carry.evolution_mask[:, evolve_batch_indices]  # [slots, M]
            
            available_slots = slot_mask.long().argmax(dim=0)  # [M]
            has_available_slot = slot_mask.any(dim=0)  # [M]
            
            if has_available_slot.any():
                available_batches = evolve_batch_indices[has_available_slot]
                available_slots_for_these = available_slots[has_available_slot]
                if available_slots_for_these.shape != available_batches.shape:
                    raise ValueError(f"Slot/batch shape mismatch: {available_slots_for_these.shape} vs {available_batches.shape}")
                carry.z_mem[available_slots_for_these, available_batches] = new_policy[available_batches].detach()
                carry.evolution_mask[available_slots_for_these, available_batches] = True
            
            no_slot_batches = evolve_batch_indices[~has_available_slot]
            if no_slot_batches.numel() > 0:
                existing_policies = carry.z_mem[:, no_slot_batches]
                K = no_slot_batches.numel()
                num_slots = existing_policies.shape[0]
                
                existing_values = torch.zeros(num_slots, K, device=device, dtype=torch.float32)
                for k in range(K):
                    for slot in range(num_slots):
                        policy = existing_policies[slot, k, :].unsqueeze(0).unsqueeze(0)
                        if use_slow_critic and self.training:
                            existing_values[slot, k] = self.slow_critic(policy).squeeze()
                        else:
                            existing_values[slot, k] = self.fast_critic(policy).squeeze()
                
                worst_slot_indices = existing_values.argmin(dim=0)  # [K]
                
                new_values_for_batches = best_values[no_slot_batches]  # [K]
                worst_existing_values = existing_values[worst_slot_indices, 
                                                      torch.arange(K, device=device)]  # [K]
                
                should_replace = new_values_for_batches > worst_existing_values
                
                if should_replace.any():
                    replace_batches = no_slot_batches[should_replace]
                    replace_slots = worst_slot_indices[should_replace]
                    
                    old_policies = carry.z_mem[replace_slots, replace_batches].clone()
                    carry.z_mem[replace_slots, replace_batches] = new_policy[replace_batches].detach()
                    del old_policies
        
        return new_policy

    def forward(self, carry: HierarchicalReasoningModel_ACTV2InnerCarry, input_ids: torch.Tensor):
        B, T = input_ids.shape
        d = self.cfg.hidden_size
        device = input_ids.device
        safe_dtype = self._get_safe_dtype(device)
    
        if input_ids.dtype not in (torch.long, torch.int64):
            input_ids = input_ids.long()
        x = self.embed(input_ids) * math.sqrt(d)
        
        cos_sin = self._ensure_rotary(T, device)
    
        if carry.z_H.dim() == 4:
            h = carry.z_H.mean(0).to(self.dtype)
            if h.shape[1] != T:
                if h.shape[1] < T:
                    padding = torch.zeros(B, T - h.shape[1], d, device=device, dtype=self.dtype)
                    h = torch.cat([h, padding], dim=1)
                else:
                    h = h[:, :T, :]
        else:
            h = carry.z_H.mean(0).to(self.dtype)
            h = h.unsqueeze(1).expand(-1, T, -1)
    
        if carry.z_mem is not None and carry.evolution_mask is not None and carry.evolution_mask.any():
            active_mask_expanded = carry.evolution_mask.unsqueeze(-1)
            active_mask_expanded = active_mask_expanded.expand(carry.evolution_mask.shape[0], carry.evolution_mask.shape[1], d)
            masked_policies = carry.z_mem * active_mask_expanded.float()
            policy_counts = carry.evolution_mask.sum(dim=0).clamp(min=1)
            policy_counts = policy_counts.unsqueeze(-1)
            policy_inject = masked_policies.sum(dim=0) / policy_counts
            policy_inject = policy_inject.unsqueeze(1)
            policy_inject = policy_inject.expand(B, T, d)
            h = h + policy_inject * self.cfg.policy_injection_weight

        t1_in = self.up1(x)
        
        total_experts = self.cfg.tier1_count + self.cfg.tier2_count + self.cfg.tier3_count
        if carry.z_H.shape[0] < total_experts:
            raise ValueError(f"z_H has {carry.z_H.shape[0]} experts but need {total_experts}")
        
        if carry.z_H.dim() == 4:
            t1_states = carry.z_H[:self.cfg.tier1_count, :, 0, :]
        else:
            t1_states = carry.z_H[:self.cfg.tier1_count]
        t1_out = self.tier1(t1_states, t1_in, cos_sin=cos_sin)
        
        t1_to_hidden = self.down1(t1_out.mean(0))
        t2_in = self.up2(t1_to_hidden)
        
        if carry.z_H.dim() == 4:
            t2_states = carry.z_H[self.cfg.tier1_count:self.cfg.tier1_count + self.cfg.tier2_count, :, 0, :]
        else:
            t2_states = carry.z_H[self.cfg.tier1_count:self.cfg.tier1_count + self.cfg.tier2_count]
        t2_out = self.tier2(t2_states, t2_in, cos_sin=cos_sin)
        
        t2_to_hidden = self.down2(t2_out.mean(0))
        t3_in = self.up3(t2_to_hidden)
        
        if carry.z_H.dim() == 4:
            t3_states = carry.z_H[-self.cfg.tier3_count:, :, 0, :]
        else:
            t3_states = carry.z_H[-self.cfg.tier3_count:]
        candidates = self.tier3(t3_states, t3_in, cos_sin=cos_sin)

        verify_in = self.down3(candidates.mean(0))
        
        slow = (carry.steps is not None and carry.steps.numel() > 0 and 
                carry.steps.shape[0] > 0 and carry.steps[0] > 0 and 
                (carry.steps[0] % self.cfg.slow_verify_every == 0))
        
        verify_out: torch.Tensor = verify_in
        
        if slow and self.training:
            if self.cfg.use_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                for i in range(self.cfg.verifier_depth):
                    block = self.verifier[i]
                    result = checkpoint(
                        lambda x, blk=block, cs=cos_sin: blk(x, inject=None, cos_sin=cs),
                        verify_out,
                        use_reentrant=False
                    )
                    verify_out = result  # type: ignore
            else:
                for i in range(self.cfg.verifier_depth):
                    verify_out = self.verifier[i](verify_out, inject=None, cos_sin=cos_sin)  # type: ignore
            value = self.slow_critic(verify_out)
        else:
            value = self.fast_critic(verify_out)
    
        if h.dim() == 2:
            h = h.unsqueeze(1).expand(-1, T, -1)
        
        if verify_out.dim() == 2:
            verify_out = verify_out.unsqueeze(1).expand(-1, T, -1)
    
        new_h = h + verify_out * 0.5
    
        logits = self.lm_head(new_h)
        q = self.halt_head(new_h)
        q_halt, q_cont = q[:, :, 0], q[:, :, 1]
        q_evolve = self.evolve_head(new_h.mean(1))  # [B, 1]

        evolve_now, num_candidates, noise_scale, use_slow_critic = self._get_evolution_conditions(
            q_evolve.squeeze(-1), carry.evolution_mask
        )

        new_policy = None
        if evolve_now.any() and num_candidates > 0:
            pooled_h = new_h.mean(dim=1)  # [B, d]
            
            new_policy = self._vectorized_policy_evolution(
                carry, pooled_h, evolve_now, num_candidates, 
                noise_scale, use_slow_critic, device
            )

        t1_out_proj = self.down1(t1_out.mean(1) if t1_out.dim() == 3 else t1_out)
        t2_out_proj = self.down2(t2_out.mean(1) if t2_out.dim() == 3 else t2_out)
        t3_out_proj = self.down3(candidates.mean(1) if candidates.dim() == 3 else candidates)

        return new_h, logits, (q_halt, q_cont), value, new_policy, t1_out_proj, t2_out_proj, t3_out_proj


class HierarchicalReasoningModel_ACTV2(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.cfg = MurderTreeV2Config(**config_dict)
        self.core = MurderTreeV2Core(self.cfg)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        device = next(self.core.parameters()).device
        safe_dtype = self.core._get_safe_dtype(device)
        bs = batch["inputs"].shape[0]
        seq_len = batch["inputs"].shape[1]
        d = self.cfg.hidden_size
        total_experts = self.cfg.tier1_count + self.cfg.tier2_count + self.cfg.tier3_count

        fake_H = torch.zeros(total_experts, bs, seq_len, d, device=device, dtype=safe_dtype)
        z_mem = torch.zeros(self.cfg.num_policy_slots, bs, d, device=device, dtype=safe_dtype)
        evolution_mask = torch.zeros(self.cfg.num_policy_slots, bs, dtype=torch.bool, device=device)
        steps = torch.zeros(bs, dtype=torch.int32, device=device)

        inner = HierarchicalReasoningModel_ACTV2InnerCarry(
            z_H=fake_H,
            z_L=torch.zeros(bs, self.cfg.seq_len, d, device=device, dtype=safe_dtype),
            z_mem=z_mem,
            evolution_mask=evolution_mask,
            steps=steps,
        )

        return HierarchicalReasoningModel_ACTV2Carry(
            inner_carry=inner,
            steps=steps,
            halted=torch.zeros(bs, dtype=torch.bool, device=device),
            current_data=batch,
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV2Carry, batch: Dict[str, torch.Tensor]):
        inner = carry.inner_carry
        inner.steps = carry.steps

        new_h, logits, (q_halt, q_cont), value, new_policy, t1_out, t2_out, t3_out = self.core(inner, batch["inputs"])
        
        T = batch["inputs"].shape[1]
        if t1_out.dim() == 2:
            t1_out = t1_out.unsqueeze(2).expand(-1, -1, T, -1)
        if t2_out.dim() == 2:
            t2_out = t2_out.unsqueeze(2).expand(-1, -1, T, -1)
        if t3_out.dim() == 2:
            t3_out = t3_out.unsqueeze(2).expand(-1, -1, T, -1)
        
        new_z_H = torch.cat([t1_out, t2_out, t3_out], dim=0)

        input_ids_embed = batch["inputs"]
        if input_ids_embed.dtype not in (torch.long, torch.int64):
            input_ids_embed = input_ids_embed.long()
        embedded = self.core.embed(input_ids_embed)
        
        if embedded.shape[1] > inner.z_L.shape[1]:
            raise ValueError(f"Input sequence length {embedded.shape[1]} exceeds z_L capacity {inner.z_L.shape[1]}")
        
        decay = 0.9
        current_z_L = inner.z_L[:, :embedded.shape[1], :]
        new_z_L = decay * current_z_L + (1 - decay) * embedded
        if new_z_L.shape[1] < inner.z_L.shape[1]:
            pad_len = inner.z_L.shape[1] - new_z_L.shape[1]
            padding = torch.zeros(embedded.shape[0], pad_len, embedded.shape[2], device=embedded.device, dtype=embedded.dtype)
            new_z_L = torch.cat([new_z_L, padding], dim=1)

        new_inner = HierarchicalReasoningModel_ACTV2InnerCarry(
            z_H=new_z_H,
            z_L=new_z_L,
            z_mem=inner.z_mem,
            evolution_mask=inner.evolution_mask,
            steps=inner.steps,
        )

        carry.steps += 1
        with torch.no_grad():
            q_halt_mean = q_halt.mean(dim=1)
            q_cont_mean = q_cont.mean(dim=1)
            
            should_halt = (carry.steps >= self.cfg.max_steps) | (q_halt_mean > q_cont_mean)
            carry.halted = carry.halted | should_halt

        carry.inner_carry = new_inner
        outputs = {
            "logits": logits,
            "value": value.squeeze(-1),
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
            "loss_aux": self.cfg.ponder_cost_weight * carry.steps.float().mean() if self.training else torch.tensor(0.0),
            "active_policies": inner.evolution_mask.sum().item() if inner.evolution_mask is not None else 0,
            "new_policy": new_policy,
            "steps": carry.steps,
        }

        return carry, outputs


def test_enhanced_evolution():
    """Test the enhanced evolution system with assertions"""
    config = {
        "vocab_size": 1000, "seq_len": 32, "hidden_size": 256,
        "tier1_dim": 128, "tier1_depth": 1, "tier1_count": 2,
        "tier2_dim": 192, "tier2_depth": 1, "tier2_count": 1,
        "tier3_dim": 256, "tier3_depth": 1, "tier3_count": 1,
        "verifier_depth": 2, "max_steps": 8, "num_policy_slots": 4,
        "evolution_candidates": 4, "evolution_trigger_threshold": 0.1,
        "allow_inference_evolution": True,
        "inference_evolution_threshold": 0.3,
        "inference_evolution_candidates": 2,
        "max_inference_evolutions": 3,
    }
    
    model = HierarchicalReasoningModel_ACTV2(config)
    batch = {"inputs": torch.randint(0, 100, (3, 32))}
    
    # Test training mode
    model.train()
    carry = model.initial_carry(batch)
    
    for step in range(6):
        prev_steps = carry.steps.clone()
        carry, outputs = model(carry, batch)
        
        # Assertions for correctness
        assert outputs["logits"].shape == (3, 32, 1000), f"Wrong logits shape: {outputs['logits'].shape}"
        assert outputs["value"].shape == (3,), f"Wrong value shape: {outputs['value'].shape}"
        assert (carry.steps == prev_steps + 1).all(), "Steps not incrementing correctly"
        assert carry.inner_carry.evolution_mask.shape == (4, 3), "Wrong evolution mask shape"
    
    # Test inference mode
    model.eval()
    carry = model.initial_carry(batch)
    
    for step in range(6):
        carry, outputs = model(carry, batch)
        
        # Assertions for inference
        assert outputs["logits"].shape == (3, 32, 1000), "Wrong logits shape in inference"
        max_evolutions = carry.inner_carry.evolution_mask.sum(dim=0).max().item()
        assert max_evolutions <= 3, f"Too many inference evolutions: {max_evolutions}"
    
    return True
    return True

if __name__ == "__main__":
    test_enhanced_evolution()
