"""GatedDeltaNet implementation based on Qwen3.5 MoE - Training only version.

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GatedDeltaNetConfig

# Try to import fast implementations
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    FusedRMSNormGated = None
    chunk_gated_delta_rule = None


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    """Torch implementation of chunk gated delta rule (training only).
    
    Args:
        query: (batch, num_heads, seq_len, head_dim)
        key: (batch, num_heads, seq_len, head_dim)
        value: (batch, num_heads, seq_len, head_dim_v)
        g: (batch, num_heads, seq_len) - gate values
        beta: (batch, num_heads, seq_len) - beta values
        chunk_size: chunk size for processing
        use_qk_l2norm_in_kernel: whether to use L2 norm for Q/K
    
    Returns:
        output: (batch, seq_len, num_heads, head_dim_v)
    """
    initial_dtype = query.dtype
    
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    # Convert to float32 for numerical stability
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) 
        for x in (query, key, value, beta, g)
    ]
    
    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    
    # Pad to multiple of chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    
    total_seq_len = seq_len + pad_size
    scale = 1.0 / math.sqrt(query.shape[-1])
    query = query * scale
    
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    
    # Reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(batch_size, num_heads, -1, chunk_size)
    
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    
    # Compute chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    
    # Compute attention within chunk
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=query.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    
    # Initialize recurrent state
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
    
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    
    # Process chunks
    num_chunks = total_seq_len // chunk_size
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_chunk @ v_new
        
        # Update recurrent state
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    
    # Reshape output
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    
    return core_attn_out


class RMSNormGated(nn.Module):
    """RMS Norm with gating."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet layer for training (no cache support).
    
    This is a simplified implementation focused on training only.
    All cache-related parameters and logic have been removed.
    """
    
    def __init__(self, config: GatedDeltaNetConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.num_key_heads
        self.num_v_heads = config.num_value_heads
        self.head_k_dim = config.key_head_dim
        self.head_v_dim = config.value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.conv_kernel_size
        
        # Convolution for QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        
        # Time step parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Normalization
        if FusedRMSNormGated is not None:
            self.norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=config.rms_norm_eps,
                activation=config.hidden_act,
            )
        else:
            self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        
        # Output projection
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        
        # Input projections
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        
        # Select implementation
        self.chunk_gated_delta_rule_fn = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: optional attention mask (not used in training mode)
        
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Input projections
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (batch, conv_dim, seq_len)
        
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        
        # Causal convolution
        if causal_conv1d_fn is not None:
            mixed_qkv = causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias if hasattr(self.conv1d, 'bias') else None,
                activation=self.config.hidden_act,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (batch, seq_len, conv_dim)
        
        # Split QKV
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        
        # Compute gating
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        
        # Handle GQA (grouped query attention)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        
        # Transpose for attention computation (batch, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Apply gated delta rule
        core_attn_out = self.chunk_gated_delta_rule_fn(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=self.config.chunk_size,
            use_qk_l2norm_in_kernel=self.config.use_qk_l2norm,
        )
        
        # Gated normalization
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        
        # Output projection
        output = self.out_proj(core_attn_out)
        
        return output
