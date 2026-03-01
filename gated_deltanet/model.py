"""GatedDeltaNet implementation based on Qwen3.5 MoE - Training only version.

Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GatedDeltaNetConfig

# Import fast implementations (required)
from causal_conv1d import causal_conv1d_fn
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


class RMSNormGated(nn.Module):
    """RMS Norm with gating (fallback if FusedRMSNormGated not available)."""
    
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
    
    This implementation requires:
    - causal-conv1d: for efficient causal convolution
    - flash-linear-attention: for optimized gated delta rule
    
    Args:
        config: GatedDeltaNetConfig
        layer_idx: layer index
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
        
        # Normalization (prefer fused version)
        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            activation=config.hidden_act,
        ) if FusedRMSNormGated is not None else RMSNormGated(
            self.head_v_dim, eps=config.rms_norm_eps
        )
        
        # Output projection
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        
        # Input projections
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: optional attention mask (not used)
        
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
        
        # Causal convolution using causal-conv1d
        mixed_qkv = causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias if hasattr(self.conv1d, 'bias') else None,
            activation=self.config.hidden_act,
        )
        
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
        
        # Apply gated delta rule using flash-linear-attention
        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
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
