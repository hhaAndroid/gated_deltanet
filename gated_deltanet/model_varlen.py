"""GatedDeltaNet with variable length sequence support - Training only version.

This module provides efficient handling of variable-length sequences in a batch
using cu_seqlens (cumulative sequence lengths) parameter.

Requires:
    - causal-conv1d: for efficient causal convolution with seq_idx support
    - flash-linear-attention: for optimized gated delta rule with cu_seqlens
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GatedDeltaNetConfig
from .model import RMSNormGated

# Import fast implementations (required)
from causal_conv1d import causal_conv1d_fn
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


class GatedDeltaNetVarlen(nn.Module):
    """Gated DeltaNet with variable length sequence support for training.
    
    This version efficiently handles batches where sequences have different lengths
    by using cu_seqlens to indicate the boundaries of each sequence.
    
    Key features:
    - Inputs/outputs are in packed format (total_tokens, hidden_size)
    - No conversion to batch/padded format
    - Each sequence is processed independently with proper boundary handling
    
    Requires:
        - causal-conv1d with seq_idx support
        - flash-linear-attention with cu_seqlens support
    
    Args:
        config: GatedDeltaNetConfig configuration
        layer_idx: layer index (for identification purposes only)
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
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass with variable length sequences for training.
        
        Args:
            hidden_states: (total_tokens, hidden_size) packed tensor
            cu_seqlens: (batch_size + 1,) cumulative sequence lengths
            max_seqlen: maximum sequence length (computed if not provided)
        
        Returns:
            output: (total_tokens, hidden_size) packed output
        
        Example:
            >>> # Three sequences with lengths 10, 20, 15
            >>> cu_seqlens = torch.tensor([0, 10, 30, 45], dtype=torch.int32)
            >>> hidden_states = torch.randn(45, 512)  # total_tokens = 45
            >>> output = model(hidden_states, cu_seqlens)
        """
        total_tokens, _ = hidden_states.shape
        batch_size = len(cu_seqlens) - 1
        
        # Input projections - keep packed format
        mixed_qkv = self.in_proj_qkv(hidden_states)  # (total_tokens, conv_dim)
        z = self.in_proj_z(hidden_states)  # (total_tokens, value_dim)
        b = self.in_proj_b(hidden_states)  # (total_tokens, num_v_heads)
        a = self.in_proj_a(hidden_states)  # (total_tokens, num_v_heads)
        
        # Transpose for conv1d: (conv_dim, total_tokens)
        mixed_qkv = mixed_qkv.transpose(0, 1)
        
        # Create seq_idx from cu_seqlens for causal_conv1d
        seq_idx = torch.zeros(total_tokens, dtype=torch.int32, device=hidden_states.device)
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_idx[start:end] = i
        
        # Causal convolution with varlen support using seq_idx
        mixed_qkv = causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias if hasattr(self.conv1d, 'bias') else None,
            activation=self.config.hidden_act,
            seq_idx=seq_idx,
        )
        
        # Transpose back: (total_tokens, conv_dim)
        mixed_qkv = mixed_qkv.transpose(0, 1)
        
        # Split QKV
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        
        # Reshape for multi-head attention - keep packed format
        query = query.reshape(total_tokens, self.num_k_heads, self.head_k_dim)
        key = key.reshape(total_tokens, self.num_k_heads, self.head_k_dim)
        value = value.reshape(total_tokens, self.num_v_heads, self.head_v_dim)
        
        # Compute gating
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        
        # Handle GQA (grouped query attention)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
        
        # Reshape to (1, total_tokens, num_heads, dim) as required by fla
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        beta = beta.unsqueeze(0)
        g = g.unsqueeze(0)
        
        # Apply gated delta rule using flash-linear-attention with cu_seqlens
        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens.to(torch.long),
            use_qk_l2norm_in_kernel=self.config.use_qk_l2norm,
        )
        
        # Squeeze back to packed format
        core_attn_out = core_attn_out.squeeze(0)  # (total_tokens, num_heads, head_v_dim)
        
        # Gated normalization on packed format
        core_attn_out = core_attn_out.reshape(total_tokens, self.head_v_dim)
        z = z.reshape(total_tokens, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(total_tokens, -1)
        
        # Output projection
        output = self.out_proj(core_attn_out)
        
        return output
