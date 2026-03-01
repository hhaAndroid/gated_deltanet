"""GatedDeltaNet with variable length sequence support - Training only version.

This module provides efficient handling of variable-length sequences in a batch
using cu_seqlens (cumulative sequence lengths) parameter.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GatedDeltaNetConfig
from .model import RMSNormGated, torch_chunk_gated_delta_rule

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


def varlen_to_batch(x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """Convert variable-length packed tensor to padded batch tensor.
    
    Args:
        x: (total_tokens, ...) packed tensor
        cu_seqlens: (batch_size + 1,) cumulative sequence lengths
        max_seqlen: maximum sequence length in the batch
    
    Returns:
        padded: (batch_size, max_seqlen, ...) padded tensor
    """
    batch_size = len(cu_seqlens) - 1
    *rest_dims, _ = x.shape
    
    # Create output tensor
    output_shape = [batch_size, max_seqlen] + rest_dims[1:]
    padded = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
    
    # Fill in values
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        padded[i, :seq_len] = x[start:end]
    
    return padded


def batch_to_varlen(padded: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Convert padded batch tensor to variable-length packed tensor.
    
    Args:
        padded: (batch_size, max_seqlen, ...) padded tensor
        cu_seqlens: (batch_size + 1,) cumulative sequence lengths
    
    Returns:
        x: (total_tokens, ...) packed tensor
    """
    batch_size = padded.shape[0]
    *rest_dims, max_seqlen = padded.shape
    
    total_tokens = cu_seqlens[-1].item()
    output_shape = [total_tokens] + rest_dims[1:]
    x = torch.zeros(output_shape, dtype=padded.dtype, device=padded.device)
    
    # Extract values
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        x[start:end] = padded[i, :seq_len]
    
    return x


class GatedDeltaNetVarlen(nn.Module):
    """Gated DeltaNet with variable length sequence support for training.
    
    This version efficiently handles batches where sequences have different lengths
    by using cu_seqlens to indicate the boundaries of each sequence.
    
    This is a simplified implementation focused on training only (no cache support).
    
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
        
        if max_seqlen is None:
            # Compute max sequence length from cu_seqlens
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        
        # Input projections
        mixed_qkv = self.in_proj_qkv(hidden_states)  # (total_tokens, conv_dim)
        z = self.in_proj_z(hidden_states)  # (total_tokens, value_dim)
        b = self.in_proj_b(hidden_states)  # (total_tokens, num_v_heads)
        a = self.in_proj_a(hidden_states)  # (total_tokens, num_v_heads)
        
        # Convert to padded format for causal convolution
        mixed_qkv_padded = varlen_to_batch(mixed_qkv, cu_seqlens, max_seqlen)
        # (batch, max_seqlen, conv_dim) -> (batch, conv_dim, max_seqlen)
        mixed_qkv_padded = mixed_qkv_padded.transpose(1, 2)
        
        if causal_conv1d_fn is not None:
            # Use fast causal conv1d if available
            mixed_qkv_padded = causal_conv1d_fn(
                x=mixed_qkv_padded,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias if hasattr(self.conv1d, 'bias') else None,
                activation=self.config.hidden_act,
            )
        else:
            # Torch fallback
            mixed_qkv_padded = F.silu(self.conv1d(mixed_qkv_padded))
        
        # Convert back to packed format
        mixed_qkv_padded = mixed_qkv_padded.transpose(1, 2)  # (batch, max_seqlen, conv_dim)
        mixed_qkv_processed = batch_to_varlen(mixed_qkv_padded, cu_seqlens)
        
        # Split QKV
        query, key, value = torch.split(
            mixed_qkv_processed,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        
        # Convert to padded format for attention computation
        query_padded = varlen_to_batch(query, cu_seqlens, max_seqlen)
        key_padded = varlen_to_batch(key, cu_seqlens, max_seqlen)
        value_padded = varlen_to_batch(value, cu_seqlens, max_seqlen)
        z_padded = varlen_to_batch(z, cu_seqlens, max_seqlen)
        b_padded = varlen_to_batch(b, cu_seqlens, max_seqlen)
        a_padded = varlen_to_batch(a, cu_seqlens, max_seqlen)
        
        # Reshape for multi-head attention
        query_padded = query_padded.reshape(batch_size, max_seqlen, self.num_k_heads, self.head_k_dim)
        key_padded = key_padded.reshape(batch_size, max_seqlen, self.num_k_heads, self.head_k_dim)
        value_padded = value_padded.reshape(batch_size, max_seqlen, self.num_v_heads, self.head_v_dim)
        
        # Compute gating
        beta_padded = b_padded.sigmoid()
        g_padded = -self.A_log.float().exp() * F.softplus(a_padded.float() + self.dt_bias)
        
        # Handle GQA
        if self.num_v_heads // self.num_k_heads > 1:
            query_padded = query_padded.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key_padded = key_padded.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        
        # Transpose for attention (batch, num_heads, seq_len, head_dim)
        query_padded = query_padded.transpose(1, 2)
        key_padded = key_padded.transpose(1, 2)
        value_padded = value_padded.transpose(1, 2)
        beta_padded = beta_padded.transpose(1, 2)
        g_padded = g_padded.transpose(1, 2)
        
        # Get actual sequence lengths
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        
        # Apply gated delta rule for each sequence
        core_attn_out_padded_list = []
        
        for i in range(batch_size):
            seq_len_i = seq_lengths[i].item()
            
            out_i = self.chunk_gated_delta_rule_fn(
                query_padded[i:i+1, :, :seq_len_i, :],
                key_padded[i:i+1, :, :seq_len_i, :],
                value_padded[i:i+1, :, :seq_len_i, :],
                g=g_padded[i:i+1, :, :seq_len_i],
                beta=beta_padded[i:i+1, :, :seq_len_i],
                chunk_size=self.config.chunk_size,
                use_qk_l2norm_in_kernel=self.config.use_qk_l2norm,
            )
            core_attn_out_padded_list.append(out_i)
        
        # Pad outputs back to same length
        core_attn_out_padded = torch.zeros(batch_size, self.num_v_heads, max_seqlen, self.head_v_dim,
                                          dtype=query_padded.dtype, device=query_padded.device)
        for i, out_i in enumerate(core_attn_out_padded_list):
            seq_len_i = seq_lengths[i].item()
            core_attn_out_padded[i, :, :seq_len_i, :] = out_i[0]
        
        # Gated normalization
        core_attn_out_padded = core_attn_out_padded.transpose(1, 2)  # (batch, max_seqlen, num_heads, head_dim_v)
        core_attn_out_padded = core_attn_out_padded.reshape(batch_size * max_seqlen, self.head_v_dim)
        z_padded = z_padded.reshape(batch_size * max_seqlen, self.head_v_dim)
        core_attn_out_padded = self.norm(core_attn_out_padded, z_padded)
        core_attn_out_padded = core_attn_out_padded.reshape(batch_size, max_seqlen, -1)
        
        # Convert back to packed format
        core_attn_out = batch_to_varlen(core_attn_out_padded, cu_seqlens)
        
        # Output projection
        output = self.out_proj(core_attn_out)
        
        return output
