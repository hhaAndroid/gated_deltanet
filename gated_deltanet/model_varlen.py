"""GatedDeltaNet with variable length sequence support.

This module provides efficient handling of variable-length sequences in a batch
using cu_seqlens (cumulative sequence lengths) parameter.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GatedDeltaNetConfig
from .model import (
    RMSNormGated,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
    torch_causal_conv1d_update,
)

# Try to import fast implementations
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
except ImportError:
    FusedRMSNormGated = None
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None


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
    """Gated DeltaNet with variable length sequence support.
    
    This version efficiently handles batches where sequences have different lengths
    by using cu_seqlens to indicate the boundaries of each sequence.
    
    Compared to the standard version, this avoids padding overhead for variable-length
    sequences by processing them in a packed format.
    
    Args:
        config: GatedDeltaNetConfig configuration
        layer_idx: layer index (for caching purposes)
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
        
        # Select implementations
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule_fn = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule_fn = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Forward pass with variable length sequences.
        
        Args:
            hidden_states: (total_tokens, hidden_size) packed tensor
            cu_seqlens: (batch_size + 1,) cumulative sequence lengths
            max_seqlen: maximum sequence length (computed if not provided)
            attention_mask: optional attention mask (not used in varlen mode)
            past_key_value: cached states for generation
            use_cache: whether to use cache
        
        Returns:
            output: (total_tokens, hidden_size) packed output
            past_key_value: updated cache (if use_cache=True)
        
        Example:
            >>> # Three sequences with lengths 10, 20, 15
            >>> cu_seqlens = torch.tensor([0, 10, 30, 45], dtype=torch.int32)
            >>> hidden_states = torch.randn(45, 512)  # total_tokens = 45
            >>> output, _ = model(hidden_states, cu_seqlens)
        """
        total_tokens, _ = hidden_states.shape
        batch_size = len(cu_seqlens) - 1
        
        if max_seqlen is None:
            # Compute max sequence length from cu_seqlens
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        
        # Check if we're in generation mode (single token per sequence)
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        use_cache_mode = past_key_value is not None and (seq_lengths == 1).all()
        
        # Input projections
        mixed_qkv = self.in_proj_qkv(hidden_states)  # (total_tokens, conv_dim)
        z = self.in_proj_z(hidden_states)  # (total_tokens, value_dim)
        b = self.in_proj_b(hidden_states)  # (total_tokens, num_v_heads)
        a = self.in_proj_a(hidden_states)  # (total_tokens, num_v_heads)
        
        # Process each sequence separately for causal convolution
        # This maintains causality within each sequence
        mixed_qkv_processed = torch.empty_like(mixed_qkv)
        
        if use_cache_mode:
            # Single token generation mode
            conv_state, recurrent_state = past_key_value if past_key_value else (None, None)
            if conv_state is None:
                conv_state = torch.zeros(batch_size, self.conv_dim, self.conv_kernel_size, 
                                        dtype=mixed_qkv.dtype, device=mixed_qkv.device)
            
            # Process each sequence
            for i in range(batch_size):
                start = cu_seqlens[i].item()
                token_qkv = mixed_qkv[start:start+1].unsqueeze(0).transpose(1, 2)  # (1, conv_dim, 1)
                
                token_qkv = self.causal_conv1d_update(
                    token_qkv,
                    conv_state[i:i+1],
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias if hasattr(self.conv1d, 'bias') else None,
                    self.config.hidden_act,
                )
                mixed_qkv_processed[start:start+1] = token_qkv.transpose(1, 2).squeeze(0)
        else:
            # Training / prefill mode
            # Convert to padded format for easier processing
            mixed_qkv_padded = varlen_to_batch(mixed_qkv, cu_seqlens, max_seqlen)
            # (batch, max_seqlen, conv_dim) -> (batch, conv_dim, max_seqlen)
            mixed_qkv_padded = mixed_qkv_padded.transpose(1, 2)
            
            if self.causal_conv1d_fn is not None:
                # Use fast causal conv1d if available
                mixed_qkv_padded = self.causal_conv1d_fn(
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
            
            if use_cache:
                # Store conv state for future generation
                conv_state = torch.zeros(batch_size, self.conv_dim, self.conv_kernel_size,
                                        dtype=mixed_qkv.dtype, device=mixed_qkv.device)
                for i in range(batch_size):
                    start = cu_seqlens[i].item()
                    end = cu_seqlens[i + 1].item()
                    seq_len = end - start
                    if seq_len > 0:
                        # Store the last conv_kernel_size-1 tokens (or padding)
                        if seq_len >= self.conv_kernel_size - 1:
                            conv_state[i] = mixed_qkv[start:end][-(self.conv_kernel_size-1):].T
                        else:
                            # Pad with zeros
                            actual_tokens = mixed_qkv[start:end].T
                            padding_size = self.conv_kernel_size - 1 - actual_tokens.shape[1]
                            conv_state[i] = F.pad(actual_tokens, (padding_size, 0))
        
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
        
        # Apply gated delta rule for each sequence
        if use_cache_mode:
            # Single token generation
            recurrent_state = past_key_value[1] if len(past_key_value) > 1 else None
            if recurrent_state is None:
                recurrent_state = torch.zeros(batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                                            dtype=query_padded.dtype, device=query_padded.device)
            
            core_attn_out_padded = torch.zeros(batch_size, self.num_v_heads, 1, self.head_v_dim,
                                              dtype=query_padded.dtype, device=query_padded.device)
            new_recurrent_state = torch.zeros_like(recurrent_state)
            
            for i in range(batch_size):
                out_i, state_i = self.recurrent_gated_delta_rule_fn(
                    query_padded[i:i+1, :, :1, :],
                    key_padded[i:i+1, :, :1, :],
                    value_padded[i:i+1, :, :1, :],
                    g=g_padded[i:i+1, :, :1],
                    beta=beta_padded[i:i+1, :, :1],
                    initial_state=recurrent_state[i],
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=self.config.use_qk_l2norm,
                )
                core_attn_out_padded[i] = out_i[0]
                new_recurrent_state[i] = state_i
        else:
            # Training / prefill mode
            core_attn_out_padded_list = []
            new_recurrent_state_list = []
            
            for i in range(batch_size):
                seq_len_i = seq_lengths[i].item()
                
                out_i, state_i = self.chunk_gated_delta_rule_fn(
                    query_padded[i:i+1, :, :seq_len_i, :],
                    key_padded[i:i+1, :, :seq_len_i, :],
                    value_padded[i:i+1, :, :seq_len_i, :],
                    g=g_padded[i:i+1, :, :seq_len_i],
                    beta=beta_padded[i:i+1, :, :seq_len_i],
                    chunk_size=self.config.chunk_size,
                    initial_state=None,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=self.config.use_qk_l2norm,
                )
                core_attn_out_padded_list.append(out_i)
                if use_cache:
                    new_recurrent_state_list.append(state_i)
            
            # Pad outputs back to same length
            core_attn_out_padded = torch.zeros(batch_size, self.num_v_heads, max_seqlen, self.head_v_dim,
                                              dtype=query_padded.dtype, device=query_padded.device)
            for i, out_i in enumerate(core_attn_out_padded_list):
                seq_len_i = seq_lengths[i].item()
                core_attn_out_padded[i, :, :seq_len_i, :] = out_i[0]
            
            if use_cache:
                new_recurrent_state = torch.stack(new_recurrent_state_list, dim=0)
        
        # Update cache
        if use_cache:
            past_key_value = (conv_state, new_recurrent_state)
        
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
        
        return output, past_key_value
