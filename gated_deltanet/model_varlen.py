"""GatedDeltaNet with variable length sequence support - Training only version.

This module provides efficient handling of variable-length sequences in a batch
using cu_seqlens (cumulative sequence lengths) parameter.

Key difference from base version:
- Inputs are in packed format (total_tokens, hidden_size)
- Uses seq_idx to handle sequence boundaries in causal_conv1d
- chunk_gated_delta_rule operates on packed format with offsets
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GatedDeltaNetConfig
from .model import RMSNormGated

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


def torch_causal_conv1d_varlen(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Torch implementation of causal conv1d for variable length sequences.
    
    Processes packed tensor with proper handling of sequence boundaries.
    Each sequence starts fresh without seeing previous sequence's padding.
    
    Args:
        x: (hidden_size, total_tokens) packed tensor
        weight: (hidden_size, kernel_size)
        bias: (hidden_size,) or None
        cu_seqlens: (batch_size + 1,) cumulative sequence lengths
        activation: activation function name
    
    Returns:
        output: (hidden_size, total_tokens) packed tensor
    """
    batch_size = len(cu_seqlens) - 1
    kernel_size = weight.shape[1]
    hidden_size = x.shape[0]
    
    output_list = []
    
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        
        # Extract sequence
        x_seq = x[:, start:end]  # (hidden_size, seq_len)
        
        # Pad for causal convolution
        x_padded = F.pad(x_seq, (kernel_size - 1, 0))
        
        # Apply conv1d
        weight_expanded = weight.unsqueeze(1)  # (hidden_size, 1, kernel_size)
        out_seq = F.conv1d(x_padded, weight_expanded, bias, groups=hidden_size)
        
        # Apply activation
        if activation == "silu":
            out_seq = F.silu(out_seq)
        
        output_list.append(out_seq)
    
    # Concatenate back to packed format
    output = torch.cat(output_list, dim=1)
    return output


def torch_chunk_gated_delta_rule_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    """Torch implementation of chunk gated delta rule for variable length sequences.
    
    Processes packed tensors with proper handling of sequence boundaries.
    Each sequence is processed independently.
    
    Args:
        query: (total_tokens, num_heads, head_k_dim) packed tensor
        key: (total_tokens, num_heads, head_k_dim) packed tensor
        value: (total_tokens, num_heads, head_v_dim) packed tensor
        g: (total_tokens, num_heads) packed tensor - gate values
        beta: (total_tokens, num_heads) packed tensor - beta values
        cu_seqlens: (batch_size + 1,) cumulative sequence lengths
        chunk_size: chunk size for processing
        use_qk_l2norm_in_kernel: whether to use L2 norm for Q/K
    
    Returns:
        output: (total_tokens, num_heads, head_v_dim) packed tensor
    """
    import math
    
    initial_dtype = query.dtype
    
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, dim=-1, eps=1e-6)
        key = F.normalize(key, dim=-1, eps=1e-6)
    
    batch_size = len(cu_seqlens) - 1
    output_list = []
    
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        
        # Extract sequence
        q_seq = query[start:end].to(torch.float32)  # (seq_len, num_heads, head_k_dim)
        k_seq = key[start:end].to(torch.float32)
        v_seq = value[start:end].to(torch.float32)
        g_seq = g[start:end].to(torch.float32)  # (seq_len, num_heads)
        beta_seq = beta[start:end].to(torch.float32)
        
        num_heads = q_seq.shape[1]
        k_head_dim = q_seq.shape[2]
        v_head_dim = v_seq.shape[2]
        
        scale = 1.0 / math.sqrt(k_head_dim)
        q_seq = q_seq * scale
        
        # Pad to multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_size > 0:
            q_seq = F.pad(q_seq, (0, 0, 0, 0, 0, pad_size))
            k_seq = F.pad(k_seq, (0, 0, 0, 0, 0, pad_size))
            v_seq = F.pad(v_seq, (0, 0, 0, 0, 0, pad_size))
            beta_seq = F.pad(beta_seq, (0, 0, 0, pad_size))
            g_seq = F.pad(g_seq, (0, 0, 0, pad_size))
        
        total_seq_len = seq_len + pad_size
        
        # Compute v_beta and k_beta
        v_beta = v_seq * beta_seq.unsqueeze(-1)
        k_beta = k_seq * beta_seq.unsqueeze(-1)
        
        # Reshape to chunks
        q_seq = q_seq.reshape(-1, chunk_size, num_heads, k_head_dim).transpose(0, 1)
        k_seq = k_seq.reshape(-1, chunk_size, num_heads, k_head_dim).transpose(0, 1)
        v_seq = v_seq.reshape(-1, chunk_size, num_heads, v_head_dim).transpose(0, 1)
        k_beta = k_beta.reshape(-1, chunk_size, num_heads, k_head_dim).transpose(0, 1)
        v_beta = v_beta.reshape(-1, chunk_size, num_heads, v_head_dim).transpose(0, 1)
        g_seq = g_seq.reshape(-1, chunk_size, num_heads).permute(1, 2, 0)
        
        # (chunk_size, num_chunks, num_heads, head_dim)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q_seq.device), diagonal=0)
        
        # Compute chunk decay
        g_seq = g_seq.cumsum(dim=-1)
        decay_mask = ((g_seq.unsqueeze(-1) - g_seq.unsqueeze(-2)).tril().exp().float()).tril()
        
        attn = -((k_beta @ k_seq.transpose(-2, -1)) * decay_mask.unsqueeze(1)).masked_fill(mask.unsqueeze(1), 0)
        
        # Compute attention within chunk
        for j in range(1, chunk_size):
            row = attn[j, :, :, :j].clone()
            sub = attn[:j, :, :, :j].clone()
            attn[j, :, :, :j] = row + (row.unsqueeze(-1) * sub).sum(-2)
        
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q_seq.device).view(chunk_size, 1, 1, chunk_size)
        v_seq = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g_seq.exp().unsqueeze(-1))
        
        # Initialize recurrent state
        last_recurrent_state = torch.zeros(num_heads, k_head_dim, v_head_dim, device=v_seq.device, dtype=v_seq.dtype)
        
        core_attn_out = torch.zeros_like(v_seq)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q_seq.device), diagonal=1)
        
        # Process chunks
        num_chunks = total_seq_len // chunk_size
        for j in range(num_chunks):
            q_j = q_seq[:, j]  # (chunk_size, num_heads, head_k_dim)
            k_j = k_seq[:, j]
            v_j = v_seq[:, j]
            g_j = g_seq[:, :, j]
            k_cumdecay_j = k_cumdecay[:, j]
            decay_mask_j = decay_mask[:, :, j, :]
            
            attn_chunk = (q_j @ k_j.transpose(-2, -1) * decay_mask_j.unsqueeze(1)).masked_fill_(mask.unsqueeze(1), 0)
            v_prime = (k_cumdecay_j) @ last_recurrent_state.unsqueeze(0)
            v_new = v_j - v_prime
            attn_inter = (q_j * g_j.exp().unsqueeze(-1)) @ last_recurrent_state.unsqueeze(0)
            core_attn_out[:, j] = attn_inter + attn_chunk @ v_new
            
            # Update recurrent state
            last_recurrent_state = (
                last_recurrent_state * g_j[-1].exp().unsqueeze(-1).unsqueeze(-1)
                + (k_j * (g_j[-1].unsqueeze(-1) - g_j).exp().unsqueeze(-1)).transpose(-2, -1) @ v_new
            )
        
        # Reshape output
        core_attn_out = core_attn_out.transpose(0, 1).reshape(total_seq_len, num_heads, v_head_dim)
        core_attn_out = core_attn_out[:seq_len]
        
        output_list.append(core_attn_out.to(initial_dtype))
    
    # Concatenate back to packed format
    output = torch.cat(output_list, dim=0)
    return output


class GatedDeltaNetVarlen(nn.Module):
    """Gated DeltaNet with variable length sequence support for training.
    
    This version efficiently handles batches where sequences have different lengths
    by using cu_seqlens to indicate the boundaries of each sequence.
    
    Key features:
    - Inputs/outputs are in packed format (total_tokens, hidden_size)
    - No conversion to batch/padded format
    - Each sequence is processed independently with proper boundary handling
    
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
            max_seqlen: maximum sequence length (not used in computation, for compatibility)
        
        Returns:
            output: (total_tokens, hidden_size) packed output
        
        Example:
            >>> # Three sequences with lengths 10, 20, 15
            >>> cu_seqlens = torch.tensor([0, 10, 30, 45], dtype=torch.int32)
            >>> hidden_states = torch.randn(45, 512)  # total_tokens = 45
            >>> output = model(hidden_states, cu_seqlens)
        """
        total_tokens, _ = hidden_states.shape
        
        # Input projections - keep packed format
        mixed_qkv = self.in_proj_qkv(hidden_states)  # (total_tokens, conv_dim)
        z = self.in_proj_z(hidden_states)  # (total_tokens, value_dim)
        b = self.in_proj_b(hidden_states)  # (total_tokens, num_v_heads)
        a = self.in_proj_a(hidden_states)  # (total_tokens, num_v_heads)
        
        # Transpose for conv1d: (conv_dim, total_tokens)
        mixed_qkv = mixed_qkv.transpose(0, 1)
        
        # Causal convolution with varlen support
        if causal_conv1d_fn is not None and hasattr(causal_conv1d_fn, '__code__') and 'seq_idx' in causal_conv1d_fn.__code__.co_varnames:
            # Use fast causal conv1d with seq_idx if available
            # Create seq_idx from cu_seqlens
            batch_size = len(cu_seqlens) - 1
            seq_idx = torch.zeros(total_tokens, dtype=torch.int32, device=hidden_states.device)
            for i in range(batch_size):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_idx[start:end] = i
            
            mixed_qkv = causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias if hasattr(self.conv1d, 'bias') else None,
                activation=self.config.hidden_act,
                seq_idx=seq_idx,
            )
        else:
            # Torch fallback - process each sequence independently
            mixed_qkv = torch_causal_conv1d_varlen(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias if hasattr(self.conv1d, 'bias') else None,
                cu_seqlens=cu_seqlens,
                activation=self.config.hidden_act,
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
        
        # Apply gated delta rule on packed format
        if chunk_gated_delta_rule is not None:
            # Use flash-linear-attention implementation with varlen support
            # Reshape to (1, total_tokens, num_heads, dim) as required by fla
            core_attn_out, _ = chunk_gated_delta_rule(
                query.unsqueeze(0),  # (1, total_tokens, num_heads, head_k_dim)
                key.unsqueeze(0),
                value.unsqueeze(0),
                g=g.unsqueeze(0),  # (1, total_tokens, num_heads)
                beta=beta.unsqueeze(0),
                cu_seqlens=cu_seqlens.to(torch.long),  # fla requires LongTensor
                use_qk_l2norm_in_kernel=self.config.use_qk_l2norm,
            )
            core_attn_out = core_attn_out.squeeze(0)  # (total_tokens, num_heads, head_v_dim)
        else:
            # Torch fallback - process each sequence independently on packed format
            core_attn_out = torch_chunk_gated_delta_rule_varlen(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                cu_seqlens=cu_seqlens,
                chunk_size=self.config.chunk_size,
                use_qk_l2norm_in_kernel=self.config.use_qk_l2norm,
            )
        
        # Gated normalization on packed format
        core_attn_out = core_attn_out.reshape(total_tokens, self.head_v_dim)
        z = z.reshape(total_tokens, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(total_tokens, -1)
        
        # Output projection
        output = self.out_proj(core_attn_out)
        
        return output
