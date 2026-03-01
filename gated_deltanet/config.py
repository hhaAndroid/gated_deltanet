"""GatedDeltaNet Configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class GatedDeltaNetConfig:
    """Configuration for GatedDeltaNet.
    
    Args:
        hidden_size: Hidden dimension size
        num_key_heads: Number of key heads
        num_value_heads: Number of value heads  
        key_head_dim: Dimension per key head
        value_head_dim: Dimension per value head
        conv_kernel_size: Convolution kernel size for causal conv1d
        hidden_act: Activation function (default: "silu")
        rms_norm_eps: Epsilon for RMS normalization (default: 1e-6)
        use_qk_l2norm: Whether to use L2 normalization for Q/K (default: True)
        chunk_size: Chunk size for chunkwise computation (default: 64)
    """
    hidden_size: int = 512
    num_key_heads: int = 4
    num_value_heads: int = 4
    key_head_dim: int = 128
    value_head_dim: int = 128
    conv_kernel_size: int = 4
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    use_qk_l2norm: bool = True
    chunk_size: int = 64
    dtype: Optional[str] = None
