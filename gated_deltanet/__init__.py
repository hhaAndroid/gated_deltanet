"""GatedDeltaNet package."""
from .config import GatedDeltaNetConfig

__all__ = [
    "GatedDeltaNetConfig",
    "GatedDeltaNet",
    "GatedDeltaNetVarlen",
]

# Lazy imports to avoid torch dependency at package level
def __getattr__(name):
    if name == "GatedDeltaNet":
        from .model import GatedDeltaNet
        return GatedDeltaNet
    elif name == "GatedDeltaNetVarlen":
        from .model_varlen import GatedDeltaNetVarlen
        return GatedDeltaNetVarlen
    raise AttributeError(f"module 'gated_deltanet' has no attribute '{name}'")
