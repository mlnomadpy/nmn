"""YAT Convolution Module (Backwards Compatibility).

This module re-exports YatConv from the new modular structure
at `nmn.nnx.conv` for backwards compatibility.

For new code, prefer importing directly from `nmn.nnx.conv`:

    from nmn.nnx.conv import YatConv
"""

# Re-export from the new modular structure
from nmn.nnx.conv import YatConv

__all__ = ["YatConv"]
