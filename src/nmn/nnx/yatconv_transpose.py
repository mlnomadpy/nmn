"""YAT Transposed Convolution Module (Backwards Compatibility).

This module re-exports YatConvTranspose from the new modular structure
at `nmn.nnx.conv` for backwards compatibility.

For new code, prefer importing directly from `nmn.nnx.conv`:

    from nmn.nnx.conv import YatConvTranspose
"""

# Re-export from the new modular structure
from nmn.nnx.conv import YatConvTranspose

__all__ = ["YatConvTranspose"]
