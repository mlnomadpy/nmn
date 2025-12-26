"""Convolution Utilities (Backwards Compatibility).

This module re-exports utilities from the new modular structure
at `nmn.nnx.conv` for backwards compatibility.

For new code, prefer importing directly from `nmn.nnx.conv`:

    from nmn.nnx.conv import (
        canonicalize_padding,
        conv_dimension_numbers,
        default_kernel_init,
        default_bias_init,
        default_alpha_init,
    )
"""

# Re-export from the new modular structure
from nmn.nnx.conv import (
    canonicalize_padding,
    conv_dimension_numbers,
    _conv_dimension_numbers,
    default_kernel_init,
    default_bias_init,
    default_alpha_init,
    DEFAULT_CONSTANT_ALPHA,
)

__all__ = [
    "canonicalize_padding",
    "conv_dimension_numbers",
    "_conv_dimension_numbers",
    "default_kernel_init",
    "default_bias_init",
    "default_alpha_init",
    "DEFAULT_CONSTANT_ALPHA",
]
