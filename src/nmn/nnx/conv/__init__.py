"""NMN Convolution Module.

This module provides YAT (You Are There) convolution layers for neural networks.

YAT Convolution applies the formula:
    y = (x * W)² / (||x - W||² + ε)

where * denotes the convolution operation, and the distance is computed
patch-wise between input patches and kernel weights.

Example:
    >>> from nmn.nnx.conv import YatConv, YatConvTranspose
    >>> from flax import nnx
    >>> import jax.numpy as jnp
    >>>
    >>> # Create a 2D convolution layer
    >>> rngs = nnx.Rngs(0)
    >>> conv = YatConv(
    ...     in_features=3,
    ...     out_features=32,
    ...     kernel_size=(3, 3),
    ...     padding='SAME',
    ...     rngs=rngs,
    ... )
    >>>
    >>> # Forward pass
    >>> x = jnp.zeros((1, 28, 28, 3))
    >>> output = conv(x)

Modules:
    - yat_conv: YatConv class for convolution
    - yat_conv_transpose: YatConvTranspose class for transposed convolution
    - utils: Utility functions and default initializers
"""

# YAT Convolution
from .yat_conv import YatConv

# YAT Transposed Convolution
from .yat_conv_transpose import YatConvTranspose

# Utilities
from .utils import (
    canonicalize_padding,
    conv_dimension_numbers,
    default_kernel_init,
    default_bias_init,
    default_alpha_init,
    DEFAULT_CONSTANT_ALPHA,
    # Backwards compatibility alias
    _conv_dimension_numbers,
)

__all__ = [
    # Core layers
    "YatConv",
    "YatConvTranspose",
    # Utilities
    "canonicalize_padding",
    "conv_dimension_numbers",
    "default_kernel_init",
    "default_bias_init",
    "default_alpha_init",
    "DEFAULT_CONSTANT_ALPHA",
    # Backwards compatibility
    "_conv_dimension_numbers",
]




