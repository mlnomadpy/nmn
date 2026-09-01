"""Custom activation and squashing functions for NNX.

This module provides specialized squashing functions optimized for neural networks,
including softermax, softer_sigmoid, and soft_tanh. These functions are:

- Fully differentiable and JAX-compatible
- JIT-compiled for performance
- Designed to work with NNX modules and training loops
- Intended for non-negative inputs to ensure proper behavior

Usage with NNX:
    >>> from flax import nnx
    >>> from nmn.nnx.layers.squashers import softermax, softer_sigmoid, soft_tanh
    >>> import jax.numpy as jnp
    >>>
    >>> # Use in a forward pass
    >>> x = jnp.array([[1.0, 2.0, 3.0]])
    >>> output = softermax(x)  # Automatic JIT compilation

Function Types:
    - softermax: Normalizes scores (similar to softmax but more flexible)
    - softer_sigmoid: S-curve squashing to [0, 1)
    - soft_tanh: Hyperbolic-tangent-like squashing to [-1, 1)

All functions accept a power parameter `n` to control sharpness:
    - n=1: Standard behavior
    - n>1: Sharper/harder transitions
    - 0<n<1: Softer/smoother transitions
"""

from .soft_tanh import soft_tanh
from .softer_sigmoid import softer_sigmoid
from .softermax import softermax

__all__ = [
    "softermax",
    "softer_sigmoid",
    "soft_tanh",
]
