"""Convolution Utility Functions.

This module provides utility functions and default initializers for
YAT convolution layers.
"""

from __future__ import annotations

import typing as tp

import jax.numpy as jnp
from jax import lax

from flax.nnx.nn import initializers
from flax.typing import PaddingLike, LaxPadding


# Default initializers
default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = jnp.sqrt(2.0)


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
    """Canonicalizes conv padding to a jax.lax supported format.

    Args:
        padding: Padding specification. Can be:
            - str: 'SAME', 'VALID', 'CIRCULAR', 'REFLECT', 'CAUSAL'
            - int: Same padding on all sides
            - Sequence: Per-dimension padding specification
        rank: Number of spatial dimensions.

    Returns:
        Padding in JAX LAX format.

    Raises:
        ValueError: If padding format is invalid.
    """
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, tp.Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(
        f"Invalid padding format: {padding}, should be str, int,"
        f" or a sequence of len {rank} where each element is an"
        " int or pair of ints."
    )


def conv_dimension_numbers(input_shape: tuple) -> lax.ConvDimensionNumbers:
    """Computes the dimension numbers based on the input shape.

    For convolutions, the input is expected to be in NHWC format
    (batch, spatial..., channels).

    Args:
        input_shape: Shape of the input tensor.

    Returns:
        ConvDimensionNumbers for use with lax.conv_general_dilated.
    """
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


# Alias for backwards compatibility
_conv_dimension_numbers = conv_dimension_numbers




