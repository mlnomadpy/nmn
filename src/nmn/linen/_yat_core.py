"""Shared YAT formula helper for Flax Linen conv / transpose-conv layers.

Unlike keras / tf / torch the Linen layers resolve bias, alpha, and the
optional learnable epsilon inside ``__call__`` (as local variables, not
``self.*``), so this helper takes them as keyword arguments rather than
duck-typing on a layer object.

The bias broadcast shape is derived from ``dot_prod_map.ndim`` — for
channels-last layouts the feature axis is always at -1.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax.custom_transpose import custom_transpose
from jax import Array


__all__ = [
    "reduction_safe_upcast",
    "safe_kernel_init",
    "saturating_downcast",
    "upcast_yat_operands",
    "yat_score",
]


@jax.custom_jvp
def _reduction_safe_upcast(value):
    return value.astype(jnp.float32)


@_reduction_safe_upcast.defjvp
def _reduction_safe_upcast_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    output = _reduction_safe_upcast(value)

    @custom_transpose
    def tangent_cast(dtype_anchor, tangent_value):
        del dtype_anchor
        return tangent_value.astype(jnp.float32)

    @tangent_cast.def_transpose
    def tangent_cast_transpose(dtype_anchor, cotangent):
        limits = jnp.finfo(dtype_anchor.dtype)
        cotangent = jnp.clip(cotangent, limits.min, limits.max)
        return (cotangent.astype(dtype_anchor.dtype),)

    tangent_output_type = jax.typeof(output).to_tangent_aval()
    tangent_output = tangent_cast(tangent_output_type, value, tangent)
    return output, tangent_output


def reduction_safe_upcast(value):
    """Upcast lowp arithmetic with JVP and operator-local VJP saturation.

    The custom transpose clips a single returning aggregate cotangent while
    leaving forward-mode tangents unchanged. Separate paths may still be added
    later at a shared low-precision leaf and overflow; use fp32 storage there.
    """
    if value.dtype in (jnp.float16, jnp.bfloat16):
        return _reduction_safe_upcast(value)
    return value


def saturating_downcast(value, dtype):
    """Two-sided finite cast to a low-precision public output dtype."""
    if dtype in (jnp.float16, jnp.bfloat16):
        limits = jnp.finfo(dtype)
        value = jnp.clip(value, limits.min, limits.max)
    return value.astype(dtype)


def safe_kernel_init(initializer):
    """Run decomposition-based initializers in fp32 before a lowp cast.

    JAX's CPU QR kernel does not accept float16, which made the default
    orthogonal convolution initializer fail before the layer could run.
    """
    def init(key, shape, dtype):
        if dtype in (jnp.float16, jnp.bfloat16):
            return initializer(key, shape, jnp.float32).astype(dtype)
        return initializer(key, shape, dtype)

    return init


def upcast_yat_operands(inputs, kernel, bias, alpha):
    """Return score operands in fp32 for fp16/bf16 computation policies."""
    output_dtype = inputs.dtype
    if output_dtype in (jnp.float16, jnp.bfloat16):
        inputs = reduction_safe_upcast(inputs)
        kernel = reduction_safe_upcast(kernel)
        bias = reduction_safe_upcast(bias) if bias is not None else None
        alpha = reduction_safe_upcast(alpha) if alpha is not None else None
    return inputs, kernel, bias, alpha, output_dtype


def yat_score(
    dot_prod_map: Array,
    distance_sq: Array,
    *,
    bias: Optional[Array],
    epsilon: float,
    epsilon_param: Optional[Array],
    alpha: Optional[Array],
    output_dtype=None,
) -> Array:
    """Apply bias / epsilon / YAT-divide / alpha to a raw conv output.

    Returns ``(dot_prod_map + bias) ** 2 / (distance_sq + eps) * alpha``,
    with ``eps = softplus(epsilon_param)`` if a learnable epsilon param
    is supplied, otherwise ``eps = epsilon``.
    """
    if bias is not None:
        # Channels-last layout: feature axis is always -1.
        bias_broadcast_shape = (1,) * (dot_prod_map.ndim - 1) + (-1,)
        dot_prod_map = dot_prod_map + bias.reshape(bias_broadcast_shape)

    if output_dtype is None:
        output_dtype = dot_prod_map.dtype
    if epsilon_param is not None:
        score_dtype = jnp.promote_types(dot_prod_map.dtype, epsilon_param.dtype)
        eps = jax.nn.softplus(epsilon_param.astype(score_dtype))
        dot_prod_map = dot_prod_map.astype(score_dtype)
        distance_sq = distance_sq.astype(score_dtype)
    else:
        eps = epsilon

    # Squared distances are non-negative; clamp cancellation noise while
    # retaining NaNs from genuinely invalid inputs.
    distance_sq = jnp.maximum(distance_sq, 0.0)
    y = dot_prod_map ** 2 / (distance_sq + eps)

    if alpha is not None:
        y = y * alpha

    return saturating_downcast(y, output_dtype) if output_dtype is not None else y
