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
from jax import Array


__all__ = ["yat_score"]


def yat_score(
    dot_prod_map: Array,
    distance_sq: Array,
    *,
    bias: Optional[Array],
    epsilon: float,
    epsilon_param: Optional[Array],
    alpha: Optional[Array],
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

    if epsilon_param is not None:
        eps = jax.nn.softplus(epsilon_param.astype(dot_prod_map.dtype))
    else:
        eps = epsilon

    y = dot_prod_map ** 2 / (distance_sq + eps)

    if alpha is not None:
        y = y * alpha

    return y
