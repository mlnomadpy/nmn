"""Shared YAT formula helper for MLX conv / transpose-conv layers.

Each ``YatConv*D`` and ``YatConvTranspose*D`` layer inlines roughly twenty
lines of identical bias / epsilon / YAT-divide / alpha logic — this module
holds the single source of truth, mirroring ``nmn.tf._yat_core``.

The helper is duck-typed on the ``layer`` object; it reads the same
instance attributes the conv classes already expose.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


__all__ = ["yat_score"]


def yat_score(
    layer,
    dot_prod_map: mx.array,
    distance_sq_map: mx.array,
) -> mx.array:
    """Apply bias / epsilon / YAT-divide / alpha to a raw conv output.

    Returns ``(dot + bias) ** 2 / (||x − W||² + ε) · α``.

    Required attributes on ``layer``:

    * ``use_bias``, ``_constant_bias_value``, ``bias``, ``dtype`` — for the
      bias resolution.
    * ``learnable_epsilon``, ``epsilon_param``, ``epsilon`` — for the
      effective epsilon (softplus-of-raw or constant).
    * ``use_alpha``, ``alpha``, ``_constant_alpha_value`` — for the optional
      alpha multiplier.
    """
    # Add bias inside the numerator square.
    if layer.use_bias:
        if layer._constant_bias_value is not None:
            bias_const = mx.full(
                (layer.filters,),
                layer._constant_bias_value,
                dtype=layer.dtype,
            )
            bias_shape = [1] * (dot_prod_map.ndim - 1) + [layer.filters]
            dot_prod_map = dot_prod_map + mx.reshape(bias_const, bias_shape)
        else:
            bias_shape = [1] * (dot_prod_map.ndim - 1) + [layer.filters]
            dot_prod_map = dot_prod_map + mx.reshape(layer.bias, bias_shape)

    # Effective epsilon (softplus-of-raw learnable, or constant).
    if layer.learnable_epsilon and getattr(layer, "epsilon_param", None) is not None:
        eps = nn.softplus(layer.epsilon_param)
    else:
        eps = layer.epsilon

    # YAT: (dot + bias)² / (||x − W||² + ε)
    y = (dot_prod_map * dot_prod_map) / (distance_sq_map + eps)

    # Optional alpha.
    if layer._constant_alpha_value is not None:
        y = y * mx.array(layer._constant_alpha_value, dtype=layer.dtype)
    elif layer.use_alpha and getattr(layer, "alpha", None) is not None:
        y = y * layer.alpha

    return y
