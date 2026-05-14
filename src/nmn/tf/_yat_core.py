"""Shared YAT formula helper for TF conv / transpose-conv layers.

Each YatConv*D and YatConvTranspose*D layer used to inline ~20 lines of
identical bias / epsilon / YAT-divide / alpha logic. This module holds
the single source of truth.

The helper is duck-typed on the ``layer`` object — it reads the same
instance attributes that the TF conv classes already expose.
"""

from __future__ import annotations

import tensorflow as tf


__all__ = ["yat_score"]


def yat_score(layer, dot_prod_map, distance_sq_map):
    """Apply bias / epsilon / YAT-divide / alpha to a raw conv output.

    Returns ``(dot_prod_map + bias) ** 2 / (distance_sq_map + eps) * alpha``.

    The `layer` is expected to expose:

    * ``use_bias``, ``_constant_bias_value``, ``bias``, ``dtype`` — for
      bias resolution.
    * ``learnable_epsilon``, ``epsilon_param``, ``epsilon`` — for the
      effective epsilon (softplus-of-raw or constant).
    * ``use_alpha``, ``alpha`` — for the optional alpha multiplier.
    """
    # Add bias before squaring (constant or learnable).
    if layer.use_bias:
        if layer._constant_bias_value is not None:
            dot_prod_map = dot_prod_map + tf.cast(layer._constant_bias_value, layer.dtype)
        else:
            dot_prod_map = dot_prod_map + layer.bias

    # Resolve effective epsilon (learnable via softplus, or constant).
    if layer.learnable_epsilon and layer.epsilon_param is not None:
        eps = tf.nn.softplus(layer.epsilon_param)
    else:
        eps = layer.epsilon

    # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps).
    y = dot_prod_map ** 2 / (distance_sq_map + eps)

    # Optional alpha (learnable; constant_alpha is folded into layer.alpha).
    if layer.use_alpha and layer.alpha is not None:
        y = y * layer.alpha

    return y
