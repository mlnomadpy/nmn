"""Shared YAT formula helpers for Keras conv / transpose-conv layers.

Each YatConv*D and YatConvTranspose*D layer used to inline ~22 lines of
identical bias / epsilon / YAT-divide / alpha logic. This module holds the
single source of truth.

The helpers are intentionally duck-typed on a `layer` object — they read
the same instance attributes that the Keras conv classes already expose.
That keeps the call sites trivial (one helper call replaces the whole
tail block) and avoids long parameter lists.
"""

from __future__ import annotations

from keras import ops


__all__ = ["yat_score"]


def yat_score(layer, dot_prod_map, distance_sq_map):
    """Apply bias / epsilon / YAT-divide / alpha to a raw conv output.

    Returns ``(dot_prod_map + bias) ** 2 / (distance_sq_map + eps) * alpha``.

    The `layer` is expected to expose:

    * ``use_bias``, ``_constant_bias_value``, ``bias``, ``data_format``,
      ``kernel_size`` — for bias resolution and channels-first reshape.
    * ``learnable_epsilon``, ``epsilon_param``, ``epsilon`` — for the
      effective epsilon (softplus-of-raw or constant).
    * ``use_alpha``, ``alpha`` — for the optional alpha multiplier.

    `dot_prod_map` and `distance_sq_map` must already be in the layer's
    output layout (channels_first or channels_last).
    """
    # Add bias before squaring (constant or learnable; reshape for channels_first).
    if layer.use_bias:
        if layer._constant_bias_value is not None:
            dot_prod_map = dot_prod_map + layer._constant_bias_value
        else:
            bias = layer.bias
            if layer.data_format == "channels_first":
                bias_shape = (1, -1) + (1,) * len(layer.kernel_size)
                bias = ops.reshape(bias, bias_shape)
            dot_prod_map = ops.add(dot_prod_map, bias)

    # Resolve effective epsilon (learnable via softplus, or constant).
    if layer.learnable_epsilon and layer.epsilon_param is not None:
        eps = ops.softplus(layer.epsilon_param)
    else:
        eps = layer.epsilon

    # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps).
    outputs = dot_prod_map ** 2 / (distance_sq_map + eps)

    # Optional alpha (constant via _constant_alpha_value is folded into
    # `layer.alpha` at __init__ time; here we only need `use_alpha` + alpha).
    if layer.use_alpha and layer.alpha is not None:
        outputs = outputs * layer.alpha

    return outputs
