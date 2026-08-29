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
from keras.src.backend import standardize_dtype


__all__ = ["stable_yat_ratio", "yat_score"]


def stable_yat_ratio(dot_product, distance_sq, epsilon):
    """Evaluate the YAT ratio without low-precision overflow/cancellation."""
    distance_sq = ops.maximum(distance_sq, ops.cast(0.0, distance_sq.dtype))
    source_dtype = standardize_dtype(dot_product.dtype)
    if source_dtype not in {"float16", "bfloat16"}:
        return ops.square(dot_product) / (distance_sq + epsilon)

    epsilon = ops.cast(epsilon, source_dtype)

    @ops.custom_gradient
    def low_precision_ratio(dot, distance, eps):
        dot32 = ops.cast(dot, "float32")
        distance32 = ops.cast(distance, "float32")
        eps32 = ops.cast(eps, "float32")
        denominator = distance32 + eps32
        raw_ratio = ops.square(dot32) / denominator
        max_value = 65504.0 if source_dtype == "float16" else 3.38953139e38
        active = raw_ratio < max_value
        result = ops.cast(ops.minimum(raw_ratio, max_value), source_dtype)

        def grad(*args, upstream=None):
            if upstream is None:
                (upstream,) = args
            upstream32 = ops.cast(upstream, "float32")
            active32 = ops.cast(active, "float32")
            dot_grad = upstream32 * (2.0 * dot32 / denominator) * active32
            denominator_grad = (
                upstream32
                * (-ops.square(dot32) / ops.square(denominator))
                * active32
            )
            distance_grad = ops.where(
                distance32 > 0.0, denominator_grad, ops.zeros_like(denominator_grad)
            )
            if source_dtype == "float16":
                dot_grad = ops.clip(dot_grad, -65504.0, 65504.0)
                distance_grad = ops.clip(distance_grad, -65504.0, 65504.0)
            # ``eps`` is scalar for every current caller, so its cotangent must
            # sum all broadcast denominator contributions.  Reduce before
            # clipping: clipping each element first can still overflow when
            # two or more fp16 contributions are added together.
            epsilon_grad = ops.sum(denominator_grad)
            epsilon_max = 65504.0 if source_dtype == "float16" else 3.38953139e38
            epsilon_grad = ops.clip(epsilon_grad, -epsilon_max, epsilon_max)
            epsilon_grad = ops.reshape(epsilon_grad, ops.shape(eps))
            return (
                ops.cast(dot_grad, source_dtype),
                ops.cast(distance_grad, source_dtype),
                ops.cast(epsilon_grad, source_dtype),
            )

        return result, grad

    return low_precision_ratio(dot_product, distance_sq, epsilon)


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

    # Squared distances assembled from norms and dot products are susceptible
    # to cancellation in float16/bfloat16.  A squared distance is
    # mathematically non-negative, so clamp before adding epsilon.  Keeping the
    # clamp here guarantees identical behaviour for all forward and transpose
    # convolution variants.
    # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps).
    outputs = stable_yat_ratio(dot_prod_map, distance_sq_map, eps)

    # Optional alpha (constant via _constant_alpha_value is folded into
    # `layer.alpha` at __init__ time; here we only need `use_alpha` + alpha).
    if layer.use_alpha and layer.alpha is not None:
        outputs = outputs * layer.alpha

    return outputs
