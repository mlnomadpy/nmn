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


__all__ = [
    "reduction_safe_upcast",
    "saturating_downcast",
    "stable_yat_ratio",
    "yat_score",
]


_LOW_PRECISION_DTYPES = {"float16", "bfloat16"}


def _dtype_max(dtype):
    return 65504.0 if dtype == "float16" else 3.38953139e38


def reduction_safe_upcast(value):
    """Upcast low-precision reduction inputs and saturate their final cotangent.

    A regular lowp-to-fp32 cast downcasts its backward result without guarding
    the already-reduced cotangent.  Multi-output conv/matmul gradients can then
    become ``inf`` at the leaf even though every individual contribution is
    finite.  This boundary keeps the reduction in fp32 and performs exactly one
    saturating cast when its aggregate cotangent returns to the lowp leaf.
    """
    dtype = standardize_dtype(value.dtype)
    if dtype not in _LOW_PRECISION_DTYPES:
        return ops.convert_to_tensor(value)
    # Materialize Keras Variables *outside* custom_gradient.  Torch's
    # autograd.Function only tracks Tensor arguments, while TensorFlow treats
    # a resource read performed inside a custom forward as a captured variable
    # and then requires a separate ``variables=`` gradient result.  A
    # dtype-changing cast creates a tracked tensor boundary without detaching
    # the underlying parameter on any backend.  The custom identity below
    # clips its fp32 cotangent before the ordinary cast backward converts that
    # finite value to the low-precision leaf dtype.
    value = ops.cast(value, "float32")
    max_value = _dtype_max(dtype)

    @ops.custom_gradient
    def upcast(x):
        result = x

        def grad(*args, upstream=None, variables=None):
            if upstream is None:
                (upstream,) = args
            # clip preserves NaNs on the supported Keras backends while
            # saturating both finite overflow and +/-inf.
            upstream32 = ops.cast(upstream, "float32")
            saturated = ops.clip(upstream32, -max_value, max_value)
            input_grad = saturated
            if variables is not None:
                # This is a defensive TensorFlow-contract fallback.  The
                # materializing cast above ensures current callers have no
                # variables captured inside this custom forward.
                variable_grads = [
                    ops.cast(input_grad, variable.dtype) for variable in variables
                ]
                return input_grad, variable_grads
            return input_grad

        return result, grad

    return upcast(value)


def saturating_downcast(value, dtype):
    """Cast an fp32 result to a lowp policy without overflow or NaN masking."""
    dtype = standardize_dtype(dtype)
    if dtype not in _LOW_PRECISION_DTYPES:
        return ops.cast(value, dtype)
    max_value = _dtype_max(dtype)

    @ops.custom_gradient
    def downcast(x):
        active = ops.logical_and(x > -max_value, x < max_value)
        result = ops.cast(ops.clip(x, -max_value, max_value), dtype)

        def grad(*args, upstream=None, variables=None):
            if upstream is None:
                (upstream,) = args
            upstream32 = ops.cast(upstream, "float32")
            # Comparisons are false for NaN.  Pass its cotangent through so the
            # upstream arithmetic remains NaN instead of silently zeroing it.
            active_or_nan = ops.logical_or(active, ops.isnan(x))
            input_grad = upstream32 * ops.cast(active_or_nan, "float32")
            if variables is not None:
                variable_grads = [
                    ops.cast(input_grad, variable.dtype) for variable in variables
                ]
                return input_grad, variable_grads
            return input_grad

        return result, grad

    return downcast(ops.cast(value, "float32"))


def stable_yat_ratio(dot_product, distance_sq, epsilon, output_dtype=None):
    """Evaluate the YAT ratio without low-precision overflow/cancellation."""
    source_dtype = standardize_dtype(dot_product.dtype)
    output_dtype = source_dtype if output_dtype is None else output_dtype
    dot_product = reduction_safe_upcast(dot_product)
    distance_sq = reduction_safe_upcast(distance_sq)
    if hasattr(epsilon, "dtype"):
        epsilon = reduction_safe_upcast(epsilon)
    else:
        epsilon = ops.cast(epsilon, "float32")
    distance_sq = ops.maximum(distance_sq, ops.cast(0.0, distance_sq.dtype))
    ratio = ops.square(dot_product) / (distance_sq + epsilon)
    return saturating_downcast(ratio, output_dtype)


def yat_score(layer, dot_prod_map, distance_sq_map, data_format=None):
    """Apply bias / epsilon / YAT-divide / alpha to a raw conv output.

    Returns ``(dot_prod_map + bias) ** 2 / (distance_sq_map + eps) * alpha``.

    The `layer` is expected to expose:

    * ``use_bias``, ``_constant_bias_value``, ``bias``, ``data_format``,
      ``kernel_size`` — for bias resolution and channels-first reshape.
    * ``learnable_epsilon``, ``epsilon_param``, ``epsilon`` — for the
      effective epsilon (softplus-of-raw or constant).
    * ``use_alpha``, ``alpha`` — for the optional alpha multiplier.

    `dot_prod_map` and `distance_sq_map` must already be in `data_format`, which
    defaults to the layer's public output layout.
    """
    data_format = layer.data_format if data_format is None else data_format
    # Add bias before squaring (constant or learnable; reshape for channels_first).
    if layer.use_bias:
        if layer._constant_bias_value is not None:
            dot_prod_map = dot_prod_map + layer._constant_bias_value
        else:
            bias = reduction_safe_upcast(layer.bias)
            if data_format == "channels_first":
                bias_shape = (1, -1) + (1,) * len(layer.kernel_size)
                bias = ops.reshape(bias, bias_shape)
            dot_prod_map = ops.add(dot_prod_map, bias)

    # Resolve effective epsilon (learnable via softplus, or constant).
    if layer.learnable_epsilon and layer.epsilon_param is not None:
        eps = ops.softplus(reduction_safe_upcast(layer.epsilon_param))
    else:
        eps = ops.cast(layer.epsilon, "float32")

    # Squared distances assembled from norms and dot products are susceptible
    # to cancellation in float16/bfloat16.  A squared distance is
    # mathematically non-negative, so clamp before adding epsilon.  Keeping the
    # clamp here guarantees identical behaviour for all forward and transpose
    # convolution variants.
    # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps).
    outputs = stable_yat_ratio(
        dot_prod_map, distance_sq_map, eps, output_dtype="float32"
    )

    # Optional alpha (constant via _constant_alpha_value is folded into
    # `layer.alpha` at __init__ time; here we only need `use_alpha` + alpha).
    if layer.use_alpha and layer.alpha is not None:
        outputs = outputs * reduction_safe_upcast(layer.alpha)

    return saturating_downcast(outputs, layer.compute_dtype)
