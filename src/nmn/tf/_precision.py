"""Autodiff-safe precision helpers for TensorFlow YAT layers."""

from __future__ import annotations

import tensorflow as tf


def reduction_safe_upcast(value: tf.Tensor) -> tf.Tensor:
    """Upcast lowp values and saturate their aggregated return cotangent."""
    value = tf.convert_to_tensor(value)
    if value.dtype not in (tf.float16, tf.bfloat16):
        return value
    max_value = 65504.0 if value.dtype == tf.float16 else 3.38953139e38
    source_dtype = value.dtype

    @tf.custom_gradient
    def upcast(tensor):
        result = tf.cast(tensor, tf.float32)

        def grad(upstream):
            upstream = tf.cast(upstream, tf.float32)
            upstream = tf.clip_by_value(upstream, -max_value, max_value)
            return tf.cast(upstream, source_dtype)

        return result, grad

    return upcast(value)
