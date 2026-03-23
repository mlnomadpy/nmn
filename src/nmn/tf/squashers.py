"""Squashing functions for TensorFlow.

Provides softermax, softer_sigmoid, and soft_tanh — activation-like functions
that use power-based normalization instead of exponentials.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf


@tf.function
def softermax(
    x: tf.Tensor,
    n: float = 1.0,
    epsilon: float = 1e-12,
    axis: Optional[int] = -1,
) -> tf.Tensor:
    """Normalizes non-negative scores using the Softermax function.

    softermax_n(x_k) = x_k^n / (epsilon + sum_i x_i^n)

    Args:
        x: Tensor of non-negative scores.
        n: Power parameter (must be positive). Higher = sharper.
        epsilon: Small constant for numerical stability.
        axis: Axis to normalize over.

    Returns:
        Normalized scores summing to ~1 along axis.
    """
    x_n = tf.pow(x, n)
    sum_x_n = tf.reduce_sum(x_n, axis=axis, keepdims=True)
    return x_n / (epsilon + sum_x_n)


@tf.function
def softer_sigmoid(
    x: tf.Tensor,
    n: float = 1.0,
) -> tf.Tensor:
    """Squashes non-negative scores into [0, 1) using soft-sigmoid.

    soft_sigmoid_n(x) = x^n / (1 + x^n)

    Args:
        x: Tensor of non-negative scores (x >= 0).
        n: Power parameter (must be positive).

    Returns:
        Squashed scores in [0, 1).
    """
    x_n = tf.pow(x, n)
    return x_n / (1.0 + x_n)


@tf.function
def soft_tanh(
    x: tf.Tensor,
    n: float = 1.0,
) -> tf.Tensor:
    """Maps non-negative scores to [-1, 1) using soft-tanh.

    soft_tanh_n(x) = (x^n - 1) / (1 + x^n)

    Args:
        x: Tensor of non-negative scores (x >= 0).
        n: Power parameter (must be positive).

    Returns:
        Mapped scores in [-1, 1).
    """
    x_n = tf.pow(x, n)
    return (x_n - 1.0) / (1.0 + x_n)
