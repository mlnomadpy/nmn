"""Squashing functions for MLX.

Provides softermax, softer_sigmoid, and soft_tanh — activation-like functions
that use power-based normalization instead of exponentials. Mirrors
``nmn.tf.squashers`` / ``nmn.nnx.layers.squashers``.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def softermax(
    x: mx.array,
    n: float = 1.0,
    epsilon: float = 1e-12,
    axis: Optional[int] = -1,
) -> mx.array:
    """Normalizes non-negative scores using the Softermax function.

    ``softermax_n(x_k) = x_k^n / (epsilon + sum_i x_i^n)``

    Args:
        x: Array of non-negative scores.
        n: Power parameter (must be positive). Higher = sharper.
        epsilon: Small constant for numerical stability.
        axis: Axis to normalize over.

    Returns:
        Normalized scores summing to ~1 along ``axis``.
    """
    if n <= 0:
        raise ValueError("Power 'n' must be positive.")
    x_n = mx.power(x, n)
    sum_x_n = mx.sum(x_n, axis=axis, keepdims=True)
    return x_n / (epsilon + sum_x_n)


def softer_sigmoid(x: mx.array, n: float = 1.0) -> mx.array:
    """Squashes non-negative scores into [0, 1) using soft-sigmoid.

    ``soft_sigmoid_n(x) = x^n / (1 + x^n)``
    """
    if n <= 0:
        raise ValueError("Power 'n' must be positive.")
    x_n = mx.power(x, n)
    return x_n / (1.0 + x_n)


def soft_tanh(x: mx.array, n: float = 1.0) -> mx.array:
    """Maps non-negative scores to [-1, 1) using soft-tanh.

    ``soft_tanh_n(x) = (x^n - 1) / (1 + x^n)``
    """
    if n <= 0:
        raise ValueError("Power 'n' must be positive.")
    x_n = mx.power(x, n)
    return (x_n - 1.0) / (1.0 + x_n)
