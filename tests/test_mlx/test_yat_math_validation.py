"""Bit-by-bit (within fp32 epsilon) validation of the YAT formula.

Compares the MLX layer output against a pure-numpy reference for every
combination of bias / alpha / spherical / weight_normalized / learnable
epsilon supported by the Tier-0 backend.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from nmn.mlx import YatNMN  # noqa: E402


def _numpy_yat(
    x: np.ndarray,
    W: np.ndarray,
    *,
    b: np.ndarray | float | None = None,
    alpha: float | None = None,
    epsilon: float = 1e-5,
    spherical: bool = False,
    weight_normalized: bool = False,
) -> np.ndarray:
    """Reference YAT forward pass.

    ``W`` has shape ``(features, in_features)`` to match the MLX layer's
    ``Linear``-style storage.
    """
    if spherical:
        x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        W = W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-8)
    if weight_normalized:
        W = W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-8)

    dot = x @ W.T
    if spherical:
        dist = np.maximum(2.0 - 2.0 * dot, 0.0)
    else:
        x_sq = (x ** 2).sum(axis=-1, keepdims=True)
        if weight_normalized:
            w_sq = np.ones(W.shape[0], dtype=W.dtype)
        else:
            w_sq = (W ** 2).sum(axis=-1)
        w_sq = w_sq.reshape((1,) * (dot.ndim - 1) + (W.shape[0],))
        dist = np.maximum(x_sq + w_sq - 2 * dot, 0.0)

    num = dot if b is None else dot + b
    y = (num ** 2) / (dist + epsilon)
    if alpha is not None:
        y = y * alpha
    return y


def _extract(layer: YatNMN) -> tuple[np.ndarray, np.ndarray | float | None, float | None]:
    W = np.array(layer.kernel)
    if layer.use_bias:
        if layer._constant_bias_value is not None:
            b: np.ndarray | float | None = float(layer._constant_bias_value)
        else:
            b = np.array(layer.bias)
    else:
        b = None
    if layer._constant_alpha_value is not None:
        a: float | None = float(layer._constant_alpha_value)
    elif layer.use_alpha:
        a = float(np.array(layer.alpha)[0])
    else:
        a = None
    return W, b, a


@pytest.mark.parametrize(
    "use_bias,use_alpha,constant_bias,constant_alpha",
    [
        (True, True, None, None),
        (False, True, None, None),
        (True, False, None, None),
        (False, False, None, None),
        (True, True, 0.25, None),
        (True, True, None, True),
        (True, True, None, 1.5),
    ],
)
def test_yat_math_all_modes(use_bias, use_alpha, constant_bias, constant_alpha):
    mx.random.seed(42)
    layer = YatNMN(
        features=7,
        use_bias=use_bias,
        constant_bias=constant_bias,
        use_alpha=use_alpha,
        constant_alpha=constant_alpha,
    )
    x = mx.random.normal(shape=(4, 5))
    y = np.array(layer(x))
    W, b, a = _extract(layer)
    ref = _numpy_yat(np.array(x), W, b=b, alpha=a, epsilon=1e-5)
    diff = np.max(np.abs(y - ref))
    assert diff < 1e-5, f"max diff {diff} exceeds tolerance"


@pytest.mark.parametrize("spherical,weight_normalized", list(product([False, True], repeat=2)))
def test_yat_math_geometry_modes(spherical, weight_normalized):
    if spherical and weight_normalized:
        pytest.skip("spherical implies its own normalization; combination is redundant")
    mx.random.seed(7)
    layer = YatNMN(
        features=6,
        spherical=spherical,
        weight_normalized=weight_normalized,
    )
    x = mx.random.normal(shape=(3, 4))
    y = np.array(layer(x))
    W, b, a = _extract(layer)
    ref = _numpy_yat(
        np.array(x), W,
        b=b, alpha=a, epsilon=1e-5,
        spherical=spherical, weight_normalized=weight_normalized,
    )
    diff = np.max(np.abs(y - ref))
    assert diff < 1e-5, f"max diff {diff} exceeds tolerance"


def test_yat_math_learnable_epsilon():
    """With a learnable ε softplus-reparameterized to 1e-3 at init, the
    forward pass should match the numpy reference at ε=1e-3."""
    mx.random.seed(11)
    layer = YatNMN(features=4, learnable_epsilon=True, epsilon=1e-3)
    x = mx.random.normal(shape=(2, 3))
    y = np.array(layer(x))
    W, b, a = _extract(layer)
    ref = _numpy_yat(np.array(x), W, b=b, alpha=a, epsilon=1e-3)
    diff = np.max(np.abs(y - ref))
    assert diff < 1e-5, f"max diff {diff} exceeds tolerance"


def test_yat_output_non_negative_when_bias_zero():
    """With no bias and no learnable alpha, output is a squared ratio of
    non-negatives → strictly non-negative."""
    mx.random.seed(3)
    layer = YatNMN(features=5, use_bias=False, use_alpha=False)
    x = mx.random.normal(shape=(8, 6))
    y = np.array(layer(x))
    assert float(np.min(y)) >= 0.0


def test_default_constant_alpha_value():
    """``constant_alpha=True`` → √2 exactly."""
    layer = YatNMN(features=3, constant_alpha=True)
    _ = layer(mx.zeros((1, 2)))
    assert layer._constant_alpha_value == math.sqrt(2.0)
