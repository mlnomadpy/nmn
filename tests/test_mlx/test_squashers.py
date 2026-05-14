"""Tests for the MLX squashing functions."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from nmn.mlx.squashers import softermax, softer_sigmoid, soft_tanh  # noqa: E402


def test_softermax_sums_to_one():
    x = mx.array([1.0, 2.0, 3.0, 4.0])
    y = np.array(softermax(x, n=1.0, epsilon=0.0))
    assert abs(y.sum() - 1.0) < 1e-6


def test_softermax_n_2_matches_formula():
    x = mx.array([1.0, 2.0, 3.0])
    y = np.array(softermax(x, n=2.0, epsilon=0.0))
    # x² / sum(x²) = [1, 4, 9] / 14
    expected = np.array([1.0, 4.0, 9.0]) / 14.0
    assert np.max(np.abs(y - expected)) < 1e-6


def test_softermax_rejects_non_positive_n():
    with pytest.raises(ValueError):
        softermax(mx.array([1.0, 2.0]), n=0.0)
    with pytest.raises(ValueError):
        softermax(mx.array([1.0, 2.0]), n=-1.0)


def test_softer_sigmoid_range():
    x = mx.array([0.0, 1.0, 10.0, 100.0])
    y = np.array(softer_sigmoid(x, n=1.0))
    assert (y >= 0).all()
    assert (y < 1.0).all()
    # x=0  -> 0,  x=1 -> 0.5,  x→∞ → 1.
    assert abs(float(y[0])) < 1e-6
    assert abs(float(y[1]) - 0.5) < 1e-6
    assert float(y[-1]) > 0.99


def test_soft_tanh_range():
    x = mx.array([0.0, 1.0, 10.0, 1000.0])
    y = np.array(soft_tanh(x, n=1.0))
    # soft_tanh(0) = -1, soft_tanh(1) = 0, soft_tanh(∞) → 1.
    # Closed form: (x - 1) / (x + 1), so x=1000 → 999/1001 ≈ 0.998.
    assert abs(float(y[0]) - (-1.0)) < 1e-6
    assert abs(float(y[1])) < 1e-6
    assert float(y[-1]) > 0.99
    assert (y >= -1.0).all()
    assert (y < 1.0).all()


def test_softer_sigmoid_rejects_non_positive_n():
    with pytest.raises(ValueError):
        softer_sigmoid(mx.array([1.0]), n=0.0)


def test_soft_tanh_rejects_non_positive_n():
    with pytest.raises(ValueError):
        soft_tanh(mx.array([1.0]), n=-2.0)
