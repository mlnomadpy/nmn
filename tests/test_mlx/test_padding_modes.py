"""Tests for the CIRCULAR / REFLECT / CAUSAL conv padding modes."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from nmn.mlx import YatConv1D, YatConv2D  # noqa: E402
from nmn.mlx.conv import _prepad_input  # noqa: E402


# ---------------------------------------------------------------------------
# _prepad_input — bit-for-bit numpy parity
# ---------------------------------------------------------------------------


def test_prepad_circular_1d_matches_numpy_wrap():
    xn = np.arange(7, dtype=np.float32)
    x = mx.array(xn.reshape(1, 7, 1))
    out = np.array(_prepad_input(x, "CIRCULAR", (5,), (1,))).flatten()
    np.testing.assert_array_equal(out, np.pad(xn, (2, 2), "wrap"))


def test_prepad_reflect_1d_matches_numpy_reflect():
    xn = np.arange(7, dtype=np.float32)
    x = mx.array(xn.reshape(1, 7, 1))
    out = np.array(_prepad_input(x, "REFLECT", (5,), (1,))).flatten()
    np.testing.assert_array_equal(out, np.pad(xn, (2, 2), "reflect"))


def test_prepad_circular_2d_matches_numpy_wrap():
    xn = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = mx.array(xn.reshape(1, 4, 4, 1))
    out = np.array(_prepad_input(x, "CIRCULAR", (3, 3), (1, 1)))
    np.testing.assert_array_equal(
        out.reshape(6, 6), np.pad(xn, ((1, 1), (1, 1)), "wrap")
    )


def test_prepad_reflect_2d_matches_numpy_reflect():
    xn = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = mx.array(xn.reshape(1, 4, 4, 1))
    out = np.array(_prepad_input(x, "REFLECT", (3, 3), (1, 1)))
    np.testing.assert_array_equal(
        out.reshape(6, 6), np.pad(xn, ((1, 1), (1, 1)), "reflect")
    )


def test_prepad_causal_1d_left_zero_only():
    xn = np.arange(5, dtype=np.float32)
    x = mx.array(xn.reshape(1, 5, 1))
    out = np.array(_prepad_input(x, "CAUSAL", (3,), (1,))).flatten()
    np.testing.assert_array_equal(out, np.pad(xn, (2, 0), "constant"))


def test_prepad_causal_with_dilation():
    """left_pad = dilation * (kernel_size - 1)."""
    xn = np.arange(4, dtype=np.float32)
    x = mx.array(xn.reshape(1, 4, 1))
    out = np.array(_prepad_input(x, "CAUSAL", (3,), (2,))).flatten()
    np.testing.assert_array_equal(out, np.pad(xn, (4, 0), "constant"))


def test_prepad_causal_rejects_multi_dim():
    x = mx.zeros((1, 4, 4, 1))
    with pytest.raises(ValueError):
        _prepad_input(x, "CAUSAL", (3, 3), (1, 1))


def test_prepad_circular_rejects_too_large_kernel():
    x = mx.zeros((1, 3, 1))  # spatial size 3
    with pytest.raises(ValueError):
        _prepad_input(x, "CIRCULAR", (9,), (1,))


def test_prepad_reflect_rejects_too_large_kernel():
    x = mx.zeros((1, 3, 1))
    with pytest.raises(ValueError):
        _prepad_input(x, "REFLECT", (9,), (1,))


# ---------------------------------------------------------------------------
# Conv forward — output shape matches input size for SAME-like modes
# ---------------------------------------------------------------------------


def test_conv1d_circular_preserves_length():
    layer = YatConv1D(filters=4, kernel_size=3, padding="circular")
    out = layer(mx.random.normal(shape=(1, 16, 2)))
    assert out.shape == (1, 16, 4)


def test_conv1d_reflect_preserves_length():
    layer = YatConv1D(filters=4, kernel_size=5, padding="reflect")
    out = layer(mx.random.normal(shape=(1, 16, 2)))
    assert out.shape == (1, 16, 4)


def test_conv1d_causal_preserves_length():
    layer = YatConv1D(filters=4, kernel_size=3, padding="causal")
    out = layer(mx.random.normal(shape=(1, 16, 2)))
    assert out.shape == (1, 16, 4)


def test_conv2d_circular_preserves_shape():
    layer = YatConv2D(filters=4, kernel_size=3, padding="circular")
    out = layer(mx.random.normal(shape=(1, 12, 12, 2)))
    assert out.shape == (1, 12, 12, 4)


def test_conv2d_reflect_preserves_shape():
    layer = YatConv2D(filters=4, kernel_size=5, padding="reflect")
    out = layer(mx.random.normal(shape=(1, 12, 12, 2)))
    assert out.shape == (1, 12, 12, 4)


def test_conv_causal_only_in_1d():
    layer = YatConv2D(filters=4, kernel_size=3, padding="causal")
    with pytest.raises(ValueError):
        layer(mx.random.normal(shape=(1, 8, 8, 2)))


# ---------------------------------------------------------------------------
# Causal correctness: output[t] must not depend on input[t' > t]
# ---------------------------------------------------------------------------


def test_conv1d_causal_does_not_leak_future():
    mx.random.seed(0)
    layer = YatConv1D(filters=4, kernel_size=3, padding="causal")
    x = mx.random.normal(shape=(1, 10, 2))
    out_a = np.array(layer(x))
    # Perturb the last timestep — output at t=0..3 must be unchanged
    # (kernel size 3, dilation 1 → each position only sees the 2 prior
    # timesteps + itself).
    x_perturbed = mx.concatenate([x[:, :9], x[:, 9:10] + 100.0], axis=1)
    out_b = np.array(layer(x_perturbed))
    assert np.max(np.abs(out_a[:, :9] - out_b[:, :9])) < 1e-4
    # But timestep 9 changes.
    assert np.max(np.abs(out_a[:, 9] - out_b[:, 9])) > 1e-2


# ---------------------------------------------------------------------------
# YAT math parity for CIRCULAR mode (sanity)
# ---------------------------------------------------------------------------


def test_conv1d_circular_yat_math_parity():
    """Compare against a hand-rolled numpy reference where we explicitly
    pre-pad the input with 'wrap' and run the YAT formula."""
    mx.random.seed(0)
    layer = YatConv1D(filters=3, kernel_size=3, padding="circular")
    x = mx.random.normal(shape=(1, 6, 2))
    y = np.array(layer(x))

    W = np.array(layer.kernel)  # (3, 3, 2)
    b = np.array(layer.bias)
    a = float(np.array(layer.alpha)[0])
    xn = np.array(x)[0]                                  # (6, 2)
    xn_padded = np.pad(xn, ((1, 1), (0, 0)), "wrap")     # (8, 2)

    out_len = 6
    ref = np.zeros((1, out_len, 3))
    for f in range(3):
        for t in range(out_len):
            patch = xn_padded[t:t + 3]                   # (3, 2)
            kf = W[f]                                    # (3, 2) per MLX (C_out, K, C_in)
            dot = (patch * kf).sum()
            dist = ((patch - kf) ** 2).sum()
            ref[0, t, f] = a * (dot + b[f]) ** 2 / (dist + 1e-5)
    assert np.max(np.abs(y - ref)) < 1e-5
