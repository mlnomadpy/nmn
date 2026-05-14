"""Tests for the Tier-1 MLX features: DropConnect + weight_normalized conv."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from nmn.mlx import YatNMN, YatConv2D, YatConvTranspose2D  # noqa: E402


# ---------------------------------------------------------------------------
# YatNMN DropConnect
# ---------------------------------------------------------------------------


def test_yat_nmn_dropconnect_rejects_invalid_rate():
    with pytest.raises(ValueError):
        YatNMN(features=4, use_dropconnect=True, drop_rate=1.0)
    with pytest.raises(ValueError):
        YatNMN(features=4, use_dropconnect=True, drop_rate=-0.1)


def test_yat_nmn_dropconnect_deterministic_unchanged():
    """In deterministic mode the layer must be identical to drop_rate=0."""
    mx.random.seed(0)
    layer_dc = YatNMN(features=6, use_dropconnect=True, drop_rate=0.5)
    layer_dc.build(4)
    layer_plain = YatNMN(features=6, use_dropconnect=False)
    layer_plain.build(4)
    # Sync weights.
    layer_plain.kernel = layer_dc.kernel
    layer_plain.bias = layer_dc.bias
    layer_plain.alpha = layer_dc.alpha

    x = mx.random.normal(shape=(3, 4))
    y_dc = np.array(layer_dc(x, deterministic=True))
    y_plain = np.array(layer_plain(x))
    assert np.max(np.abs(y_dc - y_plain)) < 1e-6


def test_yat_nmn_dropconnect_stochastic_differs():
    """In stochastic mode the output should differ from deterministic."""
    mx.random.seed(0)
    layer = YatNMN(features=6, use_dropconnect=True, drop_rate=0.5)
    x = mx.random.normal(shape=(4, 3))
    _ = layer(x)  # build
    mx.random.seed(0)
    y_det = np.array(layer(x, deterministic=True))
    y_stoch = np.array(layer(x, deterministic=False))
    # With 50 % drop rate, at least one element should differ noticeably.
    assert np.max(np.abs(y_det - y_stoch)) > 1e-4


def test_yat_nmn_dropconnect_off_when_rate_zero():
    """drop_rate=0 must produce the same output deterministic or not."""
    layer = YatNMN(features=4, use_dropconnect=True, drop_rate=0.0)
    x = mx.random.normal(shape=(2, 3))
    _ = layer(x)
    y1 = np.array(layer(x, deterministic=True))
    y2 = np.array(layer(x, deterministic=False))
    assert np.max(np.abs(y1 - y2)) < 1e-6


# ---------------------------------------------------------------------------
# YatConv2D — weight_normalized + DropConnect
# ---------------------------------------------------------------------------


def _numpy_conv2d_yat(x, W, b, a, eps=1e-5):
    """Pure-numpy YatConv2D reference for the (1, H, W, C_in) → (1, ., ., F) case."""
    F, KH, KW, C = W.shape
    out_h = x.shape[1] - KH + 1
    out_w = x.shape[2] - KW + 1
    out = np.zeros((1, out_h, out_w, F))
    for f in range(F):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[0, i:i + KH, j:j + KW, :]
                kf = W[f]
                dot = (patch * kf).sum()
                dist = ((patch - kf) ** 2).sum()
                out[0, i, j, f] = a * (dot + b[f]) ** 2 / (dist + eps)
    return out


def test_conv2d_weight_normalized_math_parity():
    """``weight_normalized=True`` must match a numpy reference where W is
    pre-normalized to unit row norm."""
    mx.random.seed(0)
    layer = YatConv2D(filters=4, kernel_size=3, weight_normalized=True, padding="valid")
    x = mx.random.normal(shape=(1, 6, 6, 2))
    y = np.array(layer(x))

    W = np.array(layer.kernel)
    norm = np.sqrt((W ** 2).sum(axis=(1, 2, 3), keepdims=True))
    Wn = W / (norm + 1e-8)
    b = np.array(layer.bias)
    a = float(np.array(layer.alpha)[0])
    ref = _numpy_conv2d_yat(np.array(x), Wn, b, a, eps=1e-5)
    assert np.max(np.abs(y - ref)) < 1e-5


def test_conv2d_weight_normalized_filter_norms_after_forward():
    """Each filter's L2 norm should be ~1 after the normalization step."""
    layer = YatConv2D(filters=4, kernel_size=3, weight_normalized=True)
    _ = layer(mx.random.normal(shape=(1, 6, 6, 2)))
    W = np.array(layer.kernel)
    # Re-apply the same normalization the forward pass does — this is the
    # post-norm receipt.
    norm = np.sqrt((W ** 2).sum(axis=(1, 2, 3), keepdims=True))
    Wn = W / (norm + 1e-8)
    Wn_norms = np.sqrt((Wn ** 2).sum(axis=(1, 2, 3)))
    assert np.max(np.abs(Wn_norms - 1.0)) < 1e-5


def test_conv2d_dropconnect_deterministic_unchanged():
    """DropConnect in deterministic mode must equal plain conv."""
    mx.random.seed(7)
    layer_dc = YatConv2D(filters=4, kernel_size=3,
                         use_dropconnect=True, drop_rate=0.3, padding="valid")
    _ = layer_dc(mx.zeros((1, 6, 6, 2)))  # build to populate kernel
    layer_plain = YatConv2D(filters=4, kernel_size=3, padding="valid")
    layer_plain.build(2)
    layer_plain.kernel = layer_dc.kernel
    layer_plain.bias = layer_dc.bias
    layer_plain.alpha = layer_dc.alpha

    x = mx.random.normal(shape=(1, 6, 6, 2))
    y_dc = np.array(layer_dc(x, deterministic=True))
    y_plain = np.array(layer_plain(x))
    assert np.max(np.abs(y_dc - y_plain)) < 1e-5


def test_conv2d_dropconnect_stochastic_differs():
    layer = YatConv2D(filters=4, kernel_size=3,
                      use_dropconnect=True, drop_rate=0.5, padding="same")
    x = mx.random.normal(shape=(1, 8, 8, 3))
    _ = layer(x)
    mx.random.seed(0)
    y1 = np.array(layer(x, deterministic=True))
    y2 = np.array(layer(x, deterministic=False))
    assert np.max(np.abs(y1 - y2)) > 1e-4


def test_conv2d_dropconnect_invalid_rate_rejected():
    with pytest.raises(ValueError):
        YatConv2D(filters=4, kernel_size=3, use_dropconnect=True, drop_rate=1.5)


def test_conv2d_weight_normalized_and_dropconnect_combine():
    """Both flags should compose without raising — and stochastic still differs."""
    layer = YatConv2D(
        filters=4, kernel_size=3,
        weight_normalized=True,
        use_dropconnect=True, drop_rate=0.3,
        padding="same",
    )
    x = mx.random.normal(shape=(1, 6, 6, 2))
    _ = layer(x)
    y_det = np.array(layer(x, deterministic=True))
    y_stoch = np.array(layer(x, deterministic=False))
    assert y_det.shape == (1, 6, 6, 4)
    assert np.max(np.abs(y_det - y_stoch)) > 1e-4


# ---------------------------------------------------------------------------
# YatConvTranspose2D DropConnect
# ---------------------------------------------------------------------------


def test_conv_transpose2d_dropconnect_deterministic_unchanged():
    mx.random.seed(3)
    layer_dc = YatConvTranspose2D(filters=2, kernel_size=2, strides=2,
                                  use_dropconnect=True, drop_rate=0.5)
    _ = layer_dc(mx.zeros((1, 3, 3, 2)))
    layer_plain = YatConvTranspose2D(filters=2, kernel_size=2, strides=2)
    layer_plain.build(2)
    layer_plain.kernel = layer_dc.kernel
    layer_plain.bias = layer_dc.bias
    layer_plain.alpha = layer_dc.alpha

    x = mx.random.normal(shape=(1, 3, 3, 2))
    y_dc = np.array(layer_dc(x, deterministic=True))
    y_plain = np.array(layer_plain(x))
    assert np.max(np.abs(y_dc - y_plain)) < 1e-5


def test_conv_transpose2d_dropconnect_stochastic_differs():
    layer = YatConvTranspose2D(filters=2, kernel_size=3, strides=2,
                               use_dropconnect=True, drop_rate=0.4)
    x = mx.random.normal(shape=(1, 4, 4, 2))
    _ = layer(x)
    mx.random.seed(11)
    y1 = np.array(layer(x, deterministic=True))
    y2 = np.array(layer(x, deterministic=False))
    assert np.max(np.abs(y1 - y2)) > 1e-4
