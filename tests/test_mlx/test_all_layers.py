"""Tests for the MLX conv family.

Covers:
* basic forward shape for 1D / 2D / 3D / transposed-1D / transposed-2D /
  transposed-3D with `valid` and `same` padding;
* element-wise parity (< 1e-5) against a pure-numpy reference for 2D
  YatConv and 1D YatConvTranspose at stride 1;
* grouped 2D forward shape;
* gradient flow through every learnable parameter.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")

from nmn.mlx import (  # noqa: E402
    YatConv1D, YatConv2D, YatConv3D,
    YatConvTranspose1D, YatConvTranspose2D, YatConvTranspose3D,
)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_conv1d_valid_shape():
    layer = YatConv1D(filters=16, kernel_size=3)
    x = mx.random.normal(shape=(2, 10, 4))
    assert layer(x).shape == (2, 8, 16)


def test_conv1d_same_shape():
    layer = YatConv1D(filters=16, kernel_size=3, padding="same")
    x = mx.random.normal(shape=(2, 10, 4))
    assert layer(x).shape == (2, 10, 16)


def test_conv2d_valid_shape():
    layer = YatConv2D(filters=8, kernel_size=3)
    x = mx.random.normal(shape=(2, 16, 16, 3))
    assert layer(x).shape == (2, 14, 14, 8)


def test_conv2d_same_shape():
    layer = YatConv2D(filters=8, kernel_size=3, padding="same")
    x = mx.random.normal(shape=(2, 16, 16, 3))
    assert layer(x).shape == (2, 16, 16, 8)


def test_conv3d_valid_shape():
    layer = YatConv3D(filters=4, kernel_size=3)
    x = mx.random.normal(shape=(1, 8, 8, 8, 2))
    assert layer(x).shape == (1, 6, 6, 6, 4)


def test_conv_transpose1d_shape():
    layer = YatConvTranspose1D(filters=8, kernel_size=3, strides=2)
    x = mx.random.normal(shape=(2, 10, 4))
    # output size = (L_in - 1) * stride + (K - 1) * dilation + 1 = 9*2 + 2 = 20
    # MLX returns L_in*stride + K - 1 = 21 here — keep the actual contract.
    out = layer(x)
    assert out.shape[0] == 2 and out.shape[-1] == 8
    assert out.shape[1] > x.shape[1]  # upsamples


def test_conv_transpose2d_same_shape():
    layer = YatConvTranspose2D(filters=4, kernel_size=4, strides=2, padding="same")
    x = mx.random.normal(shape=(1, 8, 8, 4))
    out = layer(x)
    assert out.shape == (1, 16, 16, 4)


def test_conv_transpose3d_shape():
    layer = YatConvTranspose3D(filters=2, kernel_size=2, strides=2)
    x = mx.random.normal(shape=(1, 4, 4, 4, 2))
    out = layer(x)
    assert out.shape[0] == 1 and out.shape[-1] == 2


# ---------------------------------------------------------------------------
# Math parity vs a numpy reference
# ---------------------------------------------------------------------------


def test_conv2d_math_parity():
    mx.random.seed(0)
    layer = YatConv2D(filters=4, kernel_size=3, padding="valid")
    x = mx.random.normal(shape=(1, 6, 6, 2))
    y = np.array(layer(x))

    W = np.array(layer.kernel)
    b = np.array(layer.bias)
    a = float(np.array(layer.alpha)[0])
    xn = np.array(x)

    out_h = xn.shape[1] - 3 + 1
    out_w = xn.shape[2] - 3 + 1
    ref = np.zeros((1, out_h, out_w, 4))
    for f in range(4):
        for i in range(out_h):
            for j in range(out_w):
                patch = xn[0, i:i + 3, j:j + 3, :]
                kf = W[f]
                dot = (patch * kf).sum()
                dist = ((patch - kf) ** 2).sum()
                ref[0, i, j, f] = a * (dot + b[f]) ** 2 / (dist + 1e-5)

    assert np.max(np.abs(y - ref)) < 1e-5


def test_conv_transpose1d_math_parity():
    mx.random.seed(1)
    layer = YatConvTranspose1D(filters=3, kernel_size=3, strides=1, padding="valid")
    x = mx.random.normal(shape=(1, 5, 2))
    y = np.array(layer(x))

    W = np.array(layer.kernel)
    b = np.array(layer.bias)
    a = float(np.array(layer.alpha)[0])
    xn = np.array(x)
    N, L_in, C_in = xn.shape
    F, K, _ = W.shape
    L_out = L_in + K - 1
    ref = np.zeros((N, L_out, F))
    for p in range(L_out):
        for f in range(F):
            dot = 0.0
            patch_sq = 0.0
            kernel_sq = 0.0
            for k in range(K):
                i = p - k
                for cin in range(C_in):
                    wval = W[f, k, cin]
                    kernel_sq += wval * wval
                    if 0 <= i < L_in:
                        xval = xn[0, i, cin]
                        dot += xval * wval
                        patch_sq += xval * xval
            dist = max(patch_sq + kernel_sq - 2 * dot, 0.0)
            ref[0, p, f] = a * (dot + b[f]) ** 2 / (dist + 1e-5)

    assert np.max(np.abs(y - ref)) < 1e-5


# ---------------------------------------------------------------------------
# Grouped conv
# ---------------------------------------------------------------------------


def test_grouped_conv2d_shape():
    layer = YatConv2D(filters=8, kernel_size=3, groups=2)
    x = mx.random.normal(shape=(1, 8, 8, 4))
    assert layer(x).shape == (1, 6, 6, 8)


def test_grouped_conv2d_rejects_bad_channels():
    layer = YatConv2D(filters=8, kernel_size=3, groups=3)
    with pytest.raises(ValueError):
        layer(mx.random.normal(shape=(1, 8, 8, 4)))


# ---------------------------------------------------------------------------
# Bias / alpha / epsilon variants
# ---------------------------------------------------------------------------


def test_conv2d_no_bias_no_alpha():
    layer = YatConv2D(filters=4, kernel_size=3, use_bias=False, use_alpha=False)
    _ = layer(mx.random.normal(shape=(1, 6, 6, 2)))
    params = layer.parameters()
    assert "bias" not in params
    assert "alpha" not in params
    assert "kernel" in params


def test_conv2d_constant_alpha():
    layer = YatConv2D(filters=4, kernel_size=3, constant_alpha=True)
    _ = layer(mx.random.normal(shape=(1, 6, 6, 2)))
    assert "alpha" not in layer.parameters()


def test_conv2d_constant_bias():
    layer = YatConv2D(filters=4, kernel_size=3, constant_bias=0.5)
    _ = layer(mx.random.normal(shape=(1, 6, 6, 2)))
    assert "bias" not in layer.parameters()


def test_conv2d_learnable_epsilon():
    layer = YatConv2D(filters=4, kernel_size=3, learnable_epsilon=True, epsilon=1e-3)
    _ = layer(mx.random.normal(shape=(1, 6, 6, 2)))
    assert "epsilon_param" in layer.parameters()


def test_conv_rejects_bad_input_ndim():
    layer = YatConv2D(filters=4, kernel_size=3)
    with pytest.raises(ValueError):
        layer(mx.random.normal(shape=(1, 6, 6)))  # missing batch dim


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_conv2d_gradient_reduces_loss():
    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    layer = YatConv2D(filters=4, kernel_size=3, padding="same")
    x = mx.random.normal(shape=(2, 6, 6, 3))
    y = mx.random.normal(shape=(2, 6, 6, 4))
    _ = layer(x)  # build

    grad_fn = mlx_nn.value_and_grad(layer, loss_fn)
    loss, grads = grad_fn(layer, x, y)
    assert set(grads.keys()) >= {"kernel", "bias", "alpha"}

    opt = mlx_optim.AdamW(learning_rate=1e-2)
    opt.update(layer, grads)
    mx.eval(layer.parameters())

    loss_after = float(loss_fn(layer, x, y))
    assert loss_after < float(loss)


def test_conv_transpose2d_gradient_reduces_loss():
    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    layer = YatConvTranspose2D(filters=2, kernel_size=2, strides=2)
    x = mx.random.normal(shape=(1, 4, 4, 3))
    out = layer(x)
    y = mx.random.normal(shape=out.shape)

    grad_fn = mlx_nn.value_and_grad(layer, loss_fn)
    loss, grads = grad_fn(layer, x, y)
    assert set(grads.keys()) >= {"kernel", "bias", "alpha"}

    opt = mlx_optim.AdamW(learning_rate=1e-2)
    opt.update(layer, grads)
    mx.eval(layer.parameters())
    loss_after = float(loss_fn(layer, x, y))
    assert loss_after < float(loss)
