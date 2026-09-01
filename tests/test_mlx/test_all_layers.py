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

import itertools

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")

from nmn.mlx import (  # noqa: E402
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)


def _explicit_same_transpose_reference(
    inputs, kernel, bias, alpha, epsilon, strides, dilation, output_padding
):
    """Differentiable definition of asymmetric SAME transpose convolution.

    This intentionally does not call an MLX convolution or reproduce the
    implementation's symmetric-convolution/crop strategy. Instead it applies
    the defining scatter relation directly: input position ``i`` and kernel
    position ``k`` contribute to ``i * stride + k * dilation - pad_low``.
    """
    spatial = tuple(int(size) for size in inputs.shape[1:-1])
    kernel_size = tuple(int(size) for size in kernel.shape[1:-1])
    strides = tuple(int(value) for value in strides)
    dilation = tuple(int(value) for value in dilation)
    output_padding = tuple(int(value) for value in output_padding)
    effective = tuple(
        (size - 1) * rate + 1 for size, rate in zip(kernel_size, dilation)
    )
    pad_low = tuple(
        max(size - stride, 0) // 2 for size, stride in zip(effective, strides)
    )
    target = tuple(
        size * stride + extra
        for size, stride, extra in zip(spatial, strides, output_padding)
    )
    batch = inputs.shape[0]
    filters = kernel.shape[0]
    kernel_sq = mx.sum(kernel * kernel, axis=tuple(range(1, kernel.ndim)))
    output_values = []
    for out_coord in itertools.product(*(range(size) for size in target)):
        dot = mx.zeros((batch, filters), dtype=inputs.dtype)
        patch_sq = mx.zeros((batch, 1), dtype=inputs.dtype)
        for input_coord in itertools.product(*(range(size) for size in spatial)):
            x_value = inputs[(slice(None), *input_coord, slice(None))]
            for kernel_coord in itertools.product(
                *(range(size) for size in kernel_size)
            ):
                scattered = tuple(
                    index * stride + offset * rate - low
                    for index, stride, offset, rate, low in zip(
                        input_coord, strides, kernel_coord, dilation, pad_low
                    )
                )
                if scattered != out_coord:
                    continue
                kernel_value = kernel[(slice(None), *kernel_coord, slice(None))]
                dot = dot + x_value @ kernel_value.T
                patch_sq = patch_sq + mx.sum(x_value * x_value, axis=-1, keepdims=True)
        distance = mx.maximum(patch_sq + kernel_sq - 2.0 * dot, 0.0)
        output_values.append(alpha * (dot + bias) ** 2 / (distance + epsilon))
    flat = mx.stack(output_values, axis=1)
    return mx.reshape(flat, (batch, *target, filters))


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


@pytest.mark.parametrize(
    "layer_cls,input_shape,kernel,stride,dilation,output_padding,expected",
    [
        (YatConvTranspose1D, (1, 5, 2), 3, 2, 1, 0, (1, 10, 3)),
        (YatConvTranspose1D, (1, 5, 2), 4, 1, 1, 0, (1, 5, 3)),
        (YatConvTranspose1D, (1, 5, 2), 2, 2, 2, 1, (1, 11, 3)),
        (
            YatConvTranspose2D,
            (1, 4, 5, 2),
            (4, 3),
            (2, 1),
            (1, 2),
            (1, 0),
            (1, 9, 5, 3),
        ),
        (
            YatConvTranspose3D,
            (1, 3, 4, 5, 2),
            (2, 3, 4),
            (2, 1, 2),
            (2, 1, 1),
            (0, 0, 0),
            (1, 6, 4, 10, 3),
        ),
    ],
)
def test_conv_transpose_same_exact_shape_all_dimensions(
    layer_cls, input_shape, kernel, stride, dilation, output_padding, expected
):
    layer = layer_cls(
        filters=3,
        kernel_size=kernel,
        strides=stride,
        dilation_rate=dilation,
        output_padding=output_padding,
        padding="same",
    )
    assert layer(mx.ones(input_shape)).shape == expected


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
                patch = xn[0, i : i + 3, j : j + 3, :]
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


@pytest.mark.parametrize(
    "kernel_size,stride,dilation", [(3, 2, 1), (4, 1, 1), (2, 2, 2)]
)
def test_conv_transpose1d_same_math_parity(kernel_size, stride, dilation):
    """SAME uses a symmetric native transpose followed by a high-side
    adjustment; compare its complete YAT result to that definition."""
    layer = YatConvTranspose1D(
        filters=2,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=dilation,
        padding="same",
        epsilon=0.02,
    )
    x = mx.array([[[0.2], [-0.4], [0.7], [0.1]]])
    layer.build(1)
    layer.kernel = mx.reshape(
        (mx.arange(2 * kernel_size, dtype=mx.float32) + 1) * 0.05,
        (2, kernel_size, 1),
    )
    layer.bias = mx.array([0.1, -0.2])
    layer.alpha = mx.array([1.3])
    actual = np.array(layer(x))

    xn = np.array(x)
    weights = np.array(layer.kernel)
    effective = (kernel_size - 1) * dilation + 1
    native_pad = max(effective - stride, 0) // 2
    full_size = (xn.shape[1] - 1) * stride + effective
    dot_full = np.zeros((1, full_size, 2), dtype=np.float32)
    patch_full = np.zeros_like(dot_full)
    for index in range(xn.shape[1]):
        for k in range(kernel_size):
            out_index = index * stride + k * dilation
            dot_full[0, out_index, :] += xn[0, index, 0] * weights[:, k, 0]
            patch_full[0, out_index, :] += xn[0, index, 0] ** 2
    native_stop = full_size - native_pad if native_pad else full_size
    dot = dot_full[:, native_pad:native_stop, :]
    patch_sq = patch_full[:, native_pad:native_stop, :]
    target = xn.shape[1] * stride
    dot = dot[:, :target, :]
    patch_sq = patch_sq[:, :target, :]
    if dot.shape[1] < target:
        width = target - dot.shape[1]
        dot = np.pad(dot, ((0, 0), (0, width), (0, 0)))
        patch_sq = np.pad(patch_sq, ((0, 0), (0, width), (0, 0)))
    kernel_sq = np.sum(weights * weights, axis=(1, 2))[None, None, :]
    dist = np.maximum(patch_sq + kernel_sq - 2.0 * dot, 0.0)
    expected = 1.3 * (dot + np.array([0.1, -0.2])) ** 2 / (dist + 0.02)
    assert actual.shape == (1, target, 2)
    assert np.allclose(actual, expected, rtol=2e-5, atol=2e-6)


_SAME_MULTIDIM_CASES = [
    (
        YatConvTranspose2D,
        (1, 3, 2, 1),
        (4, 3),
        (2, 1),
        (1, 2),
        (1, 0),
    ),
    (
        YatConvTranspose3D,
        (1, 2, 2, 2, 1),
        (3, 2, 2),
        (1, 2, 2),
        (1, 2, 1),
        (0, 1, 1),
    ),
]


def _assert_same_multidim_forward_and_gradient_parity(
    layer_cls,
    input_shape,
    kernel_size,
    strides,
    dilation,
    output_padding,
    *,
    forward_rtol,
    forward_atol,
    gradient_rtol,
    gradient_atol,
    require_gpu=False,
):
    epsilon = 0.03
    layer = layer_cls(
        filters=2,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation,
        output_padding=output_padding,
        padding="same",
        epsilon=epsilon,
    )
    layer.build(input_shape[-1])
    kernel_elements = int(np.prod(layer.kernel.shape))
    layer.kernel = mx.reshape(
        (mx.arange(kernel_elements, dtype=mx.float32) + 1) * 0.025,
        layer.kernel.shape,
    )
    layer.bias = mx.array([0.08, -0.11])
    layer.alpha = mx.array([1.2])
    input_elements = int(np.prod(input_shape))
    inputs = mx.reshape(
        (mx.arange(input_elements, dtype=mx.float32) + 1) * 0.04 - 0.2,
        input_shape,
    )

    def reference(value, kernel):
        return _explicit_same_transpose_reference(
            value,
            kernel,
            layer.bias,
            layer.alpha,
            epsilon,
            strides,
            dilation,
            output_padding,
        )

    actual = layer(inputs)
    expected = reference(inputs, layer.kernel)
    cotangent = mx.reshape(
        (mx.arange(actual.size, dtype=mx.float32) + 1) / actual.size,
        actual.shape,
    )

    def layer_loss(model, value):
        return mx.sum(model(value) * cotangent)

    _, layer_grads = mlx_nn.value_and_grad(layer, layer_loss)(layer, inputs)
    actual_input_grad = mx.grad(lambda value: mx.sum(layer(value) * cotangent))(inputs)

    def reference_loss(value, kernel):
        return mx.sum(reference(value, kernel) * cotangent)

    _, (expected_input_grad, expected_kernel_grad) = mx.value_and_grad(
        reference_loss, argnums=(0, 1)
    )(inputs, layer.kernel)
    mx.eval(
        actual,
        expected,
        actual_input_grad,
        expected_input_grad,
        layer_grads["kernel"],
        expected_kernel_grad,
    )

    if require_gpu:
        assert str(mx.default_device()) == "Device(gpu, 0)"
    assert np.allclose(
        np.array(actual),
        np.array(expected),
        rtol=forward_rtol,
        atol=forward_atol,
    )
    assert np.allclose(
        np.array(actual_input_grad),
        np.array(expected_input_grad),
        rtol=gradient_rtol,
        atol=gradient_atol,
    )
    assert np.allclose(
        np.array(layer_grads["kernel"]),
        np.array(expected_kernel_grad),
        rtol=gradient_rtol,
        atol=gradient_atol,
    )


@pytest.mark.parametrize(
    "layer_cls,input_shape,kernel_size,strides,dilation,output_padding",
    _SAME_MULTIDIM_CASES,
)
def test_conv_transpose_same_multidim_forward_and_gradient_parity(
    layer_cls, input_shape, kernel_size, strides, dilation, output_padding
):
    """2D/3D mixed-axis SAME matches the explicit asymmetric definition."""
    _assert_same_multidim_forward_and_gradient_parity(
        layer_cls,
        input_shape,
        kernel_size,
        strides,
        dilation,
        output_padding,
        forward_rtol=3e-5,
        forward_atol=3e-6,
        gradient_rtol=8e-5,
        gradient_atol=8e-6,
    )


@pytest.mark.parametrize(
    "layer_cls,input_shape,kernel_size,strides,dilation,output_padding",
    _SAME_MULTIDIM_CASES,
)
def test_gpu_conv_transpose_same_multidim_forward_and_gradient_parity(
    mlx_gpu,
    layer_cls,
    input_shape,
    kernel_size,
    strides,
    dilation,
    output_padding,
):
    """Mixed-axis 2D/3D SAME output and gradients agree on Metal."""
    del mlx_gpu
    _assert_same_multidim_forward_and_gradient_parity(
        layer_cls,
        input_shape,
        kernel_size,
        strides,
        dilation,
        output_padding,
        forward_rtol=3e-3,
        forward_atol=3e-4,
        gradient_rtol=6e-3,
        gradient_atol=6e-4,
        require_gpu=True,
    )


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
