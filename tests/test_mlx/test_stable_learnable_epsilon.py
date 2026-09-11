"""Stable learnable-epsilon handling across MLX YAT layer families."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")

from nmn.mlx import (  # noqa: E402
    GoatYatAttention,
    MultiHeadYatAttention,
    RotaryYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatEmbed,
    YatNMN,
    goat_yat_attention_weights,
)

EPSILONS = (1e-20, 1e-5, 1000.0)
FAMILIES = (
    (YatNMN, None, (1, 2)),
    (YatConv1D, 1, (1, 1, 1)),
    (YatConv2D, (1, 1), (1, 1, 1, 1)),
    (YatConv3D, (1, 1, 1), (1, 1, 1, 1, 1)),
    (YatConvTranspose1D, 1, (1, 1, 1)),
    (YatConvTranspose2D, (1, 1), (1, 1, 1, 1)),
    (YatConvTranspose3D, (1, 1, 1), (1, 1, 1, 1, 1)),
)


def _make(layer_cls, kernel_size, epsilon, dtype=mx.float32):
    kwargs = dict(
        epsilon=epsilon,
        learnable_epsilon=True,
        use_bias=False,
        use_alpha=False,
        dtype=dtype,
    )
    if kernel_size is None:
        return layer_cls(features=1, **kwargs)
    return layer_cls(filters=1, kernel_size=kernel_size, **kwargs)


def _evaluate(layer, inputs, compiled):
    layer(inputs)
    layer.kernel = mx.full(layer.kernel.shape, 0.3, dtype=layer.dtype)

    def loss(model, values):
        output = model(values)
        return mx.sum(output.astype(mx.float32)), output

    grad_fn = mlx_nn.value_and_grad(layer, loss)
    input_grad_fn = mx.grad(lambda values: loss(layer, values)[0])
    if compiled:

        def evaluate(values):
            value, parameter_gradients = grad_fn(layer, values)
            effective_epsilon = mlx_nn.softplus(layer.epsilon_param)
            return (
                value,
                parameter_gradients,
                input_grad_fn(values),
                effective_epsilon,
            )

        (_, output), gradients, input_gradient, effective_epsilon = mx.compile(
            evaluate, inputs=layer.state
        )(inputs)
    else:
        (_, output), gradients = grad_fn(layer, inputs)
        input_gradient = input_grad_fn(inputs)
        effective_epsilon = mlx_nn.softplus(layer.epsilon_param)
    mx.eval(output, gradients, input_gradient, effective_epsilon)
    return output, gradients, input_gradient, effective_epsilon


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("compiled", [False, True])
def test_learnable_epsilon_eager_compile_and_gradients(
    layer_cls, kernel_size, input_shape, epsilon, compiled
):
    layer = _make(layer_cls, kernel_size, epsilon)
    output, gradients, input_gradient, effective = _evaluate(
        layer, mx.full(input_shape, 0.2, dtype=mx.float32), compiled
    )

    np.testing.assert_allclose(np.array(effective), epsilon, rtol=2e-6, atol=0.0)
    assert np.isfinite(np.array(output)).all()
    assert np.isfinite(np.array(input_gradient)).all()
    assert np.isfinite(np.array(gradients["kernel"])).all()
    epsilon_gradient = np.array(gradients["epsilon_param"])
    assert np.isfinite(epsilon_gradient).all()
    assert np.abs(epsilon_gradient).max() > 0.0


def _validation_factories(epsilon):
    return (
        lambda: YatNMN(features=1, epsilon=epsilon),
        lambda: YatConv1D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConv2D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConv3D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConvTranspose1D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConvTranspose2D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConvTranspose3D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatEmbed(num_embeddings=2, features=2, epsilon=epsilon),
        lambda: MultiHeadYatAttention(embed_dim=4, num_heads=2, epsilon=epsilon),
        lambda: RotaryYatAttention(embed_dim=4, num_heads=2, epsilon=epsilon),
        lambda: GoatYatAttention(embed_dim=4, num_heads=2, epsilon=epsilon),
    )


@pytest.mark.parametrize(
    "epsilon", [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
)
def test_every_mlx_layer_rejects_non_finite_or_non_positive_epsilon(epsilon):
    for factory in _validation_factories(epsilon):
        with pytest.raises(ValueError, match="positive and finite"):
            factory()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
def test_epsilon_parameter_storage_dtype_and_representability(
    layer_cls, kernel_size, input_shape, dtype, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, dtype)
    inputs = mx.full(input_shape, 0.2, dtype=dtype)
    layer(inputs)
    expected_dtype = mx.float32 if dtype in (mx.float16, mx.bfloat16) else dtype
    assert layer.epsilon_param.dtype == expected_dtype
    effective = mlx_nn.softplus(layer.epsilon_param)
    mx.eval(effective)
    np.testing.assert_allclose(np.array(effective), epsilon, rtol=2e-6, atol=0.0)
    if dtype in (mx.float16, mx.bfloat16):
        output, gradients, input_gradient, _ = _evaluate(layer, inputs, False)
        assert output.dtype == dtype
        assert np.isfinite(np.array(output.astype(mx.float32))).all()
        assert np.isfinite(np.array(input_gradient.astype(mx.float32))).all()
        epsilon_gradient = np.array(gradients["epsilon_param"].astype(mx.float32))
        assert np.isfinite(epsilon_gradient).all()
        assert np.abs(epsilon_gradient).max() > 0.0


@pytest.mark.parametrize("epsilon", [5e-324, 1e-46, 1e39])
def test_float32_rejects_unrepresentable_epsilon(epsilon):
    layer = _make(YatNMN, None, epsilon, mx.float32)
    with pytest.raises(ValueError, match="not representable"):
        layer(mx.ones((1, 2), dtype=mx.float32))


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_dense_weight_serialization_roundtrip(tmp_path, epsilon):
    layer = _make(YatNMN, None, epsilon, mx.float16)
    inputs = mx.ones((1, 2), dtype=mx.float16)
    layer(inputs)
    layer.epsilon_param = layer.epsilon_param + mx.array([0.5], dtype=mx.float32)
    mx.eval(layer.epsilon_param)
    path = tmp_path / "epsilon.npz"
    layer.save_weights(str(path))

    restored = _make(YatNMN, None, epsilon, mx.float16)
    restored(inputs)
    mx.eval(restored.epsilon_param)
    assert not np.array_equal(
        np.array(restored.epsilon_param), np.array(layer.epsilon_param)
    )
    restored.load_weights(str(path))
    source_effective = mlx_nn.softplus(layer.epsilon_param)
    restored_effective = mlx_nn.softplus(restored.epsilon_param)
    mx.eval(restored.parameters(), source_effective, restored_effective)
    assert restored.epsilon_param.dtype == mx.float32
    np.testing.assert_array_equal(
        np.array(restored.epsilon_param), np.array(layer.epsilon_param)
    )
    np.testing.assert_array_equal(
        np.array(restored_effective), np.array(source_effective)
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("epsilon", [1e-20, 1e5])
def test_dtype_extremes_eager_compile_and_metal(dtype, epsilon, mlx_gpu):
    del mlx_gpu
    layer = _make(YatNMN, None, epsilon, dtype)
    inputs = mx.full((1, 2), 0.2, dtype=dtype)
    eager_output, eager_gradients, eager_input_gradient, _ = _evaluate(
        layer, inputs, False
    )
    compiled_output, compiled_gradients, compiled_input_gradient, _ = _evaluate(
        layer, inputs, True
    )

    assert eager_output.dtype == compiled_output.dtype == dtype
    for value in (
        eager_output,
        compiled_output,
        eager_input_gradient,
        compiled_input_gradient,
        eager_gradients["epsilon_param"],
        compiled_gradients["epsilon_param"],
    ):
        assert np.isfinite(np.array(value.astype(mx.float32))).all()
    np.testing.assert_allclose(
        np.array(compiled_output.astype(mx.float32)),
        np.array(eager_output.astype(mx.float32)),
        rtol=2e-2,
        atol=2e-3,
    )


def _direct_goat_weights(q, k, b, epsilon, mask, floor):
    qh = mx.transpose(q, (0, 2, 1, 3))
    kh = mx.transpose(k, (0, 2, 1, 3))
    dot = qh @ mx.swapaxes(kh, -1, -2)
    q_norm = mx.sum(qh * qh, axis=-1, keepdims=True)
    k_norm = mx.sum(kh * kh, axis=-1, keepdims=True)
    distance = mx.maximum(q_norm + mx.swapaxes(k_norm, -1, -2) - 2.0 * dot, 0.0)
    scores = (dot + b.reshape(1, -1, 1, 1)) ** 2 / (
        distance + epsilon.reshape(1, -1, 1, 1)
    )
    if mask is not None:
        scores = scores * mask.astype(scores.dtype)
    return scores / (mx.sum(scores, axis=-1, keepdims=True) + floor)


@pytest.mark.parametrize("mask_kind", ["unmasked", "partial"])
def test_goat_stable_normalization_matches_direct_forward_and_vjp(mask_kind):
    q = mx.array([[[[0.1, 0.2]], [[0.5, -0.2]]]], dtype=mx.float32)
    k = mx.array([[[[0.2, -0.1]], [[-0.4, 0.2]], [[0.3, 0.6]]]], dtype=mx.float32)
    b = mx.array([0.7], dtype=mx.float32)
    epsilon = mx.array([0.3], dtype=mx.float32)
    mask = None
    if mask_kind == "partial":
        mask = mx.array([[[[True, False, True], [False, True, True]]]])
    floor = 3e-3
    cotangent = mx.array([[[[0.2, -0.4, 0.7], [-0.3, 0.5, 0.1]]]])

    def actual_loss(q, k, b, epsilon):
        weights = goat_yat_attention_weights(
            q, k, b, epsilon, mask=mask, self_mask=False, floor=floor
        )
        return mx.sum(weights * cotangent)

    def reference_loss(q, k, b, epsilon):
        weights = _direct_goat_weights(q, k, b, epsilon, mask, floor)
        return mx.sum(weights * cotangent)

    actual = goat_yat_attention_weights(
        q, k, b, epsilon, mask=mask, self_mask=False, floor=floor
    )
    expected = _direct_goat_weights(q, k, b, epsilon, mask, floor)
    actual_value, actual_gradients = mx.value_and_grad(
        actual_loss, argnums=(0, 1, 2, 3)
    )(q, k, b, epsilon)
    expected_value, expected_gradients = mx.value_and_grad(
        reference_loss, argnums=(0, 1, 2, 3)
    )(q, k, b, epsilon)
    mx.eval(
        actual,
        expected,
        actual_value,
        expected_value,
        actual_gradients,
        expected_gradients,
    )

    np.testing.assert_allclose(
        np.array(actual), np.array(expected), rtol=2e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        np.array(actual_value), np.array(expected_value), rtol=2e-6, atol=1e-7
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        np.testing.assert_allclose(
            np.array(actual_gradient),
            np.array(expected_gradient),
            rtol=2e-5,
            atol=2e-6,
        )


def test_goat_fully_masked_rows_have_zero_finite_vjp():
    q = mx.array([[[[0.1, 0.2]], [[0.5, -0.2]]]], dtype=mx.float32)
    k = mx.array([[[[0.2, -0.1]], [[-0.4, 0.2]], [[0.3, 0.6]]]], dtype=mx.float32)
    b = mx.array([0.7], dtype=mx.float32)
    epsilon = mx.array([1e-20], dtype=mx.float32)
    mask = mx.zeros((1, 1, 2, 3), dtype=mx.bool_)

    def loss(q, k, b, epsilon):
        weights = goat_yat_attention_weights(
            q, k, b, epsilon, mask=mask, self_mask=False
        )
        return mx.sum(weights)

    weights = goat_yat_attention_weights(q, k, b, epsilon, mask=mask, self_mask=False)
    _, gradients = mx.value_and_grad(loss, argnums=(0, 1, 2, 3))(q, k, b, epsilon)
    mx.eval(weights, gradients)
    np.testing.assert_array_equal(np.array(weights), np.zeros((1, 1, 2, 3)))
    for gradient in gradients:
        assert np.isfinite(np.array(gradient)).all()
        np.testing.assert_array_equal(np.array(gradient), np.zeros(gradient.shape))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_goat_raw_softplus_compiled_low_precision_vjp_on_metal(dtype, mlx_gpu):
    del mlx_gpu
    values = mx.array([[[[0.1, 0.2]], [[0.5, -0.2]], [[-0.4, 0.2]]]], dtype=dtype)
    b = mlx_nn.softplus(mx.zeros((1,), dtype=dtype))
    epsilon_raw = mx.array([-46.0517], dtype=mx.float32)
    cotangent = mx.array([[[[0.2, -0.4, 0.7]]]], dtype=dtype)

    def loss(raw):
        weights = goat_yat_attention_weights(
            values,
            values,
            b,
            mlx_nn.softplus(raw),
            self_mask=False,
        )
        return mx.sum((weights.astype(dtype) * cotangent).astype(mx.float32))

    eager_gradient = mx.grad(loss)(epsilon_raw)
    compiled_gradient = mx.compile(mx.grad(loss))(epsilon_raw)
    mx.eval(eager_gradient, compiled_gradient)
    for gradient in (eager_gradient, compiled_gradient):
        assert np.isfinite(np.array(gradient)).all()
        assert np.abs(np.array(gradient)).max() > 0.0
    np.testing.assert_allclose(
        np.array(compiled_gradient),
        np.array(eager_gradient),
        rtol=2e-2,
        atol=2e-6,
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("compiled", [False, True])
def test_goat_uses_stable_epsilon_parameter(dtype, epsilon, compiled):
    layer = GoatYatAttention(
        embed_dim=4,
        num_heads=2,
        epsilon=epsilon,
        dtype=dtype,
    )
    expected_dtype = mx.float32 if dtype in (mx.float16, mx.bfloat16) else dtype
    assert layer.eps_raw.dtype == expected_dtype
    inputs = mx.array(
        [[[0.1, 0.2, -0.3, 0.4], [0.5, -0.2, 0.1, 0.3], [-0.4, 0.2, 0.6, 0.1]]],
        dtype=dtype,
    )

    def loss(model, values):
        output = model(values, self_mask=False)
        return mx.sum(output.astype(mx.float32)), output

    grad_fn = mlx_nn.value_and_grad(layer, loss)
    if compiled:

        def evaluate(values):
            value, gradients = grad_fn(layer, values)
            return value, gradients, mlx_nn.softplus(layer.eps_raw)

        (_, output), gradients, effective = mx.compile(evaluate, inputs=layer.state)(
            inputs
        )
    else:
        (_, output), gradients = grad_fn(layer, inputs)
        effective = mlx_nn.softplus(layer.eps_raw)
    mx.eval(effective, output, gradients)
    np.testing.assert_allclose(np.array(effective), epsilon, rtol=2e-6, atol=0.0)
    assert output.dtype == dtype
    assert np.isfinite(np.array(output.astype(mx.float32))).all()
    epsilon_gradient = np.array(gradients["eps_raw"].astype(mx.float32))
    assert np.isfinite(epsilon_gradient).all()
    if epsilon == 1e-5:
        assert np.abs(epsilon_gradient).max() > 0.0


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("epsilon", [1e-20, 1e5])
def test_goat_low_precision_compile_and_gradient_on_metal(dtype, epsilon, mlx_gpu):
    del mlx_gpu
    layer = GoatYatAttention(
        embed_dim=4,
        num_heads=2,
        epsilon=epsilon,
        dtype=dtype,
    )
    inputs = mx.array(
        [[[0.1, 0.2, -0.3, 0.4], [0.5, -0.2, 0.1, 0.3], [-0.4, 0.2, 0.6, 0.1]]],
        dtype=dtype,
    )
    grad_fn = mlx_nn.value_and_grad(
        layer,
        lambda model, values: mx.sum(model(values, self_mask=False).astype(mx.float32)),
    )
    _, gradients = mx.compile(
        lambda values: grad_fn(layer, values), inputs=layer.state
    )(inputs)
    mx.eval(gradients)
    assert layer.eps_raw.dtype == mx.float32
    assert np.isfinite(np.array(gradients["eps_raw"])).all()
