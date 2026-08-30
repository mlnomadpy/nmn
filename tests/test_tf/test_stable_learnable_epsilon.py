"""Stable learnable-epsilon initialization across TensorFlow YAT families."""

import numpy as np
import pytest
import tensorflow as tf

from nmn.tf import (
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatNMN,
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


def _make(layer_cls, kernel_size, epsilon, dtype=None):
    kwargs = dict(
        epsilon=epsilon,
        learnable_epsilon=True,
        use_bias=False,
        use_alpha=False,
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    if kernel_size is None:
        return layer_cls(features=1, **kwargs)
    return layer_cls(filters=1, kernel_size=kernel_size, **kwargs)


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_learnable_epsilon_tf_function_forward_and_gradients(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon)
    inputs = tf.Variable(tf.fill(input_shape, 0.2))
    layer(inputs)
    layer.kernel.assign(tf.fill(layer.kernel.shape, 0.3))

    @tf.function
    def evaluate():
        with tf.GradientTape() as tape:
            output = layer(inputs)
            loss = tf.reduce_sum(output)
        gradients = tape.gradient(
            loss, (inputs, layer.kernel, layer.epsilon_param)
        )
        return output, gradients

    output, gradients = evaluate()
    effective = tf.nn.softplus(layer.epsilon_param)
    np.testing.assert_allclose(effective.numpy(), epsilon, rtol=2e-6, atol=0.0)
    assert np.isfinite(output.numpy()).all()
    for gradient in gradients:
        assert np.isfinite(gradient.numpy()).all()
    assert np.abs(gradients[-1].numpy()).max() > 0.0


@pytest.mark.parametrize("layer_cls,kernel_size,_", FAMILIES)
@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_epsilon_must_be_finite_and_strictly_positive(
    layer_cls, kernel_size, _, epsilon
):
    with pytest.raises(ValueError, match="positive and finite"):
        _make(layer_cls, kernel_size, epsilon)


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_dense_checkpoint_roundtrip(tmp_path, epsilon):
    layer = _make(YatNMN, None, epsilon)
    inputs = tf.fill((1, 2), 0.2)
    layer(inputs)
    checkpoint = tf.train.Checkpoint(layer=layer)
    path = checkpoint.save(str(tmp_path / "epsilon"))

    restored = _make(YatNMN, None, epsilon)
    restored(inputs)
    tf.train.Checkpoint(layer=restored).restore(path).assert_consumed()
    np.testing.assert_array_equal(
        restored.epsilon_param.numpy(), layer.epsilon_param.numpy()
    )


def test_default_epsilon_remains_backward_compatible():
    layer = YatNMN(features=1, learnable_epsilon=True)
    layer(tf.ones((1, 2)))
    assert layer.epsilon == 1e-5
    np.testing.assert_allclose(
        tf.nn.softplus(layer.epsilon_param).numpy(), 1e-5, rtol=2e-6
    )


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", [1e-8, 1e-20, 1e5])
def test_float16_uses_fp32_epsilon_storage(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, tf.float16)
    inputs = tf.Variable(tf.fill(input_shape, tf.constant(0.2, tf.float16)))
    layer(inputs)
    layer.kernel.assign(tf.fill(layer.kernel.shape, tf.constant(0.3, tf.float16)))

    @tf.function
    def evaluate():
        with tf.GradientTape() as tape:
            output = layer(inputs)
            loss = tf.reduce_sum(tf.cast(output, tf.float32))
        return output, tape.gradient(loss, layer.epsilon_param)

    output, epsilon_grad = evaluate()
    assert layer.epsilon_param.dtype == tf.float32
    np.testing.assert_allclose(
        tf.nn.softplus(layer.epsilon_param).numpy(), epsilon, rtol=2e-6
    )
    assert output.dtype == tf.float16 and np.isfinite(output.numpy()).all()
    assert np.isfinite(epsilon_grad.numpy()).all()
    assert np.abs(epsilon_grad.numpy()).max() > 0


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
@pytest.mark.parametrize("epsilon", [5e-324, 1e-46, 1e39])
def test_float32_rejects_unrepresentable_epsilon(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, tf.float32)
    with pytest.raises(ValueError, match="not representable"):
        layer(tf.ones(input_shape, tf.float32))


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
@pytest.mark.parametrize("epsilon", [2.0 ** -1022, 1e150])
def test_float64_extreme_epsilon_is_effective_and_differentiable(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, tf.float64)
    inputs = tf.Variable(tf.fill(input_shape, tf.constant(0.2, tf.float64)))
    layer(inputs)
    layer.kernel.assign(tf.fill(layer.kernel.shape, tf.constant(0.3, tf.float64)))
    with tf.GradientTape() as tape:
        output = layer(inputs)
        loss = tf.reduce_sum(output)
    epsilon_grad = tape.gradient(loss, layer.epsilon_param)
    np.testing.assert_allclose(
        tf.nn.softplus(layer.epsilon_param).numpy(), epsilon, rtol=5e-14
    )
    assert np.isfinite(output.numpy()).all()
    assert np.isfinite(epsilon_grad.numpy()).all()
    assert np.abs(epsilon_grad.numpy()).max() > 0


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
def test_float64_rejects_softplus_underflow(
    layer_cls, kernel_size, input_shape
):
    layer = _make(layer_cls, kernel_size, 5e-324, tf.float64)
    with pytest.raises(ValueError, match="not representable"):
        layer(tf.ones(input_shape, tf.float64))
