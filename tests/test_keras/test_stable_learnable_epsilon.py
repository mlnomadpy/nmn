"""Stable learnable-epsilon initialization across Keras YAT families."""

from contextlib import nullcontext

import keras
from keras import ops
import numpy as np
import pytest

from nmn.keras import (
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatNMN,
)


BACKEND = keras.backend.backend()
EPSILONS = (1e-20, 1e-5, 1000.0)
FAMILIES = (
    (YatNMN, None, (1, 2)),
    (YatConv1D, (1,), (1, 1, 1)),
    (YatConv2D, (1, 1), (1, 1, 1, 1)),
    (YatConv3D, (1, 1, 1), (1, 1, 1, 1, 1)),
    (YatConvTranspose1D, (1,), (1, 1, 1)),
    (YatConvTranspose2D, (1, 1), (1, 1, 1, 1)),
    (YatConvTranspose3D, (1, 1, 1), (1, 1, 1, 1, 1)),
)


def _make(layer_cls, kernel_size, epsilon, dtype=None):
    kwargs = dict(
        epsilon=epsilon,
        learnable_epsilon=True,
        use_bias=False,
        use_alpha=False,
        kernel_initializer="zeros",
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    if kernel_size is None:
        return layer_cls(1, **kwargs)
    return layer_cls(1, kernel_size, **kwargs)


def _value_and_gradients(layer, inputs):
    variables = list(layer.trainable_variables)
    kernel_index = next(i for i, variable in enumerate(variables) if variable is layer.kernel)
    epsilon_index = next(
        i for i, variable in enumerate(variables) if variable is layer.epsilon_param
    )

    if BACKEND == "jax":
        import jax

        values = [variable.value for variable in variables]

        def evaluate(input_value, kernel_value, epsilon_value):
            current = list(values)
            current[kernel_index] = kernel_value
            current[epsilon_index] = epsilon_value
            output, _ = layer.stateless_call(
                current, layer.non_trainable_variables, input_value
            )
            return ops.sum(output), output

        (_, output), gradients = jax.jit(
            jax.value_and_grad(evaluate, argnums=(0, 1, 2), has_aux=True)
        )(inputs, values[kernel_index], values[epsilon_index])
        return output, gradients

    if BACKEND == "torch":
        torch = pytest.importorskip("torch")
        input_value = inputs.detach().clone().requires_grad_(True)
        output = layer(input_value)
        gradients = torch.autograd.grad(
            output.sum(),
            (input_value, layer.kernel.value, layer.epsilon_param.value),
        )
        return output, gradients

    if BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        input_value = tf.Variable(inputs)

        @tf.function
        def evaluate():
            with tf.GradientTape() as tape:
                output = layer(input_value)
                loss = tf.reduce_sum(output)
            gradients = tape.gradient(
                loss, (input_value, layer.kernel.value, layer.epsilon_param.value)
            )
            return output, gradients

        return evaluate()

    pytest.skip("requires the JAX, Torch, or TensorFlow Keras backend")


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_learnable_epsilon_forward_and_gradients(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon)
    inputs = ops.full(input_shape, 0.2, dtype="float32")
    layer(inputs)
    layer.kernel.assign(ops.full(layer.kernel.shape, 0.3, dtype="float32"))
    output, gradients = _value_and_gradients(layer, inputs)

    effective = ops.softplus(layer.epsilon_param)
    np.testing.assert_allclose(
        ops.convert_to_numpy(effective), epsilon, rtol=2e-6, atol=0.0
    )
    assert np.isfinite(ops.convert_to_numpy(output)).all()
    for gradient in gradients:
        assert np.isfinite(ops.convert_to_numpy(gradient)).all()
    assert np.abs(ops.convert_to_numpy(gradients[-1])).max() > 0.0


@pytest.mark.parametrize("layer_cls,kernel_size,_", FAMILIES)
@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_epsilon_must_be_finite_and_strictly_positive(
    layer_cls, kernel_size, _, epsilon
):
    with pytest.raises(ValueError, match="positive and finite"):
        _make(layer_cls, kernel_size, epsilon)


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_dense_config_and_weights_roundtrip(epsilon):
    layer = _make(YatNMN, None, epsilon)
    inputs = ops.full((1, 2), 0.2)
    layer(inputs)
    config = layer.get_config()
    restored = YatNMN.from_config(config)
    restored(inputs)
    restored.set_weights(layer.get_weights())
    assert restored.epsilon == epsilon
    np.testing.assert_array_equal(
        ops.convert_to_numpy(restored.epsilon_param),
        ops.convert_to_numpy(layer.epsilon_param),
    )


def test_default_epsilon_remains_backward_compatible():
    layer = YatNMN(1, learnable_epsilon=True)
    layer(ops.ones((1, 2)))
    assert layer.epsilon == 1e-5
    np.testing.assert_allclose(
        ops.convert_to_numpy(ops.softplus(layer.epsilon_param)),
        1e-5,
        rtol=2e-6,
    )


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", [1e-8, 1e-20, 1e5])
def test_float16_uses_fp32_epsilon_storage(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, "float16")
    inputs = ops.full(input_shape, 0.2, dtype="float16")
    layer(inputs)
    layer.kernel.assign(ops.full(layer.kernel.shape, 0.3, dtype="float16"))
    output, gradients = _value_and_gradients(layer, inputs)
    assert keras.backend.standardize_dtype(layer.epsilon_param.dtype) == "float32"
    np.testing.assert_allclose(
        ops.convert_to_numpy(ops.softplus(layer.epsilon_param)),
        epsilon,
        rtol=2e-6,
    )
    assert keras.backend.standardize_dtype(output.dtype) == "float16"
    assert np.isfinite(ops.convert_to_numpy(output)).all()
    epsilon_grad = ops.convert_to_numpy(gradients[-1])
    assert np.isfinite(epsilon_grad).all() and np.abs(epsilon_grad).max() > 0


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
@pytest.mark.parametrize("epsilon", [5e-324, 1e-46, 1e39])
def test_float32_rejects_unrepresentable_epsilon(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, "float32")
    with pytest.raises(ValueError, match="not representable"):
        layer(ops.ones(input_shape, dtype="float32"))


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
@pytest.mark.parametrize("epsilon", [2.0 ** -1022, 1e150])
def test_float64_extreme_epsilon_is_effective_and_differentiable(
    layer_cls, kernel_size, input_shape, epsilon
):
    if BACKEND == "jax":
        import jax

        dtype_context = jax.enable_x64()
    else:
        dtype_context = nullcontext()

    with dtype_context:
        layer = _make(layer_cls, kernel_size, epsilon, "float64")
        inputs = ops.full(input_shape, 0.2, dtype="float64")
        layer(inputs)
        layer.kernel.assign(ops.full(layer.kernel.shape, 0.3, dtype="float64"))
        output, gradients = _value_and_gradients(layer, inputs)
        np.testing.assert_allclose(
            ops.convert_to_numpy(ops.softplus(layer.epsilon_param)),
            epsilon,
            rtol=5e-14,
        )
        epsilon_grad = ops.convert_to_numpy(gradients[-1])
        assert np.isfinite(ops.convert_to_numpy(output)).all()
        assert np.isfinite(epsilon_grad).all() and np.abs(epsilon_grad).max() > 0


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
def test_float64_rejects_softplus_underflow(
    layer_cls, kernel_size, input_shape
):
    layer = _make(layer_cls, kernel_size, 5e-324, "float64")
    with pytest.raises(ValueError, match="not representable"):
        layer(ops.ones(input_shape, dtype="float64"))
