"""Stable learnable-epsilon initialization across Linen YAT families."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from nmn.linen import (
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
    (YatConv1D, (1,), (1, 1, 1)),
    (YatConv2D, (1, 1), (1, 1, 1, 1)),
    (YatConv3D, (1, 1, 1), (1, 1, 1, 1, 1)),
    (YatConvTranspose1D, (1,), (1, 1, 1)),
    (YatConvTranspose2D, (1, 1), (1, 1, 1, 1)),
    (YatConvTranspose3D, (1, 1, 1), (1, 1, 1, 1, 1)),
)


def _make(layer_cls, kernel_size, epsilon, dtype=None):
    kwargs = dict(
        features=1,
        epsilon=epsilon,
        learnable_epsilon=True,
        use_bias=False,
        use_alpha=False,
    )
    if dtype is not None:
        kwargs.update(
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=lambda key, shape, init_dtype: jnp.zeros(shape, init_dtype),
        )
    if kernel_size is not None:
        kwargs["kernel_size"] = kernel_size
    return layer_cls(**kwargs)


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_learnable_epsilon_jit_forward_and_gradients(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon)
    inputs = jnp.full(input_shape, 0.2, dtype=jnp.float32)
    variables = layer.init(jax.random.key(0), inputs)
    params = dict(variables["params"])
    params["kernel"] = jnp.full_like(params["kernel"], 0.3)

    def loss(param_values, input_values):
        output = layer.apply({"params": param_values}, input_values)
        return output.sum(), output

    (_, output), (param_grads, input_grads) = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    )(params, inputs)

    effective = jax.nn.softplus(params["epsilon_param"])
    np.testing.assert_allclose(np.asarray(effective), epsilon, rtol=2e-6, atol=0.0)
    assert jnp.isfinite(output).all()
    assert jnp.isfinite(input_grads).all()
    assert jnp.isfinite(param_grads["kernel"]).all()
    assert jnp.isfinite(param_grads["epsilon_param"]).all()
    assert jnp.abs(param_grads["epsilon_param"]).max() > 0.0


@pytest.mark.parametrize("layer_cls,kernel_size,_", FAMILIES)
@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_epsilon_must_be_finite_and_strictly_positive(
    layer_cls, kernel_size, _, epsilon
):
    with pytest.raises(ValueError, match="positive and finite"):
        _make(layer_cls, kernel_size, epsilon)


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_dense_params_serialize_roundtrip(epsilon):
    layer = _make(YatNMN, None, epsilon)
    inputs = jnp.full((1, 2), 0.2)
    variables = layer.init(jax.random.key(1), inputs)
    encoded = serialization.to_bytes(variables)
    restored = serialization.from_bytes(variables, encoded)
    np.testing.assert_array_equal(
        np.asarray(restored["params"]["epsilon_param"]),
        np.asarray(variables["params"]["epsilon_param"]),
    )


def test_default_epsilon_remains_backward_compatible():
    layer = YatNMN(features=1, learnable_epsilon=True)
    variables = layer.init(jax.random.key(2), jnp.ones((1, 2)))
    assert layer.epsilon == 1e-5
    np.testing.assert_allclose(
        np.asarray(jax.nn.softplus(variables["params"]["epsilon_param"])),
        1e-5,
        rtol=2e-6,
    )


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", [1e-8, 1e-20, 1e5])
def test_float16_uses_fp32_epsilon_storage(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, jnp.float16)
    inputs = jnp.full(input_shape, 0.2, dtype=jnp.float16)
    variables = layer.init(jax.random.key(3), inputs)
    variables = {
        "params": dict(
            variables["params"],
            kernel=jnp.full_like(variables["params"]["kernel"], 0.3),
        )
    }

    def loss(epsilon_param):
        params = dict(variables["params"], epsilon_param=epsilon_param)
        return layer.apply({"params": params}, inputs).astype(jnp.float32).sum()

    epsilon_param = variables["params"]["epsilon_param"]
    output = jax.jit(layer.apply)(variables, inputs)
    epsilon_grad = jax.jit(jax.grad(loss))(epsilon_param)
    assert epsilon_param.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(jax.nn.softplus(epsilon_param)), epsilon, rtol=2e-6
    )
    assert output.dtype == jnp.float16 and jnp.isfinite(output).all()
    assert jnp.isfinite(epsilon_grad).all() and jnp.abs(epsilon_grad).max() > 0


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
@pytest.mark.parametrize("epsilon", [5e-324, 1e-46, 1e39])
def test_float32_rejects_unrepresentable_epsilon(
    layer_cls, kernel_size, input_shape, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, jnp.float32)
    with pytest.raises(ValueError, match="not representable"):
        layer.init(jax.random.key(4), jnp.ones(input_shape, jnp.float32))


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
@pytest.mark.parametrize("epsilon", [2.0**-1022, 1e150])
def test_float64_extreme_epsilon_is_effective_and_differentiable(
    layer_cls, kernel_size, input_shape, epsilon
):
    with jax.enable_x64():
        layer = _make(layer_cls, kernel_size, epsilon, jnp.float64)
        inputs = jnp.full(input_shape, 0.2, dtype=jnp.float64)
        variables = layer.init(jax.random.key(5), inputs)
        variables = {
            "params": dict(
                variables["params"],
                kernel=jnp.full_like(variables["params"]["kernel"], 0.3),
            )
        }
        epsilon_param = variables["params"]["epsilon_param"]

        def loss(raw_epsilon):
            params = dict(variables["params"], epsilon_param=raw_epsilon)
            return layer.apply({"params": params}, inputs).sum()

        output = layer.apply(variables, inputs)
        epsilon_grad = jax.grad(loss)(epsilon_param)
        np.testing.assert_allclose(
            np.asarray(jax.nn.softplus(epsilon_param)), epsilon, rtol=5e-14
        )
        assert jnp.isfinite(output).all() and jnp.isfinite(epsilon_grad).all()
        assert jnp.abs(epsilon_grad).max() > 0


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES[:2])
def test_float64_rejects_softplus_underflow(layer_cls, kernel_size, input_shape):
    with jax.enable_x64():
        layer = _make(layer_cls, kernel_size, 5e-324, jnp.float64)
        with pytest.raises(ValueError, match="not representable"):
            layer.init(jax.random.key(6), jnp.ones(input_shape, jnp.float64))
