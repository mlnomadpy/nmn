"""Large-magnitude low-precision regressions for Linen YAT arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmn.linen import (
    YatConv1D, YatConv2D, YatConv3D,
    YatConvTranspose1D, YatConvTranspose2D, YatConvTranspose3D,
    YatNMN, yat_attention,
)


LOWP_DTYPES = (jnp.float16, jnp.bfloat16)
CONVS = (
    (YatConv1D, (1, 1, 1), (1,)),
    (YatConv2D, (1, 1, 1, 1), (1, 1)),
    (YatConv3D, (1, 1, 1, 1, 1), (1, 1, 1)),
    (YatConvTranspose1D, (1, 1, 1), (1,)),
    (YatConvTranspose2D, (1, 1, 1, 1), (1, 1)),
    (YatConvTranspose3D, (1, 1, 1, 1, 1), (1, 1, 1)),
)


def _as_f32(value):
    return np.asarray(value, dtype=np.float32)


def _dense_value_and_grads(dtype):
    layer = YatNMN(features=1, use_bias=False, use_alpha=False, epsilon=1.0,
                   dtype=dtype, param_dtype=dtype)
    x = jnp.array([[100.0, 100.0]], dtype=dtype)
    variables = layer.init(jax.random.key(0), x)
    kernel = jnp.array([[-100.0, -99.0]], dtype=dtype)

    def loss(input_value, kernel_value):
        params = dict(variables["params"], kernel=kernel_value)
        return layer.apply({"params": params}, input_value).astype(jnp.float32).sum()

    output = layer.apply(
        {"params": dict(variables["params"], kernel=kernel)}, x
    )
    gradients = jax.grad(loss, argnums=(0, 1))(x, kernel)
    return output, gradients


@pytest.mark.parametrize("dtype", LOWP_DTYPES)
def test_large_magnitude_dense_matches_fp32_forward_and_gradients(dtype):
    ref_output, ref_grads = _dense_value_and_grads(jnp.float32)
    output, grads = _dense_value_and_grads(dtype)
    np.testing.assert_allclose(_as_f32(output), _as_f32(ref_output), rtol=5e-3, atol=0.15)
    for actual, expected in zip(grads, ref_grads):
        np.testing.assert_allclose(_as_f32(actual), _as_f32(expected), rtol=5e-3, atol=0.15)


def _conv_value_and_grads(layer_cls, shape, kernel_size, dtype):
    def constant(key, kernel_shape, init_dtype):
        del key
        return jnp.full(kernel_shape, -100.0, init_dtype)
    layer = layer_cls(
        features=1, kernel_size=kernel_size, use_bias=False, use_alpha=False,
        epsilon=1.0, dtype=dtype, param_dtype=dtype, kernel_init=constant,
    )
    x = jnp.full(shape, 100.0, dtype=dtype)
    variables = layer.init(jax.random.key(0), x)
    kernel = variables["params"]["kernel"]

    def loss(input_value, kernel_value):
        return layer.apply(
            {"params": {"kernel": kernel_value}}, input_value
        ).astype(jnp.float32).sum()

    output = layer.apply(variables, x)
    gradients = jax.grad(loss, argnums=(0, 1))(x, kernel)
    return output, gradients


@pytest.mark.parametrize("layer_cls,shape,kernel_size", CONVS)
@pytest.mark.parametrize("dtype", LOWP_DTYPES)
def test_large_magnitude_conv_families_match_fp32(layer_cls, shape, kernel_size, dtype):
    ref_output, ref_grads = _conv_value_and_grads(
        layer_cls, shape, kernel_size, jnp.float32
    )
    output, grads = _conv_value_and_grads(layer_cls, shape, kernel_size, dtype)
    np.testing.assert_allclose(_as_f32(output), _as_f32(ref_output), rtol=5e-3, atol=0.15)
    for actual, expected in zip(grads, ref_grads):
        np.testing.assert_allclose(_as_f32(actual), _as_f32(expected), rtol=5e-3, atol=0.15)


@pytest.mark.parametrize("dtype", LOWP_DTYPES)
def test_default_orthogonal_conv_initializer_is_lowp_cpu_safe(dtype):
    layer = YatConv1D(features=2, kernel_size=(1,), dtype=dtype, param_dtype=dtype)
    variables = layer.init(jax.random.key(0), jnp.ones((1, 2, 2), dtype=dtype))
    assert variables["params"]["kernel"].dtype == dtype


@pytest.mark.parametrize("dtype", LOWP_DTYPES)
def test_large_magnitude_attention_matches_fp32_forward_and_gradients(dtype):
    def evaluate(q_dtype):
        query = jnp.full((1, 1, 1, 2), 100.0, q_dtype)
        key = jnp.full((1, 2, 1, 2), 100.0, q_dtype)
        value = jnp.array([[[[1.0]], [[2.0]]]], q_dtype)
        def fn(q, k, v):
            return (
                yat_attention(q, k, v, deterministic=True, epsilon=1.0)
                .astype(jnp.float32)
                .sum()
                * 0.015625
            )
        output = yat_attention(query, key, value, deterministic=True, epsilon=1.0)
        return output, jax.grad(fn, argnums=(0, 1, 2))(query, key, value)

    ref_output, ref_grads = evaluate(jnp.float32)
    output, grads = evaluate(dtype)
    np.testing.assert_allclose(_as_f32(output), _as_f32(ref_output), atol=2e-3)
    for actual, expected in zip(grads, ref_grads):
        np.testing.assert_allclose(_as_f32(actual), _as_f32(expected), rtol=7e-3, atol=8.0)


def test_low_precision_core_preserves_genuine_nan():
    layer = YatNMN(
        features=1, use_bias=False, use_alpha=False, dtype=jnp.float16,
        param_dtype=jnp.float16,
        kernel_init=lambda key, shape, dtype: jnp.ones(shape, dtype),
    )
    x = jnp.array([[jnp.nan, 1.0]], dtype=jnp.float16)
    assert jnp.isnan(layer.apply(layer.init(jax.random.key(0), x), x)).all()

    conv = YatConv1D(
        features=1, kernel_size=(1,), use_bias=False, use_alpha=False,
        dtype=jnp.float16, param_dtype=jnp.float16,
        kernel_init=lambda key, shape, dtype: jnp.ones(shape, dtype),
    )
    conv_x = jnp.array([[[jnp.nan]]], dtype=jnp.float16)
    assert jnp.isnan(conv.apply(conv.init(jax.random.key(1), conv_x), conv_x)).all()

    query = jnp.array([[[[jnp.nan, 100.0]]]], dtype=jnp.float16)
    key = jnp.full((1, 2, 1, 2), 100.0, dtype=jnp.float16)
    value = jnp.ones((1, 2, 1, 1), dtype=jnp.float16)
    assert jnp.isnan(yat_attention(query, key, value, deterministic=True)).all()
