"""Regression tests for signed additive bias in JAX YAT L1 attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nmn.linen.attention import yat_attention as linen_yat_attention
from nmn.linen.attention import yat_attention_weights as linen_yat_attention_weights
from nmn.nnx.layers.attention.yat_attention import yat_attention as nnx_yat_attention
from nmn.nnx.layers.attention.yat_attention import (
    yat_attention_weights as nnx_yat_attention_weights,
)


@pytest.mark.parametrize(
    "attention_weights",
    [nnx_yat_attention_weights, linen_yat_attention_weights],
    ids=["nnx", "linen"],
)
def test_l1_signed_bias_defines_nonnegative_probability_rows(attention_weights):
    query = jnp.zeros((1, 3, 1, 2), dtype=jnp.float32)
    key = jnp.zeros((1, 3, 1, 2), dtype=jnp.float32)
    bias = jnp.array(
        [[[[1.0e30, -0.5, -jnp.inf], [-3.0, -2.0, -1.0], [2.0, 1.0, 0.5]]]]
    )
    mask = jnp.array(
        [[[[True, True, True], [True, True, True], [False, False, False]]]]
    )

    weights = attention_weights(
        query,
        key,
        bias=bias,
        mask=mask,
        normalization="l1",
        deterministic=True,
    )

    assert np.all(np.isfinite(np.asarray(weights)))
    assert np.all(np.asarray(weights) >= 0.0)
    np.testing.assert_array_equal(np.asarray(weights[0, 0, 0]), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(weights[0, 0, 1]), 1.0 / 3.0)
    np.testing.assert_array_equal(np.asarray(weights[0, 0, 2]), 0.0)
    np.testing.assert_allclose(np.asarray(weights.sum(axis=-1)), [[[1.0, 1.0, 0.0]]])


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_nnx_linen_l1_forward_and_all_operand_gradients_match(compiled):
    query = jax.random.normal(jax.random.key(1), (2, 3, 2, 4)) * 0.2
    key = jax.random.normal(jax.random.key(2), (2, 4, 2, 4)) * 0.2
    value = jax.random.normal(jax.random.key(3), (2, 4, 2, 5))
    bias = jnp.linspace(-0.3, 0.2, 2 * 2 * 3 * 4).reshape((2, 2, 3, 4))
    mask = jnp.ones((2, 2, 3, 4), dtype=jnp.bool_)
    mask = mask.at[0, :, 0, :].set(False)
    mask = mask.at[1, :, 1, 2:].set(False)
    alpha = jnp.array(0.8, dtype=jnp.float32)
    epsilon = jnp.array(2.0e-3, dtype=jnp.float32)
    cotangent = jax.random.normal(jax.random.key(4), (2, 3, 2, 5))

    def evaluate(attention_fn):
        def loss(q, k, v, b, a, e):
            output = attention_fn(
                q,
                k,
                v,
                mask=mask,
                bias=b,
                alpha=a,
                epsilon=e,
                normalization="l1",
                deterministic=True,
            )
            return jnp.vdot(output, cotangent), output

        value_and_grad = jax.value_and_grad(
            loss, argnums=(0, 1, 2, 3, 4, 5), has_aux=True
        )
        if compiled:
            value_and_grad = jax.jit(value_and_grad)
        (_, output), gradients = value_and_grad(query, key, value, bias, alpha, epsilon)
        return output, gradients

    nnx_output, nnx_gradients = evaluate(nnx_yat_attention)
    linen_output, linen_gradients = evaluate(linen_yat_attention)

    np.testing.assert_allclose(linen_output, nnx_output, rtol=2e-6, atol=2e-6)
    for linen_gradient, nnx_gradient in zip(linen_gradients, nnx_gradients):
        assert np.all(np.isfinite(np.asarray(linen_gradient)))
        np.testing.assert_allclose(linen_gradient, nnx_gradient, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "attention_weights",
    [nnx_yat_attention_weights, linen_yat_attention_weights],
    ids=["nnx", "linen"],
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_unknown_normalization_is_rejected_centrally(attention_weights, compiled):
    query = jnp.ones((1, 1, 1, 2), dtype=jnp.float32)
    function = lambda q: attention_weights(  # noqa: E731
        q, q, normalization="signed-l1", deterministic=True
    )
    if compiled:
        function = jax.jit(function)

    with pytest.raises(
        ValueError,
        match="normalization must be one of 'softmax', 'l1', 'softermax'",
    ):
        function(query)
