"""Flax NNX constructor and functional validation matrix."""

import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from nmn.nnx import MultiHeadAttention, YatConv, YatConvTranspose
from nmn.nnx.layers.attention.yat_attention import yat_attention_weights


@pytest.mark.parametrize("kernel_size", [3, (3, 3), (3, 3, 3)])
@pytest.mark.parametrize("layer", [YatConv, YatConvTranspose])
@pytest.mark.parametrize("rate", [math.nan, math.inf, -0.1, 1.0, 2.0])
def test_all_conv_dimensions_reject_invalid_dropconnect(layer, kernel_size, rate):
    with pytest.raises(ValueError, match="drop_rate must be a finite real number"):
        layer(2, 2, kernel_size, drop_rate=rate, rngs=nnx.Rngs(0))


def test_attention_validates_before_rng_consumption():
    rngs = nnx.Rngs(0)
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        MultiHeadAttention(0, 8, rngs=rngs)
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        MultiHeadAttention(2, 8, dropout_rate=math.nan, rngs=rngs)
    q = jnp.zeros((1, 1, 1, 1))
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        yat_attention_weights(q, q, dropout_rate=1.0)


def test_rate_boundaries_remain_supported():
    YatConv(1, 1, 1, drop_rate=0.0, rngs=nnx.Rngs(0))
    MultiHeadAttention(1, 1, dropout_rate=0.999999, rngs=nnx.Rngs(0))


def test_functional_dropout_rate_remains_a_valid_traced_scalar():
    q = jnp.ones((1, 1, 1, 1))

    @jax.jit
    def apply(rate):
        return yat_attention_weights(q, q, dropout_rate=rate, deterministic=True)

    assert apply(jnp.array(0.0)).shape == (1, 1, 1, 1)


@pytest.mark.parametrize("rate", [jnp.array(1.0), jnp.array([0.5]), jnp.array(False)])
def test_jax_rates_still_validate_eagerly(rate):
    q = jnp.ones((1, 1, 1, 1))
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        yat_attention_weights(q, q, dropout_rate=rate, deterministic=True)


def test_boolean_tracer_is_not_accepted_as_a_rate():
    q = jnp.ones((1, 1, 1, 1))

    @jax.jit
    def apply(rate):
        return yat_attention_weights(q, q, dropout_rate=rate, deterministic=True)

    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        apply(jnp.array(False))


@pytest.mark.parametrize("rate", [jnp.array(0.5 + 0j), jnp.array(0.5 + 1j)])
def test_complex_rates_are_rejected_eagerly_and_when_traced(rate):
    q = jnp.ones((1, 1, 1, 1))

    def apply(value):
        return yat_attention_weights(q, q, dropout_rate=value, deterministic=True)

    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        apply(rate)
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        jax.jit(apply)(rate)
