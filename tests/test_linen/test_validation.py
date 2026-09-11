"""Flax Linen constructor and functional validation matrix."""

import math

import jax
import jax.numpy as jnp
import pytest

from nmn.linen import (
    MultiHeadAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)
from nmn.linen.attention import yat_attention_weights


@pytest.mark.parametrize(
    "layer",
    [
        YatConv1D,
        YatConv2D,
        YatConv3D,
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    ],
)
def test_all_conv_families_reject_invalid_feature_counts(layer):
    with pytest.raises(ValueError, match="features must be a positive integer"):
        layer(features=0, kernel_size=(3,))


def test_attention_validates_static_configuration():
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        MultiHeadAttention(num_heads=0)
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        MultiHeadAttention(num_heads=1, dropout_rate=math.nan)
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        yat_attention_weights(None, None, dropout_rate=1.0)


def test_rate_boundaries_remain_supported():
    MultiHeadAttention(num_heads=1, dropout_rate=0.0)
    MultiHeadAttention(num_heads=1, dropout_rate=0.999999)


def test_wrapper_dropout_rate_remains_a_valid_traced_scalar():
    q = jnp.ones((1, 1, 1, 1))

    @jax.jit
    def apply(rate):
        return yat_attention_weights(q, q, dropout_rate=rate, deterministic=True)

    assert apply(jnp.array(0.0)).shape == (1, 1, 1, 1)
