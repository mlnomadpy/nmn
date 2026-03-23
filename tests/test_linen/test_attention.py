"""Tests for Linen YAT attention."""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
jnp = jax.numpy

from nmn.linen.attention import (
    normalize_qk,
    yat_attention,
    yat_attention_normalized,
    MultiHeadAttention,
)


class TestAttentionFunctions:
    def test_normalize_qk(self):
        q = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 8))
        k = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4, 8))
        q_n, k_n = normalize_qk(q, k)
        q_norms = np.sqrt(np.sum(np.array(q_n) ** 2, axis=-1))
        np.testing.assert_allclose(q_norms, np.ones_like(q_norms), atol=1e-5)

    def test_attention_output_shape(self):
        q = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 8))
        k = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 4, 8))
        v = jax.random.normal(jax.random.PRNGKey(2), (2, 7, 4, 16))
        out = yat_attention(q, k, v)
        assert out.shape == (2, 5, 4, 16)

    def test_attention_no_nan(self):
        q = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 8))
        k = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 4, 8))
        v = jax.random.normal(jax.random.PRNGKey(2), (2, 7, 4, 8))
        out = np.array(yat_attention(q, k, v))
        assert not np.any(np.isnan(out))

    def test_attention_normalized(self):
        q = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 8))
        k = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 4, 8))
        v = jax.random.normal(jax.random.PRNGKey(2), (2, 7, 4, 8))
        out = yat_attention_normalized(q, k, v)
        assert out.shape == (2, 5, 4, 8)
        assert not np.any(np.isnan(np.array(out)))


class TestMultiHeadAttention:
    def test_self_attention(self):
        model = MultiHeadAttention(num_heads=4)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), x)
        out = model.apply(variables, x)
        assert out.shape == (2, 10, 32)

    def test_cross_attention(self):
        model = MultiHeadAttention(num_heads=4)
        q = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 32))
        kv = jax.random.normal(jax.random.PRNGKey(2), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), q, kv)
        out = model.apply(variables, q, kv)
        assert out.shape == (2, 5, 32)

    def test_no_nan(self):
        model = MultiHeadAttention(num_heads=4)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), x)
        out = np.array(model.apply(variables, x))
        assert not np.any(np.isnan(out))

    def test_constant_alpha(self):
        model = MultiHeadAttention(num_heads=4, constant_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), x)
        out = np.array(model.apply(variables, x))
        assert not np.any(np.isnan(out))

    def test_no_alpha(self):
        model = MultiHeadAttention(num_heads=4, use_alpha=False)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), x)
        out = np.array(model.apply(variables, x))
        assert not np.any(np.isnan(out))
        # No alpha param
        assert "alpha" not in variables.get("params", {})

    def test_no_bias(self):
        model = MultiHeadAttention(num_heads=4, use_bias=False)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), x)
        out = np.array(model.apply(variables, x))
        assert not np.any(np.isnan(out))

    def test_custom_qkv_features(self):
        model = MultiHeadAttention(num_heads=4, qkv_features=64)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), x)
        out = model.apply(variables, x)
        assert out.shape == (2, 10, 64)

    def test_custom_out_features(self):
        model = MultiHeadAttention(num_heads=4, qkv_features=64, out_features=128)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), x)
        out = model.apply(variables, x)
        assert out.shape == (2, 10, 128)

    def test_spherical(self):
        model = MultiHeadAttention(num_heads=4, spherical=True)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        variables = model.init(jax.random.PRNGKey(0), x)
        out = np.array(model.apply(variables, x))
        assert not np.any(np.isnan(out))

    def test_with_mask(self):
        model = MultiHeadAttention(num_heads=4)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 32))
        mask = jnp.ones((2, 4, 10, 10), dtype=bool)
        variables = model.init(jax.random.PRNGKey(0), x, mask=mask)
        out = model.apply(variables, x, mask=mask)
        assert out.shape == (2, 10, 32)
