"""Tests for Linen YatEmbed."""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
jnp = jax.numpy

from nmn.linen.embed import YatEmbed


class TestYatEmbed:
    def _init_and_apply(self, embed, method, *args):
        """Helper to init params and apply a method."""
        variables = embed.init(jax.random.PRNGKey(0), *args)
        return embed.apply(variables, *args, method=method)

    def test_forward_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        tokens = jnp.array([0, 5, 10, 99])
        variables = embed.init(jax.random.PRNGKey(0), tokens)
        out = embed.apply(variables, tokens)
        assert out.shape == (4, 32)

    def test_forward_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        tokens = jnp.array([[0, 1, 2], [3, 4, 5]])
        variables = embed.init(jax.random.PRNGKey(0), tokens)
        out = embed.apply(variables, tokens)
        assert out.shape == (2, 3, 16)

    def test_attend_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = jnp.ones((4, 32))
        variables = embed.init(jax.random.PRNGKey(0), query, method=embed.attend)
        out = embed.apply(variables, query, method=embed.attend)
        assert out.shape == (4, 100)

    def test_attend_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        query = jnp.ones((2, 10, 16))
        variables = embed.init(jax.random.PRNGKey(0), query, method=embed.attend)
        out = embed.apply(variables, query, method=embed.attend)
        assert out.shape == (2, 10, 50)

    def test_attend_no_nan(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 32))
        variables = embed.init(jax.random.PRNGKey(0), query, method=embed.attend)
        out = np.array(embed.apply(variables, query, method=embed.attend))
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_attend_positive(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 32))
        variables = embed.init(jax.random.PRNGKey(0), query, method=embed.attend)
        out = np.array(embed.apply(variables, query, method=embed.attend))
        assert np.all(out >= 0)

    def test_constant_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, constant_alpha=True)
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
        variables = embed.init(jax.random.PRNGKey(0), query, method=embed.attend)
        out = np.array(embed.apply(variables, query, method=embed.attend))
        assert not np.any(np.isnan(out))

    def test_no_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, use_alpha=False)
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
        variables = embed.init(jax.random.PRNGKey(0), query, method=embed.attend)
        out = np.array(embed.apply(variables, query, method=embed.attend))
        assert not np.any(np.isnan(out))
        # No alpha param should be in the variables
        assert "alpha" not in variables.get("params", {})

    def test_spherical_mode(self):
        embed = YatEmbed(num_embeddings=50, features=16, spherical=True)
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
        variables = embed.init(jax.random.PRNGKey(0), query, method=embed.attend)
        out = embed.apply(variables, query, method=embed.attend)
        assert out.shape == (4, 50)
        assert not np.any(np.isnan(np.array(out)))
