"""Tests for NNX Embed layer."""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
jnp = jax.numpy

from flax import nnx
from nmn.nnx.layers.embed import Embed


class TestEmbed:
    def test_forward_shape(self):
        embed = Embed(num_embeddings=100, features=32, rngs=nnx.Rngs(0))
        tokens = jnp.array([0, 5, 10, 99])
        out = embed(tokens)
        assert out.shape == (4, 32)

    def test_forward_batch(self):
        embed = Embed(num_embeddings=50, features=16, rngs=nnx.Rngs(0))
        tokens = jnp.array([[0, 1, 2], [3, 4, 5]])
        out = embed(tokens)
        assert out.shape == (2, 3, 16)

    def test_attend_shape(self):
        embed = Embed(num_embeddings=100, features=32, rngs=nnx.Rngs(0))
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 32))
        out = embed.attend(query)
        assert out.shape == (4, 100)

    def test_attend_batch(self):
        embed = Embed(num_embeddings=50, features=16, rngs=nnx.Rngs(0))
        query = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 16))
        out = embed.attend(query)
        assert out.shape == (2, 10, 50)

    def test_attend_no_nan(self):
        embed = Embed(num_embeddings=100, features=32, rngs=nnx.Rngs(0))
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 32))
        out = np.array(embed.attend(query))
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_attend_positive(self):
        embed = Embed(num_embeddings=100, features=32, rngs=nnx.Rngs(0))
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 32))
        out = np.array(embed.attend(query))
        assert np.all(out >= 0)

    def test_constant_alpha(self):
        embed = Embed(
            num_embeddings=50, features=16, constant_alpha=True, rngs=nnx.Rngs(0)
        )
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
        out = np.array(embed.attend(query))
        assert not np.any(np.isnan(out))

    def test_no_alpha(self):
        embed = Embed(
            num_embeddings=50, features=16, use_alpha=False, rngs=nnx.Rngs(0)
        )
        assert embed.alpha is None
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
        out = np.array(embed.attend(query))
        assert not np.any(np.isnan(out))

    def test_learnable_alpha(self):
        embed = Embed(
            num_embeddings=50, features=16, use_alpha=True, rngs=nnx.Rngs(0)
        )
        assert embed.alpha is not None

    def test_spherical_mode(self):
        embed = Embed(
            num_embeddings=50, features=16, spherical=True, rngs=nnx.Rngs(0)
        )
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 16))
        out = embed.attend(query)
        assert out.shape == (4, 50)
        assert not np.any(np.isnan(np.array(out)))

    def test_weight_normalized(self):
        embed = Embed(
            num_embeddings=50, features=16, weight_normalized=True, rngs=nnx.Rngs(0)
        )
        norms = np.sqrt(np.sum(np.array(embed.embedding[...]) ** 2, axis=1))
        np.testing.assert_allclose(norms, np.ones(50), atol=1e-5)

    def test_gradient_flow(self):
        embed = Embed(num_embeddings=50, features=16, rngs=nnx.Rngs(0))
        query = jax.random.normal(jax.random.PRNGKey(1), (4, 16))

        def loss_fn(embed, query):
            return jnp.sum(embed.attend(query))

        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(embed, query)
        assert grads.embedding[...] is not None
