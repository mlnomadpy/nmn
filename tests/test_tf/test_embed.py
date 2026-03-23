"""Tests for TensorFlow YatEmbed."""

import pytest
import numpy as np

tf = pytest.importorskip("tensorflow")

from nmn.tf.embed import YatEmbed


class TestYatEmbed:
    def test_forward_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        tokens = tf.constant([0, 5, 10, 99])
        out = embed(tokens)
        assert out.shape == (4, 32)

    def test_forward_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        tokens = tf.constant([[0, 1, 2], [3, 4, 5]])
        out = embed(tokens)
        assert out.shape == (2, 3, 16)

    def test_attend_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = tf.random.normal((4, 32))
        out = embed.attend(query)
        assert out.shape == (4, 100)

    def test_attend_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        query = tf.random.normal((2, 10, 16))
        out = embed.attend(query)
        assert out.shape == (2, 10, 50)

    def test_attend_no_nan(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = tf.random.normal((4, 32))
        out = embed.attend(query).numpy()
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_attend_positive(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = tf.random.normal((4, 32))
        out = embed.attend(query).numpy()
        assert np.all(out >= 0)

    def test_constant_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, constant_alpha=True)
        assert embed._constant_alpha_value == pytest.approx(1.4142135, abs=1e-5)
        query = tf.random.normal((4, 16))
        out = embed.attend(query).numpy()
        assert not np.any(np.isnan(out))

    def test_no_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, use_alpha=False)
        assert embed.alpha is None
        query = tf.random.normal((4, 16))
        out = embed.attend(query).numpy()
        assert not np.any(np.isnan(out))

    def test_spherical_mode(self):
        embed = YatEmbed(num_embeddings=50, features=16, spherical=True)
        query = tf.random.normal((4, 16))
        out = embed.attend(query)
        assert out.shape == (4, 50)
        assert not np.any(np.isnan(out.numpy()))

    def test_weight_normalized(self):
        embed = YatEmbed(num_embeddings=50, features=16, weight_normalized=True)
        norms = tf.sqrt(tf.reduce_sum(tf.square(embed.embedding), axis=1)).numpy()
        np.testing.assert_allclose(norms, np.ones(50), atol=1e-5)
