"""Tests for Keras YatEmbed."""

import pytest
import numpy as np

keras = pytest.importorskip("keras")

from nmn.keras.embed import YatEmbed


class TestYatEmbed:
    def test_forward_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        embed.build()
        tokens = np.array([0, 5, 10, 99])
        out = embed(tokens)
        assert out.shape == (4, 32)

    def test_forward_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        embed.build()
        tokens = np.array([[0, 1, 2], [3, 4, 5]])
        out = embed(tokens)
        assert out.shape == (2, 3, 16)

    def test_attend_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        embed.build()
        query = np.random.randn(4, 32).astype(np.float32)
        out = embed.attend(query)
        assert out.shape == (4, 100)

    def test_attend_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        embed.build()
        query = np.random.randn(2, 10, 16).astype(np.float32)
        out = embed.attend(query)
        assert out.shape == (2, 10, 50)

    def test_attend_no_nan(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        embed.build()
        query = np.random.randn(4, 32).astype(np.float32)
        out = np.array(embed.attend(query))
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_attend_positive(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        embed.build()
        query = np.random.randn(4, 32).astype(np.float32)
        out = np.array(embed.attend(query))
        assert np.all(out >= 0)

    def test_constant_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, constant_alpha=True)
        embed.build()
        assert embed._constant_alpha_value == pytest.approx(1.4142135, abs=1e-5)
        query = np.random.randn(4, 16).astype(np.float32)
        out = np.array(embed.attend(query))
        assert not np.any(np.isnan(out))

    def test_no_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, use_alpha=False)
        embed.build()
        assert embed.alpha is None
        query = np.random.randn(4, 16).astype(np.float32)
        out = np.array(embed.attend(query))
        assert not np.any(np.isnan(out))

    def test_spherical_mode(self):
        embed = YatEmbed(num_embeddings=50, features=16, spherical=True)
        embed.build()
        query = np.random.randn(4, 16).astype(np.float32)
        out = embed.attend(query)
        assert out.shape == (4, 50)

    def test_get_config(self):
        embed = YatEmbed(
            num_embeddings=100,
            features=32,
            use_alpha=True,
            constant_alpha=True,
            epsilon=1e-4,
            spherical=True,
        )
        config = embed.get_config()
        assert config["num_embeddings"] == 100
        assert config["features"] == 32
        assert config["constant_alpha"] is True
        assert config["spherical"] is True
