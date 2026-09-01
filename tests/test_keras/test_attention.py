"""Tests for Keras YAT attention."""

import pytest
import numpy as np

keras = pytest.importorskip("keras")

from nmn.keras.attention import (
    normalize_qk,
    yat_attention_weights,
    yat_attention,
    yat_attention_normalized,
    MultiHeadYatAttention,
)

to_numpy = keras.ops.convert_to_numpy


class TestAttentionFunctions:
    def test_normalize_qk(self):
        q = np.random.randn(2, 5, 4, 8).astype(np.float32)
        k = np.random.randn(2, 5, 4, 8).astype(np.float32)
        q_n, k_n = normalize_qk(q, k)
        q_norms = np.sqrt(np.sum(to_numpy(q_n) ** 2, axis=-1))
        np.testing.assert_allclose(q_norms, np.ones_like(q_norms), atol=1e-5)

    def test_attention_weights_shape(self):
        q = np.random.randn(2, 5, 4, 8).astype(np.float32)
        k = np.random.randn(2, 7, 4, 8).astype(np.float32)
        weights = yat_attention_weights(q, k)
        assert weights.shape == (2, 4, 5, 7)

    def test_attention_weights_sum_to_one(self):
        q = np.random.randn(2, 5, 4, 8).astype(np.float32)
        k = np.random.randn(2, 7, 4, 8).astype(np.float32)
        weights = to_numpy(yat_attention_weights(q, k))
        sums = weights.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_attention_output_shape(self):
        q = np.random.randn(2, 5, 4, 8).astype(np.float32)
        k = np.random.randn(2, 7, 4, 8).astype(np.float32)
        v = np.random.randn(2, 7, 4, 16).astype(np.float32)
        out = yat_attention(q, k, v)
        assert out.shape == (2, 5, 4, 16)

    def test_attention_no_nan(self):
        q = np.random.randn(2, 5, 4, 8).astype(np.float32)
        k = np.random.randn(2, 7, 4, 8).astype(np.float32)
        v = np.random.randn(2, 7, 4, 8).astype(np.float32)
        out = to_numpy(yat_attention(q, k, v))
        assert not np.any(np.isnan(out))

    def test_attention_normalized_shape(self):
        q = np.random.randn(2, 5, 4, 8).astype(np.float32)
        k = np.random.randn(2, 7, 4, 8).astype(np.float32)
        v = np.random.randn(2, 7, 4, 8).astype(np.float32)
        out = yat_attention_normalized(q, k, v)
        assert out.shape == (2, 5, 4, 8)

    def test_spherical_mode(self):
        q = np.random.randn(2, 5, 4, 8).astype(np.float32)
        k = np.random.randn(2, 7, 4, 8).astype(np.float32)
        v = np.random.randn(2, 7, 4, 8).astype(np.float32)
        out = to_numpy(yat_attention(q, k, v, spherical=True))
        assert not np.any(np.isnan(out))


class TestMultiHeadYatAttention:
    def test_keras_sequence_mask_is_expanded_and_preserved(self):
        tokens = keras.Input(shape=(3,), dtype="int32")
        embedded = keras.layers.Embedding(10, 4, mask_zero=True)(tokens)
        attended = MultiHeadYatAttention(embed_dim=4, num_heads=2)(embedded)
        assert attended._keras_mask is not None
        model = keras.Model(tokens, attended)

        token_values = np.array([[1, 2, 0], [3, 0, 0]], dtype=np.int32)
        output = to_numpy(model(token_values))

        assert output.shape == (2, 3, 4)
        np.testing.assert_array_equal(output[0, 2], np.zeros(4))
        np.testing.assert_array_equal(output[1, 1:], np.zeros((2, 4)))

    def test_explicit_rank_two_attention_mask_remains_supported(self):
        layer = MultiHeadYatAttention(embed_dim=4, num_heads=2)
        inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24.0
        attention_mask = np.tril(np.ones((3, 3), dtype=bool))

        output = layer(inputs, attention_mask=attention_mask)

        assert output.shape == (2, 3, 4)

    def test_self_attention(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = np.random.randn(2, 10, 32).astype(np.float32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_cross_attention(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        q = np.random.randn(2, 5, 32).astype(np.float32)
        kv = np.random.randn(2, 10, 32).astype(np.float32)
        out = attn(q, key=kv)
        assert out.shape == (2, 5, 32)

    def test_no_nan(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = np.random.randn(2, 10, 32).astype(np.float32)
        out = to_numpy(attn(x))
        assert not np.any(np.isnan(out))

    def test_constant_alpha(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, constant_alpha=True)
        x = np.random.randn(2, 10, 32).astype(np.float32)
        out = to_numpy(attn(x))
        assert not np.any(np.isnan(out))

    def test_no_alpha(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, use_alpha=False)
        x = np.random.randn(2, 10, 32).astype(np.float32)
        out = to_numpy(attn(x))
        assert not np.any(np.isnan(out))

    def test_no_bias(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, use_bias=False)
        x = np.random.randn(2, 10, 32).astype(np.float32)
        out = to_numpy(attn(x))
        assert not np.any(np.isnan(out))

    def test_invalid_embed_dim(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadYatAttention(embed_dim=33, num_heads=4)

    def test_get_config(self):
        attn = MultiHeadYatAttention(
            embed_dim=64, num_heads=8, constant_alpha=True, dropout=0.1
        )
        config = attn.get_config()
        assert config["embed_dim"] == 64
        assert config["num_heads"] == 8
        assert config["constant_alpha"] is True
        assert config["dropout"] == 0.1
