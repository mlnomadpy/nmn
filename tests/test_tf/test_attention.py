"""Tests for TensorFlow YAT attention."""

import pytest
import numpy as np

tf = pytest.importorskip("tensorflow")

from nmn.tf.attention import (
    normalize_qk,
    yat_attention_weights,
    yat_attention,
    yat_attention_normalized,
    MultiHeadYatAttention,
)


class TestAttentionFunctions:
    def test_normalize_qk(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 5, 4, 8))
        q_n, k_n = normalize_qk(q, k)
        q_norms = tf.sqrt(tf.reduce_sum(tf.square(q_n), axis=-1)).numpy()
        np.testing.assert_allclose(q_norms, np.ones_like(q_norms), atol=1e-5)

    def test_attention_weights_shape(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        weights = yat_attention_weights(q, k)
        assert weights.shape == (2, 4, 5, 7)

    def test_attention_weights_sum_to_one(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        weights = yat_attention_weights(q, k)
        sums = tf.reduce_sum(weights, axis=-1).numpy()
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_attention_output_shape(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        v = tf.random.normal((2, 7, 4, 16))
        out = yat_attention(q, k, v)
        assert out.shape == (2, 5, 4, 16)

    def test_attention_no_nan(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        v = tf.random.normal((2, 7, 4, 8))
        out = yat_attention(q, k, v).numpy()
        assert not np.any(np.isnan(out))

    def test_attention_normalized_shape(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        v = tf.random.normal((2, 7, 4, 8))
        out = yat_attention_normalized(q, k, v)
        assert out.shape == (2, 5, 4, 8)

    def test_attention_with_alpha(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        v = tf.random.normal((2, 7, 4, 8))
        alpha = tf.Variable([1.5])
        out = yat_attention(q, k, v, alpha=alpha)
        assert not np.any(np.isnan(out.numpy()))

    def test_attention_with_scale(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        v = tf.random.normal((2, 7, 4, 8))
        out = yat_attention(q, k, v, scale=1.414)
        assert not np.any(np.isnan(out.numpy()))

    def test_attention_with_mask(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        v = tf.random.normal((2, 7, 4, 8))
        mask = tf.ones((2, 4, 5, 7), dtype=tf.bool)
        out = yat_attention(q, k, v, mask=mask)
        assert out.shape == (2, 5, 4, 8)

    def test_spherical_mode(self):
        q = tf.random.normal((2, 5, 4, 8))
        k = tf.random.normal((2, 7, 4, 8))
        v = tf.random.normal((2, 7, 4, 8))
        out = yat_attention(q, k, v, spherical=True)
        assert not np.any(np.isnan(out.numpy()))


class TestMultiHeadYatAttention:
    def test_self_attention(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = tf.random.normal((2, 10, 32))
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_cross_attention(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        q = tf.random.normal((2, 5, 32))
        kv = tf.random.normal((2, 10, 32))
        out = attn(q, key=kv)
        assert out.shape == (2, 5, 32)

    def test_no_nan(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = tf.random.normal((2, 10, 32))
        out = attn(x).numpy()
        assert not np.any(np.isnan(out))

    def test_constant_alpha(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, constant_alpha=True)
        x = tf.random.normal((2, 10, 32))
        out = attn(x).numpy()
        assert not np.any(np.isnan(out))

    def test_no_alpha(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, use_alpha=False)
        x = tf.random.normal((2, 10, 32))
        out = attn(x).numpy()
        assert not np.any(np.isnan(out))

    def test_no_bias(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, use_bias=False)
        x = tf.random.normal((2, 10, 32))
        out = attn(x).numpy()
        assert not np.any(np.isnan(out))

    def test_no_out_proj(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, use_out_proj=False)
        x = tf.random.normal((2, 10, 32))
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_spherical(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, spherical=True)
        x = tf.random.normal((2, 10, 32))
        out = attn(x).numpy()
        assert not np.any(np.isnan(out))

    def test_invalid_embed_dim(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadYatAttention(embed_dim=33, num_heads=4)

    def test_with_mask(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = tf.random.normal((2, 10, 32))
        mask = tf.ones((2, 4, 10, 10), dtype=tf.bool)
        out = attn(x, mask=mask)
        assert out.shape == (2, 10, 32)
