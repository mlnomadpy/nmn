"""Tests for TensorFlow YAT attention."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nmn.tf.attention import (
    MultiHeadYatAttention,
    normalize_qk,
    yat_attention,
    yat_attention_normalized,
    yat_attention_weights,
)


class TestAttentionFunctions:
    def test_negative_scale_cannot_make_masked_key_win_softmax(self):
        q = tf.ones((1, 1, 1, 4))
        k = tf.ones((1, 2, 1, 4))
        weights = yat_attention_weights(
            q, k, mask=tf.constant([True, False]), scale=-1.0
        )
        np.testing.assert_array_equal(weights.numpy(), [[[[1.0, 0.0]]]])

    @pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
    @pytest.mark.parametrize("spherical", [False, True])
    def test_fully_masked_rows_are_zero_with_finite_tf_function_gradients(
        self, spherical, dtype
    ):
        q = tf.Variable(tf.random.normal((1, 2, 2, 4), seed=70, dtype=dtype))
        k = tf.Variable(tf.random.normal((1, 3, 2, 4), seed=71, dtype=dtype))
        v = tf.Variable(tf.random.normal((1, 3, 2, 5), seed=72, dtype=dtype))
        mask = tf.constant([[[[False, False, False], [True, False, True]]]])

        @tf.function
        def apply(q, k, v):
            return yat_attention(q, k, v, mask=mask, spherical=spherical)

        with tf.GradientTape() as tape:
            output = apply(q, k, v)
            loss = tf.reduce_sum(output)
        grads = tape.gradient(loss, (q, k, v))
        weights = yat_attention_weights(q, k, mask=mask, spherical=spherical)

        np.testing.assert_array_equal(output[:, 0].numpy(), 0.0)
        np.testing.assert_array_equal(weights[..., 0, :].numpy(), 0.0)
        assert all(np.all(np.isfinite(grad.numpy())) for grad in grads)

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
    @pytest.mark.parametrize("mask_rank", [2, 4])
    @pytest.mark.parametrize("cross_attention", [False, True])
    def test_fully_masked_rows_stay_zero_after_biased_projection(
        self, cross_attention, mask_rank
    ):
        attn = MultiHeadYatAttention(embed_dim=8, num_heads=2)
        query = tf.Variable(tf.random.normal((1, 2, 8), seed=73))
        context = tf.Variable(tf.random.normal((1, 3, 8), seed=74))
        _ = attn(query)
        attn.out_bias.assign(tf.fill(attn.out_bias.shape, 3.0))
        kv_length = 3 if cross_attention else 2
        shape = (2, kv_length) if mask_rank == 2 else (1, 1, 2, kv_length)
        mask_array = np.ones(shape, dtype=bool)
        mask_array[..., 0, :] = False
        mask = tf.constant(mask_array)

        @tf.function
        def apply(query, context, mask):
            if cross_attention:
                return attn(query, key=context, value=context, mask=mask)
            return attn(query, mask=mask)

        with tf.GradientTape() as tape:
            output = apply(query, context, mask)
            loss = tf.reduce_sum(tf.square(output))
        operands = (query, context) if cross_attention else (query,)
        grads = tape.gradient(loss, operands)
        np.testing.assert_array_equal(output[:, 0].numpy(), 0.0)
        assert np.all(np.isfinite(output.numpy()))
        assert all(np.all(np.isfinite(grad.numpy())) for grad in grads)

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
