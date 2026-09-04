"""Tests for Keras YAT attention."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from nmn.keras.attention import (
    MultiHeadYatAttention,
    normalize_qk,
    yat_attention,
    yat_attention_normalized,
    yat_attention_weights,
)

to_numpy = keras.ops.convert_to_numpy
BACKEND = keras.backend.backend()


def _compiled_output_and_input_gradient(
    layer, query, context, mask, *, explicit_attention_mask=False
):
    def call_layer(query_value):
        mask_kwargs = (
            {"attention_mask": mask} if explicit_attention_mask else {"mask": mask}
        )
        if context is None:
            return layer(query_value, **mask_kwargs)
        return layer(query_value, key=context, value=context, **mask_kwargs)

    if BACKEND == "jax":
        import jax
        import jax.numpy as jnp

        output = jax.jit(call_layer)(query)
        gradient = jax.jit(jax.grad(lambda value: jnp.sum(call_layer(value))))(query)
        return output, gradient

    if BACKEND == "torch":
        import torch

        query = query.detach().clone().requires_grad_(True)
        apply = torch.compile(call_layer, backend="eager")
        output = apply(query)
        return output, torch.autograd.grad(output.sum(), query)[0]

    if BACKEND == "tensorflow":
        import tensorflow as tf

        @tf.function
        def apply(query_value):
            with tf.GradientTape() as tape:
                tape.watch(query_value)
                output = call_layer(query_value)
                loss = tf.reduce_sum(output)
            return output, tape.gradient(loss, query_value)

        return apply(query)

    raise pytest.skip.Exception("requires the JAX, Torch, or TensorFlow Keras backend")


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

    @pytest.mark.parametrize("cross_attention", [False, True])
    def test_sequence_mask_is_unambiguous_when_batch_equals_query_length(
        self, cross_attention
    ):
        layer = MultiHeadYatAttention(embed_dim=4, num_heads=2)
        query = keras.ops.convert_to_tensor(
            np.arange(16, dtype=np.float32).reshape(2, 2, 4) / 16.0
        )
        context = (
            keras.ops.convert_to_tensor(
                np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24.0
            )
            if cross_attention
            else None
        )
        sequence_mask = keras.ops.convert_to_tensor(
            [[True, False], [True, True]], dtype="bool"
        )
        if context is None:
            layer(query)
        else:
            layer(query, key=context, value=context)
        layer.out_bias.assign(np.full(layer.out_bias.shape, 3.0, dtype=np.float32))

        output, gradient = _compiled_output_and_input_gradient(
            layer, query, context, sequence_mask
        )

        np.testing.assert_array_equal(to_numpy(output)[0, 1], np.zeros(4))
        assert np.all(np.isfinite(to_numpy(output)))
        assert np.all(np.isfinite(to_numpy(gradient)))

    @pytest.mark.parametrize("cross_attention", [False, True])
    def test_symbolic_sequence_mask_supports_dynamic_lengths(self, cross_attention):
        query = keras.Input(shape=(None, 4))
        sequence_mask = keras.Input(shape=(None,), dtype="bool")
        layer = MultiHeadYatAttention(embed_dim=4, num_heads=2)
        if cross_attention:
            context = keras.Input(shape=(None, 4))
            output = layer(query, key=context, value=context, mask=sequence_mask)
            model = keras.Model((query, context, sequence_mask), output)
            inputs = (
                np.ones((2, 2, 4), dtype=np.float32),
                np.ones((2, 3, 4), dtype=np.float32),
                np.array([[True, False], [True, True]]),
            )
        else:
            output = layer(query, mask=sequence_mask)
            model = keras.Model((query, sequence_mask), output)
            inputs = (
                np.ones((2, 2, 4), dtype=np.float32),
                np.array([[True, False], [True, True]]),
            )

        result = to_numpy(model(inputs))
        np.testing.assert_array_equal(result[0, 1], np.zeros(4))

    @pytest.mark.parametrize("mask_rank", [3, 4])
    def test_legacy_pairwise_mask_matches_attention_mask(self, mask_rank):
        layer = MultiHeadYatAttention(embed_dim=4, num_heads=2)
        inputs = keras.ops.convert_to_tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24.0
        )
        pairwise_mask = np.ones(
            (1, 3, 3) if mask_rank == 3 else (2, 1, 3, 3), dtype=bool
        )
        pairwise_mask[..., 1, :] = False
        layer(inputs)

        legacy_output, legacy_gradient = _compiled_output_and_input_gradient(
            layer, inputs, None, pairwise_mask
        )
        explicit_output, explicit_gradient = _compiled_output_and_input_gradient(
            layer,
            inputs,
            None,
            pairwise_mask,
            explicit_attention_mask=True,
        )

        np.testing.assert_allclose(
            to_numpy(legacy_output), to_numpy(explicit_output), atol=1e-6
        )
        np.testing.assert_allclose(
            to_numpy(legacy_gradient), to_numpy(explicit_gradient), atol=1e-6
        )
        np.testing.assert_array_equal(to_numpy(legacy_output)[:, 1], 0.0)

    def test_legacy_pairwise_mask_combines_with_attention_mask(self):
        layer = MultiHeadYatAttention(embed_dim=4, num_heads=2)
        inputs = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 24.0
        legacy_mask = np.ones((2, 1, 3, 3), dtype=bool)
        legacy_mask[:, :, 1, :] = False
        explicit_mask = np.tril(np.ones((3, 3), dtype=bool))

        combined_output = layer(inputs, mask=legacy_mask, attention_mask=explicit_mask)
        expected_output = layer(
            inputs,
            attention_mask=np.logical_and(legacy_mask, explicit_mask),
        )

        np.testing.assert_allclose(
            to_numpy(combined_output), to_numpy(expected_output), atol=1e-6
        )

    def test_rank_one_mask_is_rejected(self):
        layer = MultiHeadYatAttention(embed_dim=4, num_heads=2)
        inputs = np.ones((1, 2, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="rank-2"):
            layer(inputs, mask=np.ones((2,), dtype=bool))

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
