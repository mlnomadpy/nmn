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
    @pytest.mark.parametrize("spherical", [False, True])
    def test_fully_masked_rows_are_zero_with_finite_jit_gradients(self, spherical):
        q = jax.random.normal(jax.random.key(70), (1, 2, 2, 4))
        k = jax.random.normal(jax.random.key(71), (1, 3, 2, 4))
        v = jax.random.normal(jax.random.key(72), (1, 3, 2, 5))
        mask = jnp.array([[[[False, False, False], [True, False, True]]]])

        def apply(q, k, v):
            return yat_attention(q, k, v, mask=mask, spherical=spherical)

        eager = apply(q, k, v)
        compiled = jax.jit(apply)(q, k, v)
        grads = jax.jit(jax.grad(lambda q, k, v: jnp.sum(apply(q, k, v)), (0, 1, 2)))(
            q, k, v
        )
        np.testing.assert_array_equal(np.asarray(eager[:, 0]), 0.0)
        np.testing.assert_allclose(compiled, eager, rtol=2e-7, atol=1e-7)
        assert all(np.all(np.isfinite(np.asarray(grad))) for grad in grads)

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
    @pytest.mark.parametrize("mask_rank", [2, 4])
    @pytest.mark.parametrize("normalization", ["softmax", "l1"])
    @pytest.mark.parametrize("cross_attention", [False, True])
    def test_fully_masked_rows_stay_zero_after_biased_projection(
        self, normalization, cross_attention, mask_rank
    ):
        model = MultiHeadAttention(
            num_heads=2,
            normalization=normalization,
            bias_init=jax.nn.initializers.constant(3.0),
        )
        query = jax.random.normal(jax.random.key(73), (1, 2, 8))
        context = jax.random.normal(jax.random.key(74), (1, 3, 8))
        kv_length = 3 if cross_attention else 2
        shape = (2, kv_length) if mask_rank == 2 else (1, 1, 2, kv_length)
        mask = jnp.ones(shape, dtype=jnp.bool_).at[..., 0, :].set(False)
        args = (query, context, context) if cross_attention else (query,)
        variables = model.init(jax.random.key(75), *args, mask=mask)
        output = jax.jit(lambda variables, *args: model.apply(variables, *args, mask=mask))(
            variables, *args
        )
        np.testing.assert_array_equal(np.asarray(output[:, 0]), 0.0)
        assert np.all(np.isfinite(np.asarray(output)))
        if cross_attention:
            grads = jax.grad(
                lambda q, c: jnp.sum(model.apply(variables, q, c, c, mask=mask)),
                (0, 1),
            )(query, context)
        else:
            grads = (jax.grad(
                lambda q: jnp.sum(model.apply(variables, q, mask=mask))
            )(query),)
        assert all(np.all(np.isfinite(np.asarray(grad))) for grad in grads)

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
