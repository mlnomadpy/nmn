"""Tests for Linen YatEmbed."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
jnp = jax.numpy

from nmn.linen.embed import YatEmbed  # noqa: E402


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


def _linen_attend_value_and_grads(dtype, spherical, compiled, loss_scale=1.0):
    layer = YatEmbed(
        num_embeddings=2,
        features=2,
        epsilon=1.0,
        spherical=spherical,
        dtype=dtype,
        param_dtype=dtype,
    )
    query = jnp.array([[100.0, 100.0]], dtype=dtype)
    variables = layer.init(jax.random.key(0), query, method=layer.attend)
    params = dict(
        variables["params"],
        embedding=jnp.array([[-100.0, -99.0], [100.0, -99.0]], dtype=dtype),
        alpha=jnp.array([1.25], dtype=dtype),
    )

    def evaluate(query_value, parameter_values):
        output = layer.apply(
            {"params": parameter_values}, query_value, method=layer.attend
        )
        return output.astype(jnp.float32).sum() * loss_scale, output

    value_and_grad = jax.value_and_grad(evaluate, argnums=(0, 1), has_aux=True)
    if compiled:
        value_and_grad = jax.jit(value_and_grad)
    (_, output), gradients = value_and_grad(query, params)
    return output, gradients


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_low_precision_attend_matches_fp32_output_and_gradients(
    dtype, spherical, compiled
):
    expected_output, expected_grads = _linen_attend_value_and_grads(
        jnp.float32, spherical, compiled
    )
    output, grads = _linen_attend_value_and_grads(dtype, spherical, compiled)

    assert output.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(output, dtype=np.float32),
        np.asarray(expected_output, dtype=np.float32),
        rtol=1.5e-2,
        atol=2e-2,
    )
    for actual, expected in zip(
        jax.tree.leaves(grads), jax.tree.leaves(expected_grads)
    ):
        assert jnp.isfinite(actual).all()
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )


@pytest.mark.parametrize("spherical", [False, True])
def test_fp16_attend_saturates_output_and_gradients_and_preserves_nan(spherical):
    layer = YatEmbed(
        num_embeddings=2,
        features=2,
        spherical=spherical,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
    )
    query = jnp.array([[300.0, 300.0]], dtype=jnp.float16)
    variables = layer.init(jax.random.key(1), query, method=layer.attend)
    params = dict(
        variables["params"],
        embedding=jnp.array([[300.0, 300.0], [300.0, -300.0]], jnp.float16),
    )

    def loss(query_value, parameter_values):
        return (
            layer.apply({"params": parameter_values}, query_value, method=layer.attend)
            .astype(jnp.float32)
            .sum()
        )

    output = layer.apply({"params": params}, query, method=layer.attend)
    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(query, params)
    assert output[0, 0] == jnp.finfo(jnp.float16).max
    assert all(jnp.isfinite(value).all() for value in jax.tree.leaves(gradients))

    nan_query = query.at[0, 0].set(jnp.nan)
    nan_output = layer.apply({"params": params}, nan_query, method=layer.attend)
    assert jnp.isnan(nan_output).all()


def test_fp16_attend_returning_gradients_saturate_against_fp32():
    spherical, loss_scale = False, 1e4
    _, expected_grads = _linen_attend_value_and_grads(
        jnp.float32, spherical, True, loss_scale
    )
    _, grads = _linen_attend_value_and_grads(jnp.float16, spherical, True, loss_scale)
    limits = jnp.finfo(jnp.float16)
    for actual, expected in zip(
        jax.tree.leaves(grads), jax.tree.leaves(expected_grads)
    ):
        clipped = jnp.clip(expected, limits.min, limits.max).astype(jnp.float16)
        assert jnp.isfinite(actual).all()
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(clipped, dtype=np.float32),
            rtol=2e-2,
            atol=32.0,
        )
