"""Numerical parity tests for Pallas-fused YAT L1 attention."""

import math

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from nmn.nnx.layers.attention.pallas_yat_attention import pallas_yat_l1_attention
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX/Flax not available")


def _rand(shape, key=0):
    return jax.random.normal(jax.random.key(key), shape)


def _reference(q, k, v, *, epsilon=1e-5, causal=False):
    """Plain-JAX oracle, deliberately independent of either fused custom VJP."""
    q32, k32, v32 = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    dot = jnp.einsum("...qhd,...khd->...hqk", q32, k32)
    q_sq = jnp.einsum("...qhd,...qhd->...hq", q32, q32)[..., :, :, None]
    k_sq = jnp.einsum("...khd,...khd->...hk", k32, k32)[..., :, None, :]
    dist = jnp.maximum(q_sq + k_sq - 2.0 * dot, 0.0) + epsilon
    scores = jnp.square(dot) / (dist * math.sqrt(q.shape[-1]))
    if causal:
        q_pos = jnp.arange(q.shape[-3])[:, None]
        k_pos = jnp.arange(k.shape[-3])[None, :]
        scores = jnp.where(q_pos >= k_pos, scores, 0.0)
    weights = scores / jnp.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)
    return jnp.einsum("...hqk,...khd->...qhd", weights, v32).astype(v.dtype)


def _cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


CASES = [
    pytest.param(2, 2, 4, 4, False, 4, 4, id="tiny-single-tile"),
    pytest.param(7, 11, 4, 3, False, 4, 4, id="ragged-multitile-v-smaller"),
    pytest.param(7, 11, 4, 6, False, 4, 4, id="ragged-multitile-v-larger"),
    pytest.param(5, 9, 4, 3, True, 4, 4, id="causal-q-shorter-than-kv"),
    pytest.param(9, 5, 4, 6, True, 4, 4, id="causal-q-longer-than-kv"),
    pytest.param(8, 13, 4, 5, True, 3, 5, id="unequal-ragged-tiles"),
]


@pytest.mark.parametrize("q_len,kv_len,head_dim,v_dim,causal,block_q,block_k", CASES)
def test_forward_matches_plain_jax(
    q_len, kv_len, head_dim, v_dim, causal, block_q, block_k,
):
    q = _rand((1, q_len, 2, head_dim), 0)
    k = _rand((1, kv_len, 2, head_dim), 1)
    v = _rand((1, kv_len, 2, v_dim), 2)
    actual = pallas_yat_l1_attention(
        q, k, v, causal=causal, block_q=block_q, block_k=block_k, interpret=True)
    expected = _reference(q, k, v, causal=causal)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert actual.shape == (1, q_len, 2, v_dim)


@pytest.mark.parametrize("q_len,kv_len,head_dim,v_dim,causal,block_q,block_k", CASES)
def test_gradients_match_plain_jax(
    q_len, kv_len, head_dim, v_dim, causal, block_q, block_k,
):
    q = _rand((1, q_len, 2, head_dim), 3)
    k = _rand((1, kv_len, 2, head_dim), 4)
    v = _rand((1, kv_len, 2, v_dim), 5)
    cotangent = _rand((1, q_len, 2, v_dim), 6)

    def pallas_loss(q, k, v):
        return jnp.vdot(
            pallas_yat_l1_attention(
                q, k, v, causal=causal, block_q=block_q, block_k=block_k,
                interpret=True),
            cotangent,
        )

    def reference_loss(q, k, v):
        return jnp.vdot(_reference(q, k, v, causal=causal), cotangent)

    actual = jax.grad(pallas_loss, argnums=(0, 1, 2))(q, k, v)
    expected = jax.grad(reference_loss, argnums=(0, 1, 2))(q, k, v)
    for name, got, want in zip(("dQ", "dK", "dV"), actual, expected):
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4, err_msg=name)
        assert _cosine(got, want) > 0.99999, name


def test_no_batch_and_multiple_batch_dimensions():
    q = _rand((7, 2, 4), 7)
    k = _rand((11, 2, 4), 8)
    v = _rand((11, 2, 3), 9)
    out = pallas_yat_l1_attention(q, k, v, block_q=4, block_k=4, interpret=True)
    np.testing.assert_allclose(out, _reference(q, k, v), rtol=2e-5, atol=2e-5)

    q2 = jnp.broadcast_to(q, (2, 3) + q.shape)
    k2 = jnp.broadcast_to(k, (2, 3) + k.shape)
    v2 = jnp.broadcast_to(v, (2, 3) + v.shape)
    out2 = pallas_yat_l1_attention(q2, k2, v2, block_q=4, block_k=4, interpret=True)
    np.testing.assert_allclose(out2, _reference(q2, k2, v2), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "q_shape,k_shape,v_shape,message",
    [
        ((1, 4, 2, 3), (1, 4, 2, 4), (1, 4, 2, 5), "head_dim"),
        ((1, 4, 2, 3), (1, 5, 2, 3), (1, 4, 2, 5), "sequence"),
        ((1, 4, 2, 3), (1, 4, 3, 3), (1, 4, 2, 5), "heads"),
    ],
)
def test_shape_validation(q_shape, k_shape, v_shape, message):
    with pytest.raises(ValueError, match=message):
        pallas_yat_l1_attention(
            jnp.ones(q_shape), jnp.ones(k_shape), jnp.ones(v_shape), interpret=True)
