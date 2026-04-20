"""Tests for Pallas-fused YAT L1 attention.

Compares the tiled Pallas kernel (interpret=True for CPU) against
the custom_vjp reference implementation.
"""

import pytest

try:
    import jax
    import jax.numpy as jnp
    from nmn.nnx.layers.attention.pallas_yat_attention import pallas_yat_l1_attention
    from nmn.nnx.layers.attention.fused_yat_attention import fused_yat_l1_attention
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX/Flax not available")

BQ, BK = 8, 8


def _rand(shape, key=0):
    return jax.random.normal(jax.random.PRNGKey(key), shape)


class TestPallasForward:

    def test_basic(self):
        q, k, v = _rand((2, 16, 4, 16)), _rand((2, 16, 4, 16), 1), _rand((2, 16, 4, 16), 2)
        out_p = pallas_yat_l1_attention(q, k, v, block_q=BQ, block_k=BK, interpret=True)
        out_r = fused_yat_l1_attention(q, k, v)
        assert jnp.allclose(out_p, out_r, atol=1e-5)

    def test_causal(self):
        q, k, v = _rand((2, 16, 4, 16)), _rand((2, 16, 4, 16), 1), _rand((2, 16, 4, 16), 2)
        mask = jnp.tril(jnp.ones((16, 16), dtype=jnp.bool_))
        out_p = pallas_yat_l1_attention(q, k, v, causal=True, block_q=BQ, block_k=BK, interpret=True)
        out_r = fused_yat_l1_attention(q, k, v, mask=mask)
        assert jnp.allclose(out_p, out_r, atol=1e-5)

    def test_no_batch(self):
        q, k, v = _rand((16, 4, 16)), _rand((16, 4, 16), 1), _rand((16, 4, 16), 2)
        out = pallas_yat_l1_attention(q, k, v, block_q=BQ, block_k=BK, interpret=True)
        assert out.shape == (16, 4, 16)
        assert jnp.isfinite(out).all()

    def test_different_eps(self):
        q, k, v = _rand((2, 16, 4, 16)), _rand((2, 16, 4, 16), 1), _rand((2, 16, 4, 16), 2)
        for eps in [1e-3, 1e-5, 0.1]:
            out_p = pallas_yat_l1_attention(q, k, v, epsilon=eps, block_q=BQ, block_k=BK, interpret=True)
            out_r = fused_yat_l1_attention(q, k, v, epsilon=eps)
            assert jnp.allclose(out_p, out_r, atol=1e-5), f"eps={eps}"


class TestPallasBackward:

    def _compare_grads(self, causal=False, atol=5e-3):
        q, k, v = _rand((2, 16, 4, 16)), _rand((2, 16, 4, 16), 1), _rand((2, 16, 4, 16), 2)
        mask = jnp.tril(jnp.ones((16, 16), dtype=jnp.bool_)) if causal else None

        def loss_p(q, k, v):
            return jnp.mean(pallas_yat_l1_attention(
                q, k, v, causal=causal, block_q=BQ, block_k=BK, interpret=True) ** 2)
        def loss_r(q, k, v):
            return jnp.mean(fused_yat_l1_attention(q, k, v, mask=mask) ** 2)

        gp = jax.grad(loss_p, argnums=(0, 1, 2))(q, k, v)
        gr = jax.grad(loss_r, argnums=(0, 1, 2))(q, k, v)

        for name, a, b in zip(["dQ", "dK", "dV"], gp, gr):
            err = float(jnp.abs(a - b).max())
            assert err < atol, f"{name} err {err:.2e} > {atol}"

    def test_basic_grads(self):
        self._compare_grads()

    def test_causal_grads(self):
        self._compare_grads(causal=True)

    def test_grads_finite(self):
        q, k, v = _rand((2, 16, 4, 16)), _rand((2, 16, 4, 16), 1), _rand((2, 16, 4, 16), 2)
        def loss(q, k, v):
            return jnp.mean(pallas_yat_l1_attention(
                q, k, v, block_q=BQ, block_k=BK, interpret=True) ** 2)
        grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        for g in grads:
            assert jnp.isfinite(g).all()
