"""Tests for fused YAT L1 attention (memory-efficient custom_vjp).

Verifies forward/backward correctness by comparing against a reference
standard-autodiff implementation.
"""

import pytest

try:
    import jax
    import jax.numpy as jnp
    import math
    from nmn.nnx.layers.attention.fused_yat_attention import (
        fused_yat_l1_attention,
        fused_yat_l1_self_attention,
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX/Flax not available")


# ── Reference implementation (standard autodiff, no custom_vjp) ──

def _ref_yat_l1(q, k, v, bias=None, epsilon=1e-5, mask=None, scale=None):
    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    dot = jnp.einsum("...qhd,...khd->...hqk", q_f32, k_f32)
    q_sq = jnp.sum(q_f32 ** 2, axis=-1, keepdims=True)
    k_sq = jnp.sum(k_f32 ** 2, axis=-1, keepdims=True)
    bd = q.ndim - 3
    perm = tuple(range(bd)) + (bd + 1, bd, bd + 2)
    q_sq_t = q_sq.transpose(perm)
    k_sq_t = k_sq.transpose(perm)
    k_sq_t = jnp.swapaxes(k_sq_t, -2, -1)
    eps_val = epsilon.astype(jnp.float32) if isinstance(epsilon, jnp.ndarray) else epsilon
    dist = jnp.maximum(q_sq_t + k_sq_t - 2 * dot, 0.0) + eps_val
    if scale is None:
        scale = float(jnp.sqrt(jnp.float32(q.shape[-1])))
    num = (dot + bias.astype(jnp.float32)) if bias is not None else dot
    scores = num ** 2 / (dist * scale)
    if mask is not None:
        scores = jnp.where(mask, scores, 0.0)
    L = jnp.sum(scores, axis=-1, keepdims=True)
    W = (scores / (L + 1e-12)).astype(q.dtype)
    return jnp.einsum("...hqk,...khd->...qhd", W, v)


def _rand(shape, key=0):
    return jax.random.normal(jax.random.PRNGKey(key), shape)


def _compare_grads(loss_fused, loss_ref, args, names, atol=1e-6):
    g_fused = jax.grad(loss_fused, argnums=tuple(range(len(args))))(*args)
    g_ref = jax.grad(loss_ref, argnums=tuple(range(len(args))))(*args)
    for name, gf, gr in zip(names, g_fused, g_ref):
        err = float(jnp.abs(gf - gr).max())
        assert err < atol, f"{name} grad max err {err:.2e} > {atol}"


# ── Forward tests ──

class TestFusedAttnForward:

    def test_basic(self):
        q, k, v = _rand((2, 16, 4, 32)), _rand((2, 16, 4, 32), 1), _rand((2, 16, 4, 32), 2)
        out_f = fused_yat_l1_attention(q, k, v)
        out_r = _ref_yat_l1(q, k, v)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_with_bias(self):
        q, k, v = _rand((2, 8, 4, 16)), _rand((2, 8, 4, 16), 1), _rand((2, 8, 4, 16), 2)
        bias = _rand((4, 1, 1), 3) * 0.1
        out_f = fused_yat_l1_attention(q, k, v, bias=bias)
        out_r = _ref_yat_l1(q, k, v, bias=bias)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_with_learnable_eps(self):
        q, k, v = _rand((2, 8, 4, 16)), _rand((2, 8, 4, 16), 1), _rand((2, 8, 4, 16), 2)
        eps = jnp.array(1e-3)
        out_f = fused_yat_l1_attention(q, k, v, epsilon=eps)
        out_r = _ref_yat_l1(q, k, v, epsilon=eps)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_with_mask(self):
        q, k, v = _rand((2, 8, 4, 16)), _rand((2, 8, 4, 16), 1), _rand((2, 8, 4, 16), 2)
        mask = jnp.tril(jnp.ones((8, 8), dtype=jnp.bool_))
        out_f = fused_yat_l1_attention(q, k, v, mask=mask)
        out_r = _ref_yat_l1(q, k, v, mask=mask)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_full_combo(self):
        q, k, v = _rand((2, 8, 4, 16)), _rand((2, 8, 4, 16), 1), _rand((2, 8, 4, 16), 2)
        bias = _rand((4, 1, 1), 3) * 0.1
        eps = jnp.array(1e-3)
        mask = jnp.tril(jnp.ones((8, 8), dtype=jnp.bool_))
        out_f = fused_yat_l1_attention(q, k, v, bias=bias, epsilon=eps, mask=mask)
        out_r = _ref_yat_l1(q, k, v, bias=bias, epsilon=eps, mask=mask)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_2d_batch(self):
        q, k, v = _rand((4, 16, 8, 32)), _rand((4, 16, 8, 32), 1), _rand((4, 16, 8, 32), 2)
        out_f = fused_yat_l1_attention(q, k, v)
        out_r = _ref_yat_l1(q, k, v)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_cross_attention(self):
        q = _rand((2, 8, 4, 16))
        k = _rand((2, 32, 4, 16), 1)
        v = _rand((2, 32, 4, 16), 2)
        out_f = fused_yat_l1_attention(q, k, v)
        out_r = _ref_yat_l1(q, k, v)
        assert jnp.allclose(out_f, out_r, atol=1e-5)


# ── Backward tests ──

class TestFusedAttnBackward:

    def _make_args(self, bias=False, eps_learnable=False, mask=False):
        q = _rand((2, 8, 4, 16))
        k = _rand((2, 8, 4, 16), 1)
        v = _rand((2, 8, 4, 16), 2)
        b = _rand((4, 1, 1), 3) * 0.1 if bias else None
        e = jnp.array(1e-3) if eps_learnable else 1e-3
        m = jnp.tril(jnp.ones((8, 8), dtype=jnp.bool_)) if mask else None
        return q, k, v, b, e, m

    def _test_grads(self, bias=False, eps_learnable=False, mask=False, atol=1e-6):
        q, k, v, b, e, m = self._make_args(bias, eps_learnable, mask)

        args = [q, k, v]
        names = ["dQ", "dK", "dV"]
        if b is not None:
            args.append(b)
            names.append("dBias")
        if eps_learnable:
            args.append(e)
            names.append("dEps")

        def loss_f(*a):
            qi, ki, vi = a[0], a[1], a[2]
            kw = {}
            idx = 3
            if b is not None:
                kw["bias"] = a[idx]; idx += 1
            if eps_learnable:
                kw["epsilon"] = a[idx]; idx += 1
            else:
                kw["epsilon"] = e
            if m is not None:
                kw["mask"] = m
            return jnp.mean(fused_yat_l1_attention(qi, ki, vi, **kw) ** 2)

        def loss_r(*a):
            qi, ki, vi = a[0], a[1], a[2]
            kw = {}
            idx = 3
            if b is not None:
                kw["bias"] = a[idx]; idx += 1
            if eps_learnable:
                kw["epsilon"] = a[idx]; idx += 1
            else:
                kw["epsilon"] = e
            if m is not None:
                kw["mask"] = m
            return jnp.mean(_ref_yat_l1(qi, ki, vi, **kw) ** 2)

        _compare_grads(loss_f, loss_r, args, names, atol=atol)

    def test_basic(self):
        self._test_grads()

    def test_with_bias(self):
        self._test_grads(bias=True)

    def test_with_learnable_eps(self):
        self._test_grads(eps_learnable=True)

    def test_with_mask(self):
        self._test_grads(mask=True)

    def test_full_combo(self):
        self._test_grads(bias=True, eps_learnable=True, mask=True, atol=5e-6)

    def test_grads_finite(self):
        q, k, v = _rand((2, 8, 4, 16)), _rand((2, 8, 4, 16), 1), _rand((2, 8, 4, 16), 2)
        bias = _rand((4, 1, 1), 3) * 0.1
        eps = jnp.array(1e-3)
        def loss(q, k, v, b, e):
            return jnp.mean(fused_yat_l1_attention(q, k, v, bias=b, epsilon=e) ** 2)
        grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, bias, eps)
        for g in grads:
            assert jnp.isfinite(g).all()

    def test_cross_attention_grads(self):
        q = _rand((2, 8, 4, 16))
        k = _rand((2, 32, 4, 16), 1)
        v = _rand((2, 32, 4, 16), 2)
        def loss_f(q, k, v):
            return jnp.mean(fused_yat_l1_attention(q, k, v) ** 2)
        def loss_r(q, k, v):
            return jnp.mean(_ref_yat_l1(q, k, v) ** 2)
        _compare_grads(loss_f, loss_r, [q, k, v], ["dQ", "dK", "dV"])


class TestFusedAttnJIT:

    def test_jit_forward(self):
        q, k, v = _rand((2, 16, 4, 32)), _rand((2, 16, 4, 32), 1), _rand((2, 16, 4, 32), 2)

        @jax.jit
        def f(q, k, v):
            return fused_yat_l1_attention(q, k, v)

        out = f(q, k, v)
        assert out.shape == (2, 16, 4, 32)
        assert jnp.isfinite(out).all()

    def test_jit_train_step(self):
        q, k, v = _rand((2, 16, 4, 32)), _rand((2, 16, 4, 32), 1), _rand((2, 16, 4, 32), 2)
        bias = _rand((4, 1, 1), 3) * 0.1
        eps = jnp.array(1e-3)

        @jax.jit
        def train_step(q, k, v, b, e):
            def loss(q, k, v, b, e):
                return jnp.mean(fused_yat_l1_attention(q, k, v, bias=b, epsilon=e) ** 2)
            return jax.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, b, e)

        grads = train_step(q, k, v, bias, eps)
        for g in grads:
            assert jnp.isfinite(g).all()


# ── Self-attention (Q = K = x, no-self-yat) tests ──

def _ref_self_yat(x, v, bias=None, epsilon=1e-5, scale=None):
    """Reference self-YAT with diagonal zeroed before weights, but diagonal
    included in the L1 normalizer."""
    x_f32 = x.astype(jnp.float32)
    dot = jnp.einsum("...qhd,...khd->...hqk", x_f32, x_f32)
    x_sq = jnp.sum(x_f32 ** 2, axis=-1, keepdims=True)
    bd = x.ndim - 3
    perm = tuple(range(bd)) + (bd + 1, bd, bd + 2)
    x_sq_t = x_sq.transpose(perm)
    x_sq_k = jnp.swapaxes(x_sq_t, -2, -1)
    eps_val = epsilon.astype(jnp.float32) if isinstance(epsilon, jnp.ndarray) else epsilon
    dist = jnp.maximum(x_sq_t + x_sq_k - 2 * dot, 0.0) + eps_val
    if scale is None:
        scale = math.sqrt(float(x.shape[-1]))
    num = (dot + bias.astype(jnp.float32)) if bias is not None else dot
    scores = num ** 2 / (dist * scale)
    L = jnp.sum(scores, axis=-1, keepdims=True)
    seq = scores.shape[-1]
    eye = jnp.eye(seq, dtype=jnp.bool_)
    scores_off = jnp.where(eye, 0.0, scores)
    W = (scores_off / (L + 1e-12)).astype(x.dtype)
    return jnp.einsum("...hqk,...khd->...qhd", W, v)


class TestSelfYatForward:

    def test_basic(self):
        x = _rand((2, 8, 4, 16))
        v = _rand((2, 8, 4, 16), 1)
        out_f = fused_yat_l1_self_attention(x, v)
        out_r = _ref_self_yat(x, v)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_with_bias(self):
        x = _rand((2, 8, 4, 16))
        v = _rand((2, 8, 4, 16), 1)
        bias = _rand((4, 1, 1), 2) * 0.1
        out_f = fused_yat_l1_self_attention(x, v, bias=bias)
        out_r = _ref_self_yat(x, v, bias=bias)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_with_learnable_eps(self):
        x = _rand((2, 8, 4, 16))
        v = _rand((2, 8, 4, 16), 1)
        eps = jnp.array(1e-3)
        out_f = fused_yat_l1_self_attention(x, v, epsilon=eps)
        out_r = _ref_self_yat(x, v, epsilon=eps)
        assert jnp.allclose(out_f, out_r, atol=1e-5)

    def test_diagonal_is_zero_in_weights(self):
        """Position i must not attend to its own v[i] (diagonal zeroed).

        Use a larger epsilon so the diagonal self-score doesn't swamp the L1
        denominator, leaving meaningful off-diagonal weights.
        """
        x = _rand((2, 8, 4, 16))
        v = _rand((2, 8, 4, 16), 1)
        eps = jnp.array(10.0)  # large eps so off-diag weights are non-trivial

        out_full = fused_yat_l1_self_attention(x, v, epsilon=eps)

        # Replace v at position p with a very different value; if the diagonal
        # were not zeroed, out_full[:, p] would depend on v[:, p] and so would
        # change. Since the diagonal IS zeroed, out[:, p] must be invariant.
        p = 3
        v_mod = v.at[:, p, :, :].set(v[:, p, :, :] * 100.0 + 50.0)
        out_mod = fused_yat_l1_self_attention(x, v_mod, epsilon=eps)

        assert jnp.allclose(out_full[:, p, :, :], out_mod[:, p, :, :], atol=1e-4), \
            "Position p output changed when v[p] changed — diagonal is NOT zeroed."
        # Sanity: other positions must have changed (they attended to v[p])
        diff_other = float(jnp.abs(out_full[:, 0, :, :] - out_mod[:, 0, :, :]).max())
        assert diff_other > 1e-3, \
            f"Other positions didn't respond to v[p] change (diff={diff_other:.2e})"


class TestSelfYatBackward:

    def test_grads_basic(self):
        x = _rand((2, 8, 4, 16))
        v = _rand((2, 8, 4, 16), 1)

        def loss_f(x, v):
            return jnp.mean(fused_yat_l1_self_attention(x, v) ** 2)
        def loss_r(x, v):
            return jnp.mean(_ref_self_yat(x, v) ** 2)

        gf = jax.grad(loss_f, argnums=(0, 1))(x, v)
        gr = jax.grad(loss_r, argnums=(0, 1))(x, v)
        for name, a, b in zip(["dX", "dV"], gf, gr):
            err = float(jnp.abs(a - b).max())
            assert err < 1e-5, f"{name} err {err:.2e}"

    def test_grads_full_combo(self):
        x = _rand((2, 8, 4, 16))
        v = _rand((2, 8, 4, 16), 1)
        bias = _rand((4, 1, 1), 2) * 0.1
        eps = jnp.array(1e-3)

        def loss_f(x, v, b, e):
            return jnp.mean(fused_yat_l1_self_attention(x, v, bias=b, epsilon=e) ** 2)
        def loss_r(x, v, b, e):
            return jnp.mean(_ref_self_yat(x, v, bias=b, epsilon=e) ** 2)

        gf = jax.grad(loss_f, argnums=(0, 1, 2, 3))(x, v, bias, eps)
        gr = jax.grad(loss_r, argnums=(0, 1, 2, 3))(x, v, bias, eps)
        for name, a, b in zip(["dX", "dV", "dBias", "dEps"], gf, gr):
            err = float(jnp.abs(a - b).max())
            assert err < 5e-6, f"{name} err {err:.2e}"

    def test_grads_finite(self):
        x = _rand((2, 8, 4, 16))
        v = _rand((2, 8, 4, 16), 1)
        def loss(x, v):
            return jnp.mean(fused_yat_l1_self_attention(x, v) ** 2)
        gx, gv = jax.grad(loss, argnums=(0, 1))(x, v)
        assert jnp.isfinite(gx).all() and jnp.isfinite(gv).all()


class TestSelfYatJIT:

    def test_jit_train_step(self):
        x = _rand((2, 16, 4, 32))
        v = _rand((2, 16, 4, 32), 1)
        bias = _rand((4, 1, 1), 2) * 0.1
        eps = jnp.array(1e-3)

        @jax.jit
        def train_step(x, v, b, e):
            def loss(x, v, b, e):
                return jnp.mean(fused_yat_l1_self_attention(x, v, bias=b, epsilon=e) ** 2)
            return jax.grad(loss, argnums=(0, 1, 2, 3))(x, v, b, e)

        grads = train_step(x, v, bias, eps)
        for g in grads:
            assert jnp.isfinite(g).all()
