"""Tests for fused YatNMN custom_vjp implementation.

Verifies that YatNMN(fused=True) produces identical forward outputs
and numerically close gradients compared to YatNMN(fused=False),
across all configurations: alpha variants, bias modes, learnable epsilon.
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax._src.test_util import check_grads as _check_grads
    from flax import nnx
    from nmn.nnx.layers import YatNMN
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX/Flax not available")


# ── Helpers ──

def _make_pair(in_f, out_f, *, use_alpha=True, constant_alpha=None,
               use_bias=False, constant_bias=None, softplus_bias=False,
               scalar_bias=False, learnable_epsilon=False, epsilon=0.001):
    """Create matching fused and unfused YatNMN layers with shared weights."""
    common = dict(
        use_alpha=use_alpha, constant_alpha=constant_alpha,
        use_bias=use_bias, constant_bias=constant_bias,
        softplus_bias=softplus_bias, scalar_bias=scalar_bias,
        learnable_epsilon=learnable_epsilon, epsilon=epsilon,
    )
    ref = YatNMN(in_f, out_f, fused=False, rngs=nnx.Rngs(42), **common)
    fused = YatNMN(in_f, out_f, fused=True, rngs=nnx.Rngs(42), **common)

    for name in ('kernel', 'bias', 'alpha', 'epsilon_param'):
        r = getattr(ref, name, None)
        f = getattr(fused, name, None)
        if r is not None and f is not None:
            setattr(fused, name, nnx.Param(r[...].copy()))

    return ref, fused


def _compare_forward(ref, fused, x, atol=1e-5):
    """Assert forward outputs match."""
    out_ref = ref(x)
    out_fused = fused(x)
    max_err = float(jnp.abs(out_ref - out_fused).max())
    assert out_ref.shape == out_fused.shape, f"Shape mismatch: {out_ref.shape} vs {out_fused.shape}"
    assert max_err < atol, f"Forward max abs error {max_err:.2e} > {atol}"
    return out_ref, out_fused


def _compare_grads(ref, fused, x, atol=1e-3, rtol=1e-3):
    """Assert gradients match for all learnable parameters."""
    def loss_fn(model, x):
        return jnp.mean(model(x) ** 2)

    loss_ref, grad_ref = jax.value_and_grad(loss_fn)(ref, x)
    loss_fused, grad_fused = jax.value_and_grad(loss_fn)(fused, x)

    for name in ('kernel', 'bias', 'alpha', 'epsilon_param'):
        gr = getattr(grad_ref, name, None)
        gf = getattr(grad_fused, name, None)
        if gr is not None and gf is not None:
            assert jnp.allclose(gr[...], gf[...], atol=atol, rtol=rtol), \
                f"{name} grad max err {float(jnp.abs(gr[...] - gf[...]).max()):.2e}"

    return loss_ref, loss_fused


def _rand(shape, key=0):
    return jax.random.normal(jax.random.PRNGKey(key), shape)


# ── Forward tests ──

class TestFusedForward:

    def test_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=False)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_constant_alpha_sqrt2(self):
        ref, fused = _make_pair(64, 128, constant_alpha=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_constant_alpha_custom(self):
        ref, fused = _make_pair(64, 128, constant_alpha=1.5)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_2d_input(self):
        ref, fused = _make_pair(32, 64, use_alpha=True)
        _compare_forward(ref, fused, _rand((8, 32)))

    def test_3d_input(self):
        ref, fused = _make_pair(768, 3072, use_alpha=True)
        _compare_forward(ref, fused, _rand((2, 128, 768)))

    def test_different_epsilons(self):
        for eps in [1e-3, 1e-5, 0.1]:
            ref, fused = _make_pair(64, 128, epsilon=eps)
            _compare_forward(ref, fused, _rand((4, 16, 64)))


class TestFusedForwardBias:

    def test_bias_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, use_alpha=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_bias_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, use_alpha=False)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_bias_constant_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, constant_alpha=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_constant_bias(self):
        ref, fused = _make_pair(64, 128, constant_bias=0.5)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_softplus_bias(self):
        ref, fused = _make_pair(64, 128, use_bias=True, softplus_bias=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_scalar_bias(self):
        ref, fused = _make_pair(64, 128, scalar_bias=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_scalar_softplus_bias(self):
        ref, fused = _make_pair(64, 128, scalar_bias=True, softplus_bias=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))


class TestFusedForwardLearnableEps:

    def test_learnable_eps_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=False, learnable_epsilon=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_learnable_eps_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=True, learnable_epsilon=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_learnable_eps_constant_alpha(self):
        ref, fused = _make_pair(64, 128, constant_alpha=True, learnable_epsilon=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))


class TestFusedForwardCombined:

    def test_bias_and_learnable_eps_and_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, learnable_epsilon=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_bias_and_learnable_eps_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, use_alpha=False, learnable_epsilon=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))

    def test_scalar_softplus_bias_and_learnable_eps(self):
        ref, fused = _make_pair(64, 128, scalar_bias=True, softplus_bias=True, learnable_epsilon=True)
        _compare_forward(ref, fused, _rand((4, 16, 64)))


# ── Backward tests ──

class TestFusedBackward:

    def test_grad_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=False)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_constant_alpha(self):
        ref, fused = _make_pair(64, 128, constant_alpha=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_large_scale(self):
        ref, fused = _make_pair(768, 3072, use_alpha=True)
        _compare_grads(ref, fused, _rand((2, 64, 768)), atol=1e-2, rtol=1e-2)

    def test_grad_finite(self):
        _, fused = _make_pair(64, 128, use_alpha=True)
        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)
        _, grads = jax.value_and_grad(loss_fn)(fused, _rand((4, 16, 64)))
        for leaf in jax.tree_util.tree_leaves(grads):
            if hasattr(leaf, 'shape'):
                assert jnp.isfinite(leaf).all(), "Non-finite gradient found"


class TestFusedBackwardBias:

    def test_grad_bias_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, use_alpha=False)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_bias_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, use_alpha=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_bias_constant_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, constant_alpha=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_softplus_bias(self):
        ref, fused = _make_pair(64, 128, use_bias=True, softplus_bias=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_scalar_bias(self):
        ref, fused = _make_pair(64, 128, scalar_bias=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)), atol=2e-3)

    def test_grad_scalar_softplus_bias(self):
        ref, fused = _make_pair(64, 128, scalar_bias=True, softplus_bias=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)), atol=2e-3)


class TestFusedBackwardLearnableEps:

    def test_grad_learnable_eps_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=False, learnable_epsilon=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_learnable_eps_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=True, learnable_epsilon=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_learnable_eps_constant_alpha(self):
        ref, fused = _make_pair(64, 128, constant_alpha=True, learnable_epsilon=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))


class TestFusedBackwardCombined:

    def test_grad_all_learnable(self):
        ref, fused = _make_pair(64, 128, use_bias=True, learnable_epsilon=True, use_alpha=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_bias_and_learnable_eps_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_bias=True, use_alpha=False, learnable_epsilon=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)))

    def test_grad_scalar_softplus_bias_and_learnable_eps(self):
        ref, fused = _make_pair(64, 128, scalar_bias=True, softplus_bias=True, learnable_epsilon=True)
        _compare_grads(ref, fused, _rand((4, 16, 64)), atol=2e-3)

    def test_grad_finite_all_learnable(self):
        _, fused = _make_pair(64, 128, use_bias=True, learnable_epsilon=True, use_alpha=True)
        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)
        _, grads = jax.value_and_grad(loss_fn)(fused, _rand((4, 16, 64)))
        for leaf in jax.tree_util.tree_leaves(grads):
            if hasattr(leaf, 'shape'):
                assert jnp.isfinite(leaf).all(), "Non-finite gradient found"


class TestFusedFallback:

    def test_spherical_fallback(self):
        layer = YatNMN(64, 128, spherical=True, fused=True, rngs=nnx.Rngs(0))
        x = _rand((4, 16, 64))
        out = layer(x)
        assert out.shape == (4, 16, 128)
        assert jnp.isfinite(out).all()


class TestFusedJIT:

    def test_jit_forward(self):
        _, fused = _make_pair(64, 128)
        x = _rand((4, 16, 64))

        @jax.jit
        def forward(model, x):
            return model(x)

        out = forward(fused, x)
        assert out.shape == (4, 16, 128)
        assert jnp.isfinite(out).all()

    def test_jit_train_step(self):
        _, fused = _make_pair(64, 128)
        x = _rand((4, 16, 64))

        @jax.jit
        def train_step(model, x):
            def loss_fn(m):
                return jnp.mean(m(x) ** 2)
            loss, grads = jax.value_and_grad(loss_fn)(model)
            return loss

        loss = train_step(fused, x)
        assert jnp.isfinite(loss)

    def test_jit_train_step_bias_and_eps(self):
        _, fused = _make_pair(64, 128, use_bias=True, learnable_epsilon=True)
        x = _rand((4, 16, 64))

        @jax.jit
        def train_step(model, x):
            def loss_fn(m):
                return jnp.mean(m(x) ** 2)
            loss, grads = jax.value_and_grad(loss_fn)(model)
            return loss

        loss = train_step(fused, x)
        assert jnp.isfinite(loss)


class TestFusedGradFlow:
    """Verify that gradients are non-zero and flow to all learnable parameters."""

    def _check_nonzero_grads(self, **kwargs):
        ref, fused = _make_pair(32, 64, **kwargs)
        x = _rand((4, 8, 32))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        _, grads = jax.value_and_grad(loss_fn)(fused, x)
        for name in ('kernel', 'bias', 'alpha', 'epsilon_param'):
            g = getattr(grads, name, None)
            if g is not None:
                val = g[...]
                assert jnp.isfinite(val).all(), f"{name} has non-finite grad"
                assert float(jnp.abs(val).sum()) > 0, f"{name} grad is all zeros"

    def test_grad_flow_bias_only(self):
        self._check_nonzero_grads(use_bias=True)

    def test_grad_flow_learnable_eps_only(self):
        self._check_nonzero_grads(learnable_epsilon=True)

    def test_grad_flow_all_learnable(self):
        self._check_nonzero_grads(use_bias=True, learnable_epsilon=True)

    def test_grad_flow_scalar_softplus_bias(self):
        self._check_nonzero_grads(scalar_bias=True, softplus_bias=True, learnable_epsilon=True)
