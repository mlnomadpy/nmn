"""Tests for fused YatNMN custom_vjp implementation.

Verifies that YatNMN(fused=True) produces identical forward outputs
and numerically close gradients compared to YatNMN(fused=False),
across all alpha configurations.
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from nmn.nnx.layers import YatNMN
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX/Flax not available")


# ── Helpers ──

def _make_pair(in_f, out_f, *, use_alpha=True, constant_alpha=None, epsilon=0.001):
    """Create matching fused and unfused YatNMN layers with shared weights."""
    rngs = nnx.Rngs(42)
    ref = YatNMN(in_f, out_f, use_bias=False, use_alpha=use_alpha,
                 constant_alpha=constant_alpha, epsilon=epsilon, fused=False, rngs=rngs)

    rngs2 = nnx.Rngs(42)  # same seed → same init
    fused = YatNMN(in_f, out_f, use_bias=False, use_alpha=use_alpha,
                   constant_alpha=constant_alpha, epsilon=epsilon, fused=True, rngs=rngs2)

    # Copy weights to ensure identical
    fused.kernel = nnx.Param(ref.kernel.value.copy())
    if ref.alpha is not None and fused.alpha is not None:
        fused.alpha = nnx.Param(ref.alpha.value.copy())

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
    """Assert gradients match for x and kernel."""
    def loss_fn(model, x):
        return jnp.mean(model(x) ** 2)

    loss_ref, grad_ref = jax.value_and_grad(loss_fn)(ref, x)
    loss_fused, grad_fused = jax.value_and_grad(loss_fn)(fused, x)

    # Kernel grads
    gk_ref = grad_ref.kernel.value
    gk_fused = grad_fused.kernel.value
    gk_err = float(jnp.abs(gk_ref - gk_fused).max())
    assert jnp.allclose(gk_ref, gk_fused, atol=atol, rtol=rtol), \
        f"Kernel grad max abs error {gk_err:.2e}"

    # Alpha grads (if present)
    if hasattr(grad_ref, 'alpha') and grad_ref.alpha is not None:
        ga_ref = grad_ref.alpha.value
        ga_fused = grad_fused.alpha.value
        ga_err = float(jnp.abs(ga_ref - ga_fused).max())
        assert jnp.allclose(ga_ref, ga_fused, atol=atol, rtol=rtol), \
            f"Alpha grad max abs error {ga_err:.2e}"

    return loss_ref, loss_fused


# ── Tests ──

class TestFusedForward:
    """Forward pass equivalence tests."""

    def test_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=False)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        _compare_forward(ref, fused, x)

    def test_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        _compare_forward(ref, fused, x)

    def test_constant_alpha_sqrt2(self):
        ref, fused = _make_pair(64, 128, constant_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        _compare_forward(ref, fused, x)

    def test_constant_alpha_custom(self):
        ref, fused = _make_pair(64, 128, constant_alpha=1.5)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        _compare_forward(ref, fused, x)

    def test_2d_input(self):
        ref, fused = _make_pair(32, 64, use_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 32))
        _compare_forward(ref, fused, x)

    def test_3d_input(self):
        ref, fused = _make_pair(768, 3072, use_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 128, 768))
        _compare_forward(ref, fused, x)

    def test_different_epsilons(self):
        for eps in [1e-3, 1e-5, 0.1]:
            ref, fused = _make_pair(64, 128, epsilon=eps)
            x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
            _compare_forward(ref, fused, x)


class TestFusedBackward:
    """Backward pass gradient equivalence tests."""

    def test_grad_no_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=False)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        _compare_grads(ref, fused, x)

    def test_grad_learnable_alpha(self):
        ref, fused = _make_pair(64, 128, use_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        _compare_grads(ref, fused, x)

    def test_grad_constant_alpha(self):
        ref, fused = _make_pair(64, 128, constant_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        _compare_grads(ref, fused, x)

    def test_grad_large_scale(self):
        """GPT-124M FFN scale."""
        ref, fused = _make_pair(768, 3072, use_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 768))
        _compare_grads(ref, fused, x, atol=1e-2, rtol=1e-2)

    def test_grad_finite(self):
        """All gradients should be finite."""
        ref, fused = _make_pair(64, 128, use_alpha=True)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        _, grads = jax.value_and_grad(loss_fn)(fused, x)
        for leaf in jax.tree_util.tree_leaves(grads):
            if hasattr(leaf, 'shape'):
                assert jnp.isfinite(leaf).all(), f"Non-finite gradient found"


class TestFusedFallback:
    """Fused mode should fall back to standard path for unsupported configs."""

    def test_bias_fallback(self):
        """Fused with bias should fall back to standard (bias not supported in fused)."""
        layer = YatNMN(64, 128, use_bias=True, fused=True, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        out = layer(x)
        assert out.shape == (4, 16, 128)
        assert jnp.isfinite(out).all()

    def test_spherical_fallback(self):
        """Fused with spherical should fall back to standard."""
        layer = YatNMN(64, 128, spherical=True, fused=True, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))
        out = layer(x)
        assert out.shape == (4, 16, 128)
        assert jnp.isfinite(out).all()


class TestFusedJIT:
    """Test JIT compilation and tracing."""

    def test_jit_forward(self):
        _, fused = _make_pair(64, 128)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))

        @jax.jit
        def forward(model, x):
            return model(x)

        out = forward(fused, x)
        assert out.shape == (4, 16, 128)
        assert jnp.isfinite(out).all()

    def test_jit_train_step(self):
        _, fused = _make_pair(64, 128)
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 64))

        @jax.jit
        def train_step(model, x):
            def loss_fn(m):
                return jnp.mean(m(x) ** 2)
            loss, grads = jax.value_and_grad(loss_fn)(model)
            return loss

        loss = train_step(fused, x)
        assert jnp.isfinite(loss)
