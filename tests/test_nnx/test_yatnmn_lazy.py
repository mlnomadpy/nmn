"""Tests for YatNMN lazy mode (issue #37, nnx backend).

In lazy mode (``lazy=True`` / ``freeze_kernel=True``) ONLY the kernel is frozen:
it is stored under ``FrozenParam`` so it is excluded from
``nnx.state(model, nnx.Param)`` (and hence the optimizer/grad), while ``bias``,
``alpha`` and the learnable ``epsilon`` remain trainable ``nnx.Param``.
``lazy=False`` (the default) is fully backward compatible.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from nmn.nnx.layers.nmn import YatNMN, FrozenParam


def _param_paths(state):
    return {str(k) for k, _ in nnx.to_flat_state(state)}


class TestLazyStateFiltering:
    def test_kernel_excluded_from_param(self):
        m = YatNMN(8, 4, lazy=True, learnable_epsilon=True, rngs=nnx.Rngs(0))
        params = nnx.state(m, nnx.Param)
        frozen = nnx.state(m, FrozenParam)
        assert "('kernel',)" not in _param_paths(params)
        assert "('kernel',)" in _param_paths(frozen)

    def test_bias_alpha_epsilon_are_trainable(self):
        m = YatNMN(8, 4, lazy=True, use_alpha=True, learnable_epsilon=True, rngs=nnx.Rngs(0))
        paths = _param_paths(nnx.state(m, nnx.Param))
        assert "('bias',)" in paths
        assert "('alpha',)" in paths
        assert "('epsilon_param',)" in paths

    def test_freeze_kernel_alias(self):
        m = YatNMN(8, 4, freeze_kernel=True, rngs=nnx.Rngs(0))
        assert m.lazy is True
        assert "('kernel',)" not in _param_paths(nnx.state(m, nnx.Param))

    def test_freeze_kernel_false_overrides(self):
        # explicit freeze_kernel=False keeps the kernel trainable
        m = YatNMN(8, 4, lazy=True, freeze_kernel=False, rngs=nnx.Rngs(0))
        assert m.lazy is False
        assert "('kernel',)" in _param_paths(nnx.state(m, nnx.Param))

    def test_lazy_with_tie_kernel_bank_raises(self):
        with pytest.raises(ValueError):
            YatNMN(8, 4, lazy=True, tie_kernel_bank=True, rngs=nnx.Rngs(0))


class TestLazyTrainingStep:
    def test_kernel_frozen_others_change(self):
        m = YatNMN(8, 4, lazy=True, use_alpha=True, learnable_epsilon=True, rngs=nnx.Rngs(0))
        k0 = jnp.array(m.kernel[...])
        b0 = jnp.array(m.bias[...])
        a0 = jnp.array(m.alpha[...])
        e0 = jnp.array(m.epsilon_param[...])

        opt = nnx.Optimizer(m, optax.sgd(0.5), wrt=nnx.Param)
        x = jax.random.normal(jax.random.PRNGKey(4), (16, 8))
        y = jax.random.normal(jax.random.PRNGKey(5), (16, 4))

        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)

        eps_grad_seen = False
        for _ in range(3):
            _, grads = nnx.value_and_grad(loss_fn)(m)
            # epsilon receives a (small but nonzero) gradient -> it is trainable.
            g_eps = dict(nnx.to_flat_state(grads)).get(("epsilon_param",))
            if g_eps is not None and float(jnp.sum(jnp.abs(g_eps[...]))) > 0.0:
                eps_grad_seen = True
            opt.update(m, grads)

        # Kernel must be unchanged (frozen).
        assert bool(jnp.allclose(k0, m.kernel[...]))
        # Bias / alpha must move.
        assert not bool(jnp.allclose(b0, m.bias[...]))
        assert not bool(jnp.allclose(a0, m.alpha[...]))
        # Epsilon is trainable: it gets a nonzero gradient and is updated.
        assert eps_grad_seen
        assert float(jnp.abs(e0 - m.epsilon_param[...]).sum()) > 0.0

    def test_grads_exclude_kernel(self):
        m = YatNMN(8, 4, lazy=True, learnable_epsilon=True, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(6), (8, 8))

        def loss_fn(model):
            return jnp.mean(model(x) ** 2)

        _, grads = nnx.value_and_grad(loss_fn)(m)
        assert "('kernel',)" not in _param_paths(grads)
        assert "('bias',)" in _param_paths(grads)


class TestBackwardCompat:
    def test_default_kernel_trainable(self):
        m = YatNMN(8, 4, rngs=nnx.Rngs(0))
        assert m.lazy is False
        assert "('kernel',)" in _param_paths(nnx.state(m, nnx.Param))

    def test_lazy_false_matches_default_forward(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (5, 8))
        m_default = YatNMN(8, 4, rngs=nnx.Rngs(42))
        m_lazy_false = YatNMN(8, 4, lazy=False, rngs=nnx.Rngs(42))
        out_a = m_default(x)
        out_b = m_lazy_false(x)
        assert jnp.allclose(out_a, out_b)

    def test_lazy_forward_matches_nonlazy_same_init(self):
        # Same RNG seed -> same kernel/bias/alpha init -> identical forward,
        # whether or not the kernel is frozen.
        x = jax.random.normal(jax.random.PRNGKey(0), (5, 8))
        m_lazy = YatNMN(8, 4, lazy=True, rngs=nnx.Rngs(123))
        m_train = YatNMN(8, 4, lazy=False, rngs=nnx.Rngs(123))
        assert jnp.allclose(m_lazy(x), m_train(x))

    def test_fused_lazy_forward(self):
        # The fused path must still work in lazy mode (forward only).
        x = jax.random.normal(jax.random.PRNGKey(0), (5, 8))
        m = YatNMN(8, 4, lazy=True, fused=True, rngs=nnx.Rngs(7))
        out = m(x)
        assert out.shape == (5, 4)
        assert bool(jnp.all(jnp.isfinite(out)))
