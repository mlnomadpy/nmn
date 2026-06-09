"""Tests for YatNMN lazy mode (issue #37, Flax Linen).

When ``lazy=True`` (alias ``freeze_kernel=True``) ONLY the ``kernel`` is frozen
via ``jax.lax.stop_gradient``; ``bias``, ``alpha`` and the learnable
``epsilon`` remain trainable. ``lazy=False`` preserves the previous behavior
(backward compatible).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from flax.traverse_util import flatten_dict  # noqa: E402

from nmn.linen.nmn import YatNMN  # noqa: E402


def _make(lazy=False, epsilon=1.0, **kw):
    # Use epsilon=1.0 (not 1e-5) so the softplus-parameterized epsilon sits in
    # a sensitive region with a meaningful gradient for the training smoke test.
    layer = YatNMN(
        features=6,
        use_bias=True,
        use_alpha=True,
        learnable_epsilon=True,
        epsilon=epsilon,
        lazy=lazy,
        **kw,
    )
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    params = layer.init(key, x)
    return layer, params, x


def _grads(layer, params, x):
    def loss(p):
        return jnp.sum(layer.apply(p, x) ** 2)

    return jax.grad(loss)(params)


def test_lazy_default_false_backward_compat():
    """lazy=False yields nonzero kernel gradient (existing behavior)."""
    layer, params, x = _make(lazy=False)
    grads = _grads(layer, params, x)
    flat = flatten_dict(grads)
    kgrad = next(v for k, v in flat.items() if k[-1] == "kernel")
    assert float(jnp.linalg.norm(kgrad)) > 0.0


def test_lazy_freezes_kernel_gradient():
    """lazy=True -> kernel gradient is exactly zero (stop_gradient)."""
    layer, params, x = _make(lazy=True)
    grads = _grads(layer, params, x)
    flat = flatten_dict(grads)
    kgrad = next(v for k, v in flat.items() if k[-1] == "kernel")
    assert float(jnp.linalg.norm(kgrad)) == 0.0


def test_lazy_keeps_bias_alpha_epsilon_trainable():
    """bias, alpha, epsilon retain nonzero gradients in lazy mode."""
    layer, params, x = _make(lazy=True)
    grads = _grads(layer, params, x)
    flat = flatten_dict(grads)

    def gnorm(name):
        g = next(v for k, v in flat.items() if k[-1] == name)
        return float(jnp.linalg.norm(g))

    assert gnorm("bias") > 0.0
    assert gnorm("alpha") > 0.0
    assert gnorm("epsilon_param") > 0.0


def test_freeze_kernel_alias():
    """freeze_kernel=True behaves like lazy=True."""
    layer = YatNMN(features=6, learnable_epsilon=True, freeze_kernel=True)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    params = layer.init(key, x)
    grads = _grads(layer, params, x)
    flat = flatten_dict(grads)
    kgrad = next(v for k, v in flat.items() if k[-1] == "kernel")
    assert float(jnp.linalg.norm(kgrad)) == 0.0


def test_lazy_training_smoke():
    """One SGD step: kernel unchanged, bias/alpha/epsilon change."""
    layer, params, x = _make(lazy=True)
    flat0 = flatten_dict(params)

    grads = _grads(layer, params, x)
    lr = 0.1
    new = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    flat1 = flatten_dict(new)

    def get(flat, name):
        return next(v for k, v in flat.items() if k[-1] == name)

    # kernel unchanged
    assert np.allclose(np.asarray(get(flat0, "kernel")),
                       np.asarray(get(flat1, "kernel")))
    # bias / alpha / epsilon moved
    assert not np.allclose(np.asarray(get(flat0, "bias")),
                           np.asarray(get(flat1, "bias")))
    assert not np.allclose(np.asarray(get(flat0, "alpha")),
                           np.asarray(get(flat1, "alpha")))
    assert not np.allclose(np.asarray(get(flat0, "epsilon_param")),
                           np.asarray(get(flat1, "epsilon_param")))


def test_lazy_forward_matches_nonlazy_at_init():
    """Freezing the kernel must not change the forward pass at init."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    lazy = YatNMN(features=6, learnable_epsilon=True, lazy=True)
    eager = YatNMN(features=6, learnable_epsilon=True, lazy=False)
    p_lazy = lazy.init(key, x)
    p_eager = eager.init(key, x)
    y_lazy = lazy.apply(p_lazy, x)
    y_eager = eager.apply(p_eager, x)
    assert np.allclose(np.asarray(y_lazy), np.asarray(y_eager))


def test_optax_multi_transform_freezes_kernel():
    """The documented optax multi_transform pattern keeps the kernel exactly
    constant while updating the other params."""
    optax = pytest.importorskip("optax")
    from flax.traverse_util import path_aware_map

    layer, params, x = _make(lazy=True)

    def label_fn(p):
        return path_aware_map(
            lambda path, _: "frozen" if path[-1] == "kernel" else "trainable", p
        )

    tx = optax.multi_transform(
        {"trainable": optax.sgd(0.1), "frozen": optax.set_to_zero()}, label_fn
    )
    opt_state = tx.init(params)
    grads = _grads(layer, params, x)
    updates, _ = tx.update(grads, opt_state, params)
    new = optax.apply_updates(params, updates)

    f0, f1 = flatten_dict(params), flatten_dict(new)

    def get(flat, name):
        return next(v for k, v in flat.items() if k[-1] == name)

    assert np.allclose(np.asarray(get(f0, "kernel")), np.asarray(get(f1, "kernel")))
    assert not np.allclose(np.asarray(get(f0, "bias")), np.asarray(get(f1, "bias")))
