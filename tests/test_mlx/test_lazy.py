"""Tests for YatNMN lazy mode (freeze only the kernel) — issue #37."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
import mlx.utils as mu  # noqa: E402

from nmn.mlx import YatNMN  # noqa: E402


def _trainable_keys(model):
    return set(dict(mu.tree_flatten(model.trainable_parameters())).keys())


def _build(model, in_dim=8):
    model(mx.zeros((4, in_dim)))
    return model


# ---------------------------------------------------------------------------
# trainable-parameter membership
# ---------------------------------------------------------------------------


def test_lazy_excludes_kernel_keeps_rest_trainable():
    m = _build(YatNMN(features=6, use_bias=True, use_alpha=True,
                      learnable_epsilon=True, lazy=True))
    keys = _trainable_keys(m)
    assert not any("kernel" in k for k in keys)
    assert any("bias" in k for k in keys)
    assert any("alpha" in k for k in keys)
    assert any("epsilon" in k for k in keys)


def test_freeze_kernel_alias():
    m = _build(YatNMN(features=6, learnable_epsilon=True, freeze_kernel=True))
    keys = _trainable_keys(m)
    assert not any("kernel" in k for k in keys)
    assert any("bias" in k for k in keys)


def test_lazy_false_is_backward_compatible():
    m = _build(YatNMN(features=6, use_bias=True, use_alpha=True,
                      learnable_epsilon=True))
    keys = _trainable_keys(m)
    # Default: everything trainable, including the kernel.
    assert any("kernel" in k for k in keys)
    assert any("bias" in k for k in keys)
    assert any("alpha" in k for k in keys)
    assert any("epsilon" in k for k in keys)


def test_lazy_forward_matches_eager_with_same_weights():
    """lazy=True must not change the forward computation."""
    x = mx.random.normal(shape=(4, 8))
    eager = _build(YatNMN(features=6, lazy=False), in_dim=8)
    lazy = _build(YatNMN(features=6, lazy=True), in_dim=8)
    # Copy eager weights into lazy so the only difference is the freeze flag.
    lazy.kernel = eager.kernel
    lazy.bias = eager.bias
    lazy.alpha = eager.alpha
    out_e = np.array(eager(x))
    out_l = np.array(lazy(x))
    assert np.allclose(out_e, out_l, atol=1e-6)


# ---------------------------------------------------------------------------
# gradient / training behavior
# ---------------------------------------------------------------------------


def test_lazy_kernel_gets_no_gradient():
    m = _build(YatNMN(features=6, learnable_epsilon=True, lazy=True))
    x = mx.random.normal(shape=(4, 8))

    def loss_fn(model):
        return mx.sum(model(x) ** 2)

    grads = nn.value_and_grad(m, loss_fn)(m)[1]
    flat = dict(mu.tree_flatten(grads))
    # The frozen kernel must not appear in the gradient tree.
    assert not any("kernel" in k for k in flat)
    assert any("bias" in k for k in flat)


def test_lazy_training_step_kernel_unchanged_others_change():
    import mlx.optimizers as optim

    # Start epsilon away from the tiny 1e-5 default so softplus has a usable
    # slope and the epsilon gradient is non-negligible for the smoke test.
    m = _build(YatNMN(features=6, use_bias=True, use_alpha=True,
                      learnable_epsilon=True, epsilon=1.0, lazy=True))
    x = mx.random.normal(shape=(8, 8))
    target = mx.random.normal(shape=(8, 6))

    kernel_before = np.array(m.kernel)
    bias_before = np.array(m.bias)
    alpha_before = np.array(m.alpha)
    eps_before = np.array(m.epsilon_param)

    opt = optim.SGD(learning_rate=0.1)

    def loss_fn(model):
        return mx.mean((model(x) - target) ** 2)

    for _ in range(2):
        loss, grads = nn.value_and_grad(m, loss_fn)(m)
        opt.update(m, grads)
        mx.eval(m.parameters(), opt.state)

    # Kernel must be unchanged; bias/alpha/epsilon must move.
    assert np.array_equal(kernel_before, np.array(m.kernel))
    assert not np.array_equal(bias_before, np.array(m.bias))
    assert not np.array_equal(alpha_before, np.array(m.alpha))
    assert not np.array_equal(eps_before, np.array(m.epsilon_param))


def test_non_lazy_kernel_does_change():
    import mlx.optimizers as optim

    m = _build(YatNMN(features=6, lazy=False))
    x = mx.random.normal(shape=(8, 8))
    target = mx.random.normal(shape=(8, 6))
    kernel_before = np.array(m.kernel)
    opt = optim.SGD(learning_rate=0.1)

    def loss_fn(model):
        return mx.mean((model(x) - target) ** 2)

    for _ in range(2):
        loss, grads = nn.value_and_grad(m, loss_fn)(m)
        opt.update(m, grads)
        mx.eval(m.parameters(), opt.state)

    assert not np.array_equal(kernel_before, np.array(m.kernel))
