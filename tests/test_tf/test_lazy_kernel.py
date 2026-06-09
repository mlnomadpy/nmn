"""Tests for TensorFlow YatNMN lazy mode (issue #37).

In lazy mode ONLY the kernel (feature directions) is frozen; bias, alpha, and
the learnable epsilon remain trainable. Default (``lazy=False``) is unchanged.

TensorFlow is not installed in the local dev environment, so the module is
guarded by ``pytest.importorskip`` and exercised in CI.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nmn.tf.nmn import YatNMN


def _build(layer, in_dim=8):
    x = tf.random.normal((4, in_dim))
    layer(x)  # triggers build
    return x


def test_lazy_kernel_non_trainable():
    layer = YatNMN(features=6, lazy=True, use_bias=True, use_alpha=True, learnable_epsilon=True)
    _build(layer)
    assert layer.kernel.trainable is False
    trainable_names = [v.name for v in layer.trainable_variables]
    # kernel must be excluded from the optimizer's trainable set.
    assert not any("kernel" in n for n in trainable_names)


def test_lazy_bias_alpha_epsilon_trainable():
    layer = YatNMN(features=6, lazy=True, use_bias=True, use_alpha=True, learnable_epsilon=True)
    _build(layer)
    assert layer.bias.trainable is True
    assert layer.alpha.trainable is True
    assert layer.epsilon_param.trainable is True
    trainable_names = [v.name for v in layer.trainable_variables]
    assert any("bias" in n for n in trainable_names)
    assert any("alpha" in n for n in trainable_names)
    assert any("epsilon" in n for n in trainable_names)


def test_freeze_kernel_alias():
    layer = YatNMN(features=6, freeze_kernel=True)
    _build(layer)
    assert layer.lazy is True
    assert layer.kernel.trainable is False


def test_lazy_false_backward_compatible():
    layer = YatNMN(features=6, use_bias=True, use_alpha=True, learnable_epsilon=True)
    _build(layer)
    assert layer.lazy is False
    assert layer.kernel.trainable is True
    trainable_names = [v.name for v in layer.trainable_variables]
    assert any("kernel" in n for n in trainable_names)


def test_lazy_gradients_kernel_none_others_present():
    layer = YatNMN(features=6, lazy=True, use_bias=True, use_alpha=True, learnable_epsilon=True)
    x = _build(layer)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(layer(x)))
    grads = tape.gradient(loss, [layer.kernel, layer.bias, layer.alpha, layer.epsilon_param])
    g_kernel, g_bias, g_alpha, g_eps = grads
    # Frozen kernel: no gradient flows to it through the trainable-variable path.
    # (It is excluded from trainable_variables; explicit gradient is None.)
    assert g_kernel is None
    assert g_bias is not None and np.any(g_bias.numpy() != 0.0)
    assert g_alpha is not None
    assert g_eps is not None


def test_lazy_training_step_kernel_unchanged():
    """1-2 step smoke test: kernel stays fixed, bias/alpha/epsilon move."""
    layer = YatNMN(features=6, lazy=True, use_bias=True, use_alpha=True, learnable_epsilon=True)
    x = _build(layer)

    kernel0 = layer.kernel.numpy().copy()
    bias0 = layer.bias.numpy().copy()
    alpha0 = layer.alpha.numpy().copy()
    eps0 = layer.epsilon_param.numpy().copy()

    lr = 0.1
    target = tf.random.normal((4, 6))
    for _ in range(2):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.square(layer(x) - target))
        # Manual SGD over only the trainable variables (kernel excluded).
        grads = tape.gradient(loss, layer.trainable_variables)
        for g, var in zip(grads, layer.trainable_variables):
            if g is not None:
                var.assign_sub(lr * g)

    # Kernel frozen.
    np.testing.assert_array_equal(layer.kernel.numpy(), kernel0)
    # bias / alpha / epsilon updated.
    assert not np.array_equal(layer.bias.numpy(), bias0)
    assert not np.array_equal(layer.alpha.numpy(), alpha0)
    assert not np.array_equal(layer.epsilon_param.numpy(), eps0)


def test_lazy_forward_matches_non_lazy_at_init():
    """Lazy mode must not change the forward computation, only trainability."""
    eager = YatNMN(features=6, lazy=False, use_bias=True, use_alpha=True)
    lazy = YatNMN(features=6, lazy=True, use_bias=True, use_alpha=True)
    x = tf.random.normal((4, 8))
    eager(x)
    lazy(x)
    # Copy weights so both compute the same function.
    lazy.kernel.assign(eager.kernel)
    lazy.bias.assign(eager.bias)
    lazy.alpha.assign(eager.alpha)
    np.testing.assert_allclose(eager(x).numpy(), lazy(x).numpy(), rtol=1e-6, atol=1e-6)
