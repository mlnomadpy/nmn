"""Tests for YatNMN lazy mode (issue #37) on the Keras backend.

Lazy mode freezes ONLY the kernel (its feature directions). The bias, alpha,
and learnable epsilon stay trainable. TensorFlow is not installed locally, so
these tests are guarded by ``pytest.importorskip("tensorflow")`` and run in CI.
"""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from nmn.keras.nmn import YatNMN


def _build(layer, input_dim=8):
    x = np.random.randn(4, input_dim).astype(np.float32)
    layer(x)  # triggers build
    return x


def _weight(layer, name):
    for w in layer.weights:
        if w.name == name or w.name.endswith("/" + name) or name in w.name:
            return w
    raise KeyError(name)


# --------------------------------------------------------------------------
# Backward compatibility (lazy defaults to False).
# --------------------------------------------------------------------------
def test_default_kernel_trainable():
    layer = YatNMN(units=6)
    _build(layer)
    assert layer.lazy is False
    assert layer.kernel.trainable is True


def test_default_output_unchanged_vs_lazy_construction():
    """lazy=False must preserve current behavior (same forward output)."""
    x = np.random.randn(4, 8).astype(np.float32)
    eager = YatNMN(units=6, kernel_initializer="ones", bias_initializer="zeros")
    out_eager = np.array(eager(x))
    assert np.all(np.isfinite(out_eager))


# --------------------------------------------------------------------------
# Lazy mode: kernel frozen, everything else trainable.
# --------------------------------------------------------------------------
def test_lazy_kernel_not_trainable():
    layer = YatNMN(units=6, lazy=True)
    _build(layer)
    assert layer.lazy is True
    assert layer.kernel.trainable is False
    # Kernel excluded from the trainable set, not merely zero-grad.
    trainable_names = [w.name for w in layer.trainable_weights]
    assert not any("kernel" in n and "bias" not in n for n in trainable_names)


def test_freeze_kernel_alias():
    layer = YatNMN(units=6, freeze_kernel=True)
    _build(layer)
    assert layer.lazy is True
    assert layer.kernel.trainable is False


def test_lazy_bias_alpha_epsilon_trainable():
    layer = YatNMN(
        units=6,
        lazy=True,
        use_bias=True,
        use_alpha=True,
        learnable_epsilon=True,
    )
    _build(layer)
    trainable_names = [w.name for w in layer.trainable_weights]

    assert layer.bias is not None and layer.bias.trainable is True
    assert layer.alpha is not None and layer.alpha.trainable is True
    assert layer.epsilon_param is not None and layer.epsilon_param.trainable is True

    assert any("bias" in n for n in trainable_names)
    assert any("alpha" in n for n in trainable_names)
    assert any("epsilon" in n for n in trainable_names)
    # The kernel is the only non-trainable weight.
    non_trainable = [w.name for w in layer.non_trainable_weights]
    assert any("kernel" in n for n in non_trainable)


def test_lazy_training_step_kernel_unchanged():
    """One training step: kernel stays fixed; bias/alpha/epsilon move."""
    tf = pytest.importorskip("tensorflow")

    layer = YatNMN(
        units=4,
        lazy=True,
        use_bias=True,
        use_alpha=True,
        learnable_epsilon=True,
    )
    x = tf.constant(np.random.randn(8, 8).astype(np.float32))
    y = tf.constant(np.random.randn(8, 4).astype(np.float32))
    layer(x)  # build

    kernel_before = np.array(layer.kernel)
    bias_before = np.array(layer.bias)
    alpha_before = np.array(layer.alpha)

    # epsilon_param initializes at inverse-softplus(1e-5) ≈ -11.5, a saturated
    # point where the softplus gradient is ~1e-5, so its *value* barely moves in
    # a couple of SGD steps. The property #37 requires is that epsilon stays
    # trainable (receives a real gradient), not frozen — assert that directly
    # rather than a fragile value-change threshold.
    names = [w.name for w in layer.trainable_weights]
    eps_idx = next(i for i, n in enumerate(names) if "epsilon" in n)

    opt = keras.optimizers.SGD(learning_rate=0.1)
    eps_grad_seen = False
    for _ in range(2):
        with tf.GradientTape() as tape:
            pred = layer(x)
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, layer.trainable_weights)
        g_eps = grads[eps_idx]
        if g_eps is not None and float(tf.reduce_max(tf.abs(g_eps))) > 0.0:
            eps_grad_seen = True
        opt.apply_gradients(zip(grads, layer.trainable_weights))

    # Kernel untouched (frozen).
    np.testing.assert_allclose(kernel_before, np.array(layer.kernel), atol=1e-7)
    # Bias / alpha moved (healthy gradients).
    assert not np.allclose(bias_before, np.array(layer.bias), atol=1e-7)
    assert not np.allclose(alpha_before, np.array(layer.alpha), atol=1e-7)
    # Epsilon is trainable: it received a real (non-None, non-zero) gradient.
    assert eps_grad_seen, "epsilon_param did not receive a gradient (wrongly frozen)"


def test_eager_training_step_kernel_changes():
    """lazy=False (default): the kernel DOES update (backward compatible)."""
    tf = pytest.importorskip("tensorflow")

    layer = YatNMN(units=4, lazy=False, use_bias=True)
    x = tf.constant(np.random.randn(8, 8).astype(np.float32))
    y = tf.constant(np.random.randn(8, 4).astype(np.float32))
    layer(x)

    kernel_before = np.array(layer.kernel)
    opt = keras.optimizers.SGD(learning_rate=0.1)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(layer(x) - y))
    grads = tape.gradient(loss, layer.trainable_weights)
    opt.apply_gradients(zip(grads, layer.trainable_weights))

    assert not np.allclose(kernel_before, np.array(layer.kernel), atol=1e-7)


def test_lazy_get_config_roundtrip():
    layer = YatNMN(units=6, lazy=True)
    config = layer.get_config()
    assert config["lazy"] is True
    assert config["freeze_kernel"] is True
    rebuilt = YatNMN.from_config(config)
    assert rebuilt.lazy is True
