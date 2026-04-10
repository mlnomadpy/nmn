"""Math validation tests for Keras YAT layers.

Validates that Keras implementations compute the correct YAT formula:
    y = (x·W + b)² / (‖x - W‖² + ε)
"""

import pytest
import numpy as np


def test_keras_yat_nmn_formula_no_bias():
    """YatNMN (Keras) computes (x·W)² / (‖x-W‖² + ε) when use_bias=False."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
    except ImportError:
        pytest.skip("Keras/TF not available")

    np.random.seed(42)
    x_np = np.random.randn(2, 4).astype(np.float32)
    layer = YatNMN(units=3, use_bias=False, use_alpha=False, epsilon=1e-5)
    out = layer(x_np).numpy()

    w = layer.kernel.numpy()   # (4, 3)
    dot = x_np @ w             # (2, 3)
    x_sq = np.sum(x_np ** 2, axis=-1, keepdims=True)
    w_sq = np.sum(w ** 2, axis=0)   # (3,)
    dist = x_sq + w_sq - 2 * dot
    expected = dot ** 2 / (dist + 1e-5)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_keras_yat_nmn_formula_with_bias():
    """YatNMN (Keras) puts bias INSIDE the square: (x·W + b)² / dist."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
        import keras
    except ImportError:
        pytest.skip("Keras/TF not available")

    np.random.seed(7)
    x_np = np.random.randn(2, 4).astype(np.float32)
    layer = YatNMN(units=3, use_bias=True, use_alpha=False, epsilon=1e-5)
    _ = layer(x_np)  # build

    w_val = np.random.randn(4, 3).astype(np.float32)
    b_val = np.random.randn(3).astype(np.float32)
    layer.kernel.assign(w_val)
    layer.bias.assign(b_val)

    out = layer(x_np).numpy()

    dot = x_np @ w_val
    x_sq = np.sum(x_np ** 2, axis=-1, keepdims=True)
    w_sq = np.sum(w_val ** 2, axis=0)
    dist = x_sq + w_sq - 2 * dot
    expected = (dot + b_val) ** 2 / (dist + 1e-5)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5,
                                err_msg="Keras bias must be inside the numerator square")


def test_keras_yat_nmn_bias_not_added_after():
    """Confirm output with bias ≠ output_no_bias + bias (wrong formula check)."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
    except ImportError:
        pytest.skip("Keras/TF not available")

    np.random.seed(3)
    x_np = np.random.randn(2, 4).astype(np.float32)

    layer_no_bias = YatNMN(units=3, use_bias=False, use_alpha=False, epsilon=1e-5)
    layer_bias = YatNMN(units=3, use_bias=True, use_alpha=False, epsilon=1e-5)
    _ = layer_no_bias(x_np)
    _ = layer_bias(x_np)

    w_val = np.random.randn(4, 3).astype(np.float32)
    b_val = np.ones(3, dtype=np.float32) * 0.5
    layer_no_bias.kernel.assign(w_val)
    layer_bias.kernel.assign(w_val)
    layer_bias.bias.assign(b_val)

    out_no_bias = layer_no_bias(x_np).numpy()
    out_bias = layer_bias(x_np).numpy()

    wrong_formula = out_no_bias + b_val
    assert not np.allclose(out_bias, wrong_formula, atol=1e-4), \
        "Keras bias appears to be added AFTER squaring (wrong formula)"


def test_keras_yat_nmn_constant_alpha():
    """constant_alpha=True applies sqrt(2) scaling."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
        import math
    except ImportError:
        pytest.skip("Keras/TF not available")

    np.random.seed(1)
    x_np = np.random.randn(2, 4).astype(np.float32)
    layer = YatNMN(units=3, use_bias=False, use_alpha=False,
                   constant_alpha=True, epsilon=1e-5)
    layer_no_alpha = YatNMN(units=3, use_bias=False, use_alpha=False, epsilon=1e-5)
    _ = layer(x_np)
    _ = layer_no_alpha(x_np)

    w_val = np.random.randn(4, 3).astype(np.float32)
    layer.kernel.assign(w_val)
    layer_no_alpha.kernel.assign(w_val)

    out = layer(x_np).numpy()
    out_no_alpha = layer_no_alpha(x_np).numpy()

    np.testing.assert_allclose(out, out_no_alpha * math.sqrt(2), rtol=1e-5)


def test_keras_yat_nmn_spherical_distance_correct():
    """Spherical mode: inputs and kernel normalized per-row, distance = 2 - 2*(x·W)."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
    except ImportError:
        pytest.skip("Keras/TF not available")

    np.random.seed(11)
    x_np = np.random.randn(2, 4).astype(np.float32)

    layer = YatNMN(units=3, use_bias=True, use_alpha=False,
                   spherical=True, epsilon=1e-5)
    _ = layer(x_np)

    w_val = np.random.randn(4, 3).astype(np.float32)
    b_val = np.random.randn(3).astype(np.float32)
    layer.kernel.assign(w_val)
    layer.bias.assign(b_val)

    out = layer(x_np).numpy()

    # Manual spherical computation — normalize each input row and each kernel
    # column (neuron) to unit norm. Then ||x||=||W_col||=1, so dist = 2 - 2*(x·W).
    x_norm = x_np / (np.linalg.norm(x_np, axis=-1, keepdims=True) + 1e-8)
    # Kernel shape is (4, 3): each column is a neuron → normalize axis=0
    w_norm = w_val / (np.linalg.norm(w_val, axis=0, keepdims=True) + 1e-8)
    dot = x_norm @ w_norm
    dist = np.maximum(2 - 2 * dot, 0.0)
    expected = (dot + b_val) ** 2 / (dist + 1e-5)

    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4,
                                err_msg="Spherical mode computation mismatch")


def test_keras_yat_nmn_epsilon_positive_validation():
    """YatNMN raises ValueError for epsilon <= 0."""
    try:
        from nmn.keras.nmn import YatNMN
    except ImportError:
        pytest.skip("Keras/TF not available")

    with pytest.raises(ValueError, match="epsilon must be positive"):
        YatNMN(units=4, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        YatNMN(units=4, epsilon=-1e-5)
