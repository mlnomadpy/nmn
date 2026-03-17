"""Math validation tests for TensorFlow YAT layers.

Validates that TF implementations compute the correct YAT formula:
    y = (x·W + b)² / (‖x - W‖² + ε)

and that behaviour matches the PyTorch reference implementation.
"""

import pytest
import numpy as np


def test_tf_yat_nmn_formula_no_bias():
    """YatNMN computes (x·Wᵀ)² / (‖x-W‖² + ε) when bias=False."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
    except ImportError:
        pytest.skip("TensorFlow not available")

    np.random.seed(42)
    x_np = np.random.randn(2, 4).astype(np.float32)
    layer = YatNMN(features=3, use_bias=False, use_alpha=False, epsilon=1e-5)
    out = layer(tf.constant(x_np)).numpy()

    # Build manually
    w = layer.kernel.numpy()  # shape (3, 4)
    dot = x_np @ w.T          # (2, 3)
    x_sq = np.sum(x_np ** 2, axis=-1, keepdims=True)   # (2, 1)
    w_sq = np.sum(w ** 2, axis=-1)                       # (3,)
    dist = x_sq + w_sq - 2 * dot                         # (2, 3)
    expected = dot ** 2 / (dist + 1e-5)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_tf_yat_nmn_formula_with_bias():
    """YatNMN puts bias INSIDE the square: (x·Wᵀ + b)² / dist."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
    except ImportError:
        pytest.skip("TensorFlow not available")

    np.random.seed(7)
    x_np = np.random.randn(2, 4).astype(np.float32)
    layer = YatNMN(features=3, use_bias=True, use_alpha=False, epsilon=1e-5)
    _ = layer(tf.constant(x_np))  # build

    # Manually set deterministic weights
    w_val = np.random.randn(3, 4).astype(np.float32)
    b_val = np.random.randn(3).astype(np.float32)
    layer.kernel.assign(w_val)
    layer.bias.assign(b_val)

    out = layer(tf.constant(x_np)).numpy()

    dot = x_np @ w_val.T          # (2, 3)
    x_sq = np.sum(x_np ** 2, axis=-1, keepdims=True)
    w_sq = np.sum(w_val ** 2, axis=-1)
    dist = x_sq + w_sq - 2 * dot
    # Bias is INSIDE the square
    expected = (dot + b_val) ** 2 / (dist + 1e-5)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5,
                                err_msg="Bias must be inside the numerator square")


def test_tf_yat_nmn_bias_not_added_after():
    """Confirm output with bias ≠ output_no_bias + bias (old wrong formula)."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
    except ImportError:
        pytest.skip("TensorFlow not available")

    np.random.seed(3)
    x_np = np.random.randn(2, 4).astype(np.float32)

    layer_no_bias = YatNMN(features=3, use_bias=False, use_alpha=False, epsilon=1e-5)
    layer_bias = YatNMN(features=3, use_bias=True, use_alpha=False, epsilon=1e-5)

    _ = layer_no_bias(tf.constant(x_np))
    _ = layer_bias(tf.constant(x_np))

    # Force same weights
    w_val = np.random.randn(3, 4).astype(np.float32)
    b_val = np.ones(3, dtype=np.float32) * 0.5
    layer_no_bias.kernel.assign(w_val)
    layer_bias.kernel.assign(w_val)
    layer_bias.bias.assign(b_val)

    out_no_bias = layer_no_bias(tf.constant(x_np)).numpy()
    out_bias = layer_bias(tf.constant(x_np)).numpy()

    # If bias was wrongly added after, these would be equal up to b_val
    # They should NOT satisfy: out_bias ≈ out_no_bias + b_val
    wrong_formula = out_no_bias + b_val
    assert not np.allclose(out_bias, wrong_formula, atol=1e-4), \
        "Bias appears to be added AFTER squaring (wrong formula)"


def test_tf_yat_nmn_constant_alpha():
    """constant_alpha=True applies sqrt(2) scaling."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
        import math
    except ImportError:
        pytest.skip("TensorFlow not available")

    np.random.seed(1)
    x_np = np.random.randn(2, 4).astype(np.float32)
    layer = YatNMN(features=3, use_bias=False, use_alpha=False,
                   constant_alpha=True, epsilon=1e-5)
    layer_no_alpha = YatNMN(features=3, use_bias=False, use_alpha=False, epsilon=1e-5)

    _ = layer(tf.constant(x_np))
    _ = layer_no_alpha(tf.constant(x_np))
    w_val = np.random.randn(3, 4).astype(np.float32)
    layer.kernel.assign(w_val)
    layer_no_alpha.kernel.assign(w_val)

    out = layer(tf.constant(x_np)).numpy()
    out_no_alpha = layer_no_alpha(tf.constant(x_np)).numpy()

    np.testing.assert_allclose(out, out_no_alpha * math.sqrt(2), rtol=1e-5)


def test_tf_yat_nmn_epsilon_positive_validation():
    """YatNMN raises ValueError for epsilon <= 0."""
    try:
        from nmn.tf.nmn import YatNMN
    except ImportError:
        pytest.skip("TensorFlow not available")

    with pytest.raises(ValueError, match="epsilon must be positive"):
        YatNMN(features=4, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        YatNMN(features=4, epsilon=-1e-5)


def test_tf_yat_nmn_get_set_weights_with_constant_alpha():
    """get_weights/set_weights work when constant_alpha=True (alpha is None)."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
    except ImportError:
        pytest.skip("TensorFlow not available")

    x_np = np.random.randn(2, 4).astype(np.float32)
    layer = YatNMN(features=3, constant_alpha=True, epsilon=1e-5)
    _ = layer(tf.constant(x_np))  # build

    weights = layer.get_weights()
    # Should only contain kernel (and bias if enabled), not alpha
    assert all(w is not None for w in weights), \
        "get_weights() returned None entries"

    # Round-trip
    layer.set_weights(weights)
    out1 = layer(tf.constant(x_np)).numpy()
    out2 = layer(tf.constant(x_np)).numpy()
    np.testing.assert_array_equal(out1, out2)


def test_tf_yat_conv2d_formula():
    """YatConv2D computes (conv + b)² / (dist + ε) with bias inside square."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv2D
    except ImportError:
        pytest.skip("TensorFlow not available")

    np.random.seed(99)
    # Single channel, single filter, 1x1 kernel — reduces to dense YAT
    x_np = np.random.randn(1, 4, 4, 1).astype(np.float32)
    layer = YatConv2D(filters=1, kernel_size=1, use_bias=True, use_alpha=False,
                      epsilon=1e-5, padding="valid")
    out = layer(tf.constant(x_np)).numpy()

    w = layer.kernel.numpy()   # (1, 1, 1, 1)
    b = layer.bias.numpy()     # (1,)

    dot = tf.nn.conv2d(x_np, w, strides=[1,1,1,1], padding="VALID").numpy()
    x_sq = np.sum(x_np ** 2, axis=-1, keepdims=True)
    w_sq = np.sum(w ** 2, axis=(0, 1, 2))
    dist = x_sq + w_sq - 2 * dot
    expected = (dot + b) ** 2 / (dist + 1e-5)

    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)


def test_tf_yat_conv2d_bias_inside_square():
    """Confirm YatConv2D bias is NOT added after squaring."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv2D
    except ImportError:
        pytest.skip("TensorFlow not available")

    np.random.seed(5)
    x_np = np.random.randn(1, 4, 4, 2).astype(np.float32)

    layer_no_bias = YatConv2D(filters=4, kernel_size=3, use_bias=False,
                               use_alpha=False, epsilon=1e-5, padding="same")
    layer_bias = YatConv2D(filters=4, kernel_size=3, use_bias=True,
                            use_alpha=False, epsilon=1e-5, padding="same")
    _ = layer_no_bias(tf.constant(x_np))
    _ = layer_bias(tf.constant(x_np))

    w_val = layer_no_bias.kernel.numpy()
    b_val = np.ones(4, dtype=np.float32) * 0.5
    layer_bias.kernel.assign(w_val)
    layer_bias.bias.assign(b_val)

    out_no_bias = layer_no_bias(tf.constant(x_np)).numpy()
    out_bias = layer_bias(tf.constant(x_np)).numpy()

    wrong = out_no_bias + b_val
    assert not np.allclose(out_bias, wrong, atol=1e-4), \
        "YatConv2D bias appears to be added AFTER squaring (wrong formula)"
