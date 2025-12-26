"""Tests for TensorFlow YatConvTranspose layers - TDD: Write tests first."""

import pytest
import numpy as np


def test_tf_yat_conv_transpose2d_import():
    """Test that YatConvTranspose2D can be imported."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConvTranspose2D
        assert YatConvTranspose2D is not None
    except ImportError as e:
        pytest.skip(f"TensorFlow dependencies not available: {e}")


def test_tf_yat_conv_transpose2d_instantiation():
    """Test YatConvTranspose2D can be instantiated."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConvTranspose2D
        
        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2))
        assert layer is not None
        assert layer.filters == 8
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose2d_build():
    """Test YatConvTranspose2D builds correctly."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConvTranspose2D
        
        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2))
        # Build with input [batch, height, width, channels]
        layer.build([None, 16, 16, 16])
        
        assert layer.kernel is not None
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose2d_forward():
    """Test YatConvTranspose2D forward pass (upsamples by 2x)."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConvTranspose2D
        
        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2))
        
        # Create input [batch, height, width, channels]
        dummy_input = tf.constant(np.random.randn(2, 16, 16, 16).astype(np.float32))
        output = layer(dummy_input)
        
        # For transpose conv with stride=2, kernel=2, output should be 2x input
        assert output.shape == (2, 32, 32, 8)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose1d_forward():
    """Test YatConvTranspose1D forward pass."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConvTranspose1D
        
        layer = YatConvTranspose1D(filters=8, kernel_size=2, strides=2)
        
        # Create input [batch, length, channels]
        dummy_input = tf.constant(np.random.randn(2, 16, 16).astype(np.float32))
        output = layer(dummy_input)
        
        # For transpose conv with stride=2, kernel=2, output should be 2x input
        assert output.shape == (2, 32, 8)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose3d_forward():
    """Test YatConvTranspose3D forward pass."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConvTranspose3D
        
        layer = YatConvTranspose3D(filters=8, kernel_size=(2, 2, 2), strides=(2, 2, 2))
        
        # Create input [batch, depth, height, width, channels]
        dummy_input = tf.constant(np.random.randn(2, 8, 8, 8, 16).astype(np.float32))
        output = layer(dummy_input)
        
        # For transpose conv with stride=2, kernel=2, output should be 2x input
        assert output.shape == (2, 16, 16, 16, 8)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose2d_gradient():
    """Test that YatConvTranspose2D can compute gradients."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConvTranspose2D
        
        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2))
        
        dummy_input = tf.constant(np.random.randn(2, 16, 16, 16).astype(np.float32))
        
        with tf.GradientTape() as tape:
            output = layer(dummy_input)
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, layer.trainable_variables)
        
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose2d_no_bias():
    """Test YatConvTranspose2D without bias."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConvTranspose2D
        
        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2), use_bias=False)
        layer.build([None, 16, 16, 16])
        
        assert layer.bias is None
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")

