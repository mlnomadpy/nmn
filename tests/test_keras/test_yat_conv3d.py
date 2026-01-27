"""Tests for Keras YatConv3D - TDD: Write tests first."""

import pytest
import numpy as np


def test_keras_yat_conv3d_import():
    """Test that YatConv3D can be imported."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv3D
        assert YatConv3D is not None
    except ImportError as e:
        pytest.skip(f"Keras/TensorFlow dependencies not available: {e}")


def test_keras_yat_conv3d_instantiation():
    """Test YatConv3D can be instantiated."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv3D
        
        layer = YatConv3D(filters=16, kernel_size=3)
        assert layer is not None
        assert layer.filters == 16
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv3d_build():
    """Test YatConv3D builds correctly."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv3D
        
        layer = YatConv3D(filters=16, kernel_size=3)
        layer.build((None, 8, 8, 8, 3))
        
        assert layer.kernel is not None
        assert layer.kernel.shape == (3, 3, 3, 3, 16)
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv3d_forward():
    """Test YatConv3D forward pass."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv3D
        
        layer = YatConv3D(filters=16, kernel_size=3)
        layer.build((None, 8, 8, 8, 3))
        
        # Create input
        dummy_input = tf.constant(np.random.randn(2, 8, 8, 8, 3).astype(np.float32))
        output = layer(dummy_input)
        
        # For valid padding: output_size = input_size - kernel_size + 1 = 8 - 3 + 1 = 6
        assert output.shape == (2, 6, 6, 6, 16)
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv3d_same_padding():
    """Test YatConv3D with SAME padding."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv3D
        
        layer = YatConv3D(filters=16, kernel_size=3, padding='same')
        layer.build((None, 8, 8, 8, 3))
        
        # Create input
        dummy_input = tf.constant(np.random.randn(2, 8, 8, 8, 3).astype(np.float32))
        output = layer(dummy_input)
        
        # For same padding: output_size = input_size
        assert output.shape == (2, 8, 8, 8, 16)
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv3d_no_bias():
    """Test YatConv3D without bias."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv3D
        
        layer = YatConv3D(filters=16, kernel_size=3, use_bias=False)
        layer.build((None, 8, 8, 8, 3))
        
        assert layer.bias is None
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv3d_alpha():
    """Test YatConv3D alpha scaling."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv3D
        
        layer = YatConv3D(filters=16, kernel_size=3, use_alpha=True)
        layer.build((None, 8, 8, 8, 3))
        
        assert layer.alpha is not None
        
        layer_no_alpha = YatConv3D(filters=16, kernel_size=3, use_alpha=False)
        layer_no_alpha.build((None, 8, 8, 8, 3))
        
        assert layer_no_alpha.alpha is None
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv3d_gradient():
    """Test that YatConv3D can compute gradients."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv3D
        
        layer = YatConv3D(filters=16, kernel_size=3)
        layer.build((None, 8, 8, 8, 3))
        
        dummy_input = tf.constant(np.random.randn(2, 8, 8, 8, 3).astype(np.float32))
        
        with tf.GradientTape() as tape:
            output = layer(dummy_input)
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, layer.trainable_variables)
        
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")




