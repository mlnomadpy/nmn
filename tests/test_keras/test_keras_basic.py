"""Tests for Keras implementation."""

import pytest
import numpy as np


def test_keras_import():
    """Test that Keras module can be imported."""
    try:
        from nmn.keras import nmn
        from nmn.keras import conv
        assert hasattr(nmn, 'YatNMN')
        assert hasattr(conv, 'YatConv1D')
        assert hasattr(conv, 'YatConv2D')
    except ImportError as e:
        pytest.skip(f"Keras/TensorFlow dependencies not available: {e}")


@pytest.mark.skipif(
    True,
    reason="TensorFlow not available in test environment"
)
def test_yat_nmn_basic():
    """Test basic YatNMN functionality."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(units=10)
        
        # Build layer with input shape
        layer.build((None, 8))
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


@pytest.mark.skipif(
    True,
    reason="TensorFlow not available in test environment"
)
def test_yat_conv1d_basic():
    """Test basic YatConv1D functionality."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv1D
        
        # Create layer
        layer = YatConv1D(filters=16, kernel_size=3)
        
        # Build layer with input shape
        layer.build((None, 10, 8))
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 10, 8).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output shape for 'valid' padding: (batch, length-kernel_size+1, filters)
        assert output.shape == (4, 8, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


@pytest.mark.skipif(
    True,
    reason="TensorFlow not available in test environment"
)
def test_yat_conv2d_basic():
    """Test basic YatConv2D functionality."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv2D
        
        # Create layer
        layer = YatConv2D(filters=16, kernel_size=3)
        
        # Build layer with input shape
        layer.build((None, 32, 32, 3))
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output shape for 'valid' padding: (batch, height-kernel_size+1, width-kernel_size+1, filters)
        assert output.shape == (4, 30, 30, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


@pytest.mark.skipif(
    True,
    reason="TensorFlow not available in test environment"
)
def test_yat_conv2d_same_padding():
    """Test YatConv2D with same padding."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv2D
        
        # Create layer with same padding
        layer = YatConv2D(filters=16, kernel_size=3, padding='same')
        
        # Build layer with input shape
        layer.build((None, 32, 32, 3))
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output shape for 'same' padding: same as input spatial dims
        assert output.shape == (4, 32, 32, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


@pytest.mark.skipif(
    True,
    reason="TensorFlow not available in test environment"
)
def test_yat_nmn_no_bias():
    """Test YatNMN without bias."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
        
        # Create layer without bias
        layer = YatNMN(units=10, use_bias=False)
        
        # Build layer with input shape
        layer.build((None, 8))
        
        # Check that bias is None
        assert layer.bias is None
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


@pytest.mark.skipif(
    True,
    reason="TensorFlow not available in test environment"
)
def test_yat_nmn_epsilon():
    """Test YatNMN with custom epsilon."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
        
        # Create layer with custom epsilon
        layer = YatNMN(units=10, epsilon=1e-4)
        
        # Build layer with input shape
        layer.build((None, 8))
        
        # Check that epsilon is set
        assert layer.epsilon == 1e-4
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")