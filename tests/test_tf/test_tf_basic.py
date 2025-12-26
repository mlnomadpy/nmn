"""Tests for TensorFlow implementation."""

import pytest
import numpy as np


def test_tf_import():
    """Test that TensorFlow module can be imported."""
    try:
        from nmn.tf import nmn
        from nmn.tf import conv
        assert hasattr(nmn, 'YatNMN')
        assert hasattr(conv, 'YatConv1D')
        assert hasattr(conv, 'YatConv2D')
        assert hasattr(conv, 'YatConv3D')
    except ImportError as e:
        pytest.skip(f"TensorFlow dependencies not available: {e}")


def test_yat_nmn_basic():
    """Test basic TensorFlow YatNMN functionality."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(features=10)
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_yat_conv1d_basic():
    """Test basic TensorFlow YatConv1D functionality."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv1D
        
        # Create layer
        layer = YatConv1D(filters=16, kernel_size=3)
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 10, 8).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output shape for 'VALID' padding: (batch, length-kernel_size+1, filters)
        assert output.shape == (4, 8, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_yat_conv2d_basic():
    """Test basic TensorFlow YatConv2D functionality."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv2D
        
        # Create layer
        layer = YatConv2D(filters=16, kernel_size=(3, 3))
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output shape for 'VALID' padding: (batch, height-kernel_size+1, width-kernel_size+1, filters)
        assert output.shape == (4, 30, 30, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_yat_conv3d_basic():
    """Test basic TensorFlow YatConv3D functionality."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv3D
        
        # Create layer
        layer = YatConv3D(filters=16, kernel_size=(3, 3, 3))
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(2, 16, 16, 16, 3).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output shape for 'VALID' padding: (batch, depth-kernel_size+1, height-kernel_size+1, width-kernel_size+1, filters)
        assert output.shape == (2, 14, 14, 14, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_yat_conv2d_same_padding():
    """Test YatConv2D with SAME padding."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv2D
        
        # Create layer with SAME padding
        layer = YatConv2D(filters=16, kernel_size=(3, 3), padding='same')
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output shape for 'SAME' padding: same as input spatial dims
        assert output.shape == (4, 32, 32, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_yat_nmn_no_bias():
    """Test YatNMN without bias."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
        
        # Create layer without bias
        layer = YatNMN(features=10, use_bias=False)
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(dummy_input)
        
        assert output.shape == (4, 10)
        # Check that bias is None
        assert layer.bias is None
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_yat_nmn_custom_epsilon():
    """Test YatNMN with custom epsilon."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
        
        # Create layer with custom epsilon
        layer = YatNMN(features=10, epsilon=1e-4)
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(dummy_input)
        
        assert output.shape == (4, 10)
        # Check that epsilon is set
        assert layer.epsilon == 1e-4
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_yat_conv1d_strides():
    """Test YatConv1D with strides."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv1D
        
        # Create layer with stride=2
        layer = YatConv1D(filters=16, kernel_size=3, strides=2)
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 10, 8).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output length = (input_length - kernel_size + 1) // stride = (10 - 3 + 1) // 2 = 4
        assert output.shape == (4, 4, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_yat_conv2d_strides():
    """Test YatConv2D with strides."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv2D
        
        # Create layer with stride=(2, 2)
        layer = YatConv2D(filters=16, kernel_size=(3, 3), strides=(2, 2))
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        output = layer(dummy_input)
        
        # Expected output size = (input_size - kernel_size + 1) // stride = (32 - 3 + 1) // 2 = 15
        assert output.shape == (4, 15, 15, 16)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")