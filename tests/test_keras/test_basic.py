"""Tests for Keras implementation."""

import pytest
import numpy as np


def test_keras_import():
    """Test that Keras module can be imported."""
    try:
        from nmn.keras import nmn
        assert hasattr(nmn, 'YatDense')
    except ImportError as e:
        pytest.skip(f"Keras/TensorFlow dependencies not available: {e}")


@pytest.mark.skipif(
    True,
    reason="TensorFlow not available in test environment"
)
def test_yat_dense_basic():
    """Test basic YatDense functionality."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatDense
        
        # Create layer
        layer = YatDense(units=10)
        
        # Build layer with input shape
        layer.build((None, 8))
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")