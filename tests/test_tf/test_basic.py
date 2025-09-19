"""Tests for TensorFlow implementation."""

import pytest
import numpy as np


def test_tf_import():
    """Test that TensorFlow module can be imported."""
    try:
        from nmn.tf import nmn
        assert True
    except ImportError as e:
        pytest.skip(f"TensorFlow dependencies not available: {e}")


@pytest.mark.skipif(
    True,
    reason="TensorFlow not available in test environment"
)
def test_tf_basic_functionality():
    """Test basic TensorFlow NMN functionality."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatDense
        
        # Create layer
        layer = YatDense(units=10)
        
        # Test forward pass
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")