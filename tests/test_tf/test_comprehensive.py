"""Comprehensive tests for TensorFlow implementation."""

import pytest
import numpy as np


def test_tf_yat_conv_transpose():
    """Test YatConvTranspose layers if implemented."""
    try:
        import tensorflow as tf
        from nmn.tf import conv
        
        # Check if transpose conv is implemented
        if hasattr(conv, 'YatConvTranspose1D') or hasattr(conv, 'YatConvTranspose2D'):
            pytest.skip("YatConvTranspose not yet implemented in TensorFlow")
        
        # If implemented, add tests here
        assert False, "Update test when YatConvTranspose is implemented"
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")
    except AttributeError:
        # If not implemented, skip gracefully
        pytest.skip("YatConvTranspose not yet implemented in TensorFlow")


def test_tf_yat_nmn_gradient():
    """Test that YatNMN can compute gradients."""
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(features=10)
        
        # Create input
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        
        # Compute output with gradient tape
        with tf.GradientTape() as tape:
            tape.watch(dummy_input)
            output = layer(dummy_input)
            loss = tf.reduce_mean(output)
        
        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)
        
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_gradient():
    """Test that YatConv can compute gradients."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv2D
        
        # Create layer
        layer = YatConv2D(filters=16, kernel_size=(3, 3))
        
        # Create input
        dummy_input = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        
        # Compute output with gradient tape
        with tf.GradientTape() as tape:
            tape.watch(dummy_input)
            output = layer(dummy_input)
            loss = tf.reduce_mean(output)
        
        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)
        
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_nmn_model_save_load():
    """Test that YatNMN can be saved and loaded."""
    try:
        import tensorflow as tf
        import tempfile
        import os
        from nmn.tf.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(features=10)
        
        # Create input
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        
        # Forward pass
        output1 = layer(dummy_input)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model")
            layer.save_weights(model_path)
            
            # Create new layer and load weights
            new_layer = YatNMN(features=10)
            new_layer.load_weights(model_path)
            
            # Forward pass with loaded weights
            output2 = new_layer(dummy_input)
            
            # Outputs should be the same
            np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-5)
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_all_dimensions():
    """Test all convolution dimensions."""
    try:
        import tensorflow as tf
        from nmn.tf.conv import YatConv1D, YatConv2D, YatConv3D
        
        # Test 1D
        layer1d = YatConv1D(filters=8, kernel_size=3)
        input1d = tf.constant(np.random.randn(4, 10, 3).astype(np.float32))
        output1d = layer1d(input1d)
        assert output1d.shape[0] == 4
        assert output1d.shape[-1] == 8
        
        # Test 2D
        layer2d = YatConv2D(filters=8, kernel_size=(3, 3))
        input2d = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        output2d = layer2d(input2d)
        assert output2d.shape[0] == 4
        assert output2d.shape[-1] == 8
        
        # Test 3D
        layer3d = YatConv3D(filters=8, kernel_size=(3, 3, 3))
        input3d = tf.constant(np.random.randn(2, 16, 16, 16, 3).astype(np.float32))
        output3d = layer3d(input3d)
        assert output3d.shape[0] == 2
        assert output3d.shape[-1] == 8
        
    except ImportError:
        pytest.skip("TensorFlow dependencies not available")




