"""Comprehensive tests for Keras implementation."""

import pytest
import numpy as np


def test_keras_yat_nmn_model_compile():
    """Test that YatNMN can be used in a compiled model."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatNMN
        
        # Create a simple model
        model = tf.keras.Sequential([
            YatNMN(units=32),
            YatNMN(units=10)
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Create dummy data
        x = np.random.randn(32, 8).astype(np.float32)
        y = np.random.randn(32, 10).astype(np.float32)
        
        # Test training step
        history = model.fit(x, y, epochs=1, verbose=0)
        
        assert model is not None
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv_model_compile():
    """Test that YatConv can be used in a compiled model."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv2D
        
        # Create a simple model
        model = tf.keras.Sequential([
            YatConv2D(filters=32, kernel_size=3, input_shape=(32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Create dummy data
        x = np.random.randn(32, 32, 32, 3).astype(np.float32)
        y = np.random.randn(32, 10).astype(np.float32)
        
        # Test training step
        history = model.fit(x, y, epochs=1, verbose=0)
        
        assert model is not None
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_nmn_save_load():
    """Test that YatNMN can be saved and loaded."""
    try:
        import tensorflow as tf
        import tempfile
        import os
        from nmn.keras.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(units=10)
        layer.build((None, 8))
        
        # Create input
        dummy_input = tf.constant(np.random.randn(4, 8).astype(np.float32))
        
        # Forward pass
        output1 = layer(dummy_input)
        
        # Save weights
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "weights.weights.h5")
            model = tf.keras.Sequential([layer])
            model.save_weights(weights_path)
            
            # Create new layer and load weights
            new_layer = YatNMN(units=10)
            new_layer.build((None, 8))
            new_model = tf.keras.Sequential([new_layer])
            new_model.load_weights(weights_path)
            
            # Forward pass with loaded weights
            output2 = new_layer(dummy_input)
            
            # Outputs should be the same
            np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-5)
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv_gradient_flow():
    """Test gradient flow through YatConv layers."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv2D
        
        # Create layer
        layer = YatConv2D(filters=16, kernel_size=3)
        layer.build((None, 32, 32, 3))
        
        # Create input
        dummy_input = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        
        # Compute output with gradient tape
        with tf.GradientTape() as tape:
            output = layer(dummy_input)
            loss = tf.reduce_mean(output)
        
        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)
        
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")


def test_keras_yat_conv_all_dimensions():
    """Test all available convolution dimensions."""
    try:
        import tensorflow as tf
        from nmn.keras.conv import YatConv1D, YatConv2D
        
        # Test 1D
        layer1d = YatConv1D(filters=8, kernel_size=3)
        layer1d.build((None, 10, 3))
        input1d = tf.constant(np.random.randn(4, 10, 3).astype(np.float32))
        output1d = layer1d(input1d)
        assert output1d.shape[0] == 4
        assert output1d.shape[-1] == 8
        
        # Test 2D
        layer2d = YatConv2D(filters=8, kernel_size=3)
        layer2d.build((None, 32, 32, 3))
        input2d = tf.constant(np.random.randn(4, 32, 32, 3).astype(np.float32))
        output2d = layer2d(input2d)
        assert output2d.shape[0] == 4
        assert output2d.shape[-1] == 8
        
        # Note: YatConv3D is not implemented in Keras yet
        
    except ImportError:
        pytest.skip("Keras/TensorFlow dependencies not available")

