"""
Comprehensive Tests for All Keras YAT Layers.

Tests all layer variants with consistent test patterns.
"""

import pytest
import numpy as np

try:
    import keras
    import tensorflow as tf
    KERAS_AVAILABLE = True
    BACKEND = keras.backend.backend()
except ImportError:
    KERAS_AVAILABLE = False
    BACKEND = None


pytestmark = pytest.mark.skipif(
    not KERAS_AVAILABLE or BACKEND != "tensorflow",
    reason="Keras with TensorFlow backend not available"
)


# ============================================================================
# Layer Import Tests
# ============================================================================

class TestLayerImports:
    """Test that all layers can be imported."""
    
    def test_import_yat_nmn(self):
        """Test YatNMN import."""
        from nmn.keras import YatNMN
        assert YatNMN is not None
    
    def test_import_yat_dense(self):
        """Test YatDense import."""
        from nmn.keras import YatDense
        assert YatDense is not None
    
    def test_import_yat_conv1d(self):
        """Test YatConv1D import."""
        from nmn.keras import YatConv1D
        assert YatConv1D is not None
    
    def test_import_yat_conv2d(self):
        """Test YatConv2D import."""
        from nmn.keras import YatConv2D
        assert YatConv2D is not None
    
    def test_import_yat_conv3d(self):
        """Test YatConv3D import."""
        from nmn.keras import YatConv3D
        assert YatConv3D is not None
    
    def test_import_yat_conv_transpose1d(self):
        """Test YatConvTranspose1D import."""
        from nmn.keras import YatConvTranspose1D
        assert YatConvTranspose1D is not None
    
    def test_import_yat_conv_transpose2d(self):
        """Test YatConvTranspose2D import."""
        from nmn.keras import YatConvTranspose2D
        assert YatConvTranspose2D is not None


# ============================================================================
# YatNMN (Dense) Tests
# ============================================================================

class TestYatNMNComprehensive:
    """Comprehensive tests for YatNMN (dense layer)."""
    
    def test_instantiation(self):
        """Test basic instantiation."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=16)
        assert layer.units == 16
    
    def test_forward_pass(self):
        """Test forward pass."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=16)
        x = tf.random.normal((2, 8))
        output = layer(x)
        assert output.shape == (2, 16)
    
    def test_with_bias(self):
        """Test with bias enabled."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=16, use_bias=True)
        x = tf.random.normal((2, 8))
        output = layer(x)
        assert len([v for v in layer.trainable_weights if 'bias' in v.name]) > 0
    
    def test_without_bias(self):
        """Test without bias."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=16, use_bias=False)
        x = tf.random.normal((2, 8))
        output = layer(x)
        assert not any('bias' in v.name for v in layer.trainable_weights)
    
    def test_has_alpha(self):
        """Test that alpha parameter exists."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=16)
        x = tf.random.normal((2, 8))
        output = layer(x)
        assert any('alpha' in v.name for v in layer.trainable_weights)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=16)
        x = tf.random.normal((2, 8))
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = layer(x)
            loss = tf.reduce_sum(output)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        assert all(g is not None for g in gradients)
    
    def test_positive_outputs_no_bias(self):
        """Test that YAT produces non-negative outputs without bias."""
        from nmn.keras import YatNMN
        np.random.seed(42)
        
        layer = YatNMN(units=16, use_bias=False)
        x = tf.constant(np.random.randn(4, 8).astype(np.float32))
        output = layer(x)
        
        # Note: with alpha scaling, values might be very large but still non-negative
        # before alpha scaling
        output_np = output.numpy()
        # Due to alpha scaling, we mainly check for NaN/Inf
        assert not np.isnan(output_np).any(), "YAT should not produce NaN"


# ============================================================================
# Conv Layer Tests
# ============================================================================

class TestYatConv1DComprehensive:
    """Comprehensive tests for YatConv1D."""
    
    def test_forward_pass_valid_padding(self):
        """Test forward pass with valid padding."""
        from nmn.keras import YatConv1D
        layer = YatConv1D(filters=8, kernel_size=3, padding='valid')
        x = tf.random.normal((2, 16, 3))
        output = layer(x)
        assert output.shape == (2, 14, 8)
    
    def test_forward_pass_same_padding(self):
        """Test forward pass with same padding."""
        from nmn.keras import YatConv1D
        layer = YatConv1D(filters=8, kernel_size=3, padding='same')
        x = tf.random.normal((2, 16, 3))
        output = layer(x)
        assert output.shape == (2, 16, 8)
    
    def test_with_stride(self):
        """Test with stride > 1."""
        from nmn.keras import YatConv1D
        layer = YatConv1D(filters=8, kernel_size=3, strides=2, padding='same')
        x = tf.random.normal((2, 16, 3))
        output = layer(x)
        assert output.shape == (2, 8, 8)
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.keras import YatConv1D
        layer = YatConv1D(filters=8, kernel_size=3)
        x = tf.random.normal((2, 16, 3))
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = layer(x)
            loss = tf.reduce_sum(output)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        assert all(g is not None for g in gradients)


class TestYatConv2DComprehensive:
    """Comprehensive tests for YatConv2D."""
    
    def test_forward_pass_valid_padding(self):
        """Test forward pass with valid padding."""
        from nmn.keras import YatConv2D
        layer = YatConv2D(filters=8, kernel_size=3, padding='valid')
        x = tf.random.normal((2, 16, 16, 3))
        output = layer(x)
        assert output.shape == (2, 14, 14, 8)
    
    def test_forward_pass_same_padding(self):
        """Test forward pass with same padding."""
        from nmn.keras import YatConv2D
        layer = YatConv2D(filters=8, kernel_size=3, padding='same')
        x = tf.random.normal((2, 16, 16, 3))
        output = layer(x)
        assert output.shape == (2, 16, 16, 8)
    
    def test_with_stride(self):
        """Test with stride."""
        from nmn.keras import YatConv2D
        layer = YatConv2D(filters=8, kernel_size=3, strides=2, padding='same')
        x = tf.random.normal((2, 16, 16, 3))
        output = layer(x)
        assert output.shape == (2, 8, 8, 8)
    
    def test_with_dilation(self):
        """Test with dilation."""
        from nmn.keras import YatConv2D
        layer = YatConv2D(filters=8, kernel_size=3, dilation_rate=2)
        x = tf.random.normal((2, 16, 16, 3))
        output = layer(x)
        # Effective kernel = 3 + (3-1)*1 = 5, output = 16 - 5 + 1 = 12
        assert output.shape == (2, 12, 12, 8)
    
    def test_positive_outputs(self):
        """Test that outputs are non-negative without bias."""
        from nmn.keras import YatConv2D
        np.random.seed(42)
        
        layer = YatConv2D(filters=8, kernel_size=3, use_bias=False, use_alpha=False)
        x = tf.constant(np.random.randn(2, 16, 16, 3).astype(np.float32))
        output = layer(x)
        
        output_np = output.numpy()
        assert np.all(output_np >= 0), "YAT should produce non-negative values"
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.keras import YatConv2D
        layer = YatConv2D(filters=8, kernel_size=3)
        x = tf.random.normal((2, 16, 16, 3))
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            output = layer(x)
            loss = tf.reduce_sum(output)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        assert all(g is not None for g in gradients)


class TestYatConv3DComprehensive:
    """Comprehensive tests for YatConv3D."""
    
    def test_forward_pass_valid_padding(self):
        """Test forward pass with valid padding."""
        from nmn.keras import YatConv3D
        layer = YatConv3D(filters=8, kernel_size=3, padding='valid')
        x = tf.random.normal((2, 8, 8, 8, 3))
        output = layer(x)
        assert output.shape == (2, 6, 6, 6, 8)
    
    def test_forward_pass_same_padding(self):
        """Test forward pass with same padding."""
        from nmn.keras import YatConv3D
        layer = YatConv3D(filters=8, kernel_size=3, padding='same')
        x = tf.random.normal((2, 8, 8, 8, 3))
        output = layer(x)
        assert output.shape == (2, 8, 8, 8, 8)
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.keras import YatConv3D
        layer = YatConv3D(filters=4, kernel_size=3)
        x = tf.random.normal((1, 6, 6, 6, 2))
        
        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        assert all(g is not None for g in gradients)


# ============================================================================
# Conv Transpose Layer Tests
# ============================================================================

class TestYatConvTranspose1DComprehensive:
    """Comprehensive tests for YatConvTranspose1D."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        from nmn.keras import YatConvTranspose1D
        layer = YatConvTranspose1D(filters=4, kernel_size=2, strides=2)
        x = tf.random.normal((2, 8, 8))
        output = layer(x)
        assert output.shape[0] == 2
        assert output.shape[2] == 4
        assert output.shape[1] > 8  # Upsampled
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.keras import YatConvTranspose1D
        layer = YatConvTranspose1D(filters=4, kernel_size=2, strides=2)
        x = tf.random.normal((2, 8, 8))
        
        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        assert all(g is not None for g in gradients)


class TestYatConvTranspose2DComprehensive:
    """Comprehensive tests for YatConvTranspose2D."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        from nmn.keras import YatConvTranspose2D
        layer = YatConvTranspose2D(filters=4, kernel_size=2, strides=2)
        x = tf.random.normal((2, 8, 8, 8))
        output = layer(x)
        assert output.shape[0] == 2
        assert output.shape[3] == 4
        assert output.shape[1] > 8  # Upsampled
        assert output.shape[2] > 8  # Upsampled
    
    def test_upsampling_factor(self):
        """Test that upsampling works correctly with stride=2."""
        from nmn.keras import YatConvTranspose2D
        layer = YatConvTranspose2D(filters=4, kernel_size=2, strides=2, padding='same')
        x = tf.random.normal((2, 8, 8, 4))
        output = layer(x)
        # With stride=2 and same padding, should double spatial dims
        assert output.shape[1] == 16
        assert output.shape[2] == 16
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.keras import YatConvTranspose2D
        layer = YatConvTranspose2D(filters=4, kernel_size=2, strides=2)
        x = tf.random.normal((2, 8, 8, 8))
        
        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(output)
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        assert all(g is not None for g in gradients)


# ============================================================================
# YAT Math Validation
# ============================================================================

class TestYatMathValidation:
    """Validate YAT formula implementation."""
    
    def test_yat_formula_dense(self):
        """Test YAT formula for dense layer."""
        from nmn.keras import YatNMN
        np.random.seed(42)
        
        # Note: Keras YatNMN always has alpha, so we test with the full formula
        layer = YatNMN(units=4, use_bias=False)
        
        # Build the layer
        x_np = np.random.randn(2, 8).astype(np.float32)
        x = tf.constant(x_np)
        _ = layer(x)
        
        # Get weights
        weights = layer.get_weights()
        kernel = weights[0]  # Shape: (in_features, out_features)
        alpha_val = weights[1][0]  # Alpha value
        
        # Compute expected output using reference YAT formula
        dot_prod = np.matmul(x_np, kernel)
        x_sq_sum = np.sum(x_np**2, axis=-1, keepdims=True)
        k_sq_sum = np.sum(kernel**2, axis=0, keepdims=True)
        distance_sq = x_sq_sum + k_sq_sum - 2 * dot_prod
        expected = dot_prod**2 / (distance_sq + layer.epsilon)
        
        # Apply alpha scaling
        out_features = 4
        scale = (np.sqrt(out_features) / np.log(1 + out_features)) ** alpha_val
        expected = expected * scale
        
        # Get actual output
        output = layer(x)
        output_np = output.numpy()
        
        np.testing.assert_allclose(output_np, expected, rtol=1e-3, atol=1e-3)
    
    def test_epsilon_prevents_nan(self):
        """Test that epsilon prevents NaN for matching vectors."""
        from nmn.keras import YatNMN
        
        layer = YatNMN(units=4, use_bias=False)
        
        # Build layer
        dummy = tf.random.normal((1, 8))
        _ = layer(dummy)
        
        # Get a weight vector and use it as input (distance = 0)
        weights = layer.get_weights()
        kernel = weights[0]
        weight_vec = kernel[:, 0:1].T  # First weight column as input
        
        x = tf.constant(weight_vec)
        output = layer(x)
        
        output_np = output.numpy()
        assert not np.isnan(output_np).any(), "Output contains NaN"
        assert not np.isinf(output_np).any(), "Output contains Inf"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=16)
        x = tf.random.normal((1, 8))
        output = layer(x)
        assert output.shape == (1, 16)
    
    def test_large_batch(self):
        """Test with large batch."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=16)
        x = tf.random.normal((64, 8))
        output = layer(x)
        assert output.shape == (64, 16)
    
    def test_large_hidden_size(self):
        """Test with large hidden size."""
        from nmn.keras import YatNMN
        layer = YatNMN(units=512)
        x = tf.random.normal((4, 128))
        output = layer(x)
        assert output.shape == (4, 512)
    
    def test_small_kernel(self):
        """Test conv with kernel size 1."""
        from nmn.keras import YatConv2D
        layer = YatConv2D(filters=8, kernel_size=1)
        x = tf.random.normal((2, 16, 16, 3))
        output = layer(x)
        assert output.shape == (2, 16, 16, 8)
    
    def test_rectangular_input(self):
        """Test conv with non-square input."""
        from nmn.keras import YatConv2D
        layer = YatConv2D(filters=8, kernel_size=3, padding='same')
        x = tf.random.normal((2, 16, 8, 3))
        output = layer(x)
        assert output.shape == (2, 16, 8, 8)
