"""
Comprehensive Tests for All Flax Linen YAT Layers.

Tests all layer variants with consistent test patterns.
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX/Flax not available")


# ============================================================================
# Layer Import Tests
# ============================================================================

class TestLayerImports:
    """Test that all layers can be imported."""
    
    def test_import_yat_nmn(self):
        """Test YatNMN import."""
        from nmn.linen import YatNMN
        assert YatNMN is not None
    
    def test_import_yat_conv1d(self):
        """Test YatConv1D import."""
        from nmn.linen import YatConv1D
        assert YatConv1D is not None
    
    def test_import_yat_conv2d(self):
        """Test YatConv2D import."""
        from nmn.linen import YatConv2D
        assert YatConv2D is not None
    
    def test_import_yat_conv3d(self):
        """Test YatConv3D import."""
        from nmn.linen import YatConv3D
        assert YatConv3D is not None


# ============================================================================
# YatNMN (Dense) Tests
# ============================================================================

class TestYatNMNComprehensive:
    """Comprehensive tests for Linen YatNMN (dense layer)."""
    
    def test_instantiation(self):
        """Test basic instantiation."""
        from nmn.linen import YatNMN
        layer = YatNMN(features=16)
        assert layer.features == 16
    
    def test_forward_pass(self):
        """Test forward pass."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16)
        x = jnp.ones((2, 8))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 16)
    
    def test_with_bias(self):
        """Test with bias enabled."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16, use_bias=True)
        x = jnp.ones((2, 8))
        params = layer.init(key, x)
        assert 'bias' in params['params']
    
    def test_without_bias(self):
        """Test without bias."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16, use_bias=False)
        x = jnp.ones((2, 8))
        params = layer.init(key, x)
        assert 'bias' not in params['params']
    
    def test_with_alpha(self):
        """Test with alpha scaling."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16, use_alpha=True)
        x = jnp.ones((2, 8))
        params = layer.init(key, x)
        assert 'alpha' in params['params']
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16)
        x = jnp.ones((2, 8))
        params = layer.init(key, x)
        
        def loss_fn(params):
            return jnp.sum(layer.apply(params, x))
        
        grads = jax.grad(loss_fn)(params)
        assert 'kernel' in grads['params']
    
    def test_positive_outputs_no_bias(self):
        """Test that YAT produces non-negative outputs without bias."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(42)
        
        layer = YatNMN(features=16, use_bias=False, use_alpha=False)
        x = jax.random.normal(key, (4, 8))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        assert jnp.all(output >= 0), "YAT should produce non-negative values"


# ============================================================================
# Conv Layer Tests
# ============================================================================

class TestYatConv1DComprehensive:
    """Comprehensive tests for Linen YatConv1D."""
    
    def test_forward_pass_valid_padding(self):
        """Test forward pass with valid padding."""
        from nmn.linen import YatConv1D
        key = jax.random.PRNGKey(0)
        layer = YatConv1D(features=8, kernel_size=(3,), padding='VALID')
        x = jnp.ones((2, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 14, 8)
    
    def test_forward_pass_same_padding(self):
        """Test forward pass with same padding."""
        from nmn.linen import YatConv1D
        key = jax.random.PRNGKey(0)
        layer = YatConv1D(features=8, kernel_size=(3,), padding='SAME')
        x = jnp.ones((2, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 16, 8)
    
    def test_with_stride(self):
        """Test with stride > 1."""
        from nmn.linen import YatConv1D
        key = jax.random.PRNGKey(0)
        layer = YatConv1D(features=8, kernel_size=(3,), strides=(2,), padding='SAME')
        x = jnp.ones((2, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 8, 8)
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.linen import YatConv1D
        key = jax.random.PRNGKey(0)
        layer = YatConv1D(features=8, kernel_size=(3,))
        x = jnp.ones((2, 16, 3))
        params = layer.init(key, x)
        
        def loss_fn(params):
            return jnp.sum(layer.apply(params, x))
        
        grads = jax.grad(loss_fn)(params)
        assert 'kernel' in grads['params']


class TestYatConv2DComprehensive:
    """Comprehensive tests for Linen YatConv2D."""
    
    def test_forward_pass_valid_padding(self):
        """Test forward pass with valid padding."""
        from nmn.linen import YatConv2D
        key = jax.random.PRNGKey(0)
        layer = YatConv2D(features=8, kernel_size=(3, 3), padding='VALID')
        x = jnp.ones((2, 16, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 14, 14, 8)
    
    def test_forward_pass_same_padding(self):
        """Test forward pass with same padding."""
        from nmn.linen import YatConv2D
        key = jax.random.PRNGKey(0)
        layer = YatConv2D(features=8, kernel_size=(3, 3), padding='SAME')
        x = jnp.ones((2, 16, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 16, 16, 8)
    
    def test_with_stride(self):
        """Test with stride."""
        from nmn.linen import YatConv2D
        key = jax.random.PRNGKey(0)
        layer = YatConv2D(features=8, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        x = jnp.ones((2, 16, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 8, 8, 8)
    
    def test_with_dilation(self):
        """Test with dilation."""
        from nmn.linen import YatConv2D
        key = jax.random.PRNGKey(0)
        layer = YatConv2D(features=8, kernel_size=(3, 3), kernel_dilation=(2, 2))
        x = jnp.ones((2, 16, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        # Effective kernel = 3 + (3-1)*1 = 5, output = 16 - 5 + 1 = 12
        assert output.shape == (2, 12, 12, 8)
    
    def test_positive_outputs(self):
        """Test that outputs are non-negative without bias."""
        from nmn.linen import YatConv2D
        key = jax.random.PRNGKey(42)
        
        layer = YatConv2D(features=8, kernel_size=(3, 3), use_bias=False, use_alpha=False)
        x = jax.random.normal(key, (2, 16, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        assert jnp.all(output >= 0), "YAT should produce non-negative values"
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.linen import YatConv2D
        key = jax.random.PRNGKey(0)
        layer = YatConv2D(features=8, kernel_size=(3, 3))
        x = jnp.ones((2, 16, 16, 3))
        params = layer.init(key, x)
        
        def loss_fn(params):
            return jnp.sum(layer.apply(params, x))
        
        grads = jax.grad(loss_fn)(params)
        assert 'kernel' in grads['params']


class TestYatConv3DComprehensive:
    """Comprehensive tests for Linen YatConv3D."""
    
    def test_forward_pass_valid_padding(self):
        """Test forward pass with valid padding."""
        from nmn.linen import YatConv3D
        key = jax.random.PRNGKey(0)
        layer = YatConv3D(features=8, kernel_size=(3, 3, 3), padding='VALID')
        x = jnp.ones((2, 8, 8, 8, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 6, 6, 6, 8)
    
    def test_forward_pass_same_padding(self):
        """Test forward pass with same padding."""
        from nmn.linen import YatConv3D
        key = jax.random.PRNGKey(0)
        layer = YatConv3D(features=8, kernel_size=(3, 3, 3), padding='SAME')
        x = jnp.ones((2, 8, 8, 8, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 8, 8, 8, 8)
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.linen import YatConv3D
        key = jax.random.PRNGKey(0)
        layer = YatConv3D(features=4, kernel_size=(3, 3, 3))
        x = jnp.ones((1, 6, 6, 6, 2))
        params = layer.init(key, x)
        
        def loss_fn(params):
            return jnp.sum(layer.apply(params, x))
        
        grads = jax.grad(loss_fn)(params)
        assert 'kernel' in grads['params']


# ============================================================================
# YAT Math Validation
# ============================================================================

class TestYatMathValidation:
    """Validate YAT formula implementation."""
    
    def test_yat_formula_dense(self):
        """Test YAT formula for dense layer."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(42)
        
        layer = YatNMN(features=4, use_bias=False, use_alpha=False)
        
        # Build the layer
        x_np = np.random.randn(2, 8).astype(np.float32)
        x = jnp.array(x_np)
        params = layer.init(key, x)
        
        # Get weights - Linen YatNMN kernel is (features, in_features)
        kernel_t = np.array(params['params']['kernel'])  # Shape: (features, in_features)
        kernel = kernel_t.T  # Transpose to (in_features, features)
        
        # Compute expected output using reference YAT formula
        dot_prod = np.matmul(x_np, kernel)  # (batch, features)
        x_sq_sum = np.sum(x_np**2, axis=-1, keepdims=True)  # (batch, 1)
        k_sq_sum = np.sum(kernel_t**2, axis=-1)  # (features,) - sum over in_features
        distance_sq = x_sq_sum + k_sq_sum - 2 * dot_prod
        expected = dot_prod**2 / (distance_sq + layer.epsilon)
        
        # Get actual output
        output = layer.apply(params, x)
        output_np = np.array(output)
        
        np.testing.assert_allclose(output_np, expected, rtol=1e-3, atol=1e-3)
    
    def test_epsilon_prevents_nan(self):
        """Test that epsilon prevents NaN for matching vectors."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        
        layer = YatNMN(features=4, use_bias=False, use_alpha=False)
        
        # Build layer with consistent input size
        dummy = jnp.ones((1, 8))
        params = layer.init(key, dummy)
        
        # Get a weight vector and use it as input (distance = 0)
        # Linen YatNMN kernel is (features, in_features), so first row is a weight vector
        kernel = np.array(params['params']['kernel'])  # (features, in_features)
        weight_vec = kernel[0:1, :]  # First weight row as input (1, in_features)
        
        x = jnp.array(weight_vec)
        output = layer.apply(params, x)
        
        output_np = np.array(output)
        assert not np.isnan(output_np).any(), "Output contains NaN"
        assert not np.isinf(output_np).any(), "Output contains Inf"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16)
        x = jnp.ones((1, 8))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (1, 16)
    
    def test_large_batch(self):
        """Test with large batch."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16)
        x = jnp.ones((64, 8))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (64, 16)
    
    def test_small_kernel(self):
        """Test conv with kernel size 1."""
        from nmn.linen import YatConv2D
        key = jax.random.PRNGKey(0)
        layer = YatConv2D(features=8, kernel_size=(1, 1))
        x = jnp.ones((2, 16, 16, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 16, 16, 8)
    
    def test_rectangular_input(self):
        """Test conv with non-square input."""
        from nmn.linen import YatConv2D
        key = jax.random.PRNGKey(0)
        layer = YatConv2D(features=8, kernel_size=(3, 3), padding='SAME')
        x = jnp.ones((2, 16, 8, 3))
        params = layer.init(key, x)
        output = layer.apply(params, x)
        assert output.shape == (2, 16, 8, 8)
    
    def test_jit_compatibility(self):
        """Test that layers work with JAX JIT compilation."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16)
        x = jnp.ones((2, 8))
        params = layer.init(key, x)
        
        @jax.jit
        def forward(params, x):
            return layer.apply(params, x)
        
        output = forward(params, x)
        assert output.shape == (2, 16)
    
    def test_vmap_compatibility(self):
        """Test that layers work with JAX vmap."""
        from nmn.linen import YatNMN
        key = jax.random.PRNGKey(0)
        layer = YatNMN(features=16)
        x = jnp.ones((2, 8))
        params = layer.init(key, x)
        
        # vmap over batch dimension
        batched_x = jnp.ones((4, 2, 8))  # Extra batch dimension
        
        @jax.vmap
        def batched_forward(x):
            return layer.apply(params, x)
        
        output = batched_forward(batched_x)
        assert output.shape == (4, 2, 16)

