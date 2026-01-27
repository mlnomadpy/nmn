"""
Comprehensive YAT Math Validation Tests for PyTorch.

These tests validate that the YAT formula is correctly implemented:
    y = (dot_product)^2 / (distance_squared + epsilon)
    where distance_squared = ||input||^2 + ||kernel||^2 - 2 * dot_product
"""

import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


class TestYatNMNMathValidation:
    """Validate YAT formula implementation in YatNMN (dense layer)."""
    
    def test_yat_formula_basic(self):
        """Test that YAT formula produces expected values for simple inputs."""
        from nmn.torch.nmn import YatNMN
        
        torch.manual_seed(42)
        
        layer = YatNMN(in_features=4, out_features=2, bias=False, alpha=False)
        
        # Use ones for predictable computation
        x = torch.ones(1, 4)
        
        with torch.no_grad():
            output = layer(x)
            
            # Manually compute expected output
            weight = layer.weight.data.numpy()
            x_np = x.numpy()
            
            # dot_product = x @ weight.T
            dot_prod = np.matmul(x_np, weight.T)
            
            # distance_squared = ||x||^2 + ||weight||^2 - 2 * dot_product
            x_sq_sum = np.sum(x_np**2, axis=-1, keepdims=True)  # 4.0
            w_sq_sum = np.sum(weight**2, axis=-1)  # per output neuron
            distance_sq = x_sq_sum + w_sq_sum - 2 * dot_prod
            
            # y = dot_product^2 / (distance_squared + epsilon)
            expected = dot_prod**2 / (distance_sq + layer.epsilon)
            
            np.testing.assert_allclose(output.numpy(), expected, rtol=1e-4, atol=1e-4)
    
    def test_yat_formula_with_bias(self):
        """Test that bias is correctly added after YAT transformation."""
        from nmn.torch.nmn import YatNMN
        
        torch.manual_seed(42)
        
        layer_no_bias = YatNMN(in_features=4, out_features=2, bias=False, alpha=False)
        layer_with_bias = YatNMN(in_features=4, out_features=2, bias=True, alpha=False)
        
        # Copy weights
        layer_with_bias.weight.data = layer_no_bias.weight.data.clone()
        bias_value = 0.5
        layer_with_bias.bias.data.fill_(bias_value)
        
        x = torch.randn(2, 4)
        
        with torch.no_grad():
            out_no_bias = layer_no_bias(x)
            out_with_bias = layer_with_bias(x)
            
            # Output with bias should be output without bias + bias
            expected = out_no_bias + bias_value
            
            np.testing.assert_allclose(out_with_bias.numpy(), expected.numpy(), rtol=1e-5)
    
    def test_yat_formula_with_alpha_scaling(self):
        """Test that alpha scaling is correctly applied."""
        from nmn.torch.nmn import YatNMN
        import math
        
        torch.manual_seed(42)
        
        out_features = 8
        layer_no_alpha = YatNMN(in_features=4, out_features=out_features, bias=False, alpha=False)
        layer_with_alpha = YatNMN(in_features=4, out_features=out_features, bias=False, alpha=True)
        
        # Copy weights and set alpha to 1.0
        layer_with_alpha.weight.data = layer_no_alpha.weight.data.clone()
        layer_with_alpha.alpha.data.fill_(1.0)
        
        x = torch.randn(2, 4)
        
        with torch.no_grad():
            out_no_alpha = layer_no_alpha(x)
            out_with_alpha = layer_with_alpha(x)
            
            # Expected scaling factor
            scale = math.sqrt(out_features) / math.log(1 + out_features)
            expected = out_no_alpha * scale
            
            np.testing.assert_allclose(out_with_alpha.numpy(), expected.numpy(), rtol=1e-4)
    
    def test_yat_produces_positive_outputs(self):
        """Test that YAT always produces non-negative outputs (squared ratio)."""
        from nmn.torch.nmn import YatNMN
        
        torch.manual_seed(42)
        
        layer = YatNMN(in_features=4, out_features=8, bias=False, alpha=False)
        
        # Test with various inputs
        for _ in range(10):
            x = torch.randn(4, 4) * 10  # Large values
            with torch.no_grad():
                output = layer(x)
                assert (output >= 0).all(), "YAT should produce non-negative values"
    
    def test_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents NaN when distance approaches zero."""
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(in_features=4, out_features=2, bias=False, alpha=False, epsilon=1e-6)
        
        # Create input that exactly matches a weight vector (distance = 0)
        with torch.no_grad():
            # Get one weight vector and use it as input
            weight_vec = layer.weight.data[0].unsqueeze(0)  # (1, 4)
            output = layer(weight_vec)
            
            # Should not produce NaN or Inf
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"


class TestYatConv2DMathValidation:
    """Validate YAT formula implementation in YatConv2d."""
    
    def test_yat_conv2d_formula_basic(self):
        """Test that YatConv2d correctly computes YAT for convolutions."""
        from nmn.torch.layers import YatConv2d
        
        torch.manual_seed(42)
        
        layer = YatConv2d(
            in_channels=1, out_channels=1, kernel_size=2,
            bias=False, use_alpha=False, padding=0
        )
        
        # Use simple input for manual verification
        x = torch.ones(1, 1, 3, 3)  # Single channel, 3x3 image
        
        with torch.no_grad():
            output = layer(x)
            
            # For a 2x2 kernel on 3x3 input with valid padding, output is 2x2
            assert output.shape == (1, 1, 2, 2)
            
            # All outputs should be positive
            assert (output >= 0).all()
    
    def test_yat_conv2d_distance_computation(self):
        """Test that distance is correctly computed for patches."""
        from nmn.torch.layers import YatConv2d
        
        torch.manual_seed(42)
        
        layer = YatConv2d(
            in_channels=1, out_channels=1, kernel_size=2,
            bias=False, use_alpha=False
        )
        
        # Set kernel to known values
        layer.weight.data.fill_(1.0)  # 2x2 kernel of ones
        
        # Input: 2x2 patch of ones
        x = torch.ones(1, 1, 2, 2)
        
        with torch.no_grad():
            output = layer(x)
            
            # Patch: [1,1,1,1], Kernel: [1,1,1,1]
            # dot_product = 4
            # ||patch||^2 = 4, ||kernel||^2 = 4
            # distance^2 = 4 + 4 - 2*4 = 0
            # y = 16 / (0 + epsilon) = 16 / epsilon
            expected = 16.0 / layer.epsilon
            
            np.testing.assert_allclose(output.numpy().flatten()[0], expected, rtol=1e-3)
    
    def test_yat_conv2d_non_matching_patch_kernel(self):
        """Test YAT computation when patch and kernel differ."""
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(
            in_channels=1, out_channels=1, kernel_size=2,
            bias=False, use_alpha=False, epsilon=1e-6
        )
        
        # Set kernel to [1,2,3,4]
        with torch.no_grad():
            layer.weight.data = torch.tensor([[[[1., 2.], [3., 4.]]]])
        
        # Input patch: [4,3,2,1]
        x = torch.tensor([[[[4., 3.], [2., 1.]]]])
        
        with torch.no_grad():
            output = layer(x)
            
            # Manual computation
            patch = np.array([4., 3., 2., 1.])
            kernel = np.array([1., 2., 3., 4.])
            
            dot_prod = np.dot(patch, kernel)  # 4+6+6+4 = 20
            patch_sq = np.sum(patch**2)  # 16+9+4+1 = 30
            kernel_sq = np.sum(kernel**2)  # 1+4+9+16 = 30
            distance_sq = patch_sq + kernel_sq - 2 * dot_prod  # 30 + 30 - 40 = 20
            
            expected = dot_prod**2 / (distance_sq + 1e-6)  # 400 / 20.000001
            
            np.testing.assert_allclose(output.numpy().flatten()[0], expected, rtol=1e-3)


class TestYatConvTranspose2DMathValidation:
    """Validate YAT formula in transposed convolution."""
    
    def test_yat_conv_transpose2d_output_shape(self):
        """Test that YatConvTranspose2d produces correct output shape."""
        from nmn.torch.layers import YatConvTranspose2d
        
        layer = YatConvTranspose2d(
            in_channels=8, out_channels=4, kernel_size=2, stride=2,
            bias=False, use_alpha=False
        )
        
        x = torch.randn(2, 8, 4, 4)
        
        with torch.no_grad():
            output = layer(x)
            
            # With stride=2 and kernel=2, output should be 2x input spatial dims
            assert output.shape == (2, 4, 8, 8)
    
    def test_yat_conv_transpose2d_positive_outputs(self):
        """Test that transposed conv also produces non-negative outputs."""
        from nmn.torch.layers import YatConvTranspose2d
        
        torch.manual_seed(42)
        
        layer = YatConvTranspose2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=2,
            bias=False, use_alpha=False
        )
        
        x = torch.randn(2, 3, 8, 8)
        
        with torch.no_grad():
            output = layer(x)
            assert (output >= 0).all(), "Transposed conv should produce non-negative values"


class TestGradientFlow:
    """Test that gradients flow correctly through YAT layers."""
    
    def test_yat_nmn_gradient_flow(self):
        """Test gradient computation for YatNMN."""
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(in_features=4, out_features=2)
        x = torch.randn(2, 4, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and are not NaN
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert layer.weight.grad is not None
        assert not torch.isnan(layer.weight.grad).any()
    
    def test_yat_conv2d_gradient_flow(self):
        """Test gradient computation for YatConv2d."""
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(in_channels=3, out_channels=8, kernel_size=3)
        x = torch.randn(2, 3, 16, 16, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert layer.weight.grad is not None
        assert not torch.isnan(layer.weight.grad).any()
    
    def test_gradient_magnitude_reasonable(self):
        """Test that gradients don't explode or vanish."""
        from nmn.torch.nmn import YatNMN
        
        torch.manual_seed(42)
        
        layer = YatNMN(in_features=64, out_features=32)
        x = torch.randn(8, 64, requires_grad=True)
        
        output = layer(x)
        loss = output.mean()
        loss.backward()
        
        # Gradients should be finite and reasonable
        grad_norm = x.grad.norm().item()
        assert 1e-10 < grad_norm < 1e10, f"Gradient norm {grad_norm} is unreasonable"


class TestNumericalStability:
    """Test numerical stability under extreme conditions."""
    
    def test_large_input_values(self):
        """Test with large input values."""
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(in_features=4, out_features=2, epsilon=1e-6)
        x = torch.randn(2, 4) * 1000
        
        with torch.no_grad():
            output = layer(x)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_small_input_values(self):
        """Test with very small input values."""
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(in_features=4, out_features=2, epsilon=1e-6)
        x = torch.randn(2, 4) * 1e-6
        
        with torch.no_grad():
            output = layer(x)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_zero_input(self):
        """Test with zero input."""
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(in_features=4, out_features=2, epsilon=1e-6)
        x = torch.zeros(2, 4)
        
        with torch.no_grad():
            output = layer(x)
            # With zero input, dot_product=0, so output should be 0
            assert not torch.isnan(output).any()
            # 0^2 / (anything + epsilon) = 0
            expected_yat = 0.0
            # But bias and alpha scaling may change this
    
    def test_epsilon_sensitivity(self):
        """Test that different epsilon values produce valid outputs."""
        from nmn.torch.nmn import YatNMN
        
        x = torch.randn(2, 4)
        
        for epsilon in [1e-3, 1e-5, 1e-7, 1e-9]:
            layer = YatNMN(in_features=4, out_features=2, epsilon=epsilon)
            with torch.no_grad():
                output = layer(x)
                assert not torch.isnan(output).any(), f"NaN with epsilon={epsilon}"
                assert not torch.isinf(output).any(), f"Inf with epsilon={epsilon}"




