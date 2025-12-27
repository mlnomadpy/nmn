"""
Comprehensive Tests for All PyTorch YAT Layers.

Tests all layer variants with consistent test patterns.
"""

import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


# ============================================================================
# Layer Configurations for Parametrized Tests
# ============================================================================

CONV_LAYER_CONFIGS = [
    ("YatConv1d", {"in_channels": 3, "out_channels": 8, "kernel_size": 3}, (2, 3, 16), (2, 8, 14)),
    ("YatConv2d", {"in_channels": 3, "out_channels": 8, "kernel_size": 3}, (2, 3, 16, 16), (2, 8, 14, 14)),
    ("YatConv3d", {"in_channels": 3, "out_channels": 8, "kernel_size": 3}, (2, 3, 8, 8, 8), (2, 8, 6, 6, 6)),
]

CONV_TRANSPOSE_CONFIGS = [
    ("YatConvTranspose1d", {"in_channels": 8, "out_channels": 3, "kernel_size": 2, "stride": 2}, (2, 8, 8), (2, 3, 16)),
    ("YatConvTranspose2d", {"in_channels": 8, "out_channels": 3, "kernel_size": 2, "stride": 2}, (2, 8, 8, 8), (2, 3, 16, 16)),
    ("YatConvTranspose3d", {"in_channels": 8, "out_channels": 3, "kernel_size": 2, "stride": 2}, (2, 8, 4, 4, 4), (2, 3, 8, 8, 8)),
]


def get_layer_class(layer_name):
    """Get layer class by name."""
    from nmn.torch import layers
    return getattr(layers, layer_name)


# ============================================================================
# Parametrized Conv Layer Tests
# ============================================================================

class TestConvLayersComprehensive:
    """Comprehensive tests for all conv layer variants."""
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_LAYER_CONFIGS)
    def test_output_shape(self, layer_name, kwargs, input_shape, expected_shape):
        """Test that each conv layer produces correct output shape."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs)
        
        x = torch.randn(*input_shape)
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == expected_shape, f"{layer_name} shape mismatch"
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_LAYER_CONFIGS)
    def test_gradient_flow(self, layer_name, kwargs, input_shape, expected_shape):
        """Test gradient computation for each conv layer."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs)
        
        x = torch.randn(*input_shape, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert layer.weight.grad is not None
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_LAYER_CONFIGS)
    def test_with_alpha(self, layer_name, kwargs, input_shape, expected_shape):
        """Test each layer with alpha scaling enabled."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs, use_alpha=True)
        
        x = torch.randn(*input_shape)
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == expected_shape
        assert layer.alpha is not None
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_LAYER_CONFIGS)
    def test_without_alpha(self, layer_name, kwargs, input_shape, expected_shape):
        """Test each layer without alpha scaling."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs, use_alpha=False)
        
        x = torch.randn(*input_shape)
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == expected_shape
        assert layer.alpha is None
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_LAYER_CONFIGS)
    def test_with_bias(self, layer_name, kwargs, input_shape, expected_shape):
        """Test each layer with bias."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs, bias=True)
        
        x = torch.randn(*input_shape)
        with torch.no_grad():
            output = layer(x)
        
        assert layer.bias is not None
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_LAYER_CONFIGS)
    def test_without_bias(self, layer_name, kwargs, input_shape, expected_shape):
        """Test each layer without bias."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs, bias=False)
        
        x = torch.randn(*input_shape)
        with torch.no_grad():
            output = layer(x)
        
        assert layer.bias is None
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_LAYER_CONFIGS)
    def test_positive_outputs(self, layer_name, kwargs, input_shape, expected_shape):
        """Test that YAT produces non-negative outputs (before bias)."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs, bias=False, use_alpha=False)
        
        torch.manual_seed(42)
        x = torch.randn(*input_shape)
        
        with torch.no_grad():
            output = layer(x)
        
        # YAT formula produces non-negative values
        assert (output >= 0).all(), f"{layer_name} produced negative values"


# ============================================================================
# Parametrized Conv Transpose Layer Tests
# ============================================================================

class TestConvTransposeLayersComprehensive:
    """Comprehensive tests for all conv transpose layer variants."""
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_TRANSPOSE_CONFIGS)
    def test_output_shape(self, layer_name, kwargs, input_shape, expected_shape):
        """Test that each transpose layer produces correct output shape."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs)
        
        x = torch.randn(*input_shape)
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == expected_shape, f"{layer_name} shape mismatch"
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_TRANSPOSE_CONFIGS)
    def test_gradient_flow(self, layer_name, kwargs, input_shape, expected_shape):
        """Test gradient computation for each transpose layer."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs)
        
        x = torch.randn(*input_shape, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    @pytest.mark.parametrize("layer_name,kwargs,input_shape,expected_shape", CONV_TRANSPOSE_CONFIGS)
    def test_upsampling_effect(self, layer_name, kwargs, input_shape, expected_shape):
        """Test that transpose conv upsamples spatial dimensions."""
        LayerClass = get_layer_class(layer_name)
        layer = LayerClass(**kwargs)
        
        x = torch.randn(*input_shape)
        with torch.no_grad():
            output = layer(x)
        
        # Spatial dimensions should be larger in output
        input_spatial = input_shape[2:]
        output_spatial = output.shape[2:]
        
        for i, o in zip(input_spatial, output_spatial):
            assert o >= i, f"{layer_name} should upsample, but {i} -> {o}"


# ============================================================================
# DropConnect Tests
# ============================================================================

class TestDropConnect:
    """Test DropConnect functionality across layers."""
    
    def test_yat_conv2d_dropconnect_training_vs_eval(self):
        """Test that DropConnect only applies during training."""
        from nmn.torch.layers import YatConv2d
        
        torch.manual_seed(42)
        
        layer = YatConv2d(
            in_channels=3, out_channels=8, kernel_size=3,
            use_dropconnect=True, drop_rate=0.5
        )
        
        x = torch.randn(2, 3, 16, 16)
        
        # Training mode - should have stochastic behavior
        layer.train()
        torch.manual_seed(1)
        out1 = layer(x)
        torch.manual_seed(2)
        out2 = layer(x)
        
        # With 50% dropout, outputs should differ
        assert not torch.allclose(out1, out2, atol=1e-5)
        
        # Eval mode with deterministic=True - should be consistent
        layer.eval()
        with torch.no_grad():
            out3 = layer(x, deterministic=True)
            out4 = layer(x, deterministic=True)
        
        assert torch.allclose(out3, out4, atol=1e-6)
    
    def test_dropconnect_disabled(self):
        """Test that disabled DropConnect produces deterministic outputs."""
        from nmn.torch.layers import YatConv2d
        
        torch.manual_seed(42)
        
        layer = YatConv2d(
            in_channels=3, out_channels=8, kernel_size=3,
            use_dropconnect=False
        )
        layer.train()
        
        x = torch.randn(2, 3, 16, 16)
        
        out1 = layer(x)
        out2 = layer(x)
        
        assert torch.allclose(out1, out2, atol=1e-6)


# ============================================================================
# Device and Dtype Tests
# ============================================================================

class TestDeviceAndDtype:
    """Test device and dtype handling."""
    
    def test_float16_support(self):
        """Test layer works with float16."""
        from nmn.torch.nmn import YatNMN
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for float16 test")
        
        layer = YatNMN(in_features=4, out_features=2, dtype=torch.float16).cuda()
        x = torch.randn(2, 4, dtype=torch.float16, device='cuda')
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.dtype == torch.float16
    
    def test_float64_support(self):
        """Test layer works with float64."""
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(in_features=4, out_features=2, dtype=torch.float64)
        x = torch.randn(2, 4, dtype=torch.float64)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.dtype == torch.float64
    
    def test_cuda_support(self):
        """Test layer works on CUDA."""
        from nmn.torch.nmn import YatNMN
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        layer = YatNMN(in_features=4, out_features=2).cuda()
        x = torch.randn(2, 4, device='cuda')
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.device.type == 'cuda'
    
    def test_to_method(self):
        """Test that .to() method works correctly."""
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(in_channels=3, out_channels=8, kernel_size=3)
        layer = layer.to(torch.float64)
        
        x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.dtype == torch.float64


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(in_features=4, out_features=2)
        x = torch.randn(1, 4)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (1, 2)
    
    def test_large_batch_size(self):
        """Test with large batch size."""
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(in_features=4, out_features=2)
        x = torch.randn(128, 4)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (128, 2)
    
    def test_single_channel_conv(self):
        """Test convolution with single channel."""
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(in_channels=1, out_channels=1, kernel_size=3)
        x = torch.randn(2, 1, 16, 16)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (2, 1, 14, 14)
    
    def test_large_kernel(self):
        """Test convolution with large kernel size."""
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(in_channels=3, out_channels=8, kernel_size=7)
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (2, 8, 26, 26)
    
    def test_stride_and_padding(self):
        """Test convolution with stride and padding."""
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(
            in_channels=3, out_channels=8, kernel_size=3,
            stride=2, padding=1
        )
        x = torch.randn(2, 3, 16, 16)
        
        with torch.no_grad():
            output = layer(x)
        
        # With stride=2, padding=1, kernel=3: output = (16 + 2 - 3) / 2 + 1 = 8
        assert output.shape == (2, 8, 8, 8)
    
    def test_dilation(self):
        """Test convolution with dilation."""
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(
            in_channels=3, out_channels=8, kernel_size=3,
            dilation=2
        )
        x = torch.randn(2, 3, 16, 16)
        
        with torch.no_grad():
            output = layer(x)
        
        # Effective kernel size = 3 + (3-1)*(2-1) = 5
        # Output size = 16 - 5 + 1 = 12
        assert output.shape == (2, 8, 12, 12)
    
    def test_grouped_convolution(self):
        """Test grouped convolution."""
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(
            in_channels=6, out_channels=12, kernel_size=3,
            groups=3
        )
        x = torch.randn(2, 6, 16, 16)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (2, 12, 14, 14)




