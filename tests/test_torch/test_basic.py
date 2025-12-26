"""Tests for PyTorch implementation."""

import pytest
import numpy as np


def test_torch_import():
    """Test that PyTorch module can be imported."""
    try:
        import torch
        from nmn.torch import YatConv2d, YatNMN
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_conv2d_basic():
    """Test basic YatConv2d functionality."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        # Test parameters
        in_channels, out_channels = 3, 16
        kernel_size = 3
        
        # Create layer
        layer = YatConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        
        # Test forward pass
        batch_size = 2
        input_size = 32
        dummy_input = torch.randn(batch_size, in_channels, input_size, input_size)
        output = layer(dummy_input)
        
        # Expected output size for valid convolution
        expected_size = input_size - kernel_size + 1  # 30
        assert output.shape == (batch_size, out_channels, expected_size, expected_size)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_parameters():
    """Test YatConv2d parameter configuration."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_alpha=True,
            epsilon=1e-6
        )
        
        # Check if parameters are properly set
        assert layer.use_alpha is True
        assert layer.alpha is not None  # alpha should be a Parameter when use_alpha=True
        assert layer.epsilon == 1e-6
        
        # Test that we can set alpha value
        with torch.no_grad():
            layer.alpha.fill_(1.5)
        assert layer.alpha.item() == pytest.approx(1.5)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")