"""Unit tests for conv module."""

import pytest


def test_conv_module_imports():
    """Test that standard conv classes can be imported from conv module."""
    try:
        from nmn.torch.conv import (
            Conv1d,
            Conv2d,
            Conv3d,
            ConvTranspose1d,
            ConvTranspose2d,
            ConvTranspose3d,
            LazyConv1d,
            LazyConv2d,
            LazyConv3d,
            LazyConvTranspose1d,
            LazyConvTranspose2d,
            LazyConvTranspose3d,
        )
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_conv_from_main_module():
    """Test that standard conv classes can be imported from main torch module."""
    try:
        from nmn.torch import (
            Conv1d,
            Conv2d,
            Conv3d,
            ConvTranspose1d,
            ConvTranspose2d,
            ConvTranspose3d,
            LazyConv1d,
            LazyConv2d,
            LazyConv3d,
            LazyConvTranspose1d,
            LazyConvTranspose2d,
            LazyConvTranspose3d,
        )
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_classes_not_in_conv_module():
    """Test that YAT classes are not in conv module (they're in yat_conv)."""
    try:
        from nmn.torch import conv
        
        # YAT classes should not be in conv module
        assert not hasattr(conv, 'YatConv1d')
        assert not hasattr(conv, 'YatConv2d')
        assert not hasattr(conv, 'YatConv3d')
        assert not hasattr(conv, 'YatConvTranspose1d')
        assert not hasattr(conv, 'YatConvTranspose2d')
        assert not hasattr(conv, 'YatConvTranspose3d')
        assert not hasattr(conv, 'YatConvNd')
        assert not hasattr(conv, 'YatConvTransposeNd')
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_classes_in_yat_conv_module():
    """Test that YAT classes are in yat_conv module."""
    try:
        from nmn.torch import yat_conv
        
        # YAT classes should be in yat_conv module
        assert hasattr(yat_conv, 'YatConv1d')
        assert hasattr(yat_conv, 'YatConv2d')
        assert hasattr(yat_conv, 'YatConv3d')
        assert hasattr(yat_conv, 'YatConvTranspose1d')
        assert hasattr(yat_conv, 'YatConvTranspose2d')
        assert hasattr(yat_conv, 'YatConvTranspose3d')
        assert hasattr(yat_conv, 'YatConvNd')
        assert hasattr(yat_conv, 'YatConvTransposeNd')
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_conv2d_module_instantiation():
    """Test Conv2d can be instantiated from conv module."""
    try:
        from nmn.torch.conv import Conv2d
        
        layer = Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        assert layer is not None
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_conv_transpose2d_module_instantiation():
    """Test ConvTranspose2d can be instantiated from conv module."""
    try:
        from nmn.torch.conv import ConvTranspose2d
        
        layer = ConvTranspose2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        assert layer is not None
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_lazy_conv2d_module_instantiation():
    """Test LazyConv2d can be instantiated from conv module."""
    try:
        from nmn.torch.conv import LazyConv2d
        
        layer = LazyConv2d(
            out_channels=16,
            kernel_size=3
        )
        assert layer is not None
        assert layer.out_channels == 16
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_conv2d_forward_from_module():
    """Test Conv2d forward pass from conv module."""
    try:
        import torch
        from nmn.torch.conv import Conv2d
        
        layer = Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        
        # Test forward pass
        batch_size = 2
        height, width = 32, 32
        dummy_input = torch.randn(batch_size, 3, height, width)
        output = layer(dummy_input)
        
        # With padding=1, output should have same dimensions
        assert output.shape == (batch_size, 16, height, width)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")
