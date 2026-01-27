"""Unit tests for conv module."""

import pytest


def test_conv_module_imports():
    """Test that standard conv classes can be imported from conv module."""
    try:
        from nmn.torch.layers import (
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


def test_yat_classes_in_layers_module():
    """Test that YAT classes are exported from layers module."""
    try:
        from nmn.torch import layers
        
        # YAT classes should be exported from layers
        assert hasattr(layers, 'YatConv1d')
        assert hasattr(layers, 'YatConv2d')
        assert hasattr(layers, 'YatConv3d')
        assert hasattr(layers, 'YatConvTranspose1d')
        assert hasattr(layers, 'YatConvTranspose2d')
        assert hasattr(layers, 'YatConvTranspose3d')
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_base_classes_in_base_module():
    """Test that YAT base classes are in base module."""
    try:
        from nmn.torch import base
        
        # YAT base classes should be in base module
        assert hasattr(base, 'YatConvNd')
        assert hasattr(base, 'YatConvTransposeNd')
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_conv2d_module_instantiation():
    """Test Conv2d can be instantiated from conv module."""
    try:
        from nmn.torch.layers import Conv2d
        
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
        from nmn.torch.layers import ConvTranspose2d
        
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
        from nmn.torch.layers import LazyConv2d
        
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
        from nmn.torch.layers import Conv2d
        
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
