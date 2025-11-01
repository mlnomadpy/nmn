"""Unit tests for yat_conv module."""

import pytest


def test_yat_conv_imports():
    """Test that all YAT conv classes can be imported from yat_conv module."""
    try:
        from nmn.torch.yat_conv import (
            YatConv1d,
            YatConv2d,
            YatConv3d,
            YatConvTranspose1d,
            YatConvTranspose2d,
            YatConvTranspose3d,
        )
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_conv_from_main_module():
    """Test that YAT conv classes can be imported from main torch module."""
    try:
        from nmn.torch import (
            YatConv1d,
            YatConv2d,
            YatConv3d,
            YatConvTranspose1d,
            YatConvTranspose2d,
            YatConvTranspose3d,
        )
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_conv1d_module_instantiation():
    """Test YatConv1d can be instantiated from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConv1d
        
        layer = YatConv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )
        assert layer is not None
        assert layer.in_channels == 16
        assert layer.out_channels == 32
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_module_instantiation():
    """Test YatConv2d can be instantiated from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConv2d
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        assert layer is not None
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv3d_module_instantiation():
    """Test YatConv3d can be instantiated from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConv3d
        
        layer = YatConv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=3
        )
        assert layer is not None
        assert layer.in_channels == 8
        assert layer.out_channels == 16
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose1d_module_instantiation():
    """Test YatConvTranspose1d can be instantiated from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConvTranspose1d
        
        layer = YatConvTranspose1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )
        assert layer is not None
        assert layer.in_channels == 16
        assert layer.out_channels == 32
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose2d_module_instantiation():
    """Test YatConvTranspose2d can be instantiated from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConvTranspose2d
        
        layer = YatConvTranspose2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        assert layer is not None
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose3d_module_instantiation():
    """Test YatConvTranspose3d can be instantiated from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConvTranspose3d
        
        layer = YatConvTranspose3d(
            in_channels=8,
            out_channels=16,
            kernel_size=3
        )
        assert layer is not None
        assert layer.in_channels == 8
        assert layer.out_channels == 16
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_forward_from_module():
    """Test YatConv2d forward pass from yat_conv module."""
    try:
        import torch
        from nmn.torch.yat_conv import YatConv2d
        
        layer = YatConv2d(
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


def test_yat_conv2d_alpha_parameter():
    """Test YatConv2d alpha parameter from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConv2d
        
        layer_with_alpha = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_alpha=True
        )
        assert layer_with_alpha.use_alpha is True
        assert layer_with_alpha.alpha is not None
        
        layer_without_alpha = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_alpha=False
        )
        assert layer_without_alpha.use_alpha is False
        assert layer_without_alpha.alpha is None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_dropconnect_parameter():
    """Test YatConv2d dropconnect parameter from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConv2d
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_dropconnect=True,
            drop_rate=0.2
        )
        assert layer.use_dropconnect is True
        assert layer.drop_rate == 0.2
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_epsilon_parameter():
    """Test YatConv2d epsilon parameter from yat_conv module."""
    try:
        from nmn.torch.yat_conv import YatConv2d
        
        epsilon = 1e-6
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            epsilon=epsilon
        )
        assert layer.epsilon == epsilon
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")
