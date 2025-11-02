"""Tests for YatConvTranspose classes."""

import pytest


def test_torch_yat_conv_transpose_import():
    """Test that YatConvTranspose classes can be imported."""
    try:
        import torch
        from nmn.torch.layers import (
            YatConvTranspose1d,
            YatConvTranspose2d,
            YatConvTranspose3d,
        )
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_conv_transpose1d_instantiation():
    """Test YatConvTranspose1d can be instantiated with device and dtype parameters."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose1d
        
        # Test basic instantiation
        layer = YatConvTranspose1d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2
        )
        assert layer is not None
        assert layer.in_channels == 16
        assert layer.out_channels == 8
        
        # Test with device and dtype parameters (this was causing the bug)
        layer_with_device = YatConvTranspose1d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2,
            device=None,
            dtype=None
        )
        assert layer_with_device is not None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose2d_instantiation():
    """Test YatConvTranspose2d can be instantiated with device and dtype parameters."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose2d
        
        # Test basic instantiation (from error example in problem statement)
        layer = YatConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        assert layer is not None
        assert layer.in_channels == 128
        assert layer.out_channels == 64
        
        # Test with device and dtype parameters (this was causing the bug)
        layer_with_device = YatConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
            device=None,
            dtype=None
        )
        assert layer_with_device is not None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose3d_instantiation():
    """Test YatConvTranspose3d can be instantiated with device and dtype parameters."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose3d
        
        # Test basic instantiation
        layer = YatConvTranspose3d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2
        )
        assert layer is not None
        assert layer.in_channels == 32
        assert layer.out_channels == 16
        
        # Test with device and dtype parameters (this was causing the bug)
        layer_with_device = YatConvTranspose3d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2,
            device=None,
            dtype=None
        )
        assert layer_with_device is not None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose2d_forward():
    """Test YatConvTranspose2d forward pass."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose2d
        
        # Create layer
        layer = YatConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2
        )
        
        # Test forward pass
        batch_size = 2
        input_size = 16
        dummy_input = torch.randn(batch_size, 16, input_size, input_size)
        output = layer(dummy_input)
        
        # For transpose convolution with stride=2, kernel=2, output should be 2x input
        expected_size = input_size * 2
        assert output.shape == (batch_size, 8, expected_size, expected_size)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose2d_with_parameters():
    """Test YatConvTranspose2d with various parameters."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose2d
        
        layer = YatConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2,
            use_alpha=True,
            use_dropconnect=False,
            epsilon=1e-5,
            drop_rate=0.0
        )
        
        # Check if parameters are properly set
        assert layer.use_alpha == True
        assert layer.use_dropconnect == False
        assert layer.epsilon == 1e-5
        assert layer.drop_rate == 0.0
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose1d_forward():
    """Test YatConvTranspose1d forward pass."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose1d
        
        # Create layer
        layer = YatConvTranspose1d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2
        )
        
        # Test forward pass
        batch_size = 2
        input_length = 16
        dummy_input = torch.randn(batch_size, 16, input_length)
        output = layer(dummy_input)
        
        # For transpose convolution with stride=2, kernel=2, output should be 2x input
        expected_length = input_length * 2
        assert output.shape == (batch_size, 8, expected_length)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose3d_forward():
    """Test YatConvTranspose3d forward pass."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose3d
        
        # Create layer
        layer = YatConvTranspose3d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2
        )
        
        # Test forward pass
        batch_size = 2
        input_size = 8
        dummy_input = torch.randn(batch_size, 16, input_size, input_size, input_size)
        output = layer(dummy_input)
        
        # For transpose convolution with stride=2, kernel=2, output should be 2x input
        expected_size = input_size * 2
        assert output.shape == (batch_size, 8, expected_size, expected_size, expected_size)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose2d_device_dtype_via_to():
    """Test YatConvTranspose2d can be moved to different devices (simulating multi-GPU)."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose2d
        
        # Test instantiation and .to() method which is commonly used with DataParallel
        layer = YatConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2
        )
        
        # Test .to(device) - this is what DataParallel does internally
        # We test with CPU since GPU might not be available
        device = torch.device('cpu')
        layer_on_device = layer.to(device)
        assert layer_on_device is not None
        
        # Test forward pass on the device
        dummy_input = torch.randn(1, 32, 8, 8, device=device)
        output = layer_on_device(dummy_input)
        assert output.shape == (1, 16, 16, 16)
        assert output.device == device
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose_all_variants_with_device():
    """Test all YatConvTranspose variants can be instantiated with device parameter."""
    try:
        import torch
        from nmn.torch.layers import (
            YatConvTranspose1d,
            YatConvTranspose2d,
            YatConvTranspose3d,
        )
        
        # Test YatConvTranspose1d
        layer1d = YatConvTranspose1d(
            in_channels=8,
            out_channels=4,
            kernel_size=2,
            stride=2,
            device='cpu'
        )
        assert layer1d is not None
        assert layer1d.weight.device.type == 'cpu'
        
        # Test YatConvTranspose2d (the one from the bug report)
        layer2d = YatConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
            device='cpu'
        )
        assert layer2d is not None
        assert layer2d.weight.device.type == 'cpu'
        
        # Test YatConvTranspose3d
        layer3d = YatConvTranspose3d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2,
            device='cpu'
        )
        assert layer3d is not None
        assert layer3d.weight.device.type == 'cpu'
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose2d_dtype_parameter():
    """Test YatConvTranspose2d can be instantiated with dtype parameter."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose2d
        
        # Test with float32 dtype
        layer_f32 = YatConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2,
            dtype=torch.float32
        )
        assert layer_f32 is not None
        assert layer_f32.weight.dtype == torch.float32
        
        # Test with float64 dtype
        layer_f64 = YatConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=2,
            stride=2,
            dtype=torch.float64
        )
        assert layer_f64 is not None
        assert layer_f64.weight.dtype == torch.float64
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_transpose2d_device_and_dtype():
    """Test YatConvTranspose2d with both device and dtype parameters simultaneously."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose2d
        
        # This is the scenario that was failing before the fix
        layer = YatConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
            device='cpu',
            dtype=torch.float32
        )
        
        assert layer is not None
        assert layer.weight.device.type == 'cpu'
        assert layer.weight.dtype == torch.float32
        
        # Test forward pass
        dummy_input = torch.randn(2, 128, 16, 16, device='cpu', dtype=torch.float32)
        output = layer(dummy_input)
        assert output.shape == (2, 64, 32, 32)
        assert output.device.type == 'cpu'
        assert output.dtype == torch.float32
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")
