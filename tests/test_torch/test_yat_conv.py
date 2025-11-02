"""Tests for YatConv classes (non-transpose)."""

import pytest


def test_torch_yat_conv_import():
    """Test that YatConv classes can be imported."""
    try:
        import torch
        from nmn.torch.layers import YatConv1d, YatConv2d, YatConv3d
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_conv1d_instantiation():
    """Test YatConv1d can be instantiated."""
    try:
        import torch
        from nmn.torch.layers import YatConv1d
        
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


def test_yat_conv2d_instantiation():
    """Test YatConv2d can be instantiated."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
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


def test_yat_conv3d_instantiation():
    """Test YatConv3d can be instantiated."""
    try:
        import torch
        from nmn.torch.layers import YatConv3d
        
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


def test_yat_conv1d_forward():
    """Test YatConv1d forward pass."""
    try:
        import torch
        from nmn.torch.layers import YatConv1d
        
        layer = YatConv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        
        # Test forward pass
        batch_size = 2
        length = 10
        dummy_input = torch.randn(batch_size, 16, length)
        output = layer(dummy_input)
        
        # With padding=1, output should have same length
        assert output.shape == (batch_size, 32, length)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_forward():
    """Test YatConv2d forward pass."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
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


def test_yat_conv3d_forward():
    """Test YatConv3d forward pass."""
    try:
        import torch
        from nmn.torch.layers import YatConv3d
        
        layer = YatConv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        
        # Test forward pass
        batch_size = 2
        depth, height, width = 8, 8, 8
        dummy_input = torch.randn(batch_size, 8, depth, height, width)
        output = layer(dummy_input)
        
        # With padding=1, output should have same dimensions
        assert output.shape == (batch_size, 16, depth, height, width)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_with_alpha():
    """Test YatConv2d with alpha parameter."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_alpha=True
        )
        
        assert layer.use_alpha is True
        assert layer.alpha is not None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_without_alpha():
    """Test YatConv2d without alpha parameter."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_alpha=False
        )
        
        assert layer.use_alpha is False
        assert layer.alpha is None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_with_dropconnect():
    """Test YatConv2d with dropconnect."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
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


def test_yat_conv2d_custom_epsilon():
    """Test YatConv2d with custom epsilon."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
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


def test_yat_conv2d_stride_and_padding():
    """Test YatConv2d with stride and padding."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # Test forward pass
        batch_size = 2
        height, width = 32, 32
        dummy_input = torch.randn(batch_size, 3, height, width)
        output = layer(dummy_input)
        
        # With stride=2, output should be half size
        assert output.shape == (batch_size, 16, height // 2, width // 2)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv_all_variants_with_device():
    """Test all YatConv variants can be instantiated with device parameter."""
    try:
        import torch
        from nmn.torch.layers import YatConv1d, YatConv2d, YatConv3d
        
        # Test YatConv1d
        layer1d = YatConv1d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            device='cpu'
        )
        assert layer1d is not None
        assert layer1d.weight.device.type == 'cpu'
        
        # Test YatConv2d
        layer2d = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            device='cpu'
        )
        assert layer2d is not None
        assert layer2d.weight.device.type == 'cpu'
        
        # Test YatConv3d
        layer3d = YatConv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            device='cpu'
        )
        assert layer3d is not None
        assert layer3d.weight.device.type == 'cpu'
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_dtype_parameter():
    """Test YatConv2d can be instantiated with dtype parameter."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        # Test with float32 dtype
        layer_f32 = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            dtype=torch.float32
        )
        assert layer_f32 is not None
        assert layer_f32.weight.dtype == torch.float32
        
        # Test with float64 dtype
        layer_f64 = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            dtype=torch.float64
        )
        assert layer_f64 is not None
        assert layer_f64.weight.dtype == torch.float64
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_device_and_dtype():
    """Test YatConv2d with both device and dtype parameters simultaneously."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        # This tests the fix for the device/dtype kwargs issue
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            device='cpu',
            dtype=torch.float32
        )
        
        assert layer is not None
        assert layer.weight.device.type == 'cpu'
        assert layer.weight.dtype == torch.float32
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 32, 32, device='cpu', dtype=torch.float32)
        output = layer(dummy_input)
        assert output.device.type == 'cpu'
        assert output.dtype == torch.float32
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_device_via_to():
    """Test YatConv2d can be moved to different devices (simulating multi-GPU)."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        # Test instantiation and .to() method which is commonly used with DataParallel
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        
        # Test .to(device) - this is what DataParallel does internally
        device = torch.device('cpu')
        layer_on_device = layer.to(device)
        assert layer_on_device is not None
        
        # Test forward pass on the device
        dummy_input = torch.randn(2, 3, 32, 32, device=device)
        output = layer_on_device(dummy_input)
        assert output.device == device
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")
