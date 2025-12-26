"""Tests for PyTorch DropConnect functionality - TDD."""

import pytest
import numpy as np


def test_yat_conv2d_dropconnect_import():
    """Test that YatConv2d with DropConnect can be imported."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        assert YatConv2d is not None
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_conv2d_dropconnect_instantiation():
    """Test YatConv2d with DropConnect can be instantiated."""
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
        assert layer is not None
        assert layer.use_dropconnect is True
        assert layer.drop_rate == 0.2
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_dropconnect_training_mode():
    """Test that DropConnect affects training but not eval."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        torch.manual_seed(42)
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_dropconnect=True,
            drop_rate=0.5
        )
        
        dummy_input = torch.randn(2, 3, 32, 32)
        
        # Training mode: outputs should vary due to dropout
        layer.train()
        torch.manual_seed(1)
        output1 = layer(dummy_input)
        torch.manual_seed(2)
        output2 = layer(dummy_input)
        
        # With 50% dropout, outputs should be different
        assert not torch.allclose(output1, output2, atol=1e-5)
        
        # Eval mode: outputs should be deterministic
        layer.eval()
        with torch.no_grad():
            output3 = layer(dummy_input, deterministic=True)
            output4 = layer(dummy_input, deterministic=True)
        
        # In eval mode with deterministic=True, outputs should be same
        assert torch.allclose(output3, output4, atol=1e-5)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_dropconnect_gradient():
    """Test that gradients flow correctly with DropConnect."""
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
        layer.train()
        
        dummy_input = torch.randn(2, 3, 32, 32, requires_grad=True)
        output = layer(dummy_input)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert dummy_input.grad is not None
        assert layer.weight.grad is not None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_mask_functionality():
    """Test that custom mask works correctly."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        # Create a mask that zeros out half the weights
        mask = torch.ones(16, 3, 3, 3)
        mask[:8] = 0  # Zero out first 8 filters
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            mask=mask
        )
        
        dummy_input = torch.randn(2, 3, 32, 32)
        output = layer(dummy_input)
        
        # Output should have correct shape
        expected_h = 32 - 3 + 1
        expected_w = 32 - 3 + 1
        assert output.shape == (2, 16, expected_h, expected_w)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_conv2d_no_dropconnect():
    """Test that layer works normally without DropConnect."""
    try:
        import torch
        from nmn.torch.layers import YatConv2d
        
        torch.manual_seed(42)
        
        layer = YatConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_dropconnect=False
        )
        layer.train()
        
        dummy_input = torch.randn(2, 3, 32, 32)
        
        # Without dropconnect, outputs should be deterministic even in train mode
        output1 = layer(dummy_input)
        output2 = layer(dummy_input)
        
        assert torch.allclose(output1, output2, atol=1e-5)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

