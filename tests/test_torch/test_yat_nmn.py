"""Tests for YatNMN class."""

import pytest


def test_torch_yat_nmn_import():
    """Test that YatNMN can be imported."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_nmn_instantiation():
    """Test YatNMN can be instantiated."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        
        # Test basic instantiation
        layer = YatNMN(
            in_features=10,
            out_features=5
        )
        assert layer is not None
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.bias is not None
        assert layer.alpha is not None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_no_bias():
    """Test YatNMN without bias."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(
            in_features=10,
            out_features=5,
            bias=False
        )
        assert layer.bias is None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_no_alpha():
    """Test YatNMN without alpha."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(
            in_features=10,
            out_features=5,
            alpha=False
        )
        assert layer.alpha is None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_forward():
    """Test YatNMN forward pass."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(
            in_features=10,
            out_features=5
        )
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 10)
        output = layer(dummy_input)
        
        assert output.shape == (batch_size, 5)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_custom_epsilon():
    """Test YatNMN with custom epsilon."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        
        epsilon = 1e-6
        layer = YatNMN(
            in_features=10,
            out_features=5,
            epsilon=epsilon
        )
        assert layer.epsilon == epsilon
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_custom_dtype():
    """Test YatNMN with custom dtype."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(
            in_features=10,
            out_features=5,
            dtype=torch.float64
        )
        assert layer.dtype == torch.float64
        assert layer.weight.dtype == torch.float64
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_reset_parameters():
    """Test YatNMN reset_parameters method."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(
            in_features=10,
            out_features=5
        )
        
        # Store original weights
        original_weight = layer.weight.data.clone()
        
        # Reset parameters
        layer.reset_parameters()
        
        # Weights should have changed (with very high probability)
        assert not torch.allclose(original_weight, layer.weight.data)
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_extra_repr():
    """Test YatNMN extra_repr method."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(
            in_features=10,
            out_features=5
        )
        
        repr_str = layer.extra_repr()
        assert "in_features=10" in repr_str
        assert "out_features=5" in repr_str
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")
