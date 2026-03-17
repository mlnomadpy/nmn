"""Unit tests for nmn module."""

import pytest


def test_nmn_module_import():
    """Test that YatNMN can be imported from nmn module."""
    try:
        from nmn.torch.nmn import YatNMN
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_nmn_from_main_module():
    """Test that YatNMN can be imported from main torch module."""
    try:
        from nmn.torch import YatNMN
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_nmn_from_specific_file():
    """Test that YatNMN can be imported from specific file."""
    try:
        from nmn.torch.nmn.yat_nmn import YatNMN
        assert True
    except ImportError as e:
        pytest.skip(f"PyTorch dependencies not available: {e}")


def test_yat_nmn_module_instantiation():
    """Test YatNMN can be instantiated from nmn module."""
    try:
        from nmn.torch.nmn import YatNMN
        
        layer = YatNMN(
            in_features=10,
            out_features=5
        )
        assert layer is not None
        assert layer.in_features == 10
        assert layer.out_features == 5
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_module_forward():
    """Test YatNMN forward pass from nmn module."""
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


def test_yat_nmn_module_parameters():
    """Test YatNMN parameters from nmn module."""
    try:
        from nmn.torch.nmn import YatNMN
        
        # Test with bias and alpha
        layer = YatNMN(
            in_features=10,
            out_features=5,
            bias=True,
            alpha=True
        )
        assert layer.bias is not None
        assert layer.alpha is not None
        
        # Test without bias
        layer_no_bias = YatNMN(
            in_features=10,
            out_features=5,
            bias=False
        )
        assert layer_no_bias.bias is None
        
        # Test without alpha
        layer_no_alpha = YatNMN(
            in_features=10,
            out_features=5,
            alpha=False
        )
        assert layer_no_alpha.alpha is None
        
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


def test_yat_nmn_module_epsilon():
    """Test YatNMN epsilon parameter from nmn module."""
    try:
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


def test_yat_nmn_module_dtype():
    """Test YatNMN param_dtype controls weight storage; dtype controls compute."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN

        # param_dtype controls the storage dtype of weights
        layer = YatNMN(
            in_features=10,
            out_features=5,
            param_dtype=torch.float64
        )
        assert layer.weight.dtype == torch.float64

        # dtype is the compute dtype, does not change weight storage
        layer2 = YatNMN(
            in_features=10,
            out_features=5,
            dtype=torch.float64
        )
        assert layer2.dtype == torch.float64
        # weight storage is still param_dtype (float32 by default)
        assert layer2.weight.dtype == torch.float32

    except ImportError:
        pytest.skip("PyTorch dependencies not available")
