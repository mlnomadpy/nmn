"""Tests for NNX (Flax NNX) implementation."""

import pytest
import numpy as np


def test_nnx_import():
    """Test that NNX module can be imported."""
    try:
        import jax
        import flax.nnx as nnx
        from nmn.nnx import nmn
        from nmn.nnx import yatconv
        assert True
    except ImportError as e:
        pytest.skip(f"NNX dependencies not available: {e}")


def test_yat_nmn_basic():
    """Test basic YatNMN functionality."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.nmn import YatNMN
        
        # Test parameters
        in_features, out_features = 3, 4
        model_key, param_key, drop_key, input_key = jax.random.split(jax.random.key(0), 4)
        
        # Create layer
        layer = YatNMN(
            in_features=in_features, 
            out_features=out_features, 
            rngs=nnx.Rngs(params=param_key, dropout=drop_key)
        )
        
        # Test forward pass
        dummy_input = jax.random.normal(input_key, (2, in_features))
        output = layer(dummy_input)
        
        assert output.shape == (2, out_features)
        assert output.dtype == dummy_input.dtype
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_yat_conv_basic():
    """Test basic YatConv functionality."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.yatconv import YatConv
        
        # Test parameters
        in_channels, out_channels = 3, 8
        kernel_size = (3, 3)
        conv_key, conv_param_key, conv_input_key = jax.random.split(jax.random.key(1), 3)
        
        # Create layer with VALID padding for expected output size
        conv_layer = YatConv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            padding='VALID',
            rngs=nnx.Rngs(params=conv_param_key)
        )
        
        # Test forward pass
        dummy_conv_input = jax.random.normal(conv_input_key, (1, 28, 28, in_channels))
        conv_output = conv_layer(dummy_conv_input)
        
        # Expected output shape for valid convolution with 3x3 kernel
        expected_h = 28 - 3 + 1  # 26
        expected_w = 28 - 3 + 1  # 26
        assert conv_output.shape == (1, expected_h, expected_w, out_channels)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")