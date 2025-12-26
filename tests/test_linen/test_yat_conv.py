"""Tests for Linen YatConv layers - TDD: Write tests first."""

import pytest
import numpy as np


def test_linen_yat_conv1d_import():
    """Test that YatConv1D can be imported."""
    try:
        from nmn.linen.conv import YatConv1D
        assert YatConv1D is not None
    except ImportError as e:
        pytest.skip(f"JAX/Flax dependencies not available: {e}")


def test_linen_yat_conv2d_import():
    """Test that YatConv2D can be imported."""
    try:
        from nmn.linen.conv import YatConv2D
        assert YatConv2D is not None
    except ImportError as e:
        pytest.skip(f"JAX/Flax dependencies not available: {e}")


def test_linen_yat_conv1d_forward():
    """Test YatConv1D forward pass."""
    try:
        import jax
        import jax.numpy as jnp
        from nmn.linen.conv import YatConv1D
        
        layer = YatConv1D(features=16, kernel_size=(3,))
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 10, 8))  # [batch, length, channels]
        params = layer.init(key, dummy_input)
        
        # Forward pass
        output = layer.apply(params, dummy_input)
        
        # For valid padding: output_length = input_length - kernel_size + 1 = 10 - 3 + 1 = 8
        assert output.shape == (4, 8, 16)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_linen_yat_conv2d_forward():
    """Test YatConv2D forward pass."""
    try:
        import jax
        import jax.numpy as jnp
        from nmn.linen.conv import YatConv2D
        
        layer = YatConv2D(features=16, kernel_size=(3, 3))
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 32, 32, 3))  # [batch, height, width, channels]
        params = layer.init(key, dummy_input)
        
        # Forward pass
        output = layer.apply(params, dummy_input)
        
        # For valid padding: output_size = input_size - kernel_size + 1 = 32 - 3 + 1 = 30
        assert output.shape == (4, 30, 30, 16)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_linen_yat_conv2d_same_padding():
    """Test YatConv2D with SAME padding."""
    try:
        import jax
        import jax.numpy as jnp
        from nmn.linen.conv import YatConv2D
        
        layer = YatConv2D(features=16, kernel_size=(3, 3), padding='SAME')
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 32, 32, 3))
        params = layer.init(key, dummy_input)
        
        # Forward pass
        output = layer.apply(params, dummy_input)
        
        # For same padding: output_size = input_size
        assert output.shape == (4, 32, 32, 16)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_linen_yat_conv2d_no_bias():
    """Test YatConv2D without bias."""
    try:
        import jax
        import jax.numpy as jnp
        from nmn.linen.conv import YatConv2D
        
        layer = YatConv2D(features=16, kernel_size=(3, 3), use_bias=False)
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 32, 32, 3))
        params = layer.init(key, dummy_input)
        
        # Forward pass
        output = layer.apply(params, dummy_input)
        
        # Check no bias in params
        assert 'bias' not in params['params']
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_linen_yat_conv2d_alpha():
    """Test YatConv2D with alpha scaling."""
    try:
        import jax
        import jax.numpy as jnp
        from nmn.linen.conv import YatConv2D
        
        layer_with_alpha = YatConv2D(features=16, kernel_size=(3, 3), use_alpha=True)
        layer_no_alpha = YatConv2D(features=16, kernel_size=(3, 3), use_alpha=False)
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 32, 32, 3))
        
        params_with = layer_with_alpha.init(key, dummy_input)
        params_no = layer_no_alpha.init(key, dummy_input)
        
        assert 'alpha' in params_with['params']
        assert 'alpha' not in params_no['params']
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")



