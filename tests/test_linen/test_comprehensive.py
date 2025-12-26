"""Comprehensive tests for Linen implementation."""

import pytest
import numpy as np


def test_linen_yat_nmn_alpha():
    """Test YatNMN with alpha scaling."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from nmn.linen.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(features=10, use_alpha=True)
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 8))
        params = layer.init(key, dummy_input)
        
        # Test forward pass
        output = layer.apply(params, dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_linen_yat_nmn_no_alpha():
    """Test YatNMN without alpha scaling."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from nmn.linen.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(features=10, use_alpha=False)
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 8))
        params = layer.init(key, dummy_input)
        
        # Test forward pass
        output = layer.apply(params, dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_linen_yat_nmn_no_bias():
    """Test YatNMN without bias."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from nmn.linen.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(features=10, use_bias=False)
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 8))
        params = layer.init(key, dummy_input)
        
        # Test forward pass
        output = layer.apply(params, dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_linen_yat_nmn_custom_epsilon():
    """Test YatNMN with custom epsilon."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from nmn.linen.nmn import YatNMN
        
        # Create layer with custom epsilon
        layer = YatNMN(features=10, epsilon=1e-4)
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((4, 8))
        params = layer.init(key, dummy_input)
        
        # Test forward pass
        output = layer.apply(params, dummy_input)
        
        assert output.shape == (4, 10)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_linen_yat_nmn_forward_pass():
    """Test YatNMN forward pass with realistic data."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from nmn.linen.nmn import YatNMN
        
        # Create layer
        layer = YatNMN(features=16)
        
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jax.random.normal(key, (8, 32))
        params = layer.init(key, dummy_input)
        
        # Test forward pass
        output = layer.apply(params, dummy_input)
        
        assert output.shape == (8, 16)
        assert not jnp.isnan(output).any()
        assert not jnp.isinf(output).any()
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")



