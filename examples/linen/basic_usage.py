"""Basic usage example for NMN with Flax Linen."""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from nmn.linen.nmn import YatDense
    
    print("NMN Flax Linen Basic Example")
    print("=" * 35)
    
    # Create YAT dense layer
    model = YatDense(features=10, use_alpha=True, alpha=1.5)
    
    # Initialize model parameters
    key = jax.random.PRNGKey(42)
    input_shape = (4, 8)  # batch_size=4, features=8
    dummy_input = jnp.ones(input_shape)
    
    # Initialize parameters
    params = model.init(key, dummy_input)
    
    # Forward pass
    output = model.apply(params, dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Test with random input
    key, subkey = jax.random.split(key)
    random_input = jax.random.normal(subkey, (2, 8))
    random_output = model.apply(params, random_input)
    
    print(f"\nRandom input shape: {random_input.shape}")
    print(f"Random output shape: {random_output.shape}")
    print(f"Output mean: {jnp.mean(random_output):.4f}")
    print(f"Output std: {jnp.std(random_output):.4f}")
    
    # Demonstrate vectorized operations
    batch_inputs = jax.random.normal(key, (10, 8))
    batch_outputs = model.apply(params, batch_inputs)
    print(f"\nBatch processing:")
    print(f"Batch input shape: {batch_inputs.shape}")
    print(f"Batch output shape: {batch_outputs.shape}")
    
    print("\n✅ Flax Linen example completed successfully!")
    
except ImportError as e:
    print(f"❌ JAX/Flax not available: {e}")
    print("Install with: pip install 'nmn[linen]'")

except Exception as e:
    print(f"❌ Error running example: {e}")