"""Basic usage example for NMN with TensorFlow."""

import numpy as np

try:
    import tensorflow as tf
    from nmn.tf.nmn import YatNMN
    
    print("NMN TensorFlow Basic Example")
    print("=" * 35)
    
    # Create input data
    batch_size = 8
    input_dim = 16
    output_dim = 10
    
    # Create YAT dense layer
    yat_layer = YatNMN(features=output_dim)
    
    # Create dummy input
    dummy_input = tf.random.normal((batch_size, input_dim))
    
    # Forward pass
    output = yat_layer(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Layer built: {yat_layer.is_built}")
    
    # Test with different epsilon values
    print("\nTesting different epsilon values:")
    for epsilon_val in [1e-4, 1e-5, 1e-6, 1e-7]:
        test_layer = YatNMN(features=5, epsilon=epsilon_val)
        test_output = test_layer(dummy_input[:4, :8])  # Smaller input
        print(f"Epsilon {epsilon_val}: output range [{tf.reduce_min(test_output):.3f}, {tf.reduce_max(test_output):.3f}]")
    
    print("\n✅ TensorFlow example completed successfully!")
    
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("Install with: pip install 'nmn[tf]'")

except Exception as e:
    print(f"❌ Error running example: {e}")