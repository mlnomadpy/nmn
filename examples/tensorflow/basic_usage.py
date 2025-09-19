"""Basic usage example for NMN with TensorFlow."""

import numpy as np

try:
    import tensorflow as tf
    from nmn.tf.nmn import YatDense
    
    print("NMN TensorFlow Basic Example")
    print("=" * 35)
    
    # Create input data
    batch_size = 8
    input_dim = 16
    output_dim = 10
    
    # Create YAT dense layer
    yat_layer = YatDense(units=output_dim, use_alpha=True, alpha=1.2)
    
    # Create dummy input
    dummy_input = tf.random.normal((batch_size, input_dim))
    
    # Forward pass
    output = yat_layer(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Layer parameters: {yat_layer.count_params()}")
    
    # Test with different alpha values
    print("\nTesting different alpha values:")
    for alpha_val in [0.5, 1.0, 1.5, 2.0]:
        test_layer = YatDense(units=5, use_alpha=True, alpha=alpha_val)
        test_output = test_layer(dummy_input[:4, :8])  # Smaller input
        print(f"Alpha {alpha_val}: output range [{tf.reduce_min(test_output):.3f}, {tf.reduce_max(test_output):.3f}]")
    
    print("\n✅ TensorFlow example completed successfully!")
    
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("Install with: pip install 'nmn[tf]'")

except Exception as e:
    print(f"❌ Error running example: {e}")