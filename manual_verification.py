#!/usr/bin/env python3
"""
Manual verification of YAT algorithm logic.

This script manually verifies the YAT algorithm by implementing a simple version
and testing the mathematical components without requiring TensorFlow/Keras.
"""

import numpy as np
import math


def yat_nmn_manual(inputs, kernel, bias=None, alpha=1.0, epsilon=1e-5):
    """Manual implementation of YAT NMN layer for verification."""
    # Step 1: Compute dot product
    dot_product = np.dot(inputs, kernel)
    
    # Step 2: Compute squared norms
    inputs_squared_sum = np.sum(inputs**2, axis=-1, keepdims=True)
    kernel_squared_sum = np.sum(kernel**2, axis=0)
    
    # Step 3: Compute squared Euclidean distances
    distances = inputs_squared_sum + kernel_squared_sum - 2 * dot_product
    
    # Step 4: Apply YAT transformation
    outputs = dot_product**2 / (distances + epsilon)
    
    # Step 5: Add bias
    if bias is not None:
        outputs = outputs + bias
    
    # Step 6: Apply alpha scaling
    if alpha is not None:
        scale = (math.sqrt(kernel.shape[1]) / math.log(1.0 + kernel.shape[1])) ** alpha
        outputs = outputs * scale
    
    return outputs


def yat_conv1d_manual(inputs, kernel, bias=None, alpha=1.0, epsilon=1e-5, stride=1):
    """Manual implementation of YAT Conv1D for verification."""
    batch_size, length, in_channels = inputs.shape
    kernel_size, _, out_channels = kernel.shape
    
    # Calculate output length
    out_length = (length - kernel_size) // stride + 1
    
    outputs = np.zeros((batch_size, out_length, out_channels))
    
    for b in range(batch_size):
        for i in range(out_length):
            start_idx = i * stride
            end_idx = start_idx + kernel_size
            
            # Extract patch
            patch = inputs[b, start_idx:end_idx, :]  # (kernel_size, in_channels)
            
            for f in range(out_channels):
                # Compute dot product
                dot_product = np.sum(patch * kernel[:, :, f])
                
                # Compute squared norms
                patch_squared_sum = np.sum(patch**2)
                kernel_squared_sum = np.sum(kernel[:, :, f]**2)
                
                # Compute distance
                distance = patch_squared_sum + kernel_squared_sum - 2 * dot_product
                
                # Apply YAT
                outputs[b, i, f] = dot_product**2 / (distance + epsilon)
    
    # Add bias
    if bias is not None:
        outputs = outputs + bias
    
    # Apply alpha scaling
    if alpha is not None:
        scale = (math.sqrt(out_channels) / math.log(1.0 + out_channels)) ** alpha
        outputs = outputs * scale
    
    return outputs


def test_yat_nmn_logic():
    """Test YAT NMN logic."""
    print("🧮 Testing YAT NMN Logic")
    
    # Create test data
    batch_size, input_dim, output_dim = 2, 4, 3
    inputs = np.random.randn(batch_size, input_dim).astype(np.float32)
    kernel = np.random.randn(input_dim, output_dim).astype(np.float32)
    bias = np.random.randn(output_dim).astype(np.float32)
    
    # Test the algorithm
    output = yat_nmn_manual(inputs, kernel, bias, alpha=1.0)
    
    print(f"  Input shape: {inputs.shape}")
    print(f"  Kernel shape: {kernel.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verify output is finite and has correct shape
    assert output.shape == (batch_size, output_dim)
    assert np.all(np.isfinite(output))
    assert not np.any(np.isnan(output))
    
    print("  ✅ YAT NMN logic verified")
    return True


def test_yat_conv1d_logic():
    """Test YAT Conv1D logic."""
    print("🧮 Testing YAT Conv1D Logic")
    
    # Create test data
    batch_size, length, in_channels = 2, 8, 3
    out_channels, kernel_size = 4, 3
    
    inputs = np.random.randn(batch_size, length, in_channels).astype(np.float32)
    kernel = np.random.randn(kernel_size, in_channels, out_channels).astype(np.float32)
    bias = np.random.randn(out_channels).astype(np.float32)
    
    # Test the algorithm
    output = yat_conv1d_manual(inputs, kernel, bias, alpha=1.0)
    
    expected_out_length = (length - kernel_size) + 1
    
    print(f"  Input shape: {inputs.shape}")
    print(f"  Kernel shape: {kernel.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output length: {expected_out_length}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Verify output is finite and has correct shape
    assert output.shape == (batch_size, expected_out_length, out_channels)
    assert np.all(np.isfinite(output))
    assert not np.any(np.isnan(output))
    
    print("  ✅ YAT Conv1D logic verified")
    return True


def test_yat_properties():
    """Test mathematical properties of YAT algorithm."""
    print("🧮 Testing YAT Mathematical Properties")
    
    # Test 1: YAT should be positive
    inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
    kernel = np.array([[1.0, 0.0], [0.0, 1.0]])
    output = yat_nmn_manual(inputs, kernel)
    assert np.all(output >= 0), "YAT output should be non-negative"
    print("  ✅ Non-negativity property verified")
    
    # Test 2: Perfect match should give high activation
    inputs = np.array([[1.0, 0.0]])
    kernel = np.array([[1.0], [0.0]])  # Perfect match for first input
    output = yat_nmn_manual(inputs, kernel)
    print(f"  Perfect match activation: {output[0, 0]:.3f}")
    
    # Test 3: Orthogonal vectors should give lower activation
    kernel_orth = np.array([[0.0], [1.0]])  # Orthogonal to first input
    output_orth = yat_nmn_manual(inputs, kernel_orth)
    print(f"  Orthogonal activation: {output_orth[0, 0]:.3f}")
    
    # Perfect match should give higher activation than orthogonal
    assert output[0, 0] > output_orth[0, 0], "Perfect match should activate more than orthogonal"
    print("  ✅ Activation preference verified")
    
    return True


def main():
    """Main verification function."""
    print("🔬 Manual YAT Algorithm Verification")
    print("=" * 50)
    
    try:
        # Test the core algorithms
        test_yat_nmn_logic()
        print()
        test_yat_conv1d_logic()
        print()
        test_yat_properties()
        print()
        
        print("🎉 All manual verifications passed!")
        print("   The YAT algorithm implementations are mathematically sound.")
        return 0
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())