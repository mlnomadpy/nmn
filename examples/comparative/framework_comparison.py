"""Comparative example showing NMN usage across different frameworks."""

import numpy as np

def test_framework_imports():
    """Test which frameworks are available."""
    available_frameworks = {}
    
    # Test JAX/Flax NNX
    try:
        import jax
        import flax.nnx as nnx
        from nmn.nnx.nmn import YatNMN
        available_frameworks['nnx'] = True
        print("✅ Flax NNX available")
    except ImportError:
        available_frameworks['nnx'] = False
        print("❌ Flax NNX not available")
    
    # Test PyTorch
    try:
        import torch
        from nmn.torch.conv import YatConv2d
        available_frameworks['torch'] = True
        print("✅ PyTorch available")
    except ImportError:
        available_frameworks['torch'] = False
        print("❌ PyTorch not available")
    
    # Test Keras
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatDense
        available_frameworks['keras'] = True
        print("✅ Keras available")
    except ImportError:
        available_frameworks['keras'] = False
        print("❌ Keras not available")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        from nmn.tf.nmn import YatDense
        available_frameworks['tf'] = True
        print("✅ TensorFlow available")
    except ImportError:
        available_frameworks['tf'] = False
        print("❌ TensorFlow not available")
    
    # Test Flax Linen
    try:
        import jax
        from flax import linen as nn
        from nmn.linen.nmn import YatDense
        available_frameworks['linen'] = True
        print("✅ Flax Linen available")
    except ImportError:
        available_frameworks['linen'] = False
        print("❌ Flax Linen not available")
    
    return available_frameworks


def run_pytorch_example():
    """Run PyTorch YAT convolution example."""
    try:
        import torch
        from nmn.torch.conv import YatConv2d
        
        print("\n🔥 PyTorch YatConv2d Example")
        print("-" * 30)
        
        layer = YatConv2d(3, 16, kernel_size=3, use_alpha=True)
        input_tensor = torch.randn(1, 3, 32, 32)
        output = layer(input_tensor)
        
        print(f"Input: {input_tensor.shape}")
        print(f"Output: {output.shape}")
        return True
    except Exception as e:
        print(f"PyTorch example failed: {e}")
        return False


def run_keras_example():
    """Run Keras YAT dense example."""
    try:
        import tensorflow as tf
        from nmn.keras.nmn import YatDense
        
        print("\n🧠 Keras YatDense Example")
        print("-" * 25)
        
        layer = YatDense(10, use_alpha=True)
        input_tensor = tf.random.normal((4, 8))
        output = layer(input_tensor)
        
        print(f"Input: {input_tensor.shape}")
        print(f"Output: {output.shape}")
        return True
    except Exception as e:
        print(f"Keras example failed: {e}")
        return False


def main():
    """Run comparative examples across available frameworks."""
    print("NMN Cross-Framework Comparison")
    print("=" * 40)
    
    # Check available frameworks
    available = test_framework_imports()
    
    # Run examples for available frameworks
    success_count = 0
    total_tests = 0
    
    if available.get('torch'):
        total_tests += 1
        if run_pytorch_example():
            success_count += 1
    
    if available.get('keras'):
        total_tests += 1
        if run_keras_example():
            success_count += 1
    
    # Summary
    print(f"\n📊 Summary")
    print("-" * 15)
    print(f"Available frameworks: {sum(available.values())}/5")
    print(f"Successful tests: {success_count}/{total_tests}")
    
    if success_count == total_tests and total_tests > 0:
        print("🎉 All available frameworks working correctly!")
    elif total_tests == 0:
        print("⚠️  No frameworks available for testing")
        print("   Install frameworks with: pip install 'nmn[all]'")
    else:
        print("⚠️  Some tests failed - check framework installations")


if __name__ == "__main__":
    main()