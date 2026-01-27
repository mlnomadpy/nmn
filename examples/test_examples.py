#!/usr/bin/env python3
"""
Quick test script for NMN comprehensive examples.

This script runs a minimal version of the comprehensive examples to verify 
that everything is working correctly before running the full training pipelines.
"""

import os
import sys
import numpy as np
import tensorflow as tf

def test_basic_functionality():
    """Test basic YAT layer functionality."""
    print("🔧 Testing basic YAT layer functionality...")
    
    try:
        from nmn.keras.nmn import YatNMN
        from nmn.keras.conv import YatConv2D
        
        # Test YatNMN
        layer = YatNMN(10)
        test_input = tf.random.normal((5, 20))
        output = layer(test_input)
        assert output.shape == (5, 10), f"Expected (5, 10), got {output.shape}"
        
        # Test YatConv2D
        conv_layer = YatConv2D(8, (3, 3), padding='same')
        test_input = tf.random.normal((2, 16, 16, 3))
        output = conv_layer(test_input)
        assert output.shape == (2, 16, 16, 8), f"Expected (2, 16, 16, 8), got {output.shape}"
        
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_model_creation():
    """Test creating and compiling models."""
    print("\n🏗️ Testing model creation...")
    
    try:
        from nmn.keras.nmn import YatNMN
        from nmn.keras.conv import YatConv2D
        
        # Test vision model
        vision_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            YatConv2D(16, (3, 3), padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            YatNMN(32),
            tf.keras.layers.Activation('relu'),
            YatNMN(10),
            tf.keras.layers.Activation('softmax')
        ])
        
        vision_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Test language model
        language_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(1000, 64, input_length=100),
            tf.keras.layers.LSTM(64),
            YatNMN(32),
            tf.keras.layers.Activation('relu'),
            YatNMN(2),
            tf.keras.layers.Activation('softmax')
        ])
        
        language_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def test_training():
    """Test quick training loop."""
    print("\n🏋️ Testing quick training...")
    
    try:
        from nmn.keras.nmn import YatNMN
        
        # Create minimal model
        model = tf.keras.Sequential([
            YatNMN(16, input_shape=(10,)),
            tf.keras.layers.Activation('relu'),
            YatNMN(1),
            tf.keras.layers.Activation('sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Generate synthetic data
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, (100, 1)).astype(np.float32)
        
        # Quick training
        history = model.fit(
            X, y,
            epochs=2,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        # Test evaluation
        loss, accuracy = model.evaluate(X[:20], y[:20], verbose=0)
        
        print(f"✅ Training test passed! Final accuracy: {accuracy:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def test_save_load():
    """Test model save and load functionality."""
    print("\n💾 Testing save/load functionality...")
    
    try:
        from nmn.keras.nmn import YatNMN
        from nmn.keras.conv import YatConv2D
        
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(16, 16, 3)),
            YatConv2D(8, (3, 3), padding='same'),
            tf.keras.layers.Flatten(),
            YatNMN(16),
            tf.keras.layers.Activation('relu'),
            YatNMN(5),
            tf.keras.layers.Activation('softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Test input
        test_input = tf.random.normal((4, 16, 16, 3))
        original_prediction = model.predict(test_input, verbose=0)
        
        # Save model
        save_path = '/tmp/test_yat_model.keras'
        model.save(save_path)
        
        # Load model
        loaded_model = tf.keras.models.load_model(save_path)
        loaded_prediction = loaded_model.predict(test_input, verbose=0)
        
        # Check consistency
        diff = np.abs(original_prediction - loaded_prediction).max()
        assert diff < 1e-6, f"Predictions differ by {diff}"
        
        # Clean up
        if os.path.exists(save_path):
            if os.path.isdir(save_path):
                import shutil
                shutil.rmtree(save_path)
            else:
                os.remove(save_path)
        
        print("✅ Save/load test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 NMN Comprehensive Examples - Quick Test")
    print("=" * 50)
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    tests = [
        test_basic_functionality,
        test_model_creation,
        test_training,
        test_save_load
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You can now run the comprehensive examples:")
        print("\n📋 Next steps:")
        print("   python examples/keras/vision_cifar10.py")
        print("   python examples/tensorflow/vision_cifar10.py")
        print("   python examples/keras/language_imdb.py")
        print("   python examples/tensorflow/language_imdb.py")
        return 0
    else:
        print("❌ Some tests failed. Please check your installation:")
        print("   pip install 'nmn[keras]' tensorflow-datasets matplotlib seaborn scikit-learn")
        return 1

if __name__ == "__main__":
    sys.exit(main())