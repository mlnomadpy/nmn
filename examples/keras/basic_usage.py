"""Basic usage example for NMN with Keras/TensorFlow."""

import numpy as np

try:
    import tensorflow as tf
    from nmn.keras.nmn import YatNMN
    
    print("NMN Keras/TensorFlow Basic Example")
    print("=" * 40)
    
    # Create a simple model with YAT dense layers
    model = tf.keras.Sequential([
        YatNMN(64, input_shape=(10,)),
        tf.keras.layers.Activation('relu'),
        YatNMN(32),
        tf.keras.layers.Activation('relu'),
        YatNMN(1),
        tf.keras.layers.Activation('sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create some dummy data
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, (100, 1)).astype(np.float32)
    
    # Train for a few epochs
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=16,
        verbose=1,
        validation_split=0.2
    )
    
    # Make predictions
    X_test = np.random.randn(10, 10).astype(np.float32)
    predictions = model.predict(X_test, verbose=0)
    
    print(f"\nTest predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3].flatten()}")
    
    print("\n✅ Keras example completed successfully!")
    
except ImportError as e:
    print(f"❌ TensorFlow/Keras not available: {e}")
    print("Install with: pip install 'nmn[keras]'")

except Exception as e:
    print(f"❌ Error running example: {e}")