"""Comprehensive Keras/TensorFlow vision example with YAT layers on CIFAR-10.

This example demonstrates:
- Data loading and preprocessing
- Model definition with YAT convolutional and dense layers
- Training with validation
- Testing and evaluation
- Model saving and loading
- Performance visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

try:
    from nmn.keras.nmn import YatNMN
    from nmn.keras.conv import YatConv2D
    
    print("🎯 NMN Keras Vision Example: CIFAR-10 Classification")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 10
    NUM_CLASSES = 10
    VALIDATION_SPLIT = 0.1
    MODEL_SAVE_PATH = "/tmp/keras_cifar10_yat_model"
    
    # CIFAR-10 class names
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    def load_and_preprocess_data():
        """Load and preprocess CIFAR-10 dataset."""
        print("\n📥 Loading CIFAR-10 dataset...")
        
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Convert to float32 and normalize to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
        
        # Split training data for validation
        val_size = int(len(x_train) * VALIDATION_SPLIT)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]
        
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Image shape: {x_train[0].shape}")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def create_yat_cnn_model():
        """Create a CNN model with YAT layers."""
        print("\n🏗️ Building YAT CNN model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(32, 32, 3)),
            
            # First convolutional block with YAT Conv2D
            YatConv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            YatConv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            YatConv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            YatConv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            YatConv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            YatConv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers with YAT
            layers.Flatten(),
            YatNMN(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            YatNMN(256, activation='relu'),
            layers.Dropout(0.5),
            YatNMN(NUM_CLASSES, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        print(f"Model compiled successfully!")
        model.summary()
        
        return model
    
    def create_baseline_cnn_model():
        """Create a baseline CNN model with standard layers for comparison."""
        print("\n🏗️ Building baseline CNN model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(32, 32, 3)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        return model
    
    def train_model(model, train_data, val_data, model_name="YAT"):
        """Train the model with validation."""
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        print(f"\n🏋️ Training {model_name} model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        datagen.fit(x_train)
        
        # Train the model
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=len(x_train) // BATCH_SIZE
        )
        
        return history
    
    def evaluate_model(model, test_data, model_name="YAT"):
        """Evaluate the model on test data."""
        x_test, y_test = test_data
        
        print(f"\n📊 Evaluating {model_name} model...")
        
        # Evaluate on test set
        test_loss, test_accuracy, test_top2_accuracy = model.evaluate(
            x_test, y_test, verbose=0
        )
        
        print(f"{model_name} Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Top-2 Accuracy: {test_top2_accuracy:.4f}")
        
        # Predictions for detailed analysis
        y_pred_prob = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top2_accuracy': test_top2_accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
    
    def plot_training_history(yat_history, baseline_history=None):
        """Plot training history."""
        print("\n📈 Plotting training history...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation accuracy
        axes[0, 0].plot(yat_history.history['accuracy'], label='YAT Train Accuracy')
        axes[0, 0].plot(yat_history.history['val_accuracy'], label='YAT Val Accuracy')
        if baseline_history:
            axes[0, 0].plot(baseline_history.history['accuracy'], label='Baseline Train Accuracy', linestyle='--')
            axes[0, 0].plot(baseline_history.history['val_accuracy'], label='Baseline Val Accuracy', linestyle='--')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Plot training & validation loss
        axes[0, 1].plot(yat_history.history['loss'], label='YAT Train Loss')
        axes[0, 1].plot(yat_history.history['val_loss'], label='YAT Val Loss')
        if baseline_history:
            axes[0, 1].plot(baseline_history.history['loss'], label='Baseline Train Loss', linestyle='--')
            axes[0, 1].plot(baseline_history.history['val_loss'], label='Baseline Val Loss', linestyle='--')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Plot learning rate (if available)
        if 'lr' in yat_history.history:
            axes[1, 0].plot(yat_history.history['lr'], label='YAT Learning Rate')
            if baseline_history and 'lr' in baseline_history.history:
                axes[1, 0].plot(baseline_history.history['lr'], label='Baseline Learning Rate', linestyle='--')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
        
        # Plot top-2 accuracy
        axes[1, 1].plot(yat_history.history['top_2_accuracy'], label='YAT Train Top-2')
        axes[1, 1].plot(yat_history.history['val_top_2_accuracy'], label='YAT Val Top-2')
        if baseline_history:
            axes[1, 1].plot(baseline_history.history['top_2_accuracy'], label='Baseline Train Top-2', linestyle='--')
            axes[1, 1].plot(baseline_history.history['val_top_2_accuracy'], label='Baseline Val Top-2', linestyle='--')
        axes[1, 1].set_title('Top-2 Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Top-2 Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('/tmp/keras_cifar10_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(y_true, y_pred, model_name="YAT"):
        """Plot confusion matrix."""
        print(f"\n🎯 Plotting {model_name} confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'{model_name} Model - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'/tmp/keras_cifar10_{model_name.lower()}_confusion_matrix.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_and_load_model(model, save_path):
        """Save and load model to verify serialization."""
        print(f"\n💾 Saving model to {save_path}...")
        
        # Save the model
        model.save(save_path)
        print("Model saved successfully!")
        
        # Load the model
        print("Loading model...")
        loaded_model = keras.models.load_model(save_path)
        print("Model loaded successfully!")
        
        return loaded_model
    
    # Main execution
    def main():
        """Main execution function."""
        # Load and preprocess data
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
        
        # Create models
        yat_model = create_yat_cnn_model()
        
        # Train YAT model
        yat_history = train_model(yat_model, (x_train, y_train), (x_val, y_val), "YAT")
        
        # Evaluate YAT model
        yat_results = evaluate_model(yat_model, (x_test, y_test), "YAT")
        
        # Create and train baseline model for comparison
        print("\n" + "="*60)
        print("Training baseline model for comparison...")
        baseline_model = create_baseline_cnn_model()
        baseline_history = train_model(baseline_model, (x_train, y_train), (x_val, y_val), "Baseline")
        baseline_results = evaluate_model(baseline_model, (x_test, y_test), "Baseline")
        
        # Plot results
        plot_training_history(yat_history, baseline_history)
        plot_confusion_matrix(yat_results['y_true'], yat_results['y_pred'], "YAT")
        plot_confusion_matrix(baseline_results['y_true'], baseline_results['y_pred'], "Baseline")
        
        # Save and load model
        loaded_model = save_and_load_model(yat_model, MODEL_SAVE_PATH)
        
        # Verify loaded model works
        print("\n🔍 Verifying loaded model...")
        loaded_results = evaluate_model(loaded_model, (x_test, y_test), "Loaded YAT")
        
        # Summary comparison
        print("\n" + "="*60)
        print("📋 FINAL COMPARISON SUMMARY")
        print("="*60)
        print(f"YAT Model Test Accuracy: {yat_results['test_accuracy']:.4f}")
        print(f"Baseline Model Test Accuracy: {baseline_results['test_accuracy']:.4f}")
        print(f"Loaded Model Test Accuracy: {loaded_results['test_accuracy']:.4f}")
        print(f"YAT vs Baseline Difference: {yat_results['test_accuracy'] - baseline_results['test_accuracy']:+.4f}")
        
        if yat_results['test_accuracy'] > baseline_results['test_accuracy']:
            print("🎉 YAT model outperformed the baseline!")
        else:
            print("📊 Baseline model performed better than YAT model.")
        
        print("\n✅ Keras vision example completed successfully!")
        print(f"📁 Plots saved to /tmp/keras_cifar10_*.png")
        print(f"💾 Model saved to {MODEL_SAVE_PATH}")
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Required dependencies not available: {e}")
    print("Install with: pip install 'nmn[keras]' tensorflow-datasets matplotlib seaborn scikit-learn")

except Exception as e:
    print(f"❌ Error running Keras vision example: {e}")
    import traceback
    traceback.print_exc()