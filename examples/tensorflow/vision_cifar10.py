"""Comprehensive TensorFlow vision example with YAT layers on CIFAR-10.

This example demonstrates:
- Data loading and preprocessing with tf.data
- Model definition with YAT convolutional and dense layers using tf.Module
- Custom training loop with validation
- Testing and evaluation
- Model saving and loading with checkpoints
- Performance visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow_datasets as tfds

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

try:
    from nmn.tf.nmn import YatNMN
    from nmn.tf.conv import YatConv2D
    
    print("🎯 NMN TensorFlow Vision Example: CIFAR-10 Classification")
    print("=" * 65)
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 10
    NUM_CLASSES = 10
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.1
    MODEL_SAVE_PATH = "/tmp/tf_cifar10_yat_model"
    
    # CIFAR-10 class names
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    class YatCNNModel(tf.Module):
        """Custom YAT CNN model using tf.Module."""
        
        def __init__(self, num_classes=10, name=None):
            super().__init__(name=name)
            self.num_classes = num_classes
            
            # Convolutional layers
            self.conv1 = YatConv2D(32, kernel_size=3, padding='SAME')
            self.conv2 = YatConv2D(32, kernel_size=3, padding='SAME')
            
            self.conv3 = YatConv2D(64, kernel_size=3, padding='SAME')
            self.conv4 = YatConv2D(64, kernel_size=3, padding='SAME')
            
            self.conv5 = YatConv2D(128, kernel_size=3, padding='SAME')
            self.conv6 = YatConv2D(128, kernel_size=3, padding='SAME')
            
            # Dense layers
            self.dense1 = YatNMN(512)
            self.dense2 = YatNMN(256)
            self.dense3 = YatNMN(num_classes)
            
            # Batch normalization layers
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.bn4 = tf.keras.layers.BatchNormalization()
        
        @tf.function
        def __call__(self, x, training=False):
            # First conv block
            x = self.conv1(x)
            x = tf.nn.relu(x)
            x = self.bn1(x, training=training)
            x = self.conv2(x)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
            x = tf.nn.dropout(x, rate=0.25 if training else 0.0)
            
            # Second conv block
            x = self.conv3(x)
            x = tf.nn.relu(x)
            x = self.bn2(x, training=training)
            x = self.conv4(x)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
            x = tf.nn.dropout(x, rate=0.25 if training else 0.0)
            
            # Third conv block
            x = self.conv5(x)
            x = tf.nn.relu(x)
            x = self.bn3(x, training=training)
            x = self.conv6(x)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
            x = tf.nn.dropout(x, rate=0.25 if training else 0.0)
            
            # Flatten and dense layers
            x = tf.keras.layers.Flatten()(x)
            
            x = self.dense1(x)
            x = tf.nn.relu(x)
            x = self.bn4(x, training=training)
            x = tf.nn.dropout(x, rate=0.5 if training else 0.0)
            
            x = self.dense2(x)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, rate=0.5 if training else 0.0)
            
            x = self.dense3(x)
            return tf.nn.softmax(x)
    
    class BaselineCNNModel(tf.Module):
        """Baseline CNN model for comparison."""
        
        def __init__(self, num_classes=10, name=None):
            super().__init__(name=name)
            self.num_classes = num_classes
            
            # Standard convolutional layers
            self.conv1 = tf.keras.layers.Conv2D(32, 3, padding='same')
            self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same')
            self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same')
            self.conv4 = tf.keras.layers.Conv2D(64, 3, padding='same')
            self.conv5 = tf.keras.layers.Conv2D(128, 3, padding='same')
            self.conv6 = tf.keras.layers.Conv2D(128, 3, padding='same')
            
            # Standard dense layers
            self.dense1 = tf.keras.layers.Dense(512)
            self.dense2 = tf.keras.layers.Dense(256)
            self.dense3 = tf.keras.layers.Dense(num_classes)
            
            # Batch normalization layers
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.bn4 = tf.keras.layers.BatchNormalization()
        
        @tf.function
        def __call__(self, x, training=False):
            # First conv block
            x = self.conv1(x)
            x = tf.nn.relu(x)
            x = self.bn1(x, training=training)
            x = self.conv2(x)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
            x = tf.nn.dropout(x, rate=0.25 if training else 0.0)
            
            # Second conv block
            x = self.conv3(x)
            x = tf.nn.relu(x)
            x = self.bn2(x, training=training)
            x = self.conv4(x)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
            x = tf.nn.dropout(x, rate=0.25 if training else 0.0)
            
            # Third conv block
            x = self.conv5(x)
            x = tf.nn.relu(x)
            x = self.bn3(x, training=training)
            x = self.conv6(x)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
            x = tf.nn.dropout(x, rate=0.25 if training else 0.0)
            
            # Flatten and dense layers
            x = tf.keras.layers.Flatten()(x)
            
            x = self.dense1(x)
            x = tf.nn.relu(x)
            x = self.bn4(x, training=training)
            x = tf.nn.dropout(x, rate=0.5 if training else 0.0)
            
            x = self.dense2(x)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, rate=0.5 if training else 0.0)
            
            x = self.dense3(x)
            return tf.nn.softmax(x)
    
    def load_and_preprocess_data():
        """Load and preprocess CIFAR-10 dataset using tf.data."""
        print("\n📥 Loading CIFAR-10 dataset...")
        
        # Load CIFAR-10 from tensorflow_datasets
        (ds_train, ds_test), ds_info = tfds.load(
            'cifar10',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        
        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255.0, label
        
        def augment_img(image, label):
            """Data augmentation for training."""
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            return image, label
        
        # Prepare training dataset
        train_size = ds_info.splits['train'].num_examples
        val_size = int(train_size * VALIDATION_SPLIT)
        train_size = train_size - val_size
        
        ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(1000)
        
        # Split into train and validation
        ds_val = ds_train.take(val_size)
        ds_train = ds_train.skip(val_size)
        
        # Apply augmentation to training data
        ds_train = ds_train.map(augment_img, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        ds_train = ds_train.batch(BATCH_SIZE)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        
        ds_val = ds_val.batch(BATCH_SIZE)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
        
        # Prepare test dataset
        ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(BATCH_SIZE)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
        
        print(f"Training batches: {len(list(ds_train))}")
        print(f"Validation batches: {len(list(ds_val))}")
        print(f"Test batches: {len(list(ds_test))}")
        
        return ds_train, ds_val, ds_test
    
    def compute_loss(y_true, y_pred):
        """Compute categorical crossentropy loss."""
        y_true = tf.one_hot(y_true, NUM_CLASSES)
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    
    def compute_accuracy(y_true, y_pred):
        """Compute accuracy."""
        predictions = tf.argmax(y_pred, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, predictions), tf.float32))
    
    def compute_top_k_accuracy(y_true, y_pred, k=2):
        """Compute top-k accuracy."""
        return tf.reduce_mean(tf.cast(tf.nn.in_top_k(y_pred, y_true, k), tf.float32))
    
    @tf.function
    def train_step(model, optimizer, x, y):
        """Execute one training step."""
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = compute_loss(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        accuracy = compute_accuracy(y, predictions)
        top2_accuracy = compute_top_k_accuracy(y, predictions, k=2)
        
        return loss, accuracy, top2_accuracy
    
    @tf.function
    def val_step(model, x, y):
        """Execute one validation step."""
        predictions = model(x, training=False)
        loss = compute_loss(y, predictions)
        accuracy = compute_accuracy(y, predictions)
        top2_accuracy = compute_top_k_accuracy(y, predictions, k=2)
        
        return loss, accuracy, top2_accuracy, predictions
    
    def train_model(model, ds_train, ds_val, model_name="YAT"):
        """Train the model with custom training loop."""
        print(f"\n🏋️ Training {model_name} model...")
        
        # Optimizer with learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=1000,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Metrics tracking
        train_metrics = {
            'loss': [], 'accuracy': [], 'top2_accuracy': []
        }
        val_metrics = {
            'loss': [], 'accuracy': [], 'top2_accuracy': []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = 3
        
        for epoch in range(EPOCHS):
            print(f"\\nEpoch {epoch + 1}/{EPOCHS}")
            
            # Training
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            epoch_train_top2_accuracy = 0
            num_train_batches = 0
            
            for x_batch, y_batch in ds_train:
                loss, accuracy, top2_accuracy = train_step(model, optimizer, x_batch, y_batch)
                epoch_train_loss += loss
                epoch_train_accuracy += accuracy
                epoch_train_top2_accuracy += top2_accuracy
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            avg_train_accuracy = epoch_train_accuracy / num_train_batches
            avg_train_top2_accuracy = epoch_train_top2_accuracy / num_train_batches
            
            # Validation
            epoch_val_loss = 0
            epoch_val_accuracy = 0
            epoch_val_top2_accuracy = 0
            num_val_batches = 0
            
            for x_batch, y_batch in ds_val:
                loss, accuracy, top2_accuracy, _ = val_step(model, x_batch, y_batch)
                epoch_val_loss += loss
                epoch_val_accuracy += accuracy
                epoch_val_top2_accuracy += top2_accuracy
                num_val_batches += 1
            
            avg_val_loss = epoch_val_loss / num_val_batches
            avg_val_accuracy = epoch_val_accuracy / num_val_batches
            avg_val_top2_accuracy = epoch_val_top2_accuracy / num_val_batches
            
            # Store metrics
            train_metrics['loss'].append(float(avg_train_loss))
            train_metrics['accuracy'].append(float(avg_train_accuracy))
            train_metrics['top2_accuracy'].append(float(avg_train_top2_accuracy))
            
            val_metrics['loss'].append(float(avg_val_loss))
            val_metrics['accuracy'].append(float(avg_val_accuracy))
            val_metrics['top2_accuracy'].append(float(avg_val_top2_accuracy))
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Train Top-2: {avg_train_top2_accuracy:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}, Val Top-2: {avg_val_top2_accuracy:.4f}")
            
            # Early stopping
            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                patience_counter = 0
                # Save best model
                save_model(model, f"{MODEL_SAVE_PATH}_{model_name.lower()}_best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        history = {
            'train': train_metrics,
            'val': val_metrics
        }
        
        return history
    
    def evaluate_model(model, ds_test, model_name="YAT"):
        """Evaluate the model on test data."""
        print(f"\n📊 Evaluating {model_name} model...")
        
        test_loss = 0
        test_accuracy = 0
        test_top2_accuracy = 0
        num_test_batches = 0
        
        all_y_true = []
        all_y_pred = []
        all_y_pred_prob = []
        
        for x_batch, y_batch in ds_test:
            loss, accuracy, top2_accuracy, predictions = val_step(model, x_batch, y_batch)
            test_loss += loss
            test_accuracy += accuracy
            test_top2_accuracy += top2_accuracy
            num_test_batches += 1
            
            # Collect predictions for detailed analysis
            all_y_true.extend(y_batch.numpy())
            all_y_pred.extend(tf.argmax(predictions, axis=1).numpy())
            all_y_pred_prob.extend(predictions.numpy())
        
        avg_test_loss = test_loss / num_test_batches
        avg_test_accuracy = test_accuracy / num_test_batches
        avg_test_top2_accuracy = test_top2_accuracy / num_test_batches
        
        print(f"{model_name} Test Results:")
        print(f"  Loss: {avg_test_loss:.4f}")
        print(f"  Accuracy: {avg_test_accuracy:.4f}")
        print(f"  Top-2 Accuracy: {avg_test_top2_accuracy:.4f}")
        
        # Classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES))
        
        return {
            'test_loss': float(avg_test_loss),
            'test_accuracy': float(avg_test_accuracy),
            'test_top2_accuracy': float(avg_test_top2_accuracy),
            'y_true': np.array(all_y_true),
            'y_pred': np.array(all_y_pred),
            'y_pred_prob': np.array(all_y_pred_prob)
        }
    
    def plot_training_history(yat_history, baseline_history=None):
        """Plot training history."""
        print("\n📈 Plotting training history...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation accuracy
        axes[0, 0].plot(yat_history['train']['accuracy'], label='YAT Train Accuracy')
        axes[0, 0].plot(yat_history['val']['accuracy'], label='YAT Val Accuracy')
        if baseline_history:
            axes[0, 0].plot(baseline_history['train']['accuracy'], label='Baseline Train Accuracy', linestyle='--')
            axes[0, 0].plot(baseline_history['val']['accuracy'], label='Baseline Val Accuracy', linestyle='--')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Plot training & validation loss
        axes[0, 1].plot(yat_history['train']['loss'], label='YAT Train Loss')
        axes[0, 1].plot(yat_history['val']['loss'], label='YAT Val Loss')
        if baseline_history:
            axes[0, 1].plot(baseline_history['train']['loss'], label='Baseline Train Loss', linestyle='--')
            axes[0, 1].plot(baseline_history['val']['loss'], label='Baseline Val Loss', linestyle='--')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Plot top-2 accuracy
        axes[1, 0].plot(yat_history['train']['top2_accuracy'], label='YAT Train Top-2')
        axes[1, 0].plot(yat_history['val']['top2_accuracy'], label='YAT Val Top-2')
        if baseline_history:
            axes[1, 0].plot(baseline_history['train']['top2_accuracy'], label='Baseline Train Top-2', linestyle='--')
            axes[1, 0].plot(baseline_history['val']['top2_accuracy'], label='Baseline Val Top-2', linestyle='--')
        axes[1, 0].set_title('Top-2 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-2 Accuracy')
        axes[1, 0].legend()
        
        # Plot loss comparison
        if baseline_history:
            epochs = range(1, len(yat_history['val']['accuracy']) + 1)
            axes[1, 1].plot(epochs, yat_history['val']['accuracy'], 'b-', label='YAT Val Accuracy')
            axes[1, 1].plot(epochs, baseline_history['val']['accuracy'], 'r--', label='Baseline Val Accuracy')
            axes[1, 1].set_title('Validation Accuracy Comparison')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('/tmp/tf_cifar10_training_history.png', dpi=150, bbox_inches='tight')
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
        plt.savefig(f'/tmp/tf_cifar10_{model_name.lower()}_confusion_matrix.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(model, save_path):
        """Save model using TensorFlow SavedModel format."""
        print(f"\n💾 Saving model to {save_path}...")
        
        # Create a concrete function for saving
        @tf.function
        def serve_fn(x):
            return model(x, training=False)
        
        # Get a sample input to create the signature
        sample_input = tf.random.normal((1, 32, 32, 3))
        concrete_fn = serve_fn.get_concrete_function(
            x=tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32)
        )
        
        # Save the model
        tf.saved_model.save(
            model,
            save_path,
            signatures={'serving_default': concrete_fn}
        )
        print("Model saved successfully!")
    
    def load_model(load_path):
        """Load model from TensorFlow SavedModel format."""
        print(f"\n📂 Loading model from {load_path}...")
        loaded_model = tf.saved_model.load(load_path)
        print("Model loaded successfully!")
        return loaded_model
    
    # Main execution
    def main():
        """Main execution function."""
        # Load and preprocess data
        ds_train, ds_val, ds_test = load_and_preprocess_data()
        
        # Create models
        print("\n🏗️ Creating YAT CNN model...")
        yat_model = YatCNNModel(NUM_CLASSES)
        
        # Train YAT model
        yat_history = train_model(yat_model, ds_train, ds_val, "YAT")
        
        # Evaluate YAT model
        yat_results = evaluate_model(yat_model, ds_test, "YAT")
        
        # Create and train baseline model for comparison
        print("\n" + "="*65)
        print("Training baseline model for comparison...")
        baseline_model = BaselineCNNModel(NUM_CLASSES)
        baseline_history = train_model(baseline_model, ds_train, ds_val, "Baseline")
        baseline_results = evaluate_model(baseline_model, ds_test, "Baseline")
        
        # Plot results
        plot_training_history(yat_history, baseline_history)
        plot_confusion_matrix(yat_results['y_true'], yat_results['y_pred'], "YAT")
        plot_confusion_matrix(baseline_results['y_true'], baseline_results['y_pred'], "Baseline")
        
        # Save model
        save_model(yat_model, f"{MODEL_SAVE_PATH}_final")
        
        # Load and test the saved model
        loaded_model = load_model(f"{MODEL_SAVE_PATH}_final")
        
        # Test loaded model on a few samples
        print("\n🔍 Testing loaded model...")
        sample_batch = next(iter(ds_test.take(1)))
        x_sample, y_sample = sample_batch
        predictions = loaded_model.signatures['serving_default'](x=x_sample)
        pred_classes = tf.argmax(predictions['output_0'], axis=1)
        actual_classes = y_sample
        
        print(f"Sample predictions: {pred_classes.numpy()}")
        print(f"Actual classes: {actual_classes.numpy()}")
        
        # Summary comparison
        print("\n" + "="*65)
        print("📋 FINAL COMPARISON SUMMARY")
        print("="*65)
        print(f"YAT Model Test Accuracy: {yat_results['test_accuracy']:.4f}")
        print(f"Baseline Model Test Accuracy: {baseline_results['test_accuracy']:.4f}")
        print(f"YAT vs Baseline Difference: {yat_results['test_accuracy'] - baseline_results['test_accuracy']:+.4f}")
        
        if yat_results['test_accuracy'] > baseline_results['test_accuracy']:
            print("🎉 YAT model outperformed the baseline!")
        else:
            print("📊 Baseline model performed better than YAT model.")
        
        print("\n✅ TensorFlow vision example completed successfully!")
        print(f"📁 Plots saved to /tmp/tf_cifar10_*.png")
        print(f"💾 Model saved to {MODEL_SAVE_PATH}_final")
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Required dependencies not available: {e}")
    print("Install with: pip install 'nmn[tf]' tensorflow-datasets matplotlib seaborn scikit-learn")

except Exception as e:
    print(f"❌ Error running TensorFlow vision example: {e}")
    import traceback
    traceback.print_exc()