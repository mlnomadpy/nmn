"""Comprehensive TensorFlow language example with YAT layers for text classification.

This example demonstrates:
- Text data loading and preprocessing with tf.data
- Text vectorization and embedding
- Model definition with YAT dense layers using tf.Module
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
    
    print("📝 NMN TensorFlow Language Example: IMDB Sentiment Classification")
    print("=" * 70)
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 15
    NUM_CLASSES = 2
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    MODEL_SAVE_PATH = "/tmp/tf_imdb_yat_model"
    VOCAB_SIZE = 10000
    SEQUENCE_LENGTH = 200
    EMBEDDING_DIM = 128
    
    # Class names
    CLASS_NAMES = ['Negative', 'Positive']
    
    class YatTextClassifier(tf.Module):
        """YAT-based text classification model using tf.Module."""
        
        def __init__(self, vocab_size, embedding_dim, sequence_length, num_classes=2, name=None):
            super().__init__(name=name)
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.sequence_length = sequence_length
            self.num_classes = num_classes
            
            # Embedding layer
            self.embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=sequence_length,
                mask_zero=True
            )
            
            # LSTM layers
            self.lstm1 = tf.keras.layers.LSTM(
                128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
            )
            self.lstm2 = tf.keras.layers.LSTM(
                64, dropout=0.2, recurrent_dropout=0.2
            )
            
            # YAT dense layers
            self.dense1 = YatNMN(256)
            self.dense2 = YatNMN(128)
            self.dense3 = YatNMN(64)
            self.dense4 = YatNMN(num_classes)
            
            # Batch normalization and dropout layers
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
        
        @tf.function
        def __call__(self, x, training=False):
            # Embedding
            x = self.embedding(x)
            
            # LSTM layers
            x = self.lstm1(x, training=training)
            x = self.lstm2(x, training=training)
            
            # YAT dense layers
            x = self.dense1(x)
            x = tf.nn.relu(x)
            x = self.bn1(x, training=training)
            x = tf.nn.dropout(x, rate=0.5 if training else 0.0)
            
            x = self.dense2(x)
            x = tf.nn.relu(x)
            x = self.bn2(x, training=training)
            x = tf.nn.dropout(x, rate=0.5 if training else 0.0)
            
            x = self.dense3(x)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, rate=0.3 if training else 0.0)
            
            x = self.dense4(x)
            return tf.nn.softmax(x)
    
    class BaselineTextClassifier(tf.Module):
        """Baseline text classification model for comparison."""
        
        def __init__(self, vocab_size, embedding_dim, sequence_length, num_classes=2, name=None):
            super().__init__(name=name)
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.sequence_length = sequence_length
            self.num_classes = num_classes
            
            # Embedding layer
            self.embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=sequence_length,
                mask_zero=True
            )
            
            # LSTM layers
            self.lstm1 = tf.keras.layers.LSTM(
                128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
            )
            self.lstm2 = tf.keras.layers.LSTM(
                64, dropout=0.2, recurrent_dropout=0.2
            )
            
            # Standard dense layers
            self.dense1 = tf.keras.layers.Dense(256)
            self.dense2 = tf.keras.layers.Dense(128)
            self.dense3 = tf.keras.layers.Dense(64)
            self.dense4 = tf.keras.layers.Dense(num_classes)
            
            # Batch normalization and dropout layers
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
        
        @tf.function
        def __call__(self, x, training=False):
            # Embedding
            x = self.embedding(x)
            
            # LSTM layers
            x = self.lstm1(x, training=training)
            x = self.lstm2(x, training=training)
            
            # Standard dense layers
            x = self.dense1(x)
            x = tf.nn.relu(x)
            x = self.bn1(x, training=training)
            x = tf.nn.dropout(x, rate=0.5 if training else 0.0)
            
            x = self.dense2(x)
            x = tf.nn.relu(x)
            x = self.bn2(x, training=training)
            x = tf.nn.dropout(x, rate=0.5 if training else 0.0)
            
            x = self.dense3(x)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, rate=0.3 if training else 0.0)
            
            x = self.dense4(x)
            return tf.nn.softmax(x)
    
    def load_and_preprocess_data():
        """Load and preprocess IMDB dataset using tf.data."""
        print("\n📥 Loading IMDB dataset...")
        
        # Load IMDB dataset
        (ds_train, ds_test), ds_info = tfds.load(
            'imdb_reviews',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        
        # Create text vectorizer
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_sequence_length=SEQUENCE_LENGTH,
            standardize='lower_and_strip_punctuation',
        )
        
        # Adapt the vectorizer to the training data
        train_text = ds_train.map(lambda text, label: text)
        vectorize_layer.adapt(train_text.batch(1000))
        
        def vectorize_text(text, label):
            text = tf.expand_dims(text, -1)
            return vectorize_layer(text), label
        
        # Prepare training dataset
        train_size = ds_info.splits['train'].num_examples
        val_size = int(train_size * VALIDATION_SPLIT)
        
        ds_train = ds_train.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(1000)
        
        # Split into train and validation
        ds_val = ds_train.take(val_size)
        ds_train = ds_train.skip(val_size)
        
        # Batch and prefetch
        ds_train = ds_train.batch(BATCH_SIZE)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        
        ds_val = ds_val.batch(BATCH_SIZE)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
        
        # Prepare test dataset
        ds_test = ds_test.map(vectorize_text, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(BATCH_SIZE)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
        
        print(f"Training batches: {len(list(ds_train))}")
        print(f"Validation batches: {len(list(ds_val))}")
        print(f"Test batches: {len(list(ds_test))}")
        
        return ds_train, ds_val, ds_test, vectorize_layer
    
    def compute_loss(y_true, y_pred):
        """Compute categorical crossentropy loss."""
        y_true = tf.one_hot(y_true, NUM_CLASSES)
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    
    def compute_accuracy(y_true, y_pred):
        """Compute accuracy."""
        predictions = tf.argmax(y_pred, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, predictions), tf.float32))
    
    def compute_precision_recall(y_true, y_pred):
        """Compute precision and recall."""
        predictions = tf.argmax(y_pred, axis=1)
        
        # For binary classification
        true_positives = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(predictions, 1)), tf.float32)
        )
        predicted_positives = tf.reduce_sum(
            tf.cast(tf.equal(predictions, 1), tf.float32)
        )
        actual_positives = tf.reduce_sum(
            tf.cast(tf.equal(y_true, 1), tf.float32)
        )
        
        precision = true_positives / (predicted_positives + 1e-7)
        recall = true_positives / (actual_positives + 1e-7)
        
        return precision, recall
    
    @tf.function
    def train_step(model, optimizer, x, y):
        """Execute one training step."""
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = compute_loss(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        accuracy = compute_accuracy(y, predictions)
        precision, recall = compute_precision_recall(y, predictions)
        
        return loss, accuracy, precision, recall
    
    @tf.function
    def val_step(model, x, y):
        """Execute one validation step."""
        predictions = model(x, training=False)
        loss = compute_loss(y, predictions)
        accuracy = compute_accuracy(y, predictions)
        precision, recall = compute_precision_recall(y, predictions)
        
        return loss, accuracy, precision, recall, predictions
    
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
            'loss': [], 'accuracy': [], 'precision': [], 'recall': []
        }
        val_metrics = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = 3
        
        for epoch in range(EPOCHS):
            print(f"\\nEpoch {epoch + 1}/{EPOCHS}")
            
            # Training
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            epoch_train_precision = 0
            epoch_train_recall = 0
            num_train_batches = 0
            
            for x_batch, y_batch in ds_train:
                loss, accuracy, precision, recall = train_step(model, optimizer, x_batch, y_batch)
                epoch_train_loss += loss
                epoch_train_accuracy += accuracy
                epoch_train_precision += precision
                epoch_train_recall += recall
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            avg_train_accuracy = epoch_train_accuracy / num_train_batches
            avg_train_precision = epoch_train_precision / num_train_batches
            avg_train_recall = epoch_train_recall / num_train_batches
            
            # Validation
            epoch_val_loss = 0
            epoch_val_accuracy = 0
            epoch_val_precision = 0
            epoch_val_recall = 0
            num_val_batches = 0
            
            for x_batch, y_batch in ds_val:
                loss, accuracy, precision, recall, _ = val_step(model, x_batch, y_batch)
                epoch_val_loss += loss
                epoch_val_accuracy += accuracy
                epoch_val_precision += precision
                epoch_val_recall += recall
                num_val_batches += 1
            
            avg_val_loss = epoch_val_loss / num_val_batches
            avg_val_accuracy = epoch_val_accuracy / num_val_batches
            avg_val_precision = epoch_val_precision / num_val_batches
            avg_val_recall = epoch_val_recall / num_val_batches
            
            # Store metrics
            train_metrics['loss'].append(float(avg_train_loss))
            train_metrics['accuracy'].append(float(avg_train_accuracy))
            train_metrics['precision'].append(float(avg_train_precision))
            train_metrics['recall'].append(float(avg_train_recall))
            
            val_metrics['loss'].append(float(avg_val_loss))
            val_metrics['accuracy'].append(float(avg_val_accuracy))
            val_metrics['precision'].append(float(avg_val_precision))
            val_metrics['recall'].append(float(avg_val_recall))
            
            # Calculate F1 scores
            train_f1 = 2 * avg_train_precision * avg_train_recall / (avg_train_precision + avg_train_recall + 1e-7)
            val_f1 = 2 * avg_val_precision * avg_val_recall / (avg_val_precision + avg_val_recall + 1e-7)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
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
        test_precision = 0
        test_recall = 0
        num_test_batches = 0
        
        all_y_true = []
        all_y_pred = []
        all_y_pred_prob = []
        
        for x_batch, y_batch in ds_test:
            loss, accuracy, precision, recall, predictions = val_step(model, x_batch, y_batch)
            test_loss += loss
            test_accuracy += accuracy
            test_precision += precision
            test_recall += recall
            num_test_batches += 1
            
            # Collect predictions for detailed analysis
            all_y_true.extend(y_batch.numpy())
            all_y_pred.extend(tf.argmax(predictions, axis=1).numpy())
            all_y_pred_prob.extend(predictions.numpy())
        
        avg_test_loss = test_loss / num_test_batches
        avg_test_accuracy = test_accuracy / num_test_batches
        avg_test_precision = test_precision / num_test_batches
        avg_test_recall = test_recall / num_test_batches
        avg_test_f1 = 2 * avg_test_precision * avg_test_recall / (avg_test_precision + avg_test_recall + 1e-7)
        
        print(f"{model_name} Test Results:")
        print(f"  Loss: {avg_test_loss:.4f}")
        print(f"  Accuracy: {avg_test_accuracy:.4f}")
        print(f"  Precision: {avg_test_precision:.4f}")
        print(f"  Recall: {avg_test_recall:.4f}")
        print(f"  F1 Score: {avg_test_f1:.4f}")
        
        # Classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(all_y_true, all_y_pred, target_names=CLASS_NAMES))
        
        return {
            'test_loss': float(avg_test_loss),
            'test_accuracy': float(avg_test_accuracy),
            'test_precision': float(avg_test_precision),
            'test_recall': float(avg_test_recall),
            'test_f1': float(avg_test_f1),
            'y_true': np.array(all_y_true),
            'y_pred': np.array(all_y_pred),
            'y_pred_prob': np.array(all_y_pred_prob)
        }
    
    def plot_training_history(yat_history, baseline_history=None):
        """Plot training history."""
        print("\n📈 Plotting training history...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
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
        
        # Plot precision
        axes[0, 2].plot(yat_history['train']['precision'], label='YAT Train Precision')
        axes[0, 2].plot(yat_history['val']['precision'], label='YAT Val Precision')
        if baseline_history:
            axes[0, 2].plot(baseline_history['train']['precision'], label='Baseline Train Precision', linestyle='--')
            axes[0, 2].plot(baseline_history['val']['precision'], label='Baseline Val Precision', linestyle='--')
        axes[0, 2].set_title('Model Precision')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        
        # Plot recall
        axes[1, 0].plot(yat_history['train']['recall'], label='YAT Train Recall')
        axes[1, 0].plot(yat_history['val']['recall'], label='YAT Val Recall')
        if baseline_history:
            axes[1, 0].plot(baseline_history['train']['recall'], label='Baseline Train Recall', linestyle='--')
            axes[1, 0].plot(baseline_history['val']['recall'], label='Baseline Val Recall', linestyle='--')
        axes[1, 0].set_title('Model Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        
        # Plot F1 score
        yat_f1_train = [2 * p * r / (p + r + 1e-7) for p, r in zip(yat_history['train']['precision'], yat_history['train']['recall'])]
        yat_f1_val = [2 * p * r / (p + r + 1e-7) for p, r in zip(yat_history['val']['precision'], yat_history['val']['recall'])]
        
        axes[1, 1].plot(yat_f1_train, label='YAT Train F1')
        axes[1, 1].plot(yat_f1_val, label='YAT Val F1')
        
        if baseline_history:
            baseline_f1_train = [2 * p * r / (p + r + 1e-7) for p, r in zip(baseline_history['train']['precision'], baseline_history['train']['recall'])]
            baseline_f1_val = [2 * p * r / (p + r + 1e-7) for p, r in zip(baseline_history['val']['precision'], baseline_history['val']['recall'])]
            axes[1, 1].plot(baseline_f1_train, label='Baseline Train F1', linestyle='--')
            axes[1, 1].plot(baseline_f1_val, label='Baseline Val F1', linestyle='--')
        
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        
        # Plot validation accuracy comparison
        if baseline_history:
            epochs = range(1, len(yat_history['val']['accuracy']) + 1)
            axes[1, 2].plot(epochs, yat_history['val']['accuracy'], 'b-', label='YAT Val Accuracy')
            axes[1, 2].plot(epochs, baseline_history['val']['accuracy'], 'r--', label='Baseline Val Accuracy')
            axes[1, 2].set_title('Validation Accuracy Comparison')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('/tmp/tf_imdb_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(y_true, y_pred, model_name="YAT"):
        """Plot confusion matrix."""
        print(f"\n🎯 Plotting {model_name} confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'{model_name} Model - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'/tmp/tf_imdb_{model_name.lower()}_confusion_matrix.png', 
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
        sample_input = tf.random.uniform((1, SEQUENCE_LENGTH), maxval=VOCAB_SIZE, dtype=tf.int32)
        concrete_fn = serve_fn.get_concrete_function(
            x=tf.TensorSpec(shape=(None, SEQUENCE_LENGTH), dtype=tf.int32)
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
        ds_train, ds_val, ds_test, vectorize_layer = load_and_preprocess_data()
        
        # Create models
        print("\n🏗️ Creating YAT text classification model...")
        yat_model = YatTextClassifier(VOCAB_SIZE, EMBEDDING_DIM, SEQUENCE_LENGTH, NUM_CLASSES)
        
        # Train YAT model
        yat_history = train_model(yat_model, ds_train, ds_val, "YAT")
        
        # Evaluate YAT model
        yat_results = evaluate_model(yat_model, ds_test, "YAT")
        
        # Create and train baseline model for comparison
        print("\n" + "="*70)
        print("Training baseline model for comparison...")
        baseline_model = BaselineTextClassifier(VOCAB_SIZE, EMBEDDING_DIM, SEQUENCE_LENGTH, NUM_CLASSES)
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
        print("\n" + "="*70)
        print("📋 FINAL COMPARISON SUMMARY")
        print("="*70)
        print(f"YAT Model Test Accuracy: {yat_results['test_accuracy']:.4f}")
        print(f"YAT Model Test F1 Score: {yat_results['test_f1']:.4f}")
        print(f"Baseline Model Test Accuracy: {baseline_results['test_accuracy']:.4f}")
        print(f"Baseline Model Test F1 Score: {baseline_results['test_f1']:.4f}")
        print(f"YAT vs Baseline Accuracy Difference: {yat_results['test_accuracy'] - baseline_results['test_accuracy']:+.4f}")
        print(f"YAT vs Baseline F1 Difference: {yat_results['test_f1'] - baseline_results['test_f1']:+.4f}")
        
        if yat_results['test_accuracy'] > baseline_results['test_accuracy']:
            print("🎉 YAT model outperformed the baseline!")
        else:
            print("📊 Baseline model performed better than YAT model.")
        
        print("\n✅ TensorFlow language example completed successfully!")
        print(f"📁 Plots saved to /tmp/tf_imdb_*.png")
        print(f"💾 Model saved to {MODEL_SAVE_PATH}_final")
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Required dependencies not available: {e}")
    print("Install with: pip install 'nmn[tf]' tensorflow-datasets matplotlib seaborn scikit-learn")

except Exception as e:
    print(f"❌ Error running TensorFlow language example: {e}")
    import traceback
    traceback.print_exc()