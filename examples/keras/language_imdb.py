"""Comprehensive Keras/TensorFlow language example with YAT layers for text classification.

This example demonstrates:
- Text data loading and preprocessing
- Text vectorization and embedding
- Model definition with YAT dense layers for text classification
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
    
    print("📝 NMN Keras Language Example: IMDB Sentiment Classification")
    print("=" * 65)
    
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 15
    NUM_CLASSES = 2
    VALIDATION_SPLIT = 0.2
    MODEL_SAVE_PATH = "/tmp/keras_imdb_yat_model.keras"
    VOCAB_SIZE = 10000
    SEQUENCE_LENGTH = 200
    EMBEDDING_DIM = 128
    
    # Class names
    CLASS_NAMES = ['Negative', 'Positive']
    
    def load_and_preprocess_data():
        """Load and preprocess IMDB dataset."""
        print("\n📥 Loading IMDB dataset...")
        
        # Load IMDB dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
            num_words=VOCAB_SIZE,
            start_char=1,
            oov_char=2,
            index_from=3
        )
        
        # Pad sequences to fixed length
        x_train = keras.preprocessing.sequence.pad_sequences(
            x_train, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post'
        )
        x_test = keras.preprocessing.sequence.pad_sequences(
            x_test, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post'
        )
        
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
        print(f"Sequence length: {SEQUENCE_LENGTH}")
        print(f"Vocabulary size: {VOCAB_SIZE}")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def create_yat_text_model():
        """Create a text classification model with YAT layers."""
        print("\n🏗️ Building YAT text classification model...")
        
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                input_length=SEQUENCE_LENGTH,
                mask_zero=True
            ),
            
            # LSTM layers for sequence processing
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            
            # YAT dense layers for classification
            YatNMN(256),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            YatNMN(128),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            YatNMN(64),
            layers.Activation('relu'),
            layers.Dropout(0.3),
            
            YatNMN(NUM_CLASSES),
            layers.Activation('softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model compiled successfully!")
        model.summary()
        
        return model
    
    def create_baseline_text_model():
        """Create a baseline text classification model with standard layers."""
        print("\n🏗️ Building baseline text classification model...")
        
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                input_length=SEQUENCE_LENGTH,
                mask_zero=True
            ),
            
            # LSTM layers for sequence processing
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            
            # Standard dense layers for classification
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
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
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=f"/tmp/{model_name.lower()}_best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Train the model
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(model, test_data, model_name="YAT"):
        """Evaluate the model on test data."""
        x_test, y_test = test_data
        
        print(f"\n📊 Evaluating {model_name} model...")
        
        # Evaluate on test set
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
            x_test, y_test, verbose=0
        )
        
        # Calculate F1 score
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        print(f"{model_name} Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")
        
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
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
    
    def plot_training_history(yat_history, baseline_history=None):
        """Plot training history."""
        print("\n📈 Plotting training history...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
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
        
        # Plot precision
        axes[0, 2].plot(yat_history.history['precision'], label='YAT Train Precision')
        axes[0, 2].plot(yat_history.history['val_precision'], label='YAT Val Precision')
        if baseline_history:
            axes[0, 2].plot(baseline_history.history['precision'], label='Baseline Train Precision', linestyle='--')
            axes[0, 2].plot(baseline_history.history['val_precision'], label='Baseline Val Precision', linestyle='--')
        axes[0, 2].set_title('Model Precision')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        
        # Plot recall
        axes[1, 0].plot(yat_history.history['recall'], label='YAT Train Recall')
        axes[1, 0].plot(yat_history.history['val_recall'], label='YAT Val Recall')
        if baseline_history:
            axes[1, 0].plot(baseline_history.history['recall'], label='Baseline Train Recall', linestyle='--')
            axes[1, 0].plot(baseline_history.history['val_recall'], label='Baseline Val Recall', linestyle='--')
        axes[1, 0].set_title('Model Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        
        # Plot learning rate (if available)
        if 'lr' in yat_history.history:
            axes[1, 1].plot(yat_history.history['lr'], label='YAT Learning Rate')
            if baseline_history and 'lr' in baseline_history.history:
                axes[1, 1].plot(baseline_history.history['lr'], label='Baseline Learning Rate', linestyle='--')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')
        else:
            # Plot validation accuracy comparison
            if baseline_history:
                epochs = range(1, len(yat_history.history['val_accuracy']) + 1)
                axes[1, 1].plot(epochs, yat_history.history['val_accuracy'], 'b-', label='YAT Val Accuracy')
                axes[1, 1].plot(epochs, baseline_history.history['val_accuracy'], 'r--', label='Baseline Val Accuracy')
                axes[1, 1].set_title('Validation Accuracy Comparison')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].legend()
        
        # Plot F1 score approximation (2 * precision * recall / (precision + recall))
        yat_f1_train = [2 * p * r / (p + r) if (p + r) > 0 else 0 
                       for p, r in zip(yat_history.history['precision'], yat_history.history['recall'])]
        yat_f1_val = [2 * p * r / (p + r) if (p + r) > 0 else 0 
                     for p, r in zip(yat_history.history['val_precision'], yat_history.history['val_recall'])]
        
        axes[1, 2].plot(yat_f1_train, label='YAT Train F1')
        axes[1, 2].plot(yat_f1_val, label='YAT Val F1')
        
        if baseline_history:
            baseline_f1_train = [2 * p * r / (p + r) if (p + r) > 0 else 0 
                               for p, r in zip(baseline_history.history['precision'], baseline_history.history['recall'])]
            baseline_f1_val = [2 * p * r / (p + r) if (p + r) > 0 else 0 
                             for p, r in zip(baseline_history.history['val_precision'], baseline_history.history['val_recall'])]
            axes[1, 2].plot(baseline_f1_train, label='Baseline Train F1', linestyle='--')
            axes[1, 2].plot(baseline_f1_val, label='Baseline Val F1', linestyle='--')
        
        axes[1, 2].set_title('F1 Score')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('/tmp/keras_imdb_training_history.png', dpi=150, bbox_inches='tight')
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
        plt.savefig(f'/tmp/keras_imdb_{model_name.lower()}_confusion_matrix.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_sample_predictions(model, test_data, num_samples=5):
        """Plot sample predictions with text snippets."""
        x_test, y_test = test_data
        
        print(f"\n🔍 Analyzing sample predictions...")
        
        # Get word index for decoding
        word_index = keras.datasets.imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        reverse_word_index[0] = '<PAD>'
        reverse_word_index[1] = '<START>'
        reverse_word_index[2] = '<UNK>'
        reverse_word_index[3] = '<UNUSED>'
        
        def decode_review(text):
            return ' '.join([reverse_word_index.get(i, '?') for i in text])
        
        # Get predictions
        predictions = model.predict(x_test[:num_samples])
        
        plt.figure(figsize=(15, 10))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 1, i + 1)
            
            # Decode the review
            review = decode_review(x_test[i])
            review_words = review.split()[:50]  # First 50 words
            review_snippet = ' '.join(review_words) + '...'
            
            # Get true and predicted labels
            true_label = np.argmax(y_test[i])
            pred_label = np.argmax(predictions[i])
            confidence = predictions[i][pred_label]
            
            # Create title
            title = f"Sample {i+1}: True={CLASS_NAMES[true_label]}, Pred={CLASS_NAMES[pred_label]} ({confidence:.3f})"
            
            # Color based on correctness
            color = 'green' if true_label == pred_label else 'red'
            
            plt.text(0.05, 0.8, title, transform=plt.gca().transAxes, fontsize=12, 
                    color=color, weight='bold')
            plt.text(0.05, 0.2, review_snippet, transform=plt.gca().transAxes, fontsize=10,
                    wrap=True)
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('/tmp/keras_imdb_sample_predictions.png', dpi=150, bbox_inches='tight')
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
        yat_model = create_yat_text_model()
        
        # Train YAT model
        yat_history = train_model(yat_model, (x_train, y_train), (x_val, y_val), "YAT")
        
        # Evaluate YAT model
        yat_results = evaluate_model(yat_model, (x_test, y_test), "YAT")
        
        # Create and train baseline model for comparison
        print("\n" + "="*65)
        print("Training baseline model for comparison...")
        baseline_model = create_baseline_text_model()
        baseline_history = train_model(baseline_model, (x_train, y_train), (x_val, y_val), "Baseline")
        baseline_results = evaluate_model(baseline_model, (x_test, y_test), "Baseline")
        
        # Plot results
        plot_training_history(yat_history, baseline_history)
        plot_confusion_matrix(yat_results['y_true'], yat_results['y_pred'], "YAT")
        plot_confusion_matrix(baseline_results['y_true'], baseline_results['y_pred'], "Baseline")
        plot_sample_predictions(yat_model, (x_test, y_test))
        
        # Save and load model
        loaded_model = save_and_load_model(yat_model, MODEL_SAVE_PATH)
        
        # Verify loaded model works
        print("\n🔍 Verifying loaded model...")
        loaded_results = evaluate_model(loaded_model, (x_test, y_test), "Loaded YAT")
        
        # Summary comparison
        print("\n" + "="*65)
        print("📋 FINAL COMPARISON SUMMARY")
        print("="*65)
        print(f"YAT Model Test Accuracy: {yat_results['test_accuracy']:.4f}")
        print(f"YAT Model Test F1 Score: {yat_results['test_f1']:.4f}")
        print(f"Baseline Model Test Accuracy: {baseline_results['test_accuracy']:.4f}")
        print(f"Baseline Model Test F1 Score: {baseline_results['test_f1']:.4f}")
        print(f"Loaded Model Test Accuracy: {loaded_results['test_accuracy']:.4f}")
        print(f"YAT vs Baseline Accuracy Difference: {yat_results['test_accuracy'] - baseline_results['test_accuracy']:+.4f}")
        print(f"YAT vs Baseline F1 Difference: {yat_results['test_f1'] - baseline_results['test_f1']:+.4f}")
        
        if yat_results['test_accuracy'] > baseline_results['test_accuracy']:
            print("🎉 YAT model outperformed the baseline!")
        else:
            print("📊 Baseline model performed better than YAT model.")
        
        print("\n✅ Keras language example completed successfully!")
        print(f"📁 Plots saved to /tmp/keras_imdb_*.png")
        print(f"💾 Model saved to {MODEL_SAVE_PATH}")
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Required dependencies not available: {e}")
    print("Install with: pip install 'nmn[keras]' tensorflow-datasets matplotlib seaborn scikit-learn")

except Exception as e:
    print(f"❌ Error running Keras language example: {e}")
    import traceback
    traceback.print_exc()