# NMN Examples

This directory contains comprehensive examples demonstrating the usage of Neural-Matter Network (NMN) layers across different deep learning frameworks, including complete training pipelines for vision and language tasks.

## Directory Structure

- **`nnx/`** - Flax NNX examples (JAX-based)
- **`torch/`** - PyTorch examples
- **`keras/`** - Keras/TensorFlow examples with complete training pipelines
- **`tensorflow/`** - TensorFlow examples with custom training loops
- **`linen/`** - Flax Linen examples (JAX-based)
- **`comparative/`** - Cross-framework comparison examples

## Framework Support

| Framework | Status | Basic | Vision | Language |
|-----------|--------|-------|--------|----------|
| Flax NNX  | ✅ | ✅ | ✅ | ✅ |
| PyTorch   | ✅ | ✅ | ✅ | ❌ |
| Keras     | ✅ | ✅ | ✅ | ✅ |
| TensorFlow| ✅ | ✅ | ✅ | ✅ |
| Flax Linen| ✅ | ✅ | ❌ | ❌ |

## Quick Start

### Basic Usage Examples

Test basic functionality with minimal examples:

```bash
# Keras basic example
python examples/keras/basic_usage.py

# TensorFlow basic example
python examples/tensorflow/basic_usage.py
```

### Comprehensive Training Examples

Run full training pipelines with baseline comparisons:

```bash
# Keras vision example (CIFAR-10)
python examples/keras/vision_cifar10.py

# TensorFlow vision example (CIFAR-10) 
python examples/tensorflow/vision_cifar10.py

# Keras language example (IMDB sentiment)
python examples/keras/language_imdb.py

# TensorFlow language example (IMDB sentiment)
python examples/tensorflow/language_imdb.py
```

Each comprehensive example includes:
- **Data loading and preprocessing** (with augmentation for vision)
- **Model definition** with YAT layers
- **Baseline model comparison** using standard layers
- **Training loop** with validation and early stopping
- **Testing/evaluation** with detailed metrics
- **Model saving and loading** verification
- **Performance visualization** (training curves, confusion matrices)

## Installation Requirements

Install dependencies for the examples you want to run:

```bash
# For Keras/TensorFlow comprehensive examples
pip install 'nmn[keras]' tensorflow-datasets matplotlib seaborn scikit-learn

# For TensorFlow examples
pip install 'nmn[tf]' tensorflow-datasets matplotlib seaborn scikit-learn

# For NNX/Linen examples
pip install "nmn[nnx]"

# For PyTorch examples  
pip install "nmn[torch]"

# For all frameworks
pip install "nmn[all]"

# For development and testing
pip install "nmn[test]"
```

## Example Features

### Vision Examples (CIFAR-10)
- **Dataset**: CIFAR-10 image classification (10 classes)
- **Architecture**: CNN with YAT convolutional and dense layers
- **Comparison**: YAT vs standard Conv2D/Dense layers
- **Features**: Data augmentation, batch normalization, dropout
- **Metrics**: Accuracy, top-2 accuracy, confusion matrices
- **Runtime**: ~10-30 minutes depending on hardware

### Language Examples (IMDB)
- **Dataset**: IMDB movie review sentiment analysis
- **Architecture**: LSTM + YAT dense layers for classification
- **Comparison**: YAT vs standard Dense layers
- **Features**: Text vectorization, embeddings, sequence processing
- **Metrics**: Accuracy, precision, recall, F1-score
- **Runtime**: ~15-45 minutes depending on hardware

## Results and Outputs

All comprehensive examples save results to `/tmp/` directory:
- **Models**: Saved in framework-appropriate format
- **Plots**: Training curves, confusion matrices, sample predictions
- **Reports**: Detailed performance comparisons

## Running Examples

Navigate to the specific framework directory and run examples:

```bash
# Basic examples (quick test)
python examples/keras/basic_usage.py
python examples/tensorflow/basic_usage.py

# Comprehensive examples (full training)
python examples/keras/vision_cifar10.py
python examples/tensorflow/language_imdb.py
```

Most examples detect and use GPU automatically if available.