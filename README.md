# nmn
Not the neurons we want, but the neurons we need

[![PyPI version](https://img.shields.io/pypi/v/nmn.svg)](https://pypi.org/project/nmn/)
[![Downloads](https://static.pepy.tech/badge/nmn)](https://pepy.tech/project/nmn)
[![Downloads/month](https://static.pepy.tech/badge/nmn/month)](https://pepy.tech/project/nmn)
[![GitHub stars](https://img.shields.io/github/stars/mlnomadpy/nmn?style=social)](https://github.com/mlnomadpy/nmn)
[![GitHub forks](https://img.shields.io/github/forks/mlnomadpy/nmn?style=social)](https://github.com/mlnomadpy/nmn)
[![GitHub issues](https://img.shields.io/github/issues/mlnomadpy/nmn)](https://github.com/mlnomadpy/nmn/issues)
[![PyPI - License](https://img.shields.io/pypi/l/nmn)](https://pypi.org/project/nmn/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nmn)](https://pypi.org/project/nmn/)
[![Test Suite](https://github.com/mlnomadpy/nmn/actions/workflows/test.yml/badge.svg)](https://github.com/mlnomadpy/nmn/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/mlnomadpy/nmn/branch/master/graph/badge.svg)](https://codecov.io/gh/mlnomadpy/nmn)

## Features

*   **🔥 Activation-Free Non-linearity:** Learn complex, non-linear relationships without separate activation functions
*   **🌐 Multiple Frameworks:** Full support for Flax (Linen & NNX), Keras, PyTorch, and TensorFlow
*   **🧠 Advanced Layer Types:** Dense (YatNMN), Convolutional (YatConv, YatConvTranspose), Attention (YatAttention)
*   **🔄 Recurrent Layers:** YatSimpleRNN, YatLSTM, YatGRU with custom activation functions
*   **✨ Custom Squashing Functions:** Softermax, softer_sigmoid, soft_tanh for smoother gradients
*   **⚡ DropConnect Support:** Built-in regularization for NNX layers
*   **📐 Novel Operations:** Yat-Product and Yat-Conv based on distance-similarity tradeoff
*   **📚 Research-Driven:** Based on "Deep Learning 2.0/2.1: Artificial Neurons that Matter"

## Overview

**nmn** (Neural-Matter Network) provides cutting-edge neural network layers for multiple frameworks (Flax NNX & Linen, Keras, PyTorch, TensorFlow) that learn non-linearity without requiring traditional activation functions. The library introduces the **Yat-Product** operation—a novel approach that combines similarity and distance metrics to create inherently non-linear transformations.

### Key Innovations

1. **Distance-Similarity Tradeoff**: Unlike traditional neurons that rely on dot products and activations, Yat neurons balance alignment (similarity) with proximity (distance)
2. **Built-in Non-linearity**: The squared-ratio operation eliminates the need for ReLU, sigmoid, or tanh activations
3. **Geometric Interpretation**: Maximizes response when weights and inputs are aligned, close, and large in magnitude
4. **Production-Ready**: Comprehensive implementations across major deep learning frameworks

### Theoretical Foundation

Inspired by the papers:
> **Deep Learning 2.0**: Artificial Neurons that Matter - Reject Correlation, Embrace Orthogonality
>
> **Deep Learning 2.1**: Mind and Cosmos - Towards Cosmos-Inspired Interpretable Neural Networks

## Math

Yat-Product: 
$$
ⵟ(\mathbf{w},\mathbf{x}) := \frac{\langle \mathbf{w}, \mathbf{x} \rangle^2}{\|\mathbf{w} - \mathbf{x}\|^2 + \epsilon} = \frac{ \|\mathbf{x}\|^2  \|\mathbf{w}\|^2 \cos^2 \theta}{\|\mathbf{w}\|^2 - 2\mathbf{w}^\top\mathbf{x} + \|\mathbf{x}\|^2 + \epsilon} = \frac{ \|\mathbf{x}\|^2  \|\mathbf{w}\|^2 \cos^2 \theta}{((\mathbf{x}-\mathbf{w})\cdot(\mathbf{x}-\mathbf{w}))^2 + \epsilon}.
$$

**Explanation:**
- $\mathbf{w}$ is the weight vector, $\mathbf{x}$ is the input vector.
- $\langle \mathbf{w}, \mathbf{x} \rangle$ is the dot product between $\mathbf{w}$ and $\mathbf{x}$.
- $\|\mathbf{w} - \mathbf{x}\|^2$ is the squared Euclidean distance between $\mathbf{w}$ and $\mathbf{x}$.
- $\epsilon$ is a small constant for numerical stability.
- $\theta$ is the angle between $\mathbf{w}$ and $\mathbf{x}$.

This operation:
- **Numerator:** Squares the similarity (dot product) between $\mathbf{w}$ and $\mathbf{x}$, emphasizing strong alignments.
- **Denominator:** Penalizes large distances, so the response is high only when $\mathbf{w}$ and $\mathbf{x}$ are both similar in direction and close in space.
- **No activation needed:** The non-linearity is built into the operation itself, allowing the layer to learn complex, non-linear relationships without a separate activation function.
- **Geometric view:** The output is maximized when $\mathbf{w}$ and $\mathbf{x}$ are both large in norm, closely aligned (small $\theta$), and close together in Euclidean space.

Yat-Conv:
$$
ⵟ^*(\mathbf{W}, \mathbf{X}) := \frac{\langle \mathbf{w}, \mathbf{x} \rangle^2}{\|\mathbf{w} - \mathbf{x}\|^2 + \epsilon}
= \frac{\left(\sum_{i,j} w_{ij} x_{ij}\right)^2}{\sum_{i,j} (w_{ij} - x_{ij})^2 + \epsilon}
$$

Where:
- $\mathbf{W}$ and $\mathbf{X}$ are local patches (e.g., kernel and input patch in convolution)
- $w_{ij}$ and $x_{ij}$ are elements of the kernel and input patch, respectively
- $\epsilon$ is a small constant for numerical stability

This generalizes the Yat-product to convolutional (patch-wise) operations.


## Supported Frameworks & API

The library provides several layer types across all supported frameworks:

### Core Layers

| Layer Type | Description | Flax NNX | Flax Linen | Keras | PyTorch | TensorFlow |
|------------|-------------|----------|------------|-------|---------|------------|
| **YatNMN** | Dense/Linear layer | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConv** | Convolutional layer | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConvTranspose** | Transposed convolution | ✅ | 🚧 | 🚧 | 🚧 | 🚧 |
| **YatAttention** | Multi-head attention | ✅ | 🚧 | 🚧 | 🚧 | 🚧 |

### Recurrent Layers (Flax NNX)

| Layer Type | Description | Status |
|------------|-------------|--------|
| **YatSimpleCell** | Simple RNN cell | ✅ |
| **YatLSTMCell** | LSTM cell with Yat operations | ✅ |
| **YatGRUCell** | GRU cell with Yat operations | ✅ |

### Custom Activation Functions (Flax NNX)

| Function | Formula | Description |
|----------|---------|-------------|
| **softermax** | $\frac{x_k^n}{\epsilon + \sum_i x_i^n}$ | Generalized softmax with power parameter |
| **softer_sigmoid** | $\frac{x^n}{1 + x^n}$ | Smoother sigmoid variant |
| **soft_tanh** | $\frac{x^n}{1 + x^n} - \frac{(-x)^n}{1 + (-x)^n}$ | Smoother tanh variant |

### Advanced Features (Flax NNX)

*   **✅ DropConnect**: Built-in weight dropout for regularization
*   **✅ Alpha Scaling**: Learnable output scaling parameter
*   **🚧 Ternary Networks**: Quantized weight versions (in development)
*   **✅ Custom Initializers**: Optimized for Yat-layer convergence

*Legend: ✅ Implemented, 🚧 In Development*

## Installation

### Basic Installation
```bash
pip install nmn
```

### Framework-Specific Installation

For optimal performance and full feature access, install with framework-specific dependencies:

```bash
# For JAX/Flax (NNX and Linen)
pip install "nmn[nnx]"      # or "nmn[linen]"

# For PyTorch 
pip install "nmn[torch]"

# For TensorFlow/Keras
pip install "nmn[keras]"    # or "nmn[tf]"

# For all frameworks
pip install "nmn[all]"

# For development and testing
pip install "nmn[dev]"      # Basic dev tools
pip install "nmn[test]"     # All dependencies for testing
```

### Development Installation

```bash
git clone https://github.com/mlnomadpy/nmn.git
cd nmn
pip install -e ".[dev]"
```

## Usage Example (Flax NNX)

### Basic Dense Layer

```python
import jax
import jax.numpy as jnp
from flax import nnx
from nmn.nnx.nmn import YatNMN

# Create a simple Yat dense layer
model_key, param_key, drop_key, input_key = jax.random.split(jax.random.key(0), 4)
layer = YatNMN(
    in_features=3, 
    out_features=4, 
    rngs=nnx.Rngs(params=param_key, dropout=drop_key)
)

# Forward pass
dummy_input = jax.random.normal(input_key, (2, 3))  # Batch size 2
output = layer(dummy_input)
print("YatNMN Output Shape:", output.shape)  # (2, 4)
```

### Convolutional Layer

```python
from nmn.nnx.yatconv import YatConv

# Create a Yat convolutional layer
conv_layer = YatConv(
    in_features=3,
    out_features=8,
    kernel_size=(3, 3),
    strides=1,
    padding='SAME',
    rngs=nnx.Rngs(params=jax.random.key(1))
)

# Forward pass on image data
image = jax.random.normal(jax.random.key(2), (1, 28, 28, 3))
conv_output = conv_layer(image)
print("YatConv Output Shape:", conv_output.shape)  # (1, 28, 28, 8)
```

### Attention Layer

```python
from nmn.nnx.yatattention import YatMultiHeadAttention

# Create multi-head attention with Yat-product
attention = YatMultiHeadAttention(
    num_heads=8,
    in_features=512,
    qkv_features=512,
    out_features=512,
    use_softermax=True,  # Use custom softermax activation
    rngs=nnx.Rngs(params=jax.random.key(3))
)

# Forward pass
seq = jax.random.normal(jax.random.key(4), (2, 10, 512))  # (batch, seq_len, features)
attn_output = attention(seq)
print("Attention Output Shape:", attn_output.shape)  # (2, 10, 512)
```

### Recurrent Layers

```python
from nmn.nnx.rnn import YatLSTMCell, YatGRUCell

# LSTM cell
lstm_cell = YatLSTMCell(
    in_features=64,
    hidden_features=128,
    rngs=nnx.Rngs(params=jax.random.key(5))
)

# GRU cell
gru_cell = YatGRUCell(
    in_features=64,
    hidden_features=128,
    rngs=nnx.Rngs(params=jax.random.key(6))
)

# Initialize carry state
batch_size = 2
carry = lstm_cell.initialize_carry(jax.random.key(7), (batch_size,))

# Process sequence
x = jax.random.normal(jax.random.key(8), (batch_size, 64))
new_carry, output = lstm_cell(carry, x)
```

### DropConnect Regularization

```python
# Enable DropConnect for regularization
layer_with_dropout = YatNMN(
    in_features=128,
    out_features=64,
    use_dropconnect=True,
    drop_rate=0.2,  # 20% dropout on weights
    rngs=nnx.Rngs(params=jax.random.key(9), dropout=jax.random.key(10))
)

# Training mode (with dropout)
x_train = jax.random.normal(jax.random.key(11), (32, 128))
y_train = layer_with_dropout(x_train, deterministic=False)

# Inference mode (no dropout)
x_test = jax.random.normal(jax.random.key(12), (32, 128))
y_test = layer_with_dropout(x_test, deterministic=True)
```

*Note: For examples in other frameworks (PyTorch, Keras, TensorFlow, Linen), see the [`examples/`](examples/) directory.*

## Examples and Documentation

The [`examples/`](examples/) directory contains comprehensive, production-ready examples for all supported frameworks:

### 📁 Directory Structure

```
examples/
├── nnx/                    # Flax NNX (most feature-complete)
│   ├── vision/
│   │   └── cnn_cifar.py    # Complete CNN training on CIFAR-10
│   └── language/
│       └── mingpt.py        # GPT implementation with Yat layers
├── torch/                   # PyTorch examples
│   ├── yat_examples.py      # Basic usage patterns
│   ├── yat_cifar10.py       # CIFAR-10 training
│   └── vision/
│       └── resnet_training.py  # ResNet with Yat layers
├── keras/                   # Keras examples
│   ├── basic_usage.py       # Getting started
│   ├── vision_cifar10.py    # Image classification
│   └── language_imdb.py     # Text classification
├── tensorflow/              # TensorFlow examples
│   ├── basic_usage.py
│   ├── vision_cifar10.py
│   └── language_imdb.py
├── linen/                   # Flax Linen examples
│   └── basic_usage.py
└── comparative/             # Framework comparisons
    └── framework_comparison.py  # Side-by-side comparison
```

### 🚀 Quick Start Examples

**Check Framework Availability:**
```bash
python examples/comparative/framework_comparison.py
```

**Train Vision Model:**
```bash
# PyTorch - CIFAR-10 with Yat convolutions
python examples/torch/yat_cifar10.py

# Keras - CIFAR-10 classification
python examples/keras/vision_cifar10.py

# Flax NNX - Full CNN with data augmentation
python examples/nnx/vision/cnn_cifar.py
```

**Natural Language Processing:**
```bash
# GPT-style language model (Flax NNX)
python examples/nnx/language/mingpt.py

# Sentiment analysis (Keras)
python examples/keras/language_imdb.py
```

**Basic Usage Patterns:**
```bash
# PyTorch introduction
python examples/torch/yat_examples.py

# Keras getting started
python examples/keras/basic_usage.py
```

### 📚 Example Features

Each example includes:
- **Complete training loops** with metrics tracking
- **Data loading and preprocessing** using standard datasets
- **Model architecture definitions** with Yat layers
- **Hyperparameter configurations** and best practices
- **Evaluation and visualization** code
- **Documentation** explaining design choices

### 🎯 Real-World Applications

The examples demonstrate:
- **Computer Vision**: Image classification, feature extraction, CNNs
- **Natural Language Processing**: Language modeling, sentiment analysis, attention mechanisms
- **Regularization**: DropConnect, weight decay, data augmentation
- **Optimization**: Learning rate schedules, gradient clipping, optimizer tuning
- **Production Patterns**: Model checkpointing, logging, reproducibility

## Testing

The library includes a comprehensive test suite with high code coverage:

### Running Tests

```bash
# Install test dependencies
pip install "nmn[test]"

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=nmn --cov-report=html --cov-report=term

# Run specific framework tests
pytest tests/test_nnx/        # Flax NNX tests
pytest tests/test_torch/      # PyTorch tests
pytest tests/test_keras/      # Keras tests
pytest tests/test_tf/         # TensorFlow tests
pytest tests/test_linen/      # Flax Linen tests

# Run integration tests
pytest tests/integration/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_nnx/test_basic.py -v
```

### Test Structure

```
tests/
├── test_nnx/              # NNX layer tests
│   └── test_basic.py      # YatNMN, YatConv, Attention, RNN
├── test_torch/            # PyTorch layer tests
│   └── test_basic.py
├── test_keras/            # Keras layer tests
│   └── test_keras_basic.py
├── test_tf/               # TensorFlow layer tests
│   └── test_tf_basic.py
├── test_linen/            # Linen layer tests
│   └── test_basic.py
└── integration/           # Cross-framework compatibility
    └── test_compatibility.py
```

### Continuous Integration

The project uses GitHub Actions for automated testing:
- ✅ Tests run on every push and pull request
- ✅ Coverage reports uploaded to Codecov
- ✅ Multi-framework compatibility verified
- ✅ Code quality checks (formatting, linting)

## Roadmap

### ✅ Completed

-   **Core Architecture**: Yat-Product and Yat-Conv operations across all frameworks
-   **Flax NNX Features**: Dense, Conv, ConvTranspose, Attention, RNN cells (Simple, LSTM, GRU)
-   **Custom Activations**: Softermax, softer_sigmoid, soft_tanh
-   **Regularization**: DropConnect support in NNX
-   **Multi-Framework**: Full implementations for NNX, Linen, Keras, PyTorch, TensorFlow
-   **Examples**: Comprehensive examples for vision and language tasks
-   **Testing**: Complete test suite with CI/CD and code coverage
-   **Documentation**: API documentation and usage examples

### 🚧 In Progress

-   **Ternary Networks**: Quantized weight versions of Yat layers
-   **Additional RNN Features**: Bidirectional RNN wrappers, stacked RNNs
-   **Advanced Attention**: Sparse attention, linear attention variants
-   **Performance**: JIT compilation optimizations, memory efficiency improvements
-   **Documentation**: API reference website, tutorials, benchmark results

### 📋 Planned

-   **Graph Neural Networks**: Yat-based message passing layers
-   **Normalization Layers**: YatBatchNorm, YatLayerNorm variants
-   **Advanced Regularization**: Spectral normalization, weight normalization
-   **Mixed Precision**: FP16/BF16 training support
-   **Model Zoo**: Pre-trained models for common tasks
-   **Benchmarks**: Comprehensive performance comparisons with standard layers
-   **Interactive Demos**: Colab notebooks, live web demos

## Contributing

We welcome contributions from the community! Here's how you can help:

### Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/yourusername/nmn.git
   cd nmn
   ```

2. **Install in Development Mode**:
   ```bash
   pip install -e ".[dev]"
   # or install all dependencies for testing
   pip install -e ".[test]"
   ```

3. **Install Pre-commit Hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Code Quality

We maintain high code quality standards:

```bash
# Format code with Black
black src/ tests/ examples/

# Check code style
flake8 src/ tests/

# Sort imports
isort src/ tests/ examples/

# Type checking
mypy src/nmn
```

### Testing

Ensure all tests pass before submitting:

```bash
# Run full test suite
pytest tests/ -v

# Run tests for specific framework
pytest tests/test_nnx/ -v

# Check code coverage
pytest tests/ --cov=nmn --cov-report=term-missing
```

### Contribution Guidelines

1. **Bug Reports**: Open an issue with:
   - Clear description of the problem
   - Minimal reproducible example
   - Expected vs actual behavior
   - System information (OS, Python version, framework versions)

2. **Feature Requests**: Open an issue describing:
   - The proposed feature and its use case
   - Why it belongs in the library
   - Potential implementation approach

3. **Pull Requests**:
   - Create a new branch for your feature/fix
   - Write tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass and code is formatted
   - Reference related issues in PR description

4. **Documentation**: Help improve:
   - API documentation and docstrings
   - Usage examples and tutorials
   - README and guides
   - Example code and notebooks

### Areas for Contribution

- 🐛 **Bug Fixes**: Check [open issues](https://github.com/mlnomadpy/nmn/issues)
- ✨ **New Features**: Implement layers for additional frameworks
- 📚 **Documentation**: Improve guides, add examples
- 🧪 **Testing**: Increase test coverage, add edge cases
- ⚡ **Performance**: Optimize implementations, add benchmarks
- 🎨 **Examples**: Add new use cases and applications

## License

This project is licensed under the **GNU Affero General Public License v3 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

### What This Means

- ✅ **Free to use** for personal, academic, and commercial projects
- ✅ **Modify and distribute** with attribution
- ⚠️ **Network use is distribution**: If you run modified code on a server accessible over a network, you must make the modified source code available
- ✅ **Compatible** with open-source research and academic use

For commercial applications where AGPL terms are not suitable, please contact us to discuss alternative licensing options.

## API Reference

### Quick Reference by Framework

**Flax NNX** (Most Feature-Complete):
```python
from nmn.nnx.nmn import YatNMN
from nmn.nnx.yatconv import YatConv, YatConvTranspose
from nmn.nnx.yatattention import YatMultiHeadAttention, YatSelfAttention
from nmn.nnx.rnn import YatSimpleCell, YatLSTMCell, YatGRUCell
from nmn.nnx.squashers import softermax, softer_sigmoid, soft_tanh
```

**PyTorch**:
```python
from nmn.torch.nmn import YatNMN
from nmn.torch.conv import YatConv
```

**Keras/TensorFlow**:
```python
from nmn.keras.nmn import YatNMN
from nmn.keras.conv import YatConv
```

**Flax Linen**:
```python
from nmn.linen.nmn import YatNMN
```

### Common Parameters

**YatNMN** (Dense Layer):
- `in_features`: Input dimension
- `out_features`: Output dimension
- `use_bias`: Whether to include bias term (default: True)
- `use_alpha`: Whether to use learnable scaling (default: True)
- `use_dropconnect`: Enable DropConnect regularization (default: False)
- `drop_rate`: DropConnect probability (default: 0.0)
- `epsilon`: Numerical stability constant (default: 1e-5)

**YatConv** (Convolutional Layer):
- `in_features`: Number of input channels
- `out_features`: Number of output channels
- `kernel_size`: Convolution kernel size (int or tuple)
- `strides`: Stride of convolution (default: 1)
- `padding`: Padding mode ('SAME', 'VALID', or tuple)
- `use_bias`, `use_alpha`, `epsilon`: Same as YatNMN

**YatAttention** (Attention Layer):
- `num_heads`: Number of attention heads
- `in_features`: Input feature dimension
- `qkv_features`: Query/Key/Value dimension
- `out_features`: Output dimension
- `use_softermax`: Use custom softermax instead of softmax
- `epsilon`: Numerical stability for distance calculation

## Citation

If you use **nmn** in your research or projects, please cite the foundational papers:

```bibtex
@article{bouhsine2024dl2,
  author    = {Taha Bouhsine},
  title     = {Deep Learning 2.0: Artificial Neurons that Matter - Reject Correlation, Embrace Orthogonality},
  year      = {2024},
  note      = {Foundational paper for Yat-Product operation}
}
```

```bibtex
@article{bouhsine2025dl21,
  author    = {Taha Bouhsine},
  title     = {Deep Learning 2.1: Mind and Cosmos - Towards Cosmos-Inspired Interpretable Neural Networks},
  year      = {2025},
  note      = {Extended theoretical framework}
}
```

### Acknowledgments

This library is inspired by research into alternative neural network architectures that challenge conventional wisdom about how artificial neurons should operate. Special thanks to the JAX, Flax, PyTorch, and TensorFlow communities for creating the foundational tools that make this work possible.

## Support and Community

- 📖 **Documentation**: [GitHub Wiki](https://github.com/mlnomadpy/nmn/wiki) (coming soon)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/mlnomadpy/nmn/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/mlnomadpy/nmn/discussions)
- 📧 **Contact**: yat@mlnomads.com

## Related Projects

- [JAX](https://github.com/google/jax) - Composable transformations of Python+NumPy programs
- [Flax](https://github.com/google/flax) - Neural network library for JAX
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [TensorFlow](https://www.tensorflow.org/) - End-to-end machine learning platform

---

**Built with ❤️ by the ML Nomads team**
