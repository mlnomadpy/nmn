# NMN Examples

This directory contains comprehensive examples demonstrating the usage of Neural-Matter Network (NMN) layers across different deep learning frameworks.

## Directory Structure

- **`nnx/`** - Flax NNX examples (JAX-based)
- **`torch/`** - PyTorch examples
- **`keras/`** - Keras/TensorFlow examples
- **`tensorflow/`** - TensorFlow examples
- **`linen/`** - Flax Linen examples (JAX-based)
- **`comparative/`** - Cross-framework comparison examples

## Framework Support

| Framework | Status | Examples Available |
|-----------|--------|-------------------|
| Flax NNX  | ✅ | Dense layers, Convolutions, Vision tasks |
| PyTorch   | ✅ | Convolutions, CIFAR-10 training |
| Keras     | ✅ | Dense layers |
| TensorFlow| ✅ | Dense layers |
| Flax Linen| ✅ | Dense layers |

## Quick Start

Each framework directory contains:
- **Basic usage examples** - Simple layer instantiation and forward passes
- **Training examples** - Complete training loops for real tasks
- **Advanced examples** - Feature demonstrations and comparisons

## Installation Requirements

Different examples require different dependencies. Install the required framework:

```bash
# For NNX/Linen examples
pip install "nmn[nnx]"

# For PyTorch examples  
pip install "nmn[torch]"

# For Keras/TensorFlow examples
pip install "nmn[keras]"

# For all frameworks
pip install "nmn[all]"

# For development and testing
pip install "nmn[test]"
```

## Running Examples

Navigate to the specific framework directory and follow the README instructions there. Most examples can be run directly:

```bash
cd examples/torch/
python basic_usage.py

cd examples/nnx/
python vision_example.py
```