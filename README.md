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

## Overview

**nmn** provides neural network layers for multiple frameworks (Flax, NNX, Keras, PyTorch, TensorFlow) that do not require activation functions to learn non-linearity. The main goal is to enable deep learning architectures where the layer itself is inherently non-linear, inspired by the paper:

> Deep Learning 2.0: Artificial Neurons that Matter: Reject Correlation - Embrace Orthogonality

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


## Supported Frameworks & Tasks

### Flax (JAX)
- `YatNMN` layer implemented in `src/nmn/linen/nmn.py`
- **Tasks:**
  - [x] Core layer implementation
  - [ ] Recurrent layer (to be implemented)

### NNX (Flax NNX)
- `YatNMN` layer implemented in `src/nmn/nnx/nmn.py`
- **Tasks:**
  - [x] Core layer implementation
  - [ ] Recurrent layer (to be implemented)

### Keras
- `YatNMN` layer implemented in `src/nmn/keras/nmn.py`
- **Tasks:**
  - [x] Core layer implementation
  - [ ] Recurrent layer (to be implemented)

### PyTorch
- `YatNMN` layer implemented in `src/nmn/torch/nmn.py`
- **Tasks:**
  - [x] Core layer implementation
  - [ ] Recurrent layer (to be implemented)

### TensorFlow
- `YatNMN` layer implemented in `src/nmn/tf/nmn.py`
- **Tasks:**
  - [x] Core layer implementation
  - [ ] Recurrent layer (to be implemented)

## Installation

```bash
pip install nmn
```

## Usage Example (Flax)

```python
from nmn.nnx.nmn import YatNMN
from nmn.nnx.yatconv import YatConv
# ... use as a Flax module ...
```

## Roadmap
- [ ] Implement recurrent layers for all frameworks
- [ ] Add more examples and benchmarks
- [ ] Improve documentation and API consistency

## License
GNU Affero General Public License v3
