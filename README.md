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

## Features

*   **Activation-Free Non-linearity:** Learns complex, non-linear relationships without separate activation functions.
*   **Multiple Frameworks:** Supports Flax (Linen & NNX), Keras, PyTorch, and TensorFlow.
*   **Yat-Product & Yat-Conv:** Implements novel Yat-Product and Yat-Conv operations.
*   **Inspired by Research:** Based on the principles from "Deep Learning 2.0/2.1: Artificial Neurons that Matter".

## Overview

**nmn** provides neural network layers for multiple frameworks (Flax, NNX, Keras, PyTorch, TensorFlow) that do not require activation functions to learn non-linearity. The main goal is to enable deep learning architectures where the layer itself is inherently non-linear, inspired by the papers:

> Deep Learning 2.0: Artificial Neurons that Matter: Reject Correlation - Embrace Orthogonality
>
> Deep Learning 2.1: Deep Learning 2.1: Mind and Cosmos - Towards Cosmos-Inspired Interpretable Neural Networks

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

The `YatNMN` layer (for dense operations) and `YatConv` (for convolutional operations) are the core components. Below is a summary of their availability and features per framework:

| Framework      | `YatNMN` Path                 | `YatConv` Path                | Core Layer | DropConnect | Ternary Network | Recurrent Layer |
|----------------|-------------------------------|-------------------------------|------------|-------------|-----------------|-----------------|
| **Flax (Linen)** | `src/nmn/linen/nmn.py`        | (Available)                   | ✅         |             |                 | 🚧              |
| **Flax (NNX)**   | `src/nmn/nnx/nmn.py`          | `src/nmn/nnx/yatconv.py`      | ✅         | ✅          | 🚧              | 🚧              |
| **Keras**      | `src/nmn/keras/nmn.py`        | (Available)                   | ✅         |             |                 | 🚧              |
| **PyTorch**    | `src/nmn/torch/nmn.py`        | (Available)                   | ✅         |             |                 | 🚧              |
| **TensorFlow** | `src/nmn/tf/nmn.py`           | (Available)                   | ✅         |             |                 | 🚧              |

*Legend: ✅ Implemented, 🚧 To be implemented / In Progress, (Available) - Assumed available if NMN is, specific path might vary or be part of the NMN module.*

## Installation

```bash
pip install nmn
```

## Usage Example (Flax NNX)

```python
import jax
import jax.numpy as jnp
from flax import nnx
from nmn.nnx.nmn import YatNMN
from nmn.nnx.yatconv import YatConv

# Example YatNMN (Dense Layer)
model_key, param_key, drop_key, input_key = jax.random.split(jax.random.key(0), 4)
in_features, out_features = 3, 4
layer = YatNMN(in_features=in_features, out_features=out_features, rngs=nnx.Rngs(params=param_key, dropout=drop_key))
dummy_input = jax.random.normal(input_key, (2, in_features)) # Batch size 2
output = layer(dummy_input)
print("YatNMN Output Shape:", output.shape)

# Example YatConv (Convolutional Layer)
conv_key, conv_param_key, conv_input_key = jax.random.split(jax.random.key(1), 3)
in_channels, out_channels = 3, 8
kernel_size = (3, 3)
conv_layer = YatConv(
    in_features=in_channels,
    out_features=out_channels,
    kernel_size=kernel_size,
    rngs=nnx.Rngs(params=conv_param_key)
)
dummy_conv_input = jax.random.normal(conv_input_key, (1, 28, 28, in_channels)) # Batch 1, 28x28 image, in_channels
conv_output = conv_layer(dummy_conv_input)
print("YatConv Output Shape:", conv_output.shape)

```
*Note: Examples for other frameworks (Keras, PyTorch, TensorFlow, Flax Linen) can be found in their respective `nmn.<framework>` modules and upcoming documentation.*

## Roadmap

-   [ ] Implement recurrent layers (`YatRNN`, `YatLSTM`, `YatGRU`) for all supported frameworks.
-   [ ] Develop Ternary Network versions of Yat layers for NNX.
-   [ ] Add more comprehensive examples and benchmark scripts for various tasks (vision, language).
-   [ ] Publish detailed documentation and API references.
-   [ ] Conduct and publish thorough performance benchmarks against traditional layers.

## Contributing

Contributions are welcome! If you'd like to contribute, please feel free to:
-   Open an issue on the [Bug Tracker](https://github.com/mlnomadpy/nmn/issues) to report bugs or suggest features.
-   Submit a pull request with your improvements.
-   Help expand the documentation or add more examples.

## License

This project is licensed under the **GNU Affero General Public License v3**. See the [LICENSE](LICENSE) file for details.

## Citation

If you use `nmn` in your research, please consider citing the original papers that inspired this work:

> Deep Learning 2.0: Artificial Neurons that Matter: Reject Correlation - Embrace Orthogonality
>
> Deep Learning 2.1: Mind and Cosmos - Towards Cosmos-Inspired Interpretable Neural Networks

A BibTeX entry will be provided once the accompanying paper for this library is published.

## Citing

If you use this work, please cite the paper:

```bibtex
@article{taha2024dl2,
  author    = {Taha Bouhsine},
  title     = {Deep Learning 2.0: Artificial Neurons that Matter: Reject Correlation - Embrace Orthogonality},
}
```


```bibtex
@article{taha2025dl2,
  author    = {Taha Bouhsine},
  title     = {Deep Learning 2.1: Mind and Cosmos - Towards Cosmos-Inspired Interpretable Neural Networks},
}
```
