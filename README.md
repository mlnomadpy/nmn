<p align="center">
  <img src="https://raw.githubusercontent.com/mlnomadpy/nmn/master/assets/logo.png" alt="NMN Logo" width="200" height="200" onerror="this.style.display='none'">
</p>

<h1 align="center">⚛️ NMN — Neural Matter Networks</h1>

<p align="center">
  <strong>Not the neurons we want, but the neurons we need</strong>
</p>

<p align="center">
  <em>Activation-free neural layers that learn non-linearity through geometric operations</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/nmn/"><img src="https://img.shields.io/pypi/v/nmn.svg?style=flat-square&color=blue" alt="PyPI version"></a>
  <a href="https://pepy.tech/project/nmn"><img src="https://static.pepy.tech/badge/nmn?style=flat-square" alt="Downloads"></a>
  <a href="https://github.com/mlnomadpy/nmn"><img src="https://img.shields.io/github/stars/mlnomadpy/nmn?style=flat-square&color=yellow" alt="GitHub stars"></a>
  <a href="https://github.com/mlnomadpy/nmn/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/mlnomadpy/nmn/test.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://codecov.io/gh/mlnomadpy/nmn"><img src="https://img.shields.io/codecov/c/github/mlnomadpy/nmn?style=flat-square" alt="Coverage"></a>
  <a href="https://pypi.org/project/nmn/"><img src="https://img.shields.io/pypi/pyversions/nmn?style=flat-square" alt="Python"></a>
  <a href="https://pypi.org/project/nmn/"><img src="https://img.shields.io/pypi/l/nmn?style=flat-square&color=green" alt="License"></a>
</p>

---

## 🎯 TL;DR

**NMN** replaces traditional `Linear + ReLU` with a single geometric operation that learns non-linearity without activation functions:

```python
# Traditional approach
y = relu(linear(x))  # dot product → activation

# NMN approach  
y = yat(x)  # geometric operation with built-in non-linearity
```

The **Yat-Product** (ⵟ) balances *similarity* and *distance* to create inherently non-linear transformations—no activations needed.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔥 **Activation-Free** | Learn complex non-linear relationships without ReLU, sigmoid, or tanh |
| 🌐 **Multi-Framework** | PyTorch, TensorFlow, Keras, Flax (Linen & NNX) |
| 🧮 **Geometric Foundation** | Based on distance-similarity tradeoff, not just correlations |
| ✅ **Cross-Framework Consistency** | Verified numerical equivalence across all frameworks |
| 🧠 **Complete Layer Suite** | Dense, Conv1D/2D/3D, ConvTranspose, Attention, RNN cells |
| ⚡ **Production Ready** | Comprehensive tests, CI/CD, high code coverage |

---

## 📐 The Mathematics

### Yat-Product (ⵟ)

The core operation that powers NMN:

$$
ⵟ(\mathbf{w}, \mathbf{x}) = \frac{\langle \mathbf{w}, \mathbf{x} \rangle^2}{\|\mathbf{w} - \mathbf{x}\|^2 + \epsilon}
$$

<details>
<summary><strong>🔍 Geometric Interpretation (click to expand)</strong></summary>

Rewriting in terms of norms and angles:

$$
ⵟ(\mathbf{w}, \mathbf{x}) = \frac{\|\mathbf{w}\|^2 \|\mathbf{x}\|^2 \cos^2\theta}{\|\mathbf{w}\|^2 - 2\langle\mathbf{w}, \mathbf{x}\rangle + \|\mathbf{x}\|^2 + \epsilon}
$$

**Output is maximized when:**
- ✅ Vectors are **aligned** (small θ → large cos²θ)
- ✅ Vectors are **close** (small Euclidean distance)
- ✅ Vectors have **large magnitude** (amplifies the signal)

**This creates a fundamentally different learning dynamic:**

| Traditional Neuron | Yat Neuron |
|-------------------|------------|
| Measures correlation only | Balances similarity AND proximity |
| Requires activation for non-linearity | Non-linearity is intrinsic |
| Can fire for distant but aligned vectors | Penalizes distance between w and x |

</details>

### Yat-Convolution (ⵟ*)

The same principle applied to local patches:

$$
ⵟ^*(\mathbf{W}, \mathbf{X}) = \frac{(\sum_{i,j} w_{ij} \cdot x_{ij})^2}{\sum_{i,j}(w_{ij} - x_{ij})^2 + \epsilon}
$$

Where **W** is the kernel and **X** is the input patch.

---

## 🚀 Quick Start

### Installation

```bash
pip install nmn

# Framework-specific installations
pip install "nmn[torch]"    # PyTorch
pip install "nmn[keras]"    # Keras/TensorFlow  
pip install "nmn[nnx]"      # Flax NNX (JAX)
pip install "nmn[all]"      # Everything
```

### Basic Usage

<table>
<tr>
<td width="50%">

**PyTorch**
```python
import torch
from nmn.torch.nmn import YatNMN

# Replace nn.Linear + activation
layer = YatNMN(
    in_features=128,
    out_features=64,
    epsilon=1e-5
)

x = torch.randn(32, 128)
y = layer(x)  # (32, 64) — non-linear output!
```

</td>
<td width="50%">

**Keras**
```python
import keras
from nmn.keras.nmn import YatNMN

# Drop-in replacement for Dense
layer = YatNMN(
    features=64,
    epsilon=1e-5
)

x = keras.ops.zeros((32, 128))
y = layer(x)  # (32, 64)
```

</td>
</tr>
<tr>
<td>

**Flax NNX**
```python
from flax import nnx
from nmn.nnx.nmn import YatNMN

layer = YatNMN(
    in_features=128,
    out_features=64,
    rngs=nnx.Rngs(0)
)

x = jax.numpy.zeros((32, 128))
y = layer(x)  # (32, 64)
```

</td>
<td>

**TensorFlow**
```python
import tensorflow as tf
from nmn.tf.nmn import YatNMN

layer = YatNMN(features=64)

x = tf.zeros((32, 128))
y = layer(x)  # (32, 64)
```

</td>
</tr>
</table>

---

## 📦 Layer Support Matrix

### Core Layers

| Layer | PyTorch | TensorFlow | Keras | Flax NNX | Flax Linen |
|-------|:-------:|:----------:|:-----:|:--------:|:----------:|
| **YatNMN** (Dense) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConv1D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConv2D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConv3D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConvTranspose1D** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **YatConvTranspose2D** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **YatConvTranspose3D** | ✅ | ✅ | ❌ | ✅ | ❌ |

### Advanced Layers (Flax NNX)

| Layer | Status | Description |
|-------|:------:|-------------|
| **MultiHeadAttention** | ✅ | Yat-based attention mechanism |
| **YatSimpleCell** | ✅ | Simple RNN cell |
| **YatLSTMCell** | ✅ | LSTM with Yat operations |
| **YatGRUCell** | ✅ | GRU with Yat operations |
| **softermax** | ✅ | Generalized softmax: $\frac{x_k^n}{\epsilon + \sum_i x_i^n}$ |
| **softer_sigmoid** | ✅ | Smooth sigmoid variant |
| **soft_tanh** | ✅ | Smooth tanh variant |
| **DropConnect** | ✅ | Weight-level dropout regularization |

---

## 🔬 Cross-Framework Consistency

All implementations are verified to produce **numerically equivalent outputs** given identical inputs and weights:

```
┌─────────────────────────────────────────────────────────────┐
│              Cross-Framework Consistency Test               │
├─────────────────────────────────────────────────────────────┤
│  Framework Pair          │ Max Error    │ Status            │
├──────────────────────────┼──────────────┼───────────────────┤
│  PyTorch ↔ TensorFlow    │ < 1e-6       │ ✅ PASS           │
│  PyTorch ↔ Keras         │ < 1e-6       │ ✅ PASS           │
│  PyTorch ↔ Flax NNX      │ < 1e-6       │ ✅ PASS           │
│  PyTorch ↔ Flax Linen    │ < 1e-6       │ ✅ PASS           │
│  TensorFlow ↔ Keras      │ < 1e-7       │ ✅ PASS           │
│  Flax NNX ↔ Flax Linen   │ < 1e-7       │ ✅ PASS           │
└──────────────────────────┴──────────────┴───────────────────┘
```

This demonstrates the **robustness of the geometric YAT formulation** across different numerical backends.

---

## 📚 Examples

See **[EXAMPLES.md](EXAMPLES.md)** for comprehensive usage guides including:
- Framework-specific quick starts (PyTorch, Keras, TensorFlow, Flax)
- Architecture examples (CNN, Transformer, RNN)
- Advanced features (DropConnect, custom squashers, attention)

**Quick run:**

```bash
python examples/torch/yat_cifar10.py      # PyTorch CIFAR-10
python examples/keras/language_imdb.py    # Keras sentiment
python examples/nnx/language/mingpt.py    # JAX GPT
```

---

## 🧪 Testing

Comprehensive test suite with **cross-framework validation**:

```bash
# Install test dependencies
pip install "nmn[test]"

# Run all tests
pytest tests/ -v

# Run specific framework
pytest tests/test_torch/ -v
pytest tests/test_keras/ -v
pytest tests/test_nnx/ -v

# Run cross-framework consistency tests
pytest tests/integration/test_cross_framework_consistency.py -v

# With coverage
pytest tests/ --cov=nmn --cov-report=html
```

### Test Structure

```
tests/
├── test_torch/          # PyTorch layer tests + math validation
├── test_keras/          # Keras layer tests
├── test_tf/             # TensorFlow layer tests
├── test_nnx/            # Flax NNX tests (attention, RNN, etc.)
├── test_linen/          # Flax Linen tests
└── integration/
    ├── test_cross_framework_consistency.py  # Numerical equivalence
    └── test_compatibility.py                # API compatibility
```

---

## 📚 Theoretical Foundation

Based on the research papers:

> **Deep Learning 2.0**: *Artificial Neurons that Matter — Reject Correlation, Embrace Orthogonality*
>
> **Deep Learning 2.1**: *Mind and Cosmos — Towards Cosmos-Inspired Interpretable Neural Networks*

### Why Yat-Product?

Traditional neurons compute: $y = \sigma(\mathbf{w}^\top \mathbf{x} + b)$

This has limitations:
- **Correlation-based**: Only measures alignment, ignores proximity
- **Requires activation**: Non-linearity is external
- **Spurious activations**: Can fire strongly for distant but aligned vectors

The Yat-Product addresses these by combining:
1. **Squared dot product** (similarity) in the numerator
2. **Squared distance** (proximity) in the denominator
3. **Epsilon** for numerical stability

The result is a neuron that responds **geometrically** — activated when inputs are both similar AND close to weights.

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/mlnomadpy/nmn.git
cd nmn
pip install -e ".[dev,test]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/
```

**Areas for contribution:**
- 🐛 Bug fixes ([open issues](https://github.com/mlnomadpy/nmn/issues))
- ✨ New layer types (normalization, graph, etc.)
- 📚 Documentation and tutorials
- ⚡ Performance optimizations
- 🎨 Example applications

---

## 📖 API Reference

### Core Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_features` | int | Input dimension (Dense) or channels (Conv) |
| `out_features` | int | Output dimension or filters |
| `kernel_size` | int \| tuple | Convolution kernel size |
| `epsilon` | float | Numerical stability (default: 1e-5) |
| `use_bias` | bool | Include bias term (default: True) |
| `use_alpha` | bool | Learnable output scaling (default: True) |

### Quick Imports

```python
# PyTorch
from nmn.torch.nmn import YatNMN
from nmn.torch.layers import YatConv2d, YatConvTranspose2d

# Keras / TensorFlow
from nmn.keras.nmn import YatNMN
from nmn.keras.conv import YatConv2D

# Flax NNX (most complete)
from nmn.nnx.nmn import YatNMN
from nmn.nnx.yatconv import YatConv
from nmn.nnx.yatattention import MultiHeadAttention
from nmn.nnx.rnn import YatLSTMCell
```

📋 Full import reference → **[EXAMPLES.md](EXAMPLES.md#framework-imports-reference)**

---

## 📄 Citation

If you use NMN in your research, please cite:

```bibtex
@software{nmn2024,
  author = {Bouhsine, Taha},
  title = {NMN: Neural Matter Networks},
  year = {2024},
  url = {https://github.com/mlnomadpy/nmn}
}

@article{bouhsine2024dl2,
  author = {Taha Bouhsine},
  title = {Deep Learning 2.0: Artificial Neurons that Matter},
  year = {2024}
}

@article{bouhsine2025dl21,
  author = {Taha Bouhsine},
  title = {Deep Learning 2.1: Mind and Cosmos},
  year = {2025}
}

@article{bouhsine2025nomoredelulu,
  author = {Taha Bouhsine},
  title = {No More DeLuLu: A Kernel-Based Activation-Free Neural Networks},
  year = {2025}
}
```

---

## 📬 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/mlnomadpy/nmn/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/mlnomadpy/nmn/discussions)
- 📧 **Contact**: taha@azetta.ai

---

## 📜 License

**AGPL-3.0** — Free for personal, academic, and commercial use with attribution.

If you modify and deploy on a network, you must share the source code.

For alternative licensing, contact us.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/mlnomadpy">azetta.ai</a></sub>
</p>
