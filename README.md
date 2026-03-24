<p align="center">
  <img src="https://raw.githubusercontent.com/azettaai/nmn/master/assets/logo.png" alt="NMN Logo" width="200" height="200" onerror="this.style.display='none'">
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
  <a href="https://github.com/azettaai/nmn"><img src="https://img.shields.io/github/stars/azettaai/nmn?style=flat-square&color=yellow" alt="GitHub stars"></a>
  <a href="https://github.com/azettaai/nmn/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/azettaai/nmn/test.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://codecov.io/gh/azettaai/nmn"><img src="https://img.shields.io/codecov/c/github/azettaai/nmn?style=flat-square" alt="Coverage"></a>
  <a href="https://pypi.org/project/nmn/"><img src="https://img.shields.io/pypi/pyversions/nmn?style=flat-square" alt="Python"></a>
  <a href="https://pypi.org/project/nmn/"><img src="https://img.shields.io/pypi/l/nmn?style=flat-square&color=green" alt="License"></a>
</p>

<p align="center">
  <a href="https://azettaai.github.io/nmn/"><strong>📚 Documentation</strong></a> ·
  <a href="https://azettaai.github.io/nmn/paper/"><strong>📄 Read the Paper</strong></a> ·
  <a href="https://azettaai.github.io/nmn/blog"><strong>📝 Read the Blog</strong></a> ·
  <a href="https://github.com/azettaai/nmn/issues"><strong>🐛 Report Bug</strong></a> ·
  <a href="https://azetta.ai"><strong>🌐 Azetta.ai</strong></a>
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
| ✅ **Full Framework Parity** | Dense, Conv, ConvTranspose, Attention, Embedding, and Squashers across all 5 frameworks |
| 🧠 **Complete Layer Suite** | Dense, Conv1D/2D/3D, ConvTranspose1D/2D/3D, Multi-Head Attention, Embeddings |
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
pip install "nmn[linen]"    # Flax Linen (JAX)
pip install "nmn[all]"      # Everything
```

### Basic Usage

<table>
<tr>
<td width="50%">

**PyTorch**
```python
import torch
from nmn.torch import YatNMN

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
from nmn.keras import YatNMN

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
import jax.numpy as jnp
from flax import nnx
from nmn.nnx import YatNMN

layer = YatNMN(
    in_features=128,
    out_features=64,
    rngs=nnx.Rngs(0)
)

x = jnp.zeros((32, 128))
y = layer(x)  # (32, 64)
```

</td>
<td>

**TensorFlow**
```python
import tensorflow as tf
from nmn.tf import YatNMN

layer = YatNMN(features=64)

x = tf.zeros((32, 128))
y = layer(x)  # (32, 64)
```

</td>
</tr>
</table>

---

## 📦 Layer Support Matrix

All layers are available across **all 5 frameworks** with verified numerical equivalence.

| Layer | PyTorch | TensorFlow | Keras | Flax NNX | Flax Linen |
|-------|:-------:|:----------:|:-----:|:--------:|:----------:|
| **YatNMN** (Dense) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConv1D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConv2D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConv3D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConvTranspose1D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConvTranspose2D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatConvTranspose3D** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MultiHeadAttention** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **YatEmbed** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Squashers** | ✅ | ✅ | ✅ | ✅ | ✅ |

### Advanced Attention Variants (Flax NNX)

| Variant | Description | Complexity |
|---------|-------------|:----------:|
| **RotaryYatAttention** | YAT + Rotary Position Embeddings (RoPE) | O(n²) |
| **Spherical YAT-Performer** | YAT + FAVOR+ random features | O(n) |

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

---

## ⚙️ Advanced Features

### Attention Mechanisms

```python
# PyTorch
from nmn.torch import MultiHeadYatAttention

attn = MultiHeadYatAttention(embed_dim=512, num_heads=8)
output = attn(query, key, value)

# Flax NNX — with Rotary Position Embeddings
from nmn.nnx import RotaryYatAttention
from flax import nnx

attn = RotaryYatAttention(
    num_heads=8,
    in_features=512,
    rngs=nnx.Rngs(0)
)
output = attn(x)

# Flax NNX — Spherical YAT-Performer (O(n) linear complexity)
from nmn.nnx import MultiHeadAttention

attn = MultiHeadAttention(
    num_heads=8,
    in_features=512,
    use_performer=True,
    rngs=nnx.Rngs(0)
)
output = attn(x)
```

### Embeddings

```python
# PyTorch
from nmn.torch import YatEmbed

embed = YatEmbed(num_embeddings=10000, embedding_dim=128)
output = embed(token_ids)

# Flax NNX
from nmn.nnx import Embed
from flax import nnx

embed = Embed(
    num_embeddings=10000,
    features=128,
    constant_alpha=True,
    rngs=nnx.Rngs(0)
)
output = embed(token_ids)
# YAT attend for attention-based retrieval
scores = embed.attend(query)
```

### Squashing Functions

Alternatives to standard activation functions, available in all frameworks:

```python
from nmn.nnx import softermax, softer_sigmoid, soft_tanh

y1 = softermax(x, n=2)              # Smoother softmax with power n
y2 = softer_sigmoid(x, sharpness=1) # Smooth sigmoid variant
y3 = soft_tanh(x)                   # Smooth tanh variant
```

---

See **[EXAMPLES.md](EXAMPLES.md)** for comprehensive usage guides including:
- Framework-specific quick starts (PyTorch, Keras, TensorFlow, Flax)
- Architecture examples (CNN, Transformer)
- Advanced features (custom squashers, attention)

**Quick run:**

```bash
# PyTorch Examples
python src/nmn/torch/examples/quick_example.py         # Quick demo
python src/nmn/torch/examples/vision/resnet_training.py # ResNet training

# Flax NNX Examples
python src/nmn/nnx/examples/vision/aether_resnet50_tpu.py  # ResNet50 on TPU
python src/nmn/nnx/examples/language/m3za.py                # MiniBERT pre-training
python src/nmn/nnx/examples/language/m3za_perf.py           # Performance evaluation
```

---

## 🧪 Testing

Comprehensive test suite with **cross-framework validation**:

```bash
# Install test dependencies
pip install "nmn[test]"

# Run all tests
pytest tests/ -v

# Run specific framework tests
pytest tests/test_torch/ -v      # PyTorch
pytest tests/test_keras/ -v      # Keras
pytest tests/test_nnx/ -v        # Flax NNX

# Cross-framework consistency validation
pytest tests/integration/test_cross_framework_consistency.py -v

# With coverage report
pytest tests/ --cov=nmn --cov-report=html
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
git clone https://github.com/azettaai/nmn.git
cd nmn
pip install -e ".[dev,test]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/
```

**Areas for contribution:**
- 🐛 Bug fixes ([open issues](https://github.com/azettaai/nmn/issues))
- ✨ New layer types (normalization, graph, etc.)
- 📚 Documentation and tutorials
- ⚡ Performance optimizations
- 🎨 Example applications

---

## 📖 Quick API Reference

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_features` | int | Input dimension (Dense) or channels (Conv) |
| `out_features` | int | Output dimension or filters |
| `kernel_size` | int \| tuple | Convolution kernel size |
| `epsilon` | float | Numerical stability (default: 1e-5) |
| `use_bias` | bool | Include bias term (default: True) |
| `constant_alpha` | bool | Use fixed √2 scaling (default: varies) |
| `spherical` | bool | Enable spherical mode (default: False) |

### Framework Imports

```python
# PyTorch
from nmn.torch import YatNMN, YatConv2D, MultiHeadYatAttention, YatEmbed
from nmn.torch import softermax, softer_sigmoid, soft_tanh

# Keras
from nmn.keras import YatNMN, YatConv2D, MultiHeadYatAttention, YatEmbed
from nmn.keras import softermax, softer_sigmoid, soft_tanh

# TensorFlow
from nmn.tf import YatNMN, YatConv2D, MultiHeadYatAttention, YatEmbed
from nmn.tf import softermax, softer_sigmoid, soft_tanh

# Flax NNX (includes advanced attention variants)
from nmn.nnx import YatNMN, YatConv, MultiHeadAttention, Embed
from nmn.nnx import RotaryYatAttention, softermax

# Flax Linen
from nmn.linen import YatNMN, YatConv2D, MultiHeadAttention, YatEmbed
from nmn.linen import softermax, softer_sigmoid, soft_tanh
```

📋 Full reference → **[EXAMPLES.md](EXAMPLES.md)**

---

## 📄 Citation

If you use NMN in your research, please cite:

```bibtex
@software{nmn2024,
  author = {Bouhsine, Taha},
  title = {NMN: Neural Matter Networks},
  year = {2024},
  url = {https://github.com/azettaai/nmn}
}

@article{bouhsine2024dl2,
  author = {Bouhsine, Taha},
  title = {Deep Learning 2.0: Artificial Neurons that Matter --- Reject Correlation, Embrace Orthogonality},
  year = {2024}
}
```

---

## 📬 Support & Community

- 🐛 **Issues**: [GitHub Issues](https://github.com/azettaai/nmn/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/azettaai/nmn/discussions)
- 🌐 **Company**: [azetta.ai](https://azetta.ai)
- 📧 **Contact**: taha@azetta.ai

---

## 📜 License

**AGPL-3.0** — Free for personal, academic, and commercial use with attribution.

If you modify and deploy on a network, you must share the source code.

For alternative licensing, contact us at taha@azetta.ai.

---

## 🙏 Acknowledgments

This project was originally developed under the [mlnomadpy organization](https://github.com/mlnomadpy/nmn) and is now maintained by [Azetta.ai](https://azetta.ai).

The foundations of NMN were established through extensive research and community contributions. We're grateful to everyone who has contributed code, feedback, and ideas to make this project better.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://azetta.ai">Azetta.ai</a> ·
  Originally created by <a href="https://github.com/mlnomadpy">ML Nomad</a></sub>
</p>
