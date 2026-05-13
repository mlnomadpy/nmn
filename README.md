<h1 align="center">NMN — Neural Matter Networks</h1>

<p align="center">
  <em>Activation-free neural layers that learn non-linearity through geometric operations.</em>
  <br>
  <strong>One library. Five frameworks. Numerically equivalent.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/nmn/"><img src="https://img.shields.io/pypi/v/nmn.svg?style=flat-square&color=blue" alt="PyPI version"></a>
  <a href="https://pepy.tech/project/nmn"><img src="https://static.pepy.tech/badge/nmn?style=flat-square" alt="Downloads"></a>
  <a href="https://github.com/azettaai/nmn/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/azettaai/nmn/test.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://codecov.io/gh/azettaai/nmn"><img src="https://img.shields.io/codecov/c/github/azettaai/nmn?style=flat-square" alt="Coverage"></a>
  <a href="https://pypi.org/project/nmn/"><img src="https://img.shields.io/pypi/pyversions/nmn?style=flat-square" alt="Python"></a>
  <a href="https://pypi.org/project/nmn/"><img src="https://img.shields.io/pypi/l/nmn?style=flat-square&color=green" alt="License"></a>
</p>

<p align="center">
  <a href="docs/README.md"><strong>📚 Docs</strong></a> ·
  <a href="docs/guides/pytorch.md"><strong>🔥 PyTorch</strong></a> ·
  <a href="docs/guides/flax-nnx.md"><strong>⚡ JAX/Flax</strong></a> ·
  <a href="docs/guides/keras.md"><strong>🟨 Keras</strong></a> ·
  <a href="docs/guides/tensorflow.md"><strong>🟧 TF</strong></a> ·
  <a href="docs/architecture.md"><strong>🧮 Theory</strong></a> ·
  <a href="docs/migration.md"><strong>🔄 Migrate</strong></a> ·
  <a href="https://azettaai.github.io/nmn/paper/"><strong>📄 Paper</strong></a>
</p>

---

## Contents

- [What is NMN?](#what-is-nmn)
- [Install](#install)
- [60-second tour](#60-second-tour)
- [Choose your framework](#choose-your-framework)
- [Layer reference](#layer-reference)
- [The math, in one minute](#the-math-in-one-minute)
- [Examples](#examples)
- [Testing](#testing)
- [Project status](#project-status)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## What is NMN?

NMN is a drop-in replacement for `Linear + activation` and `Conv + activation` blocks. The non-linearity is **built into the layer** via a geometric ratio — no ReLU, no Sigmoid, no GELU.

```python
# Before
y = relu(linear(x))            # dot product → activation

# After
y = YatNMN(in_features=128, out_features=64)(x)   # geometric, intrinsically non-linear
```

**Why care?**

| Standard neuron                                        | Yat neuron                                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------------- |
| Measures *correlation* between **w** and **x**         | Balances *correlation* AND *proximity*                                  |
| Requires an external activation for non-linearity      | Non-linearity is intrinsic                                              |
| Fires for distant-but-aligned vectors (spurious)       | Penalizes distance → cleaner, prototype-like features                   |

NMN ships across **PyTorch, Flax NNX, Flax Linen, Keras 3, and TensorFlow** with numerically equivalent outputs (< 1e-6 max-abs error in fp32). Pick the framework you like; switch later without retraining math.

---

## Install

```bash
pip install nmn                   # the Yat layers, no framework deps
pip install "nmn[torch]"          # + PyTorch
pip install "nmn[nnx]"            # + Flax NNX (JAX)
pip install "nmn[linen]"          # + Flax Linen (JAX)
pip install "nmn[keras]"          # + Keras 3 / TensorFlow
pip install "nmn[tf]"             # + TensorFlow
pip install "nmn[all]"            # everything
```

**Requirements:** Python ≥ 3.10 (≥ 3.11 if you want JAX/Flax).

> **GPU/TPU note:** install the GPU/TPU build of your framework *first* (see [PyTorch](https://pytorch.org/get-started/locally/) or [JAX](https://jax.readthedocs.io/en/latest/installation.html) install pages), then `pip install nmn`.

---

## 60-second tour

The same MLP in every framework. Pick one, copy, run.

<details open>
<summary><strong>🔥 PyTorch</strong></summary>

```python
import torch, torch.nn as nn
from nmn.torch import YatNMN

model = nn.Sequential(
    nn.Flatten(),
    YatNMN(in_features=784, out_features=256),
    YatNMN(in_features=256, out_features=128),
    nn.Linear(128, 10),          # keep logits linear
)

x = torch.randn(32, 1, 28, 28)
print(model(x).shape)            # torch.Size([32, 10])
```

→ [Full PyTorch guide](docs/guides/pytorch.md)
</details>

<details>
<summary><strong>⚡ Flax NNX (JAX)</strong></summary>

```python
import jax.numpy as jnp
from flax import nnx
from nmn.nnx import YatNMN

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = YatNMN(in_features=784, out_features=256, rngs=rngs)
        self.fc2 = YatNMN(in_features=256, out_features=128, rngs=rngs)
        self.out = nnx.Linear(128, 10, rngs=rngs)
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.out(self.fc2(self.fc1(x)))

model = MLP(rngs=nnx.Rngs(0))
print(model(jnp.ones((32, 28, 28, 1))).shape)   # (32, 10)
```

→ [Full Flax NNX guide](docs/guides/flax-nnx.md)
</details>

<details>
<summary><strong>🟨 Keras 3</strong></summary>

```python
import keras
from nmn.keras import YatNMN

model = keras.Sequential([
    keras.layers.Input((28, 28)),
    keras.layers.Flatten(),
    YatNMN(features=256),
    YatNMN(features=128),
    keras.layers.Dense(10),
])
print(model(keras.ops.ones((32, 28, 28))).shape)  # (32, 10)
```

→ [Full Keras guide](docs/guides/keras.md)
</details>

<details>
<summary><strong>🟧 TensorFlow</strong></summary>

```python
import tensorflow as tf
from nmn.tf import YatNMN

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    YatNMN(features=256),
    YatNMN(features=128),
    tf.keras.layers.Dense(10),
])
print(model(tf.zeros((32, 28, 28))).shape)        # (32, 10)
```

→ [Full TensorFlow guide](docs/guides/tensorflow.md)
</details>

<details>
<summary><strong>⚡ Flax Linen (JAX, legacy API)</strong></summary>

```python
import jax, jax.numpy as jnp
import flax.linen as nn
from nmn.linen import YatNMN

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = YatNMN(features=256)(x)
        x = YatNMN(features=128)(x)
        return nn.Dense(10)(x)

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))
print(model.apply(params, jnp.ones((32, 28, 28, 1))).shape)  # (32, 10)
```

→ [Full Flax Linen guide](docs/guides/flax-linen.md)
</details>

---

## Choose your framework

All five backends expose the same operations with framework-idiomatic naming. They are **numerically equivalent** (verified in `tests/integration/`).

| Framework      | Pick it when…                                                                   | Guide                                          |
| -------------- | ------------------------------------------------------------------------------- | ---------------------------------------------- |
| **PyTorch**    | You want the most ergonomic Python API and broad GPU support.                  | [docs/guides/pytorch.md](docs/guides/pytorch.md) |
| **Flax NNX**   | You want JAX speed with Pythonic state. *Recommended JAX entry point.*         | [docs/guides/flax-nnx.md](docs/guides/flax-nnx.md) |
| **Flax Linen** | You're maintaining a legacy Linen codebase.                                     | [docs/guides/flax-linen.md](docs/guides/flax-linen.md) |
| **Keras 3**    | You want one API that runs on JAX, TF, *or* PyTorch backends.                  | [docs/guides/keras.md](docs/guides/keras.md)     |
| **TensorFlow** | You need TF-specific deployment (SavedModel, TFLite, Serving).                 | [docs/guides/tensorflow.md](docs/guides/tensorflow.md) |

---

## Layer reference

All layers are available across **all 5 frameworks** with verified parity.

| Operation                  | PyTorch                    | TF / Keras                 | Flax NNX                 | Flax Linen                 |
| -------------------------- | -------------------------- | -------------------------- | ------------------------ | -------------------------- |
| Dense                      | `YatNMN`                   | `YatNMN`                   | `YatNMN`                 | `YatNMN`                   |
| Conv 1D / 2D / 3D          | `YatConv{1,2,3}D`          | `YatConv{1,2,3}D`          | `YatConv`                | `YatConv{1,2,3}D`          |
| ConvTranspose 1D / 2D / 3D | `YatConvTranspose{1,2,3}D` | `YatConvTranspose{1,2,3}D` | `YatConvTranspose`       | `YatConvTranspose{1,2,3}D` |
| Multi-Head Attention       | `MultiHeadYatAttention`    | `MultiHeadYatAttention`    | `MultiHeadAttention`     | `MultiHeadAttention`       |
| Embedding                  | `YatEmbed`                 | `YatEmbed`                 | `Embed`                  | `YatEmbed`                 |
| Squashers                  | `softermax`, `softer_sigmoid`, `soft_tanh` | same                       | same                     | same                       |

**Flax NNX exclusives:**

| Variant                                  | What it does                                | Complexity |
| ---------------------------------------- | ------------------------------------------- | :--------: |
| `RotaryYatAttention`                     | Yat attention + RoPE                        |   O(n²)    |
| `MultiHeadAttention(use_performer=True)` | Spherical YAT-Performer (FAVOR+ features)   |   O(n)     |
| Pallas fused yat-attention kernel        | Flash-attention-style fused TPU/GPU kernel  | O(n²) mem-efficient |

### Cross-framework consistency

```
Framework Pair             │ Max Error    │ Status
───────────────────────────┼──────────────┼────────
PyTorch ↔ TensorFlow       │ < 1e-6       │ ✅
PyTorch ↔ Keras            │ < 1e-6       │ ✅
PyTorch ↔ Flax NNX         │ < 1e-6       │ ✅
PyTorch ↔ Flax Linen       │ < 1e-6       │ ✅
TensorFlow ↔ Keras         │ < 1e-7       │ ✅
Flax NNX ↔ Flax Linen      │ < 1e-7       │ ✅
```

Run yourself: `pytest tests/integration/test_cross_framework_consistency.py -v`.

---

## The math, in one minute

A Yat neuron is a ratio of *similarity* to *distance*, with the bias absorbed into the (squared) inner product:

$$
\mathrm{ⵟ}(\mathbf{w}, \mathbf{x}, b) = \frac{\bigl(\langle \mathbf{w}, \mathbf{x} \rangle + b\bigr)^2}{\lVert \mathbf{w} - \mathbf{x} \rVert^2 + \varepsilon}
$$

Maximum response requires **w** and **x** to be both *aligned* AND *close*. That's the geometric prior that lets you drop the activation function. The bias `b` shifts the affine score *inside* the polynomial (biased polynomial kernel) — not added to the output after the ratio.

For convolutions, the same identity applies per patch:

$$
\mathrm{ⵟ}^*(\mathbf{W}, \mathbf{X}, b) = \frac{\bigl(\sum_{i,j} w_{ij} x_{ij} + b\bigr)^2}{\sum_{i,j} (w_{ij} - x_{ij})^2 + \varepsilon}
$$

`ε` (`epsilon`, default `1e-5`) prevents division by zero; bump it to `1e-3` for fp16/bf16. Some layers also expose a learnable `alpha` scalar (set `use_alpha=True`, or `constant_alpha=True` to fix α = √2).

📖 **Deeper dive:** [docs/architecture.md](docs/architecture.md) — geometric reading, ε tuning, where (not) to use NMN, mental model.

---

## Examples

Runnable scripts live in-tree, organized per framework:

| Script                                                                              | What it does                                  |
| ----------------------------------------------------------------------------------- | --------------------------------------------- |
| [`src/nmn/torch/examples/quick_example.py`](src/nmn/torch/examples/quick_example.py) | Yat layers in PyTorch (weight norm, α, …)     |
| [`src/nmn/torch/examples/vision/resnet_training.py`](src/nmn/torch/examples/vision/resnet_training.py) | ResNet training on PyTorch                    |
| [`src/nmn/nnx/examples/vision/aether_resnet50_tpu.py`](src/nmn/nnx/examples/vision/aether_resnet50_tpu.py) | ResNet-50 on TPU with Flax NNX                |
| [`src/nmn/nnx/examples/language/m3za.py`](src/nmn/nnx/examples/language/m3za.py)     | MiniBERT pre-training (uses fused attention)  |
| [`src/nmn/nnx/examples/language/m3za_perf.py`](src/nmn/nnx/examples/language/m3za_perf.py) | Performance evaluation                        |

For copy-pasteable snippets across all frameworks (CNN, transformer, attention, embeddings, custom squashers), see **[EXAMPLES.md](EXAMPLES.md)**.

---

## Testing

```bash
pip install "nmn[test]"

pytest tests/                                      # everything
pytest tests/test_torch/                           # one framework
pytest tests/integration/                          # cross-framework parity
pytest tests/ -m "not slow"                        # skip slow tests
pytest tests/ --cov=nmn --cov-report=html          # coverage report
```

CI matrix: Linux × Python {3.10, 3.11, 3.12} for all frameworks, plus macOS-3.11 (PyTorch + Keras/TF) and Windows-3.11 (PyTorch). See [`.github/workflows/test.yml`](.github/workflows/test.yml).

---

## Project status

| Area                       | Status                                                              |
| -------------------------- | ------------------------------------------------------------------- |
| Core layers across 5 frameworks | ✅ Production-ready, on PyPI                                  |
| Cross-framework consistency tests | ✅ Verified < 1e-6 in fp32                                  |
| Documentation               | ✅ Per-platform guides, architecture, migration                     |
| ONNX export                | 🚧 Should work (standard ops) — not yet covered in CI ([TODO.md](TODO.md)) |
| INT8 quantization          | 🚧 Not yet implemented ([TODO.md](TODO.md))                         |
| Auto-generated API reference | 🚧 Planned (Sphinx / mkdocstrings) — see [TODO.md](TODO.md)      |

Latest changes: [CHANGELOG.md](CHANGELOG.md).

---

## Contributing

We welcome contributions of all sizes — from typo fixes to new framework backends. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for development setup, test commands, and the "add a new layer" workflow.

Quick start:

```bash
git clone https://github.com/azettaai/nmn.git
cd nmn
pip install -e ".[dev,torch]"      # or ".[dev,nnx]", etc.
pytest tests/test_torch/ -v
```

Found a bug? → [open an issue](https://github.com/azettaai/nmn/issues).
Security issue? → see [SECURITY.md](SECURITY.md) for private disclosure.

---

## Citation

```bibtex
@software{nmn2024,
  author = {Bouhsine, Taha},
  title  = {NMN: Neural Matter Networks},
  year   = {2024},
  url    = {https://github.com/azettaai/nmn}
}

@article{bouhsine2024dl2,
  author = {Bouhsine, Taha},
  title  = {Deep Learning 2.0: Artificial Neurons that Matter --- Reject Correlation, Embrace Orthogonality},
  year   = {2024}
}
```

---

## Community

- 💬 **Discussions** — [GitHub Discussions](https://github.com/azettaai/nmn/discussions)
- 🐛 **Issues** — [GitHub Issues](https://github.com/azettaai/nmn/issues)
- 🌐 **Company** — [azetta.ai](https://azetta.ai)
- 📧 **Contact** — taha@azetta.ai

---

## License

**[AGPL-3.0](LICENSE)** — free for personal, academic, and commercial use with attribution. If you modify and deploy on a network, you must share the source.

For alternative licensing, contact **taha@azetta.ai**.

---

## Acknowledgments

This project was originally developed under the [mlnomadpy organization](https://github.com/mlnomadpy/nmn) and is now maintained by [Azetta.ai](https://azetta.ai). Thanks to everyone who has contributed code, feedback, and ideas.

<p align="center">
  <sub>Built with ❤️ by <a href="https://azetta.ai">Azetta.ai</a> · Originally created by <a href="https://github.com/mlnomadpy">ML Nomad</a></sub>
</p>
