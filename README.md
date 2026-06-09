<h1 align="center">NMN — Neural Matter Networks</h1>

<p align="center">
  <em>Activation-free neural layers that learn non-linearity through geometric operations.</em>
  <br>
  <strong>One library. Six frameworks. Numerically equivalent.</strong>
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
  <a href="docs/guides/mlx.md"><strong>🍎 MLX</strong></a> ·
  <a href="docs/architecture.md"><strong>🧮 Theory</strong></a> ·
  <a href="docs/migration.md"><strong>🔄 Migrate</strong></a> ·
  <a href="https://azettaai.github.io/nmn/paper/"><strong>📄 Paper</strong></a>
</p>

---

## Contents

- [What is NMN?](#what-is-nmn)
- [Install](#install)
- [CLI / discovery](#cli--discovery)
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

NMN ships across **PyTorch, Flax NNX, Flax Linen, Keras 3, TensorFlow, and MLX** (Apple Silicon) with numerically equivalent outputs (< 1e-6 max-abs error in fp32, verified by an integration parity matrix). Pick the framework you like; switch later without retraining math.

---

## Install

```bash
pip install nmn                   # the Yat layers, no framework deps
pip install "nmn[torch]"          # + PyTorch
pip install "nmn[nnx]"            # + Flax NNX (JAX)
pip install "nmn[linen]"          # + Flax Linen (JAX)
pip install "nmn[keras]"          # + Keras 3 / TensorFlow
pip install "nmn[tf]"             # + TensorFlow
pip install "nmn[mlx]"            # + MLX (Apple Silicon only)
pip install "nmn[all]"            # everything except MLX (Linux/Windows safe)
```

**Requirements:** Python ≥ 3.10 (≥ 3.11 if you want JAX/Flax).

> **GPU/TPU note:** install the GPU/TPU build of your framework *first* (see [PyTorch](https://pytorch.org/get-started/locally/) or [JAX](https://jax.readthedocs.io/en/latest/installation.html) install pages), then `pip install nmn`.

---

## CLI / discovery

`pip install nmn` ships a small `nmn` command (also `python -m nmn`) for discovery and diagnostics. It is **import-light** — it never imports a deep-learning framework just to print help, so it's instant and works even before any backend is installed.

```bash
nmn                       # banner: version, the six backends + pip extras
nmn frameworks            # import line + YatNMN signature per framework
nmn guide nnx             # self-contained quickstart for one framework
nmn guide pytorch         # aliases: torch/pytorch, tf/tensorflow, nnx/flax-nnx, linen/flax-linen, keras, mlx
nmn features              # MAY/RAY performer maps + lazy YatNMN, incl. a nnx performer_kind snippet
nmn examples             # where to find runnable examples + a nnx quickstart
nmn version              # the version string only
nmn doctor               # which of the six backends import OK (+ versions), Python and nmn version
```

`nmn doctor` reports each backend independently and never fails on a missing one, so it's the quickest way (for humans or coding agents) to see what's installed:

```bash
$ nmn doctor
torch       ok        2.x.x
nnx/linen   ok        jax 0.9.x / flax 0.12.x
keras       missing   pip install "nmn[keras]"
tf          missing   pip install "nmn[tf]"
mlx         ok        0.x.x
```

The same content is available programmatically (both stay import-light):

```python
import nmn

nmn.help()                # prints the `nmn info` banner
status = nmn.doctor()     # prints the report AND returns {framework: version_str_or_None}
```

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
    YatNMN(units=256),
    YatNMN(units=128),
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

All six backends expose the same operations with framework-idiomatic naming. They are **numerically equivalent** (verified in `tests/integration/`).

| Framework      | Pick it when…                                                                   | Guide                                          |
| -------------- | ------------------------------------------------------------------------------- | ---------------------------------------------- |
| **PyTorch**    | You want the most ergonomic Python API and broad GPU support.                  | [docs/guides/pytorch.md](docs/guides/pytorch.md) |
| **Flax NNX**   | You want JAX speed with Pythonic state. *Recommended JAX entry point.*         | [docs/guides/flax-nnx.md](docs/guides/flax-nnx.md) |
| **Flax Linen** | You're maintaining a legacy Linen codebase.                                     | [docs/guides/flax-linen.md](docs/guides/flax-linen.md) |
| **Keras 3**    | You want one API that runs on JAX, TF, *or* PyTorch backends.                  | [docs/guides/keras.md](docs/guides/keras.md)     |
| **TensorFlow** | You need TF-specific deployment (SavedModel, TFLite, Serving).                 | [docs/guides/tensorflow.md](docs/guides/tensorflow.md) |
| **MLX**        | You're on Apple Silicon and want native Metal acceleration.                    | [docs/guides/mlx.md](docs/guides/mlx.md)         |

---

## Layer reference

All layers are available across **all 6 frameworks** with verified parity.

| Operation                  | PyTorch                    | TF / Keras                 | Flax NNX                 | Flax Linen                 | MLX                        |
| -------------------------- | -------------------------- | -------------------------- | ------------------------ | -------------------------- | -------------------------- |
| Dense                      | `YatNMN`                   | `YatNMN`                   | `YatNMN`                 | `YatNMN`                   | `YatNMN`                   |
| Conv 1D / 2D / 3D          | `YatConv{1,2,3}D`          | `YatConv{1,2,3}D`          | `YatConv`                | `YatConv{1,2,3}D`          | `YatConv{1,2,3}D`          |
| ConvTranspose 1D / 2D / 3D | `YatConvTranspose{1,2,3}D` | `YatConvTranspose{1,2,3}D` | `YatConvTranspose`       | `YatConvTranspose{1,2,3}D` | `YatConvTranspose{1,2,3}D` |
| Multi-Head Attention       | `MultiHeadYatAttention`    | `MultiHeadYatAttention`    | `MultiHeadAttention`     | `MultiHeadAttention`       | `MultiHeadYatAttention`    |
| Embedding                  | `YatEmbed`                 | `YatEmbed`                 | `Embed`                  | `YatEmbed`                 | `YatEmbed`                 |
| Squashers                  | `softermax`, `softer_sigmoid`, `soft_tanh` | same                       | same                     | same                       | same                       |

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
PyTorch ↔ MLX (CPU)        │ < 1e-6       │ ✅
TensorFlow ↔ Keras         │ < 1e-7       │ ✅
Flax NNX ↔ Flax Linen      │ < 1e-7       │ ✅
Flax NNX ↔ MLX (CPU)       │ < 1e-6       │ ✅
```

Run yourself: `pytest tests/integration/test_cross_framework_consistency.py -v`.

### Bias-aware linear-attention feature maps (MAY / RAY)

Linearized spherical-Yat attention approximates the kernel
`κ(s) = (s + b)² / ((2 + ε) − 2s)` (with `s = q̂·k̂`) by a feature map `φ` so that
`φ(q)·φ(k) ≈ κ`, giving **O(n)** attention. Three feature maps ship across all
frameworks (`create_*_projection` + `*_features` + `*_yat_attention`):

| Feature map | Module (per framework) | Bias `b` | Best regime |
| ----------- | ---------------------- | :------: | ----------- |
| **SLAY** (anchor) | `spherical_yat_performer` / `performer` | `b = 0` only | bias-free kernel |
| **MAY** (Random Maclaurin) | `maclaurin_yat` / `may` / `performer_yat` | **any `b`** | `b > 0` — near-exact |
| **RAY** (radial) | `radial_yat` / `ray` / `performer_yat` | **any `b`** | sharp-`ε` route |

Trained Yat-attention learns a per-head bias `b > 0`, which SLAY's `b = 0` anchors
cannot represent. **MAY** is bias-aware and near-exact there. Benchmark (cosine
similarity of linearized vs exact attention output, `d=64`, `N=512`, matched
`F=256`, `ε = median sq-distance`, reproduced by `tests/scripts/benchmark_may_ray.py`):

```
   b  │  MAY   │  SLAY      b  │  MAY   │  SLAY
 ─────┼────────┼──────    ─────┼────────┼──────
 0.00 │  0.20  │  0.52     1.00 │  0.89  │  0.84
 0.25 │  0.52  │  0.66     2.00 │  0.97  │  0.86
 0.50 │  0.74  │  0.78     4.00 │  0.99  │  0.87   ← SLAY floors, MAY → exact
```

SLAY wins only at its `b = 0` design point; for the `b > 0` deployment regime MAY
beats it and keeps improving with the feature budget. On NNX the attention layer
takes a selector: `performer_kind="slay" | "maclaurin" | "radial"` (default `slay`).
MAY/RAY features are sign-indefinite — see the module docstrings for the caveat.

### Lazy YatNMN training (freeze kernel)

`YatNMN(..., lazy=True)` (alias `freeze_kernel=True`) freezes **only the kernel**
(feature directions) while keeping **`bias`, `alpha`, and `epsilon` trainable** — a
cheap-adaptation / α-ε-probing regime. The freeze uses each backend's idiomatic
mechanism (NNX `FrozenParam` excluded from `nnx.state(model, nnx.Param)`, torch
`requires_grad=False`, mlx `freeze`, Keras/TF `trainable=False`, linen
`stop_gradient`). Defaults are unchanged (`lazy=False`).

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
| Core layers across 6 frameworks | ✅ Production-ready, on PyPI                                  |
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
