---
sidebar_position: 2
---

# Installation

Install Neural Matter Networks using pip. NMN ships **six independent backends**
— PyTorch, Flax NNX, Flax Linen, Keras 3, TensorFlow, and MLX — and each is a
**separate optional install**. Installing one does not install the others, and
`import nmn` by itself pulls in no ML framework.

## Quick Install

```bash
pip install nmn
```

Then add the backend(s) you want:

```bash
pip install "nmn[torch]"    # PyTorch
pip install "nmn[nnx]"      # Flax NNX  (recommended JAX entry point)
pip install "nmn[linen]"    # Flax Linen
pip install "nmn[keras]"    # Keras 3
pip install "nmn[tf]"       # TensorFlow
pip install "nmn[mlx]"      # MLX (Apple Silicon only)
```

`pip install "nmn[all]"` pulls torch + nnx + keras + tf + linen. **MLX is
Apple-Silicon-only**, so install it explicitly with `nmn[mlx]`.

## With JAX GPU Support

For GPU acceleration with the Flax backends, install JAX with CUDA support
first:

```bash
# For CUDA 12
pip install --upgrade "jax[cuda12]"

# Then install a Flax backend
pip install "nmn[nnx]"
```

## Development Installation

To install from source for development:

```bash
git clone https://github.com/azettaai/nmn.git
cd nmn
pip install -e ".[dev]"
```

## Requirements

- **Python** ≥ 3.10
- Plus the requirements of whichever backend(s) you install:
  - PyTorch backend → `torch`
  - Flax backends → `jax`, `flax`
  - Keras / TensorFlow backends → `tensorflow` / `keras`
  - MLX backend → `mlx` ≥ 0.18 (Apple Silicon, `darwin-arm64` only)

## Verify Installation

The fastest check is the bundled `nmn` CLI (it imports **no** ML framework, so
it is safe and fast):

```bash
nmn doctor
```

`nmn doctor` reports import OK/missing + version for each of the six backends
and never raises on a missing one. See the [CLI reference](/docs/cli) for the
full command set.

Programmatically:

```python
import nmn
print(nmn.__version__)
print(nmn.doctor())   # {"torch": "...", "nnx": None, ...} — version or None per backend
```

Then confirm your chosen backend imports and runs. For example, Flax NNX:

```python
from flax import nnx
from nmn.nnx import YatNMN

layer = YatNMN(in_features=64, out_features=32, rngs=nnx.Rngs(0))
print("✓ Flax NNX backend ready")
```

## Framework Support

| Framework | Import root | `pip install` | When to choose |
|-----------|-------------|---------------|----------------|
| PyTorch | `nmn.torch` | `nmn[torch]` | Most ergonomic API, broad GPU support |
| Flax NNX | `nmn.nnx` | `nmn[nnx]` | JAX speed + Pythonic API (recommended JAX entry point) |
| Flax Linen | `nmn.linen` | `nmn[linen]` | Legacy Linen codebases |
| Keras 3 | `nmn.keras` | `nmn[keras]` | Multi-backend Keras |
| TensorFlow | `nmn.tf` | `nmn[tf]` | TF / SavedModel pipelines |
| MLX | `nmn.mlx` | `nmn[mlx]` | Apple Silicon, on-device |

All six backends implement the same YAT math with numerically equivalent
outputs (< 1e-6 fp32).

## Next Steps

- [CLI Reference](/docs/cli) — discover the API from the shell
- [Quick Start Guide](/docs/quick-start)
