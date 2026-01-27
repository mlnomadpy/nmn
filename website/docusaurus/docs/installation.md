---
sidebar_position: 2
---

# Installation

Install Neural Matter Networks using pip:

## Quick Install

```bash
pip install nmn
```

## With JAX GPU Support

For GPU acceleration, first install JAX with CUDA support:

```bash
# For CUDA 12
pip install --upgrade "jax[cuda12]"

# Then install NMN
pip install nmn
```

## Development Installation

To install from source for development:

```bash
git clone https://github.com/mlnomadpy/nmn.git
cd nmn
pip install -e ".[dev]"
```

## Requirements

- **Python** ≥ 3.10
- **JAX** ≥ 0.4.20
- **Flax** ≥ 0.8.0
- **NumPy** ≥ 1.24

## Verify Installation

```python
import nmn
from nmn.nnx.nmn import YatNMN
from flax import nnx

# Create a simple layer
layer = YatNMN(in_features=64, out_features=32, rngs=nnx.Rngs(0))
print(f"NMN version: {nmn.__version__}")
print("✓ Installation successful!")
```

## Framework Support

NMN provides implementations for multiple frameworks:

| Framework | Module | Status |
|-----------|--------|--------|
| Flax NNX | `nmn.nnx` | ✅ Full support |
| Flax Linen | `nmn.linen` | ✅ Stable |
| PyTorch | `nmn.torch` | ✅ Stable |
| TensorFlow | `nmn.tf` | 🔧 Experimental |
| Keras | `nmn.keras` | 🔧 Experimental |

## Next Steps

→ [Quick Start Guide](/docs/quick-start)
