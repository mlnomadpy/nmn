# Migration Cheat Sheet

This page shows how to convert an existing model from standard `Linear/Conv + activation` blocks to NMN layers, one framework at a time.

> The **Yat-Product** (ⵟ) replaces `Linear + ReLU` with a single layer whose non-linearity is intrinsic. You usually delete the activation entirely.

For the underlying math, see [`architecture.md`](architecture.md).

---

## Drop-in replacement table

| Standard layer                  | NMN equivalent                                  | Notes                                                                  |
| ------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------- |
| `Linear(in, out) + ReLU`        | `YatNMN(in, out)`                               | Activation absorbed into the Yat operation.                            |
| `Conv2d(c_in, c_out, k) + ReLU` | `YatConv2D(c_in, c_out, k)`                     | Same for 1D / 3D and transposes.                                       |
| `MultiHeadAttention`            | `MultiHeadYatAttention` / `MultiHeadAttention`  | Yat-scored attention; drop temperature.                                |
| `Embedding`                     | `YatEmbed`                                      | Includes `.attend()` for retrieval scoring.                            |
| `Softmax(x)`                    | `softermax(x, n=2)`                             | Smoother and parameterized by power `n`.                               |
| `Sigmoid(x)`                    | `softer_sigmoid(x)`                             | Smooth alternative.                                                    |
| `Tanh(x)`                       | `soft_tanh(x)`                                  | Smooth alternative.                                                    |

---

## PyTorch

### Before

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)
```

### After

```python
import torch.nn as nn
from nmn.torch import YatNMN

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            YatNMN(in_features=784, out_features=256),
            YatNMN(in_features=256, out_features=128),
            nn.Linear(128, 10),   # keep linear for logits
        )
    def forward(self, x):
        return self.net(x)
```

### CNN example

```python
# Before
nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),

# After
from nmn.torch import YatConv2D

YatConv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1),
YatConv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1),
```

---

## TensorFlow / Keras

### Before

```python
import keras

model = keras.Sequential([
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10),
])
```

### After

```python
import keras
from nmn.keras import YatNMN

model = keras.Sequential([
    YatNMN(features=256),
    YatNMN(features=128),
    keras.layers.Dense(10),  # keep linear for logits
])
```

### TF idiom

```python
# Before
tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")

# After
from nmn.tf import YatConv2D
YatConv2D(filters=64, kernel_size=3, padding="same")
```

---

## Flax NNX

### Before

```python
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = nnx.Linear(784, 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, 128, rngs=rngs)
        self.fc3 = nnx.Linear(128, 10, rngs=rngs)
    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        return self.fc3(x)
```

### After

```python
from flax import nnx
from nmn.nnx import YatNMN

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = YatNMN(in_features=784, out_features=256, rngs=rngs)
        self.fc2 = YatNMN(in_features=256, out_features=128, rngs=rngs)
        self.fc3 = nnx.Linear(128, 10, rngs=rngs)  # keep linear for logits
    def __call__(self, x):
        return self.fc3(self.fc2(self.fc1(x)))
```

---

## Flax Linen

### Before

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(10)(x)
```

### After

```python
import flax.linen as nn
from nmn.linen import YatNMN

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = YatNMN(features=256)(x)
        x = YatNMN(features=128)(x)
        return nn.Dense(10)(x)  # keep linear for logits
```

---

## Common patterns and gotchas

### Final classification layer

Keep a plain `Linear/Dense` at the end if you need **signed logits** for `cross_entropy`. Yat outputs are non-negative — a final Yat layer makes log-likelihood comparisons asymmetric.

### Residual connections

```python
# Yat is non-negative — direct residual addition still works but is biased:
y = x + YatNMN(...)(x)         # OK but skewed

# Preferred:
y = nn.Linear(...)(x) + YatNMN(...)(x)
```

### BatchNorm / LayerNorm

Yat layers usually do **not** need BN between them — the distance term already regularizes the response distribution. Keep LN/BN only on the input to the first Yat layer or before the final classifier.

### Learning rate

When migrating from `Linear+ReLU` to `YatNMN`, you can usually keep the same learning rate. If you see slow early training, try `use_alpha=True` (learnable α) or `constant_alpha=True` (α = √2).

### Mixed precision

If you train in fp16/bf16 and see NaNs:

```python
YatNMN(in_features=..., out_features=..., epsilon=1e-3)  # bumped from 1e-5
```

See [`architecture.md` § Numerical stability](architecture.md#3-numerical-stability) for the full guidance.

---

## Quick checklist for porting a model

1. Replace every `Linear+ReLU` pair with one `YatNMN`.
2. Replace every `Conv+ReLU` pair with one `YatConv*`.
3. Keep the **final classification layer** as `Linear/Dense`.
4. Remove redundant `BatchNorm` between Yat layers; keep at the input/output boundaries.
5. Run cross-framework consistency check (if reproducing across stacks):
   ```bash
   pytest tests/integration/test_cross_framework_consistency.py -v
   ```
6. Train. If NaN within first 100 steps → bump `epsilon` an order of magnitude.

---

## Next steps

- [PyTorch step-by-step](guides/pytorch.md)
- [Flax NNX step-by-step](guides/flax-nnx.md)
- [Keras step-by-step](guides/keras.md)
- [TensorFlow step-by-step](guides/tensorflow.md)
- [Flax Linen step-by-step](guides/flax-linen.md)
