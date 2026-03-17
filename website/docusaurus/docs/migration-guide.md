---
sidebar_position: 4
---

# Migration Guide

Replace standard framework layers with NMN equivalents. NMN layers are **drop-in replacements** — same shape in, same shape out, no activation function required.

## Dense / Linear Layers

### Flax NNX

```python
# Before
from flax import nnx
layer = nnx.Linear(in_features=256, out_features=128, rngs=nnx.Rngs(0))
x = nnx.relu(layer(x))          # needs activation

# After
from nmn.nnx.nmn import YatNMN
layer = YatNMN(in_features=256, out_features=128, constant_alpha=True, rngs=nnx.Rngs(0))
x = layer(x)                     # activation-free
```

### PyTorch

```python
# Before
import torch.nn as nn
layer = nn.Linear(in_features=256, out_features=128)
x = nn.functional.relu(layer(x))

# After
from nmn.torch.nmn import YatNMN
layer = YatNMN(in_features=256, out_features=128)
x = layer(x)
```

### Keras

```python
# Before
from tensorflow import keras
layer = keras.layers.Dense(128, activation='relu')

# After
from nmn.keras.nmn import YatNMN
layer = YatNMN(features=128)
```

### TensorFlow

```python
# Before
import tensorflow as tf
layer = tf.keras.layers.Dense(128, activation='relu')

# After
from nmn.tf.nmn import YatNMN
layer = YatNMN(features=128)
```

### Flax Linen

```python
# Before
from flax import linen as nn
layer = nn.Dense(features=128)
x = nn.relu(layer(x))

# After
from nmn.linen.nmn import YatNMN
layer = YatNMN(features=128)
x = layer(x)
```

---

## 2D Convolution

### Flax NNX

```python
# Before
conv = nnx.Conv(in_features=3, out_features=32, kernel_size=(3, 3), rngs=nnx.Rngs(0))
x = nnx.relu(conv(x))

# After
from nmn.nnx.conv import YatConv
conv = YatConv(in_features=3, out_features=32, kernel_size=(3, 3), rngs=nnx.Rngs(0))
x = conv(x)
```

### PyTorch

```python
# Before
import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
x = nn.functional.relu(conv(x))

# After
from nmn.torch.layers import YatConv2D
conv = YatConv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1)
x = conv(x)
```

### Keras

```python
# Before
from tensorflow import keras
conv = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')

# After
from nmn.keras.conv import YatConv2D
conv = YatConv2D(filters=32, kernel_size=(3, 3))
```

### TensorFlow

```python
# Before
import tensorflow as tf
conv = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')

# After
from nmn.tf.conv import YatConv2D
conv = YatConv2D(filters=32, kernel_size=(3, 3))
```

---

## Transposed Convolution (Upsampling)

### PyTorch

```python
# Before
from torch.nn import ConvTranspose2d
up = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

# After
from nmn.torch.layers import YatConvTranspose2D
up = YatConvTranspose2D(in_channels=64, out_channels=32, kernel_size=2, stride=2)
```

### Keras / TensorFlow

```python
# Before
from tensorflow import keras
up = keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2))

# After
from nmn.keras.conv import YatConvTranspose2D   # or nmn.tf.conv
up = YatConvTranspose2D(filters=32, kernel_size=(2, 2), strides=(2, 2))
```

---

## Removing Activations and Normalization

Because YAT is intrinsically non-linear and self-regularizing, the pattern `Linear → Activation → Norm` collapses to a single layer.

### Flax NNX

```python
# Before — 3 operations
class Block(nnx.Module):
    def __init__(self, dim, rngs):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
    def __call__(self, x):
        return self.norm(nnx.relu(self.linear(x)))

# After — 1 operation
from nmn.nnx.nmn import YatNMN
class Block(nnx.Module):
    def __init__(self, dim, rngs):
        self.layer = YatNMN(dim, dim, constant_alpha=True, rngs=rngs)
    def __call__(self, x):
        return self.layer(x)
```

### PyTorch

```python
# Before
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(nn.functional.relu(self.linear(x)))

# After
from nmn.torch.nmn import YatNMN
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = YatNMN(dim, dim)
    def forward(self, x):
        return self.layer(x)
```

---

## Parameter Mapping

### Flax NNX `YatNMN`

| NNX `Linear` param | NMN `YatNMN` equivalent | Notes |
|---------------------|--------------------------|-------|
| `in_features` | `in_features` | same |
| `out_features` | `out_features` | same |
| `use_bias` | `use_bias` | same |
| — | `use_alpha` | new: learnable output scaling |
| — | `constant_alpha` | `True` = √2 scale, float = fixed scale |
| — | `epsilon` | numerical stability (default `1e-5`) |

### PyTorch `YatNMN`

| `nn.Linear` param | `YatNMN` equivalent | Notes |
|---------------------|----------------------|-------|
| `in_features` | `in_features` | same |
| `out_features` | `out_features` | same |
| `bias` | `bias` | same |
| `dtype` | `dtype` | compute dtype only |
| — | `param_dtype` | weight storage dtype (default `float32`) |
| — | `alpha` | enable learnable output scaling |
| — | `epsilon` | numerical stability (default `1e-5`) |

### PyTorch `YatConv2D`

| `nn.Conv2d` param | `YatConv2D` equivalent | Notes |
|--------------------|------------------------|-------|
| `in_channels` | `in_channels` | same |
| `out_channels` | `out_channels` | same |
| `kernel_size` | `kernel_size` | same |
| `stride` | `stride` | same |
| `padding` | `padding` | same |
| `dilation` | `dilation` | same |
| `groups` | `groups` | same |
| `bias` | `bias` | same |
| — | `alpha` | enable learnable output scaling |
| — | `epsilon` | numerical stability |
| — | `use_dropconnect` | weight-level dropout |
| — | `drop_rate` | DropConnect rate |

---

## Common Patterns

### Replacing a Transformer FFN block

```python
# Before (standard FFN)
class FFN(nnx.Module):
    def __init__(self, dim, hidden_dim, rngs):
        self.fc1 = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, rngs=rngs)
    def __call__(self, x):
        return self.fc2(nnx.gelu(self.fc1(x)))

# After (NMN FFN — no activation needed)
from nmn.nnx.nmn import YatNMN
class FFN(nnx.Module):
    def __init__(self, dim, hidden_dim, rngs):
        self.fc1 = YatNMN(dim, hidden_dim, constant_alpha=True, rngs=rngs)
        self.fc2 = YatNMN(hidden_dim, dim, constant_alpha=True, rngs=rngs)
    def __call__(self, x):
        return self.fc2(self.fc1(x))
```

### Replacing a ResNet basic block (PyTorch)

```python
# Before
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return nn.functional.relu(out + x)

# After — remove BN and activations
from nmn.torch.layers import YatConv2D
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = YatConv2D(channels, channels, 3, padding=1, bias=False)
        self.conv2 = YatConv2D(channels, channels, 3, padding=1, bias=False)
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x
```

---

## Tips

- **Start with `constant_alpha=True`** (Flax) or leave default alpha (PyTorch/Keras/TF). This provides √2 scaling that often matches activation-function scaling.
- **Keep epsilon at `1e-5`** for most use cases. Lower it (`1e-6`) for tasks where inputs are small-magnitude.
- **No need for activation functions** after any NMN layer. If you have existing `relu`/`gelu`/`silu` calls between layers, remove them.
- **Batch norm is optional** — YAT responses are bounded, so training often stays stable without normalization. Experiment with removing it.
- **DropConnect** (available in NNX `YatConv`) is a stronger regularizer than Dropout for convolutional weights. Use `use_dropconnect=True, drop_rate=0.1` and pass `deterministic=False` during training.
