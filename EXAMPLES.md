# 📚 NMN Examples Guide

Comprehensive examples for using Neural Matter Networks across all supported frameworks.

> **New to NMN?** Run the bundled CLI to discover what is installed and grab a
> per-framework quickstart without leaving the terminal:
>
> ```bash
> nmn                     # banner: version, the six frameworks + pip extras
> nmn guide nnx           # self-contained Flax NNX quickstart
> nmn features            # MAY / RAY performer maps + lazy YatNMN mode
> nmn doctor              # which backends are importable, with versions
> nmn examples            # pointer back to this file + nnx quickstart
> ```
>
> Equivalently from Python: `import nmn; nmn.help(); nmn.doctor()`.

---

## Table of Contents

- [Quick Start by Framework](#quick-start-by-framework)
  - [PyTorch](#pytorch)
  - [Keras](#keras)
  - [TensorFlow](#tensorflow)
  - [Flax NNX](#flax-nnx)
  - [Flax Linen](#flax-linen)
  - [MLX](#mlx)
- [Architecture Examples](#architecture-examples)
  - [Vision: CNN with Yat Layers](#vision-cnn-with-yat-layers)
  - [NLP: Transformer Block](#nlp-transformer-block-with-yat-attention)
- [Advanced Usage](#advanced-usage)
  - [DropConnect Regularization](#dropconnect-regularization)
  - [Custom Squashing Functions](#custom-squashing-functions)
  - [Multi-Head Attention](#multi-head-attention)
- [Runnable Scripts](#runnable-scripts)
- [Framework Imports Reference](#framework-imports-reference)

---

## Quick Start by Framework

### PyTorch

```python
import torch
from nmn.torch.nmn import YatNMN
from nmn.torch.layers import YatConv2D, YatConvTranspose2D

# ═══════════════════════════════════════════════════════════════
# Dense Layer — Replace nn.Linear + activation
# ═══════════════════════════════════════════════════════════════
dense = YatNMN(
    in_features=128,
    out_features=64,
    bias=True,           # Include bias term
    alpha=True,          # Learnable output scaling
    epsilon=1e-5         # Numerical stability
)

x = torch.randn(32, 128)  # (batch, features)
y = dense(x)              # (32, 64) — non-linear output!

# ═══════════════════════════════════════════════════════════════
# 2D Convolution — Replace nn.Conv2d + activation
# ═══════════════════════════════════════════════════════════════
conv = YatConv2D(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1
)

images = torch.randn(8, 3, 32, 32)  # (batch, channels, H, W)
features = conv(images)              # (8, 32, 32, 32)

# ═══════════════════════════════════════════════════════════════
# Transposed Convolution — for upsampling
# ═══════════════════════════════════════════════════════════════
deconv = YatConvTranspose2D(
    in_channels=32,
    out_channels=16,
    kernel_size=4,
    stride=2,
    padding=1
)

upsampled = deconv(features)  # (8, 16, 64, 64)
```

### Keras

```python
import keras
from nmn.keras.nmn import YatNMN
from nmn.keras.conv import YatConv1D, YatConv2D, YatConvTranspose2D

# ═══════════════════════════════════════════════════════════════
# Dense Layer
# ═══════════════════════════════════════════════════════════════
dense = YatNMN(
    units=64,
    use_bias=True,
    use_alpha=True,
    epsilon=1e-5
)

x = keras.ops.zeros((32, 128))
y = dense(x)  # (32, 64)

# ═══════════════════════════════════════════════════════════════
# 2D Convolution
# ═══════════════════════════════════════════════════════════════
conv = YatConv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same'
)

images = keras.ops.zeros((8, 32, 32, 3))  # (batch, H, W, channels)
features = conv(images)                    # (8, 32, 32, 32)

# ═══════════════════════════════════════════════════════════════
# Building a Keras Model
# ═══════════════════════════════════════════════════════════════
model = keras.Sequential([
    YatConv2D(32, (3, 3), padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    YatConv2D(64, (3, 3), padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    YatNMN(units=128),
    YatNMN(units=10),
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
```

### TensorFlow

```python
import tensorflow as tf
from nmn.tf.nmn import YatNMN
from nmn.tf.conv import YatConv2D, YatConvTranspose1D

# ═══════════════════════════════════════════════════════════════
# Dense Layer
# ═══════════════════════════════════════════════════════════════
dense = YatNMN(
    features=64,
    use_bias=True,
    use_alpha=True,
    epsilon=1e-5
)

x = tf.zeros((32, 128))
y = dense(x)  # (32, 64)

# ═══════════════════════════════════════════════════════════════
# 2D Convolution
# ═══════════════════════════════════════════════════════════════
conv = YatConv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='SAME'
)

images = tf.zeros((8, 32, 32, 3))
features = conv(images)  # (8, 32, 32, 32)

# ═══════════════════════════════════════════════════════════════
# Using with tf.function for performance
# ═══════════════════════════════════════════════════════════════
@tf.function
def forward(x):
    return dense(x)

result = forward(tf.random.normal((16, 128)))
```

### Flax NNX

```python
import jax
import jax.numpy as jnp
from flax import nnx
from nmn.nnx import YatNMN, YatConv, YatConvTranspose

# ═══════════════════════════════════════════════════════════════
# Dense Layer
# ═══════════════════════════════════════════════════════════════
rngs = nnx.Rngs(0)

dense = YatNMN(
    in_features=128,
    out_features=64,
    use_bias=True,
    use_alpha=True,
    epsilon=1e-5,
    rngs=rngs
)

x = jnp.zeros((32, 128))
y = dense(x)  # (32, 64)

# ═══════════════════════════════════════════════════════════════
# 2D Convolution
# ═══════════════════════════════════════════════════════════════
conv = YatConv(
    in_features=3,
    out_features=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='SAME',
    rngs=rngs
)

images = jnp.zeros((8, 32, 32, 3))  # (batch, H, W, channels)
features = conv(images)              # (8, 32, 32, 32)

# ═══════════════════════════════════════════════════════════════
# Transposed Convolution
# ═══════════════════════════════════════════════════════════════
deconv = YatConvTranspose(
    in_features=32,
    out_features=16,
    kernel_size=(4, 4),
    strides=(2, 2),
    padding='SAME',
    rngs=rngs
)

upsampled = deconv(features)  # (8, 64, 64, 16)

# ═══════════════════════════════════════════════════════════════
# JIT Compilation for performance
# ═══════════════════════════════════════════════════════════════
@jax.jit
def forward(x):
    return dense(x)

result = forward(jax.random.normal(jax.random.key(0), (16, 128)))
```

### Flax Linen

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
from nmn.linen.nmn import YatNMN
from nmn.linen.conv import YatConv1D, YatConv2D

# ═══════════════════════════════════════════════════════════════
# Dense Layer (Functional Style)
# ═══════════════════════════════════════════════════════════════
layer = YatNMN(
    features=64,
    use_bias=True,
    use_alpha=True,
    epsilon=1e-5
)

# Initialize parameters
key = jax.random.key(0)
x = jnp.zeros((32, 128))
params = layer.init(key, x)

# Forward pass
y = layer.apply(params, x)  # (32, 64)

# ═══════════════════════════════════════════════════════════════
# 2D Convolution
# ═══════════════════════════════════════════════════════════════
conv = YatConv2D(
    features=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='SAME'
)

images = jnp.zeros((8, 32, 32, 3))
conv_params = conv.init(key, images)
features = conv.apply(conv_params, images)  # (8, 32, 32, 32)

# ═══════════════════════════════════════════════════════════════
# Building a Linen Module
# ═══════════════════════════════════════════════════════════════
class YatMLP(nn.Module):
    hidden_dim: int = 256
    output_dim: int = 10
    
    @nn.compact
    def __call__(self, x):
        x = YatNMN(features=self.hidden_dim)(x)
        x = YatNMN(features=self.output_dim)(x)
        return x

model = YatMLP()
params = model.init(key, jnp.zeros((1, 128)))
output = model.apply(params, jnp.zeros((32, 128)))  # (32, 10)
```

### MLX

```python
import mlx.core as mx
import mlx.nn as nn
from nmn.mlx import YatNMN, YatConv2D

# ═══════════════════════════════════════════════════════════════
# Dense Layer
# ═══════════════════════════════════════════════════════════════
dense = YatNMN(features=64, constant_alpha=True)
x = mx.random.normal(shape=(32, 128))
y = dense(x)  # (32, 64)

# ═══════════════════════════════════════════════════════════════
# 2D Convolution — channels-last (N, H, W, C)
# ═══════════════════════════════════════════════════════════════
conv = YatConv2D(filters=32, kernel_size=3, padding="same")
images = mx.random.normal(shape=(8, 32, 32, 3))
features = conv(images)  # (8, 32, 32, 32)

# ═══════════════════════════════════════════════════════════════
# A small Sequential-style MLP
# ═══════════════════════════════════════════════════════════════
class YatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = YatNMN(features=128)
        self.fc2 = YatNMN(features=10)
        # Build lazy params eagerly so value_and_grad sees them.
        _ = self.fc2(self.fc1(mx.zeros((1, 28 * 28))))

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.fc2(self.fc1(x))
```

> MLX wheels are Apple-Silicon-only — install via `pip install "nmn[mlx]"`
> on macOS arm64.

---

## Architecture Examples

### Vision: CNN with Yat Layers

A complete CNN for image classification using PyTorch:

```python
import torch
import torch.nn as nn
from nmn.torch.nmn import YatNMN
from nmn.torch.layers import YatConv2D

class YatCNN(nn.Module):
    """
    A CNN using Yat layers — no activation functions needed!
    The non-linearity is built into the Yat-Product operation.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional backbone
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels
            YatConv2D(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 32x32 → 16x16
            
            # Block 2: 32 → 64 channels
            YatConv2D(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 16x16 → 8x8
            
            # Block 3: 64 → 128 channels
            YatConv2D(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 8x8 → 4x4
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            YatNMN(128 * 4 * 4, 256),
            nn.Dropout(0.5),
            YatNMN(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Usage
model = YatCNN(num_classes=10)
images = torch.randn(32, 3, 32, 32)  # CIFAR-10 sized
logits = model(images)  # (32, 10)
```

### NLP: Transformer Block with Yat Attention

Using Flax NNX for a transformer architecture:

```python
from flax import nnx
import jax.numpy as jnp
from nmn.nnx import MultiHeadAttention, YatNMN

class YatTransformerBlock(nnx.Module):
    """
    A transformer block using Yat-based attention and FFN.
    """
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        rngs: nnx.Rngs = None
    ):
        self.dim = dim
        
        # Multi-head self-attention with Yat-Product
        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            qkv_features=dim,
            out_features=dim,
            rngs=rngs
        )
        
        # Feed-forward network with Yat layers
        mlp_dim = int(dim * mlp_ratio)
        self.ffn1 = YatNMN(dim, mlp_dim, rngs=rngs)
        self.ffn2 = YatNMN(mlp_dim, dim, rngs=rngs)
        
        # Layer normalization
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        
        # Dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
    
    def __call__(self, x, deterministic: bool = True):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, deterministic=deterministic)
        x = self.dropout(x, deterministic=deterministic)
        x = x + residual
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn1(x)
        x = self.ffn2(x)
        x = self.dropout(x, deterministic=deterministic)
        x = x + residual
        
        return x

# Usage
rngs = nnx.Rngs(0)
block = YatTransformerBlock(dim=512, num_heads=8, rngs=rngs)

sequence = jnp.zeros((2, 100, 512))  # (batch, seq_len, dim)
output = block(sequence)  # (2, 100, 512)
```

---

## Advanced Usage

### DropConnect Regularization

Weight-level dropout for regularization (Flax NNX only):

```python
from flax import nnx
from nmn.nnx import YatNMN
import jax.numpy as jnp

# Create layer with DropConnect
layer = YatNMN(
    in_features=128,
    out_features=64,
    use_dropconnect=True,
    drop_rate=0.2,  # 20% of weights dropped
    rngs=nnx.Rngs(params=0, dropout=1)
)

x = jnp.zeros((32, 128))

# Training mode — dropout active
y_train = layer(x, deterministic=False)

# Inference mode — no dropout
y_eval = layer(x, deterministic=True)
```

### Custom Squashing Functions

Smooth alternatives to standard activations:

```python
from nmn.nnx import softermax, softer_sigmoid, soft_tanh
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0, 4.0])

# ═══════════════════════════════════════════════════════════════
# Softermax: Generalized softmax with power parameter
# Formula: x_k^n / (ε + Σ x_i^n)
# ═══════════════════════════════════════════════════════════════
probs = softermax(x, n=2, epsilon=1e-5)
# Smoother distribution than standard softmax

# ═══════════════════════════════════════════════════════════════
# Softer Sigmoid: Smooth sigmoid variant
# Formula: x^n / (1 + x^n)
# ═══════════════════════════════════════════════════════════════
activated = softer_sigmoid(x, n=2)
# Range: [0, 1], smoother gradients

# ═══════════════════════════════════════════════════════════════
# Soft Tanh: Smooth tanh variant
# Formula: x^n / (1 + x^n) - (-x)^n / (1 + (-x)^n)
# ═══════════════════════════════════════════════════════════════
activated = soft_tanh(x, n=2)
# Range: [-1, 1], smoother gradients
```

### Multi-Head Attention

Yat-based attention mechanism:

```python
from flax import nnx
from nmn.nnx import MultiHeadAttention
import jax.numpy as jnp

# ═══════════════════════════════════════════════════════════════
# Self-Attention
# ═══════════════════════════════════════════════════════════════
self_attn = MultiHeadAttention(
    num_heads=8,
    in_features=512,
    qkv_features=512,
    out_features=512,
    use_softermax=True,  # Use custom softermax
    rngs=nnx.Rngs(0)
)

sequence = jnp.zeros((2, 100, 512))  # (batch, seq_len, dim)
output = self_attn(sequence)  # (2, 100, 512)

# ═══════════════════════════════════════════════════════════════
# Cross-Attention (queries from one sequence, keys/values from another)
# ═══════════════════════════════════════════════════════════════
cross_attn = MultiHeadAttention(
    num_heads=8,
    in_features=512,
    qkv_features=512,
    out_features=512,
    rngs=nnx.Rngs(1)
)

queries = jnp.zeros((2, 50, 512))   # Target sequence
context = jnp.zeros((2, 100, 512))  # Source sequence
output = cross_attn(queries, context)  # (2, 50, 512)
```

### Advanced Attention Mechanisms

**Rotary YAT Attention** — Combines RoPE with YAT attention for position-aware transformers:

```python
from nmn.nnx import RotaryYatAttention
import jax.numpy as jnp
from flax import nnx

# Create Rotary YAT Attention layer
rotary_attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=2048,      # Maximum sequence length
    constant_alpha=True,   # Use constant alpha scaling
    rngs=nnx.Rngs(0)
)

# Process sequence (positions are automatically handled)
x = jnp.zeros((2, 100, 512))  # (batch, seq_len, embed_dim)
output = rotary_attn(x)  # (2, 100, 512)
```

**Performer Attention** — O(n) linear complexity using FAVOR+ approximation:

```python
from nmn.nnx import RotaryYatAttention

# Enable Performer mode for long sequences
performer_attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    use_performer=True,           # Enable linear complexity
    performer_num_features=256,   # Number of random features
    rngs=nnx.Rngs(0)
)

# Efficient attention for long sequences
long_sequence = jnp.zeros((2, 10000, 512))  # 10K tokens
output = performer_attn(long_sequence)  # Still runs in O(n) time
```

### MAY / RAY Linear Attention (Flax NNX)

The 0.3.x line adds two **bias-aware** O(n) feature maps for spherical Yat
attention. Both approximate the full spherical Yat kernel

```
kappa(s) = (s + b)^2 / ((2 + eps) - 2 s)     # s = q̂·k̂ on unit vectors
```

including the bias term `b`, where the older **SLAY** anchor map only models the
`b = 0` case `s^2 / ((2 + eps) - 2 s)`. Pick the map with `performer_kind`:

- `slay` — the bias-free anchor map (default).
- `maclaurin` (**MAY**) — Random-Maclaurin features; bias-aware, best at `b ≥ 1`.
- `radial` (**RAY**) — radial / Gauss-Laguerre features; bias-aware.

```python
import jax.numpy as jnp
from flax import nnx
from nmn.nnx import RotaryYatAttention

# Selector lives on RotaryYatAttention; performer_num_features is the budget M.
attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    use_performer=True,
    performer_kind="maclaurin",     # "slay" | "maclaurin" | "radial"
    performer_num_features=256,     # feature budget M
    performer_bias=1.0,             # the spherical-Yat bias b
    epsilon=1e-5,
    rngs=nnx.Rngs(0),
)

long_sequence = jnp.zeros((2, 8000, 512))  # (batch, seq_len, embed_dim)
output = attn(long_sequence)               # (2, 8000, 512), O(n) time
```

You can also call the feature maps directly — handy for custom attention or for
checking the unbiased-estimator guarantee `E[φ(q̂)·φ(k̂)] = kappa(q̂·k̂)`:

```python
import jax
import jax.numpy as jnp
from nmn.nnx import (
    create_maclaurin_projection, maclaurin_features, maclaurin_yat_attention,
    create_radial_projection, radial_features, radial_yat_attention,
)

key = jax.random.key(0)
heads, head_dim = 8, 64
q = jnp.zeros((2, 256, heads, head_dim))  # (batch, q_len, heads, head_dim)
k = jnp.zeros((2, 256, heads, head_dim))
v = jnp.zeros((2, 256, heads, head_dim))

# MAY — Random Maclaurin
may = create_maclaurin_projection(key, head_dim=head_dim, num_features=256,
                                  bias=1.0, epsilon=1e-5)
out_may = maclaurin_yat_attention(q, k, v, may, causal=False)  # (2, 256, 8, 64)

# RAY — radial / Gauss-Laguerre
ray = create_radial_projection(key, head_dim=head_dim, sketch_m=128,
                               num_radial=8, radial_dim=64, bias=1.0, epsilon=1e-5)
out_ray = radial_yat_attention(q, k, v, ray, causal=False)     # (2, 256, 8, 64)
```

> **Bias matters.** SLAY's `b = 0` map cannot produce a constant (degree-0)
> feature, so it cannot attend diffusely. MAY/RAY carry `b`, so at `b ≥ 1` they
> faithfully track the deployed Yat kernel where SLAY's approximation breaks
> down. Set `epsilon` to the *median squared distance* between normalized q and
> k for the regime these maps target (not the `1e-5` training default).
> See [`docs/architecture.md` § Spherical Yat attention](docs/architecture.md#9-spherical-yat-attention-slay--may--ray).

### Lazy Mode — Freeze the Kernel, Train Bias / α / ε

`YatNMN(lazy=True)` (alias `freeze_kernel=True`) freezes **only the kernel**: the
bias, the learnable α, and a learnable ε stay trainable. This is a cheap
fine-tuning / probing regime — you adapt the cheap per-neuron scalars while the
expensive prototype matrix stays fixed. It is fully backward compatible
(`lazy=False` is the default).

**Flax NNX** — the frozen kernel is excluded from `nnx.state(model, nnx.Param)`,
so the optimizer never sees it:

```python
import jax, jax.numpy as jnp
from flax import nnx
from nmn.nnx import YatNMN

layer = YatNMN(in_features=128, out_features=64, lazy=True, rngs=nnx.Rngs(0))

# Only bias / alpha (and learnable epsilon, if enabled) are trainable —
# the frozen kernel is a FrozenParam, excluded from nnx.state(..., nnx.Param).
trainable = nnx.state(layer, nnx.Param)
print([p[0] for p in jax.tree_util.tree_leaves_with_path(trainable)])  # no 'kernel'

def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

x = jnp.ones((32, 128)); y = jnp.zeros((32, 64))
grads = nnx.grad(loss_fn)(layer, x, y)   # gradients only for alpha / bias
# Feed `grads` to your optimizer (e.g. nnx.Optimizer / optax) — the kernel,
# having no gradient, stays fixed across every training step.
```

**PyTorch** — the kernel's `requires_grad` is set to `False`; α and bias remain
trainable, so a standard optimizer over `model.parameters()` skips the kernel:

```python
import torch
from nmn.torch import YatNMN

layer = YatNMN(in_features=128, out_features=64, lazy=True)  # or freeze_kernel=True
for name, p in layer.named_parameters():
    print(name, p.requires_grad)   # weight False, alpha True, bias True

opt = torch.optim.Adam(p for p in layer.parameters() if p.requires_grad)
x = torch.randn(32, 128)
loss = layer(x).pow(2).mean()
loss.backward()
opt.step()                          # only alpha / bias move
```

---

## Runnable Scripts

The repository contains complete, runnable training examples:

```
src/nmn/
├── torch/examples/
│   ├── quick_example.py             # Basic PyTorch usage patterns
│   └── vision/
│       ├── mnist.py                 # MNIST MLP — 3 epochs, ~95 % test acc
│       └── resnet_training.py       # ResNet architectures comparison
│
├── nnx/examples/
│   ├── vision/
│   │   ├── mnist.py                 # MNIST MLP — 3 epochs, ~95 % test acc
│   │   └── aether_resnet50_tpu.py   # ResNet50 training on TPU/GPU
│   └── language/
│       ├── m3za.py                  # MiniBERT pre-training (MLM + SimCSE)
│       └── m3za_perf.py             # Performance benchmarking
│
├── linen/examples/
│   └── mnist.py                     # Flax Linen MNIST MLP
│
├── tf/examples/
│   └── mnist.py                     # Native TensorFlow MNIST MLP
│
├── keras/examples/
│   └── mnist.py                     # Keras 3 (backend-agnostic) MNIST MLP
│
└── mlx/examples/
    └── mnist.py                     # MLX MNIST MLP (Apple Silicon — GPU or CPU)
```

### Running Examples

```bash
# PyTorch / Flax NNX / Flax Linen (all runnable on CPU in seconds)
PYTHONPATH=src python -m nmn.torch.examples.vision.mnist --report .context/torch_mnist.json
PYTHONPATH=src python -m nmn.nnx.examples.vision.mnist   --report .context/nnx_mnist.json
PYTHONPATH=src python -m nmn.linen.examples.mnist        --report .context/linen_mnist.json

# MLX (Apple Silicon, Metal GPU by default)
PYTHONPATH=src python -m nmn.mlx.examples.mnist --device gpu --report .context/mlx_mnist.json

# TensorFlow / Keras (need separate framework installs)
PYTHONPATH=src python -m nmn.tf.examples.mnist                          --report .context/tf_mnist.json
KERAS_BACKEND=jax PYTHONPATH=src python -m nmn.keras.examples.mnist     --report .context/keras_mnist.json

# Larger workloads
python src/nmn/torch/examples/vision/resnet_training.py
python src/nmn/nnx/examples/vision/aether_resnet50_tpu.py \
    --model resnet50 --batch-size 1024 --epochs 200
python src/nmn/nnx/examples/language/m3za.py        # MiniBERT training
python src/nmn/nnx/examples/language/m3za_perf.py   # Benchmark
```

For a cross-framework comparison with real measured numbers see
[`website/blog-pages/20-cross-framework-mnist.html`](website/blog-pages/20-cross-framework-mnist.html).

---

## Framework Imports Reference

Quick reference for all available imports:

```python
# ═══════════════════════════════════════════════════════════════
# PyTorch
# ═══════════════════════════════════════════════════════════════
from nmn.torch.nmn import YatNMN

# Convolutions
from nmn.torch.layers import YatConv1D, YatConv2D, YatConv3D

# Transposed Convolutions
from nmn.torch.layers import YatConvTranspose1D, YatConvTranspose2D, YatConvTranspose3D

# ═══════════════════════════════════════════════════════════════
# Keras
# ═══════════════════════════════════════════════════════════════
from nmn.keras.nmn import YatNMN

# Convolutions
from nmn.keras.conv import YatConv1D, YatConv2D, YatConv3D

# Transposed Convolutions
from nmn.keras.conv import YatConvTranspose1D, YatConvTranspose2D

# ═══════════════════════════════════════════════════════════════
# TensorFlow
# ═══════════════════════════════════════════════════════════════
from nmn.tf.nmn import YatNMN

# Convolutions
from nmn.tf.conv import YatConv1D, YatConv2D, YatConv3D

# Transposed Convolutions
from nmn.tf.conv import YatConvTranspose1D, YatConvTranspose2D, YatConvTranspose3D

# ═══════════════════════════════════════════════════════════════
# Flax NNX (Most Feature-Complete)
# ═══════════════════════════════════════════════════════════════
from nmn.nnx import YatNMN

# Convolutions
from nmn.nnx import YatConv

# Transposed Convolutions
from nmn.nnx import YatConvTranspose

# Attention Mechanisms
from nmn.nnx import MultiHeadAttention, RotaryYatAttention
from nmn.nnx import yat_attention, rotary_yat_attention

# Custom Squashing Functions
from nmn.nnx import softermax, softer_sigmoid, soft_tanh

# ═══════════════════════════════════════════════════════════════
# Flax Linen
# ═══════════════════════════════════════════════════════════════
from nmn.linen.nmn import YatNMN

# Convolutions
from nmn.linen.conv import YatConv1D, YatConv2D, YatConv3D

# ═══════════════════════════════════════════════════════════════
# MLX (Apple Silicon)
# ═══════════════════════════════════════════════════════════════
from nmn.mlx import YatNMN

# Convolutions
from nmn.mlx import YatConv1D, YatConv2D, YatConv3D
from nmn.mlx import YatConvTranspose1D, YatConvTranspose2D, YatConvTranspose3D

# Attention + embedding
from nmn.mlx import MultiHeadYatAttention, YatEmbed
from nmn.mlx import yat_attention, yat_attention_weights, normalize_qk

# Squashers
from nmn.mlx import softermax, softer_sigmoid, soft_tanh
```

---

## Next Steps

- Run `nmn` / `nmn guide <framework>` / `nmn features` to discover the API from the terminal
- Check out the [README](README.md) for installation and core concepts
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Browse the [examples/](examples/) directory for complete training scripts
- Run the tests: `pytest tests/ -v`

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://azetta.ai">Azetta.ai</a></sub>
</p>




