---
sidebar_position: 3
---

# Quick Start

Build your first Neural Matter Network in minutes.

## Discover the API from the shell

Before writing code, use the import-light `nmn` CLI (it imports no ML framework)
to see which backends are installed and grab a per-framework quickstart:

```bash
nmn doctor              # which of the 6 backends import + versions
nmn frameworks          # import line + YatNMN signature per framework
nmn guide nnx           # self-contained quickstart (torch/nnx/linen/keras/tf/mlx)
nmn features            # MAY/RAY performer maps + lazy YatNMN mode
```

See the [CLI reference](/docs/cli) for the full command set.

The examples below use the **Flax NNX** backend. The same YAT math is available
across all six frameworks — only the `YatNMN` size argument differs (see the
[installation table](/docs/installation#framework-support)).

## Basic Usage

### 1. Create a YatNMN Layer

```python
from flax import nnx
from nmn.nnx import YatNMN
import jax.numpy as jnp

# YatNMN replaces nn.Dense + activation
layer = YatNMN(
    in_features=784,    # Input dimension
    out_features=256,   # Output dimension
    use_bias=True,      # Include bias term
    constant_alpha=True, # Use sqrt(2) scaling
    epsilon=1e-5,       # Stability constant
    rngs=nnx.Rngs(0)
)

# Forward pass
x = jnp.ones((32, 784))  # Batch of 32 samples
y = layer(x)              # Shape: (32, 256)
```

### 2. Build a Simple Classifier

```python
class MNISTClassifier(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.layer1 = YatNMN(784, 512, constant_alpha=True, rngs=rngs)
        self.layer2 = YatNMN(512, 256, constant_alpha=True, rngs=rngs)
        self.layer3 = YatNMN(256, 10, constant_alpha=True, rngs=rngs)
    
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Create model
model = MNISTClassifier(rngs=nnx.Rngs(42))
```

### 3. Training Loop

```python
import optax

# Setup optimizer
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# Training
for batch in dataloader:
    loss = train_step(model, optimizer, batch['image'], batch['label'])
```

## Using with Convolutions

```python
from nmn.nnx import YatConv

class ConvNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = YatConv(
            in_features=1,
            out_features=32,
            kernel_size=(3, 3),
            rngs=rngs
        )
        self.conv2 = YatConv(
            in_features=32,
            out_features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            rngs=rngs
        )
        self.fc = YatNMN(64 * 7 * 7, 10, rngs=rngs)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_features` | required | Input dimension |
| `out_features` | required | Output dimension |
| `use_bias` | `True` | Add bias term |
| `use_alpha` | `True` | Enable alpha scaling |
| `constant_alpha` | `None` | Use constant (True=√2) or learnable alpha |
| `epsilon` | `1e-5` | Numerical stability constant |

## Comparison: Traditional vs NMN

```python
# Traditional (3 operations)
class TraditionalBlock(nnx.Module):
    def __init__(self, rngs):
        self.linear = nnx.Linear(256, 256, rngs=rngs)
        self.norm = nnx.LayerNorm(256)
    
    def __call__(self, x):
        x = self.linear(x)
        x = nnx.relu(x)        # Activation needed
        x = self.norm(x)       # Normalization needed
        return x

# NMN (1 operation)
class NMNBlock(nnx.Module):
    def __init__(self, rngs):
        self.layer = YatNMN(256, 256, constant_alpha=True, rngs=rngs)
    
    def __call__(self, x):
        return self.layer(x)   # That's it!
```

## Advanced Features

### Attention Mechanisms

```python
from nmn.nnx import RotaryYatAttention, MultiHeadAttention

# Multi-head attention
mha = MultiHeadAttention(
    num_heads=8,
    in_features=512,
    rngs=nnx.Rngs(0)
)

# Rotary position embeddings + YAT
rotary_attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=2048,
    rngs=nnx.Rngs(0)
)

# Performer mode for long sequences (O(n) complexity)
performer = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    use_performer=True,
    performer_kind="maclaurin",     # "slay" | "maclaurin" | "radial"
    performer_num_features=256,     # feature budget
    performer_bias=1.0,             # spherical-Yat bias b
    rngs=nnx.Rngs(0)
)
```

In torch / Keras / TensorFlow / MLX, multi-head attention is
`MultiHeadYatAttention(embed_dim, num_heads)` — there is no `key_dim` kwarg on
any backend. See [Multi-Head Attention](/docs/attention/multi-head) and
[Linear Attention](/docs/attention/linear-attention) for the SLAY / MAY / RAY
feature maps.

### Regularization

```python
from nmn.nnx import YatConv

# DropConnect: weight-level dropout
conv = YatConv(
    in_features=3,
    out_features=32,
    kernel_size=(3, 3),
    use_dropconnect=True,
    drop_rate=0.1,
    rngs=nnx.Rngs(0)
)

# Training vs inference
train_output = conv(x, deterministic=False)  # DropConnect active
eval_output = conv(x, deterministic=True)    # DropConnect disabled
```

### Custom Squashing Functions

```python
from nmn.nnx import softermax, softer_sigmoid, soft_tanh

# Smoother alternatives to standard activations
probs = softermax(logits, n=2)              # Smooth softmax
activated = softer_sigmoid(x, sharpness=1)  # Smooth sigmoid
output = soft_tanh(x)                       # Smooth tanh
```

### Lazy / Frozen-Kernel Training

Pass `lazy=True` (alias `freeze_kernel=True`) to any `YatNMN` to freeze **only
the kernel** — bias, alpha, and (if enabled) epsilon stay trainable:

```python
from nmn.nnx import YatNMN

layer = YatNMN(in_features=128, out_features=64, lazy=True, rngs=nnx.Rngs(0))
# Only bias / alpha / (learnable) epsilon are trainable; the kernel is frozen.
```

See [Lazy Training](/docs/guides/lazy-training) for the per-framework details.

## Next Steps

- [CLI Reference](/docs/cli) — `nmn doctor` / `nmn guide` / `nmn features`
- [YatNMN API Reference](/docs/layers/yat-nmn)
- [Attention Modules](/docs/attention/yat-attention)
- [Linear Attention](/docs/attention/linear-attention) — SLAY / MAY / RAY
- [Lazy Training](/docs/guides/lazy-training)
- [Example: CIFAR-10](/docs/examples/cifar10)
