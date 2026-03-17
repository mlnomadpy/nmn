# 📚 NMN Examples Guide

Comprehensive examples for using Neural Matter Networks across all supported frameworks.

---

## Table of Contents

- [Quick Start by Framework](#quick-start-by-framework)
  - [PyTorch](#pytorch)
  - [Keras](#keras)
  - [TensorFlow](#tensorflow)
  - [Flax NNX](#flax-nnx)
  - [Flax Linen](#flax-linen)
- [Architecture Examples](#architecture-examples)
  - [Vision: CNN with Yat Layers](#vision-cnn-with-yat-layers)
  - [NLP: Transformer Block](#nlp-transformer-block-with-yat-attention)
  - [Sequence: RNN Cells](#sequence-yat-rnn-cells)
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
    features=64,
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
    YatNMN(features=128),
    YatNMN(features=10),
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
from nmn.nnx.nmn import YatNMN
from nmn.nnx.conv import YatConv, YatConvTranspose

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
from nmn.nnx.attention import MultiHeadAttention
from nmn.nnx.nmn import YatNMN

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

### Sequence: Yat-RNN Cells

Using Yat-based RNN cells:

```python
from flax import nnx
import jax
import jax.numpy as jnp
from nmn.nnx.rnn import YatSimpleCell, YatLSTMCell, YatGRUCell

# ═══════════════════════════════════════════════════════════════
# Simple RNN Cell
# ═══════════════════════════════════════════════════════════════
simple_cell = YatSimpleCell(
    in_features=64,
    hidden_features=128,
    rngs=nnx.Rngs(0)
)

# Initialize hidden state
batch_size = 16
carry = simple_cell.initialize_carry(jax.random.key(0), (batch_size,))

# Process one timestep
x_t = jnp.zeros((batch_size, 64))
new_carry, output = simple_cell(carry, x_t)

# ═══════════════════════════════════════════════════════════════
# LSTM Cell
# ═══════════════════════════════════════════════════════════════
lstm_cell = YatLSTMCell(
    in_features=64,
    hidden_features=128,
    rngs=nnx.Rngs(1)
)

# LSTM carry is (cell_state, hidden_state)
lstm_carry = lstm_cell.initialize_carry(jax.random.key(1), (batch_size,))

# Process sequence
sequence = jnp.zeros((batch_size, 20, 64))  # (batch, time, features)
outputs = []
carry = lstm_carry

for t in range(20):
    carry, output = lstm_cell(carry, sequence[:, t, :])
    outputs.append(output)

final_output = jnp.stack(outputs, axis=1)  # (batch, time, hidden)

# ═══════════════════════════════════════════════════════════════
# GRU Cell
# ═══════════════════════════════════════════════════════════════
gru_cell = YatGRUCell(
    in_features=64,
    hidden_features=128,
    rngs=nnx.Rngs(2)
)

gru_carry = gru_cell.initialize_carry(jax.random.key(2), (batch_size,))
new_carry, output = gru_cell(gru_carry, x_t)
```

---

## Advanced Usage

### DropConnect Regularization

Weight-level dropout for regularization (Flax NNX only):

```python
from flax import nnx
from nmn.nnx.nmn import YatNMN
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
from nmn.nnx.squashers import softermax, softer_sigmoid, soft_tanh
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
from nmn.nnx.attention import MultiHeadAttention
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
from nmn.nnx.attention import RotaryYatAttention
import jax.numpy as jnp
from flax import nnx

# Create Rotary YAT Attention layer
rotary_attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=2048,      # Maximum sequence length
    use_yat=True,          # Use YAT formula (set False for standard attention)
    constant_alpha=True,   # Use constant alpha scaling
    rngs=nnx.Rngs(0)
)

# Process sequence (positions are automatically handled)
x = jnp.zeros((2, 100, 512))  # (batch, seq_len, embed_dim)
output = rotary_attn(x)  # (2, 100, 512)
```

**Performer Attention** — O(n) linear complexity using FAVOR+ approximation:

```python
from nmn.nnx.attention import RotaryYatAttention

# Enable Performer mode for long sequences
performer_attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    use_performer=True,    # Enable linear complexity
    num_features=256,      # Number of random features
    use_yat=True,          # Combine Performer with YAT
    rngs=nnx.Rngs(0)
)

# Efficient attention for long sequences
long_sequence = jnp.zeros((2, 10000, 512))  # 10K tokens
output = performer_attn(long_sequence)  # Still runs in O(n) time
```

### RNN Wrappers

**RNN Wrapper** — Process sequences with any RNN cell:

```python
from nmn.nnx.rnn import YatLSTMCell, RNN, Bidirectional
from flax import nnx
import jax.numpy as jnp

# Create cell
cell = YatLSTMCell(hidden_dim=256, rngs=nnx.Rngs(0))

# Wrap in RNN for automatic sequence processing
rnn = RNN(cell, return_sequences=True)  # Returns all timesteps

# Process batch of sequences
sequences = jnp.zeros((16, 50, 128))  # (batch, time, features)
outputs, final_state = rnn(sequences)
# outputs: (16, 50, 256) — all hidden states
# final_state: tuple of (cell_state, hidden_state)

# Return only final output
rnn_final = RNN(cell, return_sequences=False)
final_output, final_state = rnn_final(sequences)
# final_output: (16, 256) — only last hidden state
```

**Bidirectional RNN** — Process sequences in both directions:

```python
from nmn.nnx.rnn import Bidirectional, YatGRUCell

# Create bidirectional cell
gru_cell = YatGRUCell(hidden_dim=256, rngs=nnx.Rngs(0))
bi_rnn = Bidirectional(gru_cell)

# Processes sequences forward and backward, concatenates results
sequences = jnp.zeros((16, 50, 128))
bi_output = bi_rnn(sequences)
# Output shape: (16, 50, 512) — 256*2 from both directions
```

---

## Runnable Scripts

The repository contains complete, runnable training examples:

```
src/nmn/
├── torch/examples/
│   ├── yat_examples.py          # Basic PyTorch usage patterns
│   ├── yat_cifar10.py           # CIFAR-10 image classification
│   └── vision/
│       └── resnet_training.py   # ResNet architectures comparison
│
├── nnx/examples/
│   ├── vision/
│   │   └── aether_resnet50_tpu.py   # ResNet50 training on TPU/GPU
│   └── language/
│       ├── m3za.py                  # MiniBERT pre-training (MLM + SimCSE)
│       └── m3za_perf.py             # Performance benchmarking
```

### Running Examples

```bash
# PyTorch Examples
cd /workspaces/nmn
python src/nmn/torch/examples/yat_cifar10.py        # CIFAR-10 classification
python src/nmn/torch/examples/yat_examples.py       # Various architectures
python src/nmn/torch/examples/vision/resnet_training.py  # ResNet comparison

# Flax NNX Examples  
python src/nmn/nnx/examples/vision/aether_resnet50_tpu.py \
    --model resnet50 \
    --batch-size 1024 \
    --epochs 200
    
python src/nmn/nnx/examples/language/m3za.py        # MiniBERT training
python src/nmn/nnx/examples/language/m3za_perf.py   # Benchmark
```

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
from nmn.nnx.nmn import YatNMN

# Convolutions
from nmn.nnx.conv import YatConv

# Transposed Convolutions
from nmn.nnx.conv import YatConvTranspose

# Attention Mechanisms
from nmn.nnx.attention import MultiHeadAttention, RotaryYatAttention
from nmn.nnx.attention import yat_attention, rotary_yat_attention

# RNN Cells
from nmn.nnx.rnn import YatSimpleCell, YatLSTMCell, YatGRUCell
from nmn.nnx.rnn import RNN, Bidirectional

# Custom Squashing Functions
from nmn.nnx.squashers import softermax, softer_sigmoid, soft_tanh

# ═══════════════════════════════════════════════════════════════
# Flax Linen
# ═══════════════════════════════════════════════════════════════
from nmn.linen.nmn import YatNMN

# Convolutions
from nmn.linen.conv import YatConv1D, YatConv2D, YatConv3D
```

---

## Next Steps

- Check out the [README](README.md) for installation and core concepts
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Browse the [examples/](examples/) directory for complete training scripts
- Run the tests: `pytest tests/ -v`

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://azetta.ai">Azetta.ai</a></sub>
</p>




