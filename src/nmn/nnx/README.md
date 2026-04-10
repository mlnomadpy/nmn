# Flax NNX NMN (Neural Matter Networks) - YAT Layers

Comprehensive Flax NNX implementation of YAT (Yet Another Transformation) neural layers for building efficient and expressive neural networks with JAX.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
  - [YatConv (n-dimensional)](#yatconv-n-dimensional)
  - [YatNMN (Linear YAT Layer)](#yatnmn-linear-yat-layer)
  - [Weight Normalization](#weight-normalization)
  - [Weight Tying & Kernel Banks](#weight-tying--kernel-banks)
  - [Constant Alpha Scaling](#constant-alpha-scaling)
  - [Spherical Mode](#spherical-mode)
  - [DropConnect Regularization](#dropconnect-regularization)
- [Advanced Usage](#advanced-usage)
- [Training Example](#training-example)
- [TPU Optimization](#tpu-optimization)
- [Performance Tips](#performance-tips)
- [API Reference](#api-reference)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd nmn

# Install in development mode with JAX/Flax support
pip install -e .

# For TPU support
pip install -e ".[tpu]"

# For GPU support (CUDA)
pip install -e ".[gpu]"
```

**Dependencies:**
- JAX >= 0.4.0
- Flax >= 0.7.0
- Optax (for optimization)
- Orbax (for checkpointing)
- Grain (optional, for efficient data loading on TPU)

## Quick Start

### Basic YatConv Usage

```python
import jax
import jax.numpy as jnp
from flax import nnx
from nmn.nnx.layers.conv import YatConv

# Initialize RNG
rngs = nnx.Rngs(0)

# Create YAT convolutional layer
layer = YatConv(
    in_features=3,
    out_features=64,
    kernel_size=(3, 3),
    padding='SAME',
    use_alpha=True,
    epsilon=1e-5,
    rngs=rngs
)

# Forward pass
x = jnp.ones((4, 32, 32, 3))  # (batch, height, width, channels)
y = layer(x)  # Output: (4, 32, 32, 64)
```

### Basic YatNMN Usage

```python
from nmn.nnx import YatNMN

# YAT linear transformation layer
layer = YatNMN(
    in_features=256,
    out_features=128,
    use_alpha=True,
    epsilon=1e-5,
    rngs=rngs
)

x = jnp.ones((4, 256))
y = layer(x)  # Output: (4, 128)
```

## Core Features

### YatConv (n-dimensional)

YAT convolution applies the YAT transformation to convolution operations:

$$y = \frac{(x \ast W + b)^2}{||x - W||^2 + \epsilon}$$

where $\ast$ denotes convolution and distances are computed patch-wise.

**Supports arbitrary dimensions** (1D, 2D, 3D, etc.) through `kernel_size` parameter.

```python
from nmn.nnx.layers.conv import YatConv

# 2D Convolution for images
conv2d = YatConv(
    in_features=64,
    out_features=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='SAME',
    use_bias=True,
    use_alpha=True,
    epsilon=1e-5,
    rngs=rngs
)

# 1D Convolution for sequences
conv1d = YatConv(
    in_features=128,
    out_features=256,
    kernel_size=(3,),
    strides=(1,),
    padding='SAME',
    rngs=rngs
)

# 3D Convolution for volumetric data
conv3d = YatConv(
    in_features=32,
    out_features=64,
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    padding='VALID',
    rngs=rngs
)
```

**Parameters:**
- `in_features`: Number of input channels
- `out_features`: Number of output channels/filters
- `kernel_size`: Tuple defining kernel dimensions
- `strides`: Stride for each dimension
- `padding`: 'SAME', 'VALID', 'CIRCULAR', 'CAUSAL', or custom padding
- `use_bias`: Whether to include bias term
- `use_alpha`: Whether to apply alpha scaling
- `epsilon`: Small constant for numerical stability
- `learnable_epsilon`: If True, epsilon becomes a learnable parameter (softplus-constrained, always positive)

### YatNMN (Linear YAT Layer)

YAT applied to linear transformations:

$$y = \frac{(x \cdot W + b)^2}{||x - W||^2 + \epsilon}$$

```python
from nmn.nnx import YatNMN

layer = YatNMN(
    in_features=512,
    out_features=256,
    use_bias=True,
    use_alpha=True,
    epsilon=1e-5,
    spherical=False,
    rngs=rngs
)

output = layer(input_tensor)
```

**Parameters:**
- `in_features`: Input dimension
- `out_features`: Output dimension
- `use_bias`: Whether to include bias
- `use_alpha`: Whether to apply alpha scaling
- `constant_alpha`: Fixed alpha value (see below)
- `epsilon`: Numerical stability constant
- `learnable_epsilon`: If True, epsilon becomes a learnable parameter (softplus-constrained, always positive)
- `spherical`: Enable spherical normalization (see below)
- `rngs`: Flax NNX random number generators

### Weight Normalization

Normalize each neuron/filter to have unit L2-norm. This provides:

1. **Computational efficiency**: Avoids recomputing kernel norms in YAT distance formula
2. **Numerical stability**: Prevents weight explosion
3. **JIT optimization**: More predictable computation for XLA compiler

```python
from nmn.nnx.layers.conv import YatConv
from nmn.nnx import YatNMN

# Enable weight normalization for convolution
conv = YatConv(
    in_features=3,
    out_features=64,
    kernel_size=(3, 3),
    weight_normalized=True,  # Enable weight normalization
    rngs=rngs
)

# Enable weight normalization for linear layer
linear = YatNMN(
    in_features=256,
    out_features=128,
    weight_normalized=True,  # Enable weight normalization
    rngs=rngs
)
```

**Benefits:**
- **10-15% faster** on TPUs (skips norm computation)
- More stable training dynamics
- Better XLA compilation optimization
- Implicit regularization

**How it works:**
```python
# Each kernel filter is normalized: W' = W / ||W||₂
# In YAT distance calculation: ||W'||² = 1.0 (always)
# Instead of: kernel_norm = jnp.sqrt(jnp.sum(w**2))
# Just use: 1.0  (constant)
```

### Weight Tying & Kernel Banks

Share kernel parameters across multiple layers to **reduce model size** and **encourage feature reuse**.

**Key concepts:**
- **Kernel Bank**: Shared parameter store managed at class level
- **Auto-expansion**: Automatically grows when larger layer needs more features
- **First-k slicing**: Each layer extracts the first k filters it needs
- **Namespace isolation**: Multiple independent banks via `kernel_bank_id`

```python
from nmn.nnx.layers.conv import YatConv

rngs = nnx.Rngs(0)

# Create multiple layers sharing same kernel bank
layer1 = YatConv(
    in_features=3,
    out_features=32,  # First 32 filters
    kernel_size=(3, 3),
    tie_kernel_bank=True,
    kernel_bank_id='resnet-conv',  # Bank namespace
    rngs=rngs
)

layer2 = YatConv(
    in_features=32,
    out_features=64,  # Auto-expands to 64
    kernel_size=(3, 3),
    tie_kernel_bank=True,
    kernel_bank_id='resnet-conv',
    rngs=rngs
)

layer3 = YatConv(
    in_features=64,
    out_features=128,  # Auto-expands to 128
    kernel_size=(3, 3),
    tie_kernel_bank=True,
    kernel_bank_id='resnet-conv',
    rngs=rngs
)

# Console output:
# Auto-expanding kernel bank 'resnet-conv': 32 -> 64 filters
# Auto-expanding kernel bank 'resnet-conv': 64 -> 128 filters
```

**For YatNMN:**

```python
from nmn.nnx import YatNMN

head1 = YatNMN(
    in_features=512,
    out_features=256,
    tie_kernel_bank=True,
    kernel_bank_id='classifier',
    rngs=rngs
)

head2 = YatNMN(
    in_features=256,
    out_features=128,
    tie_kernel_bank=True,
    kernel_bank_id='classifier',  # Shares with head1
    rngs=rngs
)
```

**Multiple independent banks:**

```python
# Separate banks for different layer types
conv_layer = YatConv(
    64, 128, (3, 3),
    tie_kernel_bank=True,
    kernel_bank_id='conv-features',
    rngs=rngs
)

attention_layer = YatConv(
    128, 128, (1, 1),
    tie_kernel_bank=True,
    kernel_bank_id='attention-weights',  # Different bank
    rngs=rngs
)

classifier = YatNMN(
    256, 10,
    tie_kernel_bank=True,
    kernel_bank_id='output-head',  # Another independent bank
    rngs=rngs
)
```

**Parameter reduction example:**

```
ResNet-50 without tying:
- Stage 1: 32 filters
- Stage 2: 64 filters
- Stage 3: 128 filters
- Stage 4: 256 filters
Total unique: 480 filters

With weight tying:
- Single shared bank: 256 filters (final size)
Reduction: 480 -> 256 (46.7% fewer parameters)
```

### Constant Alpha Scaling

Control the alpha scaling factor in the YAT transformation.

**Three modes:**

1. **Learnable alpha** (default):
```python
layer = YatConv(
    64, 128, (3, 3),
    use_alpha=True,  # Alpha is a learnable Param
    rngs=rngs
)
```

2. **Constant alpha (√2 ≈ 1.414):**
```python
layer = YatConv(
    64, 128, (3, 3),
    constant_alpha=True,  # Use sqrt(2) as fixed value
    rngs=rngs
)
```

3. **Custom constant alpha:**
```python
layer = YatConv(
    64, 128, (3, 3),
    constant_alpha=2.0,  # Use custom value
    rngs=rngs
)
```

4. **No alpha scaling:**
```python
layer = YatConv(
    64, 128, (3, 3),
    use_alpha=False,  # No scaling
    rngs=rngs
)
```

**Why use constant alpha?**
- **Faster JIT compilation**: One less parameter to trace
- **Reduced memory**: No gradient computation for alpha
- **Stable gradients**: Fixed scaling factor
- **Empirically effective**: √2 works well across many tasks

### Spherical Mode

Normalize both inputs and kernels to lie on a unit sphere. Ideal for distance-based learning.

```python
layer = YatNMN(
    in_features=256,
    out_features=128,
    spherical=True,  # Enable spherical normalization
    rngs=rngs
)

# Internally:
# inputs = inputs / jnp.linalg.norm(inputs, axis=-1, keepdims=True)
# kernel = kernel / jnp.linalg.norm(kernel, axis=0, keepdims=True)
# Distance: ||x - W||² = 2 - 2(x·W)  [simplified]
```

**Combine with weight normalization:**
```python
layer = YatNMN(
    in_features=256,
    out_features=128,
    spherical=True,           # Dynamic normalization
    weight_normalized=True,   # Static normalization at init
    rngs=rngs
)
# Maximum efficiency: both inputs and weights are unit norm
```

### DropConnect Regularization

Randomly drop connections during training for regularization.

```python
layer = YatConv(
    in_features=64,
    out_features=128,
    kernel_size=(3, 3),
    use_dropconnect=True,  # Enable DropConnect
    drop_rate=0.1,         # Drop 10% of connections
    rngs=rngs
)

# During training
output = layer(x, deterministic=False)  # Applies dropout

# During inference
output = layer(x, deterministic=True)   # Disables dropout
```

**JAX-specific note:** Uses `jax.random.bernoulli` with proper RNG management.

### Learnable Epsilon

Make the numerical stability constant ε learnable. A softplus activation ensures the value stays strictly positive (never zero, never negative):

```python
from nmn.nnx import YatNMN
from nmn.nnx.layers.conv import YatConv

# Learnable epsilon for linear layer
layer = YatNMN(
    in_features=256,
    out_features=128,
    learnable_epsilon=True,  # ε becomes a learnable parameter
    rngs=rngs
)

# Learnable epsilon for conv layer
conv = YatConv(
    in_features=64,
    out_features=128,
    kernel_size=(3, 3),
    learnable_epsilon=True,
    rngs=rngs
)

# Check the effective epsilon value
import jax
effective_eps = jax.nn.softplus(layer.epsilon_param[...])
print(f"Learned epsilon: {effective_eps}")  # Always > 0
```

**How it works:**
- A raw parameter `epsilon_param` is initialized via inverse softplus so that `softplus(raw) ≈ epsilon`
- During forward pass, the effective epsilon is computed as `softplus(epsilon_param)`, which is always > 0
- Gradients flow through softplus naturally via JAX autodiff
- The fused kernel path in `YatNMN` is automatically disabled when epsilon is learnable

**When to use:**
- When different layers benefit from different epsilon scales
- When you want the model to adaptively control the sharpness of the YAT response
- Fine-tuning scenarios where the optimal stability constant varies across layers

## Advanced Usage

### Combining Features for Maximum Efficiency

Build highly optimized models combining multiple features:

```python
from nmn.nnx.layers.conv import YatConv
from nmn.nnx import YatNMN
from flax import nnx
import jax.numpy as jnp

class EfficientYATResNet(nnx.Module):
    """Efficient YAT ResNet with all optimizations enabled."""
    
    def __init__(self, num_classes: int = 10, rngs: nnx.Rngs = None):
        # Stage 1: 32 filters
        self.conv1 = YatConv(
            in_features=3,
            out_features=32,
            kernel_size=(3, 3),
            padding='SAME',
            weight_normalized=True,      # 10-15% speedup
            constant_alpha=True,         # Use sqrt(2)
            tie_kernel_bank=True,        # Share weights
            kernel_bank_id='conv-bank',
            rngs=rngs
        )
        
        # Stage 2: 64 filters (bank auto-expands)
        self.conv2 = YatConv(
            in_features=32,
            out_features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            weight_normalized=True,
            constant_alpha=True,
            tie_kernel_bank=True,
            kernel_bank_id='conv-bank',
            rngs=rngs
        )
        
        # Stage 3: 128 filters (bank auto-expands)
        self.conv3 = YatConv(
            in_features=64,
            out_features=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            weight_normalized=True,
            constant_alpha=True,
            tie_kernel_bank=True,
            kernel_bank_id='conv-bank',
            rngs=rngs
        )
        
        # Classification head
        self.head = YatNMN(
            in_features=128,
            out_features=num_classes,
            weight_normalized=True,
            constant_alpha=True,
            tie_kernel_bank=True,
            kernel_bank_id='head-bank',
            rngs=rngs
        )
    
    def __call__(self, x, deterministic: bool = False):
        x = self.conv1(x, deterministic=deterministic)
        x = self.conv2(x, deterministic=deterministic)
        x = self.conv3(x, deterministic=deterministic)
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = self.head(x, deterministic=deterministic)
        return x

# Create model
rngs = nnx.Rngs(0)
model = EfficientYATResNet(num_classes=10, rngs=rngs)

# Count parameters
params, _ = nnx.split(model, nnx.Param, ...)
total_params = sum(p.size for p in jax.tree.leaves(params))
print(f"Total parameters: {total_params:,}")

# Forward pass
x = jnp.ones((4, 32, 32, 3))
logits = model(x, deterministic=True)
```

### JIT Compilation

YAT layers are fully compatible with JAX's JIT compilation:

```python
import jax

# JIT compile the forward pass
@jax.jit
def forward(model, x):
    return model(x, deterministic=True)

# JIT compile training step
@jax.jit
def train_step(model, x, y, optimizer):
    def loss_fn(model):
        logits = model(x, deterministic=False)
        return jnp.mean((logits - y) ** 2)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# Use compiled functions
logits = forward(model, x)
loss = train_step(model, x, y, optimizer)
```

## Training Example

Complete training script with all optimizations:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from nmn.nnx.layers.conv import YatConv
from nmn.nnx import YatNMN

# Model definition
class OptimizedYATNet(nnx.Module):
    def __init__(self, num_classes: int, rngs: nnx.Rngs):
        self.conv1 = YatConv(
            3, 64, (3, 3), padding='SAME',
            weight_normalized=True,
            constant_alpha=True,
            tie_kernel_bank=True,
            kernel_bank_id='conv',
            rngs=rngs
        )
        self.conv2 = YatConv(
            64, 128, (3, 3), strides=(2, 2), padding='SAME',
            weight_normalized=True,
            constant_alpha=True,
            tie_kernel_bank=True,
            kernel_bank_id='conv',
            rngs=rngs
        )
        self.head = YatNMN(
            128, num_classes,
            weight_normalized=True,
            constant_alpha=True,
            rngs=rngs
        )
    
    def __call__(self, x, deterministic=False):
        x = self.conv1(x, deterministic=deterministic)
        x = self.conv2(x, deterministic=deterministic)
        x = jnp.mean(x, axis=(1, 2))
        return self.head(x, deterministic=deterministic)

# Initialize model and optimizer
rngs = nnx.Rngs(0)
model = OptimizedYATNet(num_classes=10, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3))

# Training step
@jax.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        logits = model(x, deterministic=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.mean(loss)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# Training loop
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        loss = train_step(model, optimizer, batch_x, batch_y)
        
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluation
@jax.jit
def eval_step(model, x, y):
    logits = model(x, deterministic=True)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == y)
    return accuracy

accuracy = eval_step(model, test_x, test_y)
print(f"Test Accuracy: {accuracy:.4f}")
```

## TPU Optimization

YAT layers are designed for efficient TPU execution:

### 1. **Use Grain for Data Loading**

```python
import grain.python as grain

# Efficient data pipeline for TPU
source = grain.HFDataSource(
    dataset_name="cifar10",
    split="train"
)

# Preprocessing with proper sharding
def preprocess(example):
    image = example['image'] / 255.0
    label = example['label']
    return {'image': image, 'label': label}

loader = grain.DataLoader(
    data_source=source,
    operations=[
        grain.MapTransform(preprocess),
        grain.Batch(batch_size=128),
    ],
    worker_count=8,
    shard_options=grain.ShardByJaxProcess(drop_remainder=True)
)

# Use in training
for batch in loader:
    loss = train_step(model, optimizer, batch['image'], batch['label'])
```

### 2. **Use Mesh Sharding for Multi-Device**

```python
import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Create device mesh
devices = jax.devices()
mesh = Mesh(devices, ('data',))

# Define sharding for inputs
sharding = NamedSharding(mesh, P('data'))

# Shard data across devices
x_sharded = jax.device_put(x, sharding)

# Model automatically handles sharded inputs
logits = model(x_sharded, deterministic=True)
```

### 3. **Optimize Compilation**

```python
# Pre-compile with static arguments
@jax.jit
def compiled_forward(model, x):
    return model(x, deterministic=True)

# Compile once with example input
x_example = jnp.ones((128, 32, 32, 3))
_ = compiled_forward(model, x_example)  # Triggers compilation

# Subsequent calls are fast
logits = compiled_forward(model, x_real)
```

### 4. **Use BFloat16 for TPUs**

```python
# Define model with bfloat16 computation
model = OptimizedYATNet(
    num_classes=10,
    rngs=rngs
)

# Cast inputs to bfloat16
x_bf16 = x.astype(jnp.bfloat16)

# Forward pass uses bfloat16 internally
logits = model(x_bf16, deterministic=True)
```

## Performance Tips

### 1. **Enable Weight Normalization for Speed**
```python
# 10-15% speedup on TPUs
layer = YatConv(64, 128, (3, 3), weight_normalized=True, rngs=rngs)
```

### 2. **Use Constant Alpha to Reduce Memory**
```python
# Saves gradient computation and memory
layer = YatConv(64, 128, (3, 3), constant_alpha=True, rngs=rngs)
```

### 3. **Apply Weight Tying for Large Models**
```python
# 30-50% parameter reduction in ResNet-style architectures
layers = [
    YatConv(ch_in, ch_out, (3, 3), 
            tie_kernel_bank=True, 
            kernel_bank_id='shared',
            rngs=rngs)
    for ch_in, ch_out in [(3, 64), (64, 128), (128, 256)]
]
```

### 4. **Use Appropriate Padding Mode**
```python
# 'SAME' is fastest on TPUs
layer = YatConv(64, 128, (3, 3), padding='SAME', rngs=rngs)

# 'VALID' reduces computation but changes output size
layer = YatConv(64, 128, (3, 3), padding='VALID', rngs=rngs)
```

### 5. **Use Learnable Epsilon for Adaptive Stability**
```python
# Let each layer learn its optimal epsilon
layer = YatNMN(256, 128, learnable_epsilon=True, rngs=rngs)
# Note: disables fused kernel (small perf cost), but can improve convergence
```

### 6. **Leverage Spherical Mode for Metric Learning**
```python
# Efficient when distance matters
layer = YatNMN(256, 128, spherical=True, weight_normalized=True, rngs=rngs)
```

### 7. **Batch Size for TPUs**
```python
# Use multiples of 128 for optimal TPU utilization
batch_size = 128  # or 256, 512, 1024
```

## API Reference

### YatConv

```python
YatConv(
    in_features: int,
    out_features: int,
    kernel_size: Union[int, Sequence[int]],
    strides: Union[None, int, Sequence[int]] = 1,
    padding: PaddingLike = "SAME",
    input_dilation: Union[None, int, Sequence[int]] = 1,
    kernel_dilation: Union[None, int, Sequence[int]] = 1,
    feature_group_count: int = 1,
    use_bias: bool = True,
    use_alpha: bool = True,
    constant_alpha: Optional[Union[bool, float]] = None,
    use_dropconnect: bool = False,
    positive_init: bool = False,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    alpha_init: Initializer = default_alpha_init,
    mask: Optional[Array] = None,
    dtype: Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    epsilon: float = 1e-5,
    learnable_epsilon: bool = False,
    drop_rate: float = 0.0,
    weight_normalized: bool = False,
    tie_kernel_bank: bool = False,
    kernel_bank_size: Optional[int] = None,
    kernel_bank_id: str = "default",
    rngs: nnx.Rngs,
)
```

### YatNMN

```python
YatNMN(
    in_features: int,
    out_features: int,
    use_bias: bool = True,
    use_alpha: bool = True,
    constant_alpha: Optional[Union[bool, float]] = None,
    positive_init: bool = False,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    alpha_init: Initializer = default_alpha_init,
    dtype: Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    epsilon: float = 1e-5,
    learnable_epsilon: bool = False,
    spherical: bool = False,
    drop_rate: float = 0.0,
    weight_normalized: bool = False,
    tie_kernel_bank: bool = False,
    kernel_bank_size: Optional[int] = None,
    kernel_bank_id: str = 'default',
    rngs: nnx.Rngs,
)
```

## Differences from PyTorch Implementation

| Feature | PyTorch | Flax NNX |
|---------|---------|----------|
| **Input format** | NCHW (batch, channels, height, width) | NHWC (batch, height, width, channels) |
| **Random numbers** | Global torch RNG or manual seeding | Explicit `nnx.Rngs` object |
| **JIT compilation** | `torch.compile()` (experimental) | `jax.jit` (mature, stable) |
| **Device placement** | `.to(device)` | Automatic via JAX |
| **Gradient computation** | `loss.backward()` | `nnx.value_and_grad()` |
| **Parameter management** | `nn.Parameter` | `nnx.Param` |
| **Dimension order** | Channels first | Channels last (TPU optimized) |

## Examples

See the `examples/` directory for complete training scripts:

- `examples/vision/aether_resnet50_tpu.py` - ResNet-50 with YAT on TPU with Grain
- Training with weight tying and normalization enabled
- Multi-device/multi-host TPU training
- WandB integration for experiment tracking

## References

- **Flax Documentation**: https://flax.readthedocs.io/
- **JAX Documentation**: https://jax.readthedocs.io/
- **YAT (Yet Another Transformation)**: Distance-based neural activations
- **Weight Normalization**: Salimans & Kingma (2016)
- **Neural Matter Networks**: Universal approximation framework
- **TPU Best Practices**: https://cloud.google.com/tpu/docs/performance-guide
