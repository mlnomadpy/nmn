---
sidebar_position: 1
---

# YatNMN

A YAT linear transformation applied over the last dimension of the input.

## Mathematical Formulation

The YatNMN layer computes:

$$
y = \frac{(\mathbf{x} \cdot \mathbf{W} + b)^2}{\|\mathbf{x} - \mathbf{W}\|^2 + \epsilon}
$$

With optional alpha scaling:

$$
y = y \cdot \left(\frac{\sqrt{\text{out\_features}}}{\log(1 + \text{out\_features})}\right)^\alpha
$$

## Import

```python
from nmn.nnx import YatNMN
```

`YatNMN` ships in all six backends with numerically equivalent outputs. The
**size argument differs per backend** — this is the #1 gotcha:

| Framework | Import | Constructor |
|-----------|--------|-------------|
| PyTorch | `from nmn.torch import YatNMN` | `YatNMN(in_features, out_features)` |
| Flax NNX | `from nmn.nnx import YatNMN` | `YatNMN(in_features, out_features, rngs=nnx.Rngs(0))` |
| Flax Linen | `from nmn.linen import YatNMN` | `YatNMN(features=N)` |
| Keras 3 | `from nmn.keras import YatNMN` | `YatNMN(units=N)` — **`units`, not `features`** |
| TensorFlow | `from nmn.tf import YatNMN` | `YatNMN(features=N)` |
| MLX | `from nmn.mlx import YatNMN` | `YatNMN(features=N)` |

Keras, TensorFlow, and MLX also export `YatDense` as an alias. The reference
below documents the Flax NNX signature.

## Constructor

```python
YatNMN(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    use_alpha: bool = True,
    constant_alpha: Optional[Union[bool, float]] = None,
    use_dropconnect: bool = False,
    lazy: bool = False,            # alias: freeze_kernel — freezes ONLY the kernel
    dtype: Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = lecun_normal(),
    bias_init: Initializer = zeros_init(),
    alpha_init: Initializer = ones_init(),
    epsilon: float = 1e-5,
    learnable_epsilon: bool = False,
    drop_rate: float = 0.0,
    rngs: nnx.Rngs,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Number of input features |
| `out_features` | `int` | required | Number of output features |
| `use_bias` | `bool` | `True` | Whether to add a bias term |
| `use_alpha` | `bool` | `True` | Whether to use alpha scaling (ignored if `constant_alpha` is set) |
| `constant_alpha` | `bool \| float \| None` | `None` | If `True`: use √2, if `float`: use that value, if `None`: learnable |
| `use_dropconnect` | `bool` | `False` | Whether to use DropConnect regularization |
| `lazy` | `bool` | `False` | Freeze **only** the kernel (alias `freeze_kernel`); bias / alpha / epsilon stay trainable. See [Lazy Training](/docs/guides/lazy-training) |
| `dtype` | `Dtype` | `None` | Computation dtype (inferred if None) |
| `param_dtype` | `Dtype` | `float32` | Parameter dtype |
| `precision` | `PrecisionLike` | `None` | JAX precision for dot product |
| `kernel_init` | `Initializer` | `lecun_normal()` | Weight initialization |
| `bias_init` | `Initializer` | `zeros_init()` | Bias initialization |
| `alpha_init` | `Initializer` | `ones_init()` | Alpha initialization (for learnable alpha) |
| `epsilon` | `float` | `1e-5` | Small constant for numerical stability |
| `learnable_epsilon` | `bool` | `False` | If `True`, epsilon becomes a learnable parameter passed through softplus to guarantee strict positivity |
| `drop_rate` | `float` | `0.0` | DropConnect dropout rate |
| `rngs` | `nnx.Rngs` | required | Random number generator state |

## Usage Examples

### Basic Usage

```python
from flax import nnx
from nmn.nnx import YatNMN
import jax.numpy as jnp

# Create layer with learnable alpha (default)
layer = YatNMN(
    in_features=784,
    out_features=256,
    rngs=nnx.Rngs(0)
)

x = jnp.ones((32, 784))
y = layer(x)  # Shape: (32, 256)
```

### Constant Alpha (Recommended)

```python
# Use sqrt(2) as constant alpha
layer = YatNMN(
    in_features=784,
    out_features=256,
    constant_alpha=True,  # Uses sqrt(2)
    rngs=nnx.Rngs(0)
)

# Or specify custom constant
layer = YatNMN(
    in_features=784,
    out_features=256,
    constant_alpha=1.5,
    rngs=nnx.Rngs(0)
)
```

### No Alpha Scaling

```python
layer = YatNMN(
    in_features=784,
    out_features=256,
    use_alpha=False,
    rngs=nnx.Rngs(0)
)
```

### With DropConnect

```python
layer = YatNMN(
    in_features=784,
    out_features=256,
    use_dropconnect=True,
    drop_rate=0.1,
    rngs=nnx.Rngs(0)
)

# During training
y_train = layer(x, deterministic=False)

# During inference
y_eval = layer(x, deterministic=True)
```

### Learnable Epsilon

```python
# Make epsilon a learnable parameter (with softplus to keep it strictly positive)
layer = YatNMN(
    in_features=784,
    out_features=256,
    learnable_epsilon=True,
    rngs=nnx.Rngs(0)
)

# epsilon_param is initialized so that softplus(param) ≈ 1e-5
# During training, the model learns the optimal epsilon value
# softplus guarantees epsilon > 0 (never zero, never negative)
```

:::tip
Learnable epsilon lets the model adapt the numerical stability constant per layer. This can be useful when different layers benefit from different epsilon scales. The softplus activation ensures the learned value stays strictly positive.
:::

### Lazy / Frozen-Kernel Mode

```python
# Freeze ONLY the kernel; bias / alpha / (learnable) epsilon stay trainable.
layer = YatNMN(
    in_features=784,
    out_features=256,
    lazy=True,          # alias: freeze_kernel=True
    rngs=nnx.Rngs(0)
)

# In NNX the frozen kernel is a FrozenParam, excluded from nnx.state(layer, nnx.Param),
# so the optimizer/gradients never touch it. Same semantics in every backend.
```

This is a lightweight regime for fine-tuning, probing, and cheap warm starts.
See [Lazy Training](/docs/guides/lazy-training) for the per-framework details.

## Comparison with nn.Linear

| Feature | `nn.Linear` | `YatNMN` |
|---------|-------------|----------|
| Activation needed | Yes (ReLU, etc.) | No |
| Normalization needed | Often | No (self-regularizing) |
| Gradient stability | Can explode/vanish | Naturally bounded |
| Non-linearity | Separate step | Built-in |

### Code Comparison

```python
# Traditional approach
class TraditionalBlock(nnx.Module):
    def __init__(self, features, rngs):
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.norm = nnx.LayerNorm(features)
    
    def __call__(self, x):
        x = self.linear(x)
        x = nnx.relu(x)
        x = self.norm(x)
        return x

# NMN approach
class NMNBlock(nnx.Module):
    def __init__(self, features, rngs):
        self.layer = YatNMN(features, features, constant_alpha=True, rngs=rngs)
    
    def __call__(self, x):
        return self.layer(x)
```

## Attributes

After initialization, the layer exposes:

- `kernel`: Weight matrix of shape `(in_features, out_features)`
- `bias`: Bias vector of shape `(out_features,)` (if `use_bias=True`)
- `alpha`: Learnable alpha parameter (if `constant_alpha=None` and `use_alpha=True`)
- `epsilon_param`: Learnable epsilon (raw) parameter of shape `(1,)` (if `learnable_epsilon=True`). Use `jax.nn.softplus(layer.epsilon_param[...])` to get the effective epsilon.

```python
layer = YatNMN(64, 32, rngs=nnx.Rngs(0))
print(layer.kernel.value.shape)  # (64, 32)
print(layer.bias.value.shape)    # (32,)
```

## See Also

- [YatConv](/docs/layers/yat-conv) - Convolutional variant
- [Lazy Training](/docs/guides/lazy-training) - Freeze-kernel mode
- [Quick Start](/docs/quick-start) - Getting started guide
- [CLI Reference](/docs/cli) - `nmn frameworks` prints the per-backend signature
