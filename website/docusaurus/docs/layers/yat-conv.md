---
sidebar_position: 2
---

# YatConv

Convolutional layer using the ⵟ-product for geometrically-aware feature extraction.

## Mathematical Formulation

For each output location $(i, j)$:

$$
(\text{ⵟ-Conv}(K, I))_{i,j} = \frac{\langle K, I_{i,j} \rangle^2}{\|K - I_{i,j}\|^2 + \epsilon}
$$

Where $K$ is the kernel and $I_{i,j}$ is the input patch at position $(i, j)$.

## Import

```python
from nmn.nnx.conv import YatConv
```

## Constructor

```python
YatConv(
    in_features: int,
    out_features: int,
    kernel_size: Union[int, Tuple[int, ...]],
    *,
    strides: Union[int, Tuple[int, ...]] = 1,
    padding: Union[str, int, Tuple[int, ...]] = 'SAME',
    input_dilation: Union[int, Tuple[int, ...]] = 1,
    kernel_dilation: Union[int, Tuple[int, ...]] = 1,
    use_bias: bool = True,
    use_alpha: bool = True,
    constant_alpha: Optional[Union[bool, float]] = None,
    dtype: Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = lecun_normal(),
    bias_init: Initializer = zeros_init(),
    epsilon: float = 1e-5,
    rngs: nnx.Rngs,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Number of input channels |
| `out_features` | `int` | required | Number of output channels (filters) |
| `kernel_size` | `int \| tuple` | required | Size of the convolutional kernel |
| `strides` | `int \| tuple` | `1` | Stride of the convolution |
| `padding` | `str \| int \| tuple` | `'SAME'` | Padding mode or explicit padding |
| `input_dilation` | `int \| tuple` | `1` | Input dilation (atrous convolution) |
| `kernel_dilation` | `int \| tuple` | `1` | Kernel dilation |
| `use_bias` | `bool` | `True` | Add bias term |
| `use_alpha` | `bool` | `True` | Use alpha scaling |
| `constant_alpha` | `bool \| float \| None` | `None` | Constant alpha value |
| `epsilon` | `float` | `1e-5` | Stability constant |
| `rngs` | `nnx.Rngs` | required | RNG state |

## Usage Examples

### Basic 2D Convolution

```python
from flax import nnx
from nmn.nnx.conv import YatConv
import jax.numpy as jnp

# Create 2D conv layer
conv = YatConv(
    in_features=3,       # RGB input
    out_features=64,     # 64 filters
    kernel_size=(3, 3),
    constant_alpha=True,
    rngs=nnx.Rngs(0)
)

x = jnp.ones((16, 32, 32, 3))  # NHWC format
y = conv(x)  # Shape: (16, 32, 32, 64)
```

### Strided Convolution (Downsampling)

```python
conv = YatConv(
    in_features=64,
    out_features=128,
    kernel_size=(3, 3),
    strides=(2, 2),      # Downsample by 2
    padding='SAME',
    rngs=nnx.Rngs(0)
)

x = jnp.ones((16, 32, 32, 64))
y = conv(x)  # Shape: (16, 16, 16, 128)
```

### Dilated Convolution

```python
conv = YatConv(
    in_features=64,
    out_features=64,
    kernel_size=(3, 3),
    kernel_dilation=(2, 2),  # Dilated (atrous) convolution
    rngs=nnx.Rngs(0)
)
```

## Building a ConvNet

```python
class SimpleConvNet(nnx.Module):
    def __init__(self, num_classes: int, rngs: nnx.Rngs):
        self.conv1 = YatConv(3, 32, (3, 3), constant_alpha=True, rngs=rngs)
        self.conv2 = YatConv(32, 64, (3, 3), strides=(2, 2), constant_alpha=True, rngs=rngs)
        self.conv3 = YatConv(64, 128, (3, 3), strides=(2, 2), constant_alpha=True, rngs=rngs)
        self.fc = YatNMN(128 * 8 * 8, num_classes, constant_alpha=True, rngs=rngs)
    
    def __call__(self, x):
        x = self.conv1(x)  # (B, 32, 32, 32)
        x = self.conv2(x)  # (B, 16, 16, 64)
        x = self.conv3(x)  # (B, 8, 8, 128)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

model = SimpleConvNet(num_classes=10, rngs=nnx.Rngs(42))
```

## Available Variants

| Class | Description |
|-------|-------------|
| `YatConv` | Standard N-dimensional convolution |
| `YatConv1D` | Convenience wrapper for 1D convolution |
| `YatConv2D` | Convenience wrapper for 2D convolution |
| `YatConv3D` | Convenience wrapper for 3D convolution |

## See Also

- [YatConvTranspose](/docs/layers/yat-conv-transpose) - Transposed convolution
- [YatNMN](/docs/layers/yat-nmn) - Dense layer
