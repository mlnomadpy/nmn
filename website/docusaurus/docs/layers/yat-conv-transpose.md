---
sidebar_position: 3
---

# YatConvTranspose

Transposed convolution (deconvolution) using the ⵟ-product for upsampling operations.

## Import

```python
from nmn.nnx.conv import YatConvTranspose
```

## Constructor

```python
YatConvTranspose(
    in_features: int,
    out_features: int,
    kernel_size: Union[int, Tuple[int, ...]],
    *,
    strides: Union[int, Tuple[int, ...]] = 1,
    padding: Union[str, int, Tuple[int, ...]] = 'SAME',
    kernel_dilation: Union[int, Tuple[int, ...]] = 1,
    use_bias: bool = True,
    use_alpha: bool = True,
    constant_alpha: Optional[Union[bool, float]] = None,
    epsilon: float = 1e-5,
    rngs: nnx.Rngs,
)
```

## Usage Example

```python
from nmn.nnx.conv import YatConvTranspose

# Upsample by 2x
deconv = YatConvTranspose(
    in_features=128,
    out_features=64,
    kernel_size=(4, 4),
    strides=(2, 2),
    padding='SAME',
    rngs=nnx.Rngs(0)
)

x = jnp.ones((16, 8, 8, 128))
y = deconv(x)  # Shape: (16, 16, 16, 64)
```

## Use Cases

- **Decoder networks**: Upsampling in autoencoders
- **Semantic segmentation**: Feature map upsampling
- **Generative models**: Image generation

## See Also

- [YatConv](/docs/layers/yat-conv) - Standard convolution
