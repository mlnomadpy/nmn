# MLX

Install on Apple Silicon with `python -m pip install "nmn[mlx]"`.

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from nmn.mlx import YatNMN

model = YatNMN(features=64, learnable_epsilon=True)
x = mx.ones((8, 128))
loss_and_grad = nn.value_and_grad(model, lambda m, z: mx.sum(m(z)))
loss, grads = loss_and_grad(model, x)
optim.Adam(1e-3).update(model, grads)
mx.eval(model.parameters(), loss)
```

MLX dense/conv modules build lazily from input width. `YatDense` aliases
`YatNMN`. Use channels-last convolution layouts.

## Fused score path

```python
from nmn.mlx import YatNMN, is_gpu_available

layer = YatNMN(features=128, fused=is_gpu_available(),
               learnable_epsilon=True)
```

The fused path dispatches a Metal kernel on a visible MLX GPU and falls back to
the eager implementation when unsupported. Array epsilon remains in the custom
VJP graph so gradients reach `epsilon_param`, including under `mx.compile` and
lazy mode. Force/evaluate GPU tests when validating the fused branch; the
repository's ordinary fixture may select CPU.

## Convolution transpose

For `padding="same"`, transpose output size is
`input * stride + output_padding` per spatial axis. The implementation handles
asymmetric high-side crop/pad for odd/even kernels, stride, and dilation. Test
2D/3D mixed-axis output and input/kernel VJPs against an independent scatter
reference on Metal.

## Attention and decode

`MultiHeadYatAttention` uses batch-major inputs. `RotaryYatAttention` adds RoPE
and incremental `decode=True` cache updates. Decode positions advance from the
cache index; invalid masks or overflow must not partially mutate cache.

MLX also exports SLAY-style performer functions, MAY, RAY, and experimental
GOAT attention. Choose the explicitly documented feature map; do not assume all
mask forms work in linear-attention paths.

## Runtime discipline

Call `mx.eval` on outputs and gradients before comparing or timing. Test both
eager and `mx.compile`. MLX can be installed on an unsupported/headless host yet
abort during Metal initialization, so probe it in a subprocess when optional
availability must not crash a parent test process.
