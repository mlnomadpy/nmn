# PyTorch

Install with `python -m pip install "nmn[torch]"`.

```python
import torch
from nmn.torch import YatNMN

layer = YatNMN(
    in_features=128,
    out_features=64,
    bias=True,
    alpha=True,
    epsilon=1e-5,
    learnable_epsilon=True,
    dtype=torch.float32,
    param_dtype=torch.float32,
)
y = layer(torch.randn(8, 128))
y.sum().backward()
```

PyTorch dense uses `bias=` and `alpha=`; convolution and attention use their
own documented `use_alpha`-style options. Inspect signatures before sharing
configuration dictionaries.

## Convolution

Use `YatConv1D/2D/3D` and transpose variants with native PyTorch NCL/NCHW/NCDHW
layouts. Group counts must divide both input and output channels.

```python
from nmn.torch import YatConv2D

conv = YatConv2D(3, 32, 3, padding="same", learnable_epsilon=True)
y = conv(torch.randn(4, 3, 64, 64))
```

Tied dense/forward-convolution banks use `tie_kernel_bank=True`, a stable
`kernel_bank_id`, and optional `kernel_bank_size`. Construct compatible
consumers on the final target device/dtype. Capacity may expand only before the
bank's first forward; tied modules deliberately reject post-construction
`.to()`/`.half()` migration that would split shared identity. Untied modules
migrate normally. Learned epsilon retains safe storage through dtype changes.

## Attention

```python
from nmn.torch import MultiHeadYatAttention

attn = MultiHeadYatAttention(embed_dim=128, num_heads=8)
x = torch.randn(2, 32, 128)
y = attn(x, mask=torch.ones(32, 32, dtype=torch.bool))
```

Pass `key=` and `value=` for cross attention. With the module in training mode,
`deterministic=False` permits dropout; `eval()` disables it. Fully masked query
rows return exact-zero weights/output.

## Precision and transforms

`dtype` is compute/output policy; `param_dtype` is parameter storage. Prefer
FP32 parameter storage with autocast for FP16 training. Safe casts support
autograd and modern `torch.func` transforms while retaining compatibility with
the declared minimum Torch version.

Checkpoint with ordinary `state_dict`; shared-bank consumers must be recreated
with compatible bank configuration before loading.
