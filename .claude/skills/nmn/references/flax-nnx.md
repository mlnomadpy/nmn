# Flax NNX

Install with `python -m pip install "nmn[nnx]"`. Every module constructor needs
`rngs=nnx.Rngs(...)`.

```python
import jax
import jax.numpy as jnp
from flax import nnx
from nmn.nnx import YatNMN

model = YatNMN(128, 64, learnable_epsilon=True, rngs=nnx.Rngs(0))
x = jnp.ones((8, 128))
y = model(x)
params = nnx.state(model, nnx.Param)
```

NNX uses channels-last, dimension-generic `YatConv` / `YatConvTranspose`; the
dimension is inferred from `kernel_size`.

```python
from nmn.nnx import YatConv

conv = YatConv(3, 32, kernel_size=(3, 3), padding="SAME", rngs=nnx.Rngs(1))
y = conv(jnp.ones((4, 64, 64, 3)))
```

`lazy=True` stores the kernel as `FrozenParam`, excluding it from ordinary
`nnx.Param` optimizer state. Tied banks are incompatible with lazy mode.

## Attention and decode

```python
from nmn.nnx import MultiHeadAttention, RotaryYatAttention

mha = MultiHeadAttention(num_heads=8, in_features=128, rngs=nnx.Rngs(2))
y = mha(jnp.ones((2, 32, 128)), deterministic=True)

rope = RotaryYatAttention(
    embed_dim=128,
    num_heads=8,
    use_performer=True,
    performer_kind="maclaurin",
    performer_num_features=256,
    rngs=nnx.Rngs(3),
)
```

Incremental `decode=True` uses mutable cache state and RoPE offsets. Initialize
cache capacity deliberately; overflow and invalid masks fail before committing
cache or RNG state. Performer modes accept key-padding masks and reject general
query-dependent masks.

## Precision modes

`compute_mode="fp32"` is the stable default. For TPU BF16 storage, prefer
`compute_mode="mixed"`; it requests FP32 accumulation without full operand
copies. `compute_mode="bf16"` is experimental and should use a suitable
`distance_floor`. All modes are covered by standard/fused parity tests.

## Pallas attention

Import the Pallas function from the attention submodule:

```python
from nmn.nnx.layers.attention import pallas_yat_l1_attention

out = pallas_yat_l1_attention(
    q, k, v,
    causal=True,
    block_q=128,
    block_k=128,
    precision=jax.lax.Precision.HIGHEST,
)
```

Q/K/V use `[..., sequence, heads, features]`; Q and KV lengths and value/head
dimensions may differ as documented. On TPU, use tile sizes legal for Mosaic
(multi-tile sequence blocks are multiples of 8) and validate native lowering.
`interpret=True` is a CPU algebra test only. `Precision.HIGHEST` prevents TPU
matrix-unit rounding of FP32 dots when parity matters.
