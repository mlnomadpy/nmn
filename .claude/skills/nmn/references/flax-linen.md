# Flax Linen

Install with `python -m pip install "nmn[linen]"`. Linen modules use immutable
variables and `init`/`apply`.

```python
import jax
import jax.numpy as jnp
from nmn.linen import YatNMN

layer = YatNMN(features=64, learnable_epsilon=True)
x = jnp.ones((8, 128))
variables = layer.init(jax.random.key(0), x)
y = layer.apply(variables, x)
```

Convolution uses channels-last and tuple geometry:

```python
from nmn.linen import YatConv2D

conv = YatConv2D(features=32, kernel_size=(3, 3), padding="SAME",
                feature_group_count=1)
variables = conv.init(jax.random.key(1), jnp.ones((4, 64, 64, 3)))
```

Grouped convolution requires input and output features divisible by
`feature_group_count`. Patch norms are group-local; preserve contiguous
group-to-output mapping when writing references or ports.

## Attention

```python
from nmn.linen import MultiHeadAttention

attn = MultiHeadAttention(num_heads=8, qkv_features=128,
                          normalization="softmax")
variables = attn.init(jax.random.key(2), jnp.ones((2, 32, 128)))
y = attn.apply(variables, jnp.ones((2, 32, 128)), deterministic=True)
```

Use `normalization="l1"` for direct normalization of non-negative YAT scores.
Constant and learnable alpha both scale scores before normalization. Masks
broadcast and fully masked rows produce exact-zero module outputs.

## Training and precision

`lazy=True` applies `stop_gradient` to the kernel. Also use an Optax mask (for
example `multi_transform` with `set_to_zero`) if the kernel must be absent from
optimizer state. Low-precision QR/orthogonal initialization is performed safely
and score reductions preserve FP16/BF16 output policy. Learned epsilon may use
wider state; do not cast the whole variables tree blindly when that would erase
its representability.

Use `jax.jit`, `grad`, `jvp`, `jacfwd`, and batched transforms normally; the
safe precision boundary is transform-compatible.
