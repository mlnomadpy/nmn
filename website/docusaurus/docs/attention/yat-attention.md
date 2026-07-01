---
sidebar_position: 1
---

# YAT Attention

Attention using the ⵟ-product for query–key similarity. Instead of scaled dot
products, YAT attention scores each query against each key with the geometric
ⵟ kernel — combining alignment and proximity — then normalizes the scores.

## Mathematical Formulation

$$
\text{ⵟ-Attn}(Q, K, V) = \text{norm}\!\left(\frac{(Q K^\top + b)^2}{\lVert Q - K\rVert^2 + \varepsilon}\right) V
$$

where the ⵟ-product computes geometric similarity between queries and keys, and
`norm` is `softmax` (default) or an L1 / softermax readout. Because YAT scores
are non-negative, the L1 readout is often more stable.

## The Attention Classes

There is **no** standalone `YatAttention` class. YAT attention is exposed as
multi-head modules, and the constructor differs per backend:

| Framework | Import | Constructor |
|-----------|--------|-------------|
| Flax NNX | `from nmn.nnx import MultiHeadAttention` | `MultiHeadAttention(num_heads, in_features, rngs=nnx.Rngs(0))` (also exported as `MultiHeadYatAttention`) |
| Flax Linen | `from nmn.linen import MultiHeadAttention` | `MultiHeadAttention(num_heads, qkv_features=, out_features=)` |
| PyTorch | `from nmn.torch import MultiHeadYatAttention` | `MultiHeadYatAttention(embed_dim, num_heads)` |
| Keras 3 | `from nmn.keras import MultiHeadYatAttention` | `MultiHeadYatAttention(embed_dim, num_heads)` |
| TensorFlow | `from nmn.tf import MultiHeadYatAttention` | `MultiHeadYatAttention(embed_dim, num_heads)` |
| MLX | `from nmn.mlx import MultiHeadYatAttention` | `MultiHeadYatAttention(embed_dim, num_heads)` |

:::caution
There is **no `key_dim` kwarg** on any backend. torch / Keras / TensorFlow / MLX
take `embed_dim` + `num_heads`; NNX takes `num_heads` + `in_features`; Linen
takes `num_heads` + `qkv_features` (**no `in_features`**).
:::

## Usage Example (Flax NNX)

```python
from flax import nnx
from nmn.nnx import MultiHeadAttention
import jax.numpy as jnp

# Multi-head YAT self-attention
attn = MultiHeadAttention(
    num_heads=8,
    in_features=512,
    dropout_rate=0.1,
    rngs=nnx.Rngs(0),
)

# Self-attention: query = key = value
x = jnp.ones((16, 128, 512))   # (batch, seq_len, embed_dim)
y = attn(x, x, x)              # Shape: (16, 128, 512)
```

## With Masking

```python
# Causal mask for autoregressive models (True = attend)
mask = jnp.tril(jnp.ones((128, 128))).astype(bool)

y = attn(x, x, x, mask=mask)
```

## MLX: GOAT (projection-free) YAT Attention

The MLX backend additionally ships **GOAT** — value-only / projection-free
YAT-kernel attention that drops the query/key projections entirely and reads out
an L1-normalized (Nadaraya–Watson) kernel smoother instead of a softmax:

```python
from nmn.mlx import GoatYatAttention

# variant="v"  → GOAT-V   (value-only, keeps W_V and W_O)
# variant="nov" → GOAT-noV (projection-free, only W_O)
attn = GoatYatAttention(embed_dim=512, num_heads=8, variant="v")
```

The self-mask is mandatory for self-attention (`self_mask=True`, the default),
so a query never attends purely to itself.

## See Also

- [MultiHeadAttention](/docs/attention/multi-head) - Multi-head details + cross-attention
- [Rotary Embeddings](/docs/attention/rotary) - RoPE positional encoding + performer
- [Linear Attention](/docs/attention/linear-attention) - O(n) SLAY / MAY / RAY feature maps
