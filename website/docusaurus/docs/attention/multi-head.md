---
sidebar_position: 2
---

# MultiHeadAttention

Multi-head attention with ⵟ-product query–key similarity.

## Import

The class name and constructor differ per backend:

| Framework | Import | Constructor |
|-----------|--------|-------------|
| Flax NNX | `from nmn.nnx import MultiHeadAttention` | `MultiHeadAttention(num_heads, in_features, rngs=nnx.Rngs(0))` |
| Flax Linen | `from nmn.linen import MultiHeadAttention` | `MultiHeadAttention(num_heads, qkv_features=, out_features=)` |
| PyTorch | `from nmn.torch import MultiHeadYatAttention` | `MultiHeadYatAttention(embed_dim, num_heads)` |
| Keras 3 | `from nmn.keras import MultiHeadYatAttention` | `MultiHeadYatAttention(embed_dim, num_heads)` |
| TensorFlow | `from nmn.tf import MultiHeadYatAttention` | `MultiHeadYatAttention(embed_dim, num_heads)` |
| MLX | `from nmn.mlx import MultiHeadYatAttention` | `MultiHeadYatAttention(embed_dim, num_heads)` |

In NNX, `MultiHeadYatAttention` is an alias of `MultiHeadAttention` (same
`num_heads`, `in_features` signature). **No backend takes a `key_dim` kwarg.**

## Usage (Flax NNX)

```python
from flax import nnx
from nmn.nnx import MultiHeadAttention
import jax.numpy as jnp

mha = MultiHeadAttention(
    num_heads=8,
    in_features=512,
    rngs=nnx.Rngs(0),
)

# Self-attention
x = jnp.ones((16, 128, 512))
y = mha(x, x, x)
```

## Usage (torch / Keras / TensorFlow / MLX)

```python
from nmn.torch import MultiHeadYatAttention   # or nmn.keras / nmn.tf / nmn.mlx

mha = MultiHeadYatAttention(embed_dim=512, num_heads=8)
```

## Cross-Attention

```python
# Encoder-decoder attention
query = jnp.ones((16, 64, 512))   # Decoder states
key = jnp.ones((16, 128, 512))    # Encoder outputs
value = jnp.ones((16, 128, 512))  # Encoder outputs

y = mha(query, key, value)  # Shape: (16, 64, 512)
```

## See Also

- [YAT Attention](/docs/attention/yat-attention) - Concept + GOAT (MLX)
- [Rotary Embeddings](/docs/attention/rotary) - RoPE + performer
- [Linear Attention](/docs/attention/linear-attention) - O(n) SLAY / MAY / RAY
