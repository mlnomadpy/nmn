---
sidebar_position: 2
---

# MultiHeadAttention

Multi-head attention with ⵟ-product query-key similarity.

## Import

```python
from nmn.nnx.attention import MultiHeadYatAttention
```

## Usage

```python
from nmn.nnx.attention import MultiHeadYatAttention

mha = MultiHeadYatAttention(
    embed_dim=512,
    num_heads=8,
    rngs=nnx.Rngs(0)
)

# Self-attention
x = jnp.ones((16, 128, 512))
y = mha(x, x, x)
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

- [YatAttention](/docs/attention/yat-attention) - Basic attention
