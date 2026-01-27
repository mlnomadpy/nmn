---
sidebar_position: 3
---

# Rotary Position Embeddings

Rotary position embeddings (RoPE) with ⵟ-attention for positional encoding.

## Import

```python
from nmn.nnx.attention import RotaryYatAttention
```

## Usage

```python
from nmn.nnx.attention import RotaryYatAttention

attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=2048,
    rngs=nnx.Rngs(0)
)

x = jnp.ones((16, 512, 512))
y = attn(x)
```

## Benefits

- **Relative position encoding**: Naturally encodes relative positions
- **Extrapolation**: Better length generalization
- **Compatible with caching**: Works with KV-cache for inference

## See Also

- [YatAttention](/docs/attention/yat-attention) - Basic attention
