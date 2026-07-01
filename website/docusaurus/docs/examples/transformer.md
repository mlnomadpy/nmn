---
sidebar_position: 3
---

# Transformer Example

Building a transformer with multi-head ⵟ-attention (Flax NNX).

## Model

```python
from flax import nnx
from nmn.nnx import MultiHeadAttention, YatNMN

class TransformerBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        # NNX: MultiHeadAttention(num_heads, in_features, rngs=...)
        self.attn = MultiHeadAttention(num_heads=num_heads, in_features=dim, rngs=rngs)
        self.ff = YatNMN(dim, dim, constant_alpha=True, rngs=rngs)

    def __call__(self, x, mask=None):
        x = x + self.attn(x, x, x, mask=mask)   # self-attention: q = k = v
        x = x + self.ff(x)
        return x

class Transformer(nnx.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, rngs: nnx.Rngs):
        self.embed = nnx.Embed(vocab_size, dim, rngs=rngs)
        self.blocks = [
            TransformerBlock(dim, num_heads, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.head = nnx.Linear(dim, vocab_size, rngs=rngs)

    def __call__(self, x, mask=None):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(x)
```

For long sequences, swap `MultiHeadAttention` for
[`RotaryYatAttention`](/docs/attention/rotary) with `use_performer=True` to get
O(n) [linear attention](/docs/attention/linear-attention) (SLAY / MAY / RAY).

In torch / Keras / TensorFlow / MLX the class is
`MultiHeadYatAttention(embed_dim, num_heads)` — there is no `key_dim` kwarg on
any backend.

## Key Points

- **No LayerNorm needed**: ⵟ-attention is self-regularizing
- **No GELU/ReLU in FFN**: YatNMN provides non-linearity
- **Simpler architecture**: Fewer components to tune
