---
sidebar_position: 3
---

# Transformer Example

Building a transformer with YatAttention.

## Model

```python
from nmn.nnx.attention import YatAttention
from nmn.nnx.nmn import YatNMN

class TransformerBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        self.attn = YatAttention(dim, num_heads, rngs=rngs)
        self.ff = YatNMN(dim, dim, constant_alpha=True, rngs=rngs)
    
    def __call__(self, x, mask=None):
        x = x + self.attn(x, mask=mask)
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

## Key Points

- **No LayerNorm needed**: YatAttention is self-regularizing
- **No GELU/ReLU in FFN**: YatNMN provides non-linearity
- **Simpler architecture**: Fewer components to tune
