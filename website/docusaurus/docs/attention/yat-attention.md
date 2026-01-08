---
sidebar_position: 1
---

# YatAttention

Self-attention mechanism using the ⵟ-product for query-key similarity.

## Mathematical Formulation

$$
\text{ⵟ-Attn}(Q, K, V) = \text{softmax}\left(s \cdot Q \text{ ⵟ } K^T\right) V
$$

Where the ⵟ-product computes geometric similarity between queries and keys.

## Import

```python
from nmn.nnx.attention import YatAttention
```

## Constructor

```python
YatAttention(
    embed_dim: int,
    num_heads: int = 8,
    *,
    qkv_features: Optional[int] = None,
    out_features: Optional[int] = None,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    epsilon: float = 1e-5,
    dtype: Optional[Dtype] = None,
    rngs: nnx.Rngs,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | `int` | required | Embedding dimension |
| `num_heads` | `int` | `8` | Number of attention heads |
| `qkv_features` | `int \| None` | `None` | Q, K, V projection dim (defaults to embed_dim) |
| `out_features` | `int \| None` | `None` | Output projection dim (defaults to embed_dim) |
| `use_bias` | `bool` | `True` | Use bias in projections |
| `dropout_rate` | `float` | `0.0` | Attention dropout rate |
| `deterministic` | `bool` | `False` | Disable dropout |
| `epsilon` | `float` | `1e-5` | Stability constant |

## Usage Example

```python
from flax import nnx
from nmn.nnx.attention import YatAttention

# Create attention layer
attn = YatAttention(
    embed_dim=512,
    num_heads=8,
    dropout_rate=0.1,
    rngs=nnx.Rngs(0)
)

# Self-attention
x = jnp.ones((16, 128, 512))  # (batch, seq_len, embed_dim)
y = attn(x)  # Shape: (16, 128, 512)
```

## With Masking

```python
# Causal mask for autoregressive models
mask = jnp.triu(jnp.ones((128, 128)), k=1) == 0

y = attn(x, mask=mask)
```

## See Also

- [MultiHeadAttention](/docs/attention/multi-head) - Full multi-head implementation
- [Rotary Embeddings](/docs/attention/rotary) - Positional encoding
