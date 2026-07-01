---
sidebar_position: 3
---

# Rotary Position Embeddings & Performer

Rotary position embeddings (RoPE) with ⵟ-attention for positional encoding, plus optional Performer mode for linear complexity.

## Import

```python
from nmn.nnx import RotaryYatAttention
```

## Basic Usage

```python
from nmn.nnx import RotaryYatAttention
from flax import nnx
import jax.numpy as jnp

attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=2048,
    constant_alpha=True,   # Use constant sqrt(2) alpha scaling
    rngs=nnx.Rngs(0)
)

x = jnp.ones((16, 512, 512))  # (batch, seq_len, embed_dim)
y = attn(x)
```

## Performer Mode — O(n) Complexity

For long sequences, enable Performer mode using linear-attention feature maps.
Pick the map with `performer_kind` — `"slay"` (bias-free anchor), `"maclaurin"`
(**MAY**, bias-aware), or `"radial"` (**RAY**, bias-aware):

```python
performer_attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=10000,
    use_performer=True,             # Enable linear complexity
    performer_kind="maclaurin",     # "slay" | "maclaurin" | "radial"
    performer_num_features=256,     # feature budget M
    performer_bias=1.0,             # spherical-Yat bias b
    epsilon=1e-5,
    rngs=nnx.Rngs(0)
)

# Efficient for very long sequences
long_seq = jnp.ones((2, 10000, 512))  # 10K tokens
output = performer_attn(long_seq)  # Runs in O(n) time instead of O(n²)
```

:::note
The kwargs are `performer_num_features` (not `num_features`) and
`performer_bias`. See [Linear Attention](/docs/attention/linear-attention) for
what the three feature maps model and when to pick each.
:::

### Complexity Comparison

| Mode | Complexity | Best For |
|------|------------|----------|
| Standard | O(n²) | Short to medium sequences (< 2K tokens) |
| Performer | O(n) | Long sequences (> 2K tokens) |

## Configuration Options

```python
attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=2048,

    # YAT Options
    constant_alpha=True,     # Use constant sqrt(2) scaling (None = learnable alpha)
    use_alpha=True,          # Enable alpha scaling
    learnable_epsilon=True,  # Make epsilon learnable (softplus-constrained)
    normalize_qk=True,       # Normalize Q and K
    normalization="softmax", # "softmax" (default), "l1", or "softermax"
    epsilon=1e-5,            # Numerical stability

    # Performer Options (linear attention)
    use_performer=False,        # Enable O(n) complexity
    performer_kind="slay",      # "slay" | "maclaurin" | "radial"
    performer_num_features=256, # feature budget (if use_performer=True)
    performer_bias=1.0,         # spherical-Yat bias b (MAY / RAY)

    rngs=nnx.Rngs(0)
)
```

## Benefits

- **Relative position encoding**: Naturally encodes relative positions
- **Extrapolation**: Better length generalization
- **Compatible with caching**: Works with KV-cache for inference
- **Linear complexity**: Optional O(n) scaling for long sequences

## See Also

- [YAT Attention](/docs/attention/yat-attention) - Concept + GOAT (MLX)
- [Multi-Head Attention](/docs/attention/multi-head) - Standard multi-head
- [Linear Attention](/docs/attention/linear-attention) - SLAY / MAY / RAY feature maps
