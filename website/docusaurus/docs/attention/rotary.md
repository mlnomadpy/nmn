---
sidebar_position: 3
---

# Rotary Position Embeddings & Performer

Rotary position embeddings (RoPE) with ⵟ-attention for positional encoding, plus optional Performer mode for linear complexity.

## Import

```python
from nmn.nnx.attention import RotaryYatAttention
```

## Basic Usage

```python
from nmn.nnx.attention import RotaryYatAttention
from flax import nnx
import jax.numpy as jnp

attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=2048,
    use_yat=True,          # Use YAT formula (default)
    constant_alpha=True,   # Use constant alpha scaling
    rngs=nnx.Rngs(0)
)

x = jnp.ones((16, 512, 512))  # (batch, seq_len, embed_dim)
y = attn(x)
```

## Performer Mode — O(n) Complexity

For long sequences, enable Performer mode using FAVOR+ random feature approximation:

```python
performer_attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    max_seq_len=2048,
    use_performer=True,    # Enable linear complexity
    num_features=256,      # Number of random features
    use_yat=True,          # Combine Performer with YAT
    rngs=nnx.Rngs(0)
)

# Efficient for very long sequences
long_seq = jnp.ones((2, 10000, 512))  # 10K tokens
output = performer_attn(long_seq)  # Runs in O(n) time instead of O(n²)
```

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
    use_yat=True,          # Use YAT formula vs standard attention
    constant_alpha=True,   # Use constant sqrt(2) scaling
    learnable_alpha=False, # Make alpha learnable
    learnable_epsilon=True, # Make epsilon learnable (softplus-constrained)
    qk_norm=True,          # Normalize Q and K
    
    # Performer Options
    use_performer=False,   # Enable linear complexity
    num_features=256,      # Random features (if performer=True)
    
    rngs=nnx.Rngs(0)
)
```

## Benefits

- **Relative position encoding**: Naturally encodes relative positions
- **Extrapolation**: Better length generalization
- **Compatible with caching**: Works with KV-cache for inference
- **Linear complexity**: Optional O(n) scaling for long sequences

## See Also

- [YatAttention](/docs/attention/yat-attention) - Basic attention
- [Multi-Head Attention](/docs/attention/multi-head) - Standard multi-head
