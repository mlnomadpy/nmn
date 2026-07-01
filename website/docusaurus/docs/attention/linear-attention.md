---
sidebar_position: 4
---

# Linear Attention (SLAY / MAY / RAY)

Quadratic YAT attention scores every query against every key — `O(n²)` in
sequence length. For long sequences, NMN provides **linear-attention feature
maps**: a function `φ` such that `φ(q̂)·φ(k̂)` approximates the YAT score, which
lets the readout be reordered into `O(n)` work (the standard Performer / FAVOR+
associativity trick).

## The Spherical YAT Kernel

On **unit vectors** (`‖q̂‖ = ‖k̂‖ = 1`, so `s = q̂·k̂ ∈ [-1, 1]`), the YAT score
reduces to a one-variable kernel:

$$
\kappa(s) = \frac{(s + b)^2}{(2 + \varepsilon) - 2s}
$$

The denominator `(2 + ε) − 2s` is `‖q̂ − k̂‖² + ε`, the numerator is the squared
(biased) inner product, `b` is a **per-head bias**, and `ε` is the YAT epsilon.
The whole job of a feature map is to reproduce this `κ(s)` cheaply and
unbiasedly.

### Why bias matters

The numerator `(s + b)²` expands to `s² + 2bs + b²`. The `b²` term is a
**constant** in `s`: it lets the kernel assign a non-zero floor to *every* key,
i.e. attend diffusely. A feature map that drops `b` (sets `b = 0`) collapses the
numerator to `s²` and **cannot** represent that constant — it can only sharpen
toward the most-aligned keys. In the deployment regime `b > 0`, ignoring bias is
a large, structural approximation error, not a small one.

## The Three Feature Maps

| Map | `performer_kind` | Kernel modelled | Idea |
|-----|------------------|-----------------|------|
| **SLAY** | `"slay"` | `s² / ((2+ε) − 2s)` (`b = 0`) | Bias-free **anchor** map; the original spherical performer. |
| **MAY** | `"maclaurin"` | full `(s+b)² / ((2+ε)−2s)` | **Random Maclaurin**: bias-aware, with a degree-0 (constant) feature. |
| **RAY** | `"radial"` | full `(s+b)² / ((2+ε)−2s)` | **Radial** Gauss-Laguerre RFF over the exponential-integral form. |

- **SLAY (anchor, `b = 0`)** approximates only `s²/((2+ε)−2s)`. It is the
  cheapest and the historical baseline, but structurally cannot carry the bias
  term — so it cannot attend diffusely.
- **MAY (Random Maclaurin, bias-aware)** expands `κ` as a power series in `s`.
  Degree-0 draws produce the **constant feature** — exactly what encodes `b²`
  and lets MAY attend diffusely. It is an *unbiased* estimator:
  `E[φ(q̂)·φ(k̂)] = κ(q̂·k̂)`. At `b > 0`, MAY clearly beats bias-free SLAY.
- **RAY (radial, bias-aware)** augments `x̂` to `z = [x̂, √b]` (folding the bias
  in geometrically) and approximates the radial denominator with a
  Gauss-Laguerre quadrature of random Fourier features. Also bias-aware; trades
  a different cost/variance profile than MAY.

:::caution Sign-indefinite features
MAY/RAY features can be **negative** (odd-degree products / Fourier features), so
the linear-attention denominator can in principle go non-positive. The
implementation adds a stabilizing `epsilon` **only to the denominator** — never
to the features, which would bias the estimator. Empirically the maps are
well-conditioned at `b > 0`.
:::

## Selecting a Map on the Attention Module

The `performer_kind` selector lives on `RotaryYatAttention` (Flax NNX), together
with `use_performer=True`:

```python
import jax.numpy as jnp
from flax import nnx
from nmn.nnx import RotaryYatAttention

attn = RotaryYatAttention(
    embed_dim=512,
    num_heads=8,
    use_performer=True,
    performer_kind="maclaurin",     # "slay" | "maclaurin" | "radial"
    performer_num_features=256,     # feature budget M
    performer_bias=1.0,             # the spherical-Yat bias b
    epsilon=1e-5,
    rngs=nnx.Rngs(0),
)

long_sequence = jnp.zeros((2, 8000, 512))  # (batch, seq_len, embed_dim)
output = attn(long_sequence)               # (2, 8000, 512), O(n) time
```

## Standalone Feature Maps

The maps are exported from **every** backend as `create_*_projection` /
`*_features` / `*_yat_attention`. The canonical kwargs are `bias=` and
`epsilon=`. This example uses Flax NNX:

```python
import jax
import jax.numpy as jnp
from nmn.nnx import (
    create_maclaurin_projection, maclaurin_features, maclaurin_yat_attention,  # MAY
    create_radial_projection,    radial_features,    radial_yat_attention,     # RAY
)

key = jax.random.key(0)
heads, head_dim = 8, 64
q = jnp.zeros((2, 256, heads, head_dim))  # (batch, seq, heads, head_dim)
k = jnp.zeros((2, 256, heads, head_dim))
v = jnp.zeros((2, 256, heads, head_dim))

# MAY — Random Maclaurin
may = create_maclaurin_projection(key, head_dim=head_dim, num_features=256,
                                  bias=1.0, epsilon=1e-5)
out_may = maclaurin_yat_attention(q, k, v, may, causal=False)  # (2, 256, 8, 64)

# RAY — radial / Gauss-Laguerre
ray = create_radial_projection(key, head_dim=head_dim, sketch_m=128,
                               num_radial=8, radial_dim=64, bias=1.0, epsilon=1e-5)
out_ray = radial_yat_attention(q, k, v, ray, causal=False)     # (2, 256, 8, 64)
```

## Tuning Note

These maps target the **deployment** regime, so set `epsilon` to the *median
squared distance* between normalized q and k for your data — **not** the `1e-5`
training default. The feature budget (`performer_num_features` / `num_features`)
trades cost for fidelity: more features raise `φ(q̂)·φ(k̂)` toward exact
attention.

Run `nmn features` to print these maps and the lazy-mode usage from the shell —
see the [CLI reference](/docs/cli).

## See Also

- [Rotary Embeddings & Performer](/docs/attention/rotary) - RoPE + performer wiring
- [YAT Attention](/docs/attention/yat-attention) - The quadratic ⵟ kernel
- [Multi-Head Attention](/docs/attention/multi-head) - Standard multi-head
