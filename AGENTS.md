# AGENTS.md — NMN cheat sheet for coding agents

Concise, accurate orientation for an agent working with the `nmn` package
(Neural Matter Networks). For prose docs see [README.md](README.md),
[docs/README.md](docs/README.md), and the per-framework guides under
[docs/guides/](docs/guides/). For a machine-readable map see [llms.txt](llms.txt).

## What NMN is

`YatNMN` is a drop-in replacement for `Linear + activation`. The non-linearity is
intrinsic: `y = (x·W)² / (||x − W||² + ε) · alpha`. The same layer family
(`YatNMN`, YAT conv, YAT embedding, YAT multi-head attention) ships across six
frameworks with numerically equivalent outputs. Current release: 0.3.2.

## Pick a framework

`nmn` (src layout, `src/nmn/`) has six independent backend subpackages. Each is a
**separate optional install** — installing one does not install the others.

| Framework  | Import root  | `pip install`  | When to choose |
| ---------- | ------------ | -------------- | -------------- |
| PyTorch    | `nmn.torch`  | `nmn[torch]`   | Most ergonomic API, broad GPU support |
| Flax NNX   | `nmn.nnx`    | `nmn[nnx]`     | JAX speed + Pythonic API (recommended JAX entry point) |
| Flax Linen | `nmn.linen`  | `nmn[linen]`   | Legacy Linen codebases |
| Keras 3    | `nmn.keras`  | `nmn[keras]`   | Multi-backend Keras |
| TensorFlow | `nmn.tf`     | `nmn[tf]`      | TF/SavedModel pipelines |
| MLX        | `nmn.mlx`    | `nmn[mlx]`     | Apple Silicon, on-device |

`pip install nmn[all]` pulls torch+nnx+keras+tf+linen (MLX is Apple-Silicon-only,
install `nmn[mlx]` explicitly).

## Exact imports + YatNMN signature (verify before you write code)

The **size argument of `YatNMN` differs per backend** — this is the #1 gotcha:

```python
from nmn.torch import YatNMN     # YatNMN(in_features=128, out_features=64)
from nmn.nnx   import YatNMN     # YatNMN(in_features=128, out_features=64, rngs=nnx.Rngs(0))
from nmn.linen import YatNMN     # YatNMN(features=64)
from nmn.keras import YatNMN     # YatNMN(units=64)        (also exported as YatDense)
from nmn.tf    import YatNMN     # YatNMN(features=64)     (also exported as YatDense)
from nmn.mlx   import YatNMN     # YatNMN(features=64)
```

Only the NNX backend requires `rngs=`. NNX uses dimension-generic names
(`YatConv`, `Embed`, `MultiHeadAttention`) but also re-exports the cross-framework
aliases (`YatConv2D`, `YatEmbed`, `MultiHeadYatAttention`) so code ports cleanly.

Multi-head YAT attention:

```python
# torch / keras / tf / mlx
from nmn.torch import MultiHeadYatAttention   # (embed_dim=256, num_heads=8)
# nnx
from nmn.nnx import MultiHeadAttention        # (num_heads=8, in_features=256, rngs=nnx.Rngs(0))
```

## Per-framework gotchas

- **`YatNMN` size kwarg**: `in_features=/out_features=` (torch, nnx),
  `features=` (linen, tf, mlx), `units=` (keras). Don't assume `in_features`
  everywhere.
- **NNX needs `rngs=`**: every NNX layer takes `rngs=nnx.Rngs(seed)`.
- **Backends are independent optional installs**: never `import nmn.tf` unless TF
  is installed. `import nmn` alone imports no ML framework. In this dev env
  numpy, jax/flax, torch, and mlx are present; **TensorFlow is NOT installed**, so
  `nmn.tf` / `nmn.keras` cannot be imported here.
- **Keras also exports `YatDense`** (alias of `YatNMN`); MLX/TF do too.

## MAY / RAY linear attention + lazy mode

Bias-aware linear-attention feature maps are exported from **every** backend; the
canonical kwargs are `bias=` and `epsilon=`:

```python
from nmn.nnx import (
    create_maclaurin_projection, maclaurin_features, maclaurin_yat_attention,  # MAY
    create_radial_projection,    radial_features,    radial_yat_attention,     # RAY
)
import jax, jax.numpy as jnp
params = create_maclaurin_projection(jax.random.key(0), head_dim=64,
                                     num_features=256, bias=1.0, epsilon=1e-5)
out = maclaurin_yat_attention(q, k, v, params)   # q,k,v: [..., seq, heads, head_dim]
```

In NNX you can also select the feature map on the attention module via
`performer_kind in {"slay", "maclaurin", "radial"}` together with
`use_performer=True` (on `RotaryYatAttention`).

**Lazy / frozen-kernel training** freezes ONLY the kernel (feature directions);
bias, alpha, and the learnable epsilon stay trainable. Pass `lazy=True` (alias
`freeze_kernel=True`) to `YatNMN` in any backend:

```python
from nmn.torch import YatNMN
layer = YatNMN(in_features=128, out_features=64, lazy=True)   # weight.requires_grad = False
# nnx / linen / tf / mlx: YatNMN(..., lazy=True)  — same semantics
```

## Discover the API from the shell

Use the import-light `nmn` CLI (also `python -m nmn`) — it imports no ML
framework, so it is safe and fast for agents:

- `nmn` / `nmn info` — banner: version, the six frameworks, pip extras.
- `nmn version` — version string only.
- `nmn frameworks` — table of import line + `YatNMN` signature per framework.
- `nmn guide <framework>` — self-contained quickstart (aliases: torch/pytorch,
  tf/tensorflow, nnx/flax-nnx, linen/flax-linen, keras, mlx).
- `nmn features` — MAY/RAY maps + lazy mode, with a NNX `performer_kind` snippet.
- `nmn doctor` — per-backend import OK/missing + version; never raises.
- `nmn examples` — runnable-example pointers + a NNX quickstart.

Programmatic: `import nmn; nmn.help()` (== `nmn info`) and `nmn.doctor()`
(== `nmn doctor`, returns `{framework: version_or_None}`). **Run `nmn doctor`
first** to see which backends are importable in the current environment.

## Run the tests

```bash
python3 -m pytest -q                 # whole suite
python3 -m pytest -q -m "not slow"   # fast subset
python3 -m pytest tests/test_torch -q # one backend: point pytest at its dir
```

Tests for a backend are skipped automatically when that framework is not
installed, so a missing TensorFlow does not fail the run. To target a single
backend, point pytest at its directory (`tests/test_torch`, `tests/test_nnx`,
`tests/test_tf`, …) rather than a `-m` marker.

## Rules for changes

- Keep everything additive and backward compatible; don't break existing tests or
  doc links.
- Match each file's surrounding style.
- Verify imports and constructor kwargs against the real code in `src/nmn/`
  before writing them down.
