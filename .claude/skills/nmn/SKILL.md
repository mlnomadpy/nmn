---
name: nmn
description: >-
  Use when writing, reviewing, or debugging code that uses the `nmn`
  (Neural-Matter Network) Python library — activation-free YAT / ⵟ neural
  layers and attention that learn non-linearity geometrically. Trigger when a
  file imports `nmn` or `from nmn.<framework> import ...`; when the user
  mentions "Neural Matter Network", "YAT layer/attention", `YatNMN`, `YatConv`,
  `MultiHeadYatAttention`, `RotaryYatAttention`, the MAY/RAY performer feature
  maps, or lazy YatNMN training; or when choosing/using nmn across PyTorch,
  Flax NNX, Flax Linen, Keras, TensorFlow, or MLX. Covers per-framework imports
  and the `units`/`features`/`in_features` constructor differences, the
  SLAY/MAY/RAY performer maps, lazy mode, and the `nmn` CLI.
---

# Using the `nmn` (Neural-Matter Network) library

`nmn` ships **YAT (ⵟ) layers** — activation-free neural layers whose response is
`(⟨w,x⟩ + b)² / (‖w − x‖² + ε)`, a ratio of *similarity* to *distance*. The same
math is implemented, numerically equivalent (< 1e-6 fp32), across **six
frameworks**. Each framework is an independent optional install.

## Step 0 — check the environment first

Before writing code, confirm which backend is installed:

```bash
nmn doctor            # which of the 6 backends import + versions
nmn guide <framework> # a self-contained quickstart (torch/nnx/linen/keras/tf/mlx)
nmn features          # MAY/RAY performer maps + lazy YatNMN usage
```

`import nmn` is import-light; a framework is only pulled in when you import its
subpackage. If a backend is missing: `pip install nmn[<framework>]`
(`torch`, `nnx`, `linen`, `keras`, `tf`, `mlx`).

## Pick the framework, then use the EXACT import + constructor

The single most common mistake is the wrong `YatNMN` constructor kwarg — it
differs per framework. Use this table verbatim:

| Framework | Import | `YatNMN` constructor |
| --- | --- | --- |
| PyTorch | `from nmn.torch import YatNMN` | `YatNMN(in_features, out_features)` |
| Flax NNX | `from nmn.nnx import YatNMN` | `YatNMN(in_features, out_features, rngs=nnx.Rngs(0))` |
| Flax Linen | `from nmn.linen import YatNMN` | `YatNMN(features=N)` — call `YatNMN(features=N)(x)` |
| Keras | `from nmn.keras import YatNMN` | `YatNMN(units)` — **`units`, not `features`** |
| TensorFlow | `from nmn.tf import YatNMN` | `YatNMN(features)` |
| MLX | `from nmn.mlx import YatNMN` | `YatNMN(features)` |

Minimal MLP (Flax NNX):

```python
from flax import nnx
from nmn.nnx import YatNMN

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.fc1 = YatNMN(in_features=784, out_features=256, rngs=rngs)
        self.fc2 = YatNMN(in_features=256, out_features=10, rngs=rngs)
    def __call__(self, x):
        return self.fc2(self.fc1(x))   # no activation between layers — YAT is non-linear
```

Common knobs (same names across frameworks): `use_bias`, `use_alpha` /
`constant_alpha`, `epsilon`, `learnable_epsilon`. Conv variants exist too:
`YatConv{1,2,3}D` / `YatConvTranspose{1,2,3}D` (torch/keras/tf/linen) and
`YatConv` (nnx).

## Attention

- `MultiHeadYatAttention(embed_dim, num_heads, ...)` — Keras / TensorFlow / MLX.
- Flax NNX: `from nmn.nnx import MultiHeadAttention` → `MultiHeadAttention(num_heads, in_features, rngs=...)` (also exported as `MultiHeadYatAttention`).
- Flax Linen: `MultiHeadAttention(num_heads, qkv_features=..., out_features=...)` — **no `in_features`**.
- NNX RoPE + O(n) performer: `RotaryYatAttention(embed_dim, num_heads, use_performer=True, performer_kind="slay"|"maclaurin"|"radial", performer_num_features=256, rngs=...)`. There is **no `key_dim`** kwarg anywhere, and RotaryYatAttention takes `embed_dim` (not `in_features`) and `performer_num_features` (not `num_features`).

## Linear-attention feature maps: SLAY / MAY / RAY

Approximate the spherical-YAT kernel `κ(s) = (s+b)²/((2+ε)−2s)` so attention runs
in **O(n)**. Available across frameworks as `create_*_projection` /
`*_features` / `*_yat_attention` (canonical kwargs `bias=`, `epsilon=`):

- **SLAY** (anchor) — bias-free (`b = 0`) only.
- **MAY** (Random Maclaurin) — bias-aware; near-exact for the deployment regime `b > 0`, where it beats SLAY.
- **RAY** (radial) — bias-aware secondary route.

```python
from nmn.nnx import create_maclaurin_projection, maclaurin_yat_attention
# epsilon here is the YAT ε. MAY is near-exact when ε ≈ the median squared
# distance (~O(1) on the unit sphere), not a tiny stabilizer like 1e-5.
p = create_maclaurin_projection(key, head_dim=64, num_features=256, bias=1.0, epsilon=0.5)
out = maclaurin_yat_attention(q, k, v, p)   # q,k,v: [batch, seq, heads, head_dim]
```

MAY/RAY features are **sign-indefinite** — never add an epsilon to the features
(it biases the estimator); only stabilize the division denominator.

## Lazy training mode (freeze the kernel)

`YatNMN(..., lazy=True)` (alias `freeze_kernel=True`) freezes **only the kernel**
feature directions; **`bias`, `alpha`, and `epsilon` stay trainable**. Uses each
backend's idiomatic mechanism (NNX `FrozenParam`, torch `requires_grad=False`,
mlx `freeze`, Keras/TF `trainable=False`, linen `stop_gradient`). Default is
`lazy=False` (unchanged).

## Gotchas

- Backends are **independent optional installs** — run `nmn doctor`; don't assume torch/tf/jax/mlx is present.
- `YatNMN` kwarg differs per framework (table above) — Keras is `units`, NOT `features`.
- No `key_dim` on any attention class; `MultiHeadYatAttention` needs `embed_dim` + `num_heads`.
- Don't insert activation functions between YAT layers — the non-linearity is built in.
- There is **no RNN module** (`nmn.nnx.rnn` does not exist); any such import is wrong.
- Squashers are at the package root: `from nmn.nnx import softermax, softer_sigmoid, soft_tanh`.

## Verify

Run the whole suite (backend tests auto-skip when a framework is absent), or
target one backend's directory:

```bash
python -m pytest -q                 # full suite
python -m pytest tests/test_torch -q # one backend
```

For deeper, self-contained per-framework quickstarts, prefer `nmn guide <framework>`
over guessing — it prints verified, runnable snippets.
