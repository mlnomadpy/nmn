# NMN Documentation

Welcome to the **Neural Matter Networks** documentation. This is the local-repo entry point — for the hosted version with paper, blog, and visualizations, see [azettaai.github.io/nmn](https://azettaai.github.io/nmn/).

---

## Start here

| If you want to…                              | Read                                                        |
| -------------------------------------------- | ----------------------------------------------------------- |
| **Understand the math**                       | [Architecture & Theory](architecture.md)                    |
| **Port an existing model**                   | [Migration Cheat Sheet](migration.md)                       |
| **Pick the right framework**                 | [Framework comparison](#framework-comparison) below          |
| **Discover the package from a terminal**     | Run `nmn` / `nmn guide <fw>` / `nmn features` / `nmn doctor`  |
| **See code snippets per framework**          | [`EXAMPLES.md`](../EXAMPLES.md) at repo root                |
| **Run a full training script**               | [`src/nmn/<framework>/examples/`](../src/nmn/)              |
| **Contribute**                                | [`CONTRIBUTING.md`](../CONTRIBUTING.md)                     |
| **Report a security issue**                  | [`SECURITY.md`](../SECURITY.md)                             |

---

## Per-platform step-by-step guides

Each guide is self-contained: install → hello world → MNIST → CNN → attention → save/load → troubleshooting.

- [🔥 **PyTorch**](guides/pytorch.md)
- [⚡ **Flax NNX**](guides/flax-nnx.md) (recommended for JAX users)
- [⚡ **Flax Linen**](guides/flax-linen.md)
- [🟨 **Keras 3**](guides/keras.md) (multi-backend)
- [🟧 **TensorFlow**](guides/tensorflow.md)
- [🍎 **MLX**](guides/mlx.md) (Apple Silicon)

---

## Framework comparison

| Framework      | When to choose it                                                                          |
| -------------- | ------------------------------------------------------------------------------------------ |
| **PyTorch**    | You want the most ergonomic Python API and the broadest GPU support.                       |
| **Flax NNX**   | You want JAX speed + Pythonic API; recommended JAX entry point.                            |
| **Flax Linen** | You're maintaining a legacy Linen codebase.                                                |
| **Keras 3**    | You want a high-level API that runs on JAX, TF, *or* PyTorch backends.                     |
| **TensorFlow** | You need TF-specific deployment (TFLite, Serving) or `tf.Module`-level control.            |
| **MLX**        | You're on Apple Silicon and want the best perf-per-watt + native Metal GPU acceleration.   |

All six **produce numerically equivalent outputs** (max abs error < 1e-6 in fp32). You can prototype in one and serve from another.

---

## Layer support matrix

All layers are available across **all 6 frameworks**.

| Layer                                      | PyTorch                | TF                     | Keras                  | NNX                                       | Linen                  | MLX                    |
| ------------------------------------------ | :--------------------: | :--------------------: | :--------------------: | :---------------------------------------: | :--------------------: | :--------------------: |
| Dense                                       | `YatNMN`               | `YatNMN`               | `YatNMN`               | `YatNMN`                                  | `YatNMN`               | `YatNMN`               |
| Conv 1D / 2D / 3D                          | `YatConv{1,2,3}D`      | `YatConv{1,2,3}D`      | `YatConv{1,2,3}D`      | `YatConv`                                 | `YatConv{1,2,3}D`      | `YatConv{1,2,3}D`      |
| ConvTranspose 1D / 2D / 3D                 | `YatConvTranspose{1,2,3}D` | `YatConvTranspose{1,2,3}D` | `YatConvTranspose{1,2,3}D` | `YatConvTranspose`                        | `YatConvTranspose{1,2,3}D` | `YatConvTranspose{1,2,3}D` (groups=1) |
| Multi-Head Attention                       | `MultiHeadYatAttention`| `MultiHeadYatAttention`| `MultiHeadYatAttention`| `MultiHeadAttention`                      | `MultiHeadAttention`   | `MultiHeadYatAttention`|
| Embedding                                  | `YatEmbed`             | `YatEmbed`             | `YatEmbed`             | `Embed`                                   | `YatEmbed`             | `YatEmbed`             |
| Squashers (`softermax`, `softer_sigmoid`, `soft_tanh`) | ✅       | ✅                     | ✅                     | ✅                                        | ✅                     | ✅                     |

**NNX-only advanced variants:**

- `RotaryYatAttention` — YAT + RoPE positional encoding
- `MultiHeadAttention(use_performer=True)` — Spherical YAT-Performer, O(n) complexity
- `RotaryYatAttention(use_performer=True, performer_kind=...)` — selects the
  linear-attention feature map: `"slay"` (default, bias-free anchor), `"maclaurin"`
  (**MAY**, bias-aware Random-Maclaurin), or `"radial"` (**RAY**, bias-aware radial
  RFF). Functional API: `create_maclaurin_projection` / `maclaurin_yat_attention`
  and `create_radial_projection` / `radial_yat_attention` (canonical kwargs
  `bias`, `epsilon`). See the [Flax NNX guide](guides/flax-nnx.md#6-linear-attention-may--ray--slay).
- Pallas fused / flash yat-attention kernel (TPU/GPU)

**Lazy mode — freeze the kernel (all frameworks):** every `YatNMN` accepts
`lazy=True` (alias `freeze_kernel=True`) to freeze **only** the kernel (its
feature directions) while `bias`, `alpha`, and the learnable `epsilon` keep
training. Each framework implements it idiomatically — `requires_grad=False`
(torch), `trainable=False` (tf/keras), `FrozenParam` (nnx), `stop_gradient`
(linen), `Module.freeze(keys=["kernel"])` (mlx). See each guide's "Lazy training
(freeze kernel)" section.

---

## Repository layout

```
nmn/
├── src/nmn/
│   ├── torch/      PyTorch implementation + examples
│   ├── nnx/        Flax NNX implementation + examples + Pallas kernels
│   ├── linen/      Flax Linen implementation
│   ├── keras/      Keras 3 implementation
│   ├── tf/         TensorFlow implementation
│   └── mlx/        MLX implementation (Apple Silicon)
├── tests/
│   ├── test_<framework>/   Per-framework unit tests
│   ├── integration/         Cross-framework consistency
│   └── benchmarks/          Performance comparisons
├── docs/           ← you are here
│   ├── architecture.md
│   ├── migration.md
│   └── guides/<framework>.md
├── website/        Docusaurus site (paper, blog, hosted docs)
├── README.md
├── EXAMPLES.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
└── CHANGELOG.md
```

---

## Troubleshooting (general)

Most issues fall into one of these buckets:

| Symptom                                  | First thing to try                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| `NaN` loss within first ~100 steps        | Bump `epsilon` from `1e-5` → `1e-3` (especially fp16/bf16)                |
| Slow / no learning early                  | Add `use_alpha=True` or `constant_alpha=True`                              |
| Numerical mismatch across frameworks      | Run `pytest tests/integration/ -v` to confirm parity on your machine     |
| Cannot install jax / version conflict     | Pin: `pip install "jax==0.9.1" "jaxlib==0.9.1" "flax==0.12.5"`            |
| Cannot import `nmn.<framework>`          | Install the matching extra: `pip install "nmn[<framework>]"`              |

Framework-specific troubleshooting tables live at the bottom of each guide.

---

## Where things are *not* (yet)

The following items are tracked in [`TODO.md`](../TODO.md):

- **ONNX export validation** — Yat layers use standard ops; export *should* work but isn't tested in CI.
- **INT8 quantization** — not yet implemented.
- **Auto-generated Sphinx/`mkdocstrings` API reference** — current API docs live in docstrings + this hand-written set.

Contributions on any of these are welcome — see [`CONTRIBUTING.md`](../CONTRIBUTING.md).
