---
name: nmn
description: Use when writing, porting, reviewing, debugging, training, exporting, or benchmarking code with the nmn Neural Matter Network Python package. Trigger on `import nmn`, `nmn.torch`, `nmn.nnx`, `nmn.linen`, `nmn.keras`, `nmn.tf`, `nmn.mlx`, YatNMN/YatDense, YAT convolution or embedding, MultiHeadYatAttention, RotaryYatAttention, SLAY/MAY/RAY/GOAT attention, Pallas YAT kernels, tied kernel banks, learnable epsilon, lazy or mixed-precision NMN training. Provides verified per-framework constructors, tensor layouts, masking semantics, serialization, accelerator guidance, and tests for PyTorch, Flax NNX, Flax Linen, Keras 3, TensorFlow, and MLX.
---

# Use NMN

Treat the installed package and repository source as authoritative. NMN ships
the YAT family across six independent backends; similar class names do not imply
identical constructor names, layouts, state APIs, or export formats.

## Start with discovery

Run these before writing backend-specific code:

```bash
nmn version
nmn doctor
nmn frameworks
nmn guide <torch|nnx|linen|keras|tf|mlx>
nmn features
```

If working from a checkout, inspect `src/nmn/<backend>/__init__.py` and the real
class signature. Do not infer one backend's kwargs from another.

Install exactly the backend needed:

```bash
python -m pip install "nmn[torch]"
python -m pip install "nmn[nnx]"
python -m pip install "nmn[linen]"
python -m pip install "nmn[keras]"
python -m pip install "nmn[tf]"
python -m pip install "nmn[mlx]"
```

`nmn[all]` excludes MLX because MLX requires Apple Silicon. Importing `nmn`
itself is framework-light; importing a backend requires that backend's extra.

## Select the backend reference

Read only the reference relevant to the task:

- [PyTorch](references/pytorch.md): eager/compile, tied banks, `param_dtype`.
- [Flax NNX](references/flax-nnx.md): NNX state, JIT, decode, performers,
  precision modes, and Pallas TPU/GPU attention.
- [Flax Linen](references/flax-linen.md): `init`/`apply`, Optax freezing,
  grouped convolution, softmax/L1 attention.
- [Keras 3](references/keras.md): backend selection, serialization, shared
  convolution banks, and channels-first portability.
- [TensorFlow](references/tensorflow.md): native TF modules, `tf.function`,
  grouped convolution, and SavedModel signatures.
- [MLX](references/mlx.md): Apple GPU, eager/fused paths, compile, decode, and
  transpose-SAME semantics.

For behavior shared by all backends, read
[the cross-framework contract](references/shared-contract.md).

## Constructor map

Use the exact dense constructor for the selected backend:

| Backend | Import | Construction |
| --- | --- | --- |
| PyTorch | `from nmn.torch import YatNMN` | `YatNMN(in_features=128, out_features=64)` |
| Flax NNX | `from nmn.nnx import YatNMN` | `YatNMN(128, 64, rngs=nnx.Rngs(0))` |
| Flax Linen | `from nmn.linen import YatNMN` | `YatNMN(features=64)` then `init`/`apply` |
| Keras 3 | `from nmn.keras import YatNMN` | `YatNMN(units=64)` |
| TensorFlow | `from nmn.tf import YatNMN` | `YatNMN(features=64)`; input width is lazy |
| MLX | `from nmn.mlx import YatNMN` | `YatNMN(features=64)`; input width is lazy |

Common aliases are intentionally incomplete: Keras/TF/MLX export `YatDense`;
NNX exports dimension-generic `YatConv` and `YatConvTranspose` plus aliases;
Linen attention is `MultiHeadAttention`; Torch/Keras/TF/MLX use
`MultiHeadYatAttention`.

## Core semantics

For an input vector `x` and output kernel vector `w`, YAT computes a learned
scale of

```text
((dot(x, w) + bias) ** 2) / (squared_distance(x, w) + epsilon)
```

The ratio is already nonlinear. Do not automatically insert ReLU/GELU after a
YAT layer. `constant_alpha=True` means the backend's documented constant;
a numeric `constant_alpha` uses that value. `lazy=True` / `freeze_kernel=True`
freezes only the kernel; bias, alpha, and learnable epsilon remain trainable.

Pass a finite, strictly positive `epsilon`; treat this as a caller precondition
because validation is not yet uniform across every backend. When
`learnable_epsilon=True`, the public effective epsilon is
`softplus(epsilon_param)`. Keep optimizer and checkpoint code aware that
low-precision layers may retain epsilon state in FP32 to avoid underflow or
overflow.

## Attention and masks

The portable functional-attention layout is `[batch, query_length, heads,
head_dim]` for Q and `[batch, key_length, heads, head_dim]` for K/V. Some JAX
functions, including Pallas attention, document extra leading batch dimensions;
do not assume that extension in Torch, TensorFlow, or MLX. Modules generally
accept batch-major `[batch, sequence, embedding]` inputs.

Boolean masks use `True` for allowed positions and may be broadcast from common
`[Q,K]`, `[H,Q,K]`, `[B,1,Q,K]`, or `[B,H,Q,K]` forms. A rank-3 mask aligns to
heads, not batches. A fully masked query row produces exactly zero attention
weights and exactly zero module output, including after an output projection
with bias. Do not replace this with a uniform row or NaNs.

Use SLAY for the bias-free spherical-anchor regime. Use MAY (`maclaurin`) or
RAY (`radial`) when the kernel bias matters. Their features are sign-indefinite;
stabilize only the final denominator, never the features themselves.

## Numeric policy

NMN promotes vulnerable low-precision score reductions where necessary,
preserves the requested public output dtype, saturates finite values that do not
fit that dtype, and preserves genuine NaNs. Compare low-precision behavior to a
synchronized FP32 reference for outputs and all trainable/input gradients.

A mathematically out-of-range gradient accumulated from multiple independent
uses of the same FP16 leaf cannot be represented in FP16. Prefer FP32 parameter
storage, autocast/loss scaling, or BF16 where appropriate; do not claim arbitrary
FP16 leaf accumulation remains exact.

## Verification workflow

Test the narrow backend first, then shared parity and the full available suite:

```bash
python -m pytest -q tests/test_<backend>
python -m pytest -q tests/integration
python -m pytest -q
```

Missing optional backends should skip cleanly. Validate on the real accelerator
for accelerator-specific paths: Pallas on TPU/GPU and fused MLX on Apple Metal.
CPU interpretation proves algebra, not native lowering.

When porting between backends, synchronize every parameter explicitly and test
forward values plus input/kernel/bias/alpha/epsilon gradients. Compare semantics,
not initializer randomness or framework-default layouts.
