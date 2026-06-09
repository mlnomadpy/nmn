# NMN Architecture & Theory

This page explains the core operation behind NMN — the **Yat-Product** (ⵟ) — and the design choices that follow from it: numerical stability, parameter semantics, and where NMN layers fit into a modern architecture.

For practical usage, see the [per-platform guides](guides/). For research depth, see the [project paper](https://azettaai.github.io/nmn/paper/).

---

## 1. The Yat-Product

A traditional artificial neuron computes:

$$
y = \sigma(\mathbf{w}^\top \mathbf{x} + b)
$$

A **Yat neuron** computes:

$$
\mathrm{ⵟ}(\mathbf{w}, \mathbf{x}) = \frac{\langle \mathbf{w}, \mathbf{x} \rangle^2}{\lVert \mathbf{w} - \mathbf{x} \rVert^2 + \varepsilon}
$$

No `σ` (activation). The non-linearity is intrinsic: the operation is a ratio of a squared inner product (similarity) and a squared distance (proximity).

### Geometric reading

Rewriting in terms of norms and angle θ between **w** and **x**:

$$
\mathrm{ⵟ}(\mathbf{w}, \mathbf{x}) = \frac{\lVert\mathbf{w}\rVert^2 \lVert\mathbf{x}\rVert^2 \cos^2\theta}{\lVert\mathbf{w}\rVert^2 - 2\langle\mathbf{w},\mathbf{x}\rangle + \lVert\mathbf{x}\rVert^2 + \varepsilon}
$$

The neuron fires strongest when **w** and **x** are simultaneously:

- **aligned** (small θ → large cos²θ),
- **close** (small Euclidean distance), and
- **non-trivial in magnitude**.

A standard ReLU neuron fires strongly for any vector that is *aligned* with **w**, regardless of how far away it is. The Yat neuron does not: distance always discounts the response. This is the key behavioral difference.

---

## 2. Convolutional form

For a kernel **W** and an input patch **X** of the same shape:

$$
\mathrm{ⵟ}^*(\mathbf{W}, \mathbf{X}) = \frac{\left(\sum_{i,j} w_{ij} x_{ij}\right)^2}{\sum_{i,j} (w_{ij} - x_{ij})^2 + \varepsilon}
$$

Same identity, applied per patch. `YatConv1D`, `YatConv2D`, `YatConv3D`, and their transposes all use this operation.

---

## 3. Numerical stability

`ε` (called `epsilon` in code) appears in the denominator to prevent division by zero when **w ≈ x**.

### Choosing epsilon

| Regime                          | Suggested `epsilon`              | Why                                                                                        |
| ------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------ |
| Default (fp32)                  | `1e-5`                           | Library default. Stable for standard inputs.                                               |
| Mixed precision (fp16 / bf16)   | `1e-3` to `1e-2`                 | fp16 underflows below ~6e-5; bf16 has limited precision near zero.                         |
| Normalized inputs (`std=1`)     | `1e-5`                           | Default is fine.                                                                           |
| Very high-magnitude inputs      | scale-aware: `epsilon = c·var(x)` | Keeps the relative contribution of `ε` constant.                                           |
| Embeddings (large vocab)        | `1e-3` to `1e-5`                 | Tune via validation loss; embeddings often cluster near zero early in training.            |

If you see `NaN` in early training steps, **increase `epsilon` first** before changing learning rate or normalization. This is the most common failure mode.

### The `alpha` knob

Several layers (`use_alpha=True`) expose a learnable scalar that rescales the Yat output:

$$
y = \alpha \cdot \mathrm{ⵟ}(\mathbf{w}, \mathbf{x})
$$

`constant_alpha=True` fixes α = √2 (motivated by matching the variance of a ReLU at init). Use this when you want to swap a `Linear+ReLU` and not retune learning rate.

---

## 4. Parameter semantics across frameworks

The same Yat operation, three slightly different parameter naming conventions inherited from each framework:

| Concept                | PyTorch                        | Keras / TF                     | Flax NNX                       | Flax Linen                     |
| ---------------------- | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ |
| Input dim (Dense)      | `in_features`                  | implicit (built on first call) | `in_features`                  | implicit                       |
| Output dim (Dense)     | `out_features`                 | `features`                     | `out_features`                 | `features`                     |
| Conv input channels    | `in_channels`                  | implicit                       | `in_features`                  | implicit                       |
| Conv output channels   | `out_channels`                 | `filters`                      | `out_features`                 | `features`                     |
| Numerical stability    | `epsilon`                      | `epsilon`                      | `epsilon`                      | `epsilon`                      |
| Learnable α            | `use_alpha`                    | `use_alpha`                    | `use_alpha` / `constant_alpha` | `use_alpha` / `constant_alpha` |
| Bias                   | `use_bias`                     | `use_bias`                     | `use_bias`                     | `use_bias`                     |

Internal math is **numerically equivalent across all six frameworks** — PyTorch, TensorFlow, Keras, Flax NNX, Flax Linen, and MLX (max abs error < 1e-6 in fp32; see `tests/integration/test_cross_framework_consistency.py` and the full `(spherical × weight_normalized × learnable_epsilon × bias_mode)` matrix in `tests/integration/test_yat_nmn_parity.py`).

---

## 5. When to use NMN layers

NMN is a **drop-in replacement** for `Linear+activation` and `Conv+activation` blocks. The most reliable wins:

| Task / pattern                           | Recommendation                                                              |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| MLP classifier head                      | Replace `Linear+ReLU` with `YatNMN`. Often matches or beats baseline.       |
| CNN feature extractor                    | Replace `Conv+ReLU` with `YatConv2D`. Especially strong on small datasets.  |
| Attention block                          | `MultiHeadYatAttention` / `RotaryYatAttention`. See per-platform guides.    |
| Embeddings                               | `YatEmbed` — supports `.attend()` for retrieval-style scoring.              |
| Final classification layer               | Optional. Many architectures benefit from a final `YatNMN` with `softermax`. |
| Output regression head                   | Use `YatNMN` *without* a squasher; the Yat output is already non-negative.  |

### When NOT to use NMN

- **Strict identity / residual passthrough**: Yat is non-negative and squared, so it cannot pass negative residuals through directly. Use a standard `Linear` in skip connections.
- **Layers expected to encode signed correlations**: e.g. signed dot-product attention scores prior to softmax. Use the standard variant or a `softermax`.
- **Hard-realtime quantized inference**: ONNX export and INT8 quantization are not yet validated (see [TODO.md](../TODO.md)).

---

## 6. Squashers (activation-style alternatives)

NMN ships smooth, "soft" alternatives to the usual squashing functions, available in all frameworks:

```python
from nmn.<framework> import softermax, softer_sigmoid, soft_tanh

softermax(x, n=2)              # Softmax with power n — sharper or smoother
softer_sigmoid(x, sharpness=1) # Smooth sigmoid variant
soft_tanh(x)                   # Smooth tanh variant
```

Pair `softermax` with `YatNMN`/attention logits for a fully Yat output pipeline.

---

## 7. Advanced attention variants (Flax NNX)

| Variant                       | Description                                                                 | Complexity |
| ----------------------------- | --------------------------------------------------------------------------- | :--------: |
| `MultiHeadAttention`          | Yat attention with standard quadratic scoring                               |   O(n²)    |
| `RotaryYatAttention`          | Yat + Rotary Position Embeddings (RoPE)                                     |   O(n²)    |
| `MultiHeadAttention(use_performer=True)` | Spherical YAT-Performer: FAVOR+ random features over Yat scores  |   O(n)     |
| **Pallas fused yat attention** | Flash-attention-style fused kernel (TPU/GPU). See [`src/nmn/nnx/`](../src/nmn/nnx/). | O(n²) memory-efficient |

The fused/pallas kernel is the recommended choice for long-context training when you have GPU/TPU access; the standard `MultiHeadAttention` is the recommended choice for CPU debugging.

---

## 8. Mental model

A useful one-liner:

> **A Yat neuron is a prototype detector.** Each row of **w** is a prototype, and the neuron fires when the input is *both* aligned with *and* close to that prototype.

This is why Yat networks often learn cleaner, more interpretable representations early in training: they cannot rely on raw magnitude alignment alone.

---

## 9. Spherical Yat attention: SLAY / MAY / RAY

Quadratic Yat attention scores every query against every key — `O(n²)` in
sequence length. For long sequences NMN provides **linear-attention feature
maps**: a function `φ` such that `φ(q̂)·φ(k̂)` approximates the Yat score, which
lets the `softmax`-free readout be reordered into `O(n)` work (the standard
Performer / FAVOR+ associativity trick).

### The spherical kernel

On **unit vectors** (`‖q̂‖ = ‖k̂‖ = 1`, so `s = q̂·k̂ ∈ [-1, 1]`), the Yat score
reduces to a one-variable kernel:

$$
\kappa(s) = \frac{(s + b)^2}{(2 + \varepsilon) - 2s}
$$

This is exactly the geometric reading from § 1 specialized to the sphere: the
denominator `(2 + ε) − 2s` is `‖q̂ − k̂‖² + ε` (since `‖q̂‖² + ‖k̂‖² = 2`), and the
numerator is the squared (biased) inner product. Here `b` is a **per-head bias**
and `ε` is the Yat epsilon. The whole job of a feature map is to reproduce this
`κ(s)` cheaply and unbiasedly.

### Why bias matters

The numerator `(s + b)²` expands to `s² + 2bs + b²`. The `b²` term is a
**constant** in `s`: it lets the kernel assign a non-zero floor to *every*
key, i.e. attend diffusely. A feature map that drops `b` (sets `b = 0`) collapses
the numerator to `s²` and **cannot** represent that constant — it can only sharpen
toward the most-aligned keys. In the deployment regime `b ≥ 1`, ignoring bias is
a large, structural approximation error, not a small one.

### The three feature maps

| Map      | `performer_kind` | Kernel modelled            | Idea                                                                 |
| -------- | ---------------- | -------------------------- | ------------------------------------------------------------------- |
| **SLAY** | `"slay"`         | `s² / ((2+ε) − 2s)` (`b=0`) | Bias-free **anchor** map; the original spherical performer.         |
| **MAY**  | `"maclaurin"`    | full `(s+b)² / ((2+ε)−2s)` | **Random Maclaurin**: bias-aware, with a degree-0 (constant) feature. |
| **RAY**  | `"radial"`       | full `(s+b)² / ((2+ε)−2s)` | **Radial** Gauss-Laguerre RFF over the exponential-integral form.   |

**SLAY (anchor, `b = 0`).** Approximates only `s²/((2+ε)−2s)`. It is the cheapest
and the historical baseline, but it structurally cannot carry the bias term — so
it cannot attend diffusely.

**MAY (Random Maclaurin, bias-aware).** Expands `κ` as a power series in `s`. The
base `1/((2+ε)−2s)` has all-positive coefficients `βₙ`; multiplying by `(s+b)²`
shifts indices to `aₙ = β_{n-2} + 2b·β_{n-1} + b²·βₙ ≥ 0`. Per feature it draws a
random degree `Nᵣ ~ Categorical(aₙ/Z)` and `Nᵣ` Rademacher vectors, forming a
product of projections. **Degree-0 draws produce the constant feature** — exactly
what encodes `b²` and lets MAY attend diffusely. It is an *unbiased* estimator:
`E[φ(q̂)·φ(k̂)] = κ(q̂·k̂)`. At `b ≥ 1`, MAY clearly beats SLAY.

**RAY (radial, bias-aware).** Augments `x̂` to `z = [x̂, √b]` (folding the bias in
geometrically), takes a degree-2 sketch of the numerator, and approximates the
radial `1/(ε + r²)` denominator with a Gauss-Laguerre quadrature of random Fourier
features (using `1/(ε+r²) = ∫₀^∞ e^{−t(ε+r²)} dt`). Also bias-aware; trades a
different cost/variance profile than MAY.

### Sign-indefinite caveat

MAY/RAY features can be negative (odd-degree products / Fourier features), so the
linear-attention denominator can in principle go non-positive. The implementation
adds a stabilizing `epsilon` **only to the denominator** — never to the features,
which would bias the estimator. Empirically the maps are well-conditioned at
`b > 0`.

### Tuning note

These maps target the **deployment** regime, so set `epsilon` to the *median
squared distance* between normalized q and k for your data — **not** the `1e-5`
training default. The feature budget (`performer_num_features` / `num_features`)
trades cost for fidelity: more features raise `φ(q̂)·φ(k̂)` toward exact attention.

```python
from nmn.nnx import RotaryYatAttention
attn = RotaryYatAttention(
    embed_dim=512, num_heads=8,
    use_performer=True, performer_kind="maclaurin",  # or "slay" / "radial"
    performer_num_features=256, performer_bias=1.0, epsilon=1e-5,
    rngs=nnx.Rngs(0),
)
```

The standalone maps (`create_maclaurin_projection` / `maclaurin_features` /
`maclaurin_yat_attention`, and the `radial_*` equivalents) are exported from
`nmn.nnx` for custom attention layers. See
[`EXAMPLES.md` § MAY / RAY Linear Attention](../EXAMPLES.md#may--ray-linear-attention-flax-nnx).

---

## 10. Lazy mode — freezing only the kernel

`YatNMN(lazy=True)` (alias `freeze_kernel=True`) freezes **only the kernel
matrix**; the per-neuron scalars — bias, the learnable α, and the learnable ε
(when enabled) — stay trainable. Everything is backward compatible: `lazy=False`
is the default and changes nothing.

This separates the two kinds of capacity in a Yat layer:

- the **kernel** holds the prototypes (the expensive `in×out` matrix), and
- **α / bias / ε** are cheap per-output scalars that rescale and shift the
  response distribution.

Freezing the kernel and training only the scalars is a lightweight regime for:

- **Fine-tuning / domain shift** — keep learned prototypes, re-calibrate the
  response (α, bias) and the numerical floor (ε) for new data.
- **Probing** — measure how much signal the frozen prototypes already carry.
- **Cheap warm starts** — fewer trainable parameters, smaller optimizer state.

The mechanism differs per framework but the semantics are identical:

| Framework | Mechanism                                                                 |
| --------- | ------------------------------------------------------------------------- |
| Flax NNX  | kernel stored under a `FrozenParam` → excluded from `nnx.state(m, nnx.Param)`, so the optimizer/grad never see it. |
| PyTorch   | kernel `weight.requires_grad = False`; α / bias keep `requires_grad = True`. |

See [`EXAMPLES.md` § Lazy Mode](../EXAMPLES.md#lazy-mode--freeze-the-kernel-train-bias--α--ε) for runnable training snippets.

---

## Further reading

- [`docs/migration.md`](migration.md) — drop-in replacement cheat sheet
- [`docs/guides/`](guides/) — full per-platform walkthroughs
- [`EXAMPLES.md`](../EXAMPLES.md) — example code per framework
- [Paper website](https://azettaai.github.io/nmn/paper/) — full theory & experiments
