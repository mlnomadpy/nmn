---
sidebar_position: 1
---

# Introduction to Neural Matter Networks

**Neural Matter Networks (NMN)** introduce a fundamentally new approach to neural computation through the **ⵟ-product** (pronounced "Yat") — a geometric operator that replaces both dot products and activation functions with a single unified operation.

## The ⵟ-Product

The core operation is defined as:

$$
\text{ⵟ}(\mathbf{w}, \mathbf{x}) = \frac{\langle \mathbf{w}, \mathbf{x} \rangle^2}{\|\mathbf{w} - \mathbf{x}\|^2 + \epsilon}
$$

Where:
- **Numerator**: Squared dot product captures **alignment** between vectors
- **Denominator**: Squared Euclidean distance captures **proximity** with ε for stability

## Why ⵟ-Product?

Traditional neural networks separate geometry from non-linearity:

| Traditional | NMN |
|-------------|-----|
| Linear transformation (dot product) | ⵟ-product unifies both |
| + Activation function (ReLU, sigmoid) | No activation needed |
| Information loss from activations | Geometric structure preserved |

### Key Benefits

- **🎯 Activation-Free**: Intrinsic non-linearity through geometry
- **📉 Self-Regularizing**: Bounded responses, decaying gradients
- **∞ Universal Approximation**: Dense in C(𝒳) on compact domains
- **🎓 Mercer Kernel**: PSD, connecting to kernel theory
- **⚡ Drop-in Replacement**: Works with existing architectures

## Quick Example

```python
from flax import nnx
from nmn.nnx import YatNMN

# YatNMN replaces nn.Dense + activation
layer = YatNMN(
    in_features=768,
    out_features=256,
    constant_alpha=True,
    rngs=nnx.Rngs(0)
)

# Forward pass - no activation function needed
y = layer(x)
```

## Six Frameworks, One Layer

The same YAT layer family ships across **six independent backends** with
numerically equivalent outputs (< 1e-6 fp32). Each backend is a **separate
optional install** — installing one does not install the others.

| Framework | Import root | `pip install` | `YatNMN` size argument |
|-----------|-------------|---------------|------------------------|
| PyTorch | `nmn.torch` | `nmn[torch]` | `YatNMN(in_features, out_features)` |
| Flax NNX | `nmn.nnx` | `nmn[nnx]` | `YatNMN(in_features, out_features, rngs=nnx.Rngs(0))` |
| Flax Linen | `nmn.linen` | `nmn[linen]` | `YatNMN(features=N)` |
| Keras 3 | `nmn.keras` | `nmn[keras]` | `YatNMN(units=N)` — **`units`, not `features`** |
| TensorFlow | `nmn.tf` | `nmn[tf]` | `YatNMN(features=N)` |
| MLX | `nmn.mlx` | `nmn[mlx]` | `YatNMN(features=N)` (Apple Silicon) |

:::tip
The `YatNMN` size argument differs per backend — this is the #1 gotcha. Run
`nmn frameworks` (or see the [CLI reference](/docs/cli)) to print the exact
import line and constructor signature for every framework.
:::

## Available Modules

| Module | Description |
|--------|-------------|
| [`YatNMN`](/docs/layers/yat-nmn) | Dense layer with ⵟ-product |
| [`YatConv`](/docs/layers/yat-conv) | Convolution with ⵟ-product |
| [`YatAttention`](/docs/attention/yat-attention) | Self-attention with ⵟ-product |
| [`MultiHeadAttention`](/docs/attention/multi-head) | Multi-head ⵟ-attention |
| [`RotaryYatAttention`](/docs/attention/rotary) | RoPE + optional O(n) performer |
| [Linear attention](/docs/attention/linear-attention) | SLAY / MAY / RAY feature maps |

## Why YAT Works: Geometric Intuition

Traditional layers separate *representation* (dot product) from *non-linearity* (activation function). YAT unifies them through **geometry**.

### Alignment × Proximity

The ⵟ-product measures two things simultaneously:

- **Numerator** `(x·W + b)²`: How aligned is the input with the weight vector? Squared alignment penalizes near-orthogonal inputs, creating a natural gating effect without a sigmoid or ReLU.
- **Denominator** `‖x − W‖² + ε`: How far is the input from the weight? Nearby inputs get amplified; distant inputs get suppressed. This is the self-regularizing effect.

The ratio combines these: **large response only when input is both aligned with and close to the weight**.

### Connection to Kernel Methods

The ⵟ-product is a **Mercer kernel** — it is positive semi-definite (PSD). This means:

1. The representation is implicitly in a reproducing kernel Hilbert space (RKHS)
2. Universal approximation follows from kernel theory on compact domains
3. The network has implicit regularization properties related to minimum-norm interpolation

### Bounded Responses

For fixed `‖x‖` and `‖W‖`, the ⵟ-product output is bounded:

$$
0 \le \text{ⵟ}(\mathbf{w}, \mathbf{x}) \le \frac{\|\mathbf{w}\|^2 \|\mathbf{x}\|^2}{\epsilon}
$$

This boundedness means gradients cannot explode through the YAT operation itself — a property that ReLU networks lack without residual connections or normalization.

### Why No Activation Function

A ReLU after a linear layer introduces non-linearity by thresholding. YAT introduces non-linearity geometrically: the division by distance is a nonlinear function of both `x` and `W`. The squared numerator creates the asymmetry (polarity) that activations normally provide.

In practice this means: **fewer components, same expressiveness** — leading to the memory and parameter count reductions reported in benchmarks.

## Next Steps

- [**Installation**](/docs/installation) - Get NMN set up
- [**Quick Start**](/docs/quick-start) - Build your first model
- [**CLI Reference**](/docs/cli) - Discover the API from the shell (`nmn doctor` / `nmn guide`)
- [**Migration Guide**](/docs/migration-guide) - Drop in NMN layers
- [**Linear Attention**](/docs/attention/linear-attention) - SLAY / MAY / RAY performer maps
- [**Lazy Training**](/docs/guides/lazy-training) - Freeze the kernel, train the scalars
- [**Interactive Paper**](/paper/) - Explore visualizations
