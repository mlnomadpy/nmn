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
from nmn.nnx.nmn import YatNMN

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

## Available Modules

| Module | Description |
|--------|-------------|
| [`YatNMN`](/docs/layers/yat-nmn) | Dense layer with ⵟ-product |
| [`YatConv`](/docs/layers/yat-conv) | Convolution with ⵟ-product |
| [`YatAttention`](/docs/attention/yat-attention) | Self-attention with ⵟ-product |
| [`YatLSTM`](/docs/rnn/lstm) | LSTM with ⵟ-product gates |

## Next Steps

- [**Installation**](/docs/installation) - Get NMN set up
- [**Quick Start**](/docs/quick-start) - Build your first model
- [**Interactive Paper**](/paper/) - Explore visualizations
