---
sidebar_position: 1
---

# Lazy Training (Freeze the Kernel)

`YatNMN(lazy=True)` (alias `freeze_kernel=True`) freezes **only the kernel
matrix** — the per-neuron scalars (bias, the learnable α, and the learnable ε
when enabled) stay trainable. Everything is backward compatible: `lazy=False` is
the default and changes nothing.

Available in **every** backend with identical semantics.

## Why Freeze Only the Kernel

A YAT layer holds two kinds of capacity:

- the **kernel** — the prototypes (the expensive `in × out` matrix), and
- **α / bias / ε** — cheap per-output scalars that rescale and shift the
  response distribution.

Freezing the kernel and training only the scalars is a lightweight regime for:

- **Fine-tuning / domain shift** — keep learned prototypes, re-calibrate the
  response (α, bias) and the numerical floor (ε) for new data.
- **Probing** — measure how much signal the frozen prototypes already carry.
- **Cheap warm starts** — fewer trainable parameters, smaller optimizer state.

## Per-Framework Mechanism

The mechanism differs per framework, but the semantics are identical — only the
kernel is frozen:

| Framework | Mechanism |
|-----------|-----------|
| Flax NNX | Kernel stored under a `FrozenParam` → excluded from `nnx.state(m, nnx.Param)`, so the optimizer/grad never see it. |
| PyTorch | Kernel `weight.requires_grad = False`; α / bias keep `requires_grad = True`. |
| MLX | Kernel `freeze`-d; α / bias stay trainable. |
| Keras / TensorFlow | Kernel `trainable = False`. |
| Flax Linen | Kernel wrapped in `stop_gradient`. |

## Flax NNX

The frozen kernel is excluded from `nnx.state(model, nnx.Param)`, so the
optimizer never sees it:

```python
import jax, jax.numpy as jnp
from flax import nnx
from nmn.nnx import YatNMN

layer = YatNMN(in_features=128, out_features=64, lazy=True, rngs=nnx.Rngs(0))

# Only bias / alpha (and learnable epsilon, if enabled) are trainable —
# the frozen kernel is a FrozenParam, excluded from nnx.state(..., nnx.Param).
trainable = nnx.state(layer, nnx.Param)
print([p[0] for p in jax.tree_util.tree_leaves_with_path(trainable)])  # no 'kernel'

def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

x = jnp.ones((32, 128)); y = jnp.zeros((32, 64))
grads = nnx.grad(loss_fn)(layer, x, y)   # gradients only for alpha / bias
# Feed `grads` to your optimizer (e.g. nnx.Optimizer / optax) — the kernel,
# having no gradient, stays fixed across every training step.
```

## PyTorch

The kernel's `requires_grad` is set to `False`; α and bias remain trainable, so
a standard optimizer over `model.parameters()` skips the kernel:

```python
import torch
from nmn.torch import YatNMN

layer = YatNMN(in_features=128, out_features=64, lazy=True)  # or freeze_kernel=True
for name, p in layer.named_parameters():
    print(name, p.requires_grad)   # weight False, alpha True, bias True

opt = torch.optim.Adam(p for p in layer.parameters() if p.requires_grad)
x = torch.randn(32, 128)
loss = layer(x).pow(2).mean()
loss.backward()
opt.step()                          # only alpha / bias move
```

## Other Backends

The same flag works across Flax Linen, Keras, TensorFlow, and MLX:

```python
# Flax Linen / TensorFlow / MLX
from nmn.mlx import YatNMN
layer = YatNMN(features=64, lazy=True)        # or freeze_kernel=True

# Keras — units, not features
from nmn.keras import YatNMN
layer = YatNMN(units=64, lazy=True)
```

## See Also

- [YatNMN Reference](/docs/layers/yat-nmn) - Full constructor
- [CLI Reference](/docs/cli) - `nmn features` prints lazy-mode usage
- [Migration Guide](/docs/migration-guide) - Porting existing models
