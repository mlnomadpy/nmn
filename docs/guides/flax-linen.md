# Flax Linen Guide

End-to-end walkthrough for using NMN with **Flax Linen** (the older, pure-functional Flax API).

- **Module path**: `nmn.linen`
- **Minimum versions**: Python 3.11+, JAX ≥ 0.9.1, Flax ≥ 0.12.5

> If you are starting a new project, prefer the [Flax NNX guide](flax-nnx.md). Use Linen only for legacy code or projects that rely on `nn.compact`-style modules.

---

## 1. Install

```bash
pip install "nmn[linen]"
```

Verify:

```python
import jax, jax.numpy as jnp
import flax.linen as nn
from nmn.linen import YatNMN

class M(nn.Module):
    @nn.compact
    def __call__(self, x):
        return YatNMN(features=64)(x)

m = M()
params = m.init(jax.random.PRNGKey(0), jnp.ones((32, 128)))
y = m.apply(params, jnp.ones((32, 128)))
print(y.shape)  # (32, 64)
```

---

## 2. Hello world

```python
import flax.linen as nn
from nmn.linen import YatNMN

class FC(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Before:
        # return nn.relu(nn.Dense(64)(x))

        # After:
        return YatNMN(features=64)(x)
```

No `nn.relu`. The Yat layer is non-linear by itself.

---

## 3. MNIST end-to-end

```python
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
from nmn.linen import YatNMN

class YatMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = YatNMN(features=256)(x)
        x = YatNMN(features=128)(x)
        return nn.Dense(10)(x)

model = YatMLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optax.adamw(3e-4),
)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

ds = tfds.load("mnist", split="train", as_supervised=True, batch_size=128)
for epoch in range(3):
    for x, y in tfds.as_numpy(ds):
        x = jnp.asarray(x, dtype=jnp.float32) / 255.0
        state, loss = train_step(state, x, jnp.asarray(y))
    print(f"epoch {epoch}: loss={loss:.4f}")
```

A runnable version (no tfds, uses torchvision for data) is at
[`src/nmn/linen/examples/mnist.py`](../../src/nmn/linen/examples/mnist.py):

```bash
PYTHONPATH=src python -m nmn.linen.examples.mnist \
    --epochs 3 --report .context/linen_mnist.json
```

**Measured baseline** (CPU JAX, fp32, batch=128, lr=3e-4, seed=0):

| Epoch | Train loss | Test loss | Test acc |
| ----- | ---------- | --------- | -------- |
| 0     | 0.8967     | 0.2652    | 91.79 %  |
| 1     | 0.2386     | 0.1973    | 93.79 %  |
| 2     | 0.1808     | 0.1538    | **95.28 %** |

3 epochs ≈ 1.3 s on CPU. Final accuracy matches the NNX run (95.39 %) within
seed noise, confirming the Linen and NNX backends are numerically aligned.

---

## 4. CNN

```python
from nmn.linen import YatConv2D, YatNMN
import flax.linen as nn
import jax.numpy as jnp

class YatCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = YatConv2D(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = YatConv2D(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = jnp.mean(x, axis=(1, 2))   # global average pool
        x = YatNMN(features=64)(x)
        return nn.Dense(10)(x)
```

---

## 5. Attention

```python
from nmn.linen import MultiHeadAttention
import flax.linen as nn

class Block(nn.Module):
    @nn.compact
    def __call__(self, x):
        return MultiHeadAttention(num_heads=8)(x)
```

`nmn.linen.MultiHeadAttention` mirrors `flax.linen.MultiHeadDotProductAttention` but with Yat-scored attention.

---

## 6. Saving & loading

```python
import flax
import pickle

# Save raw params
with open("params.pkl", "wb") as f:
    pickle.dump(flax.serialization.to_state_dict(state.params), f)

# Or use Orbax (preferred):
import orbax.checkpoint as ocp
ckpt = ocp.PyTreeCheckpointer()
ckpt.save("/tmp/yat_linen", state.params)
```

---

## 7. Lazy training (freeze kernel)

Pass `lazy=True` (alias: `freeze_kernel=True`) to freeze **only** the kernel.
Linen parameters are functional — there is no per-variable "trainable" flag — so
in lazy mode the kernel is read through `jax.lax.stop_gradient` inside
`__call__`. No gradient flows into the kernel, while `bias`, `alpha`, and the
learnable `epsilon` stay fully trainable. Default `lazy=False`, fully backward
compatible.

```python
import jax, jax.numpy as jnp
import flax.linen as nn
from nmn.linen import YatNMN

class FC(nn.Module):
    @nn.compact
    def __call__(self, x):
        return YatNMN(features=64, lazy=True)(x)   # kernel frozen via stop_gradient

m = FC()
params = m.init(jax.random.PRNGKey(0), jnp.ones((32, 128)))
grads = jax.grad(lambda p: m.apply(p, jnp.ones((32, 128))).sum())(params)
# The 'kernel' leaf receives zero gradient; bias / alpha / epsilon update.
```

Because the kernel still lives in the `params` pytree, its leaf simply gets a
zero gradient. For *true* exclusion from the optimizer state (so it is not even
carried), pair `lazy=True` with an `optax.multi_transform` that routes the
`kernel` leaf to `optax.set_to_zero()` — see the class docstring for the
pattern. Use lazy mode for a fixed random/anchored feature basis where only the
scale/shift/`epsilon` adapt.

> CLI: `nmn guide flax-linen` prints the quickstart; `nmn features` summarizes
> lazy mode and the MAY/RAY performer maps.

---

## 8. Differences vs NNX

| Aspect              | Linen                                            | NNX                                                  |
| ------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| Module style         | `@nn.compact` or `setup()`                       | `__init__` + `__call__` (Pythonic)                   |
| Params               | External `params` dict passed to `apply`         | Stored on the module; access with `nnx.state(m)`     |
| RNG handling         | Pass `rngs={...}` to `apply`                     | `nnx.Rngs(0)` on module construction                  |
| Mutation             | Functional only                                  | Mutable; supports `model.train()` / `model.eval()`   |

If you're new, learn NNX. If you're maintaining a Linen codebase, `nmn.linen` slots in transparently.

---

## 9. Troubleshooting

| Symptom                                  | Likely cause                                            | Fix                                                                |
| ---------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
| `NaN` loss after one JIT compile          | `epsilon` too small for input scale                     | `YatNMN(features=..., epsilon=1e-3)`                                |
| `KeyError: 'YatNMN_0'` on `apply`         | Initialized with old params after architecture change   | Reinit `params = model.init(...)`                                   |
| Slow first step                            | JIT compile cost                                        | Reuse the JIT-compiled function across steps                         |
| `TypeError: 'YatNMN' object is not callable` | Old Flax version                                       | `pip install --upgrade flax`                                         |

---

## 10. Next steps

- [Flax NNX guide](flax-nnx.md) — recommended for new code
- [Architecture & theory](../architecture.md)
- [Migration cheat sheet](../migration.md)
- [EXAMPLES.md](../../EXAMPLES.md)
