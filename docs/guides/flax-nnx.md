# Flax NNX Guide

End-to-end walkthrough for using NMN with **Flax NNX** (the recommended JAX entry point — explicit state, eager-friendly).

- **Module path**: `nmn.nnx`
- **Minimum versions**: Python 3.11+, JAX ≥ 0.9.1, Flax ≥ 0.12.5

---

## 1. Install

```bash
pip install "nmn[nnx]"
```

For GPU/TPU JAX, install JAX *first* with the matching CUDA/TPU wheel (see [JAX install docs](https://jax.readthedocs.io/en/latest/installation.html)), then:

```bash
pip install "nmn[nnx]" --no-deps   # avoid clobbering JAX
pip install "flax>=0.12.5"
```

Verify:

```python
import jax, jax.numpy as jnp, nmn
from flax import nnx
from nmn.nnx import YatNMN

print(nmn.__version__, jax.__version__)
layer = YatNMN(in_features=128, out_features=64, rngs=nnx.Rngs(0))
print(layer(jnp.ones((32, 128))).shape)  # (32, 64)
```

---

## 2. Hello world

```python
import jax.numpy as jnp
from flax import nnx
from nmn.nnx import YatNMN

# Before:
# fc = nnx.Linear(128, 64, rngs=nnx.Rngs(0))
# y = nnx.relu(fc(x))

# After:
fc = YatNMN(in_features=128, out_features=64, rngs=nnx.Rngs(0))
x = jnp.ones((32, 128))
y = fc(x)
print(y.shape)  # (32, 64)
```

No `nnx.relu` needed.

---

## 3. MNIST end-to-end

```python
import jax, jax.numpy as jnp
import optax
from flax import nnx
from nmn.nnx import YatNMN
import tensorflow_datasets as tfds  # pip install tensorflow-datasets

class YatMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.fc1 = YatNMN(in_features=28*28, out_features=256, rngs=rngs)
        self.fc2 = YatNMN(in_features=256,   out_features=128, rngs=rngs)
        self.out = nnx.Linear(128, 10, rngs=rngs)
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.out(self.fc2(self.fc1(x)))

model = YatMLP(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adamw(3e-4))

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(m):
        logits = m(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

ds = tfds.load("mnist", split="train", as_supervised=True, batch_size=128)
for epoch in range(3):
    for x, y in tfds.as_numpy(ds):
        x = jnp.asarray(x, dtype=jnp.float32) / 255.0
        loss = train_step(model, optimizer, x, jnp.asarray(y))
    print(f"epoch {epoch}: loss={loss:.4f}")
```

A complete runnable version (using torchvision to avoid the tfds dependency)
lives at [`src/nmn/nnx/examples/vision/mnist.py`](../../src/nmn/nnx/examples/vision/mnist.py):

```bash
PYTHONPATH=src python -m nmn.nnx.examples.vision.mnist \
    --epochs 3 --report .context/nnx_mnist.json
```

**Measured baseline** (CPU JAX, fp32, batch=128, lr=3e-4, seed=0):

| Epoch | Train loss | Test loss | Test acc |
| ----- | ---------- | --------- | -------- |
| 0     | 0.8841     | 0.2567    | 92.06 %  |
| 1     | 0.2249     | 0.1824    | 94.38 %  |
| 2     | 0.1710     | 0.1477    | **95.39 %** |

3 epochs ≈ 1.7 s on CPU — JAX's JIT is ~3× faster than PyTorch eager here on
the same model. See the [cross-framework benchmark blog post](https://github.com/azettaai/nmn/blob/master/website/blog-pages/20-cross-framework-mnist.html) for a side-by-side.

---

## 4. Convolutional model

```python
from nmn.nnx import YatConv, YatNMN
from flax import nnx
import jax.numpy as jnp

class YatCNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.c1 = YatConv(in_features=3,  out_features=32, kernel_size=(3, 3), rngs=rngs)
        self.c2 = YatConv(in_features=32, out_features=64, kernel_size=(3, 3), rngs=rngs)
        self.fc = YatNMN(in_features=64,  out_features=10, rngs=rngs)
    def __call__(self, x):
        x = nnx.max_pool(self.c1(x), window_shape=(2, 2), strides=(2, 2))
        x = nnx.max_pool(self.c2(x), window_shape=(2, 2), strides=(2, 2))
        x = jnp.mean(x, axis=(1, 2))   # global average pool
        return self.fc(x)
```

For a real ResNet-50 trained on TPU, see [`src/nmn/nnx/examples/vision/aether_resnet50_tpu.py`](../../src/nmn/nnx/examples/vision/aether_resnet50_tpu.py).

---

## 5. Attention — three variants

### a) Standard Yat attention

```python
from nmn.nnx import MultiHeadAttention
from flax import nnx
import jax.numpy as jnp

attn = MultiHeadAttention(num_heads=8, in_features=512, rngs=nnx.Rngs(0))
x = jnp.ones((4, 16, 512))
print(attn(x).shape)  # (4, 16, 512)
```

### b) RoPE (Rotary Position Embeddings)

```python
from nmn.nnx import RotaryYatAttention

attn = RotaryYatAttention(num_heads=8, in_features=512, rngs=nnx.Rngs(0))
y = attn(x)
```

### c) Spherical YAT-Performer — O(n) linear attention

```python
attn = MultiHeadAttention(num_heads=8, in_features=512,
                          use_performer=True, rngs=nnx.Rngs(0))
y = attn(x)
```

### d) Fused / Pallas flash attention

For long-context training on TPU/GPU, NMN ships a fused yat-attention kernel (Pallas). See [`src/nmn/nnx/`](../../src/nmn/nnx/) for the kernel and the language example
[`src/nmn/nnx/examples/language/m3za.py`](../../src/nmn/nnx/examples/language/m3za.py) for end-to-end usage. Required: TPU runtime *or* a recent CUDA build of JAX.

---

## 6. Embeddings

```python
from nmn.nnx import Embed
from flax import nnx

embed = Embed(num_embeddings=10_000, features=128,
              constant_alpha=True, rngs=nnx.Rngs(0))

token_ids = jnp.array([[1, 2, 3, 4, 5]])
h = embed(token_ids)            # (1, 5, 128)
scores = embed.attend(h[0, -1]) # (10_000,) — retrieval scoring
```

---

## 7. Saving & loading

Use [Orbax checkpointing](https://orbax.readthedocs.io/), the JAX-recommended pattern:

```python
import orbax.checkpoint as ocp
from flax import nnx

state = nnx.state(model)
ckpt = ocp.PyTreeCheckpointer()
ckpt.save("/tmp/yat_mlp", state)

# Later:
restored = ckpt.restore("/tmp/yat_mlp")
nnx.update(model, restored)
```

---

## 8. Sharding / distributed training

`YatNMN` and `YatConv` are plain `nnx.Module`s. They compose with:

- `jax.sharding.PartitionSpec` for tensor parallelism
- `jax.pmap` for data parallelism
- `nnx.shard_map` / `jax.experimental.shard_map` for SPMD code

See the TPU ResNet example for a worked pattern.

---

## 9. Troubleshooting

| Symptom                                  | Likely cause                                            | Fix                                                                |
| ---------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
| `NaN` after first JIT compile            | `epsilon` too small relative to input scale             | `YatNMN(..., epsilon=1e-3)`                                        |
| `XlaRuntimeError` on TPU                 | Pallas kernel ran on a CPU/GPU runtime                  | Use standard `MultiHeadAttention` on CPU; pallas only on TPU/GPU   |
| Slow first step, fast after              | Normal — JIT compile cost                               | Reuse the JIT-compiled function across steps                       |
| Output range collapses to zero           | Inputs and weights nearly identical → numerator ≈ 0     | Add input normalization *before* first Yat layer                   |
| `flax` import fails                      | Mismatched JAX/Flax versions                            | `pip install "jax==0.9.1" "jaxlib==0.9.1" "flax==0.12.5"`         |

---

## 10. Next steps

- [Architecture & theory](../architecture.md)
- [Flax Linen guide](flax-linen.md) — for legacy Linen codebases
- [Full NNX examples in repo](../../src/nmn/nnx/examples/)
- [EXAMPLES.md](../../EXAMPLES.md)
