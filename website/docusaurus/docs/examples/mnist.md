---
sidebar_position: 1
---

# MNIST Classification

A complete example of training an NMN classifier on MNIST.

## Full Code

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from nmn.nnx.nmn import YatNMN

# Model
class MNISTClassifier(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.layer1 = YatNMN(784, 512, constant_alpha=True, rngs=rngs)
        self.layer2 = YatNMN(512, 256, constant_alpha=True, rngs=rngs)
        self.layer3 = YatNMN(256, 10, constant_alpha=True, rngs=rngs)
    
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

# Training
model = MNISTClassifier(rngs=nnx.Rngs(42))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(m):
        logits = m(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# Evaluation
@nnx.jit
def eval_step(model, x, y):
    logits = model(x)
    return (logits.argmax(-1) == y).mean()
```

## Key Observations

- **No activation functions** between layers
- **No normalization layers** needed
- **Simple architecture** with just 3 YatNMN layers
