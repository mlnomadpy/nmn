# TensorFlow Guide

End-to-end walkthrough for using NMN with native **TensorFlow** (not via Keras).

- **Module path**: `nmn.tf`
- **Minimum versions**: Python 3.10+, TensorFlow ≥ 2.10

> If you're using Keras 3 (multi-backend), follow the [Keras guide](keras.md) instead. Use `nmn.tf` only when you need the lower-level `tf.Module` API.

---

## 1. Install

```bash
pip install "nmn[tf]"
```

Verify:

```python
import tensorflow as tf
from nmn.tf import YatNMN

print(tf.__version__)
layer = YatNMN(features=64)
print(layer(tf.zeros((32, 128))).shape)  # (32, 64)
```

---

## 2. Hello world

```python
import tensorflow as tf
from nmn.tf import YatNMN

# Before:
# y = tf.nn.relu(tf.keras.layers.Dense(64)(x))

# After:
fc = YatNMN(features=64)
y = fc(tf.zeros((32, 128)))
print(y.shape)  # (32, 64)
```

---

## 3. MNIST end-to-end with `tf.GradientTape`

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from nmn.tf import YatNMN

class YatMLP(tf.Module):
    def __init__(self):
        self.fc1 = YatNMN(features=256)
        self.fc2 = YatNMN(features=128)
        self.out = tf.keras.layers.Dense(10)
    def __call__(self, x):
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        return self.out(self.fc2(self.fc1(x)))

ds, info = tfds.load("mnist", split="train", as_supervised=True, with_info=True)
ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).batch(128)

model = YatMLP()
opt = tf.keras.optimizers.AdamW(3e-4)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)
        )
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(3):
    for x, y in ds:
        loss = train_step(x, y)
    print(f"epoch {epoch}: loss={loss.numpy():.4f}")
```

---

## 4. Convolutional model

```python
from nmn.tf import YatConv2D, YatNMN
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input((32, 32, 3)),
    YatConv2D(filters=32, kernel_size=3, padding="same"),
    tf.keras.layers.MaxPooling2D(),
    YatConv2D(filters=64, kernel_size=3, padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.GlobalAveragePooling2D(),
    YatNMN(features=64),
    tf.keras.layers.Dense(10),
])
```

---

## 5. Attention

```python
from nmn.tf import MultiHeadYatAttention
import tensorflow as tf

attn = MultiHeadYatAttention(num_heads=8, key_dim=64)
x = tf.ones((4, 16, 512))
y = attn(x, x)
print(y.shape)
```

---

## 6. Saving & loading

### Checkpoints

```python
ckpt = tf.train.Checkpoint(model=model, optimizer=opt)
ckpt.save("/tmp/yat_mlp")

# Later:
ckpt.restore(tf.train.latest_checkpoint("/tmp"))
```

### SavedModel

```python
tf.saved_model.save(model, "yat_savedmodel")

restored = tf.saved_model.load("yat_savedmodel")
```

### TFLite

```python
converter = tf.lite.TFLiteConverter.from_saved_model("yat_savedmodel")
tflite_bytes = converter.convert()
open("yat.tflite", "wb").write(tflite_bytes)
```

Yat layers only use standard TF ops (mul, sum, sub, div, square). TFLite conversion has worked in our smoke tests but is **not currently covered by CI** — please report results.

---

## 7. Distributed training

`YatNMN` and `YatConv*` are `tf.Module`s with standard `tf.Variable`s. They work with:

- `tf.distribute.MirroredStrategy` (multi-GPU)
- `tf.distribute.TPUStrategy` (TPU)
- `tf.distribute.MultiWorkerMirroredStrategy` (multi-host)

Standard pattern:

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = YatMLP()
    opt = tf.keras.optimizers.AdamW(3e-4)
```

---

## 8. Troubleshooting

| Symptom                                | Likely cause                                  | Fix                                                                          |
| -------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------- |
| `NaN` in early steps                    | `epsilon` too small                           | `YatNMN(features=..., epsilon=1e-3)`                                          |
| `tf.function` retraces every step       | Python-int args mixed with tensors            | Wrap input shapes; ensure `train_step` signature is consistent               |
| Wrong shape on attention output         | `key_dim` missing                             | Pass `key_dim=embed_dim // num_heads` to `MultiHeadYatAttention`              |
| `SavedModel` reload fails                | Custom layer class not on import path         | Ensure `nmn.tf` is importable in the loading process                          |

---

## 9. Next steps

- [Architecture & theory](../architecture.md)
- [Keras guide](keras.md) — usually a nicer API for new TF code
- [EXAMPLES.md](../../EXAMPLES.md)
