# Keras Guide

End-to-end walkthrough for using NMN with **Keras 3** (backend-agnostic: works on JAX, TensorFlow, or PyTorch backends).

- **Module path**: `nmn.keras`
- **Minimum versions**: Python 3.10+, Keras 3.x, TensorFlow ≥ 2.10

---

## 1. Install

```bash
pip install "nmn[keras]"
```

Verify:

```python
import keras
from nmn.keras import YatNMN

print(keras.__version__)
layer = YatNMN(features=64)
print(layer(keras.ops.ones((32, 128))).shape)  # (32, 64)
```

To switch Keras backend, set the env var **before** importing keras:

```bash
export KERAS_BACKEND=jax      # or tensorflow, torch
```

---

## 2. Hello world

```python
import keras
from nmn.keras import YatNMN

# Before:
# fc = keras.layers.Dense(64, activation="relu")

# After:
fc = YatNMN(features=64)

x = keras.ops.ones((32, 128))
print(fc(x).shape)  # (32, 64)
```

---

## 3. MNIST end-to-end

```python
import keras
from nmn.keras import YatNMN

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

model = keras.Sequential([
    keras.layers.Input((28, 28)),
    keras.layers.Flatten(),
    YatNMN(features=256),
    YatNMN(features=128),
    keras.layers.Dense(10),
])

model.compile(
    optimizer=keras.optimizers.AdamW(3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=128, epochs=3,
          validation_data=(x_test, y_test))
```

Expect ~97–98% test accuracy.

---

## 4. CNN

```python
from nmn.keras import YatConv2D, YatNMN
import keras

model = keras.Sequential([
    keras.layers.Input((32, 32, 3)),
    YatConv2D(filters=32, kernel_size=3, padding="same"),
    keras.layers.MaxPooling2D(),
    YatConv2D(filters=64, kernel_size=3, padding="same"),
    keras.layers.MaxPooling2D(),
    keras.layers.GlobalAveragePooling2D(),
    YatNMN(features=64),
    keras.layers.Dense(10),
])
```

---

## 5. Attention

```python
from nmn.keras import MultiHeadYatAttention
import keras

attn = MultiHeadYatAttention(num_heads=8, key_dim=64)
x = keras.ops.ones((4, 16, 512))
y = attn(x, x)
print(y.shape)  # (4, 16, 512)
```

Drop in as a replacement for `keras.layers.MultiHeadAttention`.

---

## 6. Saving & loading

```python
model.save("yat_model.keras")

# Custom layers need to be registered for deserialization:
import keras
from nmn.keras import YatNMN, YatConv2D, MultiHeadYatAttention

restored = keras.models.load_model(
    "yat_model.keras",
    custom_objects={
        "YatNMN": YatNMN,
        "YatConv2D": YatConv2D,
        "MultiHeadYatAttention": MultiHeadYatAttention,
    },
)
```

---

## 7. Export & deployment

- **TensorFlow SavedModel**: `model.export("yat_model_savedmodel")`
- **TFLite**: convert from SavedModel with `tf.lite.TFLiteConverter.from_saved_model(...)` — Yat layers use only standard TF ops and should convert cleanly. Not currently in CI; please report results.
- **ONNX**: `tf2onnx` works for SavedModel exports of Yat models in our smoke tests. Not yet in CI.

---

## 8. Troubleshooting

| Symptom                                | Likely cause                                       | Fix                                                            |
| -------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| `NaN` in early training                 | `epsilon` too small                                | `YatNMN(features=..., epsilon=1e-3)`                            |
| `ValueError` on model load              | Custom layers not registered                       | Pass `custom_objects=` (see § 6)                                |
| Slow training on CPU                    | Keras backend defaulted to TF and op fallback      | Switch to JAX backend: `KERAS_BACKEND=jax`                      |
| Shape mismatch in attention block       | Forgot `key_dim`                                   | `MultiHeadYatAttention(num_heads=8, key_dim=embed_dim // 8)`    |

---

## 9. Next steps

- [Architecture & theory](../architecture.md)
- [Migration cheat sheet](../migration.md)
- [EXAMPLES.md](../../EXAMPLES.md) — more Keras snippets
