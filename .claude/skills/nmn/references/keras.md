# Keras 3

Install Keras plus one backend, and set the backend before importing Keras:

```bash
python -m pip install "nmn[keras]" jax
export KERAS_BACKEND=jax        # or tensorflow / torch
```

```python
import keras
from nmn.keras import YatNMN

model = keras.Sequential([
    keras.Input((128,)),
    YatNMN(units=64, learnable_epsilon=True),
    YatNMN(units=10),
])
model.compile(optimizer="adam", loss="mse")
```

Keras uses `units`, not `features` or `out_features`. `YatDense` is an alias.
Do not add an activation merely to make the layer nonlinear.

## Convolution

`YatConv1D/2D/3D` and transpose variants follow Keras channels-last defaults
and accept `data_format`. On the TensorFlow CPU backend, channels-first public
inputs are internally converted to channels-last for the complete YAT path and
converted back, including grouped/dilated/causal and transpose output-padding
cases.

Shared convolution banks use `tie_kernel_bank=True` with a compatible
`kernel_bank_id`/capacity or an explicit bank object. Registry compatibility
includes backend and dtype policy. Clone/save/load should preserve one tracked
shared variable without duplicate optimizer slots.

## Attention

```python
from nmn.keras import MultiHeadYatAttention

attn = MultiHeadYatAttention(embed_dim=128, num_heads=8)
y = attn(x, attention_mask=mask, training=True)
```

A rank-2 `mask` is always a Keras query sequence mask shaped `[batch,
query_length]`; in self-attention it also masks the corresponding keys. Use
`attention_mask` for pairwise masks, including rank-2 `[query_length,
key_length]` masks. Legacy rank-3 and rank-4 pairwise masks passed through
`mask` remain supported. The two masks are combined when both are supplied.
Fully masked query rows yield exact zeros even when output projection bias is
enabled.

## Dtype and serialization

Use Keras dtype policies (`float32`, `mixed_float16`, etc.). Vulnerable YAT
reductions use a safe FP32 boundary and cast/saturate back while preserving
NaNs. On JAX, declared float64 learned-epsilon state requires x64 enabled;
otherwise construction rejects rather than silently storing FP32/inf.

Round-trip with `model.save("model.keras")` and
`keras.models.load_model(...)`; NMN layers are registered serializables. With
the TensorFlow backend, use `model.export(...)` for SavedModel.
