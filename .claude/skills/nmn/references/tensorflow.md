# TensorFlow

Install with `python -m pip install "nmn[tf]"`.

```python
import tensorflow as tf
from nmn.tf import YatNMN

layer = YatNMN(features=64, learnable_epsilon=True)
x = tf.ones((8, 128))
with tf.GradientTape() as tape:
    tape.watch(x)
    loss = tf.reduce_sum(layer(x))
grads = tape.gradient(loss, [x] + layer.trainable_variables)
```

Native TF dense, embedding, and convolution modules build lazily from input
shape. `YatDense` aliases `YatNMN`. Convolutions are channels-last; native TF
does not expose Keras `data_format`. Grouped convolution uses explicit
split/apply/concat so CPU forward and gradients remain portable.

## Attention

```python
from nmn.tf import MultiHeadYatAttention

attn = MultiHeadYatAttention(embed_dim=128, num_heads=8)
y = attn(x, mask=tf.ones((32, 32), dtype=tf.bool), training=True)
```

Use batch-major `[B,S,E]` module inputs. Boolean `True` means allowed. Fully
masked rows return exact-zero output after projection.

## SavedModel

NMN native modules expose explicit export methods with stable signatures:

```python
layer.export("saved_dense", tf.TensorSpec([None, 128], tf.float32))

from nmn.tf import YatEmbed
embed = YatEmbed(num_embeddings=32000, features=128)
embed.export(
    "saved_embed",
    tf.TensorSpec([None, None], tf.int32),
    attend_signature=tf.TensorSpec([None, 128], tf.float32),
)
```

Convolution modules also accept one `input_signature`. Attention export accepts
query plus optional key/value/mask signatures and creates self/cross serving
paths as applicable. Load with `tf.saved_model.load` and invoke the named
signature; overwrite behavior is explicit.

## Precision

FP16/BF16 score math promotes vulnerable reductions, saturates finite output
overflow, and preserves NaNs. Learned epsilon uses representable positive state.
Test eager and `tf.function`; note that some dilated convolution gradients are
limited by TensorFlow CPU itself and may require accelerator validation.
