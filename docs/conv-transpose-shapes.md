# Transposed-convolution output shapes

NMN defines one canonical spatial-size contract for transposed convolution.
For each spatial axis, let `L` be the input size, `k` the kernel size, `s` the
stride, `d` the kernel dilation, `o` the output padding, and
`e = d * (k - 1) + 1` the effective kernel size. Then:

```text
VALID: (L - 1) * s + e + o
SAME:  L * s + o
```

`k`, `s`, and `d` must be positive. `o` must satisfy `0 <= o < s`. Multi-axis
1D/2D/3D layers apply these equations independently to every axis. Output
padding extends only the high side; it does not add symmetric input padding.

## Backward-compatible selection

Frameworks historically disagreed when `s > e`: PyTorch and MLX used the
canonical `VALID` result, while JAX, Keras, and TensorFlow inferred
`L * s + max(e - s, 0)` when output padding was omitted. NMN preserves those
legacy defaults so loading an existing model cannot silently change its shape.

| Backend | Select the canonical contract |
| --- | --- |
| PyTorch | Existing numeric `padding=0` and `output_padding=o` (`VALID`) |
| Flax NNX | Pass `output_padding=o` explicitly; omission keeps JAX sizing |
| Flax Linen | Pass `output_padding=o` explicitly; omission keeps JAX sizing |
| Keras 3 | Pass `output_shape_mode="nmn"`; omitted output padding means zero |
| TensorFlow | Pass `output_padding=o` explicitly; omission keeps TF sizing |
| MLX | Existing `padding="valid"`/`"same"` and `output_padding=o` |

Keras keeps `output_shape_mode="framework"` as its default for configuration
and saved-model compatibility. Its `"nmn"` mode is serialized by `get_config`.
TensorFlow, Linen, and NNX added only optional keyword arguments, so existing
constructor calls and implicit shapes are unchanged.

PyTorch exposes numeric rather than string padding. Its general native formula
is `(L - 1) * s - 2 * p + e + o`; `p=0` is canonical `VALID`. A numeric padding
configuration is canonical `SAME` only when that equation equals `L * s + o`.
Use a framework with asymmetric string padding when no such integer `p` exists.
PyTorch can also accept `o >= s` when dilation is larger than the stride. NMN
retains that native behavior for backward compatibility, but those configurations
are outside the canonical contract and are not portable; use `0 <= o < s` for
cross-framework models.

## Example

For `L=3`, `k=2`, `s=3`, `d=1`, and `o=0`, canonical `VALID` output length is
`(3 - 1) * 3 + 2 = 8`. The legacy JAX/Keras/TensorFlow inferred length is 9.
