# Cross-framework contract

## Public families

Every backend provides dense YAT, 1D/2D/3D convolution and transpose
convolution, embedding, attention, squashers, and MAY/RAY feature maps. Names
and construction differ; use the backend reference rather than guessing.

## State and training

- `lazy=True` and `freeze_kernel=True` freeze only the kernel.
- `learnable_epsilon=True` stores a raw parameter and evaluates its softplus.
- `weight_normalized=True` uses normalized kernel geometry where supported.
- DropConnect is stochastic only during training/non-deterministic calls. Use a
  rate in `[0, 1)` and pass the framework's training/deterministic flag.
- Tied banks share only compatible geometry, device, backend, and dtype policy.
  Preallocate capacity when architecture width is known.

## Data layouts

- Dense operates on the final feature dimension.
- JAX/Linen/NNX, Keras, TF, and MLX convolution examples are channels-last.
- Native PyTorch convolution is channels-first.
- Attention functional APIs use `[..., Q, H, D]`; attention modules use
  `[B, S, E]` unless their backend reference says otherwise.

## Masks

- Boolean `True` means allowed/visible.
- Broadcast masks to score shape rather than indexing fixed mask ranks.
- Partially masked rows normalize over valid keys only.
- Fully masked rows yield exact-zero weights and outputs with finite gradients.
- Performer paths accept key-padding-style masks, not arbitrary query-dependent
  masks, unless the selected function documents broader support.

## Precision

- Form YAT dots, squared norms, distances, and ratios using the backend's safe
  compute path for FP16/BF16.
- Preserve genuine NaNs. Saturation is for finite overflow, not corruption.
- Keep learned epsilon representable and positive; use FP32 epsilon state for
  low-precision training where implemented.
- Verify forward and VJP parity against synchronized FP32 parameters. Use
  realistic collision, large-distance, multi-output, and aggregate-reduction
  cases—not only random small tensors.

## Porting checklist

1. Map constructor names and tensor layout.
2. Build lazy layers before assigning weights.
3. Transpose kernels explicitly between framework layouts.
4. Match bias, alpha, epsilon, spherical/normalized mode, and score
   normalization.
5. Match mask convention and output projection.
6. Compare output and every gradient with dtype-appropriate tolerances.
7. Round-trip the native checkpoint/export format.
