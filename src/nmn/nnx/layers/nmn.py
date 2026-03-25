from __future__ import annotations

import threading
import typing as tp

import jax
import jax.numpy as jnp
from jax import lax

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Dtype,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  PromoteDtypeFn,
)

Array = jax.Array
Axis = int
Size = int


default_kernel_init = initializers.xavier_normal()
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()

class YatNMN(Module):
  """A YAT linear transformation applied over the last dimension of the input.

  The YAT  operation computes:
    y = (x · W)² / (||x - W||² + ε)

  With optional scaling:
    y = y * alpha (learnable) or y = y * sqrt(2) (constant)

  Example usage::

    >>> from flax import nnx
    >>> from nmn.nnx.nmn import YatNMN
    >>> import jax.numpy as jnp

    >>> # Learnable alpha (default)
    >>> layer = YatNMN(in_features=3, out_features=4, rngs=nnx.Rngs(0))
    
    >>> # Constant alpha = sqrt(2) (recommended default)
    >>> layer = YatNMN(in_features=3, out_features=4, constant_alpha=True, rngs=nnx.Rngs(0))
    
    >>> # Custom constant alpha value
    >>> layer = YatNMN(in_features=3, out_features=4, constant_alpha=1.5, rngs=nnx.Rngs(0))
    
    >>> # No alpha scaling
    >>> layer = YatNMN(in_features=3, out_features=4, use_alpha=False, rngs=nnx.Rngs(0))

  Args:
    in_features: the number of input features.
    out_features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    use_alpha: whether to use alpha scaling (default: True). Ignored if constant_alpha is set.
    constant_alpha: if True, use sqrt(2) as constant alpha. If a float, use that value.
      If None (default), use learnable alpha when use_alpha=True.
    use_dropconnect: whether to use DropConnect (default: False).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    alpha_init: initializer function for the learnable alpha (only used if constant_alpha is None).
    dot_general: dot product function.
    promote_dtype: function to promote the dtype of the arrays to the desired
      dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
      and a ``dtype`` keyword argument, and return a tuple of arrays with the
      promoted dtype.
    epsilon: A small float added to the denominator to prevent division by zero.
    drop_rate: dropout rate for DropConnect (default: 0.0).
    weight_normalized: if True, normalize each neuron (column) of the kernel to
      have norm 1. This optimization avoids recomputing kernel norms in YAT
      distance calculation since they are guaranteed to be 1.0.
    tie_kernel_bank: if True, reuse a shared kernel bank across compatible
      YatNMN instances and slice the first ``out_features`` neurons.
    kernel_bank_size: total neurons in the shared bank. If None, defaults to
      ``out_features`` for the creating instance.
    kernel_bank_id: optional bank namespace to control sharing groups.
    rngs: rng key.
  """

  __data__ = ('kernel', 'bias', 'alpha', 'dropconnect_key')
  _KERNEL_BANKS: dict[tuple[tp.Any, ...], nnx.Param] = {}
  _KERNEL_BANKS_LOCK = threading.Lock()

  # Default constant alpha value (sqrt(2))
  DEFAULT_CONSTANT_ALPHA = jnp.sqrt(2.0)  # jnp.sqrt(2.0)

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    use_alpha: bool = True,
    constant_alpha: tp.Optional[tp.Union[bool, float]] = None,
    positive_init: bool = False,
    use_dropconnect: bool = False,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    alpha_init: Initializer = default_alpha_init,
    dot_general: DotGeneralT = lax.dot_general,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    epsilon: float = 1e-5,
    spherical: bool = False,
    drop_rate: float = 0.0,
    weight_normalized: bool = False,
    tie_kernel_bank: bool = False,
    kernel_bank_size: tp.Optional[int] = None,
    kernel_bank_id: str = 'default',
    rngs: rnglib.Rngs,
  ):

    self._tie_kernel_bank = tie_kernel_bank
    self._kernel_slice = slice(None)
    self.kernel_shape = (in_features, out_features)

    if tie_kernel_bank:
      # Auto-calculate bank size: use specified size or default to current layer's out_features
      # Bank will auto-expand if a later layer needs more features
      bank_out_features = kernel_bank_size or out_features

      bank_shape = (in_features, bank_out_features)
      bank_key = (
        kernel_bank_id,
        in_features,
        param_dtype,
        kernel_init,
        positive_init,
      )

      with YatNMN._KERNEL_BANKS_LOCK:
        shared_kernel = YatNMN._KERNEL_BANKS.get(bank_key)
        if shared_kernel is None:
          # First layer using this bank: create with auto-sized dimensions
          kernel_key = rngs.params()
          kernel_val = kernel_init(kernel_key, bank_shape, param_dtype)
          if positive_init:
            kernel_val = jnp.abs(kernel_val)
          shared_kernel = nnx.Param(kernel_val)
          YatNMN._KERNEL_BANKS[bank_key] = shared_kernel
        else:
          # Bank exists: auto-expand if needed
          existing_shape = shared_kernel.value.shape
          existing_bank_size = existing_shape[-1]

          if bank_out_features > existing_bank_size:
            # Auto-expand bank to accommodate larger layer
            import logging as _logging
            _logging.getLogger(__name__).info(
              "Auto-expanding kernel bank '%s': %d -> %d neurons",
              kernel_bank_id, existing_bank_size, bank_out_features
            )
            new_shape = (in_features, bank_out_features)
            old_kernel = shared_kernel.value
            # Pad with random initialization for new neurons
            kernel_key = rngs.params()
            new_kernel_val = kernel_init(kernel_key, new_shape, param_dtype)
            if positive_init:
              new_kernel_val = jnp.abs(new_kernel_val)
            # Copy old values, new neurons already initialized
            new_kernel_val = new_kernel_val.at[:, :existing_bank_size].set(old_kernel)
            shared_kernel.value = new_kernel_val
          # elif bank_out_features < existing_bank_size: bank is larger, slice used below

      self.kernel = shared_kernel
      self._kernel_slice = slice(0, out_features)
    else:
      kernel_key = rngs.params()
      kernel_val = kernel_init(kernel_key, (in_features, out_features), param_dtype)
      if positive_init:
        kernel_val = jnp.abs(kernel_val)
      self.kernel = nnx.Param(kernel_val)
    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
    else:
      self.bias = None

    # Handle alpha configuration
    # Priority: constant_alpha > use_alpha
    # 
    # Options:
    #   1. constant_alpha=True -> use sqrt(2) as constant
    #   2. constant_alpha=<float> -> use that value as constant
    #   3. use_alpha=True (default) -> learnable alpha parameter
    #   4. use_alpha=False -> no alpha scaling
    
    self.alpha: nnx.Param[jax.Array] | None
    
    if constant_alpha is not None:
      # Use constant alpha (no learnable parameter)
      if constant_alpha is True:
        self._constant_alpha_value = self.DEFAULT_CONSTANT_ALPHA
      else:
        self._constant_alpha_value = float(constant_alpha)
      self.alpha = None
      use_alpha = True  # Alpha scaling is enabled (but constant)
    else:
      self._constant_alpha_value = None
      if use_alpha:
        # Use learnable alpha
        alpha_key = rngs.params()
        self.alpha = nnx.Param(alpha_init(alpha_key, (1,), param_dtype))
      else:
        # No alpha scaling
        self.alpha = None
    
    self.use_alpha = use_alpha

    self.in_features = in_features
    self.out_features = out_features
    self.use_bias = use_bias
    self.constant_alpha = constant_alpha
    self.use_dropconnect = use_dropconnect
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.dot_general = dot_general
    self.promote_dtype = promote_dtype
    if epsilon <= 0:
      raise ValueError(f"epsilon must be positive, got {epsilon}")
    self.epsilon = epsilon
    self.spherical = spherical
    self.drop_rate = drop_rate
    self.weight_normalized = weight_normalized
    self.tie_kernel_bank = tie_kernel_bank
    self.kernel_bank_size = kernel_bank_size
    self.kernel_bank_id = kernel_bank_id

    # Normalize kernel if requested: normalize each neuron (column) to have norm 1
    if self.weight_normalized:
      kernel_val = self.kernel[...]
      kernel_norm = jnp.sqrt(jnp.sum(kernel_val**2, axis=0, keepdims=True))
      self.kernel.value = kernel_val / (kernel_norm + 1e-8)

    if use_dropconnect:
      self.dropconnect_key = rngs.params()
    else:
      self.dropconnect_key = None

  def __call__(self, inputs: Array, *, deterministic: bool = False) -> Array:
    """Applies a YAT linear transformation to the inputs along the last dimension.

    Computes: y = (x · W)² / (||x - W||² + ε), with optional alpha scaling.

    Args:
      inputs: The nd-array to be transformed.
      deterministic: If true, DropConnect is not applied (e.g., during inference).

    Returns:
      The transformed input.
    """
    kernel = self.kernel[...]
    if self._tie_kernel_bank:
      kernel = kernel[:, self._kernel_slice]
    bias = self.bias[...] if self.bias is not None else None
    
    # Get alpha value (either learnable or constant)
    if self._constant_alpha_value is not None:
      alpha = jnp.array(self._constant_alpha_value, dtype=self.param_dtype)
    elif self.alpha is not None:
      alpha = self.alpha[...]
    else:
      alpha = None

    if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
      keep_prob = 1.0 - self.drop_rate
      mask = jax.random.bernoulli(self.dropconnect_key, p=keep_prob, shape=kernel.shape)
      kernel = (kernel * mask) / keep_prob

    # Normalize kernel if weight normalization is enabled
    if self.weight_normalized:
      kernel = kernel / (jnp.sqrt(jnp.sum(kernel**2, axis=0, keepdims=True)) + 1e-8)

    if self.spherical:
       inputs = inputs / jnp.linalg.norm(inputs, axis=-1, keepdims=True)
       kernel = kernel / jnp.linalg.norm(kernel, axis=0, keepdims=True)

    inputs, kernel, bias, alpha = self.promote_dtype(
      (inputs, kernel, bias, alpha), dtype=self.dtype
    )
    
    # Compute dot product: x · W
    y = self.dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )

    if self.spherical:
      # Spherical YAT: inputs and kernel are normalized
      # distances = ||x||² + ||W||² - 2(x · W) = 1 + 1 - 2(x · W) = 2 - 2(x · W)
      # Clamp to zero: bf16 cancellation can make distance negative when x ≈ W
      distances = jnp.maximum(2 - 2 * y, 0.0)
    else:
      # Compute squared Euclidean distance: ||x||² + ||W||² - 2(x · W)
      inputs_squared_sum = jnp.sum(inputs**2, axis=-1, keepdims=True)

      # Optimization: if weights are normalized, ||W||² = 1 for each neuron
      if self.weight_normalized:
        kernel_squared_sum = jnp.ones((1, kernel.shape[-1]), dtype=kernel.dtype)
      else:
        kernel_squared_sum = jnp.sum(kernel**2, axis=0, keepdims=True)

      # Clamp to zero: bf16 cancellation can make distance negative when x ≈ W
      distances = jnp.maximum(inputs_squared_sum + kernel_squared_sum - 2 * y, 0.0)

    # Add bias
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

    # YAT operation: (x · W)² / (||x - W||² + ε)
    y = y ** 2 / (distances + self.epsilon)

    # Apply alpha scaling
    if self._constant_alpha_value is not None:
      # Constant alpha: use directly as the scale factor (e.g. sqrt(2))
      y = y * self._constant_alpha_value
    elif alpha is not None:
      # Simple learnable alpha scaling
      y = y * alpha

    return y
