from __future__ import annotations

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


default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()

class YatNMN(Module):
  """A YAT linear transformation applied over the last dimension of the input.

  The YAT  operation computes:
    y = (x · W)² / (||x - W||² + ε)

  With optional scaling:
    y = y * scale, where scale = (sqrt(out_features) / log(1 + out_features))^alpha

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
    rngs: rng key.
  """

  __data__ = ('kernel', 'bias', 'alpha', 'dropconnect_key')

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
    rngs: rnglib.Rngs,
  ):

    kernel_key = rngs.params()
    self.kernel = nnx.Param(
      kernel_init(kernel_key, (in_features, out_features), param_dtype)
    )
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
    self.epsilon = epsilon
    self.spherical = spherical
    self.drop_rate = drop_rate

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
    kernel = self.kernel.value
    bias = self.bias.value if self.bias is not None else None
    
    # Get alpha value (either learnable or constant)
    if self._constant_alpha_value is not None:
      alpha = jnp.array(self._constant_alpha_value, dtype=self.param_dtype)
    elif self.alpha is not None:
      alpha = self.alpha.value
    else:
      alpha = None

    if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
      keep_prob = 1.0 - self.drop_rate
      mask = jax.random.bernoulli(self.dropconnect_key, p=keep_prob, shape=kernel.shape)
      kernel = (kernel * mask) / keep_prob

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
      distances = 2 - 2 * y
    else:
      # Compute squared Euclidean distance: ||x||² + ||W||² - 2(x · W)
      inputs_squared_sum = jnp.sum(inputs**2, axis=-1, keepdims=True)
      kernel_squared_sum = jnp.sum(kernel**2, axis=0, keepdims=True)
      distances = inputs_squared_sum + kernel_squared_sum - 2 * y

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
      scale = (jnp.sqrt(self.out_features) / jnp.log(1 + self.out_features)) ** alpha
      y = y * scale

    return y
