from __future__ import annotations

import functools
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
    constant_bias: if a float, use that value as a fixed (non-learnable) bias constant.
      If None (default), use learnable bias when use_bias=True.
    softplus_bias: if True, the learnable bias parameter is passed through
      softplus in the forward pass to guarantee strict positivity (default: False).
      Ignored when constant_bias is set or use_bias=False.
    scalar_bias: if True, the learnable bias is a single scalar (shape ``(1,)``)
      shared and broadcast across all ``out_features`` neurons (default: False).
      Ignored when constant_bias is set or use_bias=False.
    use_alpha: whether to use alpha scaling (default: True). Ignored if constant_alpha is set.
    constant_alpha: if True, use sqrt(2) as constant alpha. If a float, use that value.
      If None (default) or False, use learnable alpha when use_alpha=True.
      Note: False is treated as None — pass a float (e.g. 0.0) to actually freeze
      alpha at a numeric value.
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
    learnable_epsilon: if True, epsilon becomes a learnable parameter passed
      through softplus to guarantee strict positivity (default: False).
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

  __data__ = ('kernel', 'bias', 'alpha', 'epsilon_param', 'dropconnect_key')
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
    constant_bias: tp.Optional[float] = None,
    softplus_bias: bool = False,
    scalar_bias: bool = False,
    use_alpha: bool = True,
    constant_alpha: tp.Optional[tp.Union[bool, float]] = None,
    positive_init: bool = False,
    use_dropconnect: bool = False,
    fused: bool = False,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    alpha_init: Initializer = default_alpha_init,
    dot_general: DotGeneralT = lax.dot_general,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    epsilon: float = 1e-5,
    learnable_epsilon: bool = False,
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
    self._constant_bias_value: tp.Optional[float] = None
    if constant_bias is not None and constant_bias is not False:
      self._constant_bias_value = float(constant_bias)
      self.bias = None
      use_bias = True  # Bias is applied (but constant)
    elif use_bias:
      bias_key = rngs.params()
      bias_shape = (1,) if scalar_bias else (out_features,)
      self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
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
    
    if constant_alpha is not None and constant_alpha is not False:
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
    self.constant_bias = constant_bias
    self.softplus_bias = softplus_bias and self.bias is not None
    self.scalar_bias = scalar_bias and self.bias is not None
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
    self.learnable_epsilon = learnable_epsilon
    self.epsilon_param: nnx.Param[jax.Array] | None
    if learnable_epsilon:
      # Initialize so that softplus(raw) ≈ epsilon: raw = log(exp(eps) - 1)
      raw_eps = jnp.log(jnp.exp(jnp.array(epsilon, dtype=param_dtype)) - 1.0)
      self.epsilon_param = nnx.Param(raw_eps.reshape((1,)))
    else:
      self.epsilon_param = None
    self.spherical = spherical
    self.drop_rate = drop_rate
    self.fused = fused
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

    # Get bias value (either learnable or constant)
    if self._constant_bias_value is not None:
      bias = jnp.full((self.out_features,), self._constant_bias_value, dtype=self.param_dtype)
    elif self.bias is not None:
      bias = self.bias[...]
      if self.softplus_bias:
        bias = jax.nn.softplus(bias)
    else:
      bias = None
    
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
       inputs = inputs / (jnp.linalg.norm(inputs, axis=-1, keepdims=True) + 1e-8)
       kernel = kernel / (jnp.linalg.norm(kernel, axis=0, keepdims=True) + 1e-8)

    inputs, kernel, bias, alpha = self.promote_dtype(
      (inputs, kernel, bias, alpha), dtype=self.dtype
    )

    # Resolve effective epsilon (learnable via softplus, or constant)
    if self.learnable_epsilon and self.epsilon_param is not None:
      eps = jax.nn.softplus(self.epsilon_param[...].astype(jnp.float32))
    else:
      eps = self.epsilon

    # ── Fused path: custom_vjp saves only (x, W, dot, dist, bias, eps) ──
    if self.fused and not self.spherical:
      return _fused_yat_call(
        inputs, kernel, alpha, bias, eps,
        self._constant_alpha_value,
      )

    # ── Standard path ──────────────────────────────────────────────────
    # Upcast to float32 for YAT score computation.
    # YAT scores grow as O(dot²) — squaring causes overflow in bf16.
    inputs_f32 = inputs.astype(jnp.float32)
    kernel_f32 = kernel.astype(jnp.float32)

    # Compute dot product: x · W (in f32)
    y = self.dot_general(
      inputs_f32,
      kernel_f32,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )

    if self.spherical:
      distances = jnp.maximum(2 - 2 * y, 0.0)
    else:
      inputs_squared_sum = jnp.sum(inputs_f32**2, axis=-1, keepdims=True)

      if self.weight_normalized:
        kernel_squared_sum = jnp.ones((1, kernel_f32.shape[-1]), dtype=jnp.float32)
      else:
        kernel_squared_sum = jnp.sum(kernel_f32**2, axis=0, keepdims=True)

      distances = jnp.maximum(inputs_squared_sum + kernel_squared_sum - 2 * y, 0.0)

    # Add bias (in f32)
    if bias is not None:
      y += jnp.reshape(bias.astype(jnp.float32), (1,) * (y.ndim - 1) + (-1,))

    # YAT operation: (x · W)² / (||x - W||² + ε) — safe in f32
    y = y ** 2 / (distances + eps)

    # Apply alpha scaling
    if self._constant_alpha_value is not None:
      y = y * self._constant_alpha_value
    elif alpha is not None:
      y = y * alpha.astype(jnp.float32)

    # Cast back to caller's dtype
    return y.astype(inputs.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Fused YatNMN kernel — unified custom_vjp for reduced activation memory
#
# Standard autodiff saves 5+ intermediates (dot, dot_sq, x_sq, w_sq, dist, out).
# The fused version saves only (x, kernel, alpha, bias, eps, dot, dist) and
# recomputes dot_sq, x_sq, w_sq during backward.
#
# Supports all combinations of: bias / no-bias, learnable / constant epsilon,
# learnable / constant / no alpha.  Boolean flags via nondiff_argnums let JAX
# eliminate dead branches at trace time — zero runtime overhead.
# ══════════════════════════════════════════════════════════════════════════════


def _fused_yat_call(inputs, kernel, alpha, bias, eps, constant_alpha_value):
  """Dispatch to unified fused forward."""
  # Alpha array + grad flag
  if constant_alpha_value is not None:
    a = jnp.array(constant_alpha_value, dtype=jnp.float32)
    has_alpha_grad = False
  elif alpha is not None:
    a = alpha
    has_alpha_grad = True
  else:
    a = jnp.array(1.0, dtype=jnp.float32)
    has_alpha_grad = False

  # Bias array + flag
  if bias is not None:
    b = bias
    has_bias = True
  else:
    b = jnp.zeros(1, dtype=jnp.float32)
    has_bias = False

  # Epsilon — if it's a JAX array (from softplus), it's learnable
  if isinstance(eps, jnp.ndarray):
    e = eps
    has_eps_grad = True
  else:
    e = jnp.array(eps, dtype=jnp.float32)
    has_eps_grad = False

  return _fused_yat(inputs, kernel, a, b, e, has_alpha_grad, has_bias, has_eps_grad)


# ── Unified custom_vjp ──

@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _fused_yat(x, kernel, alpha, bias, eps, has_alpha_grad, has_bias, has_eps_grad):
  x_f32 = x.astype(jnp.float32)
  w_f32 = kernel.astype(jnp.float32)
  a_f32 = alpha.astype(jnp.float32)
  dot = x_f32 @ w_f32
  x_sq = jnp.sum(x_f32 ** 2, axis=-1, keepdims=True)
  w_sq = jnp.sum(w_f32 ** 2, axis=0, keepdims=True)
  dist = jnp.maximum(x_sq + w_sq - 2 * dot, 0.0) + eps.astype(jnp.float32)
  num = dot + bias.astype(jnp.float32) if has_bias else dot
  return (a_f32 * num ** 2 / dist).astype(x.dtype)


def _fused_yat_fwd(x, kernel, alpha, bias, eps, has_alpha_grad, has_bias, has_eps_grad):
  x_f32 = x.astype(jnp.float32)
  w_f32 = kernel.astype(jnp.float32)
  a_f32 = alpha.astype(jnp.float32)
  e_f32 = eps.astype(jnp.float32)
  dot = x_f32 @ w_f32
  x_sq = jnp.sum(x_f32 ** 2, axis=-1, keepdims=True)
  w_sq = jnp.sum(w_f32 ** 2, axis=0, keepdims=True)
  dist = jnp.maximum(x_sq + w_sq - 2 * dot, 0.0) + e_f32
  num = dot + bias.astype(jnp.float32) if has_bias else dot
  out = (a_f32 * num ** 2 / dist).astype(x.dtype)
  return out, (x, kernel, alpha, bias, eps, dot, dist)


def _fused_yat_bwd(has_alpha_grad, has_bias, has_eps_grad, res, g):
  x, kernel, alpha, bias, eps, dot, dist = res
  a_f32 = alpha.astype(jnp.float32)
  g_f32 = g.astype(jnp.float32)

  num = dot + bias.astype(jnp.float32) if has_bias else dot

  g_x, g_w = _fused_yat_grad_xw(x, kernel, num, dist, g, alpha_val=a_f32)

  # Alpha grad
  if has_alpha_grad:
    raw_yat = num ** 2 / dist
    g_alpha = jnp.sum(g_f32 * raw_yat).reshape(alpha.shape).astype(alpha.dtype)
  else:
    g_alpha = jnp.zeros_like(alpha)

  # Bias grad
  if has_bias:
    inv_dist = 1.0 / dist
    g_bias = jnp.sum(g_f32 * a_f32 * 2 * num * inv_dist,
                     axis=tuple(range(g.ndim - 1)))
    if bias.shape != g_bias.shape:
      g_bias = jnp.sum(g_bias, keepdims=True)
    g_bias = g_bias.astype(bias.dtype)
  else:
    g_bias = jnp.zeros_like(bias)

  # Epsilon grad
  if has_eps_grad:
    inv_dist_sq = 1.0 / (dist ** 2)
    g_eps = jnp.sum(g_f32 * (-a_f32 * num ** 2 * inv_dist_sq))
    g_eps = g_eps.reshape(eps.shape).astype(eps.dtype)
  else:
    g_eps = jnp.zeros_like(eps)

  return g_x, g_w, g_alpha, g_bias, g_eps


_fused_yat.defvjp(_fused_yat_fwd, _fused_yat_bwd)


# ── Shared gradient computation ──

def _fused_yat_grad_xw(x, kernel, num, dist, g, alpha_val):
  """Compute gradients for x and kernel.

  out = alpha * num^2 / dist,  where num = dot + bias (or just dot)
  dist = ||x||^2 + ||W||^2 - 2*(x@W) + eps

  Partials (holding other intermediates constant):
    d_out/d_num  = alpha * 2*num / dist
    d_out/d_dist = -alpha * num^2 / dist^2

  The dist→dot path (-2) is handled separately in the gradient assembly.
  """
  g_f32 = g.astype(jnp.float32)
  x_f32 = x.astype(jnp.float32)
  w_f32 = kernel.astype(jnp.float32)

  num_sq = num ** 2
  inv_dist = 1.0 / dist
  inv_dist_sq = inv_dist ** 2

  d_out_d_num = alpha_val * 2 * num * inv_dist
  d_out_d_dist = -alpha_val * num_sq * inv_dist_sq

  g_num = g_f32 * d_out_d_num
  g_dist = g_f32 * d_out_d_dist

  # Total dot gradient = through numerator (d_num/d_dot=1) + through dist (d_dist/d_dot=-2)
  g_dot_total = g_num + g_dist * (-2)

  # Grad x: through dot + through x_sq in dist
  g_dist_summed = jnp.sum(g_dist, axis=-1, keepdims=True)
  g_x = g_dot_total @ w_f32.T + g_dist_summed * (2 * x_f32)

  # Grad kernel: through dot + through w_sq in dist
  x_flat = x_f32.reshape(-1, x_f32.shape[-1])
  g_dot_flat = g_dot_total.reshape(-1, g_dot_total.shape[-1])
  g_dist_flat = g_dist.reshape(-1, g_dist.shape[-1])
  g_w = x_flat.T @ g_dot_flat
  g_w += 2 * w_f32 * jnp.sum(g_dist_flat, axis=0, keepdims=True)

  return g_x.astype(x.dtype), g_w.astype(kernel.dtype)
