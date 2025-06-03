import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from flax import nnx
from flax.nnx.module import Module
from flax.nnx import rnglib
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Dtype,
  Initializer,
  PrecisionLike,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
  PromoteDtypeFn,
)

Array = jax.Array

# Default initializers
default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()

# Helper functions
def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
  """ "Canonicalizes conv padding to a jax.lax supported format."""
  if isinstance(padding, str):
    return padding
  if isinstance(padding, int):
    return [(padding, padding)] * rank
  if isinstance(padding, tp.Sequence) and len(padding) == rank:
    new_pad = []
    for p in padding:
      if isinstance(p, int):
        new_pad.append((p, p))
      elif isinstance(p, tuple) and len(p) == 2:
        new_pad.append(p)
      else:
        break
    if len(new_pad) == rank:
      return new_pad
  raise ValueError(
    f'Invalid padding format: {padding}, should be str, int,'
    f' or a sequence of len {rank} where each element is an'
    ' int or pair of ints.'
  )

def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

class YatConv(Module):
  """Yat Convolution Module wrapping ``lax.conv_general_dilated``.

  Example usage::

    >>> from flax.nnx import conv # Assuming this file is flax/nnx/conv.py
    >>> import jax, jax.numpy as jnp
    >>> from flax.nnx import rnglib, state

    >>> rngs = rnglib.Rngs(0)
    >>> x = jnp.ones((1, 8, 3))

    >>> # valid padding
    >>> layer = conv.Conv(in_features=3, out_features=4, kernel_size=(3,),
    ...                  padding='VALID', rngs=rngs)
    >>> s = state(layer)
    >>> print(s['kernel'].value.shape)
    (3, 3, 4)
    >>> print(s['bias'].value.shape)
    (4,)
    >>> out = layer(x)
    >>> print(out.shape)
    (1, 6, 4)

  Args:
    in_features: int or tuple with number of input features.
    out_features: int or tuple with number of output features.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer, which will be interpreted
      as a tuple of the single integer. For all other cases, it must be a
      sequence of integers.
    strides: an integer or a sequence of ``n`` integers, representing the
      inter-window strides (default: 1).
    padding: either the string ``'SAME'``, the string ``'VALID'``, the string
      ``'CIRCULAR'`` (periodic boundary conditions), the string `'REFLECT'`
      (reflection across the padding boundary), or a sequence of ``n``
      ``(low, high)`` integer pairs that give the padding to apply before and after each
      spatial dimension. A single int is interpeted as applying the same padding
      in all dims and passign a single int in a sequence causes the same padding
      to be used on both sides. ``'CAUSAL'`` padding for a 1D convolution will
      left-pad the convolution axis, resulting in same-sized output.
    input_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of ``inputs``
      (default: 1). Convolution with input dilation ``d`` is equivalent to
      transposed convolution with stride ``d``.
    kernel_dilation: an integer or a sequence of ``n`` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel (default: 1). Convolution with kernel dilation
      is also known as 'atrous convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    mask: Optional mask for the weights during masked convolution. The mask must
          be the same shape as the convolution weight matrix.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    promote_dtype: function to promote the dtype of the arrays to the desired
      dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
      and a ``dtype`` keyword argument, and return a tuple of arrays with the
      promoted dtype.
    epsilon: A small float added to the denominator to prevent division by zero.
    rngs: rng key.
  """

  __data__ = ('kernel', 'bias', 'mask')

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: int | tp.Sequence[int],
    strides: tp.Union[None, int, tp.Sequence[int]] = 1,
    *,
    padding: PaddingLike = 'SAME',
    input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
    feature_group_count: int = 1,
    use_bias: bool = True,
    mask: tp.Optional[Array] = None,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    bias_init: Initializer = default_bias_init,
    conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    epsilon: float = 1e-5,
    rngs: rnglib.Rngs,
  ):
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)
    else:
      kernel_size = tuple(kernel_size)

    self.kernel_shape = kernel_size + (
      in_features // feature_group_count,
      out_features,
    )
    kernel_key = rngs.params()
    self.kernel = nnx.Param(kernel_init(kernel_key, self.kernel_shape, param_dtype))

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      bias_shape = (out_features,)
      bias_key = rngs.params()
      self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
    else:
      self.bias = None

    self.in_features = in_features
    self.out_features = out_features
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.input_dilation = input_dilation
    self.kernel_dilation = kernel_dilation
    self.feature_group_count = feature_group_count
    self.use_bias = use_bias
    self.mask = mask 
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.precision = precision
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.conv_general_dilated = conv_general_dilated
    self.promote_dtype = promote_dtype
    self.epsilon = epsilon

  def __call__(self, inputs: Array) -> Array:
    assert isinstance(self.kernel_size, tuple)

    def maybe_broadcast(
      x: tp.Optional[tp.Union[int, tp.Sequence[int]]],
    ) -> tuple[int, ...]:
      if x is None:
        x = 1
      if isinstance(x, int):
        return (x,) * len(self.kernel_size)
      return tuple(x)

    num_batch_dimensions = inputs.ndim - (len(self.kernel_size) + 1)
    if num_batch_dimensions != 1:
      input_batch_shape = inputs.shape[:num_batch_dimensions]
      total_batch_size = int(np.prod(input_batch_shape))
      flat_input_shape = (total_batch_size,) + inputs.shape[
        num_batch_dimensions:
      ]
      inputs_flat = jnp.reshape(inputs, flat_input_shape)
    else:
      inputs_flat = inputs
      input_batch_shape = () 

    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax = canonicalize_padding(self.padding, len(self.kernel_size))
    if padding_lax in ('CIRCULAR', 'REFLECT'):
      assert isinstance(padding_lax, str)
      kernel_size_dilated = [
        (k - 1) * d + 1 for k, d in zip(self.kernel_size, kernel_dilation)
      ]
      zero_pad: tp.List[tuple[int, int]] = [(0, 0)]
      pads = (
        zero_pad
        + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
        + [(0, 0)]
      )
      padding_mode = {'CIRCULAR': 'wrap', 'REFLECT': 'reflect'}[padding_lax]
      inputs_flat = jnp.pad(inputs_flat, pads, mode=padding_mode)
      padding_lax = 'VALID'
    elif padding_lax == 'CAUSAL':
      if len(self.kernel_size) != 1:
        raise ValueError(
          'Causal padding is only implemented for 1D convolutions.'
        )
      left_pad = kernel_dilation[0] * (self.kernel_size[0] - 1)
      pads = [(0, 0), (left_pad, 0), (0, 0)]
      inputs_flat = jnp.pad(inputs_flat, pads)
      padding_lax = 'VALID'

    dimension_numbers = _conv_dimension_numbers(inputs_flat.shape)
    assert self.in_features % self.feature_group_count == 0

    kernel_val = self.kernel.value
    
    current_mask = self.mask 
    if current_mask is not None:
      if current_mask.shape != self.kernel_shape:
        raise ValueError(
          'Mask needs to have the same shape as weights. '
          f'Shapes are: {current_mask.shape}, {self.kernel_shape}'
        )
      kernel_val *= current_mask

    bias_val = self.bias.value if self.bias is not None else None

    inputs_promoted, kernel_promoted, bias_promoted = self.promote_dtype(
      (inputs_flat, kernel_val, bias_val), dtype=self.dtype
    )
    inputs_flat = inputs_promoted
    kernel_val = kernel_promoted
    bias_val = bias_promoted

    dot_prod_map = self.conv_general_dilated(
      inputs_flat,
      kernel_val,
      strides,
      padding_lax,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision,
    )

    inputs_flat_squared = inputs_flat**2
    kernel_in_channels_for_sum_sq = self.kernel_shape[-2]
    kernel_for_patch_sq_sum_shape = self.kernel_size + (kernel_in_channels_for_sum_sq, 1)
    kernel_for_patch_sq_sum = jnp.ones(kernel_for_patch_sq_sum_shape, dtype=kernel_val.dtype)

    patch_sq_sum_map_raw = self.conv_general_dilated(
      inputs_flat_squared,
      kernel_for_patch_sq_sum,
      strides,
      padding_lax,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=self.precision,
    )

    if self.feature_group_count > 1:
      num_out_channels_per_group = self.out_features // self.feature_group_count
      if num_out_channels_per_group == 0 :
          raise ValueError(
              "out_features must be a multiple of feature_group_count and greater or equal."
          )
      patch_sq_sum_map = jnp.repeat(patch_sq_sum_map_raw, num_out_channels_per_group, axis=-1)
    else:
      patch_sq_sum_map = patch_sq_sum_map_raw

    reduce_axes_for_kernel_sq = tuple(range(kernel_val.ndim - 1))
    kernel_sq_sum_per_filter = jnp.sum(kernel_val**2, axis=reduce_axes_for_kernel_sq)

    distance_sq_map = patch_sq_sum_map + kernel_sq_sum_per_filter - 2 * dot_prod_map
    y = dot_prod_map**2 / (distance_sq_map + self.epsilon)

    if self.use_bias and bias_val is not None:
      bias_reshape_dims = (1,) * (y.ndim - 1) + (-1,)
      y += jnp.reshape(bias_val, bias_reshape_dims)

    if num_batch_dimensions != 1:
      output_shape = input_batch_shape + y.shape[1:]
      y = jnp.reshape(y, output_shape)
    return y 
