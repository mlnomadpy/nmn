from __future__ import annotations

import functools
from typing import Any, Callable, Optional
import typing as tp

import jax
import jax.numpy as jnp
from jax import lax, random

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import initializers
from flax.nnx.nn.dtypes import promote_dtype
from flax.nnx.nn.linear import (
  LinearGeneral,
  default_kernel_init,
)
from flax.nnx.nn.normalization import LayerNorm
from flax.typing import (
  Dtype,
  Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)

from nmn.nnx.nmn import YatNMN
from jax import Array

from nmn.nnx.squashers import softermax
def yat_attention_weights(
  query: Array,
  key: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
  epsilon: float = 1e-5,
  use_softermax: bool = False,
):
  """Computes attention weights using YatNMN distance-based calculation."""
  query, key = promote_dtype((query, key), dtype=dtype)
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # YatNMN-style attention calculation using the cleaner approach
  # query shape: [..., q_length, num_heads, head_dim]
  # key shape: [..., kv_length, num_heads, head_dim]

  # Calculate dot product attention scores
  attn = jnp.einsum('...qhd,...khd->...hqk', query, key, precision=precision)
  squared_dot_product = jnp.square(attn)

  # Calculate norms
  q_norm = jnp.sum(jnp.square(query), axis=-1, keepdims=True)  # [..., q_length, num_heads, 1]
  k_norm = jnp.sum(jnp.square(key), axis=-1, keepdims=True)    # [..., kv_length, num_heads, 1]
  qk_norm_sum = q_norm + k_norm  # Broadcasting: [..., q_length, num_heads, 1] + [..., kv_length, num_heads, 1]

  # Transpose to match attention dimensions [..., num_heads, q_length, kv_length]
  # The transpose converts [..., q_length, num_heads, kv_length] -> [..., num_heads, q_length, kv_length]
  batch_dims = len(qk_norm_sum.shape) - 3
  transpose_axes = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims, batch_dims + 2)
  qk_norm_sum_transposed = qk_norm_sum.transpose(transpose_axes)

  # Calculate squared distances: ||q||² + ||k||² - 2*(q·k)²
  squared_dist = qk_norm_sum_transposed - 2.0 * squared_dot_product

  # YatNMN attention scores: (q·k)² / (squared_distance + ε)
  attn_weights = squared_dot_product / (squared_dist + epsilon)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  if use_softermax:
    attn_weights = softermax(attn_weights).astype(dtype)
  else:
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if module:
    module.sow(nnx.Intermediate, 'attention_weights', attn_weights)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
    attn_weights = attn_weights * multiplier

  return attn_weights


def yat_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
  epsilon: float = 1e-5,
  use_softermax: bool = False,
):
  """Computes attention using YatNMN distance-based calculation."""
  query, key, value = promote_dtype((query, key, value), dtype=dtype)
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights using YatNMN
  attn_weights = yat_attention_weights(
    query,
    key,
    bias,
    mask,
    broadcast_dropout,
    dropout_rng,
    dropout_rate,
    deterministic,
    dtype,
    precision,
    module,
    epsilon,
    use_softermax,
  )

  # return weighted sum over values for each query position
  return jnp.einsum(
    '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )



def dot_product_attention_weights(
  query: Array,
  key: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      ``nnx.Intermediate`` collection. If ``module`` is None, the attention
      weights will not be sowed.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  query, key = promote_dtype((query, key), dtype=dtype)  # type: ignore[bad-unpacking]
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
    '...qhd,...khd->...hqk', query, key, precision=precision
  )

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if module:
    module.sow(nnx.Intermediate, 'attention_weights', attn_weights)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
    multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
    attn_weights = attn_weights * multiplier

  return attn_weights


def dot_product_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Optional[Array] = None,
  mask: Optional[Array] = None,
  broadcast_dropout: bool = True,
  dropout_rng: Optional[Array] = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Optional[Dtype] = None,
  precision: PrecisionLike = None,
  module: Optional[Module] = None,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  .. note::
    ``query``, ``key``, ``value`` needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of ``[batch..., q_length,
      num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    value: values to be used in attention with shape of ``[batch..., kv_length,
      num_heads, v_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      ``nnx.Intermediate`` collection. If ``module`` is None, the attention
      weights will not be sowed.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  query, key, value = promote_dtype((query, key, value), dtype=dtype)  # type: ignore[bad-unpacking]
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = dot_product_attention_weights(
    query,
    key,
    bias,
    mask,
    broadcast_dropout,
    dropout_rng,
    dropout_rate,
    deterministic,
    dtype,
    precision,
    module,
  )

  # return weighted sum over values for each query position
  return jnp.einsum(
    '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )


class MultiHeadAttention(Module):
  def __init__(
    self,
    num_heads: int,
    in_features: int,
    qkv_features: int | None = None,
    out_features: int | None = None,
    *,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.0,
    deterministic: bool | None = None,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    out_kernel_init: Initializer | None = None,
    bias_init: Initializer = initializers.zeros_init(),
    out_bias_init: Initializer | None = None,
    use_bias: bool = True,
    attention_fn: Callable[..., Array] = yat_attention,
    decode: bool | None = None,
    normalize_qk: bool = False,
    use_alpha: bool = True,
    alpha_init: Initializer = initializers.ones_init(),
    use_dropconnect: bool = False,
    dropconnect_rate: float = 0.0,
    # Deprecated, will be removed.
    qkv_dot_general: DotGeneralT | None = None,
    out_dot_general: DotGeneralT | None = None,
    qkv_dot_general_cls: Any = None,
    out_dot_general_cls: Any = None,
    rngs: rnglib.Rngs,
    epsilon: float = 1e-5,
    use_softermax: bool = False,
  ):
    self.num_heads = num_heads
    self.in_features = in_features
    self.qkv_features = (
      qkv_features if qkv_features is not None else in_features
    )
    self.out_features = (
      out_features if out_features is not None else in_features
    )
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.broadcast_dropout = broadcast_dropout
    self.dropout_rate = dropout_rate
    self.deterministic = deterministic
    self.precision = precision
    self.kernel_init = kernel_init
    self.out_kernel_init = out_kernel_init
    self.bias_init = bias_init
    self.out_bias_init = out_bias_init
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.decode = decode
    self.normalize_qk = normalize_qk
    self.qkv_dot_general = qkv_dot_general
    self.out_dot_general = out_dot_general
    self.qkv_dot_general_cls = qkv_dot_general_cls
    self.out_dot_general_cls = out_dot_general_cls
    self.epsilon = epsilon
    self.use_softermax = use_softermax
    self.use_alpha = use_alpha
    self.alpha_init = alpha_init
    self.use_dropconnect = use_dropconnect
    self.dropconnect_rate = dropconnect_rate

    if self.qkv_features % self.num_heads != 0:
      raise ValueError(
        f'Memory dimension ({self.qkv_features}) must be divisible by '
        f"'num_heads' heads ({self.num_heads})."
      )

    self.head_dim = self.qkv_features // self.num_heads

    # Replace LinearGeneral with YatNMN for query, key, value projections
    yat_linear = functools.partial(
      YatNMN,
      in_features=self.in_features,
      out_features=self.qkv_features,  # Output total features, will reshape later
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
      epsilon=self.epsilon,
      use_alpha=self.use_alpha,
      alpha_init=self.alpha_init,
      use_dropconnect=self.use_dropconnect,
      drop_rate=self.dropconnect_rate,
    )

    # project inputs_q to multi-headed q/k/v
    # dimensions will be reshaped to [batch..., length, n_heads, n_features_per_head]
    self.query = yat_linear(rngs=rngs)
    self.key = yat_linear(rngs=rngs)
    self.value = yat_linear(rngs=rngs)

    self.query_ln: LayerNorm | None
    self.key_ln: LayerNorm | None
    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      self.query_ln = LayerNorm(
        self.head_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        rngs=rngs,
      )
      self.key_ln = LayerNorm(
        self.head_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        rngs=rngs,
      )
    else:
      self.query_ln = None
      self.key_ln = None

    # Remove the output layer - no more self.out
    self.rngs = rngs if dropout_rate > 0.0 else None

    self.cached_key: nnx.Cache[Array] | None = None
    self.cached_value: nnx.Cache[Array] | None = None
    self.cache_index: nnx.Cache[Array] | None = None

  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Array | None = None,
    inputs_v: Array | None = None,
    *,
    mask: Array | None = None,
    deterministic: bool | None = None,
    rngs: rnglib.Rngs | None = None,
    sow_weights: bool = False,
    decode: bool | None = None,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    If both inputs_k and inputs_v are None, they will both copy the value of
    inputs_q (self attention).
    If only inputs_v is None, it will copy the value of inputs_k.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
        inputs_k will copy the value of inputs_q.
      inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
        inputs_v will copy the value of inputs_k.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      rngs: container for random number generators to generate the dropout
        mask when `deterministic` is False. The `rngs` container should have a
        `dropout` key.
      sow_weights: if ``True``, the attention weights are sowed into the
        'intermediates' collection.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if rngs is None:
      rngs = self.rngs

    if inputs_k is None:
      if inputs_v is not None:
        raise ValueError(
          '`inputs_k` cannot be None if `inputs_v` is not None. '
          'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
          'value to `inputs_k` and leave `inputs_v` as None.'
        )
      inputs_k = inputs_q
    if inputs_v is None:
      inputs_v = inputs_k

    if inputs_q.shape[-1] != self.in_features:
      raise ValueError(
        f'Incompatible input dimension, got {inputs_q.shape[-1]} '
        f'but module expects {self.in_features}.'
      )

    is_deterministic: bool = False
    if self.dropout_rate > 0.0 or (
      self.use_dropconnect and self.dropconnect_rate > 0.0
    ):
      is_deterministic = first_from(
        deterministic,
        self.deterministic,
        error_msg="""No `deterministic` argument was provided to MultiHeadAttention
          as either a __call__ argument, class attribute, or nnx.flag.""",
      )
    else:
      is_deterministic = True

    # Apply YatNMN transformations and reshape to multi-head format
    query = self.query(inputs_q, deterministic=is_deterministic)
    key = self.key(inputs_k, deterministic=is_deterministic)
    value = self.value(inputs_v, deterministic=is_deterministic)

    # Reshape from [batch..., length, qkv_features] to [batch..., length, num_heads, head_dim]
    query = query.reshape(query.shape[:-1] + (self.num_heads, self.head_dim))
    key = key.reshape(key.shape[:-1] + (self.num_heads, self.head_dim))
    value = value.reshape(value.shape[:-1] + (self.num_heads, self.head_dim))

    if self.normalize_qk:
      assert self.query_ln is not None and self.key_ln is not None
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = self.query_ln(query)
      key = self.key_ln(key)

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    decode = first_from(
      decode,
      self.decode,
      error_msg="""No `decode` argument was provided to MultiHeadAttention
        as either a __call__ argument, class attribute, or nnx.flag.""",
    )

    if decode:
      if (
        self.cached_key is None
        or self.cached_value is None
        or self.cache_index is None
      ):
        raise ValueError(
          'Autoregressive cache not initialized, call ``init_cache`` first.'
        )
      (
        *batch_dims,
        max_length,
        num_heads,
        depth_per_head,
      ) = self.cached_key.value.shape
      # shape check of cached keys against query input
      expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
      if expected_shape != query.shape:
        raise ValueError(
          'Autoregressive cache shape error, '
          'expected query shape %s instead got %s.'
#           % (expected_shape, query.shape)
        )
      # update key, value caches with our new 1d spatial slices
      cur_index = self.cache_index.value
      zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
      indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
      key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
      value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
      self.cached_key.value = key
      self.cached_value.value = value
      self.cache_index.value += 1
      # causal mask for cached decoder self-attention:
      # our single query position should only attend to those key
      # positions that have already been generated and cached,
      # not the remaining zero elements.
      mask = combine_masks(
        mask,
        jnp.broadcast_to(
          jnp.arange(max_length) <= cur_index,
          tuple(batch_dims) + (1, 1, max_length),
        ),
      )

    dropout_rng = None
    if self.dropout_rate > 0.0 and not is_deterministic:
      if rngs is None:
        raise ValueError("'rngs' must be provided for dropout.")
      dropout_rng = rngs.dropout()

    # apply attention with epsilon parameter for YatNMN
    x = self.attention_fn(
      query,
      key,
      value,
      mask=mask,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      deterministic=is_deterministic,
      dtype=self.dtype,
      precision=self.precision,
      module=self if sow_weights else None,
      epsilon=self.epsilon,  # Pass epsilon to yat_attention
      use_softermax=self.use_softermax,
    )
    # Reshape attention output back to original embedding dimension
    # from [batch..., length, num_heads, head_dim] to [batch..., length, qkv_features]
    x = x.reshape(x.shape[:-2] + (self.qkv_features,))
    return x

  def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
 
    cache_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
    self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype))
    self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype))
    self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))


# mask-making utility functions


def make_attention_mask(
  query_input: Array,
  key_input: Array,
  pairwise_fn: Callable[..., Any] = jnp.multiply,
  extra_batch_dims: int = 0,
  dtype: Dtype = jnp.float32,
):

  mask = pairwise_fn(
    jnp.expand_dims(query_input, axis=-1), jnp.expand_dims(key_input, axis=-2)
  )
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(
  x: Array, extra_batch_dims: int = 0, dtype: Dtype = jnp.float32
) -> Array:

  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
    idxs,
    idxs,
    jnp.greater_equal,
    extra_batch_dims=extra_batch_dims,
    dtype=dtype,
  )


def combine_masks(
  *masks: Optional[Array], dtype: Dtype = jnp.float32
) -> Array | None:

  masks_list = [m for m in masks if m is not None]
  if not masks_list:
    return None
  assert all(
    map(lambda x: x.ndim == masks_list[0].ndim, masks_list)
  ), f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks_list))}'
  mask, *other_masks = masks_list
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)



# Define a triangular mask for causal attention with `jax.numpy.tril` and `jax.numpy.ones`.
def causal_attention_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))
