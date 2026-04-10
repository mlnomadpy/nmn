
from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Dtype,
  Initializer,
  PromoteDtypeFn,
)

Array = jax.Array

default_embed_init = initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)
default_alpha_init = initializers.ones_init()


class Embed(Module):
  """Embedding Module with YAT support.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> layer = nnx.Embed(num_embeddings=5, features=3, rngs=nnx.Rngs(0))
    >>> nnx.state(layer)
    State({
      'embedding': Param( # 15 (60 B)
        value=Array([[ 0.57966787, -0.523274  , -0.43195742],
               [-0.676289  , -0.50300646,  0.33996582],
               [ 0.41796115, -0.59212935,  0.95934135],
               [-1.0917838 , -0.7441663 ,  0.07713798],
               [-0.66570747,  0.13815777,  1.007365  ]], dtype=float32)
      )
    })
    >>> # get the first three and last three embeddings
    >>> indices_input = jnp.array([[0, 1, 2], [-1, -2, -3]])
    >>> layer(indices_input)
    Array([[[ 0.57966787, -0.523274  , -0.43195742],
            [-0.676289  , -0.50300646,  0.33996582],
            [ 0.41796115, -0.59212935,  0.95934135]],
    <BLANKLINE>
           [[-0.66570747,  0.13815777,  1.007365  ],
            [-1.0917838 , -0.7441663 ,  0.07713798],
            [ 0.41796115, -0.59212935,  0.95934135]]], dtype=float32)

  A parameterized function from integers [0, ``num_embeddings``) to
  ``features``-dimensional vectors. This ``Module`` will create an ``embedding``
  matrix with shape ``(num_embeddings, features)``. When calling this layer,
  the input values will be used to 0-index into the ``embedding`` matrix.
  Indexing on a value greater than or equal to ``num_embeddings`` will result
  in ``nan`` values. When ``num_embeddings`` equals to 1, it will
  broadcast the ``embedding`` matrix to input shape with ``features``
  dimension appended.

  Args:
    num_embeddings: number of embeddings / vocab size.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: same as embedding).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    embedding_init: embedding initializer.
    promote_dtype: function to promote the dtype of the arrays to the desired
      dtype. The function should accept a tuple of ``(embedding,)`` during ``__call__``
      or ``(query, embedding)`` during ``attend``, and a ``dtype`` keyword argument,
      and return a tuple of arrays with the promoted dtype.
    use_alpha: whether to use alpha scaling in YAT attend (default: True).
    constant_alpha: if True, use sqrt(2) as constant alpha. If a float, use that value.
      If None (default), use learnable alpha when use_alpha=True.
    epsilon: small value added to denominator for numerical stability in YAT (default: 1e-5).
    learnable_epsilon: if True, epsilon becomes a learnable parameter passed
      through softplus to guarantee strict positivity (default: False).
    spherical: if True, normalize query and embeddings before YAT computation (default: False).
      Ignored if weight_normalized=True.
    weight_normalized: if True, normalize embeddings to unit norm at initialization only.
      Does not maintain normalization during training. (default: False)
    alpha_init: initializer function for learnable alpha (only used if constant_alpha is None).
    rngs: rng key.
  """

  # Default constant alpha value (sqrt(2))
  DEFAULT_CONSTANT_ALPHA = jnp.sqrt(2.0)

  def __init__(
    self,
    num_embeddings: int,
    features: int,
    *,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    embedding_init: Initializer = default_embed_init,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    use_alpha: bool = True,
    constant_alpha: tp.Optional[tp.Union[bool, float]] = None,
    epsilon: float = 1e-5,
    learnable_epsilon: bool = False,
    spherical: bool = False,
    weight_normalized: bool = False,
    alpha_init: Initializer = default_alpha_init,
    rngs: rnglib.Rngs,
  ):
    self.embedding = nnx.Param(
      embedding_init(rngs.params(), (num_embeddings, features), param_dtype)
    )

    # Handle alpha configuration
    # Priority: constant_alpha > use_alpha
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
    
    self.num_embeddings = num_embeddings
    self.features = features
    self.dtype = dtype or self.embedding[...].dtype
    self.param_dtype = param_dtype
    self.embedding_init = embedding_init
    self.promote_dtype = promote_dtype
    self.use_alpha = use_alpha
    self.constant_alpha = constant_alpha
    self.epsilon = epsilon
    self.learnable_epsilon = learnable_epsilon
    self.epsilon_param: nnx.Param[jax.Array] | None
    if learnable_epsilon:
      raw_eps = jnp.log(jnp.exp(jnp.array(epsilon, dtype=param_dtype)) - 1.0)
      self.epsilon_param = nnx.Param(raw_eps.reshape((1,)))
    else:
      self.epsilon_param = None
    self.spherical = spherical
    self.weight_normalized = weight_normalized

    # Normalize embeddings if requested: normalize each embedding to have norm 1
    if self.weight_normalized:
      embedding_val = self.embedding[...]
      embedding_norm = jnp.sqrt(jnp.sum(embedding_val**2, axis=1, keepdims=True))
      self.embedding.value = embedding_val / (embedding_norm + 1e-8)

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.
        Values in the input array must be integers.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional ``features`` dimension appended.
    """
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    # Use take because fancy indexing numpy arrays with JAX indices does not
    # work correctly.
    (embedding,) = self.promote_dtype(
      (self.embedding[...],), dtype=self.dtype, inexact=False
    )
    if self.num_embeddings == 1:
      return jnp.broadcast_to(embedding, inputs.shape + (self.features,))
    return jnp.take(embedding, inputs, axis=0)

  def attend(self, query: jax.Array) -> jax.Array:
    """Attend over the embedding using YAT.

    Computes: y = (query · embedding)² / (||query - embedding||² + ε)
    with optional alpha scaling.

    Args:
      query: array with last dimension equal the feature depth ``features`` of the
        embedding.

    Returns:
      An array with final dim ``num_embeddings`` corresponding to the YAT
      similarity between the query vectors and each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    embedding = self.embedding[...]
    
    # Get alpha value (either learnable or constant)
    if self._constant_alpha_value is not None:
      alpha = jnp.array(self._constant_alpha_value, dtype=self.param_dtype)
    elif self.alpha is not None:
      alpha = self.alpha[...]
    else:
      alpha = None

    # Spherical normalization (only if not using weight normalization)
    # weight_normalized only normalizes at init, doesn't maintain it during training
    if self.spherical:
      query = query / jnp.linalg.norm(query, axis=-1, keepdims=True)
      embedding = embedding / jnp.linalg.norm(embedding, axis=1, keepdims=True)

    query, embedding, alpha = self.promote_dtype(
      (query, embedding, alpha), dtype=self.dtype
    )
    
    # Compute dot product: query · embedding^T
    y = jnp.dot(query, embedding.T)

    if self.spherical:
      # Spherical YAT: query and embedding are normalized
      # distances = ||query||² + ||embedding||² - 2(query · embedding) = 1 + 1 - 2(query · embedding) = 2 - 2(query · embedding)
      distances = 2 - 2 * y
    else:
      # Compute squared Euclidean distance: ||query||² + ||embedding||² - 2(query · embedding)
      query_squared_sum = jnp.sum(query**2, axis=-1, keepdims=True)
      embedding_squared_sum = jnp.sum(embedding**2, axis=1, keepdims=True).T
      distances = query_squared_sum + embedding_squared_sum - 2 * y

    # Resolve effective epsilon (learnable via softplus, or constant)
    if self.learnable_epsilon and self.epsilon_param is not None:
      eps = jax.nn.softplus(self.epsilon_param[...].astype(jnp.float32))
    else:
      eps = self.epsilon

    # YAT operation: (query · embedding)² / (||query - embedding||² + ε)
    y = y ** 2 / (distances + eps)

    # Apply alpha scaling
    if self._constant_alpha_value is not None:
      # Constant alpha: use directly as the scale factor (e.g. sqrt(2))
      y = y * self._constant_alpha_value
    elif alpha is not None:
      # Learnable alpha scaling
      y = y * alpha

    return y
