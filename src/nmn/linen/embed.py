"""YAT Embedding layer for Flax Linen.

Provides token embedding lookup with a YAT-based attend() method for
weight-sharing between embedding and output layers in language models.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module, compact
from flax import linen as nn

__all__ = ["YatEmbed"]

DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)

default_embed_init = nn.initializers.variance_scaling(
    1.0, "fan_in", "normal", out_axis=0
)


class YatEmbed(Module):
    """Embedding with YAT attend method (Flax Linen).

    Standard embedding lookup via __call__(), plus attend() which computes
    YAT similarity between queries and all embeddings.

    Attributes:
        num_embeddings: Size of the vocabulary.
        features: Dimension of each embedding vector.
        use_alpha: Whether to use alpha scaling (default: True).
        constant_alpha: If True, use sqrt(2). If float, use that value.
            If None, use learnable alpha when use_alpha=True.
        epsilon: Numerical stability constant (default: 1e-5).
        spherical: If True, normalize query and embeddings before YAT.
        weight_normalized: If True, normalize embeddings at init only.
        dtype: Computation dtype.
        param_dtype: Parameter dtype.
        embedding_init: Initializer for embedding matrix.
    """

    num_embeddings: int = 1
    features: int = 1
    use_alpha: bool = True
    constant_alpha: Optional[Any] = None
    epsilon: float = 1e-5
    spherical: bool = False
    weight_normalized: bool = False
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    embedding_init: Any = default_embed_init
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)

    @compact
    def __call__(self, inputs):
        """Standard embedding lookup.

        Args:
            inputs: Integer tensor of token indices.

        Returns:
            Embedded vectors of shape (*inputs.shape, features).
        """
        embedding = self.param(
            "embedding",
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

        if self.weight_normalized:
            norms = jnp.sqrt(jnp.sum(embedding ** 2, axis=1, keepdims=True))
            embedding = embedding / (norms + 1e-8)

        (embedding,) = promote_dtype(embedding, dtype=self.dtype, inexact=False)

        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, inputs.shape + (self.features,))
        return jnp.take(embedding, inputs, axis=0)

    @compact
    def attend(self, query):
        """Compute YAT similarity between query and all embeddings.

        Args:
            query: Array with last dim equal to features.

        Returns:
            Array of shape (*query.shape[:-1], num_embeddings).
        """
        embedding = self.param(
            "embedding",
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

        # Alpha configuration
        _constant_alpha_value = None
        if self.constant_alpha is not None:
            if self.constant_alpha is True:
                _constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                _constant_alpha_value = float(self.constant_alpha)
            alpha = None
        elif self.use_alpha:
            alpha = self.param("alpha", self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None

        if self.weight_normalized:
            norms = jnp.sqrt(jnp.sum(embedding ** 2, axis=1, keepdims=True))
            embedding = embedding / (norms + 1e-8)

        query, embedding, alpha = promote_dtype(
            query, embedding, alpha, dtype=self.dtype
        )

        # Spherical normalization
        if self.spherical:
            query = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-8)
            embedding = embedding / (
                jnp.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
            )

        # Dot product
        y = jnp.dot(query, embedding.T)

        # Distances
        if self.spherical:
            distances = 2.0 - 2.0 * y
        else:
            query_sq = jnp.sum(query ** 2, axis=-1, keepdims=True)
            embed_sq = jnp.sum(embedding ** 2, axis=1, keepdims=True).T
            distances = query_sq + embed_sq - 2.0 * y

        # YAT
        y = y ** 2 / (distances + self.epsilon)

        # Alpha scaling
        if _constant_alpha_value is not None:
            y = y * _constant_alpha_value
        elif alpha is not None:
            y = y * alpha

        return y
