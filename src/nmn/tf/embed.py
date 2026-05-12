"""YAT Embedding layer for TensorFlow.

Provides token embedding lookup with a YAT-based attend() method for
weight-sharing between embedding and output layers in language models.
"""

from __future__ import annotations

import math
from typing import Optional, Union, List

import tensorflow as tf

__all__ = ["YatEmbed"]

DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatEmbed(tf.Module):
    """Embedding with YAT attend method (tf.Module).

    Standard embedding lookup via __call__(), plus attend() which computes
    YAT similarity between queries and all embeddings.

    Args:
        num_embeddings: Size of the vocabulary.
        features: Dimension of each embedding vector.
        use_alpha: Whether to use alpha scaling (default: True).
        constant_alpha: If True, use sqrt(2). If float, use that value.
            If None, use learnable alpha when use_alpha=True.
        epsilon: Numerical stability constant (default: 1e-5).
        spherical: If True, normalize query and embeddings before YAT.
        weight_normalized: If True, normalize embeddings at init only.
        dtype: TensorFlow dtype (default: tf.float32).
        name: Optional name scope.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        epsilon: float = 1e-5,
        spherical: bool = False,
        weight_normalized: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.features = features
        self.epsilon = epsilon
        self.spherical = spherical
        self.weight_normalized = weight_normalized
        self.dtype = dtype

        # Initialize embedding with variance scaling
        std = 1.0 / math.sqrt(features)
        initial = tf.random.normal(
            (num_embeddings, features), stddev=std, dtype=dtype
        )

        if weight_normalized:
            norms = tf.sqrt(tf.reduce_sum(tf.square(initial), axis=1, keepdims=True))
            initial = initial / (norms + 1e-8)

        self.embedding = tf.Variable(initial, trainable=True, name="embedding")

        # Alpha configuration
        self._constant_alpha_value = None
        if constant_alpha is not None and constant_alpha is not False:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
            self.alpha = None
        elif use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=dtype), trainable=True, name="alpha"
            )
        else:
            self.alpha = None
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Standard embedding lookup.

        Args:
            inputs: Integer tensor of token indices.

        Returns:
            Embedded vectors of shape (*inputs.shape, features).
        """
        return tf.gather(self.embedding, inputs)

    @tf.Module.with_name_scope
    def attend(self, query: tf.Tensor) -> tf.Tensor:
        """Compute YAT similarity between query and all embeddings.

        Args:
            query: Tensor with last dim equal to features.

        Returns:
            Tensor of shape (*query.shape[:-1], num_embeddings).
        """
        query = tf.cast(query, self.dtype)
        embedding = self.embedding

        if self.spherical:
            query = query / (
                tf.sqrt(tf.reduce_sum(tf.square(query), axis=-1, keepdims=True)) + 1e-8
            )
            embedding = embedding / (
                tf.sqrt(tf.reduce_sum(tf.square(embedding), axis=1, keepdims=True))
                + 1e-8
            )

        y = tf.matmul(query, tf.transpose(embedding))

        if self.spherical:
            distances = 2.0 - 2.0 * y
        else:
            query_sq = tf.reduce_sum(tf.square(query), axis=-1, keepdims=True)
            embed_sq = tf.transpose(
                tf.reduce_sum(tf.square(embedding), axis=1, keepdims=True)
            )
            distances = query_sq + embed_sq - 2.0 * y

        y = tf.square(y) / (distances + self.epsilon)

        if self._constant_alpha_value is not None:
            y = y * tf.cast(self._constant_alpha_value, self.dtype)
        elif self.alpha is not None:
            y = y * self.alpha

        return y
