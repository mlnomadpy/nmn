"""YAT Embedding layer for Keras.

Provides token embedding lookup with a YAT-based attend() method for
weight-sharing between embedding and output layers in language models.
"""

from __future__ import annotations

import math
from typing import Optional, Union

from keras.src import initializers, ops
from keras.src.layers.layer import Layer

__all__ = ["YatEmbed"]

DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatEmbed(Layer):
    """Embedding with YAT attend method (Keras Layer).

    Standard embedding lookup via call(), plus attend() which computes
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
        embedding_initializer: Initializer for embedding matrix.
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
        embedding_initializer="glorot_normal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.features = features
        self.epsilon = epsilon
        self.spherical = spherical
        self.weight_normalized = weight_normalized
        self.embedding_initializer = initializers.get(embedding_initializer)

        self._constant_alpha_value = None
        if constant_alpha is not None:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

    def build(self, input_shape=None):
        self.embedding = self.add_weight(
            name="embedding",
            shape=(self.num_embeddings, self.features),
            initializer=self.embedding_initializer,
            trainable=True,
        )

        if self.weight_normalized:
            norms = ops.sqrt(
                ops.sum(ops.square(self.embedding), axis=1, keepdims=True)
            )
            self.embedding.assign(self.embedding / (norms + 1e-8))

        if self.use_alpha and self._constant_alpha_value is None:
            self.alpha = self.add_weight(
                name="alpha", shape=(1,), initializer="ones", trainable=True
            )
        else:
            self.alpha = None

        self.built = True

    def call(self, inputs):
        """Standard embedding lookup.

        Args:
            inputs: Integer tensor of token indices.

        Returns:
            Embedded vectors of shape (*inputs.shape, features).
        """
        return ops.take(self.embedding, inputs, axis=0)

    def attend(self, query):
        """Compute YAT similarity between query and all embeddings.

        Args:
            query: Tensor with last dim equal to features.

        Returns:
            Tensor of shape (*query.shape[:-1], num_embeddings).
        """
        if not self.built:
            self.build()

        embedding = self.embedding

        if self.spherical:
            query = query / (
                ops.sqrt(ops.sum(ops.square(query), axis=-1, keepdims=True)) + 1e-8
            )
            embedding = embedding / (
                ops.sqrt(ops.sum(ops.square(embedding), axis=1, keepdims=True)) + 1e-8
            )

        y = ops.matmul(query, ops.transpose(embedding))

        if self.spherical:
            distances = 2.0 - 2.0 * y
        else:
            query_sq = ops.sum(ops.square(query), axis=-1, keepdims=True)
            embed_sq = ops.transpose(
                ops.sum(ops.square(embedding), axis=1, keepdims=True)
            )
            distances = query_sq + embed_sq - 2.0 * y

        y = ops.square(y) / (distances + self.epsilon)

        if self._constant_alpha_value is not None:
            y = y * self._constant_alpha_value
        elif self.alpha is not None:
            y = y * self.alpha

        return y

    def compute_output_shape(self, input_shape):
        return input_shape + (self.features,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_embeddings": self.num_embeddings,
                "features": self.features,
                "use_alpha": self.use_alpha,
                "constant_alpha": self.constant_alpha,
                "epsilon": self.epsilon,
                "spherical": self.spherical,
                "weight_normalized": self.weight_normalized,
                "embedding_initializer": initializers.serialize(
                    self.embedding_initializer
                ),
            }
        )
        return config
