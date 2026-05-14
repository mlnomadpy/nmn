"""YAT embedding layer for MLX.

Standard embedding lookup via ``__call__`` plus an ``attend`` method that
computes YAT similarity between query vectors and every embedding — useful
for weight-tying between embedding and output layers in LMs. Mirrors
``nmn.tf.embed.YatEmbed``.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn


__all__ = ["YatEmbed"]


DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatEmbed(nn.Module):
    """Embedding table with a YAT-flavoured ``attend`` method.

    Args:
        num_embeddings: Vocabulary size.
        features: Dimension of each embedding vector.
        use_alpha: Whether to apply an alpha scale in ``attend`` (default ``True``).
            Ignored when ``constant_alpha`` is set.
        constant_alpha: If ``True``, use √2; if a ``float``, use that value;
            if ``None``, fall back to a learnable scalar when ``use_alpha=True``.
        epsilon: Stability constant added to the YAT denominator.
        spherical: If ``True``, L2-normalize the query and each embedding row
            before computing the YAT score.
        weight_normalized: If ``True``, normalize embeddings to unit row norm
            **at init only** (the matrix is not re-normalized during training).
        dtype: MLX dtype for parameters and computation.
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
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.num_embeddings = num_embeddings
        self.features = features
        self.epsilon = epsilon
        self.spherical = spherical
        self.weight_normalized = weight_normalized
        self.dtype = dtype

        std = 1.0 / math.sqrt(features)
        initial = mx.random.normal(shape=(num_embeddings, features)) * std
        if weight_normalized:
            norms = mx.sqrt(mx.sum(initial * initial, axis=1, keepdims=True))
            initial = initial / (norms + 1e-8)
        self.embedding = initial.astype(dtype)

        # Alpha configuration.
        self._constant_alpha_value: Optional[float] = None
        if constant_alpha is not None and constant_alpha is not False:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        elif use_alpha:
            self.alpha = mx.ones((1,), dtype=dtype)
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

    def __call__(self, inputs: mx.array) -> mx.array:
        """Standard ``gather`` lookup.

        Args:
            inputs: Integer array of token ids; any shape.

        Returns:
            Array of shape ``inputs.shape + (features,)``.
        """
        if inputs.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64):
            raise ValueError(
                f"YatEmbed inputs must be integer; got dtype {inputs.dtype}"
            )
        return mx.take(self.embedding, inputs, axis=0)

    def attend(self, query: mx.array) -> mx.array:
        """YAT similarity between ``query`` and every embedding row.

        Args:
            query: Array with last dimension equal to ``features``.

        Returns:
            Array of shape ``query.shape[:-1] + (num_embeddings,)``.
        """
        if query.dtype != self.dtype:
            query = query.astype(self.dtype)
        embedding = self.embedding

        if self.spherical:
            query = query / (
                mx.sqrt(mx.sum(query * query, axis=-1, keepdims=True)) + 1e-8
            )
            embedding = embedding / (
                mx.sqrt(mx.sum(embedding * embedding, axis=1, keepdims=True)) + 1e-8
            )

        y = query @ embedding.T

        if self.spherical:
            distances = mx.maximum(2.0 - 2.0 * y, 0.0)
        else:
            query_sq = mx.sum(query * query, axis=-1, keepdims=True)
            embed_sq = mx.sum(embedding * embedding, axis=1, keepdims=True).T
            distances = mx.maximum(query_sq + embed_sq - 2.0 * y, 0.0)

        y = (y * y) / (distances + self.epsilon)

        if self._constant_alpha_value is not None:
            y = y * mx.array(self._constant_alpha_value, dtype=self.dtype)
        elif self.use_alpha and getattr(self, "alpha", None) is not None:
            y = y * self.alpha
        return y
