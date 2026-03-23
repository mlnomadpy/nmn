"""YAT Embedding layer for PyTorch.

Provides token embedding lookup with a YAT-based attend() method for
weight-sharing between embedding and output layers in language models.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

__all__ = ["YatEmbed"]

DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatEmbed(nn.Module):
    """Embedding with YAT attend method.

    Standard embedding lookup via forward(), plus attend() which computes
    YAT similarity between queries and all embeddings — useful for
    weight-sharing between input embeddings and output logits.

    YAT attend formula:
        y = (query . embedding)^2 / (||query - embedding||^2 + epsilon)

    Args:
        num_embeddings: Size of the vocabulary.
        features: Dimension of each embedding vector.
        use_alpha: Whether to use alpha scaling (default: True).
        constant_alpha: If True, use sqrt(2). If float, use that value.
            If None (default), use learnable alpha when use_alpha=True.
        epsilon: Numerical stability constant (default: 1e-5).
        spherical: If True, normalize query and embeddings before YAT.
        weight_normalized: If True, normalize embeddings at init only.
        device: Device for parameters.
        dtype: Parameter dtype.

    Example::

        >>> embed = YatEmbed(num_embeddings=1000, features=128)
        >>> tokens = torch.tensor([0, 5, 3])
        >>> vectors = embed(tokens)        # (3, 128) — standard lookup
        >>> query = torch.randn(2, 128)
        >>> logits = embed.attend(query)   # (2, 1000) — YAT similarity
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
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.features = features
        self.epsilon = epsilon
        self.spherical = spherical
        self.weight_normalized = weight_normalized

        factory_kwargs = {"device": device, "dtype": dtype}

        # Initialize embedding with variance scaling (fan_in, normal)
        std = 1.0 / math.sqrt(features)
        self.embedding = Parameter(
            torch.randn(num_embeddings, features, **factory_kwargs) * std
        )

        # Weight normalization at init
        if weight_normalized:
            with torch.no_grad():
                norms = self.embedding.norm(dim=1, keepdim=True).clamp(min=1e-8)
                self.embedding.div_(norms)

        # Alpha configuration
        self._constant_alpha_value = None
        if constant_alpha is not None:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            self.register_parameter("alpha", None)
            use_alpha = True
        elif use_alpha:
            self.alpha = Parameter(torch.ones(1, **factory_kwargs))
        else:
            self.register_parameter("alpha", None)
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

    def forward(self, inputs: Tensor) -> Tensor:
        """Standard embedding lookup.

        Args:
            inputs: Integer tensor of token indices.

        Returns:
            Embedded vectors of shape (*inputs.shape, features).
        """
        return torch.nn.functional.embedding(inputs, self.embedding)

    def attend(self, query: Tensor) -> Tensor:
        """Compute YAT similarity between query and all embeddings.

        y = (query . embedding)^2 / (||query - embedding||^2 + epsilon)

        Args:
            query: Tensor with last dim equal to features.

        Returns:
            Tensor of shape (*query.shape[:-1], num_embeddings) with
            YAT similarity scores.
        """
        embedding = self.embedding

        # Spherical normalization
        if self.spherical:
            query = query / query.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Dot product: query @ embedding^T
        y = torch.matmul(query, embedding.t())

        # Squared distances
        if self.spherical:
            distances = 2.0 - 2.0 * y
        else:
            query_sq = torch.sum(query ** 2, dim=-1, keepdim=True)
            embed_sq = torch.sum(embedding ** 2, dim=1, keepdim=True).t()
            distances = query_sq + embed_sq - 2.0 * y

        # YAT: (dot)^2 / (dist + eps)
        y = y ** 2 / (distances + self.epsilon)

        # Alpha scaling
        if self._constant_alpha_value is not None:
            y = y * self._constant_alpha_value
        elif self.alpha is not None:
            y = y * self.alpha

        return y
