"""YAT Attention Functions (PyTorch).

Implements the YAT attention mechanism:
    ⵟ(q, k) = (q·k)² / (||q - k||² + ε)

This balances:
- Similarity: squared dot product (numerator)
- Proximity: squared Euclidean distance (denominator)

Spherical YAT:
    When spherical=True, Q and K are L2-normalized to unit vectors,
    simplifying the formula to: (q·k)² / (2(1 - q·k) + ε)
    This only requires ONE dot product instead of separate norm computations.

Alpha Scaling (learnable):
    scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha

Note: Constant alpha (e.g. sqrt(2)) is handled by the calling module
    as a direct scale factor, not passed through these functions.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def normalize_qk(
    query: Tensor,
    key: Tensor,
    epsilon: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Normalizes query and key to unit vectors.

    When Q and K are unit vectors, YAT simplifies:
        ||q - k||² = 2(1 - q·k)
    So: ⵟ(q, k) = (q·k)² / (2(1 - q·k) + ε)

    Args:
        query: [..., head_dim]
        key: [..., head_dim]
        epsilon: Numerical stability constant.

    Returns:
        Tuple of (normalized_query, normalized_key).
    """
    q_norm = torch.sqrt(torch.sum(query ** 2, dim=-1, keepdim=True) + epsilon)
    k_norm = torch.sqrt(torch.sum(key ** 2, dim=-1, keepdim=True) + epsilon)
    return query / q_norm, key / k_norm


def yat_attention_weights(
    query: Tensor,
    key: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    epsilon: float = 1e-5,
    alpha: Optional[Tensor] = None,
    scale: Optional[float] = None,
    spherical: bool = False,
) -> Tensor:
    """Computes YAT attention weights: softmax((Q·K)² / (||Q-K||² + ε))

    Args:
        query: (batch, q_len, num_heads, head_dim)
        key: (batch, kv_len, num_heads, head_dim)
        mask: Optional boolean mask broadcastable to
            (batch, num_heads, q_len, kv_len). True = attend, False = mask out.
        dropout_p: Dropout probability on attention weights.
        training: Whether in training mode (for dropout).
        epsilon: Numerical stability constant.
        alpha: Optional learnable alpha parameter. Used as exponent:
            attn *= (sqrt(head_dim) / log(1+head_dim))^alpha
        scale: Optional direct scale factor applied before softmax
            (e.g. sqrt(2) for constant_alpha).
        spherical: If True, L2-normalize Q/K and use simplified formula
            (q·k)² / (2(1-q·k) + ε).

    Returns:
        Attention weights of shape (batch, num_heads, q_len, kv_len)
    """
    assert query.ndim == key.ndim == 4, "Expected (batch, seq, heads, dim)"
    head_dim = query.shape[-1]

    if spherical:
        # Spherical YAT: normalize Q/K to unit vectors, simplified formula
        query, key = normalize_qk(query, key, epsilon)

        # Only one dot product needed
        dot_product = torch.einsum("bqhd,bkhd->bhqk", query, key)
        squared_dot = dot_product.square()

        # ||q-k||² = 2(1 - q·k) for unit vectors
        distance_sq = 2.0 - 2.0 * dot_product
        attn_weights = squared_dot / (distance_sq + epsilon)
    else:
        # Standard YAT: full norm computation
        dot_product = torch.einsum("bqhd,bkhd->bhqk", query, key)
        squared_dot = dot_product.square()

        q_norm_sq = query.square().sum(dim=-1, keepdim=True)
        k_norm_sq = key.square().sum(dim=-1, keepdim=True)

        q_norm_sq = q_norm_sq.permute(0, 2, 1, 3)
        k_norm_sq = k_norm_sq.permute(0, 2, 1, 3).transpose(-2, -1)

        squared_dist = q_norm_sq + k_norm_sq - 2.0 * dot_product
        attn_weights = squared_dot / (squared_dist + epsilon)

    # Alpha scaling (before softmax)
    # Either learnable alpha (exponent formula) or direct constant scale, never both
    if alpha is not None:
        alpha_scale = (math.sqrt(head_dim) / math.log(1 + head_dim)) ** alpha
        attn_weights = attn_weights * alpha_scale
    elif scale is not None:
        attn_weights = attn_weights * scale

    # Apply mask
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

    # Softmax normalization
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Dropout
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    return attn_weights


def yat_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    epsilon: float = 1e-5,
    alpha: Optional[Tensor] = None,
    scale: Optional[float] = None,
    spherical: bool = False,
) -> Tensor:
    """Computes YAT attention: softmax((Q·K)² / (||Q-K||² + ε)) · V

    Args:
        query: (batch, q_len, num_heads, head_dim)
        key: (batch, kv_len, num_heads, head_dim)
        value: (batch, kv_len, num_heads, v_dim)
        mask: Optional boolean mask (batch, num_heads, q_len, kv_len).
        dropout_p: Dropout probability.
        training: Whether in training mode.
        epsilon: Numerical stability constant.
        alpha: Optional learnable alpha parameter (exponent scaling).
        scale: Optional direct scale factor applied before softmax.
        spherical: If True, L2-normalize Q/K and use simplified formula.

    Returns:
        Output of shape (batch, q_len, num_heads, v_dim)
    """
    attn_weights = yat_attention_weights(
        query, key,
        mask=mask,
        dropout_p=dropout_p,
        training=training,
        epsilon=epsilon,
        alpha=alpha,
        scale=scale,
        spherical=spherical,
    )

    # Weighted sum: (B, H, Q, K) × (B, K, H, D) → (B, Q, H, D)
    return torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)


def yat_attention_normalized(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
    epsilon: float = 1e-5,
    alpha: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """YAT attention with normalized Q/K (optimized).

    Normalizes Q, K to unit vectors, then uses simplified formula:
        (q·k)² / (2(1 - q·k) + ε)

    Only needs ONE dot product instead of separate norm computations.

    Args:
        query: (batch, q_len, num_heads, head_dim) — will be normalized.
        key: (batch, kv_len, num_heads, head_dim) — will be normalized.
        value: (batch, kv_len, num_heads, v_dim)
        mask: Optional boolean mask.
        dropout_p: Dropout probability.
        training: Whether in training mode.
        epsilon: Numerical stability constant.
        alpha: Optional learnable alpha scaling.
        scale: Optional direct scale factor applied before softmax.

    Returns:
        Output of shape (batch, q_len, num_heads, v_dim)
    """
    head_dim = query.shape[-1]

    # Normalize to unit vectors
    query_n, key_n = normalize_qk(query, key, epsilon)

    # Dot product: q·k → (B, H, Q, K)
    dot_product = torch.einsum("bqhd,bkhd->bhqk", query_n, key_n)

    # Simplified: (q·k)² / (2(1 - q·k) + ε)
    squared_dot = dot_product.square()
    distance_sq = 2.0 - 2.0 * dot_product

    attn_weights = squared_dot / (distance_sq + epsilon)

    # Alpha scaling (before softmax)
    # Either learnable alpha (exponent formula) or direct constant scale, never both
    if alpha is not None:
        alpha_scale = (math.sqrt(head_dim) / math.log(1 + head_dim)) ** alpha
        attn_weights = attn_weights * alpha_scale
    elif scale is not None:
        attn_weights = attn_weights * scale

    # Mask
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Dropout
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Weighted sum
    return torch.einsum("bhqk,bkhd->bqhd", attn_weights, value)
