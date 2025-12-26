"""Performer Attention Module.

This module implements FAVOR+ (Fast Attention Via positive Orthogonal Random 
features) from the paper "Rethinking Attention with Performers".

Performer approximates softmax attention in O(n) time instead of O(n²) by:
1. Using random feature maps φ(x) to approximate the softmax kernel
2. Computing attention as (φ(Q) @ (φ(K)^T @ V)) instead of softmax(QK^T) @ V

Reference:
    - Rethinking Attention with Performers (https://arxiv.org/abs/2009.14794)
    - Choromanski et al., 2021
"""

from __future__ import annotations

from typing import Optional, Tuple
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from flax import nnx
from flax.nnx.module import Module
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax import Array


def orthogonal_random_features(
    key: Array,
    num_features: int,
    head_dim: int,
    dtype: Dtype = jnp.float32,
) -> Array:
    """Generates orthogonal random features for FAVOR+.

    Uses Gram-Schmidt orthogonalization to create orthogonal random projections,
    which provide better approximation than purely random projections.

    Args:
        key: JAX random key.
        num_features: Number of random features (m).
        head_dim: Dimension of each head (d).
        dtype: Data type for the output.

    Returns:
        Random projection matrix of shape [num_features, head_dim].
    """
    # Number of blocks needed
    num_blocks = (num_features + head_dim - 1) // head_dim

    blocks = []
    for i in range(num_blocks):
        key, subkey = random.split(key)
        # Generate random matrix
        random_matrix = random.normal(subkey, (head_dim, head_dim), dtype=dtype)
        # QR decomposition for orthogonalization
        q, _ = jnp.linalg.qr(random_matrix)
        blocks.append(q)

    # Stack and slice to get exact num_features
    projection = jnp.concatenate(blocks, axis=0)[:num_features]

    # Scale by sqrt(head_dim) for proper normalization
    return projection * jnp.sqrt(head_dim).astype(dtype)


def softmax_kernel_feature_map(
    x: Array,
    projection: Array,
    is_query: bool = True,
    epsilon: float = 1e-6,
) -> Array:
    """Applies the FAVOR+ softmax kernel feature map.

    Computes φ(x) such that φ(q)^T φ(k) ≈ exp(q·k / sqrt(d)).

    The feature map is:
        φ(x) = exp(||x||² / 2) * [exp(w_1·x), exp(w_2·x), ..., exp(w_m·x)] / sqrt(m)

    For numerical stability, we use:
        φ(x) = exp(x @ W^T - ||x||² / 2) / sqrt(m)

    Args:
        x: Input tensor of shape [..., seq_len, num_heads, head_dim].
        projection: Random projection matrix of shape [num_features, head_dim].
        is_query: Whether this is the query (for potential asymmetric treatment).
        epsilon: Small constant for numerical stability.

    Returns:
        Feature-mapped tensor of shape [..., seq_len, num_heads, num_features].
    """
    # x: [..., seq_len, num_heads, head_dim]
    # projection: [num_features, head_dim]

    head_dim = x.shape[-1]
    num_features = projection.shape[0]

    # Normalize by sqrt(head_dim) for the softmax kernel
    x_scaled = x / jnp.sqrt(head_dim).astype(x.dtype)

    # Compute x @ W^T: [..., seq_len, num_heads, num_features]
    # projection.T: [head_dim, num_features]
    x_proj = jnp.einsum("...d,md->...m", x_scaled, projection)

    # Compute ||x||² / 2 for normalization
    x_norm_sq = jnp.sum(x_scaled ** 2, axis=-1, keepdims=True) / 2.0

    # Feature map: exp(x @ W^T - ||x||² / 2) / sqrt(m)
    # This ensures φ(q)^T φ(k) ≈ exp(q·k / d)
    features = jnp.exp(x_proj - x_norm_sq + epsilon)
    features = features / jnp.sqrt(num_features).astype(x.dtype)

    return features


def relu_kernel_feature_map(
    x: Array,
    projection: Array,
    is_query: bool = True,
) -> Array:
    """Applies ReLU kernel feature map (simpler alternative to softmax kernel).

    Uses φ(x) = ReLU(x @ W^T) which approximates a different kernel but
    is simpler and sometimes works well in practice.

    Args:
        x: Input tensor of shape [..., seq_len, num_heads, head_dim].
        projection: Random projection matrix of shape [num_features, head_dim].
        is_query: Whether this is the query.

    Returns:
        Feature-mapped tensor of shape [..., seq_len, num_heads, num_features].
    """
    head_dim = x.shape[-1]
    num_features = projection.shape[0]

    # Normalize
    x_scaled = x / jnp.sqrt(head_dim).astype(x.dtype)

    # Project and apply ReLU
    x_proj = jnp.einsum("...d,md->...m", x_scaled, projection)
    features = jax.nn.relu(x_proj)
    features = features / jnp.sqrt(num_features).astype(x.dtype)

    return features


def performer_attention_weights(
    query: Array,
    key: Array,
    projection: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[Module] = None,
    use_softmax_kernel: bool = True,
    epsilon: float = 1e-6,
) -> Array:
    """Computes Performer attention weights using random features.

    Note: This returns the full attention weight matrix for compatibility,
    but the main benefit of Performer is avoiding this computation.
    Use `performer_attention` for the efficient O(n) computation.

    Args:
        query: Queries of shape [..., q_length, num_heads, head_dim].
        key: Keys of shape [..., kv_length, num_heads, head_dim].
        projection: Random projection matrix.
        bias: Optional attention bias.
        mask: Optional attention mask.
        broadcast_dropout: Whether to broadcast dropout.
        dropout_rng: RNG for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout.
        dtype: Computation dtype.
        precision: JAX precision.
        module: Optional module for sowing.
        use_softmax_kernel: If True, use softmax kernel; else use ReLU.
        epsilon: Numerical stability constant.

    Returns:
        Attention weights of shape [..., num_heads, q_length, kv_length].
    """
    query, key = promote_dtype((query, key), dtype=dtype)
    dtype = query.dtype

    # Apply feature map
    if use_softmax_kernel:
        q_features = softmax_kernel_feature_map(query, projection, is_query=True, epsilon=epsilon)
        k_features = softmax_kernel_feature_map(key, projection, is_query=False, epsilon=epsilon)
    else:
        q_features = relu_kernel_feature_map(query, projection, is_query=True)
        k_features = relu_kernel_feature_map(key, projection, is_query=False)

    # Compute approximate attention weights: φ(Q) @ φ(K)^T
    # q_features: [..., q_len, num_heads, num_features]
    # k_features: [..., kv_len, num_heads, num_features]
    attn_weights = jnp.einsum(
        "...qhm,...khm->...hqk", q_features, k_features, precision=precision
    )

    # Apply bias if provided
    if bias is not None:
        attn_weights = attn_weights + bias

    # Apply mask if provided
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # Normalize (row-wise softmax approximation)
    attn_weights = attn_weights / (jnp.sum(attn_weights, axis=-1, keepdims=True) + epsilon)

    # Sow attention weights if module provided
    if module:
        module.sow(nnx.Intermediate, "attention_weights", attn_weights)

    # Apply dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights


def performer_attention(
    query: Array,
    key: Array,
    value: Array,
    projection: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[Module] = None,
    use_softmax_kernel: bool = True,
    epsilon: float = 1e-6,
    causal: bool = False,
) -> Array:
    """Computes Performer attention in O(n) time.

    Uses FAVOR+ random feature approximation:
        Output ≈ φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)

    By computing in the order φ(K)^T @ V first, we get O(n) complexity.

    Args:
        query: Queries of shape [..., q_length, num_heads, head_dim].
        key: Keys of shape [..., kv_length, num_heads, head_dim].
        value: Values of shape [..., kv_length, num_heads, v_dim].
        projection: Random projection matrix of shape [num_features, head_dim].
        bias: Optional attention bias (not used in efficient mode).
        mask: Optional attention mask (not used in efficient mode).
        broadcast_dropout: Whether to broadcast dropout.
        dropout_rng: RNG for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout.
        dtype: Computation dtype.
        precision: JAX precision.
        module: Optional module for sowing.
        use_softmax_kernel: If True, use softmax kernel; else use ReLU.
        epsilon: Numerical stability constant.
        causal: If True, use causal (unidirectional) attention.

    Returns:
        Output of shape [..., q_length, num_heads, v_dim].
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)
    dtype = query.dtype

    # Apply feature map
    if use_softmax_kernel:
        q_features = softmax_kernel_feature_map(query, projection, is_query=True, epsilon=epsilon)
        k_features = softmax_kernel_feature_map(key, projection, is_query=False, epsilon=epsilon)
    else:
        q_features = relu_kernel_feature_map(query, projection, is_query=True)
        k_features = relu_kernel_feature_map(key, projection, is_query=False)

    # q_features: [..., q_len, num_heads, num_features]
    # k_features: [..., kv_len, num_heads, num_features]
    # value: [..., kv_len, num_heads, v_dim]

    if causal:
        # Causal attention requires prefix sums
        # This is more complex and uses cumulative sums
        return _causal_performer_attention(
            q_features, k_features, value, epsilon, precision
        )
    else:
        # Non-causal: efficient O(n) computation
        # Step 1: Compute φ(K)^T @ V: [num_features, v_dim] per head
        # k_features: [..., kv_len, num_heads, num_features]
        # value: [..., kv_len, num_heads, v_dim]
        kv = jnp.einsum(
            "...khm,...khd->...hmd", k_features, value, precision=precision
        )  # [..., num_heads, num_features, v_dim]

        # Step 2: Compute φ(Q) @ (φ(K)^T @ V)
        # q_features: [..., q_len, num_heads, num_features]
        # kv: [..., num_heads, num_features, v_dim]
        qkv = jnp.einsum(
            "...qhm,...hmd->...qhd", q_features, kv, precision=precision
        )  # [..., q_len, num_heads, v_dim]

        # Step 3: Compute normalizer φ(Q) @ φ(K)^T @ 1 = φ(Q) @ (sum_k φ(K))
        k_sum = jnp.sum(k_features, axis=-3, keepdims=False)  # [..., num_heads, num_features]
        normalizer = jnp.einsum(
            "...qhm,...hm->...qh", q_features, k_sum, precision=precision
        )  # [..., q_len, num_heads]
        normalizer = normalizer[..., None] + epsilon  # [..., q_len, num_heads, 1]

        # Normalize
        output = qkv / normalizer

        return output


def _causal_performer_attention(
    q_features: Array,
    k_features: Array,
    value: Array,
    epsilon: float,
    precision: PrecisionLike,
) -> Array:
    """Causal Performer attention using prefix sums.

    For causal attention, we need to compute:
        output[i] = (sum_{j<=i} φ(q[i]) @ φ(k[j]) @ v[j]) / (sum_{j<=i} φ(q[i]) @ φ(k[j]))

    This requires cumulative sums which can be computed in O(n).

    Args:
        q_features: Query features [..., q_len, num_heads, num_features].
        k_features: Key features [..., kv_len, num_heads, num_features].
        value: Values [..., kv_len, num_heads, v_dim].
        epsilon: Numerical stability constant.
        precision: JAX precision.

    Returns:
        Output of shape [..., q_len, num_heads, v_dim].
    """
    # Compute cumulative sums for causal attention
    # kv_cumsum[i] = sum_{j<=i} φ(k[j]) ⊗ v[j]
    # k_cumsum[i] = sum_{j<=i} φ(k[j])

    # k_features: [..., kv_len, num_heads, num_features]
    # value: [..., kv_len, num_heads, v_dim]

    # Outer product: [..., kv_len, num_heads, num_features, v_dim]
    kv = jnp.einsum("...khm,...khd->...khmd", k_features, value, precision=precision)

    # Cumulative sum along sequence dimension
    kv_cumsum = jnp.cumsum(kv, axis=-4)  # [..., kv_len, num_heads, num_features, v_dim]
    k_cumsum = jnp.cumsum(k_features, axis=-3)  # [..., kv_len, num_heads, num_features]

    # Compute output: φ(Q) @ kv_cumsum / (φ(Q) @ k_cumsum)
    # q_features: [..., q_len, num_heads, num_features]
    # kv_cumsum: [..., kv_len, num_heads, num_features, v_dim]

    # Numerator: [..., q_len, num_heads, v_dim]
    numerator = jnp.einsum(
        "...qhm,...qhmd->...qhd", q_features, kv_cumsum, precision=precision
    )

    # Denominator: [..., q_len, num_heads]
    denominator = jnp.einsum(
        "...qhm,...qhm->...qh", q_features, k_cumsum, precision=precision
    )
    denominator = denominator[..., None] + epsilon

    return numerator / denominator


def create_performer_projection(
    key: Array,
    num_features: int,
    head_dim: int,
    dtype: Dtype = jnp.float32,
    orthogonal: bool = True,
) -> Array:
    """Creates random projection matrix for Performer.

    Args:
        key: JAX random key.
        num_features: Number of random features.
        head_dim: Dimension of each head.
        dtype: Data type.
        orthogonal: If True, use orthogonal random features.

    Returns:
        Projection matrix of shape [num_features, head_dim].
    """
    if orthogonal:
        return orthogonal_random_features(key, num_features, head_dim, dtype)
    else:
        return random.normal(key, (num_features, head_dim), dtype=dtype) * jnp.sqrt(head_dim)

