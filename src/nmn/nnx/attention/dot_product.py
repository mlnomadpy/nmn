"""Standard Scaled Dot-Product Attention Functions.

This module implements the standard scaled dot-product attention mechanism
as described in "Attention Is All You Need" (Vaswani et al., 2017).

The attention score is: softmax(Q·K / sqrt(d_k)) · V

This is provided for comparison with YAT attention and for cases where
standard attention is preferred.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax import random

from flax import nnx
from flax.nnx.module import Module
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax import Array


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
) -> Array:
    """Computes scaled dot-product attention weights.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
        query: Queries with shape [..., q_length, num_heads, qk_depth_per_head]
        key: Keys with shape [..., kv_length, num_heads, qk_depth_per_head]
        bias: Optional bias for attention weights, broadcastable to
            [..., num_heads, q_length, kv_length]. Can be used for
            causal masks, padding masks, proximity bias, etc.
        mask: Optional boolean mask, broadcastable to
            [..., num_heads, q_length, kv_length]. False values are masked out.
        broadcast_dropout: If True, dropout is broadcast across batch dims.
        dropout_rng: JAX PRNGKey for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout is applied.
        dtype: Computation dtype (inferred from inputs if None).
        precision: JAX precision for einsum operations.
        module: Optional Module to sow attention weights for introspection.

    Returns:
        Attention weights of shape [..., num_heads, q_length, kv_length]
    """
    query, key = promote_dtype((query, key), dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # Calculate scaled attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    
    # Attention weight shape: [..., num_heads, q_length, kv_length]
    attn_weights = jnp.einsum(
        "...qhd,...khd->...hqk", query, key, precision=precision
    )

    # Apply attention bias
    if bias is not None:
        attn_weights = attn_weights + bias

    # Apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # Normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # Sow attention weights for introspection
    if module:
        module.sow(nnx.Intermediate, "attention_weights", attn_weights)

    # Apply attention dropout
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
) -> Array:
    """Computes scaled dot-product attention.

    This is the core function for applying attention based on
    "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    It calculates the attention weights given query and key, then combines
    values using the attention weights.

    Note:
        ``query``, ``key``, ``value`` needn't have any batch dimensions.

    Args:
        query: Queries with shape [..., q_length, num_heads, qk_depth_per_head]
        key: Keys with shape [..., kv_length, num_heads, qk_depth_per_head]
        value: Values with shape [..., kv_length, num_heads, v_depth_per_head]
        bias: Optional bias for attention weights.
        mask: Optional boolean mask for attention weights.
        broadcast_dropout: If True, dropout is broadcast across batch dims.
        dropout_rng: JAX PRNGKey for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout is applied.
        dtype: Computation dtype.
        precision: JAX precision for einsum operations.
        module: Optional Module for sowing attention weights.

    Returns:
        Output of shape [..., q_length, num_heads, v_depth_per_head]
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)
    dtype = query.dtype

    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # Compute attention weights
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

    # Return weighted sum over values for each query position
    return jnp.einsum(
        "...hqk,...khd->...qhd", attn_weights, value, precision=precision
    )

