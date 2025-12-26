"""YAT Attention Functions.

This module implements the YAT attention mechanism, which replaces the standard
scaled dot-product attention with a distance-similarity tradeoff formula:

    ⵟ(q, k) = (q·k)² / (||q - k||² + ε)

This balances:
- Similarity: squared dot product (numerator)
- Proximity: squared Euclidean distance (denominator)

The result is attention that activates when queries and keys are both
similar in direction AND close in Euclidean space.
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
    power: float = 1.0,
) -> Array:
    """Computes YAT attention weights: softmax((Q·K)² / (||Q-K||² + ε))

    Uses the YAT formula to compute attention scores that balance similarity
    (squared dot product) with proximity (squared Euclidean distance).

    The YAT attention score: ⵟ(q, k) = (q·k)² / (||q - k||² + ε)

    Args:
        query: Queries with shape [..., q_length, num_heads, head_dim]
        key: Keys with shape [..., kv_length, num_heads, head_dim]
        bias: Optional bias for attention weights, broadcastable to
            [..., num_heads, q_length, kv_length]
        mask: Optional boolean mask, broadcastable to
            [..., num_heads, q_length, kv_length]. False values are masked out.
        broadcast_dropout: If True, dropout is broadcast across batch dims.
        dropout_rng: JAX PRNGKey for dropout.
        dropout_rate: Dropout probability (0.0 = no dropout).
        deterministic: If True, no dropout is applied.
        dtype: Computation dtype (inferred from inputs if None).
        precision: JAX precision for einsum operations.
        module: Optional Module to sow attention weights for introspection.
        epsilon: Small constant for numerical stability in denominator.
        use_softermax: If True, use softermax instead of standard softmax.
        power: Power parameter for softermax (only used if use_softermax=True).

    Returns:
        Attention weights of shape [..., num_heads, q_length, kv_length]
    """
    query, key = promote_dtype((query, key), dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # YAT-style attention: softmax((q·k)² / (||q-k||² + ε)) · V
    #
    # query shape: [..., q_length, num_heads, head_dim]
    # key shape: [..., kv_length, num_heads, head_dim]
    #
    # The YAT formula: ⵟ(q, k) = (q·k)² / (||q - k||² + ε)
    # where ||q - k||² = ||q||² - 2(q·k) + ||k||²

    # Calculate dot product: q·k
    # Output shape: [..., num_heads, q_length, kv_length]
    dot_product = jnp.einsum(
        "...qhd,...khd->...hqk", query, key, precision=precision
    )

    # Squared dot product for numerator: (q·k)²
    squared_dot_product = jnp.square(dot_product)

    # Calculate squared norms
    # q_norm: [..., q_length, num_heads, 1]
    # k_norm: [..., kv_length, num_heads, 1]
    q_norm_sq = jnp.sum(jnp.square(query), axis=-1, keepdims=True)
    k_norm_sq = jnp.sum(jnp.square(key), axis=-1, keepdims=True)

    # We need to compute ||q - k||² for each (q, k) pair
    # ||q - k||² = ||q||² + ||k||² - 2*(q·k)
    #
    # Reshape norms to broadcast correctly with dot_product shape
    # [..., num_heads, q_length, kv_length]
    batch_dims = query.ndim - 3

    # Transpose q_norm: [..., q_length, num_heads, 1] -> [..., num_heads, q_length, 1]
    q_axes = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims, batch_dims + 2)
    q_norm_transposed = q_norm_sq.transpose(q_axes)

    # Transpose k_norm: [..., kv_length, num_heads, 1] -> [..., num_heads, 1, kv_length]
    k_axes = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims, batch_dims + 2)
    k_norm_transposed = k_norm_sq.transpose(k_axes)
    k_norm_transposed = jnp.swapaxes(k_norm_transposed, -2, -1)

    # Squared Euclidean distance: ||q||² + ||k||² - 2*(q·k)
    squared_dist = q_norm_transposed + k_norm_transposed - 2.0 * dot_product

    # YAT attention scores: (q·k)² / (||q - k||² + ε)
    attn_weights = squared_dot_product / (squared_dist + epsilon)

    # Apply attention bias
    if bias is not None:
        attn_weights = attn_weights + bias

    # Apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # Normalize the attention weights
    if use_softermax:
        attn_weights = softermax(attn_weights, n=power).astype(dtype)
    else:
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
    power: float = 1.0,
) -> Array:
    """Computes YAT attention: softmax((Q·K)² / (||Q-K||² + ε)) · V

    This replaces the standard scaled dot-product attention with the YAT formula,
    which balances similarity (dot product) with proximity (Euclidean distance).

    The YAT attention score for each (query, key) pair is:
        ⵟ(q, k) = (q·k)² / (||q - k||² + ε)

    Args:
        query: Queries with shape [..., q_length, num_heads, head_dim]
        key: Keys with shape [..., kv_length, num_heads, head_dim]
        value: Values with shape [..., kv_length, num_heads, v_dim]
        bias: Optional bias for attention weights.
        mask: Optional boolean mask for attention weights.
        broadcast_dropout: If True, dropout is broadcast across batch dims.
        dropout_rng: JAX PRNGKey for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout is applied.
        dtype: Computation dtype.
        precision: JAX precision for einsum operations.
        module: Optional Module for sowing attention weights.
        epsilon: Small constant for numerical stability in denominator.
        use_softermax: If True, use softermax instead of standard softmax.
        power: Power parameter for softermax.

    Returns:
        Output of shape [..., q_length, num_heads, v_dim]
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

    # Compute attention weights using YAT formula
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
        power,
    )

    # Return weighted sum over values for each query position
    return jnp.einsum(
        "...hqk,...khd->...qhd", attn_weights, value, precision=precision
    )

