"""Attention Mask Utility Functions.

This module provides utility functions for creating and combining attention masks
used in transformer architectures.

Masks are used to:
- Implement causal (autoregressive) attention
- Handle variable-length sequences (padding masks)
- Combine multiple mask types
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax.numpy as jnp
from flax.typing import Dtype
from jax import Array


def make_attention_mask(
    query_input: Array,
    key_input: Array,
    pairwise_fn: Callable[..., Any] = jnp.multiply,
    extra_batch_dims: int = 0,
    dtype: Dtype = jnp.float32,
) -> Array:
    """Creates an attention mask from query and key input arrays.

    This function creates a mask by applying a pairwise function to all
    combinations of query and key positions. Common use cases:
    
    - Padding mask: Pass boolean arrays indicating valid positions
    - Custom masks: Use custom pairwise functions

    Args:
        query_input: Array of shape [..., q_length] indicating query positions.
        key_input: Array of shape [..., kv_length] indicating key positions.
        pairwise_fn: Function to apply to each (query, key) pair.
            Default is multiply (for boolean AND).
        extra_batch_dims: Number of extra batch dimensions to add.
        dtype: Output dtype of the mask.

    Returns:
        Mask of shape [..., 1, q_length, kv_length] (with extra batch dims prepended)

    Example:
        >>> # Create padding mask from valid position indicators
        >>> query_valid = jnp.array([1, 1, 1, 0, 0])  # 3 valid positions
        >>> key_valid = jnp.array([1, 1, 0, 0])       # 2 valid positions
        >>> mask = make_attention_mask(query_valid, key_valid)
        >>> # mask[..., i, j] = 1 if both query[i] and key[j] are valid
    """
    mask = pairwise_fn(
        jnp.expand_dims(query_input, axis=-1),
        jnp.expand_dims(key_input, axis=-2),
    )
    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
    return mask.astype(dtype)


def make_causal_mask(
    x: Array,
    extra_batch_dims: int = 0,
    dtype: Dtype = jnp.float32,
) -> Array:
    """Creates a causal (autoregressive) attention mask.

    A causal mask ensures that each position can only attend to earlier
    positions (and itself), which is essential for autoregressive models
    like GPT and other decoder-only transformers.

    Args:
        x: Array of shape [..., length] used to determine the sequence length.
        extra_batch_dims: Number of extra batch dimensions to add.
        dtype: Output dtype of the mask.

    Returns:
        Lower triangular mask of shape [..., 1, length, length]
        where mask[..., i, j] = 1 if j <= i, else 0.

    Example:
        >>> x = jnp.zeros((2, 5))  # batch_size=2, seq_len=5
        >>> mask = make_causal_mask(x)
        >>> # mask allows position i to attend to positions 0...i
    """
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        jnp.greater_equal,
        extra_batch_dims=extra_batch_dims,
        dtype=dtype,
    )


def combine_masks(
    *masks: Optional[Array],
    dtype: Dtype = jnp.float32,
) -> Array | None:
    """Combines multiple attention masks using logical AND.

    This is useful for combining different types of masks, e.g.,
    a causal mask with a padding mask.

    Args:
        *masks: Variable number of mask arrays. None values are ignored.
            All non-None masks must have the same rank.
        dtype: Output dtype of the combined mask.

    Returns:
        Combined mask (logical AND of all input masks), or None if no masks provided.

    Example:
        >>> causal = make_causal_mask(x)
        >>> padding = make_attention_mask(query_valid, key_valid)
        >>> combined = combine_masks(causal, padding)
    """
    masks_list = [m for m in masks if m is not None]
    if not masks_list:
        return None

    assert all(
        map(lambda x: x.ndim == masks_list[0].ndim, masks_list)
    ), f"masks must have same rank: {tuple(map(lambda x: x.ndim, masks_list))}"

    mask, *other_masks = masks_list
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)

    return mask.astype(dtype)


def causal_attention_mask(seq_len: int) -> Array:
    """Creates a simple lower triangular causal mask.

    A convenience function for creating a basic causal mask without
    batch dimensions.

    Args:
        seq_len: Length of the sequence.

    Returns:
        Lower triangular mask of shape [seq_len, seq_len]
        where mask[i, j] = 1 if j <= i, else 0.

    Example:
        >>> mask = causal_attention_mask(4)
        >>> # [[1, 0, 0, 0],
        >>> #  [1, 1, 0, 0],
        >>> #  [1, 1, 1, 0],
        >>> #  [1, 1, 1, 1]]
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))

