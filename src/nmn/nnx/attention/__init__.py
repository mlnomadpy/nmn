"""NMN Attention Module.

This module provides attention mechanisms for neural networks, including
the novel YAT (You Are There) attention and standard scaled dot-product attention.

YAT Attention:
    Uses the formula: softmax((Q·K)² / (||Q-K||² + ε)) · V
    This balances similarity (dot product) with proximity (Euclidean distance).

Standard Attention:
    Uses the formula: softmax(Q·K / sqrt(d_k)) · V
    The classic scaled dot-product attention from "Attention Is All You Need".

Example:
    >>> from nmn.nnx.attention import MultiHeadAttention, yat_attention
    >>> from flax import nnx
    >>> import jax.numpy as jnp
    >>>
    >>> # Create multi-head attention with YAT
    >>> rngs = nnx.Rngs(0)
    >>> attn = MultiHeadAttention(
    ...     num_heads=8,
    ...     in_features=512,
    ...     rngs=rngs,
    ...     decode=False,
    ... )
    >>>
    >>> # Forward pass
    >>> x = jnp.zeros((2, 10, 512))
    >>> output = attn(x, deterministic=True)

Modules:
    - yat_attention: YAT attention functions
    - dot_product: Standard scaled dot-product attention functions
    - multi_head: MultiHeadAttention class
    - masks: Mask utility functions
"""

# YAT Attention
from .yat_attention import (
    yat_attention,
    yat_attention_weights,
)

# Standard Dot-Product Attention
from .dot_product import (
    dot_product_attention,
    dot_product_attention_weights,
)

# Multi-Head Attention Module
from .multi_head import MultiHeadAttention

# Mask Utilities
from .masks import (
    make_attention_mask,
    make_causal_mask,
    combine_masks,
    causal_attention_mask,
)

__all__ = [
    # YAT Attention
    "yat_attention",
    "yat_attention_weights",
    # Standard Attention
    "dot_product_attention",
    "dot_product_attention_weights",
    # Multi-Head Attention
    "MultiHeadAttention",
    # Masks
    "make_attention_mask",
    "make_causal_mask",
    "combine_masks",
    "causal_attention_mask",
]

