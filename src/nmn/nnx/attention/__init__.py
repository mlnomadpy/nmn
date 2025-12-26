"""NMN Attention Module.

This module provides attention mechanisms for neural networks, including
the novel YAT attention, Rotary YAT attention, Performer attention, and
standard scaled dot-product attention.

YAT Attention:
    Uses the formula: softmax((Q·K)² / (||Q-K||² + ε)) · V
    This balances similarity (dot product) with proximity (Euclidean distance).
    
    With optional alpha scaling:
        scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha
    
    Alpha can be learnable, constant (e.g., sqrt(2)), or disabled.

Rotary YAT Attention:
    Combines Rotary Position Embeddings (RoPE) with YAT attention:
        1. Apply RoPE: q' = RoPE(q, pos), k' = RoPE(k, pos)
        2. Compute YAT: softmax((q'·k')² / (||q' - k'||² + ε)) · V

Performer Attention:
    Uses FAVOR+ random feature approximation for O(n) linear complexity.
    Can be combined with YAT and Rotary embeddings.

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
    >>> # Rotary YAT with Performer for O(n) complexity
    >>> attn_performer = RotaryYatAttention(
    ...     embed_dim=512,
    ...     num_heads=8,
    ...     use_performer=True,
    ...     num_features=256,
    ...     rngs=rngs,
    ... )

Modules:
    - yat_attention: YAT attention functions (quadratic and Performer)
    - rotary_yat: Rotary YAT attention with RoPE
    - performer: Performer attention with FAVOR+ approximation
    - dot_product: Standard scaled dot-product attention functions
    - multi_head: MultiHeadAttention class
    - masks: Mask utility functions
"""

# YAT Attention (with Performer support)
from .yat_attention import (
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
    yat_performer_attention,
    yat_performer_feature_map,
    create_yat_projection,
    normalize_qk,
)

# Rotary YAT Attention (with Performer support)
from .rotary_yat import (
    RotaryYatAttention,
    rotary_yat_attention,
    rotary_yat_attention_weights,
    rotary_yat_performer_attention,
    precompute_freqs_cis,
    apply_rotary_emb,
)

# Performer Attention
from .performer import (
    performer_attention,
    performer_attention_weights,
    softmax_kernel_feature_map,
    relu_kernel_feature_map,
    orthogonal_random_features,
    create_performer_projection,
)

# Standard Dot-Product Attention
from .dot_product import (
    dot_product_attention,
    dot_product_attention_weights,
)

# Multi-Head Attention Module
from .multi_head import MultiHeadAttention, DEFAULT_CONSTANT_ALPHA

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
    "yat_attention_normalized",
    "yat_performer_attention",
    "yat_performer_feature_map",
    "create_yat_projection",
    "normalize_qk",
    # Rotary YAT Attention
    "RotaryYatAttention",
    "rotary_yat_attention",
    "rotary_yat_attention_weights",
    "rotary_yat_performer_attention",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    # Performer Attention
    "performer_attention",
    "performer_attention_weights",
    "softmax_kernel_feature_map",
    "relu_kernel_feature_map",
    "orthogonal_random_features",
    "create_performer_projection",
    # Standard Attention
    "dot_product_attention",
    "dot_product_attention_weights",
    # Multi-Head Attention
    "MultiHeadAttention",
    "DEFAULT_CONSTANT_ALPHA",
    # Masks
    "make_attention_mask",
    "make_causal_mask",
    "combine_masks",
    "causal_attention_mask",
]

