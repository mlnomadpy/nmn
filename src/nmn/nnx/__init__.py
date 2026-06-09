"""Neural-Matter Network (NMN) - Flax NNX Implementation.

This module provides YAT (You Are There) neural network layers and utilities
for Flax NNX.

Architecture Overview
---------------------
YAT layers compute: y = (x · W)² / (||x - W||² + ε) * alpha
This formula balances similarity (dot product) with distance (Euclidean norm).

Module Organization
-------------------
1. Core Layers: YatNMN (linear), Embed (embedding)
2. Convolution: YatConv, YatConvTranspose
3. Attention: Multi-head attention with YAT/Rotary/Performer variants
4. Activations: Custom squashing functions

Quick Start
-----------
    >>> from nmn.nnx import YatNMN, Embed, MultiHeadAttention, YatConv
    >>> from flax import nnx
    >>> import jax.numpy as jnp
    >>>
    >>> rngs = nnx.Rngs(0)
    >>>
    >>> # Core YAT linear layer with constant alpha = sqrt(2) (recommended)
    >>> layer = YatNMN(
    ...     in_features=128,
    ...     out_features=256,
    ...     constant_alpha=True,  # Use sqrt(2) scaling
    ...     spherical=False,      # Standard YAT
    ...     rngs=rngs
    ... )
    >>>
    >>> # YAT embedding for token embeddings
    >>> embed = Embed(
    ...     num_embeddings=10000,
    ...     features=128,
    ...     constant_alpha=True,
    ...     rngs=rngs
    ... )
    >>>
    >>> # Multi-head YAT attention
    >>> attn = MultiHeadAttention(
    ...     num_heads=8,
    ...     in_features=128,
    ...     use_rotary=False,     # Set True for rotary position embeddings
    ...     use_performer=False,  # Set True for O(n) linear complexity
    ...     rngs=rngs
    ... )
    >>>
    >>> # YAT convolution for vision tasks
    >>> conv = YatConv(
    ...     in_features=3,
    ...     out_features=64,
    ...     kernel_size=(3, 3),
    ...     strides=(1, 1),
    ...     padding='SAME',
    ...     constant_alpha=True,
    ...     rngs=rngs
    ... )
"""

# =============================================================================
# Core YAT Layers
# =============================================================================

from nmn.nnx.layers import YatNMN, Embed, FrozenParam


# =============================================================================
# Convolution Layers
# =============================================================================

from nmn.nnx.layers import (
    # Layers
    YatConv,
    YatConvTranspose,
    # Utilities
    canonicalize_padding,
    conv_dimension_numbers,
    default_kernel_init,
    default_bias_init,
    default_alpha_init,
    CONV_DEFAULT_CONSTANT_ALPHA,
)


# =============================================================================
# Attention Mechanisms
# =============================================================================

# Multi-Head Attention Module
from nmn.nnx.layers import (
    MultiHeadAttention,
    ATTENTION_DEFAULT_CONSTANT_ALPHA,
)

# YAT Attention Functions
from nmn.nnx.layers import (
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
    yat_performer_attention,
    yat_performer_feature_map,
    create_yat_projection,
    normalize_qk,
)

# Rotary YAT Attention (RoPE + YAT)
from nmn.nnx.layers import (
    RotaryYatAttention,
    rotary_yat_attention,
    rotary_yat_attention_weights,
    rotary_yat_performer_attention,
    precompute_freqs_cis,
    apply_rotary_emb,
)

# Spherical YAT-Performer (Linear Complexity)
from nmn.nnx.layers import (
    yat_tp_attention,
    yat_tp_features,
    create_yat_tp_projection,
)

# MAY / RAY bias-aware linear-attention feature maps
from nmn.nnx.layers import (
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    maclaurin_coeffs,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
)

# Standard Dot-Product Attention
from nmn.nnx.layers import (
    dot_product_attention,
    dot_product_attention_weights,
)

# Attention Masks
from nmn.nnx.layers import (
    make_attention_mask,
    make_causal_mask,
    combine_masks,
    causal_attention_mask,
)


# =============================================================================
# Activation Functions
# =============================================================================

from nmn.nnx.layers import (
    softermax,
    softer_sigmoid,
    soft_tanh,
)


# =============================================================================
# Cross-framework name aliases
# =============================================================================
# Torch / Keras / Linen / TF backends expose `YatEmbed`,
# `YatConv1D/2D/3D`, `YatConvTranspose1D/2D/3D`, and
# `MultiHeadYatAttention`. The NNX backend uses shorter / dimension-generic
# names internally; expose the longer names too so user code can stay the
# same when porting between backends.
YatEmbed = Embed
YatConv1D = YatConv
YatConv2D = YatConv
YatConv3D = YatConv
YatConvTranspose1D = YatConvTranspose
YatConvTranspose2D = YatConvTranspose
YatConvTranspose3D = YatConvTranspose
MultiHeadYatAttention = MultiHeadAttention


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Core Layers
    # -------------------------------------------------------------------------
    "YatNMN",
    "FrozenParam",
    "Embed",
    "YatEmbed",  # alias of Embed for cross-framework consistency

    # -------------------------------------------------------------------------
    # Convolution Layers
    # -------------------------------------------------------------------------
    "YatConv",
    "YatConvTranspose",
    # Dimension-specific aliases (match torch / keras / linen / tf names).
    # They all resolve to the dimension-generic YatConv / YatConvTranspose
    # in NNX; the dim is inferred from kernel_size at call time.
    "YatConv1D",
    "YatConv2D",
    "YatConv3D",
    "YatConvTranspose1D",
    "YatConvTranspose2D",
    "YatConvTranspose3D",
    # Conv Utilities
    "canonicalize_padding",
    "conv_dimension_numbers",
    "default_kernel_init",
    "default_bias_init",
    "default_alpha_init",
    "CONV_DEFAULT_CONSTANT_ALPHA",
    
    # -------------------------------------------------------------------------
    # Attention Mechanisms
    # -------------------------------------------------------------------------
    # Multi-Head Attention
    "MultiHeadAttention",
    "MultiHeadYatAttention",  # alias of MultiHeadAttention for cross-framework consistency
    "ATTENTION_DEFAULT_CONSTANT_ALPHA",
    
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
    
    # Spherical YAT-Performer
    "yat_tp_attention",
    "yat_tp_features",
    "create_yat_tp_projection",

    # MAY / RAY bias-aware feature maps
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "maclaurin_coeffs",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",

    # Standard Attention
    "dot_product_attention",
    "dot_product_attention_weights",
    
    # Attention Masks
    "make_attention_mask",
    "make_causal_mask",
    "combine_masks",
    "causal_attention_mask",
    
    # -------------------------------------------------------------------------
    # Activation Functions
    # -------------------------------------------------------------------------
    "softermax",
    "softer_sigmoid",
    "soft_tanh",
]
