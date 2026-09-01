"""Neural-Matter Network Layers.

This module contains all YAT (You Are There) layer implementations and related components:
- Core layers: YatNMN (linear), Embed (embedding)
- Convolution: YatConv, YatConvTranspose
- Attention: MultiHeadAttention, RotaryYatAttention, attention functions
- Activations: softermax, softer_sigmoid, soft_tanh
"""

# =============================================================================
# Core YAT Layers
# =============================================================================

# Attention Masks
# Standard Dot-Product Attention
# MAY / RAY bias-aware linear-attention feature maps
# Spherical YAT-Performer (Linear Complexity)
# Rotary YAT Attention (RoPE + YAT)
# YAT Attention Functions
# Multi-Head Attention Module
from nmn.nnx.layers.attention import (
    DEFAULT_CONSTANT_ALPHA as ATTENTION_DEFAULT_CONSTANT_ALPHA,
)
from nmn.nnx.layers.attention import (
    MultiHeadAttention,
    RotaryYatAttention,
    apply_rotary_emb,
    causal_attention_mask,
    combine_masks,
    create_maclaurin_projection,
    create_radial_projection,
    create_yat_projection,
    create_yat_tp_projection,
    dot_product_attention,
    dot_product_attention_weights,
    maclaurin_coeffs,
    maclaurin_features,
    maclaurin_yat_attention,
    make_attention_mask,
    make_causal_mask,
    normalize_qk,
    precompute_freqs_cis,
    radial_features,
    radial_yat_attention,
    rotary_yat_attention,
    rotary_yat_attention_weights,
    rotary_yat_performer_attention,
    yat_attention,
    yat_attention_normalized,
    yat_attention_weights,
    yat_performer_attention,
    yat_performer_feature_map,
    yat_tp_attention,
    yat_tp_features,
)
from nmn.nnx.layers.conv import (
    DEFAULT_CONSTANT_ALPHA,
    YatConv,
    YatConvTranspose,
    canonicalize_padding,
    conv_dimension_numbers,
    default_alpha_init,
    default_bias_init,
    default_kernel_init,
)
from nmn.nnx.layers.embed import Embed
from nmn.nnx.layers.nmn import FrozenParam, YatNMN
from nmn.nnx.layers.squashers import (
    soft_tanh,
    softer_sigmoid,
    softermax,
)

CONV_DEFAULT_CONSTANT_ALPHA = DEFAULT_CONSTANT_ALPHA

# =============================================================================
# Convolution Layers
# =============================================================================


# =============================================================================
# Attention Mechanisms
# =============================================================================


# =============================================================================
# Activation Functions
# =============================================================================


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
    # -------------------------------------------------------------------------
    # Convolution Layers
    # -------------------------------------------------------------------------
    "YatConv",
    "YatConvTranspose",
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
