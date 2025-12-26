"""YAT Attention Module (Backwards Compatibility).

This module re-exports all attention components from the new modular structure
at `nmn.nnx.attention` for backwards compatibility.

For new code, prefer importing directly from `nmn.nnx.attention`:

    from nmn.nnx.attention import (
        MultiHeadAttention,
        RotaryYatAttention,
        yat_attention,
        yat_attention_weights,
        yat_performer_attention,
        rotary_yat_attention,
        performer_attention,
        dot_product_attention,
        make_causal_mask,
    )
"""

# Re-export everything from the new modular structure
from nmn.nnx.attention import (
    # YAT Attention
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
    yat_performer_attention,
    yat_performer_feature_map,
    create_yat_projection,
    normalize_qk,
    # Rotary YAT Attention
    RotaryYatAttention,
    rotary_yat_attention,
    rotary_yat_attention_weights,
    rotary_yat_performer_attention,
    precompute_freqs_cis,
    apply_rotary_emb,
    # Performer Attention
    performer_attention,
    performer_attention_weights,
    softmax_kernel_feature_map,
    relu_kernel_feature_map,
    orthogonal_random_features,
    create_performer_projection,
    # Standard Attention  
    dot_product_attention,
    dot_product_attention_weights,
    # Multi-Head Attention
    MultiHeadAttention,
    # Masks
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
    # Rotary YAT
    "RotaryYatAttention",
    "rotary_yat_attention",
    "rotary_yat_attention_weights",
    "rotary_yat_performer_attention",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    # Performer
    "performer_attention",
    "performer_attention_weights",
    "softmax_kernel_feature_map",
    "relu_kernel_feature_map",
    "orthogonal_random_features",
    "create_performer_projection",
    # Standard
    "dot_product_attention",
    "dot_product_attention_weights",
    "MultiHeadAttention",
    # Masks
    "make_attention_mask",
    "make_causal_mask",
    "combine_masks",
    "causal_attention_mask",
]
