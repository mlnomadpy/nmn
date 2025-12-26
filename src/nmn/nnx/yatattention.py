"""YAT Attention Module (Backwards Compatibility).

This module re-exports all attention components from the new modular structure
at `nmn.nnx.attention` for backwards compatibility.

For new code, prefer importing directly from `nmn.nnx.attention`:

    from nmn.nnx.attention import (
        MultiHeadAttention,
        yat_attention,
        yat_attention_weights,
        dot_product_attention,
        make_causal_mask,
    )
"""

# Re-export everything from the new modular structure
from nmn.nnx.attention import (
    # YAT Attention
    yat_attention,
    yat_attention_weights,
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
    "yat_attention",
    "yat_attention_weights",
    "dot_product_attention",
    "dot_product_attention_weights",
    "MultiHeadAttention",
    "make_attention_mask",
    "make_causal_mask",
    "combine_masks",
    "causal_attention_mask",
]
