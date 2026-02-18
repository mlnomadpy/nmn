"""PyTorch YAT Attention Module.

Provides:
- yat_attention: Full YAT attention (quadratic complexity)
- yat_attention_weights: YAT attention weight computation
- yat_attention_normalized: Optimized YAT for unit-normed Q/K
- normalize_qk: Normalize Q, K to unit vectors
- MultiHeadYatAttention: Multi-head attention module using YAT
"""

from .yat_attention import (
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
    normalize_qk,
)

from .multi_head import MultiHeadYatAttention, DEFAULT_CONSTANT_ALPHA

__all__ = [
    "yat_attention",
    "yat_attention_weights",
    "yat_attention_normalized",
    "normalize_qk",
    "MultiHeadYatAttention",
    "DEFAULT_CONSTANT_ALPHA",
]
