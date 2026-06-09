"""PyTorch YAT Attention Module.

Provides:
- yat_attention: Full YAT attention (quadratic complexity)
- yat_attention_weights: YAT attention weight computation
- yat_attention_normalized: Optimized YAT for unit-normed Q/K
- normalize_qk: Normalize Q, K to unit vectors
- MultiHeadYatAttention: Multi-head attention module using YAT
- MAY (create_maclaurin_projection / maclaurin_features / maclaurin_yat_attention):
  bias-aware Random Maclaurin linear-attention feature map
- RAY (create_radial_projection / radial_features / radial_yat_attention):
  bias-aware sketched degree-2 × radial-RFF linear-attention feature map
"""

from .yat_attention import (
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
    normalize_qk,
)

from .multi_head import MultiHeadYatAttention, DEFAULT_CONSTANT_ALPHA

from .performer_yat import (
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
    linear_attention_readout,
)

__all__ = [
    "yat_attention",
    "yat_attention_weights",
    "yat_attention_normalized",
    "normalize_qk",
    "MultiHeadYatAttention",
    "DEFAULT_CONSTANT_ALPHA",
    # MAY / RAY linear-attention feature maps (issue #36)
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
    "linear_attention_readout",
]
