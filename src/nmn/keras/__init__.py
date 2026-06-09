"""Keras backend for Neural Matter Network (NMN)."""

from .nmn import YatNMN, YatDense
from .embed import YatEmbed
from .attention import (
    MultiHeadYatAttention,
    normalize_qk,
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
)
from .squashers import softermax, softer_sigmoid, soft_tanh
from .performer_yat import (
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
)

try:
    from .conv import (
        YatConv1D, YatConv2D, YatConv3D,
        YatConv1d, YatConv2d, YatConv3d,
        YatConvTranspose1D, YatConvTranspose2D, YatConvTranspose3D,
        YatConvTranspose1d, YatConvTranspose2d, YatConvTranspose3d,
    )
    _conv_all = [
        "YatConv1D", "YatConv2D", "YatConv3D",
        "YatConv1d", "YatConv2d", "YatConv3d",
        "YatConvTranspose1D", "YatConvTranspose2D", "YatConvTranspose3D",
        "YatConvTranspose1d", "YatConvTranspose2d", "YatConvTranspose3d",
    ]
except ImportError:
    _conv_all = []

__all__ = [
    "YatNMN", "YatDense",
    # Embedding
    "YatEmbed",
    # Attention
    "MultiHeadYatAttention",
    "normalize_qk",
    "yat_attention",
    "yat_attention_weights",
    "yat_attention_normalized",
    # Squashers
    "softermax", "softer_sigmoid", "soft_tanh",
    # Performer (MAY / RAY bias-aware linear-attention feature maps)
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
] + _conv_all