"""Flax Linen backend for Neural Matter Network (NMN)."""

from .nmn import YatNMN
from .embed import YatEmbed
from .attention import (
    MultiHeadAttention,
    normalize_qk,
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
)
from .squashers import softermax, softer_sigmoid, soft_tanh
from .performer_yat import (
    spherical_kappa,
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
    linear_attention,
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
    "YatNMN",
    # Embedding
    "YatEmbed",
    # Attention
    "MultiHeadAttention",
    "normalize_qk",
    "yat_attention",
    "yat_attention_weights",
    "yat_attention_normalized",
    # Squashers
    "softermax", "softer_sigmoid", "soft_tanh",
    # Performer / linear-attention feature maps (MAY + RAY)
    "spherical_kappa",
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
    "linear_attention",
] + _conv_all

