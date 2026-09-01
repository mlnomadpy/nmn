"""TensorFlow backend for Neural Matter Network (NMN)."""

from .attention import (
    MultiHeadYatAttention,
    normalize_qk,
    yat_attention,
    yat_attention_normalized,
    yat_attention_weights,
)
from .embed import YatEmbed
from .nmn import YatDense, YatNMN
from .performer_yat import (
    create_maclaurin_projection,
    create_radial_projection,
    maclaurin_coeffs,
    maclaurin_features,
    maclaurin_yat_attention,
    radial_features,
    radial_yat_attention,
)
from .squashers import soft_tanh, softer_sigmoid, softermax

try:
    from .conv import (
        YatConv1D,
        YatConv1d,
        YatConv2D,
        YatConv2d,
        YatConv3D,
        YatConv3d,
        YatConvTranspose1D,
        YatConvTranspose1d,
        YatConvTranspose2D,
        YatConvTranspose2d,
        YatConvTranspose3D,
        YatConvTranspose3d,
    )

    _conv_all = [
        "YatConv1D",
        "YatConv2D",
        "YatConv3D",
        "YatConv1d",
        "YatConv2d",
        "YatConv3d",
        "YatConvTranspose1D",
        "YatConvTranspose2D",
        "YatConvTranspose3D",
        "YatConvTranspose1d",
        "YatConvTranspose2d",
        "YatConvTranspose3d",
    ]
except ImportError:
    _conv_all = []

__all__ = [
    "YatNMN",
    "YatDense",
    # Embedding
    "YatEmbed",
    # Attention
    "MultiHeadYatAttention",
    "normalize_qk",
    "yat_attention",
    "yat_attention_weights",
    "yat_attention_normalized",
    # Squashers
    "softermax",
    "softer_sigmoid",
    "soft_tanh",
    # Performer feature maps (MAY / RAY)
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
] + _conv_all
