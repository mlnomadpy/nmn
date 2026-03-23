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
] + _conv_all

