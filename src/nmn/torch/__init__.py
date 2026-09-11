"""PyTorch implementation of Neural Matter Network (NMN) layers."""

from . import _dependency as _dependency

# Import all layers from the layers module
# Import attention
from .attention import (
    MultiHeadYatAttention,
    create_maclaurin_projection,
    create_radial_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    radial_features,
    radial_yat_attention,
    yat_attention,
    yat_attention_weights,
)

# Import embedding
from .embed import YatEmbed
from .kernel_bank import KernelBank
from .layers import (
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)

# Import YatNMN from nmn module
from .nmn import YatNMN

# Import squashers
from .squashers import soft_tanh, softer_sigmoid, softermax

__all__ = [
    # YAT Conv layers
    "YatConv1D",
    "YatConv2D",
    "YatConv3D",
    "YatConvTranspose1D",
    "YatConvTranspose2D",
    "YatConvTranspose3D",
    # YAT NMN
    "YatNMN",
    "KernelBank",
    # YAT Attention
    "MultiHeadYatAttention",
    "yat_attention",
    "yat_attention_weights",
    # MAY / RAY linear-attention feature maps
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
    # YAT Embedding
    "YatEmbed",
    # Squashers
    "softermax",
    "softer_sigmoid",
    "soft_tanh",
]
