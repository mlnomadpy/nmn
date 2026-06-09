"""PyTorch implementation of Neural Matter Network (NMN) layers."""

# Import all layers from the layers module
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

# Import attention
from .attention import (
    MultiHeadYatAttention,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
)

# Import embedding
from .embed import YatEmbed

# Import squashers
from .squashers import softermax, softer_sigmoid, soft_tanh


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
    # YAT Attention
    "MultiHeadYatAttention",
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


