"""PyTorch implementation of Neural Matter Network (NMN) layers."""

# Import standard conv layers from conv module
from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    LazyConv1d,
    LazyConv2d,
    LazyConv3d,
    LazyConvTranspose1d,
    LazyConvTranspose2d,
    LazyConvTranspose3d,
)

# Import YAT conv layers from yat_conv module
from .yat_conv import (
    YatConv1d,
    YatConv2d,
    YatConv3d,
    YatConvTranspose1d,
    YatConvTranspose2d,
    YatConvTranspose3d,
)

# Import YatNMN from nmn module
from .nmn import YatNMN


__all__ = [
    # Standard Conv layers
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    # Lazy Conv layers
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    # YAT Conv layers
    "YatConv1d",
    "YatConv2d",
    "YatConv3d",
    "YatConvTranspose1d",
    "YatConvTranspose2d",
    "YatConvTranspose3d",
    # YAT NMN
    "YatNMN",
]
