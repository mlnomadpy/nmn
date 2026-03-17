"""Flax Linen backend for Neural Matter Network (NMN)."""

from .nmn import YatNMN

try:
    from .conv import (
        YatConv1D, YatConv2D, YatConv3D,
        YatConv1d, YatConv2d, YatConv3d,
        YatConvTranspose1D, YatConvTranspose2D, YatConvTranspose3D,
        YatConvTranspose1d, YatConvTranspose2d, YatConvTranspose3d,
    )
    __all__ = [
        "YatNMN",
        "YatConv1D", "YatConv2D", "YatConv3D",
        "YatConv1d", "YatConv2d", "YatConv3d",
        "YatConvTranspose1D", "YatConvTranspose2D", "YatConvTranspose3D",
        "YatConvTranspose1d", "YatConvTranspose2d", "YatConvTranspose3d",
    ]
except ImportError:
    __all__ = ["YatNMN"]

