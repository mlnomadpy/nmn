"""Neural-Matter Network (NMN) - beyond blinded neurons."""

try:
    from nmn._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # Framework subpackages — import the one you need:
    #   from nmn.torch import YatNMN, YatConv2D, ...
    #   from nmn.nnx import YatNMN, YatConv, ...
    #   from nmn.keras import YatNMN, YatConv2D, ...
    #   from nmn.tf import YatNMN, YatConv2D, ...
    #   from nmn.linen import YatNMN, YatConv2D, ...
    "torch",
    "nnx",
    "keras",
    "tf",
    "linen",
]
