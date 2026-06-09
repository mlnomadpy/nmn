"""MLX backend for Neural Matter Network (NMN).

Apple-Silicon-native implementation of the YAT family of layers, mirroring
the surface of ``nmn.tf`` / ``nmn.keras``. Requires ``mlx``.
"""

from .nmn import YatNMN, YatDense
from .embed import YatEmbed
from .attention import (
    MultiHeadYatAttention,
    normalize_qk,
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
)
from .rotary import (
    RotaryYatAttention,
    precompute_freqs_cis,
    apply_rotary_emb,
    rotary_yat_attention,
    rotary_yat_attention_weights,
)
from .performer import (
    create_yat_tp_projection,
    yat_tp_features,
    yat_tp_attention,
)
from .may import (
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
)
from .ray import (
    create_radial_projection,
    radial_features,
    radial_yat_attention,
)
from .goat import (
    GoatYatAttention,
    goat_yat_attention,
    goat_yat_attention_weights,
)
from .fused import fused_yat_score, is_gpu_available
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
    "YatNMN", "YatDense",
    "YatEmbed",
    "MultiHeadYatAttention",
    "normalize_qk",
    "yat_attention",
    "yat_attention_weights",
    "yat_attention_normalized",
    "RotaryYatAttention",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "rotary_yat_attention",
    "rotary_yat_attention_weights",
    "create_yat_tp_projection",
    "yat_tp_features",
    "yat_tp_attention",
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
    "GoatYatAttention",
    "goat_yat_attention",
    "goat_yat_attention_weights",
    "fused_yat_score",
    "is_gpu_available",
    "softermax", "softer_sigmoid", "soft_tanh",
] + _conv_all
