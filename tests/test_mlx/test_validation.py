"""MLX constructor and functional validation matrix."""

import math

import pytest

pytest.importorskip("mlx.core")

from nmn.mlx import (
    MultiHeadYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)
from nmn.mlx.attention import yat_attention_weights

CONVS = [
    (YatConv1D, 1),
    (YatConv2D, (1, 1)),
    (YatConv3D, (1, 1, 1)),
    (YatConvTranspose1D, 1),
    (YatConvTranspose2D, (1, 1)),
    (YatConvTranspose3D, (1, 1, 1)),
]


@pytest.mark.parametrize(("layer", "kernel_size"), CONVS)
@pytest.mark.parametrize("rate", [math.nan, math.inf, -0.1, 1.0, 2.0])
def test_all_conv_families_reject_invalid_dropconnect(layer, kernel_size, rate):
    with pytest.raises(ValueError, match="drop_rate must be a finite real number"):
        layer(2, kernel_size, drop_rate=rate)


def test_attention_validates_before_allocation():
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        MultiHeadYatAttention(8, 0)
    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(8, 2, dropout=math.nan)
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        yat_attention_weights(None, None, dropout_rate=1.0)


def test_rate_boundaries_remain_supported():
    YatConv1D(1, 1, drop_rate=0.0)
    MultiHeadYatAttention(2, 1, dropout=0.999999)


def test_boolean_array_is_not_accepted_as_a_rate():
    import mlx.core as mx

    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(2, 1, dropout=mx.array(False))


def test_complex_array_is_not_accepted_as_a_rate():
    import mlx.core as mx

    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(2, 1, dropout=mx.array(0.5 + 1j))
