"""Keras constructor and functional validation matrix."""

import math

import pytest

from nmn.keras import (
    MultiHeadYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)
from nmn.keras.attention import yat_attention_weights

CONVS = [
    (YatConv1D, 3),
    (YatConv2D, (3, 3)),
    (YatConv3D, (3, 3, 3)),
    (YatConvTranspose1D, 3),
    (YatConvTranspose2D, (3, 3)),
    (YatConvTranspose3D, (3, 3, 3)),
]


@pytest.mark.parametrize(("layer", "kernel_size"), CONVS)
@pytest.mark.parametrize("rate", [math.nan, math.inf, -0.1, 1.0, 2.0])
def test_all_conv_families_reject_invalid_dropconnect(layer, kernel_size, rate):
    with pytest.raises(ValueError, match="drop_rate must be a finite real number"):
        layer(2, kernel_size, drop_rate=rate)


def test_attention_validates_before_build():
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        MultiHeadYatAttention(8, 0)
    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(8, 2, dropout=math.nan)
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        yat_attention_weights(None, None, dropout_rate=1.0)


def test_rate_boundaries_remain_supported():
    YatConv1D(1, 1, drop_rate=0.0)
    MultiHeadYatAttention(2, 1, dropout=0.999999)
