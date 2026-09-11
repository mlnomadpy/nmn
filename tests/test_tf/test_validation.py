"""TensorFlow constructor and functional validation matrix."""

import math

import pytest

tf = pytest.importorskip("tensorflow")

from nmn.tf import (
    MultiHeadYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)
from nmn.tf.attention import yat_attention_weights


@pytest.mark.parametrize(
    "layer",
    [
        YatConv1D,
        YatConv2D,
        YatConv3D,
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    ],
)
def test_all_conv_families_reject_invalid_feature_counts(layer):
    with pytest.raises(ValueError, match="filters must be a positive integer"):
        layer(filters=0, kernel_size=1)


def test_attention_validates_before_allocation():
    with pytest.raises(ValueError, match="num_heads must be a positive integer"):
        MultiHeadYatAttention(8, 0)
    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(8, 2, dropout=math.nan)
    with pytest.raises(ValueError, match="dropout_rate must be a finite real number"):
        yat_attention_weights(None, None, dropout_rate=1.0)


def test_rate_boundaries_remain_supported():
    MultiHeadYatAttention(2, 1, dropout=0.0)
    MultiHeadYatAttention(2, 1, dropout=0.999999)


@pytest.mark.parametrize(
    "rate", [tf.constant(False), tf.constant("0.5"), tf.constant(0.5 + 0j)]
)
def test_non_real_tensors_are_not_accepted_as_rates(rate):
    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(2, 1, dropout=rate)
