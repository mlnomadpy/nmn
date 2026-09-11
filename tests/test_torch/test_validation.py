"""PyTorch constructor and functional validation matrix."""

import math

import pytest
import torch

from nmn.torch import (
    MultiHeadYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)
from nmn.torch.attention import yat_attention_weights

CONVS = [
    (YatConv1D, 3),
    (YatConv2D, (3, 3)),
    (YatConv3D, (3, 3, 3)),
    (YatConvTranspose1D, 3),
    (YatConvTranspose2D, (3, 3)),
    (YatConvTranspose3D, (3, 3, 3)),
]
INVALID_RATES = [math.nan, math.inf, -0.1, 1.0, 2.0]


@pytest.mark.parametrize(("layer", "kernel_size"), CONVS)
@pytest.mark.parametrize("rate", INVALID_RATES)
def test_all_conv_families_reject_invalid_dropconnect(layer, kernel_size, rate):
    with pytest.raises(ValueError, match="drop_rate must be a finite real number"):
        layer(2, 2, kernel_size, drop_rate=rate)


@pytest.mark.parametrize("rate", INVALID_RATES)
def test_attention_rejects_invalid_dropout(rate):
    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(8, 2, dropout=rate)
    q = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="dropout_p must be a finite real number"):
        yat_attention_weights(q, q, dropout_p=rate)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dim": 0, "num_heads": 1},
        {"embed_dim": 8, "num_heads": 0},
        {"embed_dim": 8, "num_heads": -2},
    ],
)
def test_attention_dimensions_fail_before_modulo(kwargs):
    with pytest.raises(ValueError, match="must be a positive integer"):
        MultiHeadYatAttention(**kwargs)


def test_rate_boundaries_remain_supported():
    YatConv1D(1, 1, 1, drop_rate=0.0)
    MultiHeadYatAttention(2, 1, dropout=0.999999)


def test_boolean_tensor_is_not_accepted_as_a_rate():
    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(2, 1, dropout=torch.tensor(False))


@pytest.mark.parametrize("rate", [torch.tensor(0.5 + 0j), torch.tensor(0.5 + 1j)])
def test_complex_tensors_are_not_accepted_as_rates(rate):
    with pytest.raises(ValueError, match="dropout must be a finite real number"):
        MultiHeadYatAttention(2, 1, dropout=rate)
