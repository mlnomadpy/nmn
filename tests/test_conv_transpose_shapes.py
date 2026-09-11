"""Framework-independent tests for the ConvTranspose shape contract."""

import pytest

from nmn._conv_transpose import (
    canonical_jax_transpose_padding,
    canonical_same_crop_or_pad,
    canonical_transpose_output_spatial,
)


@pytest.mark.parametrize(
    "spatial,kernel,strides,padding,dilation,extra,expected",
    [
        ((3,), (2,), (3,), "VALID", (1,), (0,), (8,)),
        ((3,), (2,), (3,), "SAME", (1,), (0,), (9,)),
        ((3,), (2,), (3,), "SAME", (1,), (1,), (10,)),
        ((2, 3), (2, 3), (3, 2), "VALID", (1, 1), (0, 0), (5, 7)),
        (
            (2, 2, 2),
            (2, 3, 2),
            (3, 2, 4),
            "VALID",
            (1, 2, 1),
            (0, 1, 2),
            (5, 8, 8),
        ),
    ],
)
def test_canonical_spatial_formula(
    spatial, kernel, strides, padding, dilation, extra, expected
):
    assert (
        canonical_transpose_output_spatial(
            spatial, kernel, strides, padding, dilation, extra
        )
        == expected
    )


def test_jax_padding_distinguishes_legacy_gap_from_explicit_zero():
    assert canonical_jax_transpose_padding((2,), (3,), "VALID") == ((1, 1),)
    assert canonical_jax_transpose_padding((2,), (3,), "SAME", output_padding=1) == (
        (1, 3),
    )


def test_same_adjustment_uncrops_values_before_it_zero_pads_stride_gaps():
    assert canonical_same_crop_or_pad(3, 2, output_padding=0) == ((0, 1),)
    assert canonical_same_crop_or_pad(3, 2, output_padding=1) == ((0, 0),)
    assert canonical_same_crop_or_pad(2, 3, output_padding=0) == ((0, -1),)
    assert canonical_same_crop_or_pad(2, 3, output_padding=1) == ((0, -2),)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"strides": 0}, "strides values must be positive"),
        ({"output_padding": -1}, "output_padding values must be nonnegative"),
        ({"output_padding": 2}, "smaller than the corresponding stride"),
        ({"padding": "causal"}, "padding must be"),
    ],
)
def test_contract_rejects_nonportable_configuration(kwargs, match):
    arguments = dict(
        input_spatial=(3,),
        kernel_size=2,
        strides=2,
        padding="valid",
        dilation_rate=1,
        output_padding=0,
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        canonical_transpose_output_spatial(**arguments)
