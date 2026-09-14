"""Exact interval operations at zero crossings and binary float boundaries."""

from fractions import Fraction as F

import pytest

from nmn.research.intervals import RationalInterval as Interval


def test_square_product_and_positive_reciprocal():
    assert Interval(F(-2), F(1)).square().to_list() == ["0", "4"]
    assert (Interval(F(-2), F(1)) * Interval(F(-3), F(4))).to_list() == ["-8", "6"]
    assert Interval(F(2), F(3)).positive_reciprocal().to_list() == ["1/3", "1/2"]
    assert Interval.point(0.1).lower == F.from_float(0.1)
    with pytest.raises(ValueError, match="strictly positive"):
        Interval(F(0), F(1)).positive_reciprocal()
