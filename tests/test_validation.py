"""Framework-independent scalar validation contract."""

import math

import numpy as np
import pytest

from nmn._validation import validate_positive_int, validate_rate


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        -0.1,
        1.0,
        2.0,
        True,
        False,
        "0.5",
        b"0.5",
        [0.5],
        (0.5,),
        np.array([0.5]),
        np.array(False),
        np.array("0.5"),
        np.array(b"0.5"),
        np.complex64(0.5 + 1j),
        np.array(0.5 + 0j),
    ],
)
def test_invalid_rates_have_one_diagnostic(value):
    with pytest.raises(
        ValueError, match=r"rate must be a finite real number in \[0, 1\)"
    ):
        validate_rate(value, "rate")


@pytest.mark.parametrize("value", [0.0, 0.5, 0.999999, np.array(0.5)])
def test_valid_rates_are_preserved(value):
    assert validate_rate(value, "rate") == float(value)


@pytest.mark.parametrize("value", [0, -1, 1.0, True, "1"])
def test_positive_integer_contract(value):
    with pytest.raises(ValueError, match="count must be a positive integer"):
        validate_positive_int(value, "count")
