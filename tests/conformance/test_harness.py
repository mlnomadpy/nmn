"""Fail-closed checks for conformance comparison mechanics."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conformance.harness import compare


def test_compare_rejects_broadcastable_shape_mismatch():
    with pytest.raises(AssertionError, match="shape mismatch"):
        compare("example", np.asarray(1.0), np.ones((2, 3)), rtol=0.0, atol=0.0)


def test_compare_rejects_wrong_backend_dtype():
    with pytest.raises(AssertionError, match="dtype mismatch"):
        compare(
            "example",
            np.ones((2,), dtype=np.float64),
            np.ones((2,), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            expected_dtype="float32",
        )


def test_compare_reports_exact_shape_dtype_and_errors():
    result = compare(
        "example",
        np.ones((2,), dtype=np.float32),
        np.ones((2,), dtype=np.float64),
        rtol=0.0,
        atol=0.0,
        expected_dtype="float32",
    )
    assert result.actual_dtype == "float32"
    assert result.max_absolute_error == 0.0
    assert result.max_relative_error == 0.0
