"""Adversarial checks for the independent float64 dense oracle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tests.conformance.oracle import (
    canonical_attention_case,
    canonical_dense_case,
    read_dense_fixture,
    write_dense_fixture,
    yat_attention,
    yat_dense,
)


def _loss(case):
    return float(np.sum(yat_dense(case).output * case.cotangent))


def test_dense_oracle_gradients_match_centered_finite_differences():
    case = canonical_dense_case()
    result = yat_dense(case)
    step = 1e-6
    for name in ("inputs", "kernel", "bias", "alpha", "epsilon"):
        operand = np.asarray(getattr(case, name), dtype=np.float64)
        numerical = np.zeros_like(operand)
        for index in np.ndindex(operand.shape):
            plus = operand.copy()
            minus = operand.copy()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (
                _loss(replace(case, **{name: plus}))
                - _loss(replace(case, **{name: minus}))
            ) / (2.0 * step)
        gradient_name = {"inputs": "input"}.get(name, name)
        np.testing.assert_allclose(
            result.gradients[gradient_name], numerical, rtol=2e-6, atol=2e-7
        )


def test_canonical_fixture_round_trip_is_pickle_free(tmp_path):
    path = tmp_path / "dense-v1.npz"
    write_dense_fixture(path)
    case, stored = read_dense_fixture(path)
    regenerated = yat_dense(case)
    np.testing.assert_array_equal(stored.output, regenerated.output)
    for name in stored.gradients:
        np.testing.assert_array_equal(
            stored.gradients[name], regenerated.gradients[name]
        )


def test_attention_oracle_mask_simplex_all_masked_and_gradients():
    result = yat_attention(canonical_attention_case())
    np.testing.assert_allclose(result.weights[0, 0, 0].sum(), 1.0)
    np.testing.assert_array_equal(result.weights[0, 0, 1], 0.0)
    np.testing.assert_array_equal(result.output[0, 1], 0.0)
    assert set(result.gradients) == {"query", "key", "value", "alpha", "epsilon"}
    assert all(np.isfinite(value).all() for value in result.gradients.values())
