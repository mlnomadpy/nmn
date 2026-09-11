"""Dependency-light adapter loading and conformance comparison."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tests.conformance.oracle import (
    AttentionCase,
    AttentionResult,
    DenseCase,
    DenseConfiguration,
    LinearAttentionCase,
    LinearAttentionResult,
    OracleResult,
)


class Adapter(Protocol):
    """Thin backend boundary used by generated conformance cases."""

    @staticmethod
    def available() -> bool:
        """Return whether the backend runtime can be used."""
        raise NotImplementedError

    @staticmethod
    def dense(
        case: DenseCase,
        *,
        compiled: bool = False,
        configuration: DenseConfiguration | None = None,
    ) -> np.ndarray:
        """Run the canonical dense case."""
        raise NotImplementedError

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        """Run dense and differentiate its canonical cotangent projection."""
        raise NotImplementedError

    @staticmethod
    def attention_value_and_grad(
        case: AttentionCase, *, compiled: bool = False
    ) -> AttentionResult:
        """Run masked attention and differentiate its canonical projection."""
        raise NotImplementedError

    @staticmethod
    def linear_attention_value_and_grad(
        case: LinearAttentionCase, *, compiled: bool = False
    ) -> LinearAttentionResult:
        """Run fixed-projection MAY/RAY and differentiate q/k/v only."""
        raise NotImplementedError


@dataclass(frozen=True)
class Comparison:
    """Measured difference between a backend and the canonical oracle."""

    backend: str
    max_absolute_error: float
    max_relative_error: float
    actual_dtype: str


def load_adapter(reference: str) -> type[Adapter]:
    """Load ``package.module:Class`` only when a backend is exercised."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"invalid adapter reference: {reference!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def compare(
    backend: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
    expected_dtype: str | None = None,
) -> Comparison:
    """Assert parity and return stable error metrics for diagnostics."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{backend} shape mismatch: actual {actual_array.shape}, "
            f"expected {expected_array.shape}"
        )
    if expected_dtype is not None and actual_array.dtype.name != expected_dtype:
        raise AssertionError(
            f"{backend} dtype mismatch: actual {actual_array.dtype.name}, "
            f"expected {expected_dtype}"
        )
    actual64 = actual_array.astype(np.float64, copy=False)
    expected64 = expected_array.astype(np.float64, copy=False)
    np.testing.assert_allclose(actual64, expected64, rtol=rtol, atol=atol)
    absolute = np.abs(actual64 - expected64)
    relative = absolute / np.maximum(np.abs(expected64), np.finfo(np.float64).tiny)
    return Comparison(
        backend,
        float(np.max(absolute)),
        float(np.max(relative)),
        actual_array.dtype.name,
    )
