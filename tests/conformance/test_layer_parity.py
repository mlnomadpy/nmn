"""Canonical lookup and 1D convolution parity across executable adapters."""

from __future__ import annotations

import os

import pytest

from nmn.conformance import load_contract
from tests.conformance.harness import compare, load_adapter
from tests.conformance.oracle import (
    canonical_convolution_case,
    canonical_embedding_attend_case,
    canonical_embedding_case,
    canonical_transpose_convolution_case,
    yat_conv1d,
    yat_conv_transpose1d,
    yat_embedding,
    yat_embedding_attend,
)

CONTRACT = load_contract()
TOLERANCE = CONTRACT["tolerances"]["float32"]


def _cases(operation):
    for backend_name, backend in CONTRACT["backends"].items():
        profile = CONTRACT["profiles"][backend_name][operation]
        for mode in profile["modes"]["tested"]:
            yield pytest.param(
                backend_name,
                backend,
                mode,
                id=f"{backend_name}-{mode}",
            )


def _adapter_or_skip(backend_name, backend):
    adapter = load_adapter(backend["adapter"])
    if adapter.available():
        return adapter
    platform = os.environ.get("NMN_CONFORMANCE_PLATFORM")
    if backend["required_in_ci"] and backend["ci_platform"] == platform:
        pytest.fail(f"required {platform} backend is unavailable: {backend_name}")
    pytest.skip(f"backend unavailable on this platform: {backend_name}")
    raise AssertionError("pytest.skip unexpectedly returned")


def _convolution_cases():
    operations = (
        ("convolution", False, canonical_convolution_case, yat_conv1d),
        (
            "transpose_convolution",
            True,
            canonical_transpose_convolution_case,
            yat_conv_transpose1d,
        ),
    )
    for operation, transpose, case_factory, oracle in operations:
        for backend_name, backend in CONTRACT["backends"].items():
            profile = CONTRACT["profiles"][backend_name][operation]
            for mode in profile["modes"]["tested"]:
                yield pytest.param(
                    backend_name,
                    backend,
                    mode,
                    transpose,
                    case_factory,
                    oracle,
                    id=f"{operation}-{backend_name}-{mode}",
                )


@pytest.mark.parametrize("backend,backend_contract,mode", list(_cases("embed")))
def test_embedding_lookup_and_embedding_gradient_match_float64_oracle(
    backend, backend_contract, mode
):
    adapter = _adapter_or_skip(backend, backend_contract)
    expected = yat_embedding(canonical_embedding_case())
    actual = adapter.embedding_value_and_grad(
        canonical_embedding_case(), compiled=mode != "eager"
    )
    compare(
        f"{backend}:embedding-output",
        actual.output,
        expected.output,
        expected_dtype="float32",
        **TOLERANCE,
    )
    assert set(actual.gradients) == {"embedding"}
    compare(
        f"{backend}:embedding-gradient",
        actual.gradients["embedding"],
        expected.gradients["embedding"],
        expected_dtype="float32",
        **TOLERANCE,
    )


@pytest.mark.parametrize("backend,backend_contract,mode", list(_cases("embed")))
def test_embedding_attend_outputs_and_all_supported_gradients_match_float64_oracle(
    backend, backend_contract, mode
):
    adapter = _adapter_or_skip(backend, backend_contract)
    case = canonical_embedding_attend_case()
    expected = yat_embedding_attend(case)
    actual = adapter.embedding_attend_value_and_grad(case, compiled=mode != "eager")
    compare(
        f"{backend}:embed-attend-output",
        actual.output,
        expected.output,
        expected_dtype="float32",
        **TOLERANCE,
    )
    assert set(actual.gradients) == set(expected.gradients)
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{backend}:embed-attend-{name}",
            actual.gradients[name],
            expected_gradient,
            expected_dtype="float32",
            **TOLERANCE,
        )
    compare(
        f"{backend}:embedding-gradient",
        actual.gradients["embedding"],
        expected.gradients["embedding"],
        expected_dtype="float32",
        **TOLERANCE,
    )


@pytest.mark.parametrize(
    "backend,backend_contract,mode,transpose,case_factory,oracle",
    list(_convolution_cases()),
)
def test_convolution_outputs_and_all_gradients_match_float64_oracle(
    backend, backend_contract, mode, transpose, case_factory, oracle
):
    adapter = _adapter_or_skip(backend, backend_contract)
    case = case_factory()
    expected = oracle(case)
    actual = adapter.convolution_value_and_grad(
        case, transpose=transpose, compiled=mode != "eager"
    )
    compare(
        f"{backend}:conv-output",
        actual.output,
        expected.output,
        expected_dtype="float32",
        **TOLERANCE,
    )
    assert set(actual.gradients) == set(expected.gradients)
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{backend}:conv-{name}",
            actual.gradients[name],
            expected_gradient,
            expected_dtype="float32",
            **TOLERANCE,
        )
