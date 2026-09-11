"""Contract-generated dense forward parity across thin backend adapters."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nmn.conformance import load_contract
from tests.conformance.harness import compare, load_adapter
from tests.conformance.oracle import (
    DenseConfiguration,
    canonical_dense_case,
    read_dense_fixture,
    yat_dense,
    yat_dense_configured,
)

CONTRACT = load_contract()

CONFIGURATIONS = [
    DenseConfiguration(spherical, weight_normalized, learnable_epsilon, bias_mode)
    for spherical in (False, True)
    for weight_normalized in (False, True)
    for learnable_epsilon in (False, True)
    for bias_mode in ("learnable", "constant", "none")
]


def _cases():
    include_all_modes = os.environ.get("NMN_CONFORMANCE_ALL_MODES") == "1"
    for backend_name, backend in CONTRACT["backends"].items():
        modes = backend["execution"] if include_all_modes else ["eager"]
        for mode in modes:
            yield pytest.param(
                backend_name,
                backend,
                mode,
                id=f"{backend_name}-{mode}",
            )


@pytest.mark.parametrize("backend_name,backend,mode", list(_cases()))
def test_dense_forward_matches_float64_oracle(backend_name, backend, mode):
    adapter = load_adapter(backend["adapter"])
    if not adapter.available():
        platform = os.environ.get("NMN_CONFORMANCE_PLATFORM")
        required_here = backend["required_in_ci"] and backend["ci_platform"] == platform
        if required_here:
            pytest.fail(f"required {platform} backend is unavailable: {backend_name}")
        pytest.skip(f"optional backend unavailable: {backend_name}")

    fixture_path = os.environ.get("NMN_CONFORMANCE_FIXTURE")
    if backend_name == "mlx" and fixture_path:
        case, expected = read_dense_fixture(Path(fixture_path))
    else:
        case = canonical_dense_case(CONTRACT["canonical"]["random_seed"])
        expected = yat_dense(case)
    actual = adapter.dense(case, compiled=mode != "eager")
    tolerance = CONTRACT["tolerances"]["float32"]
    compare(
        backend_name,
        actual,
        expected.output,
        expected_dtype="float32",
        **tolerance,
    )


@pytest.mark.parametrize("backend_name,backend,mode", list(_cases()))
def test_dense_output_and_all_gradients_match_float64_oracle(
    backend_name, backend, mode
):
    adapter = load_adapter(backend["adapter"])
    if not adapter.available():
        platform = os.environ.get("NMN_CONFORMANCE_PLATFORM")
        required_here = backend["required_in_ci"] and backend["ci_platform"] == platform
        if required_here:
            pytest.fail(f"required {platform} backend is unavailable: {backend_name}")
        pytest.skip(f"optional backend unavailable: {backend_name}")

    fixture_path = os.environ.get("NMN_CONFORMANCE_FIXTURE")
    if backend_name == "mlx" and fixture_path:
        case, expected = read_dense_fixture(Path(fixture_path))
    else:
        case = canonical_dense_case(CONTRACT["canonical"]["random_seed"])
        expected = yat_dense(case)
    actual = adapter.dense_value_and_grad(case, compiled=mode != "eager")
    tolerance = CONTRACT["tolerances"]["float32"]
    compare(
        backend_name,
        actual.output,
        expected.output,
        expected_dtype="float32",
        **tolerance,
    )
    assert set(actual.gradients) == set(expected.gradients)
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{backend_name}:{name}",
            actual.gradients[name],
            expected_gradient,
            expected_dtype="float32",
            **tolerance,
        )


@pytest.mark.parametrize(
    "configuration",
    CONFIGURATIONS,
    ids=lambda item: (
        f"spherical={item.spherical}-normalized={item.weight_normalized}-"
        f"learnable-epsilon={item.learnable_epsilon}-bias={item.bias_mode}"
    ),
)
@pytest.mark.parametrize(
    "backend_name,backend",
    [
        pytest.param(name, backend, id=name)
        for name, backend in CONTRACT["backends"].items()
    ],
)
def test_dense_configuration_matrix_matches_float64_oracle(
    backend_name, backend, configuration
):
    """Preserve every unique configuration from the retired adapter suites."""
    adapter = load_adapter(backend["adapter"])
    if not adapter.available():
        platform = os.environ.get("NMN_CONFORMANCE_PLATFORM")
        required_here = backend["required_in_ci"] and backend["ci_platform"] == platform
        if required_here:
            pytest.fail(f"required {platform} backend is unavailable: {backend_name}")
        pytest.skip(f"optional backend unavailable: {backend_name}")

    case = canonical_dense_case(CONTRACT["canonical"]["random_seed"])
    actual = adapter.dense(case, configuration=configuration)
    expected = yat_dense_configured(case, configuration)
    assert np.all(np.isfinite(actual)), f"{backend_name} produced a non-finite value"
    compare(
        f"{backend_name}:{configuration}",
        actual,
        expected,
        expected_dtype="float32",
        **CONTRACT["tolerances"]["float32"],
    )


@pytest.mark.parametrize("stress", ["large", "small", "matching"])
@pytest.mark.parametrize(
    "backend_name,backend",
    [
        pytest.param(name, backend, id=name)
        for name, backend in CONTRACT["backends"].items()
    ],
)
def test_dense_numerical_stress_cases_match_oracle(backend_name, backend, stress):
    """Retain the non-degenerate stability cases from the legacy suites."""
    adapter = load_adapter(backend["adapter"])
    if not adapter.available():
        platform = os.environ.get("NMN_CONFORMANCE_PLATFORM")
        required_here = backend["required_in_ci"] and backend["ci_platform"] == platform
        if required_here:
            pytest.fail(f"required {platform} backend is unavailable: {backend_name}")
        pytest.skip(f"optional backend unavailable: {backend_name}")

    case = canonical_dense_case(CONTRACT["canonical"]["random_seed"])
    if stress == "large":
        case = replace(case, inputs=case.inputs * 1_000.0)
    elif stress == "small":
        case = replace(case, inputs=case.inputs * 1e-6)
    else:
        case = replace(
            case,
            inputs=case.kernel[:, :1].T,
            cotangent=case.cotangent[:1],
        )

    actual = adapter.dense(case)
    expected = yat_dense(case).output
    assert np.all(np.isfinite(actual)), f"{backend_name} produced a non-finite value"
    compare(
        f"{backend_name}:{stress}",
        actual,
        expected,
        expected_dtype="float32",
        **CONTRACT["tolerances"]["float32"],
    )


def test_every_adapter_reference_is_loadable_without_importing_its_framework():
    for backend in CONTRACT["backends"].values():
        adapter = load_adapter(backend["adapter"])
        assert callable(adapter.available)
        assert callable(adapter.dense)
        assert callable(adapter.dense_value_and_grad)


def test_every_declared_api_symbol_exists_when_backend_is_available():
    for backend_name, backend in CONTRACT["backends"].items():
        adapter = load_adapter(backend["adapter"])
        if not adapter.available():
            continue
        module = __import__(f"nmn.{backend_name}", fromlist=["nmn"])
        for capability in backend["operations"].values():
            for api_name in capability["api"].split("|"):
                assert hasattr(module, api_name.rsplit(".", 1)[1]), api_name
