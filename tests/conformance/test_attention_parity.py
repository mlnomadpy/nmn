"""Canonical masked-attention output and gradient parity."""

from __future__ import annotations

import os

import pytest

from nmn.conformance import load_contract
from tests.conformance.harness import compare, load_adapter
from tests.conformance.oracle import canonical_attention_case, yat_attention

CONTRACT = load_contract()


def _cases():
    for backend_name, backend in CONTRACT["backends"].items():
        profile = CONTRACT["profiles"][backend_name]["attention"]
        for mode in profile["modes"]["tested"]:
            yield pytest.param(backend_name, backend, mode, id=f"{backend_name}-{mode}")


@pytest.mark.parametrize("backend_name,backend,mode", list(_cases()))
def test_attention_outputs_masks_and_gradients_match_oracle(
    backend_name, backend, mode
):
    adapter = load_adapter(backend["adapter"])
    if not adapter.available():
        platform = os.environ.get("NMN_CONFORMANCE_PLATFORM")
        if backend["required_in_ci"] and backend["ci_platform"] == platform:
            pytest.fail(f"required {platform} backend is unavailable: {backend_name}")
        pytest.skip(f"backend unavailable on this platform: {backend_name}")
    case = canonical_attention_case(CONTRACT["canonical"]["random_seed"])
    expected = yat_attention(case)
    actual = adapter.attention_value_and_grad(case, compiled=mode != "eager")
    tolerance = CONTRACT["tolerances"]["float32"]
    compare(
        f"{backend_name}:weights",
        actual.weights,
        expected.weights,
        expected_dtype="float32",
        **tolerance,
    )
    compare(
        f"{backend_name}:output",
        actual.output,
        expected.output,
        expected_dtype="float32",
        **tolerance,
    )
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{backend_name}:{name}",
            actual.gradients[name],
            expected_gradient,
            expected_dtype="float32",
            **tolerance,
        )
