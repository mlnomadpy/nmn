"""Canonical fixed-projection MAY/RAY feature-map and readout parity."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nmn.conformance import load_contract
from tests.conformance.harness import compare, load_adapter
from tests.conformance.oracle import (
    canonical_linear_attention_case,
    linear_yat_attention,
)

CONTRACT = load_contract()
ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = CONTRACT["tolerances"]["float32"]


def _cases():
    for operation in ("may", "ray"):
        for backend_name, backend in CONTRACT["backends"].items():
            profile = CONTRACT["profiles"][backend_name][operation]
            for mode in profile["modes"]["tested"]:
                yield pytest.param(
                    operation,
                    backend_name,
                    backend,
                    mode,
                    id=f"{operation}-{backend_name}-{mode}",
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


@pytest.mark.parametrize("operation,backend_name,backend,mode", list(_cases()))
def test_fixed_projection_linear_attention_matches_float64_oracle(
    operation, backend_name, backend, mode
):
    """Check both feature maps and the causal q/k/v readout VJP."""
    adapter = _adapter_or_skip(backend_name, backend)
    case = canonical_linear_attention_case(operation)
    expected = linear_yat_attention(case)
    actual = adapter.linear_attention_value_and_grad(case, compiled=mode != "eager")
    for name in ("query_features", "key_features", "output"):
        compare(
            f"{operation}:{backend_name}:{name}",
            getattr(actual, name),
            getattr(expected, name),
            expected_dtype="float32",
            **TOLERANCE,
        )
    assert set(actual.gradients) == {"query", "key", "value"}
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{operation}:{backend_name}:{name}",
            actual.gradients[name],
            expected_gradient,
            expected_dtype="float32",
            **TOLERANCE,
        )


@pytest.mark.parametrize("operation", ["may", "ray"])
@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_nnx_key_padding_matches_the_canonical_factorable_mask(operation, mode):
    """NNX additionally verifies its public factorable key-padding contract."""
    backend = CONTRACT["backends"]["nnx"]
    adapter = _adapter_or_skip("nnx", backend)
    case = canonical_linear_attention_case(operation, key_padding=True)
    expected = linear_yat_attention(case)
    actual = adapter.linear_attention_value_and_grad(case, compiled=mode == "jit")
    compare(
        f"{operation}:nnx:key-padding-output",
        actual.output,
        expected.output,
        expected_dtype="float32",
        **TOLERANCE,
    )
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{operation}:nnx:key-padding-{name}",
            actual.gradients[name],
            expected_gradient,
            expected_dtype="float32",
            **TOLERANCE,
        )


def test_epsilon_semantic_divergences_are_explicit_and_not_tolerance_exceptions():
    """Guard the known public-API mismatch until it receives a compatibility fix.

    TensorFlow feeds public ``epsilon`` into both normalization and readout;
    NNX sign-stabilizes the denominator.  The canonical profile therefore fixes
    epsilon at 1e-6, omits its VJP, and constructs positive denominators.
    """
    tf_source = (ROOT / "src" / "nmn" / "tf" / "performer_yat.py").read_text()
    nnx_source = (
        ROOT / "src" / "nmn" / "nnx" / "layers" / "attention" / "maclaurin_yat.py"
    ).read_text()
    assert (
        "maclaurin_features(query, params, normalize=True, epsilon=epsilon)"
        in tf_source
    )
    assert "den + jnp.sign(den) * epsilon + (den == 0) * epsilon" in nnx_source
    result = linear_yat_attention(canonical_linear_attention_case("may"))
    assert set(result.gradients) == {"query", "key", "value"}
