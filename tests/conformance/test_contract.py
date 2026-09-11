"""Schema, documentation, and dependency-boundary contract tests."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nmn.conformance import ContractError, load_contract, render_support_markdown
from tests.conformance.oracle import canonical_linear_attention_case

ROOT = Path(__file__).resolve().parents[2]


def test_packaged_contract_is_valid_json_and_complete():
    contract = load_contract()
    assert contract["schema_version"] == 1
    assert len(contract["backends"]) == 6
    assert len(contract["operations"]) == 7
    assert all(
        set(backend["operations"]) == set(contract["operations"])
        for backend in contract["backends"].values()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("tolerances"), "missing required keys"),
        (lambda value: value.__setitem__("schema_version", 2), "must be 1"),
        (
            lambda value: value["backends"]["torch"]["operations"].pop("ray"),
            "must declare every operation",
        ),
        (
            lambda value: value["backends"]["mlx"]["operations"]["dense"].__setitem__(
                "status", "maybe"
            ),
            "status is invalid",
        ),
        (
            lambda value: value["profiles"]["torch"]["dense"].__setitem__(
                "dtypes", "float32"
            ),
            "dtypes must be an object",
        ),
        (
            lambda value: value["profiles"]["torch"]["dense"].__setitem__("config", {}),
            "config must not be empty",
        ),
        (
            lambda value: (
                value["profiles"]["torch"]["dense"]["dtypes"].__setitem__(
                    "supported", []
                ),
                value["profiles"]["torch"]["dense"]["dtypes"].__setitem__("tested", []),
            ),
            "supported capability must name supported dtypes and modes",
        ),
        (
            lambda value: (
                value["profiles"]["mlx"]["transpose_convolution"]["modes"].__setitem__(
                    "supported", []
                ),
                value["profiles"]["mlx"]["transpose_convolution"]["modes"].__setitem__(
                    "tested", []
                ),
            ),
            "partial capability must name supported dtypes and modes",
        ),
        (
            lambda value: value["profiles"]["torch"]["dense"]["evidence"].__setitem__(
                "tests", []
            ),
            "verified evidence must name tested dtypes, modes, and tests",
        ),
        (
            lambda value: (
                value["backends"]["torch"]["operations"]["dense"].__setitem__(
                    "conformance", "declared"
                ),
                value["profiles"]["torch"]["dense"]["dtypes"].__setitem__("tested", []),
                value["profiles"]["torch"]["dense"]["modes"].__setitem__("tested", []),
                value["profiles"]["torch"]["dense"]["evidence"].update(
                    kind="declared", tests=[], fixture=None
                ),
            ),
            "supported capability must use oracle or fixture evidence",
        ),
        (
            lambda value: (
                value["backends"]["tf"]["operations"]["may"].__setitem__(
                    "conformance", "declared"
                ),
                value["profiles"]["tf"]["may"]["evidence"].__setitem__(
                    "kind", "declared"
                ),
            ),
            "declared evidence must not claim test coverage",
        ),
        (
            lambda value: value["backends"]["torch"]["operations"]["dense"].__setitem__(
                "status", "unsupported"
            ),
            "unsupported capability must not claim support or tests",
        ),
        (lambda value: value.__setitem__("typo", True), "unexpected keys"),
        (
            lambda value: value["backends"]["torch"].__setitem__("typo", True),
            "unexpected keys",
        ),
        (
            lambda value: value["backends"]["torch"]["operations"]["dense"].__setitem__(
                "typo", True
            ),
            "unexpected keys",
        ),
        (
            lambda value: value["backends"]["torch"]["operations"][
                "convolution"
            ].__setitem__("declared_api", "nmn.keras.YatConv2D"),
            "declared_api must contain fully qualified nmn.torch symbols",
        ),
        (
            lambda value: value["backends"]["torch"]["operations"][
                "convolution"
            ].__setitem__("declared_api", "nmn.torch.YatConv1D"),
            "api and declared_api must not overlap",
        ),
        (
            lambda value: value["backends"]["torch"]["operations"]["convolution"].pop(
                "declared_api"
            ),
            "missing required keys: declared_api",
        ),
        (
            lambda value: value["backends"]["nnx"]["operations"][
                "transpose_convolution"
            ].pop("declared_api"),
            "missing required keys: declared_api",
        ),
        (
            lambda value: value["profiles"]["torch"]["convolution"][
                "config"
            ].__setitem__("spatial_rank", 2),
            "spatial_rank must be 1 for 'convolution_1d_valid_v1'",
        ),
        (
            lambda value: value["profiles"]["mlx"]["transpose_convolution"][
                "config"
            ].__setitem__("spatial_rank", 3),
            "spatial_rank must be 1 for 'transpose_convolution_1d_valid_v1'",
        ),
        (
            lambda value: value["profiles"]["torch"]["dense"].__setitem__("typo", True),
            "unexpected keys",
        ),
        (
            lambda value: value["profiles"]["torch"]["dense"]["evidence"].__setitem__(
                "typo", True
            ),
            "unexpected keys",
        ),
        (
            lambda value: value["profiles"]["torch"]["dense"]["config"].__setitem__(
                "typo", True
            ),
            "unexpected keys",
        ),
    ],
)
def test_contract_validator_fails_closed(mutation, message):
    from nmn.conformance import validate_contract

    contract = copy.deepcopy(load_contract())
    mutation(contract)
    with pytest.raises(ContractError, match=message):
        validate_contract(contract)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("key", ["rtol", "atol"])
def test_contract_validator_rejects_nonfinite_tolerances(value, key):
    from nmn.conformance import validate_contract

    contract = copy.deepcopy(load_contract())
    contract["tolerances"]["float32"][key] = value
    with pytest.raises(ContractError, match="must be finite and nonnegative"):
        validate_contract(contract)


def test_every_capability_has_an_exact_profile_cell():
    contract = load_contract()
    for backend_name, backend in contract["backends"].items():
        profiles = contract["profiles"][backend_name]
        assert set(profiles) == set(backend["operations"])
        for operation_name, profile in profiles.items():
            assert set(profile) == {
                "dtypes",
                "modes",
                "keras_engine",
                "config",
                "layout",
                "masking",
                "serialization",
                "evidence",
            }
            assert profile["dtypes"]["supported"]
            assert profile["modes"]["supported"]
            assert profile["config"]
            assert profile["serialization"]
            if backend["operations"][operation_name]["conformance"] == "declared":
                assert profile["dtypes"]["tested"] == []
                assert profile["modes"]["tested"] == []
                assert profile["evidence"]["tests"] == []


def test_linear_attention_profiles_exactly_match_canonical_projection_shapes():
    contract = load_contract()
    for operation in ("may", "ray"):
        case = canonical_linear_attention_case(operation)
        for backend_name in contract["backends"]:
            config = contract["profiles"][backend_name][operation]["config"]
            assert config["num_heads"] == case.query.shape[-2]
            assert config["head_features"] == case.query.shape[-1]
            if operation == "may":
                assert config["num_features"] == case.projection["num_features"]
            else:
                for key in ("sketch_m", "num_radial", "radial_dim"):
                    assert config[key] == case.projection[key]


def test_attention_capabilities_name_the_exact_functional_api_under_test():
    contract = load_contract()
    for backend_name, backend in contract["backends"].items():
        assert backend["operations"]["attention"]["api"].split("|") == [
            f"nmn.{backend_name}.yat_attention",
            f"nmn.{backend_name}.yat_attention_weights",
        ]


def test_convolution_oracle_claims_are_scoped_to_the_tested_spatial_rank():
    contract = load_contract()
    for backend_name, backend in contract["backends"].items():
        for operation, stem in (
            ("convolution", "YatConv"),
            ("transpose_convolution", "YatConvTranspose"),
        ):
            capability = backend["operations"][operation]
            assert capability["api"] == f"nmn.{backend_name}.{stem}1D"
            assert capability["declared_api"].split("|") == [
                f"nmn.{backend_name}.{stem}2D",
                f"nmn.{backend_name}.{stem}3D",
            ]
            assert (
                contract["profiles"][backend_name][operation]["config"]["spatial_rank"]
                == 1
            )


@pytest.mark.parametrize("operation", ["convolution", "transpose_convolution"])
def test_renderer_fails_closed_without_convolution_declared_api(operation):
    contract = copy.deepcopy(load_contract())
    contract["backends"]["torch"]["operations"][operation].pop("declared_api")

    with pytest.raises(ContractError, match="missing required keys: declared_api"):
        render_support_markdown(contract)


def test_apple_fixture_evidence_names_the_ci_generated_artifact():
    contract = load_contract()
    assert (
        contract["profiles"]["mlx"]["dense"]["evidence"]["fixture"]
        == "dense-v1.npz (generated by Linux CI artifact)"
    )


def test_contract_loader_imports_no_optional_framework():
    program = """
import json, sys
from nmn.conformance import load_contract
load_contract()
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {
    'torch', 'tensorflow', 'keras', 'jax', 'flax', 'mlx'
})))
"""
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source, environment.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize("engine", ["jax", "torch"])
def test_keras_adapter_does_not_claim_unimplemented_engines(engine):
    program = """
from tests.conformance.adapters.keras import KerasAdapter
assert KerasAdapter.available() is False
"""
    environment = os.environ.copy()
    environment["KERAS_BACKEND"] = engine
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), source, environment.get("PYTHONPATH", "")) if part
    )
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_generated_support_document_is_current():
    expected = (ROOT / "docs" / "generated" / "conformance.md").read_text()
    assert expected == render_support_markdown()
    website_copy = (
        ROOT / "website" / "docusaurus" / "docs" / "conformance.md"
    ).read_text()
    assert website_copy == expected
