"""Explicit scope, budget exhaustion and target/protection distinctions."""

import json

import pytest

from nmn.cli import main
from nmn.research.contracts import check, default_contract, validate
from nmn.research.model import default_model


def test_default_contract_target_and_protection():
    data = check(default_model(), default_contract())
    assert data["status"] == "certified-under-assumptions"
    assert data["cases_checked"] == data["cases_total"] == 25
    assert data["target_status"] == "passed"


def test_budget_exhaustion_cannot_pass():
    data = check(default_model(), default_contract(), max_cases=24)
    assert data["status"] == "inconclusive"
    assert not data["coverage_complete"]
    assert data["target_status"] == "not-evaluated"
    assert data["protected_violations_observed"] == 0


def test_failure_before_budget_exhaustion_is_valid_counterexample():
    model = default_model()
    model["protected_leak"] = "1"
    data = check(model, default_contract(), max_cases=6)
    assert data["status"] == "counterexample-found"
    assert not data["coverage_complete"]
    assert data["counterexample"]["violations"] == ["protected tolerance exceeded"]


def test_target_can_fail_while_protection_passes():
    contract = default_contract()
    contract["target"]["minimum_decrease"] = "4"
    data = check(default_model(), contract)
    assert data["status"] == "counterexample-found"
    assert data["protected_violations_observed"] == 0
    assert data["target_status"] == "failed"
    assert data["counterexample"]["target_decrease"] == "7/2"


def test_nonzero_tolerance_boundary_and_arbitrary_gate_pair():
    contract = default_contract()
    contract["inputs"] = {"u": ["1"], "v": ["1"]}
    contract["gate"] = {"baseline": "1/2", "edited": "0"}
    contract["target"]["minimum_decrease"] = "13/10"
    contract["protected_tolerance"] = "13/10"
    model = default_model()
    model["protected_leak"] = "1"
    assert check(model, contract)["status"] == "certified-under-assumptions"
    contract["protected_tolerance"] = "129/100"
    assert check(model, contract)["status"] == "counterexample-found"


def test_no_target_request_is_distinct():
    contract = default_contract()
    contract["target"] = None
    assert check(default_model(), contract)["target_status"] == "not-requested"


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c["inputs"].update(u=[]),
        lambda c: c["inputs"].update(u=["1", "2/2"]),
        lambda c: c["inputs"].update(u=["2"]),
        lambda c: c["inputs"].update(u=["0"]),
        lambda c: c.update(protected_tolerance="-1"),
        lambda c: c.update(protected_tolerance="1e999999999"),
        lambda c: c.update(unrecognized="x"),
    ],
)
def test_invalid_contracts(edit):
    data = default_contract()
    edit(data)
    with pytest.raises(ValueError):
        validate(data)


def test_case_budget_validation():
    for budget in (0, -1, True, 4097):
        with pytest.raises(ValueError):
            check(default_model(), default_contract(), budget)


def test_cli_inconclusive_exit_and_contract_identity(tmp_path, capsys):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(default_contract()))
    assert (
        main(["research", "verify", "--contract", str(path), "--max-cases", "1"]) == 3
    )
    partial = json.loads(capsys.readouterr().out)
    assert main(["research", "verify", "--contract", str(path)]) == 0
    full = json.loads(capsys.readouterr().out)
    assert partial["contract_sha256"] == full["contract_sha256"]
    assert main(["research", "verify", "--max-cases", "1"]) == 2
