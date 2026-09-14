"""Budget and contract fidelity through save, replay and export."""

import json

import pytest

from nmn.cli import main
from nmn.research.bundles import create_configured
from nmn.research.contract_bundles import create_contract_bundle, export, reproduce
from nmn.research.contracts import default_contract
from nmn.research.model import default_model
from nmn.research.reference import digest


@pytest.mark.parametrize(
    "budget,minimum,status",
    [
        (4096, "7/2", "certified-under-assumptions"),
        (10, "7/2", "inconclusive"),
        (4096, "4", "counterexample-found"),
    ],
)
def test_replay_preserves_scope_and_outcome(tmp_path, budget, minimum, status):
    contract = default_contract()
    contract["target"]["minimum_decrease"] = minimum
    source = tmp_path / "source"
    create_contract_bundle(default_model(), contract, source, budget)
    moved = tmp_path / "moved"
    source.rename(moved)
    result = reproduce(moved)
    assert result["verification_status"] == status
    assert result["case_budget"] == budget
    assert result["cases_checked"] == min(25, budget)
    destination = tmp_path / "vault" / "export"
    export(moved, destination)
    assert reproduce(destination)["verification_status"] == status
    for path in moved.iterdir():
        assert path.read_bytes() == (destination / path.name).read_bytes()


@pytest.mark.parametrize(
    "filename,field,value",
    [("run.json", "max_cases", 4096), ("contract.json", "protected_tolerance", "1")],
)
def test_rehashed_scope_change_rejected(tmp_path, filename, field, value):
    bundle = tmp_path / "bundle"
    create_contract_bundle(default_model(), default_contract(), bundle, 10)
    path = bundle / filename
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][filename] = digest(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recomputed"):
        reproduce(bundle)


def test_cli_incomplete_bundle_is_successful_artifact_creation(tmp_path, capsys):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(default_contract()))
    output = tmp_path / "bundle"
    assert (
        main(
            [
                "research",
                "demo",
                "--contract",
                str(path),
                "--max-cases",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert main(["research", "reproduce", str(output)]) == 0
    assert "inconclusive" in capsys.readouterr().out
    assert reproduce(output)["cases_checked"] == 1
    assert (
        main(
            [
                "research",
                "demo",
                "--max-cases",
                "1",
                "--output",
                str(tmp_path / "invalid"),
            ]
        )
        == 2
    )


def test_existing_configured_bundles_still_dispatch(tmp_path):
    bundle = tmp_path / "old"
    create_configured(default_model(), bundle)
    assert reproduce(bundle)["verification_status"] == "certified-under-assumptions"
