"""Configured evidence, failed-contract replay, relocation and tamper checks."""

import json

import pytest

from nmn.cli import main
from nmn.research.bundles import create_configured, export_any, reproduce_any
from nmn.research.model import default_model
from nmn.research.reference import digest, write_bundle


@pytest.mark.parametrize(
    "leak,expected",
    [("0", "certified-under-assumptions"), ("1", "counterexample-found")],
)
def test_configured_relocation_and_export(tmp_path, leak, expected):
    model = default_model()
    model["protected_leak"] = leak
    source = tmp_path / "source"
    create_configured(model, source)
    moved = tmp_path / "moved"
    source.rename(moved)
    assert reproduce_any(moved)["verification_status"] == expected
    exported = tmp_path / "vault" / "run"
    export_any(moved, exported)
    assert reproduce_any(exported)["verification_status"] == expected
    for path in moved.iterdir():
        assert path.read_bytes() == (exported / path.name).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        export_any(moved, exported)


@pytest.mark.parametrize("name", ["model.json", "evidence.json", "Report.md"])
def test_tampered_files_rejected(tmp_path, name):
    bundle = tmp_path / "bundle"
    create_configured(default_model(), bundle)
    (bundle / name).write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        reproduce_any(bundle)


def test_modified_result_with_updated_hash_is_recomputed(tmp_path):
    bundle = tmp_path / "bundle"
    create_configured(default_model(), bundle)
    path = bundle / "evidence.json"
    data = json.loads(path.read_text())
    data["coverage"] = 100
    path.write_text(json.dumps(data))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["evidence.json"] = digest(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recomputed"):
        reproduce_any(bundle)


def test_source_mismatch_and_missing_file(tmp_path):
    bundle = tmp_path / "bundle"
    create_configured(default_model(), bundle)
    path = bundle / "manifest.json"
    original = path.read_bytes()
    manifest = json.loads(original)
    manifest["implementation"]["model.py"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="implementation differs"):
        reproduce_any(bundle)
    path.write_bytes(original)
    (bundle / "model.json").unlink()
    with pytest.raises(ValueError, match="regular"):
        reproduce_any(bundle)


def test_failure_is_reproduced_successfully_in_cli(tmp_path, capsys):
    model = default_model()
    model["protected_leak"] = "1"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    bundle = tmp_path / "bundle"
    assert (
        main(["research", "demo", "--model", str(path), "--output", str(bundle)]) == 0
    )
    assert main(["research", "reproduce", str(bundle)]) == 0
    assert "counterexample-found" in capsys.readouterr().out


def test_legacy_bundle_dispatch(tmp_path):
    legacy = tmp_path / "legacy"
    write_bundle(legacy)
    assert reproduce_any(legacy)["status"] == "reproduced"
    export_any(legacy, tmp_path / "exported")
    assert reproduce_any(tmp_path / "exported")["status"] == "reproduced"


def test_untrusted_manifest_inventory_and_duplicate_keys(tmp_path):
    bundle = tmp_path / "bundle"
    create_configured(default_model(), bundle)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside"] = "bad"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory"):
        reproduce_any(bundle)
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        reproduce_any(bundle)


def test_failed_export_does_not_create_destination(tmp_path):
    bundle = tmp_path / "bundle"
    create_configured(default_model(), bundle)
    (bundle / "Report.md").write_text("bad")
    destination = tmp_path / "export"
    with pytest.raises(ValueError):
        export_any(bundle, destination)
    assert not destination.exists()
