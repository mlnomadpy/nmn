"""Backend-free reusable extraction preserves source linkage and input files."""

import hashlib
import json

import pytest

from nmn.research.components import extract_native_component, list_native_components


def test_maps_extraction_and_no_overwrite(tmp_path):
    source = tmp_path / "summary.json"
    maps = {"encoder": {"weight": [[1]], "bias": [0]}}
    raw = json.dumps({"schema": "nmn.reduction-study.v1", "maps": maps}).encode()
    source.write_bytes(raw)
    assert list_native_components(source)["components"] == ["maps"]
    output = tmp_path / "extracted"
    result = extract_native_component(source, "maps", output)
    assert json.loads((output / "data.json").read_text()) == maps
    assert result["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        result["data_sha256"]
        == hashlib.sha256((output / "data.json").read_bytes()).hexdigest()
    )
    assert result["recomputed"] is False
    with pytest.raises(FileExistsError):
        extract_native_component(source, "maps", output)
    with pytest.raises(ValueError, match="unavailable"):
        extract_native_component(source, "evaluation", tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    assert source.read_bytes() == raw


def test_selected_edit_retains_name_and_requires_frozen_candidate(tmp_path):
    source = tmp_path / "selection.json"
    record = {
        "schema": "nmn.edit-selection.v1",
        "status": "selected",
        "selected": "winner",
        "candidates": {"winner": {"h": {"gate": 0.0}}},
        "ledger": {"selected": {"candidate_id": "winner"}},
    }
    source.write_text(json.dumps(record))
    output = tmp_path / "edit"
    extract_native_component(source, "selected-edit", output)
    assert json.loads((output / "data.json").read_text()) == {
        "winner": {"h": {"gate": 0.0}}
    }
    record["ledger"]["selected"]["candidate_id"] = "different"
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="freeze ledger"):
        list_native_components(source)
    record["selected"] = None
    record["status"] = "no-feasible-measured-candidate"
    source.write_text(json.dumps(record))
    assert "selected-edit" not in list_native_components(source)["components"]
