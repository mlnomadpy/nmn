"""Native report exports preserve evidence and distinguish integrity from replay."""

import copy
import hashlib
import json

import pytest

from nmn.research.native_export import (
    export_native_record,
    render_native_note,
    verify_native_export,
)


def snapshot():
    identity = {
        "configuration": {"class": "nmn.torch.ThreeNeuronYat"},
        "parameters": {"h.coefficients": [[1.0]]},
    }
    return {
        "schema": "nmn.native-model.v1",
        **identity,
        "model_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
    }


def test_export_preserves_bytes_and_detects_changes(tmp_path):
    source = tmp_path / "source.json"
    payload = (json.dumps(snapshot(), indent=3) + "\n").encode()
    source.write_bytes(payload)
    dest = tmp_path / "vault"
    export_native_record(source, dest)
    assert (dest / "data.json").read_bytes() == payload
    assert verify_native_export(dest)["recomputed"] is False
    with pytest.raises(FileExistsError):
        export_native_record(source, dest)
    (dest / "Report.md").write_text("changed")
    with pytest.raises(ValueError, match="content mismatch"):
        verify_native_export(dest)


def test_identity_mismatch_rejected_before_export(tmp_path):
    data = snapshot()
    data["parameters"]["h.coefficients"][0][0] = 2
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity mismatch"):
        export_native_record(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_path_render_is_pure_and_retains_nonfinite_status():
    data = {
        "schema": "nmn.gate-path-study.v1",
        "model_snapshot": snapshot(),
        "limitations": ["parent limitation"],
        "path": {
            "residuals": {"bad": [["nan"]]},
            "cost": {"intervals": 2, "model_forward_calls": 3},
            "limitations": ["path limitation"],
        },
    }
    before = copy.deepcopy(data)
    report = render_native_note(data)
    assert "unavailable/nonfinite" in report
    assert data == before
    assert render_native_note(data) == report
