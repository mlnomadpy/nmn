"""Replay checks measured values, not merely parameter identity hashes."""

import copy

import torch

from nmn.research.native_export import render_native_note
from nmn.torch import ThreeNeuronYat
from nmn.torch.paths import gate_path
from nmn.torch.replay import replay_native_record
from nmn.torch.research import _json_value, collect_research_data


def test_replay_detects_changed_observation_and_retains_source():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    saved = collect_research_data(
        model,
        torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        sample_ids=["one"],
        derivatives=False,
    )
    before = copy.deepcopy(saved)
    matching = replay_native_record(saved, atol=0, rtol=0)
    assert matching["status"] == "matched" and saved == before
    saved["observations"]["baseline"][0][0] += 1
    mismatch = replay_native_record(saved)
    assert mismatch["status"] == "mismatch"
    assert mismatch["mismatches"][0]["path"] == "/observations/baseline/0/0"
    assert mismatch["maximum_absolute_error"] == 1
    assert "Mismatch count: 1" in render_native_note(mismatch)
    assert mismatch["record_sha256"] != matching["record_sha256"]


def test_path_replay_ignores_timing_but_checks_curvature():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    inputs = torch.ones(1, 2, dtype=torch.float64)
    record = {
        "schema": "nmn.gate-path-study.v1",
        "model_snapshot": collect_research_data(
            model, inputs, sample_ids=["one"], derivatives=False
        ),
        "path": _json_value(gate_path(model, inputs, [1, 1, 1], [0, 1, 1], steps=2)),
    }
    record["path"]["cost"]["seconds"] = -1
    result = replay_native_record(record)
    assert result["status"] == "matched"
    assert "/path/cost/seconds" in result["ignored_paths"]
    record["path"]["directional_curvature"][0][0][0] += 1
    result = replay_native_record(record)
    assert result["status"] == "mismatch"
    assert any("directional_curvature" in item["path"] for item in result["mismatches"])
