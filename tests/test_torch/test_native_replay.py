"""Replay checks measured values, not merely parameter identity hashes."""

import copy

import torch

from nmn.research.native_export import render_native_note
from nmn.torch import ThreeNeuronYat
from nmn.torch.replay import replay_native_record
from nmn.torch.research import collect_research_data


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
