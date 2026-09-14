"""Actual protected replay retains conditional populations and missing strata."""

import pytest
import torch

from nmn.research.datasets import ResearchDataset
from nmn.research.native_export import render_native_note
from nmn.torch import Intervention, ThreeNeuronYat
from nmn.torch.protection import protection_study


def test_protection_replay_and_empty_eligibility():
    dataset = ResearchDataset.from_dict(
        {
            "schema": "nmn.research-dataset.v1",
            "name": "protection fixture",
            "provenance": "arithmetic fixture",
            "samples": [
                {
                    "sample_id": "a",
                    "inputs": [1.0, 1.0],
                    "split": "evaluation",
                    "group_id": "a",
                    "semantics": {"kind": "positive"},
                },
                {
                    "sample_id": "b",
                    "inputs": [0.0, 1.0],
                    "split": "evaluation",
                    "group_id": "b",
                    "semantics": {},
                },
            ],
        }
    )
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    tasks = {
        "protected": {
            "outputs": ["protected"],
            "rule": "threshold",
            "threshold": 0.5,
            "labels": {"a": 1, "b": 0},
        }
    }
    record = protection_study(
        model,
        dataset,
        edits={"remove-p": {"p": Intervention(gate=0)}},
        tasks=tasks,
        provenance="synthetic declared labels v1",
        strata=["kind"],
    )
    result = record["results"]["protected"]["remove-p"]
    overall = result["strata"][0]["metrics"]
    assert overall["accuracy_before"] == overall["accuracy_after"] == 0.5
    assert overall["conditional_damage_rate"] == 1.0
    assert overall["disagreement_rate"] == 1.0
    missing = result["strata"][2]
    assert missing["missing"] and missing["metrics"]["conditional_damage_rate"] is None
    assert missing["metrics"]["fixed"] == [True]
    assert "not zero damage" in render_native_note(record)
    tasks["protected"]["labels"]["a"] = True
    with pytest.raises(ValueError, match="valid integer classes"):
        protection_study(
            model, dataset, edits={"identity": {}}, tasks=tasks, provenance="fixture"
        )
