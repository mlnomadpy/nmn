"""Validation cannot retroactively choose a different candidate."""

import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import ThreeNeuronYat
from nmn.torch.selection import select_edit


def test_freeze_precedes_validation_even_when_validation_prefers_identity():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    ds = ResearchDataset(
        [
            ResearchSample("t", (1.0, 1.0), "tuning", "t"),
            ResearchSample("v", (1.0, 1.0), "validation", "v"),
        ],
        name="selection fixture",
        provenance="supplied contradictory split targets",
    )
    result = select_edit(
        model,
        ds,
        candidates={
            "identity": {},
            "remove-h": {"h": {"gate": 0}},
            "invalid": {"unknown": {"gate": 0}},
        },
        targets={"t": {"target": 0.0}, "v": {"target": 10.0}},
        target_outputs=["target"],
        protected_outputs=["protected"],
        protection_tolerance=0.0,
        provenance="synthetic selection ordering fixture",
        max_candidates=3,
    )
    assert result["selected"] == "remove-h"
    assert result["selection"]["invalid"]["status"] == "failed"
    assert result["validation"]["target_mse"] == 90.25
    ledger = result["ledger"]
    assert ledger["selected"]["after_event"] == 2
    assert [event["phase"] for event in ledger["events"]] == ["selection"] * 3 + [
        "validation"
    ]
    assert ledger["events"][-1]["candidate_id"] == "remove-h"
    assert result["validation_snapshot"]["sample_ids"] == ["v"]
    assert result["model_snapshot"]["sample_ids"] == ["t"]
