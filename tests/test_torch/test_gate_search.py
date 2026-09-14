"""Gate proposals use selection data and preserve native parameters."""

import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import ThreeNeuronYat
from nmn.torch.gate_search import search_gates
from nmn.torch.replay import replay_native_record


def test_gate_search_freezes_before_validation():
    model = ThreeNeuronYat(dtype=torch.float64)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1)
    original = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    data = ResearchDataset(
        [
            ResearchSample("t", [1.0, 1.0], "tuning", "t", {}),
            ResearchSample("v", [1.0, 1.0], "validation", "v", {}),
        ],
        name="gate fixture",
        provenance="supplied arithmetic values",
    )
    args = dict(
        modules=["h"],
        targets={"t": {"target": 0.0}, "v": {"target": 10.0}},
        target_outputs=["target"],
        protected_outputs=["protected"],
        protection_tolerance=0.0,
        provenance="supplied target and protected coordinate",
        max_steps=12,
        max_seconds=30.0,
        learning_rate=0.1,
        protection_weight=1.0,
    )
    result = search_gates(model, data, **args)
    changed = search_gates(
        model,
        data,
        **{**args, "targets": {"t": {"target": 0.0}, "v": {"target": -10.0}}},
    )
    assert result["candidates"] == changed["candidates"]
    selected = result["selection"]["selected"]
    assert selected == changed["selection"]["selected"]
    assert (
        result["selection"]["selection"][selected]["target_mse"]
        < result["proposals"][0]["target_mse"]
    )
    assert replay_native_record(result["selection"])["status"] == "matched"
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, original[name])
        assert value.grad is None
