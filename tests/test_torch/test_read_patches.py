"""Receiving-slot patches leave shared residual state and other readers intact."""

import pytest
import torch

from nmn.research.datasets import DonorPair, ResearchDataset, ResearchSample
from nmn.torch import YatGraph, YatModuleSpec
from nmn.torch.studies import donor_study


def model():
    graph = YatGraph(
        ["x", "h", "y", "p"],
        ["x"],
        ["y", "p"],
        [
            [YatModuleSpec("a", ["x"], ["h"])],
            [YatModuleSpec("b", ["h"], ["y"]), YatModuleSpec("c", ["h"], ["p"])],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for block in graph.blocks.values():
            block.centers.fill_(1)
            block.coefficients.fill_(1)
    return graph


def test_read_isolation_and_gradient():
    graph = model()
    x = torch.ones(1, 1, dtype=torch.float64)
    patch = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    output, trace = graph.forward_with_trace(x, read_patches={"b": {"h": patch}})
    assert output[0, 1] == 1
    assert trace["state.1"][0, 1] == 1
    assert trace["b.input_original"].item() == trace["c.input"].item() == 1
    assert trace["b.input"].item() == 0.5
    derivative = torch.autograd.grad(output[0, 0], patch)[0]
    assert derivative.item() != 0
    with pytest.raises(ValueError, match="input slots"):
        graph(x, read_patches={"b": {"x": patch}})


def test_donor_read_replay():
    graph = model()
    dataset = ResearchDataset(
        [
            ResearchSample("base", (1.0,), "evaluation", "a"),
            ResearchSample("donor", (0.0,), "evaluation", "b"),
        ],
        name="read-patch fixture",
        provenance="declared arithmetic inputs",
    )
    pairs = [
        DonorPair("transfer", "base", "donor", ("b",), {"y": 0.0}),
        DonorPair("self", "base", "base", ("b",), {"y": 1.0}),
    ]
    record = donor_study(
        graph, dataset, pairs, read_slots={"b": ["h"]}, protected_outputs=["p"]
    )
    row = record["rows"][0]
    assert row["edited_outputs"] == [0.0, 1.0]
    assert row["donor_reads"] == {"b": {"h": [0.0]}}
    assert row["donor_writes"] == {}
    assert row["protected_delta"] == {"p": 0.0}
    assert record["rows"][1]["edited_outputs"] == [1.0, 1.0]
    assert "read-slot" in record["protocol"]["donor_execution"]
