"""Explicit linear graph modules retain geometry, gradients and replay."""

import torch

from nmn.torch import LinearExpansion, YatGraph, YatModuleSpec
from nmn.torch.baselines import baseline_geometry
from nmn.torch.enclosure import enclose_native
from nmn.torch.replay import replay_native_record
from nmn.torch.research import collect_research_data


def test_linear_expansion_matches_effective_map_and_norm():
    module = LinearExpansion(2, 2, 3, dtype=torch.float64)
    with torch.no_grad():
        module.centers.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        module.coefficients.copy_(torch.tensor([[1.0, 2.0, -1.0], [0.0, 0.0, 1.0]]))
    x = torch.tensor([[0.5, -0.25]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(module(x), x @ module.effective_weight.T)
    torch.testing.assert_close(module.contributions(x).sum(-1), module(x))
    geometry = baseline_geometry(module, x)
    torch.testing.assert_close(
        geometry["rkhs_inner_products"],
        module.effective_weight @ module.effective_weight.T,
    )
    (gradient,) = torch.autograd.grad(module(x).sum(), x)
    torch.testing.assert_close(gradient, module.effective_weight.sum(0)[None])


def test_linear_graph_replays_and_has_exact_box_image():
    model = YatGraph(
        ["x", "y"],
        ["x"],
        ["y"],
        [[YatModuleSpec("identity", ["x"], ["y"], family="linear")]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    x = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    record = collect_research_data(
        model, x, sample_ids=["a", "b", "c"], derivatives=True
    )
    assert replay_native_record(record)["status"] == "matched"
    enclosure = enclose_native(record, {"x": [-1, 1]})
    assert enclosure["output_bounds"]["y"] == ["-1", "1"]
    assert enclosure["denominator_bounds"]["identity"] == []


def test_linear_enclosure_contracts_factors_without_float_rounding():
    model = YatGraph(
        ["x", "y"],
        ["x"],
        ["y"],
        [[YatModuleSpec("a", ["x"], ["y"], family="linear", num_centers=3)]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.blocks["a"].centers.fill_(1.0)
        model.blocks["a"].coefficients.copy_(
            torch.tensor([[1e16, 1.0, -1e16]], dtype=torch.float64)
        )
    record = collect_research_data(
        model,
        torch.zeros(1, 1, dtype=torch.float64),
        sample_ids=["origin"],
        derivatives=False,
    )
    # In exact real arithmetic the factors sum to 1, even if float contraction
    # rounds away the middle term. This adapter does not cover runtime roundoff.
    assert enclose_native(record, {"x": ["-2", "3"]})["output_bounds"]["y"] == [
        "-2",
        "3",
    ]
