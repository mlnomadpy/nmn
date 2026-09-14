"""Explicit linear graph modules retain geometry, gradients and replay."""

import pytest
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


def test_linear_graph_replays_but_unsupported_enclosure_rejects():
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
    with pytest.raises(ValueError, match="only fixed Yat and IMQ"):
        enclose_native(record, {"x": [-1, 1]})
