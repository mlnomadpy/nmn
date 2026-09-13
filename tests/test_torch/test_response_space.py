"""An unseen response direction remains visible in evaluation residuals."""

import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import Intervention, YatGraph, YatModuleSpec
from nmn.torch.response_space import response_space_study


def test_fit_basis_does_not_absorb_evaluation_direction():
    model = YatGraph(
        ["u", "v", "y", "p"],
        ["u", "v"],
        ["y", "p"],
        [[YatModuleSpec("a", ["u"], ["y"]), YatModuleSpec("b", ["v"], ["p"])]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for block in model.blocks.values():
            block.centers.fill_(1)
            block.coefficients.fill_(1)
    ds = ResearchDataset(
        [
            ResearchSample("fit", (1.0, 0.0), "tuning", "a"),
            ResearchSample("eval", (0.0, 1.0), "validation", "b"),
        ],
        name="Independent response directions",
        provenance="Arithmetic fixture",
    )
    record = response_space_study(
        model,
        ds,
        edits={
            "remove-a": {"a": Intervention(gate=0)},
            "remove-b": {"b": Intervention(gate=0)},
        },
        rank=1,
    )
    assert record["numerical_rank"] == 1
    assert record["fit"]["residual_norm"] == 0.0
    assert record["evaluation"]["residual_norm"] == 1.0
    assert record["evaluation"]["relative_residual"] == 1.0
    assert record["cost"]["model_forward_calls"] == 6
