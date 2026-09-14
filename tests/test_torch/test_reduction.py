"""A summary can be transition-exact while losing downstream behavior."""

import copy

import pytest
import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import YatGraph, YatModuleSpec
from nmn.torch.reduction import reduction_study
from nmn.torch.replay import replay_native_record


def test_summary_closure_does_not_imply_output_preservation():
    model = YatGraph(
        ["x", "h", "y"],
        ["x"],
        ["y"],
        [[YatModuleSpec("a", ["x"], ["h"])], [YatModuleSpec("b", ["h"], ["y"])]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for block in model.blocks.values():
            block.centers.fill_(1)
            block.coefficients.fill_(1)
    dataset = ResearchDataset(
        [ResearchSample("one", [1.0], "evaluation", "one", {})],
        name="summary fixture",
        provenance="supplied arithmetic fixture",
    )
    # The x coordinate is unchanged through layer b, but omitting h loses y.
    maps = {
        "encoder": {"weight": [[1.0], [0.0], [0.0]], "bias": [0.0]},
        "decoder": {"weight": [[1.0, 0.0, 0.0]], "bias": [0.0, 0.0, 0.0]},
        "transition": {"weight": [[1.0]], "bias": [0.0]},
    }
    result = reduction_study(
        model,
        dataset,
        start_layer=1,
        maps=maps,
        provenance="supplied x-only summary",
        split="evaluation",
    )
    observed = result["observations"]
    assert observed["summary_transition_residual"] == [[0.0]]
    assert observed["baseline_outputs"] == [[1.0]]
    assert observed["reconstruction_output_residual"] == [[-1.0]]
    assert observed["prediction_output_residual"] == [[-1.0]]
    assert replay_native_record(result)["status"] == "matched"
    altered = copy.deepcopy(result)
    altered["observations"]["prediction_output_residual"][0][0] = 0.0
    assert replay_native_record(altered)["status"] == "mismatch"
    with pytest.raises(ValueError, match="successor"):
        reduction_study(model, dataset, start_layer=2, maps=maps, provenance="fixture")
    invalid = copy.deepcopy(maps)
    invalid["decoder"]["weight"] = [[1.0]]
    with pytest.raises(ValueError, match="dimensions"):
        reduction_study(
            model, dataset, start_layer=1, maps=invalid, provenance="fixture"
        )
