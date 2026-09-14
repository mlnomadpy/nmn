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


def test_fitting_excludes_evaluation_and_keeps_model_fixed():
    from nmn.torch.reduction import fit_reduction_study

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
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    def population(evaluation):
        return ResearchDataset(
            [
                ResearchSample(f"fit{i}", [x], "tuning", f"fit{i}", {})
                for i, x in enumerate([0.0, 0.5, 1.0])
            ]
            + [ResearchSample("eval", [evaluation], "validation", "eval", {})],
            name="fit-only fixture",
            provenance="arithmetic inputs",
        )

    result = fit_reduction_study(
        model, population(2.0), start_layer=1, rank=1, ridge=0.01
    )
    shifted = fit_reduction_study(
        model, population(20.0), start_layer=1, rank=1, ridge=0.01
    )
    assert result["maps"] == shifted["maps"]
    assert result["fitting"] == shifted["fitting"]
    assert result["fit"]["observations"] == shifted["fit"]["observations"]
    assert result["evaluation"]["observations"] != shifted["evaluation"]["observations"]
    assert replay_native_record(result["evaluation"])["status"] == "matched"
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, before[name])
        assert value.grad is None
    with pytest.raises(ValueError, match="distinct"):
        fit_reduction_study(
            model,
            population(2.0),
            start_layer=1,
            rank=1,
            ridge=0.01,
            evaluation_split="tuning",
        )
    with pytest.raises(ValueError, match="strictly positive"):
        fit_reduction_study(model, population(2.0), start_layer=1, rank=1, ridge=0)
