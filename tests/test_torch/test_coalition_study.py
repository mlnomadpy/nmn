"""Full and budget-limited native lattices preserve finite coverage."""

import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import Intervention, ThreeNeuronYat, YatGraph, YatModuleSpec
from nmn.torch.coalitions import coalition_study
from nmn.torch.research import coalition_effects


def test_full_lattice_matches_reference_and_reconstructs():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    ds = ResearchDataset(
        [ResearchSample("one", (1.0, 1.0), "eval", "one")],
        name="coalition fixture",
        provenance="arithmetic",
    )
    result = coalition_study(model, ds, modules=["h", "p", "y"], max_evaluations=8)
    expected = coalition_effects(model, torch.ones(1, 2, dtype=torch.float64))
    torch.testing.assert_close(
        torch.tensor(result["subset_coefficients"], dtype=torch.float64),
        expected["subset_coefficients"],
    )
    assert result["coverage"] == {"evaluated": 8, "total": 8, "complete": True}
    assert torch.tensor(result["reconstruction_error"]).abs().max() == 0
    partial = coalition_study(
        model, ds, modules=["h", "p", "y"], max_evaluations=3, background={"y": 0.5}
    )
    assert (
        partial["status"] == "inconclusive" and partial["subset_coefficients"] is None
    )
    assert partial["masks"] == [0, 1, 2]
    assert (
        partial["outputs"][0]
        == model(
            torch.ones(1, 2, dtype=torch.float64), {"y": Intervention(gate=0.5)}
        ).tolist()
    )
    assert partial["cost"]["coalition_forward_calls"] == 3


def test_four_module_graph_has_sixteen_masks():
    model = YatGraph(
        ["x", "y"],
        ["x"],
        ["y"],
        [[YatModuleSpec(f"m{i}", ["x"], ["y"]) for i in range(4)]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for block in model.blocks.values():
            block.centers.fill_(1)
            block.coefficients.fill_(1)
    ds = ResearchDataset(
        [ResearchSample("one", (1.0,), "eval", "one")],
        name="four additive writes",
        provenance="arithmetic",
    )
    result = coalition_study(model, ds, modules=model.state_names, max_evaluations=16)
    assert result["coverage"]["total"] == 16 and result["status"] == "observed"
    assert result["outputs"][0] == [[4.0]] and result["outputs"][15] == [[0.0]]
    for mask, coefficient in enumerate(result["subset_coefficients"]):
        if mask and mask & (mask - 1):
            assert coefficient == [[0.0]]
