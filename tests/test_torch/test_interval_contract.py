"""Check coverage and genuine point witnesses, not merely saved outcomes."""

import copy

import pytest
import torch

from nmn.torch import YatGraph, YatModuleSpec
from nmn.torch.interval_contract import check_box_certificate, verify_box
from nmn.torch.research import collect_research_data


def test_subdivision_closes_cancellation_bound_and_checker_rejects_missing_leaf():
    model = YatGraph(
        ["x", "y"],
        ["x"],
        ["y"],
        [[YatModuleSpec("a", ["x"], ["y"], 2)]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.blocks["a"].centers.fill_(1)
        model.blocks["a"].coefficients.copy_(torch.tensor([[1.0, -1.0]]))
    snapshot = collect_research_data(
        model,
        torch.zeros(1, 1, dtype=torch.float64),
        sample_ids=["zero"],
        derivatives=False,
    )
    contract = dict(
        schema="nmn.interval-contract.v1",
        provenance="Cancellation fixture",
        input_box={"x": ["0", "1"]},
        outputs={"y": ["-1/2", "1/2"]},
    )
    partial = verify_box(snapshot, contract, max_boxes=1)
    assert partial["status"] == "inconclusive"
    assert check_box_certificate(partial)["counts"]["pending"] == 2
    complete = verify_box(snapshot, contract, max_boxes=31)
    assert complete["status"] == "certified-under-assumptions"
    assert check_box_certificate(complete)["counts"]["split"] > 0
    altered = copy.deepcopy(complete)
    del altered["nodes"][
        next(k for k, v in altered["nodes"].items() if v["kind"] == "covered")
    ]
    with pytest.raises(ValueError, match="missing nodes"):
        check_box_certificate(altered)
    false_contract = {**contract, "outputs": {"y": ["1", "2"]}}
    false = verify_box(snapshot, false_contract, max_boxes=1)
    assert check_box_certificate(false)["outcome"] == "counterexample-found"
    false["nodes"][""]["output_bounds"]["y"] = ["1", "1"]
    with pytest.raises(ValueError, match="does not violate"):
        check_box_certificate(false)
