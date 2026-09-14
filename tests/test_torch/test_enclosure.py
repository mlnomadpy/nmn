"""Rational native enclosures checked against a separate exact point evaluator."""

from fractions import Fraction as F

import torch

from nmn.research.reference import kernel
from nmn.torch import ThreeNeuronYat
from nmn.torch.enclosure import enclose_native
from nmn.torch.research import collect_research_data


def test_reference_box_contains_exact_points_and_point_box_is_exact():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    snapshot = collect_research_data(
        model,
        torch.zeros(1, 2, dtype=torch.float64),
        sample_ids=["identity"],
        derivatives=False,
    )
    box = {"u": ["0", "1"], "v": ["0", "1"]}
    result = enclose_native(snapshot, box)
    for u in (F(0), F(1, 3), F(1)):
        for v in (F(0), F(1, 7), F(1)):
            h = kernel((F(1),), (u,))
            y = kernel((F(1), F(1)), (h, v))
            lo, hi = map(F, result["output_bounds"]["target"])
            assert lo <= y <= hi
            point = enclose_native(snapshot, {"u": [str(u)] * 2, "v": [str(v)] * 2})
            assert list(map(F, point["output_bounds"]["target"])) == [y, y]
    for values in result["denominator_bounds"].values():
        assert all(F(bounds[0]) > 0 for bounds in values)


def test_residual_imq_then_yat_and_reject_tanh():
    import pytest

    from nmn.torch import YatGraph, YatModuleSpec

    for family in ("imq", "tanh"):
        model = YatGraph(
            ["x", "h", "y"],
            ["x"],
            ["y"],
            [
                [YatModuleSpec("a", ["x"], ["h"], family=family)],
                [YatModuleSpec("b", ["h"], ["y"])],
            ],
            dtype=torch.float64,
        )
        with torch.no_grad():
            for block in model.blocks.values():
                for parameter in block.parameters():
                    parameter.fill_(1)
        snapshot = collect_research_data(
            model,
            torch.zeros(1, 1, dtype=torch.float64),
            sample_ids=["zero"],
            derivatives=False,
        )
        if family == "tanh":
            with pytest.raises(ValueError, match="only fixed Yat and IMQ"):
                enclose_native(snapshot, {"x": ["0", "0"]})
        else:
            assert enclose_native(snapshot, {"x": ["0", "0"]})["output_bounds"][
                "y"
            ] == ["1/5", "1/5"]
