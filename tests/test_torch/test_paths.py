"""Finite effects, quadrature and held-out selection separation."""

import pytest

torch = pytest.importorskip("torch")

from nmn.research.datasets import ResearchDataset, ResearchSample  # noqa: E402
from nmn.research.selection import SelectionLedger  # noqa: E402
from nmn.torch import ThreeNeuronYat  # noqa: E402
from nmn.torch.paths import gate_path  # noqa: E402


def test_native_path_reconstructs_effect_without_assuming_second_order_improves():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    x = torch.ones(1, 2, dtype=torch.float64)
    path = gate_path(model, x, [1, 1, 1], [0, 1, 1], steps=64)
    torch.testing.assert_close(path["actual_delta"], x.new_tensor([[-3.5, 0]]))
    assert path["directional_curvature"][0, 0, 0] == -6
    assert path["residuals"]["integrated_gradient"].abs().max() < 0.001
    assert path["residuals"]["integrated_curvature"].abs().max() < 0.001
    assert (
        path["residuals"]["second_order"].abs().max()
        > path["residuals"]["first_order"].abs().max()
    )
    assert all(p.grad is None for p in model.parameters())
    same = gate_path(model, x, [1, 1, 1], [1, 1, 1], steps=2)
    assert same["actual_delta"].abs().max() == 0
    assert same["directional_curvature"].abs().max() == 0


def test_selection_freezes_before_validation():
    ds = ResearchDataset(
        [
            ResearchSample("a", (1.0,), "tuning", "a"),
            ResearchSample("b", (2.0,), "validation", "b"),
        ],
        name="small",
        provenance="explicit",
    )
    ledger = SelectionLedger(ds, model_sha256="a" * 64)
    ledger.register("keep", {"h": 1})
    ledger.register("remove", {"h": 0})
    with pytest.raises(ValueError, match="frozen candidate"):
        ledger.record("keep", phase="validation", sample_ids=["b"], measurements={})
    with pytest.raises(ValueError, match="splits"):
        ledger.record("keep", phase="selection", sample_ids=["b"], measurements={})
    ledger.record("keep", phase="selection", sample_ids=["a"], measurements={"loss": 0})
    ledger.freeze("keep", rule="lowest measured target loss under protection budget")
    with pytest.raises(ValueError, match="frozen"):
        ledger.register("another", {})
    with pytest.raises(ValueError, match="frozen candidate"):
        ledger.record("remove", phase="validation", sample_ids=["b"], measurements={})
    ledger.record(
        "keep", phase="validation", sample_ids=["b"], measurements={"loss": 1}
    )
    assert [e["phase"] for e in ledger.to_dict()["events"]] == [
        "selection",
        "validation",
    ]
