"""Alternative module execution, snapshots and comparison failure retention."""

import pytest

torch = pytest.importorskip("torch")

from nmn.research.datasets import ResearchDataset, ResearchSample  # noqa: E402
from nmn.torch import Intervention, YatGraph, YatModuleSpec  # noqa: E402
from nmn.torch.benchmark import benchmark_models  # noqa: E402
from nmn.torch.research import collect_research_data, model_from_snapshot  # noqa: E402


@pytest.mark.parametrize("family", ["imq", "tanh"])
def test_alternative_blocks_keep_trace_gradients_and_snapshot(family):
    model = YatGraph(
        ["x", "y"],
        ["x"],
        ["y"],
        [[YatModuleSpec("a", ["x"], ["y"], 2, family=family)]],
        dtype=torch.float64,
    )
    x = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    y, trace = model.forward_with_trace(x)
    torch.testing.assert_close(trace["a.contributions"].sum(-1), y)
    y.sum().backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )
    snapshot = collect_research_data(model, x, sample_ids=["a", "b"], derivatives=False)
    restored = model_from_snapshot(snapshot)
    torch.testing.assert_close(restored(x), y)
    torch.testing.assert_close(
        model(x, {"a": Intervention(gate=0)}), torch.zeros_like(y)
    )


def test_comparison_retains_failed_methods_and_budget():
    def model(family, writes="y"):
        return YatGraph(
            ["x", "y"],
            ["x"],
            ["y"],
            [[YatModuleSpec("a", ["x"], [writes], family=family)]],
        )

    ds = ResearchDataset(
        [ResearchSample("a", (0.5,), "eval", "a")],
        name="fixture",
        provenance="supplied",
    )
    result = benchmark_models(
        {"yat": model("yat"), "imq": model("imq"), "wrong-routing": model("yat", "x")},
        ds,
        edits={"off": {"a": Intervention(gate=0)}},
        repeats=2,
        warmup=0,
    )
    assert (
        result["methods"]["yat"]["status"]
        == result["methods"]["imq"]["status"]
        == "measured"
    )
    assert result["methods"]["wrong-routing"]["status"] == "failed"
    assert result["contract"]["repeats"] == 2
    assert result["methods"]["yat"]["baseline_cost"]["forward_calls"] == 2
