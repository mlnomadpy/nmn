"""Check execution semantics of explicit-state residual graphs."""

import pytest

torch = pytest.importorskip("torch")

from nmn.torch import Intervention, YatGraph, YatModuleSpec  # noqa: E402
from nmn.torch.research import collect_research_data  # noqa: E402


def graph(layers):
    model = YatGraph(["x", "h", "y"], ["x"], ["y"], layers, dtype=torch.float64)
    with torch.no_grad():
        for block in model.blocks.values():
            block.centers.fill_(1)
    return model


def test_simultaneous_reads_and_overlapping_writes():
    a = YatModuleSpec("a", ["x"], ["h"])
    b = YatModuleSpec("b", ["h"], ["y"])
    c = YatModuleSpec("c", ["x"], ["h"])
    x = torch.ones(1, 1, dtype=torch.float64)
    parallel = graph([[a, b, c]])
    output, trace = parallel.forward_with_trace(x)
    assert output.item() == 0  # b reads original h=0, not a's write
    assert trace["state.1"][0, 1] == 2  # overlapping writes add
    sequential = graph([[a], [b]])
    assert sequential(x).item() == 1
    assert sequential(x, {"a": Intervention(gate=0)}).item() == 0
    assert sequential.dependencies()["y"] == ["input:x", "module:a", "module:b"]
    assert parallel.dependencies()["y"] == ["module:b"]


def test_graph_collection_gradients_and_checkpoint():
    model = graph(
        [[YatModuleSpec("a", ["x"], ["h"], 2)], [YatModuleSpec("b", ["h"], ["y"])]]
    )
    x = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    data = collect_research_data(model, x, sample_ids=["one", "two"])
    assert len(data["gate_derivatives"]["jacobian"][0][0]) == 2
    clone = YatGraph.from_configuration(data["configuration"], dtype=torch.float64)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(x), model(x))
    gate = torch.tensor(0.5, dtype=x.dtype, requires_grad=True)
    model(x, {"a": Intervention(gate=gate)}).sum().backward()
    assert gate.grad.isfinite() and gate.grad.abs() > 0
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )
    wrong = YatGraph.from_configuration(
        {**data["configuration"], "output_names": ["h"]}, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="routing/configuration"):
        wrong.load_state_dict(model.state_dict())
