"""Native JSON restoration must preserve computation and reject stale identity."""

import copy

import pytest

torch = pytest.importorskip("torch")

from nmn.torch import ThreeNeuronYat, YatGraph, YatModuleSpec  # noqa: E402
from nmn.torch.research import collect_research_data, model_from_snapshot  # noqa: E402


@pytest.mark.parametrize("graph", [False, True])
def test_native_snapshot_restoration_and_tampering(graph):
    if graph:
        model = YatGraph(
            ["x", "y"],
            ["x"],
            ["y"],
            [[YatModuleSpec("block", ["x"], ["y"], 3)]],
            dtype=torch.float64,
        )
        x = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    else:
        model = ThreeNeuronYat.reference(dtype=torch.float64)
        x = torch.tensor([[0.25, 1.0], [0.75, 1.0]], dtype=torch.float64)
    data = collect_research_data(model, x, sample_ids=["a", "b"], derivatives=False)
    restored = model_from_snapshot(data)
    torch.testing.assert_close(restored(x), model(x))
    changed = copy.deepcopy(data)
    key = next(iter(changed["parameters"]))
    changed["parameters"][key][0][0] += 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        model_from_snapshot(changed)
