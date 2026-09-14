"""Producer-specific patches retain other residual writers and readers."""

import pytest
import torch

from nmn.torch import Intervention, YatGraph, YatModuleSpec


def test_residual_edge_isolation_uses_current_effective_write():
    model = YatGraph(
        ["x", "h", "y", "p"],
        ["x"],
        ["y", "p"],
        [
            [YatModuleSpec("a", ["x"], ["h"]), YatModuleSpec("c", ["x"], ["h"])],
            [
                YatModuleSpec("b", ["h"], ["y"]),
                YatModuleSpec("protected", ["h"], ["p"]),
            ],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for block in model.blocks.values():
            block.centers.fill_(1)
            block.coefficients.fill_(1)
        model.blocks["c"].coefficients.fill_(2)
    x = torch.ones(1, 1, dtype=torch.float64)
    baseline, trace = model.forward_with_trace(x)
    edges = {"b": {"h": {"a": 0.0}}}
    result, patched = model.forward_with_trace(x, edge_patches=edges)
    assert patched["b.input"].item() == 2.0
    assert patched["b.input_original"].item() == 3.0
    assert patched["state.1"][0, 1].item() == 3.0
    assert patched["b.edge_delta.h"].item() == -1.0
    assert result[0, 0].item() == 2.0
    torch.testing.assert_close(result[:, 1], baseline[:, 1])
    torch.testing.assert_close(model(x, edge_patches=edges), result)
    _, controlled = model.forward_with_trace(
        x, {"a": Intervention(gate=2)}, edge_patches=edges
    )
    assert controlled["b.input_original"].item() == 4.0
    assert controlled["b.input"].item() == 2.0
    replacement = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    value = model(x, edge_patches={"b": {"h": {"a": replacement}}})[:, 0].sum()
    assert torch.autograd.grad(value, replacement)[0].abs().item() > 0
    with pytest.raises(ValueError, match="both"):
        model(x, read_patches={"b": {"h": 0}}, edge_patches=edges)
    with pytest.raises(ValueError, match="earlier"):
        model.forward_from_state(trace["state.1"], start_layer=1, edge_patches=edges)
    with pytest.raises(ValueError, match="earlier"):
        model(x, edge_patches={"a": {"x": {"c": 0}}})
