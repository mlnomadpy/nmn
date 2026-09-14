"""Suffix execution shares graph semantics and retains state gradients."""

import pytest
import torch

from nmn.torch import Intervention, YatGraph, YatModuleSpec


def test_every_boundary_reconstructs_and_rejects_skipped_controls():
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
    x = torch.ones(2, 1, dtype=torch.float64)
    output, trace = model.forward_with_trace(x)
    for boundary in range(3):
        replayed, suffix = model.forward_from_state(
            trace[f"state.{boundary}"], start_layer=boundary
        )
        torch.testing.assert_close(replayed, output)
        assert f"state.{boundary}" in suffix
    state = trace["state.1"].detach().clone().requires_grad_()
    edited, _ = model.forward_from_state(state, start_layer=1)
    assert torch.autograd.grad(edited.sum(), state)[0][:, 1].abs().sum() > 0
    with pytest.raises(ValueError, match="unknown intervention"):
        model.forward_from_state(
            state, start_layer=1, interventions={"a": Intervention(gate=0)}
        )
    before = state.detach().clone()
    model.forward_from_state(state, start_layer=1, read_patches={"b": {"h": 0.0}})
    torch.testing.assert_close(state, before)
