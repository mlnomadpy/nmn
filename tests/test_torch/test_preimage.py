"""Finite native preimage search and frozen-parameter behavior."""

import torch

from nmn.torch import ThreeNeuronYat
from nmn.torch.preimage import search_preimages


def test_bounded_solver_produces_executable_input_without_parameter_updates():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    record = search_preimages(
        model.y,
        [[1.0, 1.0]],
        [[0.5]],
        lower=[0.0, 1.0],
        upper=[1.0, 1.0],
        sample_ids=["a"],
        provenance="fixture",
        max_steps=100,
        max_seconds=10.0,
        learning_rate=0.1,
    )
    assert record["selected_inputs"] == [[0.0, 1.0]]
    assert record["feature_residuals"] == [[0.0]]
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name])
        torch.testing.assert_close(p.grad, torch.ones_like(p))


def test_unattainable_target_retains_residual_without_certification():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    record = search_preimages(
        model.h,
        [[0.0]],
        [[-1.0]],
        lower=[0.0],
        upper=[0.0],
        sample_ids=["a"],
        provenance="negative feature target",
        max_steps=0,
        max_seconds=10.0,
        learning_rate=0.1,
    )
    assert record["selected_step"] == 0
    assert record["feature_residuals"] == [[1.0]]
    assert record["status"] == "step-budget-completed"
