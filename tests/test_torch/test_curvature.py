"""Check directional products against a small dense reference and exact linear gate."""

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import Intervention, ThreeNeuronYat
from nmn.torch.curvature import curvature_study
from nmn.torch.replay import replay_native_record


def test_products_axes_linear_channel_and_parameter_preservation():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7.0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    dataset = ResearchDataset(
        [ResearchSample("x", [0.7, 0.3], "evaluation", "x")],
        name="curvature fixture",
        provenance="explicit native gates",
    )
    vectors = {"h": [-1.0, 0.0, 0.0], "p": [0.0, -1.0, 0.0], "mixed": [-0.2, 0.0, -0.3]}
    background = [0.8, 0.9, 0.6]
    record = curvature_study(
        model, dataset, directions=vectors, background=background, provenance="fixture"
    )
    observed = record["observations"]
    for j in range(2):

        def scalar(g):
            return model(
                torch.tensor([0.7, 0.3], dtype=torch.float64),
                {n: Intervention(gate=g[i]) for i, n in enumerate(model.state_names)},
            )[j]

        dense = torch.autograd.functional.hessian(
            scalar, torch.tensor(background, dtype=torch.float64)
        )
        expected = torch.tensor(list(vectors.values()), dtype=torch.float64) @ dense.T
        torch.testing.assert_close(
            torch.tensor(
                observed["hessian_vector_products"][0][j], dtype=torch.float64
            ),
            expected,
        )
    assert observed["directional_curvature"][0][1][1] == 0.0
    assert abs(observed["first_order_residual"][0][1][1]) < 1e-15
    assert record["cost"]["dense_hessian_allocated"] is False
    for name, parameter in model.named_parameters():
        assert torch.equal(before[name], parameter)
        assert torch.equal(parameter.grad, torch.full_like(parameter, 7.0))
    assert (
        replay_native_record(json.loads(json.dumps(record, sort_keys=True)))["status"]
        == "matched"
    )
    corrupt = copy.deepcopy(record)
    corrupt["observations"]["hessian_vector_products"][0][0][0][0] += 1.0
    assert replay_native_record(corrupt)["status"] == "mismatch"
