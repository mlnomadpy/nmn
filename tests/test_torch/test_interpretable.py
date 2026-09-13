"""Native architecture routing, intervention gradients and persistence."""

import pytest

torch = pytest.importorskip("torch")

from nmn.torch import Intervention, ThreeNeuronYat, YatExpansion  # noqa: E402


def test_reference_intervention_and_restore():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    x = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    normal, trace = model.forward_with_trace(x)
    torch.testing.assert_close(normal, x.new_tensor([[4.0, 1.0]]))
    gated = model(x, {"h": Intervention(gate=0)})
    torch.testing.assert_close(gated, x.new_tensor([[0.5, 1.0]]))
    restored = model(x, {"h": Intervention(gate=0, replacement=trace["h"])})
    torch.testing.assert_close(restored, normal)


def test_native_gradients_and_protected_routing():
    torch.manual_seed(5)
    model = ThreeNeuronYat(num_centers=3, dtype=torch.float64)
    x = torch.rand(7, 2, dtype=torch.float64)
    gate = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    y, trace = model.forward_with_trace(x, {"h": Intervention(gate=gate)})
    protected_grad = torch.autograd.grad(y[:, 1].sum(), gate, retain_graph=True)[0]
    assert protected_grad == 0
    y.sum().backward()
    assert gate.grad.isfinite() and gate.grad.abs() > 0
    for param in model.parameters():
        assert param.grad is not None and param.grad.isfinite().all()
    for name in model.state_names:
        torch.testing.assert_close(
            trace[f"{name}.contributions"].sum(-1), trace[f"{name}.raw"]
        )
    replacement = torch.ones(7, 1, dtype=x.dtype, requires_grad=True)
    model.zero_grad(set_to_none=True)
    model(x, {"h": Intervention(replacement=replacement)}).sum().backward()
    assert replacement.grad.isfinite().all()
    assert model.h.centers.grad is None
    torch.testing.assert_close(y[:, 1], model(x)[:, 1])


def test_expansion_and_checkpoint(tmp_path):
    model = ThreeNeuronYat(num_centers=4).double()
    x = torch.rand(2, 3, 2, dtype=torch.float64)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    clone = ThreeNeuronYat(num_centers=4).double()
    clone.load_state_dict(torch.load(path, map_location="cpu"))
    torch.testing.assert_close(model(x), clone(x))
    expansion = YatExpansion(2, 3, 4).double()
    torch.testing.assert_close(expansion(x), expansion.contributions(x).sum(-1))
    with pytest.raises(ValueError, match="unknown intervention"):
        model(x, {"missing": Intervention()})
    with pytest.raises(ValueError, match="broadcast"):
        model(x, {"h": Intervention(gate=torch.ones(4))})


def test_research_measurements(tmp_path):
    import json

    from nmn.torch.research import (
        coalition_effects,
        collect_research_data,
        protection_metrics,
        save_research_data,
    )

    model = ThreeNeuronYat.reference(dtype=torch.float64)
    x = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    data = collect_research_data(
        model, x, sample_ids=["one"], edits={"remove_h": {"h": Intervention(gate=0)}}
    )
    assert data["observations"]["edits"]["remove_h"]["delta"] == [[-3.5, 0.0]]
    assert data["geometry"]["y"]["rkhs_inner_products"] == [[4.0]]
    assert data["gate_derivatives"]["jacobian"][0][1][0] == 0
    assert all(p.grad is None for p in model.parameters())
    # Local derivatives agree with finite differences at this smooth point.
    jac = torch.tensor(data["input_jacobian"], dtype=x.dtype)[0]
    step = 1e-5
    for i in range(2):
        delta = torch.zeros_like(x)
        delta[0, i] = step
        finite_diff = (model(x + delta) - model(x - delta))[0] / (2 * step)
        torch.testing.assert_close(jac[:, i], finite_diff, atol=1e-7, rtol=1e-7)
    path = tmp_path / "data.json"
    save_research_data(data, path)
    assert json.loads(path.read_text())["model_sha256"] == data["model_sha256"]
    with pytest.raises(FileExistsError):
        save_research_data(data, path)
    coalition = coalition_effects(model, x)
    for mask in range(8):
        reconstructed = sum(
            coalition["subset_coefficients"][s] for s in range(8) if s & mask == s
        )
        torch.testing.assert_close(reconstructed, coalition["values"][mask])
    metrics = protection_metrics(
        torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([0, 0])
    )
    assert metrics["accuracy_before"] == metrics["accuracy_after"] == 0.5
    assert metrics["conditional_damage_rate"] == metrics["disagreement_rate"] == 1
