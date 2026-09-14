"""Check the fit/evaluation boundary and measured projection algebra."""

import copy

import pytest

torch = pytest.importorskip("torch")

from nmn.research.datasets import ResearchDataset, ResearchSample  # noqa: E402
from nmn.torch import ThreeNeuronYat  # noqa: E402
from nmn.torch.erasure import erasure_study  # noqa: E402


def test_fit_projection_is_independent_of_evaluation_labels():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    samples = [
        ResearchSample(str(i), [i / 8, 1.0], "fit" if i < 6 else "eval", str(i))
        for i in range(8)
    ]
    dataset = ResearchDataset(samples, name="fixture", provenance="synthetic")
    labels = {s.sample_id: [s.inputs[0] ** 2] for s in samples}
    state = copy.deepcopy(model.state_dict())
    kwargs = dict(
        module_name="y",
        provenance="fixture",
        rtol=1e-10,
        fit_split="fit",
        evaluation_split="eval",
    )
    record = erasure_study(model, dataset, labels=labels, **kwargs)
    assert record["projection"]["removed_rank"] == 1
    assert record["fit"]["covariance_before_norm"] > 0
    assert record["fit"]["covariance_after_norm"] < 1e-12
    labels["6"], labels["7"] = [100.0], [-50.0]
    changed = erasure_study(model, dataset, labels=labels, **kwargs)
    assert changed["projection"] == record["projection"]
    assert changed["targets"] == record["targets"]
    assert all(
        torch.equal(value, state[name]) for name, value in model.state_dict().items()
    )
    assert all(p.grad is None for p in model.parameters())
    labels["extra"] = [0.0]
    with pytest.raises(ValueError, match="exactly"):
        erasure_study(model, dataset, labels=labels, **kwargs)
