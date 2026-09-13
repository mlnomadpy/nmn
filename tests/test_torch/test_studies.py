"""Donor execution and explicit data separation checks."""

import pytest

torch = pytest.importorskip("torch")

from nmn.research.datasets import (  # noqa: E402
    DonorPair,
    ResearchDataset,
    ResearchSample,
)
from nmn.torch import ThreeNeuronYat  # noqa: E402
from nmn.torch.studies import donor_study  # noqa: E402


def dataset():
    return ResearchDataset(
        [
            ResearchSample("a", (1.0, 1.0), "eval", "a", {"role": "scalar"}),
            ResearchSample("b", (0.0, 1.0), "eval", "b", {"role": "scalar"}),
            ResearchSample("c", (0.5, 1.0), "train", "c", {"role": "scalar"}),
        ],
        name="fixture",
        provenance="designed values",
    )


def test_donor_recompute_and_independent_reference():
    data = dataset()
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    study = donor_study(
        model,
        data,
        [DonorPair("p", "a", "b", ("h",))],
        reference=lambda base, donor, modules: {"target": 0.5},
        reference_id="v1",
        protected_outputs=["protected"],
        match_semantics=["role"],
    )
    row = study["rows"][0]
    assert row["edited_outputs"] == [0.5, 1.0]
    assert row["absolute_reference_error"] == {"target": 0.0}
    assert row["protected_delta"] == {"protected": 0.0}
    assert row["donor_writes"]["h"] == [[0.0]]
    assert all(p.grad is None for p in model.parameters())
    assert ResearchDataset.from_dict(data.to_dict()).sha256 == data.sha256


def test_split_and_group_access_are_explicit():
    data = dataset()
    pairs = [DonorPair("cross", "a", "c", ("h",))]
    with pytest.raises(ValueError, match="crosses splits"):
        data.validate_pairs(pairs)
    data.validate_pairs(pairs, allow_cross_split=True)
    with pytest.raises(ValueError, match="crosses splits"):
        ResearchDataset(
            [
                ResearchSample("a", (1.0,), "train", "same"),
                ResearchSample("b", (1.0,), "eval", "same"),
            ],
            name="bad",
            provenance="test",
        )
    with pytest.raises(ValueError, match="reference_id"):
        donor_study(
            ThreeNeuronYat.reference(),
            data,
            [DonorPair("p", "a", "b", ("h",))],
            reference=lambda *args: {},
        )
    with pytest.raises(ValueError, match="duplicate sample"):
        ResearchDataset(
            [data.sample("a"), data.sample("a")], name="bad", provenance="test"
        )
