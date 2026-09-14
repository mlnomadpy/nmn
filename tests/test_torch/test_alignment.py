"""Selection leakage and incomplete-search controls for native assignment fitting."""

import copy

import pytest

torch = pytest.importorskip("torch")

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.semantics import TabulatedReference
from nmn.torch import ThreeNeuronYat
from nmn.torch.alignment import alignment_study
from nmn.torch.replay import replay_native_record


def fixture():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    samples = [
        ResearchSample(f"{s}-{i}", x, s, f"{s}-{i}")
        for s in ("tuning", "validation")
        for i, x in enumerate(((1.0, 1.0), (0.0, 0.0)))
    ]
    dataset = ResearchDataset(
        samples, name="assignment fixture", provenance="supplied arithmetic"
    )
    reference = dict(
        schema="nmn.semantic-reference.v1",
        reference_id="transfer",
        provenance="analytic zero and one outputs",
        variables=["square"],
        baseline={
            s.sample_id: dict(target=1.0 if s.inputs[0] else 0.0, protected=s.inputs[1])
            for s in samples
        },
        cases=[
            dict(
                case_id=s,
                base_id=s + "-0",
                donor_id=s + "-1",
                variables=["square"],
                outputs=dict(target=0.0, protected=1.0),
            )
            for s in ("tuning", "validation")
        ],
    )
    return model, dataset, reference


def test_selection_ignores_evaluation_labels_and_retains_budget_limit():
    model, dataset, reference = fixture()
    args = dict(module_pool=["p", "h"], max_candidates=2, provenance="fixture")
    full = alignment_study(
        model, dataset, reference=TabulatedReference(reference), **args
    )
    assert full["selected"]["mapping"] == {"square": ["h"]}
    changed = copy.deepcopy(reference)
    changed["cases"][1]["outputs"]["target"] = 100.0
    second = alignment_study(
        model, dataset, reference=TabulatedReference(changed), **args
    )
    assert full["candidates"] == second["candidates"]
    assert full["selected"] == second["selected"]
    assert full["evaluation"]["mse"] < second["evaluation"]["mse"]
    partial = alignment_study(
        model,
        dataset,
        reference=TabulatedReference(reference),
        **dict(args, max_candidates=1),
    )
    assert partial["status"] == "candidate-budget-stopped"
    assert partial["selected"]["mapping"] == {"square": ["p"]}
    assert replay_native_record(full)["status"] == "matched"
    corrupt = copy.deepcopy(full)
    corrupt["selection_executions"][0]["donor_execution"]["rows"][0]["edited_outputs"][
        0
    ] += 1.0
    assert replay_native_record(corrupt)["status"] == "mismatch"


def test_cross_split_reference_is_rejected():
    model, dataset, reference = fixture()
    reference["cases"][0]["donor_id"] = "validation-1"
    with pytest.raises(ValueError, match="within declared"):
        alignment_study(
            model,
            dataset,
            reference=TabulatedReference(reference),
            module_pool=["h"],
            max_candidates=1,
            provenance="fixture",
        )
