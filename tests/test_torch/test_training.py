"""Small optimizer checks for the explicit training API, not a research study."""

import copy

import pytest

torch = pytest.importorskip("torch")

from nmn.research.datasets import (  # noqa: E402
    DonorPair,
    ResearchDataset,
    ResearchSample,
)
from nmn.torch import ThreeNeuronYat  # noqa: E402
from nmn.torch.research import collect_research_data  # noqa: E402
from nmn.torch.training import TrainingConfig, train_native  # noqa: E402


def fixture():
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    model.p.requires_grad_(False)
    data = ResearchDataset(
        [
            ResearchSample("a", (1.0, 1.0), "train", "a"),
            ResearchSample("b", (0.5, 1.0), "train", "b"),
            ResearchSample("v", (0.75, 1.0), "validation", "v"),
            ResearchSample("untouched", (0.25, 1.0), "test", "test"),
        ],
        name="optimizer fixture",
        provenance="synthetic implementation check",
    )
    snapshot = collect_research_data(
        model,
        torch.ones(1, 2, dtype=torch.float64),
        sample_ids=["initial"],
        derivatives=False,
    )
    contract = {
        name: "implementation fixture; no semantic claim"
        for name in (
            "architecture_id",
            "semantic_specification",
            "intervention_specification",
            "guarantee_scope",
            "worked_example",
        )
    }
    return snapshot, data, contract


def test_bounded_task_training_preserves_frozen_parameters_and_initial_snapshot():
    snapshot, data, contract = fixture()
    original = copy.deepcopy(snapshot)
    record = train_native(
        snapshot,
        data,
        {"a": [0.0, 1.0], "b": [0.0, 1.0], "v": [0.0, 1.0]},
        config=TrainingConfig(max_steps=2, evaluate_every=1, learning_rate=0.01),
        architecture_contract=contract,
        target_provenance="fixture",
        seeds=[0, 1],
    )
    assert snapshot == original
    assert len(record["runs"]) == 2
    for run in record["runs"]:
        assert run["status"] == "completed" and run["steps_completed"] == 2
        assert run["best_step"] > 0
        selected = run["selected_checkpoint"]
        assert (
            selected["parameters"]["p.kernel.weight"]
            == snapshot["parameters"]["p.kernel.weight"]
        )
        assert not selected["trainability"]["p.kernel.weight"]
    assert "untouched" not in record["train_ids"] + record["checkpoint_selection_ids"]


def test_donor_loss_is_separate_and_validation_pair_access_is_rejected():
    snapshot, data, contract = fixture()
    config = TrainingConfig(max_steps=1, evaluate_every=1, intervention_weight=1)
    args = dict(
        config=config, architecture_contract=contract, target_provenance="fixture"
    )
    targets = {"a": [0.0, 1.0], "b": [0.0, 1.0], "v": [0.0, 1.0]}
    record = train_native(
        snapshot,
        data,
        targets,
        pairs=[DonorPair("pair", "a", "b", ("h",), {"target": 0.5})],
        **args,
    )
    assert record["protocol"] == "task-and-detached-donor-supervision"
    assert record["runs"][0]["history"][1]["pre_update_intervention_mse"] > 0
    with pytest.raises(ValueError, match="splits"):
        train_native(
            snapshot,
            data,
            targets,
            pairs=[DonorPair("bad", "a", "v", ("h",), {"target": 0.5})],
            **args,
        )


def test_budget_stops_and_failed_seeds_remain_in_report(monkeypatch):
    snapshot, data, contract = fixture()
    targets = {"a": [0.0, 1.0], "b": [0.0, 1.0], "v": [0.0, 1.0]}
    common = dict(architecture_contract=contract, target_provenance="fixture")
    record = train_native(
        snapshot,
        data,
        targets,
        config=TrainingConfig(max_steps=1, max_seconds=1e-12),
        **common,
    )
    assert record["runs"][0]["status"] == "budget-stopped"
    assert record["runs"][0]["steps_completed"] == 0

    def fail_step(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(torch.optim.Adam, "step", fail_step)
    record = train_native(
        snapshot,
        data,
        targets,
        config=TrainingConfig(max_steps=1),
        seeds=[0, 1],
        **common,
    )
    assert [run["status"] for run in record["runs"]] == ["failed", "failed"]
    assert all("injected optimizer failure" in run["error"] for run in record["runs"])


def test_joint_donor_gradient_reaches_donor_only_parameters():
    snapshot, _, contract = fixture()
    data = ResearchDataset(
        [
            ResearchSample("a", (0.0, 0.0), "train", "a"),
            ResearchSample("b", (1.0, 0.0), "train", "b"),
            ResearchSample("v", (1.0, 0.0), "validation", "v"),
        ],
        name="gradient route",
        provenance="zero base isolates donor gradient",
    )
    from nmn.torch.research import model_from_snapshot

    coefficients = []
    for detached in (True, False):
        record = train_native(
            snapshot,
            data,
            {"a": [0.0, 0.0], "b": [1.0, 0.0], "v": [1.0, 0.0]},
            config=TrainingConfig(
                max_steps=1,
                batch_size=1,
                evaluate_every=1,
                learning_rate=0.01,
                intervention_weight=1.0,
                separate_pair_rng=True,
                detach_donor=detached,
            ),
            architecture_contract=contract,
            target_provenance="gradient fixture",
            seeds=[0],
            pairs=[DonorPair("pair", "a", "b", ("h",), {"target": 1.0})],
        )
        result = record["runs"][0]
        assert result["status"] == "completed" and result["best_step"] == 1
        fitted = model_from_snapshot(result["selected_checkpoint"])
        coefficients.append(float(fitted.h.coefficients.detach().item()))
        assert record["configuration"]["detach_donor"] is detached
    assert coefficients[0] == 1.0
    assert coefficients[1] > coefficients[0]


def test_fixed_edit_checkpoint_selection_and_evaluation_exclusion():
    snapshot, data, contract = fixture()
    objective = {
        "schema": "nmn.fixed-edit-objective.v1",
        "controls": {"h": {"gate": 0.0}},
        "output_names": ["target"],
        "protected_outputs": ["protected"],
        "targets": {s: [0.0] for s in ("a", "b", "v")},
        "provenance": "Synthetic deletion target",
    }
    args = dict(
        config=TrainingConfig(
            max_steps=3,
            evaluate_every=1,
            fixed_edit_weight=2.0,
            protection_weight=1.0,
            checkpoint_objective="task-plus-fixed-edit",
        ),
        architecture_contract=contract,
        target_provenance="fixture",
        seeds=[0],
    )
    targets = {s: [0.0, 1.0] for s in ("a", "b", "v")}
    record = train_native(snapshot, data, targets, fixed_edit=objective, **args)
    run = record["runs"][0]
    assert run["status"] == "completed"
    for h in run["history"]:
        assert h["validation_selection_score"] == pytest.approx(
            h["validation_mse"]
            + 2 * h["validation_fixed_edit_mse"]
            + h["validation_protection_mse"]
        )
    assert run["best_selection_score"] == min(
        h["validation_selection_score"] for h in run["history"]
    )
    assert (
        run["history"][-1]["validation_fixed_edit_mse"]
        < run["history"][0]["validation_fixed_edit_mse"]
    )
    objective["targets"]["untouched"] = [0.0]
    with pytest.raises(ValueError, match="exactly training"):
        train_native(snapshot, data, targets, fixed_edit=objective, **args)
