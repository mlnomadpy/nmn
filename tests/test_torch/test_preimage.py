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


def test_dataset_study_uses_module_inputs_and_exact_target_population():
    import pytest

    from nmn.research.datasets import ResearchDataset, ResearchSample
    from nmn.torch.preimage import preimage_study

    dataset = ResearchDataset(
        [
            ResearchSample("a", (1.0, 1.0), "evaluation", "a"),
            ResearchSample("b", (0.0, 0.0), "fit", "b"),
        ],
        name="population",
        provenance="synthetic",
    )
    model = ThreeNeuronYat.reference(dtype=torch.float64)
    kwargs = dict(
        module_name="y",
        lower=[0.0, 1.0],
        upper=[1.0, 1.0],
        provenance="fixture",
        max_steps=0,
        max_seconds=1.0,
        learning_rate=0.1,
        split="evaluation",
    )
    record = preimage_study(model, dataset, targets={"a": [0.5]}, **kwargs)
    assert record["sample_ids"] == ["a"]
    assert record["search"]["inputs"] == [[1.0, 1.0]]
    assert record["proposed_inputs"] == {"a": [1.0, 1.0]}
    assert record["dataset_sha256"] == dataset.sha256
    with pytest.raises(ValueError, match="exactly"):
        preimage_study(model, dataset, targets={"a": [0.5], "b": [0.5]}, **kwargs)


def test_proposals_execute_only_selected_reader_and_replay():
    import copy

    import pytest

    from nmn.research.datasets import ResearchDataset, ResearchSample
    from nmn.torch import YatGraph, YatModuleSpec
    from nmn.torch.preimage import execute_preimage_study, preimage_study
    from nmn.torch.replay import replay_native_record

    model = YatGraph(
        ("x", "h", "y", "p"),
        ("x",),
        ("y", "p"),
        [
            [YatModuleSpec("a", ("x",), ("h",))],
            [YatModuleSpec("b", ("h",), ("y",)), YatModuleSpec("c", ("h",), ("p",))],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    dataset = ResearchDataset(
        [ResearchSample("a", (1.0,), "evaluation", "a")],
        name="reader fixture",
        provenance="synthetic",
    )
    proposal = preimage_study(
        model,
        dataset,
        module_name="b",
        targets={"a": [0.0]},
        lower=[0.0],
        upper=[1.0],
        provenance="zero bank target",
        max_steps=100,
        max_seconds=10.0,
        learning_rate=0.1,
    )
    result = execute_preimage_study(proposal)
    assert result["outputs"] == [[0.0, 1.0]]
    assert result["trace"]["c.input"] == [[1.0]]
    assert replay_native_record(result)["status"] == "matched"
    result["output_delta"][0][1] = 2.0
    assert replay_native_record(result)["status"] == "mismatch"
    altered = copy.deepcopy(proposal)
    altered["proposed_inputs"]["a"] = [0.5]
    with pytest.raises(ValueError, match="selected inputs"):
        execute_preimage_study(altered)
