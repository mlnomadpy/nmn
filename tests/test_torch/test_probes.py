"""Fit-only decoding and edited held-out evidence are kept separate."""

import copy

import pytest
import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import Intervention, YatGraph, YatModuleSpec
from nmn.torch.probes import probe_study
from nmn.torch.replay import replay_native_record


def test_probe_freezing_damage_and_replay():
    model = YatGraph(
        ["x", "h"],
        ["x"],
        ["h"],
        [[YatModuleSpec("a", ["x"], ["h"])]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.blocks["a"].centers.fill_(1)
        model.blocks["a"].coefficients.fill_(1)
    dataset = ResearchDataset(
        [
            ResearchSample(sid, [value], split, sid, {})
            for sid, value, split in (
                ("f0", 0.0, "tuning"),
                ("f1", 1.0, "tuning"),
                ("e0", 0.0, "validation"),
                ("e1", 1.0, "validation"),
            )
        ],
        name="decoding fixture",
        provenance="supplied binary labels",
    )
    labels = {"f0": "zero", "f1": "one", "e0": "zero", "e1": "one"}
    kwargs = dict(
        feature="a.input",
        labels=labels,
        classes=["zero", "one"],
        provenance="arithmetic fixture",
        ridge=0.01,
        edits={"remove": {"a": Intervention(gate=0)}},
    )
    result = probe_study(model, dataset, **kwargs)
    assert result["evaluation"]["baseline"]["accuracy"] == 1.0
    assert result["evaluation"]["edits"]["remove"]["accuracy"] == 1.0
    # The effective write loses decoding while its preserved input still decodes.
    kwargs["feature"] = "a"
    result = probe_study(model, dataset, **kwargs)
    assert result["evaluation"]["edits"]["remove"]["accuracy"] == 0.5
    assert result["evaluation"]["edits"]["remove"]["conditional_damage"] == 0.5
    changed = copy.deepcopy(labels)
    changed.update(e0="one", e1="zero")
    other = probe_study(model, dataset, **{**kwargs, "labels": changed})
    assert result["probe"] == other["probe"]
    assert result["fit"] == other["fit"]
    assert replay_native_record(result)["status"] == "matched"
    altered = copy.deepcopy(result)
    altered["evaluation"]["baseline"]["scores"][0][0] += 1
    assert replay_native_record(altered)["status"] == "mismatch"
    with pytest.raises(ValueError, match="distinct"):
        probe_study(model, dataset, **kwargs, evaluation_split="tuning")


def test_refitted_probe_recovers_sign_change_but_not_constant_write():
    model = YatGraph(
        ["x", "h"],
        ["x"],
        ["h"],
        [[YatModuleSpec("a", ["x"], ["h"])]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.blocks["a"].centers.fill_(1)
        model.blocks["a"].coefficients.fill_(1)
    data = ResearchDataset(
        [
            ResearchSample(sid, [value], split, sid, {})
            for sid, value, split in (
                ("f0", 0.0, "tuning"),
                ("f1", 1.0, "tuning"),
                ("e0", 0.0, "validation"),
                ("e1", 1.0, "validation"),
            )
        ],
        name="refitted decoding",
        provenance="arithmetic fixture",
    )
    kwargs = dict(
        feature="a",
        labels=dict(f0="zero", f1="one", e0="zero", e1="one"),
        classes=["zero", "one"],
        provenance="supplied arithmetic labels",
        ridge=0.01,
        edits={
            "negate": {"a": Intervention(gate=-1)},
            "remove": {"a": Intervention(gate=0)},
        },
        refit_edits=True,
    )
    result = probe_study(model, data, **kwargs)
    negated = result["evaluation"]["edits"]["negate"]
    assert negated["accuracy"] == 0.5
    assert negated["refitted"]["evaluation"]["accuracy"] == 1.0
    assert (
        result["evaluation"]["edits"]["remove"]["refitted"]["evaluation"]["accuracy"]
        == 0.5
    )
    changed = probe_study(
        model,
        data,
        **{**kwargs, "labels": dict(f0="zero", f1="one", e0="one", e1="zero")},
    )
    for name in kwargs["edits"]:
        assert (
            result["evaluation"]["edits"][name]["refitted"]["probe"]
            == changed["evaluation"]["edits"][name]["refitted"]["probe"]
        )
    assert result["cost"] == dict(full_forwards=6, linear_solves=3)
    assert replay_native_record(result)["status"] == "matched"
    altered = copy.deepcopy(result)
    altered["evaluation"]["edits"]["negate"]["refitted"]["probe"]["weight"][0][0] += 1
    assert replay_native_record(altered)["status"] == "mismatch"
    with pytest.raises(ValueError, match="scalar"):
        probe_study(
            model,
            data,
            **{**kwargs, "edits": {"bad": {"a": Intervention(gate=[0, 1])}}},
        )
