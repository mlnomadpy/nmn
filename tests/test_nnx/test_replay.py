"""NNX snapshot identity and numerical replay failure paths."""

import copy
import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nmn.nnx import ThreeNeuronYat
from nmn.nnx.replay import model_from_snapshot, replay_native_record
from nmn.nnx.research import collect_research_data
from nmn.research.datasets import ResearchDataset, ResearchSample


def fixture_record():
    model = ThreeNeuronYat(num_centers=2, rngs=nnx.Rngs(7))
    dataset = ResearchDataset(
        [
            ResearchSample("eval", (0.25, 0.75), "evaluation", "eval"),
            ResearchSample("fit", (1.0, 1.0), "fit", "fit"),
        ],
        name="roundtrip",
        provenance="designed test",
    )
    return collect_research_data(
        model,
        dataset,
        split="evaluation",
        edits={"delete": {"h": {"gate": 0.0}}},
        derivatives=False,
    )


def test_nonreference_parameters_reload_and_replay():
    record = fixture_record()
    restored = model_from_snapshot(record)
    np.testing.assert_array_equal(
        restored(jnp.array([[0.25, 0.75]])), record["outputs"]
    )
    result = replay_native_record(record)
    assert result["status"] == "matched"
    assert result["execution"]["model_sha256"] == record["model_sha256"]
    assert result["execution"]["sample_ids"] == ["eval"]


def test_modified_measurement_and_missing_trace_do_not_match():
    record = fixture_record()
    changed = copy.deepcopy(record)
    changed["outputs"][0][0] += 1.0
    del changed["trace"]["h.raw"]
    result = replay_native_record(changed)
    assert result["status"] == "mismatch"
    paths = {item["path"] for item in result["mismatches"]}
    assert "/outputs/0/0" in paths
    assert "/trace/h.raw" in paths


def test_identity_and_unsupported_routing_rejected():
    record = fixture_record()
    record["parameters"]["h"]["centers"][0][0] += 1.0
    with pytest.raises(ValueError, match="identity"):
        model_from_snapshot(record)
    record = fixture_record()
    record["configuration"]["routing"]["p"] = ["h"]
    record["model_sha256"] = hashlib.sha256(
        json.dumps(
            {key: record[key] for key in ("configuration", "parameters")},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="configuration"):
        model_from_snapshot(record)
    record = fixture_record()
    record["dataset"]["samples"][0]["inputs"][0] = 2.0
    with pytest.raises(ValueError, match="dataset identity"):
        replay_native_record(record)


def test_cli_model_definition_collect_and_inspect(tmp_path, capsys):
    from nmn.research.native_cli import main

    model = tmp_path / "model.json"
    dataset = tmp_path / "dataset.json"
    observations = tmp_path / "observations.json"
    dataset.write_text(json.dumps(fixture_record()["dataset"]))
    assert (
        main(["init", "--backend", "nnx", "--dtype", "float32", "--output", str(model)])
        == 0
    )
    definition = json.loads(model.read_text())
    assert definition["schema"] == "nmn.nnx-model.v1"
    assert "outputs" not in definition
    assert (
        main(
            [
                "collect",
                "--model",
                str(model),
                "--dataset",
                str(dataset),
                "--split",
                "evaluation",
                "--no-derivatives",
                "--output",
                str(observations),
            ]
        )
        == 0
    )
    assert json.loads(observations.read_text())["sample_ids"] == ["eval"]
    assert main(["inspect", "--model", str(model)]) == 0
    assert (
        main(["init", "--backend", "nnx", "--dtype", "float32", "--output", str(model)])
        == 2
    )
    capsys.readouterr()
