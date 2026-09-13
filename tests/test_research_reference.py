"""Behavioral checks for the exact reference and evidence lifecycle."""

import json
from fractions import Fraction as F

import pytest

from nmn.cli import main
from nmn.research.reference import experiment, rational, replay, trace, write_bundle


def test_known_intervention_and_full_suffix():
    baseline = trace(F(1), F(1))
    edited = trace(F(1), F(1), F(0))
    assert baseline["outputs"] == {"target": "4", "protected": "1", "leaky": "5"}
    assert edited["outputs"] == {"target": "1/2", "protected": "1", "leaky": "3/2"}
    assert trace(F(1), F(1), F(0), F(1))["outputs"] == baseline["outputs"]
    assert edited["layer2"]["reads"]["h"] == "0"


def test_exhaustive_coverage_and_zero_response():
    data = experiment()
    inputs = {
        (c["baseline"]["input"]["u"], c["baseline"]["input"]["v"])
        for c in data["cases"]
    }
    assert len(inputs) == 25
    assert data["results"]["protected_violations"] == 0
    assert data["results"]["target_changes"] == 20
    assert data["results"]["target_witness_pass"]
    assert trace(F(0), F(1), F(0))["outputs"] == trace(F(0), F(1))["outputs"]


@pytest.mark.parametrize(
    "value", ["nan", "inf", "1e999999999", "1/0", "-1", "2", "x", "1" * 65]
)
def test_invalid_inputs(value):
    with pytest.raises(ValueError):
        rational(value)


def test_bundle_replay_and_tamper(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    assert replay(bundle)["status"] == "reproduced"
    with pytest.raises(ValueError, match="already exists"):
        write_bundle(bundle)
    (bundle / "Report.md").write_text("incorrect explanation")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay(bundle)


def test_forged_hash_does_not_replace_recomputation(tmp_path):
    from nmn.research.reference import digest

    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    path = bundle / "evidence.json"
    evidence = json.loads(path.read_text())
    evidence["results"]["protected_violations"] = 999
    path.write_text(json.dumps(evidence))
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"]["evidence.json"] = digest(path.read_bytes())
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recomputed"):
        replay(bundle)


def test_cli_outcomes_and_export(tmp_path, capsys):
    assert main(["research", "verify"]) == 0
    assert main(["research", "verify", "--leaky"]) == 1
    assert main(["research", "trace", "--u", "nan"]) == 2
    bundle, export = tmp_path / "bundle", tmp_path / "vault" / "reference"
    assert main(["research", "demo", "--output", str(bundle)]) == 0
    assert main(["research", "export", str(bundle), "--output", str(export)]) == 0
    assert (export / "Report.md").exists()
    assert replay(export)["status"] == "reproduced"
    assert "research error" in capsys.readouterr().err


def test_evaluator_mismatch(tmp_path):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["evaluator_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evaluator differs"):
        replay(bundle)


@pytest.mark.parametrize(
    "manifest", [[], {"schema": "nmn.three-neuron.v1", "files": []}]
)
def test_malformed_manifest_is_cli_error(tmp_path, manifest):
    bundle = tmp_path / "bad"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    assert main(["research", "reproduce", str(bundle)]) == 2


def test_research_imports_no_backend():
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = "from nmn.cli import main; main(['research', 'verify']); import sys; assert not ({'numpy', 'torch', 'jax', 'flax', 'tensorflow', 'keras', 'mlx'} & set(sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_comparison_exact_deltas_and_replacement():
    from nmn.research.comparison import compare

    changed = compare(F(1), F(1))
    assert changed["output_deltas"] == {
        "target": "-7/2",
        "protected": "0",
        "leaky": "-7/2",
    }
    assert changed["protected_unchanged"]
    assert not changed["leaky_readout_unchanged"]
    assert compare(F(1), F(1), F(0), F(1))["output_deltas"]["target"] == "0"
    assert not compare(F(0), F(1))["target_changed"]


def test_compare_cli_fractional_gate(capsys):
    assert main(["research", "compare", "--gate", "1/2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["edited"]["outputs"]["target"] == "9/5"
    assert result["output_deltas"]["target"] == "-11/5"
    assert result["protected_unchanged"]
    assert "one specified input" in result["scope"]
