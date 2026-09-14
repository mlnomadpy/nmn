"""Model schema, parameter effects and contract-scope regressions."""

import copy
import json
from fractions import Fraction as F

import pytest

from nmn.cli import main
from nmn.research.model import default_model, load, model_trace, model_verify, validate
from nmn.research.reference import trace


def test_default_matches_original_all_grid_and_gates():
    model = default_model()
    for i in range(5):
        for j in range(5):
            for gate in (F(0), F(1, 2), F(1)):
                expected = trace(F(i, 4), F(j, 4), gate)
                actual = model_trace(model, F(i, 4), F(j, 4), gate)
                for key in ("target", "protected"):
                    assert actual["outputs"][key] == expected["outputs"][key]


def test_leak_and_changed_parameters_are_executed():
    model = default_model()
    model["protected_leak"] = "1"
    result = model_verify(model)
    assert result["status"] == "counterexample-found"
    assert result["protected_violations"] == 20
    model["neurons"]["y"]["coefficient"] = "-2"
    assert model_trace(model, F(1), F(1))["outputs"] == {
        "target": "-8",
        "protected": "-7",
    }
    model["epsilon"] = "2"
    assert model_trace(model, F(1), F(1))["outputs"]["target"] == "-2"


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m.update(epsilon="0"),
        lambda m: m.update(epsilon="1e999999999"),
        lambda m: m.update(epsilon="-1"),
        lambda m: m.update(epsilon=1.0),
        lambda m: m.update(extra=True),
        lambda m: m["neurons"]["y"].update(center=["1"]),
        lambda m: m["neurons"]["h"].update(coefficient="nan"),
        lambda m: m.update(protected_leak=True),
    ],
)
def test_schema_rejects_bad_models(edit):
    model = default_model()
    edit(model)
    with pytest.raises(ValueError):
        validate(model)


def test_normalization_and_model_identity():
    a = default_model()
    b = copy.deepcopy(a)
    b["epsilon"] = "2/2"
    assert validate(a) == validate(b)
    assert (
        model_trace(a, F(1), F(1))["model_sha256"]
        == model_trace(b, F(1), F(1))["model_sha256"]
    )


def test_file_rejects_duplicate_keys_and_oversized_data(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError, match="duplicate"):
        load(path)
    path.write_text(" " * 65537)
    with pytest.raises(ValueError, match="64 KiB"):
        load(path)


def test_cli_configured_scope_and_exit_codes(tmp_path, capsys):
    path = tmp_path / "model.json"
    model = default_model()
    path.write_text(json.dumps(model))
    assert main(["research", "verify", "--model", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["coverage"] == 25
    assert "no target-success requirement" in result["contract"]["target"]
    model["protected_leak"] = "1"
    path.write_text(json.dumps(model))
    assert main(["research", "verify", "--model", str(path)]) == 1
    assert main(["research", "verify", "--model", str(path), "--leaky"]) == 2


def test_inspection_uses_configured_parameters_and_tracks_leak():
    from nmn.research.inspection import inspect_model

    model = default_model()
    model["epsilon"] = "2"
    assert inspect_model(model)["neurons"][0]["epsilon"] == "2"
    assert inspect_model(model)["h_to_protected_paths"] == []
    model["protected_leak"] = "1"
    assert inspect_model(model)["h_to_protected_paths"] == [["h", "y", "protected"]]


def test_computation_has_a_nonadditive_interaction():
    from fractions import Fraction as F

    model = default_model()

    def y(u, v):
        return F(model_trace(model, F(u), F(v))["outputs"]["target"])

    assert y(1, 1) - y(0, 1) - y(1, 0) + y(0, 0) == 3
