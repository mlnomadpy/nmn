"""Cross-reference and numeric rejection for backend-free graph specifications."""

import copy
import json
from pathlib import Path

import pytest

from nmn.research.architecture import validate_architecture


def fixture():
    return json.loads(
        (
            Path(__file__).parents[1] / "examples/research/linear-protected-graph.json"
        ).read_text()
    )


def test_normalization_is_idempotent_and_does_not_mutate_input():
    config = fixture()
    del config["layers"][0][0]["epsilon"]
    original = copy.deepcopy(config)
    record = validate_architecture(config)
    assert config == original
    assert record == validate_architecture(record["configuration"])
    assert record["dimensions"][2]["center_or_hidden_weight_shape"] == [1, 2]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("reads", ["missing"], "reads"),
        ("epsilon", 0.0, "epsilon"),
        ("epsilon", float("nan"), "epsilon"),
        ("num_centers", True, "num_centers"),
        ("name", "y", "globally unique"),
    ],
)
def test_invalid_module_fields_report_location(field, value, message):
    config = fixture()
    config["layers"][0][0][field] = value
    with pytest.raises(ValueError, match=message):
        validate_architecture(config)
