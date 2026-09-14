"""Reference tables preserve supplied context and reject ambiguous case IDs."""

import pytest

from nmn.research.semantics import TabulatedReference


def test_reference_is_copied_and_cases_are_unique():
    data = {
        "schema": "nmn.semantic-reference.v1",
        "reference_id": "copy-v1",
        "provenance": "supplied",
        "variables": ["x"],
        "baseline": {"a": {"y": 1}},
        "cases": [
            {
                "case_id": "self",
                "base_id": "a",
                "donor_id": "a",
                "variables": ["x"],
                "outputs": {"y": 1},
            }
        ],
    }
    reference = TabulatedReference(data)
    reference.evaluate("a")["y"] = 0
    assert reference.counterfactual("self")["outputs"]["y"] == 1
    assert reference.evaluate("a")["y"] == 1
    data["cases"].append(data["cases"][0])
    with pytest.raises(ValueError, match="unique"):
        TabulatedReference(data)
