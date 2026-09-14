"""Explicit finite semantic reference tables, independent of ML backends."""

import copy
import json
import math


class TabulatedReference:
    """Supplied baseline and counterfactual labels; no inferred semantic meaning.

    Cases identify base/donor IDs and semantic variables transferred together.
    Tables need not cover any unlisted context. They are reference data, not
    proof that a semantic model or correspondence is correct.
    """

    def __init__(self, record):
        record = json.loads(json.dumps(record, allow_nan=False))
        if record.get("schema") != "nmn.semantic-reference.v1":
            raise ValueError("unsupported semantic reference schema")
        if not all(
            isinstance(record.get(k), str) and record[k]
            for k in ("reference_id", "provenance")
        ):
            raise ValueError("reference identity and provenance are required")
        variables = record["variables"]
        if (
            not isinstance(variables, list)
            or not variables
            or any(not isinstance(v, str) or not v for v in variables)
            or len(set(variables)) != len(variables)
        ):
            raise ValueError("semantic variables must be unique nonempty names")

        def outputs(values):
            if (
                not isinstance(values, dict)
                or not values
                or any(
                    not isinstance(k, str)
                    or not k
                    or isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for k, v in values.items()
                )
            ):
                raise ValueError("reference outputs must be named finite numbers")

        if not isinstance(record["baseline"], dict) or not record["baseline"]:
            raise ValueError("baseline reference table must be nonempty")
        for sid, values in record["baseline"].items():
            if not isinstance(sid, str) or not sid:
                raise ValueError("baseline IDs must be nonempty")
            outputs(values)
        seen = set()
        if not record["cases"]:
            raise ValueError("supply reference counterfactual cases")
        for case in record["cases"]:
            if (
                not isinstance(case["case_id"], str)
                or not case["case_id"]
                or case["case_id"] in seen
            ):
                raise ValueError("case IDs must be unique and nonempty")
            seen.add(case["case_id"])
            if (
                case["base_id"] not in record["baseline"]
                or case["donor_id"] not in record["baseline"]
            ):
                raise ValueError("case sample IDs require baseline reference outputs")
            selected = case["variables"]
            if (
                not isinstance(selected, list)
                or not selected
                or len(set(selected)) != len(selected)
                or not set(selected) <= set(variables)
            ):
                raise ValueError("case variables must be unique declared variables")
            outputs(case["outputs"])
        self._record = record

    def to_dict(self):
        return copy.deepcopy(self._record)

    def evaluate(self, sample_id):
        return copy.deepcopy(self._record["baseline"][sample_id])

    def counterfactual(self, case_id):
        return copy.deepcopy(
            next(c for c in self._record["cases"] if c["case_id"] == case_id)
        )
