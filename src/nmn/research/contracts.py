"""Finite-domain intervention contracts for the configurable reference model."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from itertools import islice, product
from pathlib import Path
from typing import Any, Dict

from .model import _fields, _unique_pairs, model_trace
from .model import validate as validate_model
from .reference import digest, encoded

SCHEMA = "nmn.finite-intervention-contract.v1"


def _fraction(value: Any, field: str, unit: bool = False) -> Fraction:
    # Exponent notation is deliberately excluded to keep parsing resource-bounded.
    if (
        not isinstance(value, str)
        or len(value) > 64
        or not re.fullmatch(r"[+-]?(?:\d+(?:/\d+|\.\d*)?|\.\d+)", value)
    ):
        raise ValueError(f"{field}: expected a bounded rational/decimal string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{field}: invalid rational") from exc
    if result < 0 or (unit and result > 1):
        raise ValueError(
            f"{field}: must be nonnegative" + (" and at most 1" if unit else "")
        )
    return result


def validate(data: Any) -> Dict[str, Any]:
    _fields(
        data, {"schema", "inputs", "gate", "protected_tolerance", "target"}, "contract"
    )
    if data["schema"] != SCHEMA:
        raise ValueError("unsupported contract schema")
    _fields(data["inputs"], {"u", "v"}, "inputs")
    inputs = {}
    for name in ("u", "v"):
        values = data["inputs"][name]
        if not isinstance(values, list) or not 1 <= len(values) <= 64:
            raise ValueError(f"inputs.{name}: expected 1 to 64 values")
        numbers = [_fraction(x, f"inputs.{name}", unit=True) for x in values]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f"inputs.{name}: duplicate rational values")
        inputs[name] = [str(x) for x in sorted(numbers)]
    _fields(data["gate"], {"baseline", "edited"}, "gate")
    gates = {
        name: str(_fraction(data["gate"][name], f"gate.{name}", unit=True))
        for name in ("baseline", "edited")
    }
    target = data["target"]
    normalized_target = None
    if target is not None:
        _fields(target, {"u", "v", "minimum_decrease"}, "target")
        normalized_target = {
            name: str(_fraction(target[name], f"target.{name}", unit=True))
            for name in ("u", "v")
        }
        if any(normalized_target[name] not in inputs[name] for name in ("u", "v")):
            raise ValueError("target witness must belong to the declared input grid")
        normalized_target["minimum_decrease"] = str(
            _fraction(target["minimum_decrease"], "target.minimum_decrease")
        )
    return {
        "schema": SCHEMA,
        "inputs": inputs,
        "gate": gates,
        "protected_tolerance": str(
            _fraction(data["protected_tolerance"], "protected_tolerance")
        ),
        "target": normalized_target,
    }


def default_contract() -> Dict[str, Any]:
    grid = ["0", "1/4", "1/2", "3/4", "1"]
    return validate(
        {
            "schema": SCHEMA,
            "inputs": {"u": grid, "v": grid},
            "gate": {"baseline": "1", "edited": "0"},
            "protected_tolerance": "0",
            "target": {"u": "1", "v": "1", "minimum_decrease": "7/2"},
        }
    )


def load_contract(path: Path) -> Dict[str, Any]:
    with path.open("rb") as stream:
        content = stream.read(65537)
    if len(content) > 65536:
        raise ValueError("contract exceeds 64 KiB")
    return validate(json.loads(content, object_pairs_hook=_unique_pairs))


def check(
    model: Dict[str, Any], contract: Dict[str, Any], max_cases: int = 4096
) -> Dict[str, Any]:
    model, contract = validate_model(model), validate(contract)
    if type(max_cases) is not int or not 1 <= max_cases <= 4096:
        raise ValueError("max_cases must be an integer in [1, 4096]")
    cases = []
    first_failure = None
    target_status = "not-requested" if contract["target"] is None else "not-evaluated"
    target = contract["target"]
    tolerance = Fraction(contract["protected_tolerance"])
    violations = 0
    grid = product(contract["inputs"]["u"], contract["inputs"]["v"])
    for u, v in islice(grid, max_cases):
        before = model_trace(
            model, Fraction(u), Fraction(v), Fraction(contract["gate"]["baseline"])
        )
        after = model_trace(
            model, Fraction(u), Fraction(v), Fraction(contract["gate"]["edited"])
        )
        deviation = abs(
            Fraction(after["outputs"]["protected"])
            - Fraction(before["outputs"]["protected"])
        )
        decrease = Fraction(before["outputs"]["target"]) - Fraction(
            after["outputs"]["target"]
        )
        protected_pass = deviation <= tolerance
        reasons = []
        if not protected_pass:
            violations += 1
            reasons.append("protected tolerance exceeded")
        if target is not None and u == target["u"] and v == target["v"]:
            target_status = (
                "passed"
                if decrease >= Fraction(target["minimum_decrease"])
                else "failed"
            )
            if target_status == "failed":
                reasons.append("target minimum decrease not met")
        case = {
            "baseline": before,
            "edited": after,
            "protected_deviation": str(deviation),
            "target_decrease": str(decrease),
            "violations": reasons,
        }
        cases.append(case)
        if reasons and first_failure is None:
            first_failure = case
    total = len(contract["inputs"]["u"]) * len(contract["inputs"]["v"])
    complete = len(cases) == total
    status = (
        "counterexample-found"
        if first_failure
        else ("certified-under-assumptions" if complete else "inconclusive")
    )
    return {
        "schema": "nmn.finite-contract-evidence.v1",
        "model": model,
        "contract": contract,
        "model_sha256": digest(encoded(model)),
        "contract_sha256": digest(encoded(contract)),
        "scope": "declared finite grid and one shared gate edit; exact rational arithmetic",
        "status": status,
        "cases_checked": len(cases),
        "cases_total": total,
        "coverage_complete": complete,
        "case_budget": max_cases,
        "protected_violations_observed": violations,
        "target_status": target_status,
        "counterexample": first_failure,
        "cases": cases,
        "limitations": "No continuous-domain, floating-point, learned-semantic or independent-proof guarantee.",
    }
