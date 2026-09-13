"""Validated JSON parameters for the fixed, explicit three-neuron topology."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional

from .reference import digest, encoded, rational

SCHEMA = "nmn.three-neuron-model.v1"


def _fields(value: Any, expected: set, location: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location}: expected exactly {sorted(expected)}")


def _number(value: Any, location: str) -> Fraction:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError(
            f"{location}: expected a rational string of at most 64 characters"
        )
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{location}: expected a finite rational string") from exc


def validate(data: Any) -> Dict[str, Any]:
    """Normalize all rationals; reject extra keys and unsupported topology."""
    _fields(data, {"schema", "epsilon", "neurons", "protected_leak"}, "model")
    if data["schema"] != SCHEMA:
        raise ValueError("unsupported model schema")
    epsilon = _number(data["epsilon"], "epsilon")
    if epsilon <= 0:
        raise ValueError("epsilon: must be strictly positive")
    _fields(data["neurons"], {"h", "p", "y"}, "neurons")
    neurons = {}
    for name, dimension in (("h", 1), ("p", 1), ("y", 2)):
        neuron = data["neurons"][name]
        _fields(neuron, {"center", "coefficient"}, f"neurons.{name}")
        center = neuron["center"]
        if not isinstance(center, list) or len(center) != dimension:
            raise ValueError(f"neurons.{name}.center: expected {dimension} coordinates")
        neurons[name] = {
            "center": [str(_number(x, f"neurons.{name}.center")) for x in center],
            "coefficient": str(
                _number(neuron["coefficient"], f"neurons.{name}.coefficient")
            ),
        }
    return {
        "schema": SCHEMA,
        "epsilon": str(epsilon),
        "neurons": neurons,
        "protected_leak": str(_number(data["protected_leak"], "protected_leak")),
    }


def default_model() -> Dict[str, Any]:
    return validate(
        {
            "schema": SCHEMA,
            "epsilon": "1",
            "neurons": {
                "h": {"center": ["1"], "coefficient": "1"},
                "p": {"center": ["1"], "coefficient": "1"},
                "y": {"center": ["1", "1"], "coefficient": "1"},
            },
            "protected_leak": "0",
        }
    )


def _unique_pairs(pairs: list) -> dict:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> Dict[str, Any]:
    with path.open("rb") as stream:
        content = stream.read(65537)
    if len(content) > 65536:
        raise ValueError("model file exceeds 64 KiB")
    return validate(json.loads(content, object_pairs_hook=_unique_pairs))


def model_trace(
    model: Dict[str, Any],
    u: Fraction,
    v: Fraction,
    gate: Fraction = Fraction(1),
    replacement_h: Optional[Fraction] = None,
) -> Dict[str, Any]:
    model = validate(model)
    for value in (u, v, gate):
        if not isinstance(value, Fraction):
            raise ValueError("inputs and gates must be Fraction values")
        rational(str(value))
    if replacement_h is not None:
        if not isinstance(replacement_h, Fraction):
            raise ValueError("replacement h must be a Fraction value")
        rational(str(replacement_h))
    epsilon = Fraction(model["epsilon"])

    def neuron(name: str, inputs: tuple) -> Fraction:
        spec = model["neurons"][name]
        center = [Fraction(x) for x in spec["center"]]
        dot = sum((c * x for c, x in zip(center, inputs)), Fraction(0))
        denominator = epsilon + sum(
            ((c - x) ** 2 for c, x in zip(center, inputs)), Fraction(0)
        )
        return Fraction(spec["coefficient"]) * dot**2 / denominator

    h_raw = neuron("h", (u,))
    h = h_raw * gate if replacement_h is None else replacement_h
    p = neuron("p", (v,))
    y = neuron("y", (h, v))
    return {
        "model_sha256": digest(encoded(model)),
        "input": {"u": str(u), "v": str(v)},
        "action": {
            "gate": str(gate),
            "replacement_h": None if replacement_h is None else str(replacement_h),
        },
        "layer1": {
            "h_section": str(h_raw),
            "h_gated": str(h_raw * gate),
            "h": str(h),
            "p": str(p),
        },
        "layer2": {"reads": {"h": str(h), "v": str(v)}, "y": str(y)},
        "outputs": {
            "target": str(y),
            "protected": str(p + Fraction(model["protected_leak"]) * y),
        },
    }


def model_compare(
    model: Dict[str, Any],
    u: Fraction,
    v: Fraction,
    gate: Fraction = Fraction(0),
    replacement_h: Optional[Fraction] = None,
) -> Dict[str, Any]:
    model = validate(model)
    baseline = model_trace(model, u, v)
    edited = model_trace(model, u, v, gate, replacement_h)
    deltas = {
        key: str(Fraction(value) - Fraction(baseline["outputs"][key]))
        for key, value in edited["outputs"].items()
    }
    return {
        "schema": "nmn.configured-comparison.v1",
        "model": model,
        "scope": "one input and native edit; exact rational arithmetic",
        "baseline": baseline,
        "edited": edited,
        "output_deltas": deltas,
        "protected_unchanged": deltas["protected"] == "0",
        "target_changed": deltas["target"] != "0",
    }


def model_verify(model: Dict[str, Any]) -> Dict[str, Any]:
    """Only protection is contracted here; target changes are descriptive."""
    model = validate(model)
    grid = [Fraction(i, 4) for i in range(5)]
    cases = [model_compare(model, u, v) for u, v in product(grid, repeat=2)]
    violations = [case for case in cases if not case["protected_unchanged"]]
    return {
        "schema": "nmn.configured-verification.v1",
        "model": model,
        "model_sha256": digest(encoded(model)),
        "contract": {
            "input_grid": [str(x) for x in grid],
            "action": "shared gate 1 -> 0",
            "protected": "p + protected_leak*y; exact equality",
            "scope": "25 explicit inputs; exact rational execution",
            "target": "no target-success requirement; changes are descriptive only",
        },
        "status": (
            "counterexample-found" if violations else "certified-under-assumptions"
        ),
        "coverage": len(cases),
        "protected_violations": len(violations),
        "target_changes": sum(case["target_changed"] for case in cases),
        "counterexample": violations[0] if violations else None,
        "cases": cases,
    }
