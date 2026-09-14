"""Pointwise native-edit comparisons, distinct from domain verification."""

from fractions import Fraction
from typing import Any, Dict, Optional

from .reference import trace


def compare(
    u: Fraction,
    v: Fraction,
    gate: Fraction = Fraction(0),
    replacement_h: Optional[Fraction] = None,
) -> Dict[str, Any]:
    """Compare to the unedited gate-one model at exactly one input."""
    baseline = trace(u, v)
    edited = trace(u, v, gate, replacement_h)
    deltas = {
        name: str(Fraction(value) - Fraction(baseline["outputs"][name]))
        for name, value in edited["outputs"].items()
    }
    return {
        "schema": "nmn.three-neuron-comparison.v1",
        "scope": "one specified input and native action; exact rational execution",
        "baseline": baseline,
        "edited": edited,
        "output_deltas": deltas,
        "target_changed": deltas["target"] != "0",
        "protected_unchanged": deltas["protected"] == "0",
        "leaky_readout_unchanged": deltas["leaky"] == "0",
        "interpretation": (
            "Target change is not automatically target success. "
            "This comparison does not establish a domain-wide guarantee."
        ),
    }
