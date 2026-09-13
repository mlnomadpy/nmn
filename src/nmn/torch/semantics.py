"""Compare supplied semantic correspondence against actual native donor replay."""

import hashlib
import json
import math
from pathlib import Path

from ..research.datasets import DonorPair
from .studies import donor_study


def semantic_study(model, dataset, *, reference, correspondence, tolerance=1e-8):
    """Measure baseline and counterfactual agreement for a declared alignment.

    Correspondence maps every reference variable to native whole-write modules.
    Its origin, anchors and ambiguity are retained, never promoted to discovery.
    This finite table workflow does not load Python reference code from JSON.
    """
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("tolerance must be finite and nonnegative")
    correspondence = json.loads(json.dumps(correspondence, allow_nan=False))
    table = reference.to_dict()
    mapping = correspondence["mapping"]
    if correspondence.get("origin") not in ("supplied", "supervised", "inferred"):
        raise ValueError(
            "declare correspondence origin: supplied, supervised or inferred"
        )
    if (
        not isinstance(correspondence.get("provenance"), str)
        or not correspondence["provenance"]
    ):
        raise ValueError("correspondence provenance is required")
    if not all(
        isinstance(correspondence.get(k), list) for k in ("anchors", "ambiguities")
    ):
        raise ValueError(
            "declare anchors and ambiguities as lists, including when empty"
        )
    if set(mapping) != set(table["variables"]):
        raise ValueError("correspondence must cover exactly the reference variables")
    for modules in mapping.values():
        if (
            not isinstance(modules, list)
            or not modules
            or len(set(modules)) != len(modules)
            or not set(modules) <= set(model.state_names)
        ):
            raise ValueError("each variable maps to unique native module names")
    pairs = []
    for case in table["cases"]:
        modules = tuple(
            dict.fromkeys(
                m for variable in case["variables"] for m in mapping[variable]
            )
        )
        pairs.append(
            DonorPair(
                case["case_id"],
                case["base_id"],
                case["donor_id"],
                modules,
                case["outputs"],
            )
        )
    study = donor_study(model, dataset, pairs)
    snapshot = study["model_snapshot"]
    indices = {name: i for i, name in enumerate(model.output_names)}
    baseline = []
    for sid, outputs in zip(
        snapshot["sample_ids"], snapshot["observations"]["baseline"]
    ):
        expected = reference.evaluate(sid)
        if not set(expected) <= set(indices):
            raise ValueError("reference outputs must name native output coordinates")
        errors = {
            name: abs(outputs[indices[name]] - value)
            for name, value in expected.items()
        }
        baseline.append(
            dict(
                sample_id=sid,
                split=dataset.sample(sid).split,
                expected=expected,
                absolute_error=errors,
                agrees=all(
                    math.isfinite(v) and v <= tolerance for v in errors.values()
                ),
            )
        )
    rows = []
    for case, row in zip(table["cases"], study["rows"]):
        errors = row["absolute_reference_error"]
        agrees = all(
            isinstance(v, (int, float)) and math.isfinite(v) and v <= tolerance
            for v in errors.values()
        )
        rows.append(
            dict(
                case_id=case["case_id"],
                variables=case["variables"],
                native_modules=list(row["pair"]["modules"]),
                base_id=case["base_id"],
                donor_id=case["donor_id"],
                base_split=row["base_split"],
                donor_split=row["donor_split"],
                expected=row["expected"],
                edited_outputs=row["edited_outputs"],
                absolute_error=errors,
                agrees=agrees,
            )
        )
    return dict(
        schema="nmn.semantic-study.v1",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status=(
            "observed-agreement"
            if all(row["agrees"] for row in baseline + rows)
            else "counterfactual-or-baseline-disagreement"
        ),
        reference=table,
        correspondence=correspondence,
        tolerance=tolerance,
        dataset=dataset.to_dict(),
        dataset_sha256=dataset.sha256,
        model_snapshot=snapshot,
        donor_execution=study,
        baseline=baseline,
        cases=rows,
        coverage=dict(
            baseline_samples=len(baseline),
            reference_baseline_samples=len(table["baseline"]),
            counterfactual_cases=len(rows),
        ),
        limitations=[
            "Finite supplied-reference agreement does not identify a unique internal mechanism.",
            "Correspondence origin is a caller declaration, not established by this study.",
            "Only listed counterfactual contexts are checked; no held-out independence is inferred.",
            "Native actions replace whole module writes; read-slot/path correspondences are unsupported.",
        ],
    )
