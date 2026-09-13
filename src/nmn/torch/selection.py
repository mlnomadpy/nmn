"""Finite candidate selection followed by frozen-edit validation."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from ..research.selection import SelectionLedger
from .interpretable import Intervention
from .research import _json_value, collect_research_data


def select_edit(
    model,
    dataset,
    *,
    candidates,
    targets,
    target_outputs,
    protected_outputs,
    protection_tolerance,
    provenance,
    max_candidates,
    selection_split="tuning",
    validation_split="validation",
):
    """Choose a feasible edit by selection MSE, then evaluate only that edit.

    Feasibility is per-coordinate absolute protection change on selection data.
    Ties use lexicographic candidate IDs. Validation cannot change the winner.
    Candidate limits cover sorted candidates; unexecuted candidates stay visible.
    No optimizer or statistical/population certificate is invoked.
    """
    candidates = json.loads(json.dumps(candidates, allow_nan=False))
    targets = json.loads(json.dumps(targets, allow_nan=False))
    if (
        selection_split == validation_split
        or not selection_split
        or not validation_split
    ):
        raise ValueError(
            "selection and validation splits must be distinct and nonempty"
        )
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if not candidates or any(not isinstance(k, str) or not k for k in candidates):
        raise ValueError("provide nonempty named candidates")
    if not isinstance(provenance, str) or not provenance:
        raise ValueError("target/protocol provenance is required")
    if (
        isinstance(protection_tolerance, bool)
        or not isinstance(protection_tolerance, (int, float))
        or not math.isfinite(protection_tolerance)
        or protection_tolerance < 0
    ):
        raise ValueError("protection tolerance must be finite and nonnegative")
    for names in (target_outputs, protected_outputs):
        if (
            isinstance(names, str)
            or len(set(names)) != len(names)
            or not set(names) <= set(model.output_names)
        ):
            raise ValueError(
                "target/protected outputs must be unique native output names"
            )
    if not target_outputs or set(target_outputs) & set(protected_outputs):
        raise ValueError(
            "target outputs must be nonempty and disjoint from protected outputs"
        )
    ids = dataset.sample_ids(split=selection_split)
    heldout = dataset.sample_ids(split=validation_split)
    if not ids or not heldout or set(targets) != set(ids) | set(heldout):
        raise ValueError(
            "targets must cover exactly nonempty selection and validation populations"
        )
    for values in targets.values():
        if set(values) != set(target_outputs) or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values.values()
        ):
            raise ValueError(
                "targets must provide finite values for exactly the target outputs"
            )
    parameter = next(model.parameters())

    def inputs(population):
        return torch.tensor(
            [dataset.sample(s).inputs for s in population],
            dtype=parameter.dtype,
            device=parameter.device,
        )

    selection_inputs = inputs(ids)
    snapshot = collect_research_data(
        model, selection_inputs, sample_ids=ids, derivatives=False
    )
    ledger = SelectionLedger(
        dataset,
        model_sha256=snapshot["model_sha256"],
        selection_splits=(selection_split,),
        validation_splits=(validation_split,),
    )
    for name in sorted(candidates):
        ledger.register(name, candidates[name])
    ti = [model.output_names.index(name) for name in target_outputs]
    pi = [model.output_names.index(name) for name in protected_outputs]

    def evaluate(name, population, x, baseline):
        try:
            controls = {
                module: Intervention(**value)
                for module, value in candidates[name].items()
            }
            with torch.no_grad():
                output, trace = model.forward_with_trace(x, controls)
                expected = output.new_tensor(
                    [[targets[s][k] for k in target_outputs] for s in population]
                )
                squared_error = (output[:, ti] - expected).square()
                delta = output[:, pi] - baseline[:, pi]
                if not bool(
                    torch.isfinite(output).all()
                    and torch.isfinite(squared_error).all()
                    and torch.isfinite(delta).all()
                ):
                    raise ValueError("nonfinite candidate measurements")
                return _json_value(
                    dict(
                        status="measured",
                        outputs=output,
                        trace=trace,
                        target_squared_error=squared_error,
                        target_mse=squared_error.mean(),
                        protected_delta=delta,
                        protection_satisfied=bool(
                            (delta.abs() <= protection_tolerance).all()
                        ),
                    )
                )
        except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as exc:
            return dict(status="failed", error=str(exc))

    baseline = parameter.new_tensor(snapshot["observations"]["baseline"])
    rows: dict[str, Any] = {}
    for name in sorted(candidates):
        if len(rows) >= max_candidates:
            break
        row = evaluate(name, ids, selection_inputs, baseline)
        rows[name] = row
        ledger.record(
            name,
            phase="selection",
            sample_ids=ids,
            measurements={
                k: v for k, v in row.items() if k not in ("trace", "outputs")
            },
            costs={"attempted_forward_calls": 1},
        )
    feasible = [
        name
        for name, row in rows.items()
        if row["status"] == "measured" and row["protection_satisfied"]
    ]
    selected = (
        min(feasible, key=lambda name: (rows[name]["target_mse"], name))
        if feasible
        else None
    )
    validation = None
    validation_snapshot = None
    if selected is not None:
        ledger.freeze(
            selected,
            rule="minimum selection target MSE among protection-feasible measured candidates; ties by candidate ID",
        )
        x = inputs(heldout)
        validation_snapshot = collect_research_data(
            model, x, sample_ids=heldout, derivatives=False
        )
        validation = evaluate(
            selected,
            heldout,
            x,
            parameter.new_tensor(validation_snapshot["observations"]["baseline"]),
        )
        ledger.record(
            selected,
            phase="validation",
            sample_ids=heldout,
            measurements={
                k: v for k, v in validation.items() if k not in ("trace", "outputs")
            },
            costs={"attempted_forward_calls": 1},
        )
    return dict(
        schema="nmn.edit-selection.v1",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status="selected" if selected is not None else "no-feasible-measured-candidate",
        dataset=dataset.to_dict(),
        dataset_sha256=dataset.sha256,
        model_snapshot=snapshot,
        candidates=candidates,
        targets=targets,
        protocol=dict(
            target_outputs=list(target_outputs),
            protected_outputs=list(protected_outputs),
            protection_tolerance=protection_tolerance,
            provenance=provenance,
            max_candidates=max_candidates,
            selection_split=selection_split,
            validation_split=validation_split,
        ),
        selection=rows,
        selected=selected,
        unexecuted=[name for name in sorted(candidates) if name not in rows],
        validation=validation,
        validation_snapshot=validation_snapshot,
        ledger=ledger.to_dict(),
        limitations=[
            "Finite empirical selection; no independence, population-risk or simultaneous statistical guarantee.",
            "Selection feasibility is measured only on selection samples; validation cannot change the winner.",
            "Unexecuted and failed candidates are not assumed infeasible.",
            "Target and protection declarations are supplied, not inferred semantics.",
        ],
    )
