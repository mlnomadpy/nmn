"""Declared classification protection measurements from actual native replay."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .research import _json_value, collect_research_data, protection_metrics


def protection_study(
    model, dataset, *, edits, tasks, provenance, split=None, strata=()
):
    """Measure each declared task separately, retaining eligibility and raw data.

    A task declares ``outputs``, ``rule`` (argmax or threshold), and ``labels``
    keyed by selected sample ID. Threshold uses >=; argmax ties use the first
    declared output. Labels are indices in that declared order. Strata name
    supplied semantic fields; missing values form an explicit separate stratum.
    No task is inferred, and no population guarantee or edit selection is made.
    """
    ids = dataset.sample_ids(split=split)
    if (
        not ids
        or not edits
        or not tasks
        or not isinstance(provenance, str)
        or not provenance
    ):
        raise ValueError("require samples, edits, tasks and label/protocol provenance")
    if len(set(strata)) != len(strata) or any(
        not isinstance(s, str) or not s for s in strata
    ):
        raise ValueError("strata must be unique nonempty semantic field names")
    normalized = {}
    for name, task in tasks.items():
        if not isinstance(name, str) or not name:
            raise ValueError("task names must be nonempty strings")
        outputs, rule, labels = task["outputs"], task["rule"], task["labels"]
        if (
            not isinstance(outputs, (list, tuple))
            or not outputs
            or len(set(outputs)) != len(outputs)
            or not set(outputs) <= set(model.output_names)
        ):
            raise ValueError("task outputs must be unique model output names")
        if (
            rule not in ("threshold", "argmax")
            or (rule == "threshold" and len(outputs) != 1)
            or (rule == "argmax" and len(outputs) < 2)
        ):
            raise ValueError(
                "threshold requires one output; argmax requires at least two"
            )
        threshold = task.get("threshold", 0.0)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
        ):
            raise ValueError("threshold must be finite")
        classes = 2 if rule == "threshold" else len(outputs)
        if set(labels) != set(ids) or any(
            type(v) is not int or not 0 <= v < classes for v in labels.values()
        ):
            raise ValueError(
                "labels must cover exactly selected IDs with valid integer classes"
            )
        normalized[name] = dict(
            outputs=list(outputs), rule=rule, threshold=threshold, labels=dict(labels)
        )
    parameter = next(model.parameters())
    inputs = torch.tensor(
        [dataset.sample(s).inputs for s in ids],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    snapshot = collect_research_data(
        model, inputs, sample_ids=ids, edits=edits, derivatives=False
    )
    observations = snapshot["observations"]
    groups: list[dict[str, Any]] = [
        {
            "field": None,
            "value": None,
            "missing": False,
            "indices": list(range(len(ids))),
        }
    ]
    for field in strata:
        partitions: dict[tuple[bool, str], list[int]] = {}
        for index, sid in enumerate(ids):
            semantics = dataset.sample(sid).semantics
            missing = field not in semantics
            value = semantics.get(field)
            key = (missing, json.dumps(value, sort_keys=True, allow_nan=False))
            partitions.setdefault(key, []).append(index)
        for (missing, value), indices in partitions.items():
            groups.append(
                dict(
                    field=field,
                    value=json.loads(value),
                    missing=missing,
                    indices=indices,
                )
            )
    results: dict[str, Any] = {}
    for name, task in normalized.items():
        columns = [model.output_names.index(out) for out in task["outputs"]]

        def predict(values):
            scores = torch.tensor(values, dtype=torch.float64)[:, columns]
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(
                    "nonfinite task scores; protection metrics unavailable"
                )
            return (
                (scores[:, 0] >= task["threshold"]).long()
                if task["rule"] == "threshold"
                else scores.argmax(dim=1)
            )

        before = predict(observations["baseline"])
        labels = torch.tensor([task["labels"][sid] for sid in ids])
        results[name] = {}
        for edit, observation in observations["edits"].items():
            after = predict(observation["outputs"])
            summaries = []
            for group in groups:
                indices = group["indices"]
                summaries.append(
                    {
                        **group,
                        "sample_ids": [ids[i] for i in indices],
                        "metrics": protection_metrics(
                            before[indices], after[indices], labels[indices]
                        ),
                    }
                )
            results[name][edit] = dict(
                predictions_before=before,
                predictions_after=after,
                labels=labels,
                strata=summaries,
            )
    return _json_value(
        dict(
            schema="nmn.protection-study.v1",
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            model_snapshot=snapshot,
            sample_ids=ids,
            tasks=normalized,
            protocol=dict(
                provenance=provenance,
                split=split,
                strata=list(strata),
                threshold_tie="positive",
                argmax_tie="first declared output",
            ),
            results=results,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "Empirical classification measurements; no population or uniform guarantee.",
                "Labels and semantic strata are supplied, not discovered.",
                "Strata are reported separately, not as intersections; missing values remain explicit.",
                "No edit selection or multiple-comparison correction is performed.",
            ],
        )
    )
