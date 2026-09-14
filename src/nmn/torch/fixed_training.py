"""Validation of a shared attenuation objective for native model training."""

import json
import math

import torch

from .interpretable import Intervention


def prepare_fixed_objective(objective, model, sample_ids):
    objective = json.loads(json.dumps(objective, allow_nan=False))
    if (
        not isinstance(objective, dict)
        or set(objective)
        != {
            "schema",
            "controls",
            "output_names",
            "protected_outputs",
            "targets",
            "provenance",
        }
        or objective["schema"] != "nmn.fixed-edit-objective.v1"
    ):
        raise ValueError(
            "fixed_edit requires the documented fixed-edit-objective fields"
        )
    if (
        not isinstance(objective["provenance"], str)
        or not objective["provenance"].strip()
    ):
        raise ValueError("fixed edit target provenance is required")
    controls = objective["controls"]
    if (
        not isinstance(controls, dict)
        or not controls
        or not set(controls) <= set(model.state_names)
    ):
        raise ValueError("fixed controls must name native modules")
    for name, value in controls.items():
        if (
            not isinstance(value, dict)
            or set(value) != {"gate"}
            or isinstance(value["gate"], bool)
            or not isinstance(value["gate"], (int, float))
            or not math.isfinite(value["gate"])
            or not 0 <= value["gate"] <= 1
        ):
            raise ValueError(
                "fixed controls support only shared scalar attenuation gates in [0,1]"
            )
    indices = {}
    for field in ("output_names", "protected_outputs"):
        names = objective[field]
        if (
            not isinstance(names, list)
            or (field == "output_names" and not names)
            or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
            or not set(names) <= set(model.output_names)
        ):
            raise ValueError(
                "fixed objective output lists must contain unique declared outputs"
            )
        indices[field] = [model.output_names.index(n) for n in names]
    if not isinstance(objective["targets"], dict) or set(objective["targets"]) != set(
        sample_ids
    ):
        raise ValueError(
            "fixed targets must cover exactly training and checkpoint-selection samples"
        )
    targets = torch.tensor(
        [objective["targets"][s] for s in sample_ids], dtype=torch.float64
    )
    if targets.shape != (len(sample_ids), len(indices["output_names"])) or not bool(
        torch.isfinite(targets).all()
    ):
        raise ValueError("fixed targets must be finite vectors matching output_names")
    return (
        objective,
        {n: Intervention(**v) for n, v in controls.items()},
        indices,
        targets,
    )
