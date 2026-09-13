"""Budgeted native coalition replay with explicit background gates."""

import hashlib
import math
import time
from pathlib import Path

import torch

from .interpretable import Intervention
from .research import _json_value, collect_research_data


def coalition_study(
    model, dataset, *, modules, max_evaluations, background=None, split=None
):
    """Replay a declared subset lattice, disabling selected modules by mask.

    Mask zero uses the supplied background gates (one elsewhere). Bit i disables
    modules[i] regardless of its background gate. Other modules retain their
    background gates. Replay proceeds in integer-mask order within the explicit
    budget; partial records retain observations but no incomplete coefficients.
    Complete records include per-example subset coefficients and reconstruction
    residuals. This is numerical finite-difference analysis, not a certificate.
    """
    modules = tuple(modules)
    if (
        not modules
        or len(set(modules)) != len(modules)
        or not set(modules) <= set(model.state_names)
    ):
        raise ValueError("modules must be nonempty, unique model module names")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer")
    background = {} if background is None else dict(background)
    if not set(background) <= set(model.state_names) or any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
        for v in background.values()
    ):
        raise ValueError("background must map model modules to finite scalar gates")
    ids = dataset.sample_ids(split=split)
    if not ids:
        raise ValueError("selected split has no samples")
    parameter = next(model.parameters())
    inputs = torch.tensor(
        [dataset.sample(s).inputs for s in ids],
        device=parameter.device,
        dtype=parameter.dtype,
    )
    total = 1 << len(modules)
    count = min(total, max_evaluations)
    started = time.perf_counter()
    snapshot = collect_research_data(model, inputs, sample_ids=ids, derivatives=False)
    preparation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    gates = {name: background.get(name, 1.0) for name in model.state_names}
    values = []
    with torch.no_grad():
        for mask in range(count):
            controls = {name: Intervention(gate=gate) for name, gate in gates.items()}
            for bit, name in enumerate(modules):
                if mask & (1 << bit):
                    controls[name] = Intervention(gate=0.0)
            values.append(model(inputs, controls))
        response = torch.stack(values)
        delta = response - response[0]
        finite = bool(torch.isfinite(response).all() and torch.isfinite(delta).all())
        complete = count == total and finite
        coefficients = None
        reconstruction_error = None
        if complete:
            coefficients = response.clone()
            for bit in range(len(modules)):
                for mask in range(count):
                    if mask & (1 << bit):
                        coefficients[mask] -= coefficients[mask ^ (1 << bit)]
            reconstruction = coefficients.clone()
            for bit in range(len(modules)):
                for mask in range(count):
                    if mask & (1 << bit):
                        reconstruction[mask] += reconstruction[mask ^ (1 << bit)]
            reconstruction_error = reconstruction - response
            if not bool(
                torch.isfinite(coefficients).all()
                and torch.isfinite(reconstruction_error).all()
            ):
                complete = False
                finite = False
                coefficients = reconstruction_error = None
    elapsed = time.perf_counter() - started
    return _json_value(
        {
            "schema": "nmn.coalition-study.v1",
            "status": (
                "failed" if not finite else "observed" if complete else "inconclusive"
            ),
            "reason": (
                "nonfinite arithmetic"
                if not finite
                else (
                    "complete finite lattice"
                    if complete
                    else "evaluation budget exhausted"
                )
            ),
            "dataset": dataset.to_dict(),
            "dataset_sha256": dataset.sha256,
            "sample_ids": ids,
            "model_snapshot": snapshot,
            "protocol": {
                "modules": modules,
                "background_gates": gates,
                "split": split,
                "mask_semantics": "bit i disables modules[i]; other gates retain background",
                "mask_order": "ascending integer",
                "max_evaluations": max_evaluations,
            },
            "coverage": {"evaluated": count, "total": total, "complete": complete},
            "masks": list(range(count)),
            "outputs": response,
            "delta_from_background": delta,
            "subset_coefficients": coefficients,
            "reconstruction_error": reconstruction_error,
            "cost": {
                "preparation_seconds": preparation_seconds,
                "host_replay_and_transform_seconds": elapsed,
                "coalition_forward_calls": count,
                "coalition_sample_evaluations": count * len(ids),
                "preparation_forward_calls": 1,
            },
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": [
                "Observed finite lattice at declared background gates; no unqueried-background guarantee.",
                "Coefficients are numerical subset differences, not evidence of global interaction sparsity.",
                "A partial lattice has no full coefficient table; unqueried coalitions are not zero.",
                "Preparation stores the unchanged model trace; coalition outputs are stored without every internal trace.",
            ],
        }
    )
