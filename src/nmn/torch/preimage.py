"""Bounded native preimage search for a fixed finite kernel feature bank."""

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .interpretable import YatExpansion
from .research import _json_value


def search_preimages(
    module,
    inputs,
    target_features,
    *,
    lower,
    upper,
    sample_ids,
    provenance,
    max_steps,
    max_seconds,
    learning_rate,
):
    """Optimize input coordinates, holding the native kernel bank fixed.

    Targets are unweighted finite-bank kernel evaluations, not arbitrary RKHS
    coordinates or output coefficients. Minimize their mean squared residual
    with projected Adam. The best finite iterate (including initialization) is
    selected by aggregate loss, with earlier ties retained. A residual is a
    numerical observation, never proof of existence or impossibility.
    """
    if type(module) is not YatExpansion:
        raise TypeError("preimage search requires a strict native YatExpansion")
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    for name, value in [("max_seconds", max_seconds), ("learning_rate", learning_rate)]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(name + " must be finite and positive")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("target feature provenance is required")
    parameter = module.centers
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ValueError("preimage search requires float32 or float64 parameters")
    points = (
        torch.as_tensor(inputs, dtype=parameter.dtype, device=parameter.device)
        .detach()
        .clone()
    )
    targets = (
        torch.as_tensor(target_features, dtype=parameter.dtype, device=parameter.device)
        .detach()
        .clone()
    )
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != module.in_features:
        raise ValueError("inputs must be a nonempty matrix of module input width")
    if targets.shape != (points.shape[0], module.num_centers):
        raise ValueError("targets must contain one feature per center and sample")
    if (
        not isinstance(sample_ids, (list, tuple))
        or len(sample_ids) != len(points)
        or any(not isinstance(s, str) or not s for s in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("supply a unique nonempty ID for every sample")
    bounds = []
    for value in (lower, upper):
        bound = torch.as_tensor(value, dtype=points.dtype, device=points.device)
        try:
            bounds.append(torch.broadcast_to(bound, points.shape).detach().clone())
        except RuntimeError as exc:
            raise ValueError("bounds must broadcast to the input matrix") from exc
    lo, hi = bounds
    if (
        any(not bool(torch.isfinite(v).all()) for v in (points, targets, lo, hi))
        or bool((lo > hi).any())
        or bool(((points < lo) | (points > hi)).any())
    ):
        raise ValueError("finite ordered bounds must contain the finite initial inputs")
    snapshot = _json_value(
        {
            "centers": parameter,
            "coefficients": module.coefficients,
            "epsilon": module.kernel.epsilon,
            "dtype": str(parameter.dtype),
            "distance_mode": "direct",
        }
    )
    identity = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    variable = points.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([variable], lr=learning_rate)
    history = []
    best, best_step, best_loss = points.clone(), 0, math.inf
    status = "step-budget-completed"
    started = time.perf_counter()
    # Enable input differentiation even if a caller is collecting under no_grad.
    with torch.enable_grad():
        for step in range(max_steps + 1):
            if step and time.perf_counter() - started >= max_seconds:
                status = "time-budget-stopped"
                break
            features = module._features(variable)
            losses = (features - targets).square().mean(dim=1)
            loss = losses.mean()
            if not bool(torch.isfinite(loss)):
                status = "nonfinite-stopped"
                break
            value = float(loss.detach())
            history.append(
                {
                    "step": step,
                    "mean_squared_error": value,
                    "per_sample_squared_error": losses.detach().cpu().tolist(),
                }
            )
            if value < best_loss:
                best_loss, best_step, best = value, step, variable.detach().clone()
            if step == max_steps:
                break
            (gradient,) = torch.autograd.grad(loss, variable)
            if not bool(torch.isfinite(gradient).all()):
                status = "nonfinite-stopped"
                break
            variable.grad = gradient
            optimizer.step()
            with torch.no_grad():
                variable.copy_(torch.minimum(torch.maximum(variable, lo), hi))
    with torch.no_grad():
        initial_features = module._features(points)
        selected_features = module._features(best)
    record = _json_value(
        {
            "schema": "nmn.preimage-search.v1",
            "status": status,
            "module_snapshot": snapshot,
            "module_sha256": identity,
            "sample_ids": list(sample_ids),
            "provenance": provenance,
            "protocol": {
                "objective": "mean squared finite-bank feature residual",
                "optimizer": "projected Adam",
                "max_steps": max_steps,
                "max_seconds": max_seconds,
                "learning_rate": learning_rate,
                "selection": "lowest aggregate loss; earliest tie",
            },
            "inputs": points,
            "target_features": targets,
            "lower": lo,
            "upper": hi,
            "initial_features": initial_features,
            "selected_step": best_step,
            "selected_inputs": best,
            "selected_features": selected_features,
            "feature_residuals": selected_features - targets,
            "per_sample_squared_error": (selected_features - targets)
            .square()
            .mean(dim=1),
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": [
                "Inputs are optimized per example; no generalizing mapper is learned.",
                "Finite-bank residual is not a whole-RKHS norm or erasure guarantee.",
                "A nonzero residual does not prove that no preimage exists.",
                "Downstream utility/protection must be measured by executing the frozen suffix.",
                "Time limit is checked between iterations; initialization and final reporting are excluded.",
            ],
        }
    )
    json.dumps(record, allow_nan=False)
    return record
