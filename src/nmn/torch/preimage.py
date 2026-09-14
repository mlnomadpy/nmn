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


def preimage_study(
    model,
    dataset,
    *,
    module_name,
    targets,
    lower,
    upper,
    provenance,
    max_steps,
    max_seconds,
    learning_rate,
    split=None,
):
    """Search from actual module inputs on an explicitly selected population.

    Targets map exactly the selected sample IDs to finite-bank feature vectors.
    Selected input coordinates are proposals, not automatically installed edits.
    A parent model/dataset snapshot links the frozen bank to its executed inputs.
    """
    from .graph import YatGraph
    from .interpretable import ThreeNeuronYat
    from .research import collect_research_data

    if not isinstance(model, (YatGraph, ThreeNeuronYat)):
        raise TypeError("preimage studies require a native graph or three-neuron model")
    if not isinstance(module_name, str) or module_name not in model.state_names:
        raise ValueError("unknown preimage module")
    block = (
        model.blocks[module_name]
        if isinstance(model, YatGraph)
        else getattr(model, module_name)
    )
    if type(block) is not YatExpansion:
        raise ValueError(
            "preimage studies support fixed unbiased YatExpansion modules only"
        )
    ids = dataset.sample_ids(split=split)
    if not ids or not isinstance(targets, dict) or set(targets) != set(ids):
        raise ValueError("targets must cover exactly the nonempty selected population")
    json.dumps(targets, allow_nan=False)
    x = torch.tensor(
        [dataset.sample(sid).inputs for sid in ids],
        dtype=block.centers.dtype,
        device=block.centers.device,
    )
    with torch.no_grad():
        _, trace = model.forward_with_trace(x)
        points = trace[module_name + ".input"]
    search = search_preimages(
        block,
        points,
        [targets[sid] for sid in ids],
        lower=lower,
        upper=upper,
        sample_ids=ids,
        provenance=provenance,
        max_steps=max_steps,
        max_seconds=max_seconds,
        learning_rate=learning_rate,
    )
    snapshot = collect_research_data(model, x, sample_ids=ids, derivatives=False)
    return {
        "schema": "nmn.preimage-study.v1",
        "status": search["status"],
        "model_snapshot": snapshot,
        "module": module_name,
        "dataset": dataset.to_dict(),
        "dataset_sha256": dataset.sha256,
        "sample_ids": list(ids),
        "split": split,
        "targets": targets,
        "search": search,
        "proposed_inputs": dict(zip(ids, search["selected_inputs"])),
        "limitations": [
            "Selected module inputs are proposals; no native edit is installed by this study.",
            "No held-out generalization is evaluated: every selected sample is optimized.",
            "Feature residual does not certify erasure, protection, or preimage impossibility.",
        ],
    }


def execute_preimage_study(record):
    """Apply proposals as complete reads of one declared graph receiver.

    Shared state and other readers keep their original values. Recompute the
    chosen receiver and downstream graph; this is a read intervention, not a
    claim that the proposed state is globally reachable or that erasure occurred.
    """
    from ..research.datasets import ResearchDataset
    from ..research.native_export import _check_identities
    from .graph import YatGraph
    from .research import collect_research_data, model_from_snapshot

    if record.get("schema") != "nmn.preimage-study.v1":
        raise ValueError("expected a dataset-linked preimage study")
    serialized = json.dumps(record, sort_keys=True, allow_nan=False)
    _check_identities(record)
    dataset = ResearchDataset.from_dict(record["dataset"])
    if dataset.sha256 != record["dataset_sha256"]:
        raise ValueError("preimage dataset identity mismatch")
    ids = list(dataset.sample_ids(split=record["split"]))
    search = record["search"]
    if not ids or ids != record["sample_ids"] or ids != search["sample_ids"]:
        raise ValueError("preimage sample order mismatch")
    if record["proposed_inputs"] != dict(zip(ids, search["selected_inputs"])):
        raise ValueError("proposals differ from the recorded selected inputs")
    if (
        set(record["targets"]) != set(ids)
        or [record["targets"][sid] for sid in ids] != search["target_features"]
    ):
        raise ValueError("preimage target mapping mismatch")
    snapshot = record["model_snapshot"]
    dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
    dtype = dtypes.get(snapshot["runtime"]["dtype"])
    if dtype is None:
        raise ValueError("execution supports saved float32/float64 models")
    model = model_from_snapshot(snapshot, device="cpu", dtype=dtype)
    if not isinstance(model, YatGraph):
        raise ValueError("automatic read execution requires an explicit YatGraph")
    name = record["module"]
    if name not in model.state_names or type(model.blocks[name]) is not YatExpansion:
        raise ValueError("proposal receiver must be a strict YatExpansion")
    block = model.blocks[name]
    current_bank = _json_value(
        {
            "centers": block.centers,
            "coefficients": block.coefficients,
            "epsilon": block.kernel.epsilon,
            "dtype": str(dtype),
            "distance_mode": "direct",
        }
    )
    if current_bank != search["module_snapshot"]:
        raise ValueError("search bank differs from parent model receiver")
    spec = next(
        spec for layer in model.layer_specs for spec in layer if spec.name == name
    )
    x = torch.tensor([dataset.sample(sid).inputs for sid in ids], dtype=dtype)
    if snapshot["sample_ids"] != ids or snapshot["inputs"] != x.tolist():
        raise ValueError("parent snapshot population differs from the proposal")
    selected = torch.tensor(search["selected_inputs"], dtype=dtype)
    expected_shape = (len(ids), len(spec.reads))
    if selected.shape != expected_shape or not bool(torch.isfinite(selected).all()):
        raise ValueError("proposals must have one finite coordinate per receiver read")
    lo, hi = torch.tensor(search["lower"], dtype=dtype), torch.tensor(
        search["upper"], dtype=dtype
    )
    if (
        not bool(torch.isfinite(lo).all())
        or not bool(torch.isfinite(hi).all())
        or lo.shape != selected.shape
        or hi.shape != selected.shape
        or bool(((selected < lo) | (selected > hi)).any())
    ):
        raise ValueError("proposals must satisfy recorded bounds")
    with torch.no_grad():
        baseline, baseline_trace = model.forward_with_trace(x)
        if baseline_trace[name + ".input"].tolist() != search["inputs"]:
            raise ValueError(
                "search initialization differs from baseline receiver inputs"
            )
        patches = {name: {slot: selected[:, i] for i, slot in enumerate(spec.reads)}}
        output, trace = model.forward_with_trace(x, read_patches=patches)
        features = block._features(trace[name + ".input"])
        expected = torch.tensor(search["target_features"], dtype=dtype)
        if expected.shape != features.shape:
            raise ValueError("target feature shape differs from receiver features")
        saved_features = torch.tensor(search["selected_features"], dtype=dtype)
        if saved_features.shape != features.shape or not torch.allclose(
            features, saved_features, atol=1e-10, rtol=1e-8
        ):
            raise ValueError("executed features disagree with the recorded proposal")
    result = _json_value(
        {
            "schema": "nmn.preimage-execution.v1",
            "proposal_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "proposal": record,
            "model_snapshot": collect_research_data(
                model, x, sample_ids=ids, derivatives=False
            ),
            "dataset": dataset.to_dict(),
            "dataset_sha256": dataset.sha256,
            "sample_ids": ids,
            "module": name,
            "read_slots": list(spec.reads),
            "read_patches": patches,
            "baseline_outputs": baseline,
            "baseline_trace": baseline_trace,
            "outputs": output,
            "output_delta": output - baseline,
            "trace": trace,
            "executed_features": features,
            "feature_residuals": features - expected,
            "protocol": "replace complete reads of the selected receiver; shared state unchanged",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": [
                "Read interventions do not imply a globally reachable state.",
                "Per-example proposals do not define a learned input-to-input mapper.",
                "No target success or protected-output certificate is inferred.",
            ],
        }
    )
    json.dumps(result, allow_nan=False)
    return result
