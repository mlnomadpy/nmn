"""Numerical research observations from native NMN models.

These APIs expose tensors and measured finite-sample effects. They do not
produce continuous-domain certificates or infer semantic labels.
"""

import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import torch

from .baselines import baseline_geometry
from .graph import YatGraph
from .interpretable import Intervention, ThreeNeuronYat, YatExpansion


def yat_gram(x: torch.Tensor, y: Optional[torch.Tensor] = None, *, epsilon=1.0):
    """Unbiased shared-epsilon ⵟ matrix for two 2D point banks.

    Explicit differences avoid cancellation in the expanded squared distance.
    The result is a kernel Gram matrix only when the banks are the same.
    """
    y = x if y is None else y
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("point banks must be matrices with equal feature dimension")
    if not x.is_floating_point() or not y.is_floating_point():
        raise ValueError("point banks must be floating-point tensors")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    distances = (x[:, None, :] - y[None, :, :]).square().sum(-1)
    return (x @ y.T).square() / (distances + epsilon)


def expansion_geometry(module: YatExpansion, inputs: torch.Tensor):
    """Expose the center geometry and finite-expansion RKHS norm matrix.

    ``rkhs_inner_products[j,k] = a_j.T K_centers a_k`` for the fixed,
    unbiased, shared-epsilon kernel at the current parameter snapshot. This is
    local module geometry, not an RKHS norm for the composed network. Eigenvalues
    and rank are numerical diagnostics; singular banks are permitted.
    """
    if inputs.ndim != 2 or inputs.shape[0] == 0:
        raise ValueError("inputs must be a nonempty 2D point bank")
    # Promote all geometry to float64, preserving autograd through casts.
    centers = module.centers.to(dtype=torch.float64)
    points = inputs.to(device=centers.device, dtype=torch.float64)
    coefficients = module.coefficients.to(dtype=torch.float64)
    epsilon = module.kernel.epsilon
    gram = yat_gram(centers, epsilon=epsilon)
    eigenvalues = torch.linalg.eigvalsh(gram)
    tolerance = torch.finfo(gram.dtype).eps * gram.shape[0] * eigenvalues.abs().max()
    rank = (eigenvalues > tolerance).sum()
    condition = (
        eigenvalues[-1] / eigenvalues[0]
        if bool(eigenvalues[0] > tolerance)
        else eigenvalues.new_tensor(float("inf"))
    )
    return {
        "centers": centers,
        "coefficients": coefficients,
        "dot_products": points @ centers.T,
        "squared_distances": (points[:, None, :] - centers[None, :, :])
        .square()
        .sum(-1),
        "kernel_values": yat_gram(points, centers, epsilon=epsilon),
        "center_gram": gram,
        "gram_eigenvalues": eigenvalues,
        "rank_tolerance": tolerance,
        "numerical_rank": rank,
        "condition_number": condition,
        "rkhs_inner_products": coefficients @ gram @ coefficients.T,
    }


def input_jacobian(model, inputs: torch.Tensor):
    """Per-example output/input Jacobians, shape ``(N, output_dim, input_dim)``.

    Evaluates each example separately, suitable for sample-independent modules.
    For a batch-coupled model this describes singleton execution, not the
    Jacobian of the full batch. Parameter gradient buffers are not populated.
    """
    if inputs.ndim != 2 or inputs.shape[0] == 0:
        raise ValueError("inputs must be a nonempty 2D tensor")
    with torch.enable_grad():
        return torch.stack(
            [torch.autograd.functional.jacobian(model, row) for row in inputs]
        )


def gate_derivatives(
    model: Union[ThreeNeuronYat, YatGraph], inputs: torch.Tensor, *, gates=None
):
    """Jacobians and Hessians of outputs under shared scalar module gates.

    Returns Jacobian ``(N, O, M)`` and Hessian ``(N, O, M, M)`` for O outputs
    and M modules, in model.state_names order.
    These are local derivatives at the supplied gates, not finite-edit bounds.
    The Hessian includes mixed downstream effects through actual execution.
    """
    if inputs.ndim != 2 or inputs.shape[0] == 0:
        raise ValueError("inputs must be a nonempty 2D tensor")
    g = (
        inputs.new_ones(len(model.state_names))
        if gates is None
        else torch.as_tensor(gates, dtype=inputs.dtype, device=inputs.device)
    )
    if g.shape != (len(model.state_names),):
        raise ValueError("provide one gate per model.state_names entry")
    jacobians, hessians = [], []
    with torch.enable_grad():
        for row in inputs:

            def execute(values):
                controls = {
                    name: Intervention(gate=values[i])
                    for i, name in enumerate(model.state_names)
                }
                return model(row, controls)

            jacobians.append(torch.autograd.functional.jacobian(execute, g))
            hessians.append(
                torch.stack(
                    [
                        torch.autograd.functional.hessian(lambda v: execute(v)[j], g)
                        for j in range(len(model.output_names))
                    ]
                )
            )
    return {
        "gates": g,
        "jacobian": torch.stack(jacobians),
        "hessian": torch.stack(hessians),
    }


def intervention_table(
    model: Union[ThreeNeuronYat, YatGraph],
    inputs: torch.Tensor,
    edits: Mapping[str, Mapping[str, Intervention]],
):
    """Replay named, possibly joint edits and retain every example's effects.

    Returns baseline, edited outputs, signed/absolute deltas, and raw/effective
    internal traces. Does not aggregate away collateral changes or assign pass
    thresholds. Edits run independently from the same model parameters.
    """
    with torch.no_grad():
        baseline, baseline_trace = model.forward_with_trace(inputs)
        rows = {}
        for name, controls in edits.items():
            output, trace = model.forward_with_trace(inputs, controls)
            rows[name] = {
                "outputs": output,
                "delta": output - baseline,
                "absolute_delta": (output - baseline).abs(),
                "trace": trace,
            }
    return {"baseline": baseline, "baseline_trace": baseline_trace, "edits": rows}


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0 else "-infinity" if value < 0 else "nan"
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def collect_research_data(
    model: Union[ThreeNeuronYat, YatGraph],
    inputs: torch.Tensor,
    *,
    sample_ids: Sequence[str],
    edits: Optional[Mapping[str, Mapping[str, Intervention]]] = None,
    metadata: Optional[dict] = None,
    derivatives: bool = True,
):
    """Collect a JSON-ready native-model research snapshot.

    Supports ThreeNeuronYat and explicit-state YatGraph networks.
    Caller supplies sample IDs and task/split/semantic metadata. The snapshot
    records all parameters, architecture configuration, input values and IDs,
    edit controls, local geometry, responses and optional derivatives. Collection
    does not train, modify parameters or fill their gradient buffers. Designed
    for small research banks: full Gram matrices/Hessians can be expensive.
    """
    if not isinstance(model, (ThreeNeuronYat, YatGraph)):
        raise TypeError("collector supports ThreeNeuronYat and YatGraph")
    if (
        inputs.ndim != 2
        or inputs.shape[0] == 0
        or inputs.shape[1]
        != (len(model.input_names) if isinstance(model, YatGraph) else 2)
    ):
        raise ValueError("inputs must be nonempty and match the model input dimension")
    if len(sample_ids) != len(inputs) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("provide one unique sample ID per input")
    if any(not isinstance(s, str) or not s for s in sample_ids):
        raise ValueError("sample IDs must be nonempty strings")
    edits = {} if edits is None else edits
    started = time.perf_counter()
    with torch.no_grad():
        observations = intervention_table(model, inputs, edits)
        geometry = {}
        for name in model.state_names:
            block = (
                model.blocks[name]
                if isinstance(model, YatGraph)
                else getattr(model, name)
            )
            points = observations["baseline_trace"][f"{name}.input"]
            geometry[name] = (
                expansion_geometry(block, points)
                if isinstance(block, YatExpansion)
                else baseline_geometry(block, points)
            )
    configuration = (
        model.configuration()
        if isinstance(model, YatGraph)
        else {
            "class": "nmn.torch.ThreeNeuronYat",
            "distance_mode": "direct",
            "num_centers": model.h.num_centers,
            "epsilon": {
                name: getattr(model, name).kernel.epsilon for name in model.state_names
            },
            "state_names": model.state_names,
            "output_names": model.output_names,
            "routing": {"h": ["u"], "p": ["v"], "y": ["h", "v"]},
        }
    )
    payload = {
        "schema": "nmn.native-research.v1",
        "assurance": "floating-point observations",
        "configuration": configuration,
        "parameters": dict(model.named_parameters()),
        "trainability": {name: p.requires_grad for name, p in model.named_parameters()},
        "source_sha256": {
            str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("interpretable.py"),
                Path(__file__).with_name("graph.py"),
                Path(__file__).with_name("baselines.py"),
                Path(__file__).parent / "nmn" / "yat_nmn.py",
            )
        },
        "inputs": inputs,
        "sample_ids": list(sample_ids),
        "metadata": {} if metadata is None else metadata,
        "controls": {
            name: {
                state: {"gate": c.gate, "replacement": c.replacement}
                for state, c in controls.items()
            }
            for name, controls in edits.items()
        },
        "observations": observations,
        "geometry": geometry,
        "runtime": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(inputs.device),
            "dtype": str(inputs.dtype),
        },
    }
    if derivatives:
        payload["input_jacobian"] = input_jacobian(model, inputs)
        payload["gate_derivatives"] = gate_derivatives(model, inputs)
    payload["collection_seconds"] = time.perf_counter() - started
    result = _json_value(payload)
    identity = {key: result[key] for key in ("configuration", "parameters")}
    result["model_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    result["limitations"] = [
        "Finite observations and local derivatives are not uniform certificates.",
        "RKHS norms describe each local expansion, not the composed response.",
        "Semantics, split discipline and population definitions are caller supplied.",
        "Collection time includes diagnostics; it is not a comparative replay benchmark.",
    ]
    return result


def save_research_data(data: dict, path):
    """Write a snapshot as strict JSON; refuse to overwrite existing evidence."""
    from ..research.io import write_json_exclusive

    write_json_exclusive(data, path, sort_keys=True)


def coalition_effects(model: ThreeNeuronYat, inputs: torch.Tensor):
    """Execute every subset of h,p,y deletions and compute subset coefficients.

    Bit i disables state_names[i]; mask zero is the unchanged network. Returns
    output values and their subset Möbius transform, both shaped ``(8, N, 2)``.
    Coefficients reconstruct each queried coalition by summing over its subsets.
    No unqueried-background or interaction-sparsity claim is made.
    """
    if not isinstance(model, ThreeNeuronYat):
        raise TypeError("use coalition_study for general graph coalitions")
    with torch.no_grad():
        values = torch.stack(
            [
                model(
                    inputs,
                    {
                        name: Intervention(gate=0.0)
                        for bit, name in enumerate(model.state_names)
                        if mask & (1 << bit)
                    },
                )
                for mask in range(8)
            ]
        )
        coefficients = values.clone()
        for bit in range(3):
            for mask in range(8):
                if mask & (1 << bit):
                    coefficients[mask] -= coefficients[mask ^ (1 << bit)]
    return {
        "state_names": model.state_names,
        "mask_semantics": "bit=1 disables state",
        "values": values,
        "subset_coefficients": coefficients,
    }


def protection_metrics(before: torch.Tensor, after: torch.Tensor, labels: torch.Tensor):
    """Keep accuracy, damage to correct answers, and disagreement separate.

    Arguments are one-dimensional class-label tensors for the same population.
    Returns per-example flags and separate rates. Conditional damage is None
    when no example was originally correct; no eligibility is silently dropped.
    Apply separately to caller-declared strata; labels/strata are not inferred.
    """
    if (
        before.ndim != 1
        or before.numel() == 0
        or (before.shape != after.shape or before.shape != labels.shape)
    ):
        raise ValueError("provide matching nonempty 1D prediction and label tensors")
    correct_before, correct_after = before == labels, after == labels
    broken = correct_before & ~correct_after
    fixed = ~correct_before & correct_after
    eligible = correct_before.sum()
    return {
        "count": before.numel(),
        "originally_correct_count": eligible,
        "correct_before": correct_before,
        "correct_after": correct_after,
        "broken": broken,
        "fixed": fixed,
        "disagreement": before != after,
        "accuracy_before": correct_before.double().mean(),
        "accuracy_after": correct_after.double().mean(),
        "conditional_damage_rate": (
            broken.sum().double() / eligible if bool(eligible) else None
        ),
        "disagreement_rate": (before != after).double().mean(),
    }


def model_from_snapshot(snapshot: dict, *, device=None, dtype=torch.float64):
    """Restore a native research model from JSON configuration and parameters.

    No pickle or executable model code is loaded. The content hash and parameter
    names/shapes are checked. Historical expanded-distance research snapshots are
    rejected rather than silently reinterpreted as direct-distance models.
    """
    if snapshot.get("schema") not in ("nmn.native-research.v1", "nmn.native-model.v1"):
        raise ValueError("unsupported native snapshot schema")
    config, params = snapshot["configuration"], snapshot["parameters"]
    identity = {"configuration": config, "parameters": params}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if snapshot.get("model_sha256") != digest:
        raise ValueError("model content hash mismatch")
    if config.get("distance_mode") != "direct":
        raise ValueError("snapshot does not declare direct-distance execution")
    if config.get("class") == "nmn.torch.YatGraph":
        model = YatGraph.from_configuration(config, device=device, dtype=dtype)
    elif config.get("class") == "nmn.torch.ThreeNeuronYat":
        model = ThreeNeuronYat(config["num_centers"], device=device, dtype=dtype)
        for name in model.state_names:
            epsilon = config["epsilon"][name]
            if not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("snapshot epsilon must be finite and positive")
            getattr(model, name).kernel.epsilon = epsilon
        if config["state_names"] != list(model.state_names) or config[
            "output_names"
        ] != list(model.output_names):
            raise ValueError("snapshot names do not match the reference architecture")
        if config["routing"] != {"h": ["u"], "p": ["v"], "y": ["h", "v"]}:
            raise ValueError(
                "snapshot routing does not match the reference architecture"
            )
    else:
        raise ValueError("unsupported native model class")
    targets = dict(model.named_parameters())
    if set(params) != set(targets):
        raise ValueError("snapshot parameter names do not match model")
    trainability = snapshot.get("trainability", {name: True for name in targets})
    if set(trainability) != set(targets) or any(
        not isinstance(v, bool) for v in trainability.values()
    ):
        raise ValueError("invalid parameter trainability declaration")
    for name, parameter in targets.items():
        parameter.requires_grad_(trainability[name])
    tensors = {
        name: torch.as_tensor(value, device=device, dtype=dtype)
        for name, value in params.items()
    }
    for name, value in tensors.items():
        if value.shape != targets[name].shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid shape or nonfinite parameter: {name}")
    with torch.no_grad():
        for name, value in tensors.items():
            targets[name].copy_(value)
    return model
