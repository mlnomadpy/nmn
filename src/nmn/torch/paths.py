"""Actual joint gate paths and numerical finite-effect comparisons."""

import hashlib
import time
from pathlib import Path

import torch

from .interpretable import Intervention


def gate_path(model, inputs: torch.Tensor, start, end, *, steps: int = 32):
    """Measure a straight joint gate path and compare effect approximations.

    For g(t)=start+t*(end-start), retain actual outputs, gate gradients and
    directional curvature of every output. Trapezoidal integration estimates
    integral grad_g(output) * direction dt per module and the weighted curvature
    integral in the second-order remainder identity. Curvature is differentiated
    directly along scalar t; no module-by-module Hessian is materialized.

    Returns first-order and endpoint second-order predictions, path-integrated
    predictions, and their signed residuals against actual endpoint differences.
    Integration residuals are numerical diagnostics, not certified error bounds.
    This is an actual gate path, not the input interpolation path of EAP-IG.
    Parameter gradient buffers are not populated and parameters are not changed.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer >= 2")
    if inputs.ndim != 2 or not len(inputs) or not inputs.is_floating_point():
        raise ValueError("inputs must be a nonempty floating-point matrix")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must be finite")
    shape = (len(model.state_names),)
    start = torch.as_tensor(start, device=inputs.device, dtype=inputs.dtype)
    end = torch.as_tensor(end, device=inputs.device, dtype=inputs.dtype)
    if (
        start.shape != shape
        or end.shape != shape
        or not bool(torch.isfinite(start).all() & torch.isfinite(end).all())
    ):
        raise ValueError("start/end must contain one finite gate per module")
    direction = end - start
    times = torch.linspace(0, 1, steps + 1, device=inputs.device, dtype=inputs.dtype)
    values, gradients, curvatures = [], [], []
    calls = 0
    began = time.perf_counter()
    with torch.enable_grad():
        for row in inputs:

            def execute(gates):
                nonlocal calls
                calls += 1
                return model(
                    row,
                    {
                        name: Intervention(gate=gates[i])
                        for i, name in enumerate(model.state_names)
                    },
                )

            sample_values, sample_gradients, sample_curvature = [], [], []
            for t in times:
                gates = start + t * direction
                sample_values.append(execute(gates).detach())
                sample_gradients.append(
                    torch.autograd.functional.jacobian(execute, gates)
                )
                sample_curvature.append(
                    torch.stack(
                        [
                            torch.autograd.functional.hessian(
                                lambda position: execute(start + position * direction)[
                                    j
                                ],
                                t,
                            )
                            for j in range(len(model.output_names))
                        ]
                    )
                )
            values.append(torch.stack(sample_values))
            gradients.append(torch.stack(sample_gradients))
            curvatures.append(torch.stack(sample_curvature))
    outputs = torch.stack(values)  # N, time, outputs
    gradient = torch.stack(gradients)  # N, time, outputs, modules
    curvature = torch.stack(curvatures)  # N, time, outputs
    contributions = gradient * direction
    # Uniform-grid trapezoids; no dependency on newer torch integration APIs.
    integrated = (contributions[:, :-1] + contributions[:, 1:]).sum(1) / (2 * steps)
    weighted = curvature * (1 - times)[None, :, None]
    remainder = (weighted[:, :-1] + weighted[:, 1:]).sum(1) / (2 * steps)
    actual = outputs[:, -1] - outputs[:, 0]
    first = contributions[:, 0].sum(-1)
    second = first + curvature[:, 0] / 2
    predictions = {
        "first_order": first,
        "second_order": second,
        "integrated_gradient": integrated.sum(-1),
        "integrated_curvature": first + remainder,
    }
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "module_names": model.state_names,
        "output_names": model.output_names,
        "start": start.detach(),
        "end": end.detach(),
        "times": times,
        "outputs": outputs,
        "gate_gradients": gradient,
        "directional_curvature": curvature,
        "integrated_module_contributions": integrated,
        "actual_delta": actual,
        "predictions": predictions,
        "residuals": {name: value - actual for name, value in predictions.items()},
        "cost": {
            "model_forward_calls": calls,
            "seconds": time.perf_counter() - began,
            "intervals": steps,
            "samples": len(inputs),
        },
        "assurance": "floating-point path samples and trapezoidal quadrature",
        "limitations": [
            "Quadrature residuals are measured, not uniform certificates.",
            "Local second-order correction need not improve a finite edit prediction.",
            "Per-module integrated contributions are not isolated edit effects.",
            "Timing includes autograd and instrumentation; no speedup is established.",
        ],
    }
