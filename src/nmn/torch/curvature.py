"""Directional gate Hessian products and executed finite-edit residuals."""

import hashlib
import time
from pathlib import Path

import torch

from .interpretable import Intervention
from .research import _json_value, collect_research_data


def curvature_study(
    model,
    dataset,
    *,
    directions,
    provenance,
    background=None,
    split=None,
    max_directions=16,
):
    """Reuse one gate-gradient graph per sample for supplied Hessian products.

    No dense gate Hessian is allocated. Every direction is also executed as a
    finite gate displacement. Products are derivatives at the background, not
    integrated interactions or certified finite-edit errors. Gates are unrestricted
    finite real scalars; attenuation constraints must be declared separately.
    """
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("direction provenance is required")
    if type(max_directions) is not int or max_directions < 1:
        raise ValueError("max_directions must be a positive integer")
    if (
        not isinstance(directions, dict)
        or not directions
        or len(directions) > max_directions
        or any(not isinstance(k, str) or not k for k in directions)
    ):
        raise ValueError("supply named directions within max_directions")
    parameter = next(model.parameters())
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ValueError("curvature requires float32 or float64 parameters")
    names = list(model.state_names)

    def tensor(v):
        return torch.as_tensor(v, device=parameter.device, dtype=parameter.dtype)

    base = tensor([1.0] * len(names) if background is None else background)
    vectors = tensor(list(directions.values()))
    if (
        base.shape != (len(names),)
        or vectors.shape != (len(directions), len(names))
        or not bool(torch.isfinite(base).all() & torch.isfinite(vectors).all())
    ):
        raise ValueError("background and directions need one finite value per module")
    if not bool(torch.isfinite(base + vectors).all()):
        raise ValueError("finite gate endpoints are required")
    ids = dataset.sample_ids(split=split)
    if not ids:
        raise ValueError("selected population is empty")
    inputs = tensor([dataset.sample(s).inputs for s in ids])
    snapshot = collect_research_data(model, inputs, sample_ids=ids, derivatives=False)
    outputs, gradients, products, endpoints = [], [], [], []
    forward_calls, gradient_calls = 0, 0
    began = time.perf_counter()
    for row in inputs:

        def execute(g):
            return model(
                row, {name: Intervention(gate=g[i]) for i, name in enumerate(names)}
            )

        with torch.enable_grad():
            gates = base.detach().clone().requires_grad_(True)
            output = execute(gates)
            forward_calls += 1
            sample_gradients, sample_products = [], []
            for value in output:
                gradient = None
                if value.requires_grad:
                    gradient = torch.autograd.grad(
                        value,
                        gates,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    gradient_calls += 1
                if gradient is None:
                    gradient = torch.zeros_like(gates)
                hvps = []
                for vector in vectors:
                    product = None
                    if gradient.requires_grad:
                        product = torch.autograd.grad(
                            (gradient * vector).sum(),
                            gates,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        gradient_calls += 1
                    hvps.append(
                        torch.zeros_like(gates) if product is None else product.detach()
                    )
                sample_gradients.append(gradient.detach())
                sample_products.append(torch.stack(hvps))
            outputs.append(output.detach())
            gradients.append(torch.stack(sample_gradients))
            products.append(torch.stack(sample_products))
        with torch.no_grad():
            endpoints.append(torch.stack([execute(base + v) for v in vectors]))
            forward_calls += len(vectors)
    baseline = torch.stack(outputs)  # N,O
    gradient = torch.stack(gradients)  # N,O,M
    hvp = torch.stack(products)  # N,O,K,M
    edited = torch.stack(endpoints)  # N,K,O
    first = torch.einsum("nom,km->nko", gradient, vectors)
    directional = torch.einsum("nokm,km->nko", hvp, vectors)
    mixed = torch.einsum("im,nokm->noik", vectors, hvp)
    delta = edited - baseline[:, None, :]
    observations = dict(
        baseline=baseline,
        gate_gradients=gradient,
        hessian_vector_products=hvp,
        directional_curvature=directional,
        mixed_direction_curvature=mixed,
        edited_outputs=edited,
        actual_delta=delta,
        first_order_prediction=first,
        second_order_prediction=first + directional / 2,
        first_order_residual=first - delta,
        second_order_residual=first + directional / 2 - delta,
    )
    if any(not bool(torch.isfinite(v).all()) for v in observations.values()):
        raise ValueError("curvature execution produced nonfinite observations")
    return _json_value(
        dict(
            schema="nmn.curvature-study.v1",
            status="observed",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_snapshot=snapshot,
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            sample_ids=ids,
            module_names=names,
            output_names=list(model.output_names),
            direction_names=list(directions),
            protocol=dict(
                directions=directions,
                background=base,
                provenance=provenance,
                split=split,
                max_directions=max_directions,
            ),
            observations=observations,
            cost=dict(
                derivative_and_endpoint_forward_calls=forward_calls,
                autograd_calls=gradient_calls,
                seconds=time.perf_counter() - began,
                samples=len(ids),
                directions=len(directions),
                dense_hessian_allocated=False,
                timing_scope="derivative and finite-endpoint execution, excluding initial snapshot",
            ),
            axes=dict(
                gate_gradients=["sample", "output", "module"],
                hessian_vector_products=["sample", "output", "direction", "module"],
                mixed_direction_curvature=[
                    "sample",
                    "output",
                    "left_direction",
                    "right_direction",
                ],
                finite_predictions=["sample", "direction", "output"],
            ),
            limitations=[
                "Local floating-point derivatives are not finite-edit interaction certificates.",
                "A second-order approximation may be worse than first order; raw residuals are retained.",
                "Graph reuse avoids a dense module Hessian but does not establish runtime speedup.",
                "Directions are supplied, not acquired or selected by a sparse recovery algorithm.",
            ],
        )
    )
