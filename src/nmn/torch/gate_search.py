"""Bounded gradient proposals followed by measured finite gate-edit selection."""

import hashlib
import math
import time
from pathlib import Path

import torch

from .interpretable import Intervention
from .research import _json_value
from .selection import select_edit


def search_gates(
    model,
    dataset,
    *,
    modules,
    targets,
    target_outputs,
    protected_outputs,
    protection_tolerance,
    provenance,
    max_steps,
    max_seconds,
    learning_rate,
    protection_weight,
    selection_split="tuning",
    validation_split="validation",
):
    """Propose gates in [0,1] with Adam, then freeze a measured feasible winner.

    Only gates are optimized. Generation uses selection samples; the shared
    selector reexecutes recorded candidates and validates only its frozen winner.
    The time budget covers proposal generation, not final candidate evaluation.
    """
    if (
        not isinstance(modules, (list, tuple))
        or not modules
        or any(not isinstance(m, str) for m in modules)
        or len(set(modules)) != len(modules)
        or not set(modules) <= set(model.state_names)
    ):
        raise ValueError("modules must be unique native module names")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    for name, value, positive in (
        ("max_seconds", max_seconds, True),
        ("learning_rate", learning_rate, True),
        ("protection_weight", protection_weight, False),
        ("protection_tolerance", protection_tolerance, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)
        ):
            raise ValueError(name + " has an invalid finite range")
    for names in (target_outputs, protected_outputs):
        if (
            not isinstance(names, (list, tuple))
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
    if (
        not selection_split
        or not validation_split
        or selection_split == validation_split
        or not ids
        or not heldout
    ):
        raise ValueError(
            "require separate nonempty selection and validation populations"
        )
    if (
        not isinstance(provenance, str)
        or not provenance.strip()
        or not isinstance(targets, dict)
        or set(targets) != set(ids) | set(heldout)
    ):
        raise ValueError("provide provenance and targets for exactly both populations")
    for row in targets.values():
        if (
            not isinstance(row, dict)
            or set(row) != set(target_outputs)
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in row.values()
            )
        ):
            raise ValueError(
                "targets must contain finite values for the declared outputs"
            )
    parameter = next(model.parameters())
    x = torch.tensor(
        [dataset.sample(s).inputs for s in ids],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    expected = x.new_tensor(
        [[targets[s][name] for name in target_outputs] for s in ids]
    )
    ti = [model.output_names.index(name) for name in target_outputs]
    pi = [model.output_names.index(name) for name in protected_outputs]
    with torch.no_grad():
        baseline = model(x).detach()
    gates = torch.ones(len(modules), dtype=x.dtype, device=x.device, requires_grad=True)
    optimizer = torch.optim.Adam([gates], lr=learning_rate)
    candidates, history = {}, []
    started = time.perf_counter()
    status, error = "step-budget-completed", None
    width = len(str(max_steps))
    for step in range(max_steps + 1):
        if step and time.perf_counter() - started >= max_seconds:
            status = "time-budget-stopped"
            break
        controls = {name: Intervention(gate=gates[i]) for i, name in enumerate(modules)}
        output = model(x, controls)
        target_loss = (output[:, ti] - expected).square().mean()
        protected_delta = output[:, pi] - baseline[:, pi]
        protected_loss = protected_delta.square().mean() if pi else output.new_zeros(())
        loss = target_loss + protection_weight * protected_loss
        if not bool(torch.isfinite(loss)):
            status, error = "nonfinite-stopped", "nonfinite proposal loss"
            break
        candidate_id = "step-" + str(step).zfill(width)
        candidates[candidate_id] = {
            name: {"gate": float(gates[i].detach())} for i, name in enumerate(modules)
        }
        history.append(
            dict(
                candidate_id=candidate_id,
                gates=gates.detach().clone(),
                target_mse=target_loss.detach(),
                protection_mse=protected_loss.detach(),
                objective=loss.detach(),
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        if step == max_steps:
            break
        gradient = (
            torch.autograd.grad(loss, gates, allow_unused=True)[0]
            if loss.requires_grad
            else None
        )
        if gradient is None:
            gradient = torch.zeros_like(gates)
        if not bool(torch.isfinite(gradient).all()):
            status, error = "nonfinite-stopped", "nonfinite proposal gradient"
            break
        gates.grad = gradient.detach()
        optimizer.step()
        with torch.no_grad():
            gates.clamp_(0, 1)
    if not candidates:
        raise ValueError("no finite gate candidate could be generated")
    selection = select_edit(
        model,
        dataset,
        candidates=candidates,
        targets=targets,
        target_outputs=target_outputs,
        protected_outputs=protected_outputs,
        protection_tolerance=protection_tolerance,
        provenance=provenance,
        max_candidates=len(candidates),
        selection_split=selection_split,
        validation_split=validation_split,
    )
    return _json_value(
        dict(
            schema="nmn.gate-search.v1",
            status=status,
            error=error,
            model_snapshot=selection["model_snapshot"],
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            protocol=dict(
                modules=list(modules),
                gate_interval=[0, 1],
                max_steps=max_steps,
                max_seconds=max_seconds,
                learning_rate=learning_rate,
                protection_weight=protection_weight,
                selection_split=selection_split,
                validation_split=validation_split,
                objective="target MSE + protection_weight * protected-output-change MSE",
                time_budget_scope="proposal loop only; final selection and validation excluded",
            ),
            proposals=history,
            candidates=candidates,
            selection=selection,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "Projected Adam proposals do not establish global optimality or feasibility outside the measured selection population.",
                "The penalty is not a protection certificate; final feasibility uses actual per-coordinate measured changes.",
                "Validation cannot choose the winner, but repeated user-guided runs may leak validation information.",
                "The embedded edit-selection record supports numerical replay; proposal optimization itself has no replay adapter.",
            ],
        )
    )
