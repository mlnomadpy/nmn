"""Explicit, bounded CPU task and donor-supervised training protocols."""

import copy
import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .interpretable import Intervention
from .research import _json_value, collect_research_data, model_from_snapshot


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    batch_size: int = 16
    learning_rate: float = 1e-3
    train_split: str = "train"
    validation_split: str = "validation"
    evaluate_every: int = 10
    max_seconds: float = 60.0
    intervention_weight: float = 0.0
    separate_pair_rng: bool = False
    detach_donor: bool = True

    def validate(self):
        if type(self.detach_donor) is not bool:
            raise ValueError("detach_donor must be a boolean")
        if type(self.separate_pair_rng) is not bool:
            raise ValueError("separate_pair_rng must be a boolean")
        for name in ("max_steps", "batch_size", "evaluate_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.intervention_weight) or self.intervention_weight < 0:
            raise ValueError("intervention_weight must be finite and nonnegative")
        if (
            not self.train_split
            or not self.validation_split
            or self.train_split == self.validation_split
        ):
            raise ValueError(
                "declare distinct training and checkpoint-selection splits"
            )


def train_native(
    initial_snapshot,
    dataset,
    targets,
    *,
    config: TrainingConfig,
    architecture_contract: dict,
    target_provenance: str,
    seeds=(0,),
    pairs=(),
):
    """Explicitly fit independent copies of a supplied native model on CPU.

    Seeds govern minibatch/pair sampling, not initialization: every run starts
    from the SAME supplied parameter snapshot. Validation is checkpoint-selection
    data, not final held-out evidence. Donor-supervised loss uses current donor
    writes with an explicit detached/joint gradient policy and recomputes base
    descendants; this declared protocol is not an automatic reproduction of a published IIT algorithm.

    Requires declared architecture/semantic/intervention/scope/example references.
    Those declarations are retained, not independently proved by this function.
    Returns all runs, failures and budget stops, with selected checkpoints.
    Never mutates the caller's model or launches external compute.
    """
    config.validate()
    required = (
        "architecture_id",
        "semantic_specification",
        "intervention_specification",
        "guarantee_scope",
        "worked_example",
    )
    if any(
        not isinstance(architecture_contract.get(k), str)
        or not architecture_contract[k]
        for k in required
    ):
        raise ValueError("complete the declared architecture contract before training")
    if not isinstance(target_provenance, str) or not target_provenance:
        raise ValueError("target_provenance is required")
    seeds = tuple(seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 0 or s >= 2**63
            for s in seeds
        )
    ):
        raise ValueError("seeds must be unique integers in [0, 2^63)")
    train_ids = dataset.sample_ids(split=config.train_split)
    validation_ids = dataset.sample_ids(split=config.validation_split)
    if not train_ids or not validation_ids:
        raise ValueError(
            "training and checkpoint-selection splits must both have samples"
        )
    if set(targets) != set(train_ids + validation_ids):
        raise ValueError(
            "targets must cover exactly training and checkpoint-selection samples"
        )
    if bool(pairs) != bool(config.intervention_weight):
        raise ValueError(
            "donor pairs require a positive intervention weight, and vice versa"
        )
    if pairs:
        dataset.validate_pairs(pairs)
        if any(
            dataset.sample(sid).split != config.train_split
            for p in pairs
            for sid in (p.base_id, p.donor_id)
        ):
            raise ValueError("training donor pairs may access only training samples")
    # Validate restoration, supervision shapes and pair measurements before updates.
    with torch.random.fork_rng(devices=[]):
        prototype = model_from_snapshot(initial_snapshot)
    if not any(p.requires_grad for p in prototype.parameters()):
        raise ValueError("model has no trainable parameters")
    outputs = tuple(prototype.output_names)
    target_tensor = torch.as_tensor(
        [targets[sid] for sid in train_ids + validation_ids], dtype=torch.float64
    )
    if target_tensor.shape != (
        len(train_ids) + len(validation_ids),
        len(outputs),
    ) or not bool(torch.isfinite(target_tensor).all()):
        raise ValueError("targets must be finite vectors in model output order")
    for pair in pairs:
        if (
            not pair.expected
            or not set(pair.expected) <= set(outputs)
            or not set(pair.modules) <= set(prototype.state_names)
        ):
            raise ValueError(
                "training pairs need valid modules and named expected outputs"
            )
    x_train = torch.tensor(
        [dataset.sample(sid).inputs for sid in train_ids], dtype=torch.float64
    )
    x_validation = torch.tensor(
        [dataset.sample(sid).inputs for sid in validation_ids], dtype=torch.float64
    )
    y_train, y_validation = (
        target_tensor[: len(train_ids)],
        target_tensor[len(train_ids) :],
    )
    train_index = {sid: i for i, sid in enumerate(train_ids)}
    runs = []
    for seed in seeds:
        with torch.random.fork_rng(devices=[]):
            model = model_from_snapshot(initial_snapshot)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pair_seed = (seed + 2**32) % 2**63
        pair_generator = (
            torch.Generator(device="cpu").manual_seed(pair_seed)
            if config.separate_pair_rng
            else generator
        )
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
        )
        history = []
        best_state = copy.deepcopy(model.state_dict())
        completed, best_step, status = 0, 0, "completed"
        started = time.perf_counter()
        error = None
        best_loss = None
        model.train()
        try:
            with torch.no_grad():
                initial_validation = (
                    ((model(x_validation) - y_validation) ** 2).mean().item()
                )
            if not math.isfinite(initial_validation):
                raise ValueError("nonfinite initial validation loss")
            best_loss = initial_validation
            history.append({"step": 0, "validation_mse": best_loss})
            for step in range(1, config.max_steps + 1):
                if time.perf_counter() - started >= config.max_seconds:
                    status = "budget-stopped"
                    break
                indices = torch.randperm(len(train_ids), generator=generator)[
                    : config.batch_size
                ]
                optimizer.zero_grad(set_to_none=True)
                task_loss = ((model(x_train[indices]) - y_train[indices]) ** 2).mean()
                intervention_loss = task_loss.new_zeros(())
                if pairs:
                    losses = []
                    chosen = torch.randperm(len(pairs), generator=pair_generator)[
                        : config.batch_size
                    ]
                    for idx in chosen.tolist():
                        pair = pairs[idx]
                        donor = x_train[
                            train_index[pair.donor_id] : train_index[pair.donor_id] + 1
                        ]
                        base = x_train[
                            train_index[pair.base_id] : train_index[pair.base_id] + 1
                        ]
                        with torch.set_grad_enabled(not config.detach_donor):
                            _, donor_trace = model.forward_with_trace(donor)
                        prediction = model(
                            base,
                            {
                                name: Intervention(
                                    replacement=(
                                        donor_trace[name].detach()
                                        if config.detach_donor
                                        else donor_trace[name]
                                    )
                                )
                                for name in pair.modules
                            },
                        )
                        coordinate_ids = [outputs.index(name) for name in pair.expected]
                        expected = prediction.new_tensor(list(pair.expected.values()))
                        losses.append(
                            (prediction[0, coordinate_ids] - expected).square().mean()
                        )
                    intervention_loss = torch.stack(losses).mean()
                loss = task_loss + config.intervention_weight * intervention_loss
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                if any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                ):
                    raise ValueError("nonfinite training gradient")
                optimizer.step()
                completed = step
                if step % config.evaluate_every == 0 or step == config.max_steps:
                    with torch.no_grad():
                        validation = (
                            ((model(x_validation) - y_validation) ** 2).mean().item()
                        )
                    if not math.isfinite(validation):
                        raise ValueError("nonfinite validation loss")
                    history.append(
                        {
                            "step": step,
                            "pre_update_task_mse": task_loss.item(),
                            "pre_update_intervention_mse": intervention_loss.item(),
                            "validation_mse": validation,
                        }
                    )
                    if validation < best_loss:
                        best_loss, best_step = validation, step
                        best_state = copy.deepcopy(model.state_dict())
        except (ValueError, RuntimeError) as exc:
            status, error = "failed", str(exc)
        optimization_seconds = time.perf_counter() - started
        # Always restore the last finite selected checkpoint before instrumentation.
        model.load_state_dict(best_state)
        model.eval()
        checkpoint_error = None
        try:
            snapshot = collect_research_data(
                model,
                x_validation,
                sample_ids=validation_ids,
                derivatives=False,
                metadata={"role": "checkpoint-selection; not final evaluation"},
            )
        except (ValueError, RuntimeError) as exc:
            snapshot, checkpoint_error = None, str(exc)
            status = "failed"
        runs.append(
            {
                "seed": seed,
                "status": status,
                "error": error,
                "steps_completed": completed,
                "best_step": best_step,
                "best_validation_mse": best_loss,
                "optimization_seconds": optimization_seconds,
                "history": history,
                "selected_checkpoint": snapshot,
                "checkpoint_error": checkpoint_error,
            }
        )
    return _json_value(
        {
            "schema": "nmn.native-training.v1",
            "configuration": asdict(config),
            "architecture_contract": architecture_contract,
            "target_provenance": target_provenance,
            "initial_model_sha256": initial_snapshot["model_sha256"],
            "initial_trainability": {
                name: p.requires_grad for name, p in prototype.named_parameters()
            },
            "dataset": dataset.to_dict(),
            "dataset_sha256": dataset.sha256,
            "targets": targets,
            "train_ids": train_ids,
            "checkpoint_selection_ids": validation_ids,
            "protocol": (
                "task-only"
                if not pairs
                else (
                    "task-and-detached-donor-supervision"
                    if config.detach_donor
                    else "task-and-joint-donor-supervision"
                )
            ),
            "donor_pairs": [asdict(pair) for pair in pairs],
            "seed_scope": (
                "minibatches use seed; pairs use (seed + 2**32) modulo 2**63; common initialization"
                if config.separate_pair_rng
                else "minibatch/pair order; common initialization"
            ),
            "selection_rule": "lowest observed validation MSE; strict improvement; ties retain earlier checkpoint",
            "runs": runs,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": [
                "Organization/routing is imposed; training does not prove semantic recovery.",
                "Validation chooses checkpoints and is not untouched final evaluation.",
                "The wall-time cap is checked between steps, not a preemptive execution limit.",
                "Multi-seed measurements are not a training success-rate theorem.",
            ],
        }
    )
