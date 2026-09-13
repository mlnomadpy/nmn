"""Matched finite-population replay comparisons with explicit cost accounting."""

import hashlib
import os
import platform
import time
from pathlib import Path

import torch

from .graph import YatGraph
from .interpretable import ThreeNeuronYat
from .research import _json_value, collect_research_data


def benchmark_models(
    models,
    dataset,
    *,
    edits,
    expected_outputs=None,
    protected_outputs=(),
    split=None,
    repeats=5,
    warmup=1,
):
    """Evaluate every supplied native model on the same inputs and native edits.

    CPU-only timing of actual batched forwards; traces/geometry serialization is
    timed separately. Keeps every result or execution failure. No model fitting,
    predictor selection, certification algorithm or performance winner is implied.
    Optional expected_outputs is an (N, O) matrix aligned to selected sample IDs
    and the common model output order, supplied by the research task.
    """
    if len(models) < 2 or any(not isinstance(name, str) or not name for name in models):
        raise ValueError("provide at least two named models")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    ids = dataset.sample_ids(split=split)
    if not ids:
        raise ValueError("selected population is empty")

    def routing(model):
        if isinstance(model, YatGraph):
            config = model.configuration()
            return {
                "slots": config["slots"],
                "inputs": config["input_names"],
                "outputs": config["output_names"],
                "update": config["update"],
                "layers": [
                    [
                        {k: spec[k] for k in ("name", "reads", "writes")}
                        for spec in layer
                    ]
                    for layer in config["layers"]
                ],
            }
        if isinstance(model, ThreeNeuronYat):
            return {"class": "ThreeNeuronYat", "routing": "h=H(u), p=P(v), y=Y(h,v)"}
        raise TypeError("benchmark requires native graph or reference models")

    first = next(iter(models.values()))
    common_routing = routing(first)
    output_names, module_names = tuple(first.output_names), tuple(first.state_names)
    if len(set(protected_outputs)) != len(protected_outputs) or not set(
        protected_outputs
    ) <= set(output_names):
        raise ValueError("protected outputs must be unique common output names")
    expected = (
        None
        if expected_outputs is None
        else torch.as_tensor(expected_outputs, dtype=torch.float64)
    )
    if expected is not None and (
        expected.shape != (len(ids), len(output_names))
        or not bool(torch.isfinite(expected).all())
    ):
        raise ValueError(
            "expected_outputs must be finite with shape (samples, outputs)"
        )
    controls = {
        name: {
            key: {"gate": value.gate, "replacement": value.replacement}
            for key, value in edit.items()
        }
        for name, edit in edits.items()
    }
    rows = {}
    with torch.no_grad():
        for name, model in models.items():
            started = time.perf_counter()
            try:
                if (
                    tuple(model.output_names) != output_names
                    or tuple(model.state_names) != module_names
                ):
                    raise ValueError(
                        "model output/module names differ from the common contract"
                    )
                if routing(model) != common_routing:
                    raise ValueError("model routing differs from the common contract")
                parameter = next(model.parameters())
                if parameter.device.type != "cpu":
                    raise ValueError(
                        "this comparison runner measures CPU execution only"
                    )
                inputs = torch.tensor(
                    [dataset.sample(sid).inputs for sid in ids], dtype=parameter.dtype
                )

                def measure(control):
                    for _ in range(warmup):
                        model(inputs, control)
                    durations = []
                    for _ in range(repeats):
                        began = time.perf_counter()
                        value = model(inputs, control)
                        durations.append(time.perf_counter() - began)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("nonfinite model output")
                    return value, {
                        "seconds_per_batch": durations,
                        "median_seconds": sorted(durations)[len(durations) // 2],
                        "forward_calls": warmup + repeats,
                    }

                baseline, baseline_cost = measure({})
                effects = {}
                for edit_name, control in edits.items():
                    output, cost = measure(control)
                    effects[edit_name] = {
                        "outputs": output,
                        "delta": output - baseline,
                        "protected_delta": {
                            key: output[:, output_names.index(key)]
                            - baseline[:, output_names.index(key)]
                            for key in protected_outputs
                        },
                        "cost": cost,
                    }
                instrumentation_start = time.perf_counter()
                snapshot = collect_research_data(
                    model, inputs, sample_ids=ids, derivatives=False
                )
                instrumentation_seconds = time.perf_counter() - instrumentation_start
                rows[name] = {
                    "status": "measured",
                    "model_snapshot": snapshot,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "trainable_parameter_count": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "baseline_outputs": baseline,
                    "baseline_cost": baseline_cost,
                    "baseline_squared_error": (
                        None
                        if expected is None
                        else (baseline.double() - expected).square()
                    ),
                    "baseline_mse": (
                        None
                        if expected is None
                        else (baseline.double() - expected).square().mean()
                    ),
                    "edits": effects,
                    "instrumentation_seconds": instrumentation_seconds,
                    "total_seconds": time.perf_counter() - started,
                    "training_cost": "not measured; models supplied",
                    "certification": "not performed",
                }
            except (
                ValueError,
                RuntimeError,
                TypeError,
                KeyError,
                AttributeError,
            ) as exc:
                rows[name] = {
                    "status": "failed",
                    "error": str(exc),
                    "total_seconds": time.perf_counter() - started,
                }
    return _json_value(
        {
            "schema": "nmn.native-benchmark.v1",
            "dataset": dataset.to_dict(),
            "dataset_sha256": dataset.sha256,
            "method_order": list(models),
            "hardware": {
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
            },
            "sample_ids": ids,
            "contract": {
                "routing": common_routing,
                "split": split,
                "output_names": output_names,
                "module_names": module_names,
                "protected_outputs": list(protected_outputs),
                "edits": controls,
                "expected_outputs": expected,
                "repeats": repeats,
                "warmup": warmup,
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
            },
            "methods": rows,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": [
                "Equal routing or width does not imply matched capacity, task competence or compute.",
                "This is direct replay; no certification or acquisition algorithm is benchmarked.",
                "Timing is instrumented CPU wall time and is not a hardware-independent speed claim.",
                "Finite untrained comparisons cannot establish a kernel-specific advantage.",
            ],
        }
    )
