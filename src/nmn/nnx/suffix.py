"""Finite NNX downstream responses to supplied complete boundary states."""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp

from .interpretable import ThreeNeuronYat
from .research import _json, model_snapshot


def suffix_study(model, dataset, *, start_layer, states, provenance, split=None):
    """Measure baseline reconstruction and per-sample downstream state effects."""
    if type(model) is not ThreeNeuronYat:
        raise TypeError("suffix studies support the strict NNX ThreeNeuronYat")
    if type(start_layer) is not int or start_layer not in (0, 1, 2):
        raise ValueError("start_layer must be boundary 0, 1 or 2")
    if (
        not isinstance(provenance, str)
        or not provenance.strip()
        or not isinstance(states, dict)
        or not states
    ):
        raise ValueError("supply named state variants and provenance")
    ids = dataset.sample_ids(split=split)
    if not ids or dataset.input_width != 2:
        raise ValueError("selected population must contain two-dimensional inputs")
    json.dumps(states, allow_nan=False)
    x = jnp.asarray(
        [dataset.sample(sid).inputs for sid in ids], dtype=model.h.centers.dtype
    )
    outputs, trace = model.forward_with_trace(x)
    original = (
        x
        if start_layer == 0
        else (
            jnp.concatenate((trace["h"], x[..., 1:], trace["p"]), axis=-1)
            if start_layer == 1
            else outputs
        )
    )
    replayed, baseline_trace = model.forward_from_state(
        original, start_layer=start_layer
    )
    rows = {}
    for name, values in states.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(values, dict)
            or set(values) != set(ids)
        ):
            raise ValueError(
                "variants must map names to exactly the selected sample IDs"
            )
        state = jnp.asarray([values[sid] for sid in ids], dtype=x.dtype)
        if state.shape != original.shape or not bool(jnp.isfinite(state).all()):
            raise ValueError(
                "states must be finite and have the selected boundary width"
            )
        edited, suffix_trace = model.forward_from_state(state, start_layer=start_layer)
        rows[name] = {
            "state": state,
            "state_delta": state - original,
            "outputs": edited,
            "output_delta": edited - outputs,
            "trace": suffix_trace,
        }
    result = _json(
        {
            "schema": "nmn.nnx-suffix-study.v1",
            "model_snapshot": model_snapshot(model),
            "dataset": dataset.to_dict(),
            "dataset_sha256": dataset.sha256,
            "sample_ids": ids,
            "protocol": {
                "start_layer": start_layer,
                "slot_order": [("u", "v"), ("h", "v", "p"), ("target", "protected")][
                    start_layer
                ],
                "split": split,
                "provenance": provenance,
            },
            "original_state": original,
            "baseline_outputs": outputs,
            "baseline_trace": trace,
            "baseline_suffix_outputs": replayed,
            "baseline_reconstruction_error": replayed - outputs,
            "baseline_suffix_trace": baseline_trace,
            "variants": rows,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": [
                "Finite floating-point observations, not closure or protection certificates.",
                "Supplied states may be unreachable; no decoder or semantic alignment is fitted.",
            ],
        }
    )
    json.dumps(result, allow_nan=False)
    return result
