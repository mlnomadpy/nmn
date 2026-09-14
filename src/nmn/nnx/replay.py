"""Restore and numerically replay strict NNX observations without Torch or pickle."""

import hashlib
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..research.datasets import ResearchDataset
from ..research.native_export import _check_identities
from .interpretable import ThreeNeuronYat, YatExpansion
from .research import collect_research_data


def model_from_snapshot(record):
    """Restore the recorded strict architecture and parameter dtype.

    Configuration, routing, shapes, finite values and content identity are
    checked. JAX x64 must already be enabled for float64 records. This is not a
    converter for general NNX layers; only the explicit research schema loads.
    """
    if record.get("schema") not in ("nmn.nnx-research.v1", "nmn.nnx-model.v1"):
        raise ValueError("unsupported NNX snapshot schema")
    _check_identities(record)
    config = record["configuration"]
    required = {
        "architecture",
        "backend",
        "dtype",
        "distance_mode",
        "modules",
        "routing",
    }
    if (
        set(config) != required
        or config["architecture"] != "three-neuron-yat"
        or config["backend"] != "flax-nnx"
        or config["distance_mode"] != "direct"
        or config["routing"] != {"h": ["u"], "p": ["v"], "y": ["h", "v"]}
        or config["dtype"] not in ("float32", "float64")
    ):
        raise ValueError("unsupported NNX execution configuration")
    names = ThreeNeuronYat.state_names
    if set(config["modules"]) != set(names) or set(record["parameters"]) != set(names):
        raise ValueError("snapshot must contain exactly h, p and y")
    dtype = np.dtype(config["dtype"])
    if dtype == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("enable JAX x64 before loading a float64 snapshot")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        spec = config["modules"][name]
        if set(spec) != {"epsilon", "num_centers"}:
            raise ValueError("unsupported module configuration")
        n = spec["num_centers"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("num_centers must be a positive integer")
        parameters = record["parameters"][name]
        if set(parameters) != {"centers", "coefficients"}:
            raise ValueError("unsupported module parameters")
        arrays[name] = {}
        for key, shape in [
            ("centers", (n, 2 if name == "y" else 1)),
            ("coefficients", (1, n)),
        ]:
            source = np.asarray(parameters[key])
            if source.shape != shape or source.dtype.kind not in "fi":
                raise ValueError("invalid parameter shape or numeric type")
            array = source.astype(dtype)
            if not np.isfinite(array).all() or array.tolist() != source.tolist():
                raise ValueError("parameters must be finite and exact in saved dtype")
            arrays[name][key] = array
    model = ThreeNeuronYat(dtype=dtype, rngs=nnx.Rngs(0))
    for name in names:
        module = YatExpansion(
            2 if name == "y" else 1,
            1,
            dtype=dtype,
            rngs=nnx.Rngs(0),
            **config["modules"][name],
        )
        module.kernel.kernel[...] = jnp.asarray(arrays[name]["centers"].T)
        module.coefficients[...] = jnp.asarray(arrays[name]["coefficients"])
        setattr(model, name, module)
    return model


def replay_native_record(record, *, atol=1e-10, rtol=1e-8):
    """Recompute observations on CPU; compare measurements with declared tolerances."""
    for value in (atol, rtol):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("tolerances must be finite nonnegative numbers")
    # Strict JSON rejects nonfinite values even outside the compared fields.
    serialized = json.dumps(record, sort_keys=True, allow_nan=False)
    dataset = ResearchDataset.from_dict(record["dataset"])
    if dataset.sha256 != record["dataset_sha256"]:
        raise ValueError("dataset identity mismatch")
    if record.get("schema") == "nmn.nnx-suffix-study.v1":
        from .suffix import suffix_study

        with jax.default_device(jax.devices("cpu")[0]):
            model = model_from_snapshot(record["model_snapshot"])
            actual = suffix_study(
                model,
                dataset,
                start_layer=record["protocol"]["start_layer"],
                split=record["protocol"]["split"],
                provenance=record["protocol"]["provenance"],
                states={
                    name: dict(zip(record["sample_ids"], row["state"]))
                    for name, row in record["variants"].items()
                },
            )
        fields = [
            "model_snapshot",
            "dataset_sha256",
            "sample_ids",
            "protocol",
            "original_state",
            "baseline_outputs",
            "baseline_trace",
            "baseline_suffix_outputs",
            "baseline_reconstruction_error",
            "baseline_suffix_trace",
            "variants",
            "limitations",
        ]
    else:
        with jax.default_device(jax.devices("cpu")[0]):
            model = model_from_snapshot(record)
            actual = collect_research_data(
                model,
                dataset,
                split=record["split"],
                edits={name: row["controls"] for name, row in record["edits"].items()},
                derivatives=record["derivatives"] is not None,
            )
        fields = [
            "configuration",
            "parameters",
            "model_sha256",
            "dataset_sha256",
            "split",
            "sample_ids",
            "output_names",
            "outputs",
            "trace",
            "geometry",
            "edits",
            "derivatives",
            "capabilities",
            "assurance",
        ]
    mismatches = []
    checked = 0
    maximum = 0.0
    availability_changes = {}

    def compare(saved, current, path):
        nonlocal checked, maximum
        # Suffix API availability does not alter a saved full-forward execution.
        # Execution facts such as graph completeness remain normal comparisons.
        if (
            path == "/capabilities/suffix_replay"
            and type(saved) is bool
            and type(current) is bool
        ):
            if saved != current:
                availability_changes[path] = {"saved": saved, "current": current}
            return
        checked += 1
        if isinstance(saved, dict) and isinstance(current, dict):
            for key in sorted(set(saved) | set(current)):
                child = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key not in saved or key not in current:
                    mismatches.append({"path": child, "reason": "missing field"})
                else:
                    compare(saved[key], current[key], child)
            return
        if (
            isinstance(saved, list)
            and isinstance(current, list)
            and len(saved) == len(current)
        ):
            for i, (a, b) in enumerate(zip(saved, current)):
                compare(a, b, path + "/" + str(i))
            return
        if (
            isinstance(saved, (int, float))
            and not isinstance(saved, bool)
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
        ):
            error = abs(saved - current)
            maximum = max(maximum, error)
            if error <= atol + rtol * abs(saved):
                return
        elif type(saved) is type(current) and saved == current:
            return
        mismatches.append({"path": path, "saved": saved, "replayed": current})

    for field in fields:
        if field not in record:
            mismatches.append({"path": "/" + field, "reason": "missing field"})
        else:
            compare(record[field], actual[field], "/" + field)
    return {
        "schema": "nmn.native-replay.v1",
        "source_schema": record["schema"],
        "status": "mismatch" if mismatches else "matched",
        "record_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "model_sha256": actual.get(
            "model_sha256", actual.get("model_snapshot", {}).get("model_sha256")
        ),
        "compared_fields": fields,
        "tolerances": {
            "atol": atol,
            "rtol": rtol,
            "rule": "abs(saved-replayed) <= atol + rtol*abs(saved)",
        },
        "checked_nodes": checked,
        "maximum_absolute_error": maximum,
        "mismatches": mismatches,
        "execution": actual,
        "availability_changes": availability_changes,
        "ignored_paths": [
            "/runtime",
            "/source_sha256",
            "/elapsed_seconds",
            "/capabilities/suffix_replay",
        ],
        "limitations": [
            "CPU numerical replay is not a scientific certificate.",
            "No training, cached execution or external model conversion is replayed.",
        ],
    }
