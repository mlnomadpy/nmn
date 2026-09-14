"""JSON research observations from the strict NNX three-neuron architecture."""

import hashlib
import json
import platform
import time
from pathlib import Path

import flax
import jax
import jax.numpy as jnp

from .interpretable import ThreeNeuronYat


def _json(value):
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def model_snapshot(model):
    """Serialize a strict NNX model without running an observation or using pickle."""
    if type(model) is not ThreeNeuronYat:
        raise TypeError("only the strict NNX ThreeNeuronYat is supported")
    dtype = model.h.centers.dtype
    configuration = {
        "architecture": "three-neuron-yat",
        "backend": "flax-nnx",
        "dtype": str(dtype),
        "distance_mode": "direct",
        "modules": {
            name: {
                "epsilon": getattr(model, name).epsilon,
                "num_centers": getattr(model, name).num_centers,
            }
            for name in model.state_names
        },
        "routing": {"h": ["u"], "p": ["v"], "y": ["h", "v"]},
    }
    parameters = {
        name: {
            "centers": _json(getattr(model, name).centers),
            "coefficients": _json(getattr(model, name).coefficients[...]),
        }
        for name in model.state_names
    }
    identity = {"configuration": configuration, "parameters": parameters}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return {"schema": "nmn.nnx-model.v1", **identity, "model_sha256": digest}


def collect_research_data(model, dataset, *, split, edits=None, derivatives=True):
    """Collect per-example traces, center geometry and optional local derivatives.

    No cached execution, semantic inference, statistical or interval assurance is
    supplied. Input/gate Jacobians and gate Hessians are evaluated at baseline.
    Collection is synchronous and rejects nonfinite JSON before returning.
    """
    if type(model) is not ThreeNeuronYat:
        raise TypeError("only the strict NNX ThreeNeuronYat is supported")
    ids = dataset.sample_ids(split=split)
    if not ids or dataset.input_width != 2:
        raise ValueError("split must contain two-dimensional samples")
    edits = {} if edits is None else edits
    if not isinstance(edits, dict) or any(
        not isinstance(k, str) or not k for k in edits
    ):
        raise ValueError("edits must map nonempty names to controls")
    # Reject nonfinite controls even if replacement would override a gate.
    json.dumps(_json(edits), allow_nan=False)
    start = time.perf_counter()
    dtype = model.h.centers.dtype
    x = jnp.asarray([dataset.sample(sid).inputs for sid in ids], dtype=dtype)
    snapshot = model_snapshot(model)
    identity = {key: snapshot[key] for key in ("configuration", "parameters")}
    digest = snapshot["model_sha256"]
    outputs, trace = model.forward_with_trace(x)
    geometry = {}
    for name in model.state_names:
        module = getattr(model, name)
        centers = module.centers
        gram = (centers @ centers.T) ** 2 / (
            jnp.sum((centers[:, None] - centers[None, :]) ** 2, axis=-1)
            + module.epsilon
        )
        coefficients = module.coefficients[...]
        geometry[name] = {
            "center_gram": gram,
            "center_gram_eigenvalues": jnp.linalg.eigvalsh(gram),
            "local_rkhs_inner_products": coefficients @ gram @ coefficients.T,
            "features": module.features(trace[name + ".input"]),
        }
    observations = {}
    for name, controls in edits.items():
        edited, edited_trace = model.forward_with_trace(x, controls)
        observations[name] = {
            "controls": controls,
            "outputs": edited,
            "delta": edited - outputs,
            "trace": edited_trace,
        }
    differential = None
    if derivatives:

        def gated(point, gates):
            return model(
                point,
                {name: {"gate": gates[i]} for i, name in enumerate(model.state_names)},
            )

        gates = jnp.ones(3, dtype=dtype)
        differential = {
            "input_jacobian": jax.vmap(jax.jacfwd(lambda point: model(point)))(x),
            "gate_jacobian": jax.vmap(jax.jacfwd(gated, argnums=1), (0, None))(
                x, gates
            ),
            "gate_hessian": jax.vmap(
                jax.jacfwd(jax.jacrev(gated, argnums=1), argnums=1), (0, None)
            )(x, gates),
            "gate_order": model.state_names,
        }
    record = _json(
        {
            "schema": "nmn.nnx-research.v1",
            **identity,
            "model_sha256": digest,
            "dataset": dataset.to_dict(),
            "dataset_sha256": dataset.sha256,
            "split": split,
            "sample_ids": ids,
            "output_names": model.output_names,
            "outputs": outputs,
            "trace": trace,
            "geometry": geometry,
            "edits": observations,
            "derivatives": differential,
            "source_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in ("research.py", "interpretable.py")
            },
            "runtime": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "flax": flax.__version__,
                "devices": [str(d) for d in x.devices()],
            },
            "capabilities": {
                "complete_graph": True,
                "cached_execution": False,
                "suffix_replay": False,
                "interval_certificates": False,
                "arbitrary_external_models": False,
            },
            "assurance": "floating-point observations; no statistical or continuous-domain guarantee",
        }
    )
    # Array conversion synchronizes asynchronous JAX work before timing is recorded.
    record["elapsed_seconds"] = time.perf_counter() - start
    json.dumps(record, allow_nan=False)
    return record
