"""Observed downstream responses to supplied intermediate graph states."""

import hashlib
from pathlib import Path

import torch

from .graph import YatGraph
from .research import _json_value, collect_research_data


def suffix_study(model, dataset, *, start_layer, states, provenance, split=None):
    """Compare explicit states against the unchanged boundary state on each sample.

    ``states`` maps variant names to sample-ID/vector mappings covering exactly
    the selected population. Complete residual states are supplied in graph slot
    order. This measures downstream reconstruction/perturbation effects without
    discovering a summary map, fitting a decoder or certifying domain closure.
    """
    if not isinstance(model, YatGraph):
        raise ValueError("suffix studies require YatGraph")
    if type(start_layer) is not int or not 0 <= start_layer <= len(model.layer_specs):
        raise ValueError("start_layer must identify a graph boundary")
    if not isinstance(provenance, str) or not provenance or not states:
        raise ValueError("supply state variants and their provenance")
    ids = dataset.sample_ids(split=split)
    if not ids:
        raise ValueError("selected split has no samples")
    parameter = next(model.parameters())
    inputs = torch.tensor(
        [dataset.sample(sid).inputs for sid in ids],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    variants = {}
    for name, values in states.items():
        if not isinstance(name, str) or not name or set(values) != set(ids):
            raise ValueError(
                "state variants require names and exactly the selected sample IDs"
            )
        value = torch.as_tensor(
            [values[sid] for sid in ids], dtype=parameter.dtype, device=parameter.device
        )
        if value.shape != (len(ids), len(model.slots)) or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(
                "state variants must be finite matrices with one column per graph slot"
            )
        variants[name] = value
    snapshot = collect_research_data(model, inputs, sample_ids=ids, derivatives=False)
    with torch.no_grad():
        output, trace = model.forward_with_trace(inputs)
        original = trace[f"state.{start_layer}"]
        replayed, baseline_trace = model.forward_from_state(
            original, start_layer=start_layer
        )
        rows = {}
        for name, state in variants.items():
            edited, suffix_trace = model.forward_from_state(
                state, start_layer=start_layer
            )
            rows[name] = dict(
                state=state,
                state_delta=state - original,
                outputs=edited,
                output_delta=edited - output,
                trace=suffix_trace,
            )
    return _json_value(
        dict(
            schema="nmn.suffix-study.v1",
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            model_snapshot=snapshot,
            sample_ids=ids,
            protocol=dict(
                start_layer=start_layer,
                slot_order=list(model.slots),
                split=split,
                provenance=provenance,
            ),
            original_state=original,
            baseline_outputs=output,
            baseline_suffix_outputs=replayed,
            baseline_reconstruction_error=replayed - output,
            baseline_suffix_trace=baseline_trace,
            variants=rows,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "Supplied state responses are finite numerical observations, not a closure or protection certificate.",
                "State variants may be unreachable from model inputs; no realizability is inferred.",
                "No summary encoder/decoder or semantic alignment is learned by this study.",
            ],
        )
    )
