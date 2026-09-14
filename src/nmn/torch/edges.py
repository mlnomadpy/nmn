"""Portable per-example measurements of residual edge replacements."""

import hashlib
from pathlib import Path

import torch

from .graph import YatGraph
from .research import _json_value, collect_research_data


def edge_study(model, dataset, *, patches, provenance, split=None):
    """Execute named producer replacements against the unchanged graph.

    Each condition maps receiver -> read slot -> producer -> replacement.
    Values are supplied numeric scalars or arrays, with rows aligned to the
    selected sample IDs. No donor inference, fitting, or causal claim is made.
    """
    if not isinstance(model, YatGraph):
        raise ValueError("edge studies require YatGraph")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("edge replacement provenance is required")
    if (
        not isinstance(patches, dict)
        or not patches
        or any(not isinstance(name, str) or not name for name in patches)
    ):
        raise ValueError("supply named edge-patch conditions")
    ids = dataset.sample_ids(split=split)
    if not ids:
        raise ValueError("selected split has no samples")
    parameter = next(model.parameters())
    normalized: dict[str, dict] = {}
    for name, receivers in patches.items():
        if not isinstance(receivers, dict) or not receivers:
            raise ValueError("each condition requires receiving modules")
        normalized[name] = {}
        for receiver, slots in receivers.items():
            if not isinstance(slots, dict) or not slots:
                raise ValueError("each receiver requires read slots")
            normalized[name][receiver] = {}
            for slot, producers in slots.items():
                if not isinstance(producers, dict) or not producers:
                    raise ValueError("each read slot requires producer replacements")
                values = {}
                for producer, value in producers.items():
                    tensor = torch.as_tensor(
                        value, dtype=parameter.dtype, device=parameter.device
                    )
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError("edge replacement values must be finite")
                    values[producer] = tensor
                normalized[name][receiver][slot] = values
    inputs = torch.tensor(
        [dataset.sample(sid).inputs for sid in ids],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    snapshot = collect_research_data(model, inputs, sample_ids=ids, derivatives=False)
    baseline = torch.as_tensor(
        snapshot["observations"]["baseline"],
        dtype=parameter.dtype,
        device=parameter.device,
    )
    results = {}
    finite = bool(torch.isfinite(baseline).all())
    with torch.no_grad():
        for name, replacements in normalized.items():
            output, trace = model.forward_with_trace(inputs, edge_patches=replacements)
            delta = output - baseline
            finite = (
                finite
                and bool(torch.isfinite(delta).all())
                and all(bool(value.isfinite().all()) for value in trace.values())
            )
            results[name] = dict(
                outputs=output, delta=delta, absolute_delta=delta.abs(), trace=trace
            )
    return _json_value(
        dict(
            schema="nmn.edge-study.v1",
            status="observed" if finite else "nonfinite-observation",
            model_snapshot=snapshot,
            dataset=dataset.to_dict(),
            dataset_sha256=dataset.sha256,
            sample_ids=ids,
            patches=normalized,
            baseline_outputs=baseline,
            results=results,
            protocol=dict(
                split=split,
                provenance=provenance,
                output_order=list(model.output_names),
                semantics="receiver read += replacement - current effective producer write",
                array_order="selected sample ID order; no donor selection",
            ),
            cost=dict(full_forwards=1 + len(patches), samples=len(ids)),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=[
                "Supplied edge replacements isolate explicit additive writes; indirect effects through other producers remain.",
                "Finite numerical effects are not semantic causality, recursive scrubbing, or continuous-domain guarantees.",
                "Replacement arrays are supplied in selected sample order; their origin is declared rather than inferred or verified.",
            ],
        )
    )
