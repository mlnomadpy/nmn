"""Native donor interventions with explicit datasets and reference measurements."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import torch

from ..research.datasets import DonorPair, ResearchDataset, ResearchSample
from .graph import YatGraph
from .interpretable import Intervention
from .research import _json_value, collect_research_data

Reference = Callable[[ResearchSample, ResearchSample, tuple], Mapping[str, float]]


def donor_study(
    model,
    dataset: ResearchDataset,
    pairs: Sequence[DonorPair],
    *,
    protected_outputs: Sequence[str] = (),
    reference: Optional[Reference] = None,
    reference_id: Optional[str] = None,
    allow_cross_split: bool = False,
    match_semantics: Sequence[str] = (),
    read_slots: Optional[Mapping[str, Sequence[str]]] = None,
    edge_routes=None,
):
    """Run actual donor replacements and compare supplied reference measurements.

    Donor states come from the unchanged network and remain fixed during each
    base replay. All requested modules are replaced together, with descendants
    recomputed. On YatGraph, replacement applies to a module WRITE vector, not
    an entire residual state slot. Each row retains donor/base/edited traces.

    With read_slots on YatGraph, modules instead receive the listed donor input
    coordinates from their unchanged donor trace. Other readers and shared state
    are untouched; the receiving module and descendants recompute. Routes must
    cover exactly the modules named across pairs. This is a receiving-slot patch,
    not isolation of a particular producer's contribution to a residual sum.

    With edge_routes, pair modules name receivers and routes identify producer
    writes to transfer at those receivers. Unchanged donor writes remain fixed;
    other residual terms survive. Whole-slot and edge routes are mutually exclusive.

    Reference callbacks receive data records, never the model or its outputs.
    Supply a stable versioned reference_id; callbacks are not discovered semantic
    truth. Pair expectations and callbacks are mutually exclusive. No threshold,
    population guarantee, fitting or automatic donor selection is implied.
    """
    dataset.validate_pairs(
        pairs, allow_cross_split=allow_cross_split, match_semantics=match_semantics
    )
    if reference is not None and (
        not isinstance(reference_id, str) or not reference_id
    ):
        raise ValueError(
            "a versioned reference_id is required with a reference callback"
        )
    if reference is None and reference_id is not None:
        raise ValueError("reference_id requires a reference callback")
    if reference is not None and any(pair.expected for pair in pairs):
        raise ValueError("use pair expectations or a reference callback, not both")
    if len(set(protected_outputs)) != len(protected_outputs) or not set(
        protected_outputs
    ) <= set(model.output_names):
        raise ValueError("protected outputs must be unique model output names")
    for pair in pairs:
        unknown = set(pair.modules) - set(model.state_names)
        if unknown:
            raise ValueError(f"unknown donor modules: {sorted(unknown)}")
    routes: Optional[dict[str, tuple]] = None
    if read_slots is not None:
        if not isinstance(model, YatGraph):
            raise ValueError("read-slot donor patching requires YatGraph")
        specifications = {
            spec.name: spec for layer in model.layer_specs for spec in layer
        }
        if set(read_slots) != {name for pair in pairs for name in pair.modules}:
            raise ValueError("read-slot routes must cover exactly the paired modules")
        routes = {}
        for name, slots in read_slots.items():
            if (
                isinstance(slots, str)
                or not slots
                or len(set(slots)) != len(slots)
                or not set(slots) <= set(specifications[name].reads)
            ):
                raise ValueError("routes require unique receiving-module read slots")
            routes[name] = tuple(slots)
    normalized_edges: Optional[dict[str, dict[str, list[str]]]] = None
    if edge_routes is not None:
        if read_slots is not None:
            raise ValueError("choose whole-slot read routes or producer edge routes")
        if not isinstance(model, YatGraph):
            raise ValueError("donor edge routing requires YatGraph")
        specifications = {
            spec.name: spec for layer in model.layer_specs for spec in layer
        }
        layers = {
            spec.name: i for i, layer in enumerate(model.layer_specs) for spec in layer
        }
        if not isinstance(edge_routes, Mapping) or set(edge_routes) != {
            name for pair in pairs for name in pair.modules
        }:
            raise ValueError("edge routes must cover exactly paired receiving modules")
        normalized_edges = {}
        for receiver, slots in edge_routes.items():
            if (
                not isinstance(slots, Mapping)
                or not slots
                or set(slots) - set(specifications[receiver].reads)
            ):
                raise ValueError("edge routes must name receiving-module read slots")
            normalized_edges[receiver] = {}
            for slot, producers in slots.items():
                if (
                    not isinstance(producers, (list, tuple))
                    or not producers
                    or any(not isinstance(p, str) for p in producers)
                    or len(set(producers)) != len(producers)
                ):
                    raise ValueError("edge routes require unique producer name lists")
                for producer in producers:
                    if (
                        producer not in specifications
                        or layers[producer] >= layers[receiver]
                        or slot not in specifications[producer].writes
                    ):
                        raise ValueError(
                            "edge donor producer must execute earlier and write the receiving slot"
                        )
                normalized_edges[receiver][slot] = list(producers)
    parameter = next(model.parameters())
    ids = tuple(
        dict.fromkeys(sid for pair in pairs for sid in (pair.base_id, pair.donor_id))
    )
    inputs = torch.tensor(
        [dataset.sample(sid).inputs for sid in ids],
        device=parameter.device,
        dtype=parameter.dtype,
    )
    # Freeze a complete snapshot of the actual population touched by this study.
    snapshot = collect_research_data(
        model,
        inputs,
        sample_ids=ids,
        derivatives=False,
        metadata={"dataset_sha256": dataset.sha256},
    )
    index = {sid: i for i, sid in enumerate(ids)}
    output_index = {name: i for i, name in enumerate(model.output_names)}
    rows = []
    with torch.no_grad():
        population_outputs, population_trace = model.forward_with_trace(inputs)
        for pair in pairs:
            base, donor = dataset.sample(pair.base_id), dataset.sample(pair.donor_id)
            b, d = index[pair.base_id], index[pair.donor_id]
            controls = {
                name: Intervention(
                    replacement=population_trace[name][d : d + 1].clone()
                )
                for name in pair.modules
            }
            read_patches: dict[str, dict[str, torch.Tensor]] = {}
            donor_edges = {}
            if normalized_edges is not None:
                controls = {}
                donor_edges = {
                    receiver: {
                        slot: {
                            producer: population_trace[producer][
                                d : d + 1, specifications[producer].writes.index(slot)
                            ].clone()
                            for producer in producers
                        }
                        for slot, producers in normalized_edges[receiver].items()
                    }
                    for receiver in pair.modules
                }
                edited, trace = model.forward_with_trace(
                    inputs[b : b + 1], edge_patches=donor_edges
                )
            elif routes is not None:
                controls = {}
                for name in pair.modules:
                    read_patches[name] = {
                        slot: population_trace[f"{name}.input"][
                            d : d + 1, specifications[name].reads.index(slot)
                        ].clone()
                        for slot in routes[name]
                    }
                edited, trace = model.forward_with_trace(
                    inputs[b : b + 1], read_patches=read_patches
                )
            else:
                edited, trace = model.forward_with_trace(inputs[b : b + 1], controls)
            expected = (
                dict(reference(base, donor, tuple(pair.modules)))
                if reference
                else dict(pair.expected)
            )
            # Reuse the contract's finite-value validation for reference outputs.
            dataset.validate_pairs(
                [
                    DonorPair(
                        pair.pair_id,
                        pair.base_id,
                        pair.donor_id,
                        pair.modules,
                        expected,
                    )
                ],
                allow_cross_split=allow_cross_split,
            )
            if not set(expected) <= set(output_index):
                raise ValueError(
                    "reference expectations must name model output coordinates"
                )
            baseline = population_outputs[b]
            error = {
                name: edited[0, output_index[name]] - value
                for name, value in expected.items()
            }
            rows.append(
                {
                    "pair": asdict(pair),
                    "base_split": base.split,
                    "donor_split": donor.split,
                    "self_donor": pair.base_id == pair.donor_id,
                    "expected": expected,
                    "edited_outputs": edited[0],
                    "signed_reference_error": error,
                    "absolute_reference_error": {k: v.abs() for k, v in error.items()},
                    "protected_delta": {
                        name: edited[0, output_index[name]]
                        - baseline[output_index[name]]
                        for name in protected_outputs
                    },
                    "donor_reads": read_patches,
                    "donor_writes": {
                        name: control.replacement for name, control in controls.items()
                    },
                    "edited_trace": trace,
                    **(
                        {"donor_edges": donor_edges}
                        if normalized_edges is not None
                        else {}
                    ),
                }
            )
    return _json_value(
        {
            "schema": "nmn.donor-study.v1",
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(__file__).parent.parent / "research" / "datasets.py",
                )
            },
            "dataset": dataset.to_dict(),
            "dataset_sha256": dataset.sha256,
            "model_snapshot": snapshot,
            "protocol": {
                "reference_id": reference_id,
                "expected_origin": (
                    "reference-callback" if reference else "pair-supplied"
                ),
                "protected_outputs": list(protected_outputs),
                "allow_cross_split": allow_cross_split,
                "match_semantics": list(match_semantics),
                "donor_execution": (
                    "unchanged-model; producer-specific residual edge replacement"
                    if normalized_edges is not None
                    else (
                        "unchanged-model; receiving-module read-slot replacement"
                        if routes is not None
                        else "unchanged-model; full module-write replacement"
                    )
                ),
                "read_slots": routes,
                **(
                    {"edge_routes": normalized_edges}
                    if normalized_edges is not None
                    else {}
                ),
            },
            "rows": rows,
            "limitations": [
                "Supplied semantics are not learned or identified by this study.",
                "Split/group checks do not establish independence or population coverage.",
                "Effects are finite numerical observations, not uniform protection bounds.",
            ],
        }
    )


def donor_study_from_plan(
    model, dataset, plan, *, protected_outputs=(), read_slots=None, edge_routes=None
):
    """Check and execute the selected prefix of a saved donor planning protocol.

    Budget-stopped plans remain valid partial selections; their status and full
    eligibility decisions are embedded without promoting coverage to complete.
    """
    import json

    from ..research.donor_planning import validate_donor_plan

    pairs = validate_donor_plan(plan, dataset)
    result = donor_study(
        model,
        dataset,
        pairs,
        protected_outputs=protected_outputs,
        match_semantics=plan["protocol"]["match_semantics"],
        read_slots=read_slots,
        edge_routes=edge_routes,
    )
    encoded = json.dumps(plan, sort_keys=True, allow_nan=False)
    result["selection_plan"] = json.loads(encoded)
    result["selection_plan_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return result
