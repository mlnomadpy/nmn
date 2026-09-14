"""Recompute supported native records, separating replay from file integrity."""

import hashlib
import json
import math

import torch

from ..research.datasets import DonorPair, ResearchDataset
from ..research.native_export import _check_identities
from ..research.semantics import TabulatedReference
from .alignment import alignment_study
from .benchmark import benchmark_models
from .coalitions import coalition_study
from .curvature import curvature_study
from .edges import edge_study
from .graph import YatGraph
from .interpretable import Intervention, YatExpansion
from .paths import gate_path
from .probes import probe_study
from .protection import protection_study
from .reduction import reduction_study
from .research import _json_value, collect_research_data, model_from_snapshot
from .response_space import response_space_study
from .sampled_contract import evaluate_sampled_contract
from .selection import select_edit
from .semantics import semantic_study
from .studies import donor_study, donor_study_from_plan
from .suffix import suffix_study


def replay_native_record(record, *, atol=1e-10, rtol=1e-8):
    """Recompute saved measurements on CPU without training or executable loaders.

    Supports native observations, donor studies (saved reference expectations),
    classification protection, coalition, gate-path and kernel diagnostic studies.
    Match means agreement with
    stored measurements within explicit tolerances, not a scientific certificate.
    Timings/source versions are retained in the new execution but not compared.
    """
    for value in (atol, rtol):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("tolerances must be finite nonnegative numbers")
    supported = {
        "nmn.sampled-contract-evidence.v1",
        "nmn.curvature-study.v1",
        "nmn.alignment-study.v1",
        "nmn.preimage-execution.v1",
        "nmn.native-research.v1",
        "nmn.response-space.v1",
        "nmn.native-benchmark.v1",
        "nmn.edit-selection.v1",
        "nmn.semantic-study.v1",
        "nmn.suffix-study.v1",
        "nmn.reduction-study.v1",
        "nmn.edge-study.v1",
        "nmn.probe-study.v1",
        "nmn.gate-path-study.v1",
        "nmn.kernel-diagnostics.v1",
        "nmn.donor-study.v1",
        "nmn.protection-study.v1",
        "nmn.coalition-study.v1",
    }
    schema = record.get("schema")
    if schema not in supported:
        raise ValueError("unsupported native replay schema")
    _check_identities(record)
    if schema == "nmn.native-benchmark.v1":
        if len(record["methods"]) < 2 or any(
            row["status"] != "measured" or "model_snapshot" not in row
            for row in record["methods"].values()
        ):
            raise ValueError(
                "benchmark replay requires at least two measured methods with snapshots; failed historical methods cannot be reconstructed"
            )
        snapshot = next(iter(record["methods"].values()))["model_snapshot"]
    else:
        snapshot = (
            record if schema == "nmn.native-research.v1" else record["model_snapshot"]
        )
    dtype_name = snapshot["runtime"]["dtype"]
    dtypes = {"torch.float64": torch.float64, "torch.float32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError("native replay supports saved float32/float64 execution")
    model = model_from_snapshot(snapshot, device="cpu", dtype=dtypes[dtype_name])

    def edits(controls):
        return {
            name: {
                module: Intervention(**control) for module, control in mapping.items()
            }
            for name, mapping in controls.items()
        }

    if schema == "nmn.native-benchmark.v1":
        dataset = ResearchDataset.from_dict(record["dataset"])
        if dataset.sha256 != record["dataset_sha256"]:
            raise ValueError("dataset content hash mismatch")
        contract = record["contract"]
        models = {
            name: model_from_snapshot(
                record["methods"][name]["model_snapshot"],
                dtype=dtypes[
                    record["methods"][name]["model_snapshot"]["runtime"]["dtype"]
                ],
            )
            for name in record["method_order"]
        }
        if set(models) != set(record["methods"]) or len(record["method_order"]) != len(
            models
        ):
            raise ValueError("method order must cover exactly the recorded methods")
        actual = benchmark_models(
            models,
            dataset,
            edits=edits(contract["edits"]),
            expected_outputs=contract["expected_outputs"],
            protected_outputs=contract["protected_outputs"],
            split=contract["split"],
            repeats=1,
            warmup=0,
        )
        keys = ["method_order", "sample_ids", "methods"]
    elif schema == "nmn.native-research.v1":
        actual = collect_research_data(
            model,
            torch.tensor(snapshot["inputs"], dtype=dtypes[dtype_name]),
            sample_ids=snapshot["sample_ids"],
            edits=edits(snapshot["controls"]),
            metadata=snapshot.get("metadata"),
            derivatives="input_jacobian" in snapshot or "gate_derivatives" in snapshot,
        )
        keys = ["observations", "geometry"] + [
            k for k in ("input_jacobian", "gate_derivatives") if k in record
        ]
    elif schema in ("nmn.gate-path-study.v1", "nmn.kernel-diagnostics.v1"):
        inputs = torch.tensor(snapshot["inputs"], dtype=dtypes[dtype_name])
        ids = snapshot["sample_ids"]
        if "dataset" in record:
            dataset = ResearchDataset.from_dict(record["dataset"])
            expected_hash = record.get(
                "dataset_sha256", snapshot.get("metadata", {}).get("dataset_sha256")
            )
            if expected_hash is not None and dataset.sha256 != expected_hash:
                raise ValueError("dataset content hash mismatch")
            if not torch.equal(
                torch.tensor(
                    [dataset.sample(s).inputs for s in ids], dtype=dtypes[dtype_name]
                ),
                inputs,
            ):
                raise ValueError("snapshot inputs differ from declared dataset")
        actual = {
            "schema": schema,
            "model_snapshot": collect_research_data(
                model, inputs, sample_ids=ids, derivatives=False
            ),
        }
        if "dataset" in record:
            actual["dataset"] = record["dataset"]
        if schema == "nmn.gate-path-study.v1":
            path = record["path"]
            actual["path"] = _json_value(
                gate_path(
                    model,
                    inputs,
                    path["start"],
                    path["end"],
                    steps=path["cost"]["intervals"],
                )
            )
            keys = ["path"]
        else:
            from .diagnostics import diagnose_layer, sensor_diagnostics

            name = record["module"]
            if name not in model.state_names:
                raise ValueError("unknown diagnostic module")
            block = (
                model.blocks[name]
                if isinstance(model, YatGraph)
                else getattr(model, name)
            )
            if not isinstance(block, YatExpansion):
                raise ValueError("kernel replay requires a YatExpansion module")
            with torch.no_grad():
                _, trace = model.forward_with_trace(inputs)
                points = trace[f"{name}.input"]
            actual.update(
                {
                    "module": name,
                    "sample_ids": ids,
                    "layer": _json_value(diagnose_layer(block, points)),
                    "sensors": _json_value(
                        sensor_diagnostics(
                            block.centers,
                            points,
                            epsilon=block.kernel.epsilon,
                            noise_radius=record["sensors"]["noise_radius"],
                        )
                    ),
                }
            )
            keys = ["layer", "sensors"]
    else:
        dataset = ResearchDataset.from_dict(record["dataset"])
        if dataset.sha256 != record["dataset_sha256"]:
            raise ValueError("dataset content hash mismatch")
        protocol = record.get("protocol", {})
        if schema == "nmn.response-space.v1":
            actual = response_space_study(
                model,
                dataset,
                edits=edits(snapshot["controls"]),
                rank=protocol["rank"],
                fit_split=protocol["fit_split"],
                evaluation_split=protocol["evaluation_split"],
            )
            keys = [
                "protocol",
                "singular_values",
                "numerical_rank",
                "numerical_rank_threshold",
                "fit",
                "evaluation",
                "cost",
            ]
        elif schema == "nmn.edit-selection.v1":
            actual = select_edit(
                model,
                dataset,
                candidates=record["candidates"],
                targets=record["targets"],
                **protocol,
            )
            keys = [
                "status",
                "selection",
                "selected",
                "unexecuted",
                "validation",
                "ledger",
            ]
        elif schema == "nmn.sampled-contract-evidence.v1":
            actual = evaluate_sampled_contract(model, dataset, record["contract"])
            keys = [
                "status",
                "assessment",
                "contract_sha256",
                "sample_ids",
                "rows",
                "violations",
                "coverage",
            ]
        elif schema == "nmn.curvature-study.v1":
            curvature_protocol = dict(record["protocol"])
            if len(set(record["direction_names"])) != len(
                record["direction_names"]
            ) or set(record["direction_names"]) != set(
                curvature_protocol["directions"]
            ):
                raise ValueError(
                    "curvature direction names must match the saved protocol"
                )
            curvature_protocol["directions"] = {
                name: curvature_protocol["directions"][name]
                for name in record["direction_names"]
            }
            actual = curvature_study(model, dataset, **curvature_protocol)
            keys = [
                "sample_ids",
                "module_names",
                "output_names",
                "direction_names",
                "observations",
                "axes",
            ]
        elif schema == "nmn.alignment-study.v1":
            actual = alignment_study(
                model,
                dataset,
                reference=TabulatedReference(record["reference"]),
                **{
                    k: record["protocol"][k]
                    for k in (
                        "module_pool",
                        "max_candidates",
                        "provenance",
                        "selection_split",
                        "evaluation_split",
                        "tolerance",
                    )
                },
            )
            keys = [
                "status",
                "coverage",
                "candidates",
                "selected",
                "exact_ties",
                "evaluation",
            ]
        elif schema == "nmn.semantic-study.v1":
            actual = semantic_study(
                model,
                dataset,
                reference=TabulatedReference(record["reference"]),
                correspondence=record["correspondence"],
                tolerance=record["tolerance"],
            )
            keys = ["status", "baseline", "cases", "coverage"]
        elif schema == "nmn.donor-study.v1":
            keys = ["rows"]
            if "selection_plan" in record:
                actual = donor_study_from_plan(
                    model,
                    dataset,
                    record["selection_plan"],
                    protected_outputs=protocol["protected_outputs"],
                    read_slots=protocol.get("read_slots"),
                    edge_routes=protocol.get("edge_routes"),
                )
                keys += ["protocol", "selection_plan", "selection_plan_sha256"]
            else:
                pairs = [
                    DonorPair(**{**row["pair"], "expected": row["expected"]})
                    for row in record["rows"]
                ]
                actual = donor_study(
                    model,
                    dataset,
                    pairs,
                    protected_outputs=protocol["protected_outputs"],
                    allow_cross_split=protocol["allow_cross_split"],
                    match_semantics=protocol["match_semantics"],
                    read_slots=protocol.get("read_slots"),
                    edge_routes=protocol.get("edge_routes"),
                )
                # Reuse supplied expectations, never execute saved callback names.
                for current, saved in zip(actual["rows"], record["rows"]):
                    current["pair"]["expected"] = saved["pair"]["expected"]
                    if "donor_reads" not in saved:
                        current.pop("donor_reads", None)
        elif schema == "nmn.probe-study.v1":
            actual = probe_study(
                model,
                dataset,
                feature=protocol["feature"],
                feature_map=protocol.get("feature_map", "linear"),
                max_expanded_features=protocol.get("max_expanded_features", 1024),
                labels=record["labels"],
                classes=record["classes"],
                provenance=protocol["provenance"],
                ridge=protocol["ridge"],
                refit_edits=protocol.get("refit_edits", False),
                fit_split=protocol["fit_split"],
                evaluation_split=protocol["evaluation_split"],
                edits=edits(record["evaluation_snapshot"]["controls"]),
            )
            keys = [
                "status",
                "labels",
                "classes",
                "protocol",
                "probe",
                "fit",
                "evaluation",
                "cost",
            ]
        elif schema == "nmn.preimage-execution.v1":
            from .preimage import execute_preimage_study

            actual = execute_preimage_study(record["proposal"])
            keys = [
                "proposal_sha256",
                "sample_ids",
                "module",
                "read_slots",
                "read_patches",
                "baseline_outputs",
                "baseline_trace",
                "outputs",
                "output_delta",
                "trace",
                "executed_features",
                "feature_residuals",
                "protocol",
            ]
        elif schema == "nmn.edge-study.v1":
            actual = edge_study(
                model,
                dataset,
                patches=record["patches"],
                provenance=protocol["provenance"],
                split=protocol["split"],
            )
            keys = [
                "status",
                "sample_ids",
                "patches",
                "baseline_outputs",
                "protocol",
                "results",
                "cost",
            ]
        elif schema == "nmn.reduction-study.v1":
            actual = reduction_study(
                model,
                dataset,
                start_layer=protocol["start_layer"],
                maps=record["maps"],
                provenance=protocol["provenance"],
                split=protocol["split"],
            )
            keys = ["status", "sample_ids", "maps", "protocol", "observations", "cost"]
        elif schema == "nmn.suffix-study.v1":
            actual = suffix_study(
                model,
                dataset,
                start_layer=protocol["start_layer"],
                states={
                    name: dict(zip(record["sample_ids"], row["state"]))
                    for name, row in record["variants"].items()
                },
                provenance=protocol["provenance"],
                split=protocol["split"],
            )
            keys = [
                "sample_ids",
                "original_state",
                "baseline_outputs",
                "baseline_suffix_outputs",
                "baseline_reconstruction_error",
                "baseline_suffix_trace",
                "variants",
            ]
        elif schema == "nmn.protection-study.v1":
            actual = protection_study(
                model,
                dataset,
                edits=edits(snapshot["controls"]),
                tasks=record["tasks"],
                provenance=protocol["provenance"],
                split=protocol["split"],
                strata=protocol["strata"],
            )
            keys = ["sample_ids", "results"]
        else:
            actual = coalition_study(
                model,
                dataset,
                modules=protocol["modules"],
                max_evaluations=protocol["max_evaluations"],
                background=protocol["background_gates"],
                split=protocol["split"],
            )
            keys = [
                "status",
                "coverage",
                "masks",
                "outputs",
                "delta_from_background",
                "subset_coefficients",
                "reconstruction_error",
            ]

    def segment(value):
        return str(value).replace("~", "~0").replace("/", "~1")

    ignored_paths = (
        {"/path/source_sha256", "/path/cost/seconds"}
        if schema == "nmn.gate-path-study.v1"
        else set()
    )
    if schema == "nmn.response-space.v1":
        ignored_paths.update({"/fit/coordinates", "/evaluation/coordinates"})
    if schema == "nmn.native-benchmark.v1":
        for name in record["methods"]:
            prefix = "/methods/" + segment(name)
            ignored_paths.update(
                prefix + "/" + field
                for field in (
                    "model_snapshot",
                    "baseline_cost",
                    "instrumentation_seconds",
                    "total_seconds",
                )
            )
            ignored_paths.update(
                prefix + "/edits/" + segment(name) + "/cost"
                for name in contract["edits"]
            )
    mismatches = []
    checked = 0
    maximum_error = 0.0

    def compare(saved, current, path):
        nonlocal checked, maximum_error
        if path in ignored_paths:
            return
        checked += 1
        if isinstance(saved, dict) and isinstance(current, dict):
            if set(saved) != set(current):
                mismatches.append({"path": path, "reason": "mapping keys differ"})
            for key in sorted(set(saved) & set(current)):
                compare(saved[key], current[key], path + "/" + segment(key))
        elif isinstance(saved, list) and isinstance(current, list):
            if len(saved) != len(current):
                mismatches.append({"path": path, "reason": "array lengths differ"})
            for index, (left, right) in enumerate(zip(saved, current)):
                compare(left, right, path + "/" + str(index))
        elif type(saved) in (int, float) and type(current) in (int, float):
            error = abs(saved - current)
            if math.isfinite(error):
                maximum_error = max(maximum_error, error)
            if (
                not math.isfinite(saved)
                or not math.isfinite(current)
                or not error <= atol + rtol * abs(saved)
            ):
                mismatches.append(
                    {
                        "path": path,
                        "reason": "numeric mismatch",
                        "saved": saved,
                        "replayed": current,
                    }
                )
        elif isinstance(saved, str) and saved in ("nan", "inf", "-inf"):
            mismatches.append({"path": path, "reason": "nonfinite saved measurement"})
        elif type(saved) is not type(current) or saved != current:
            mismatches.append(
                {
                    "path": path,
                    "reason": "value mismatch",
                    "saved": saved,
                    "replayed": current,
                }
            )

    for key in keys:
        compare(record[key], actual[key], "/" + key)
    # Always compare the unchanged population outputs, too, for composite studies.
    if schema not in ("nmn.native-research.v1", "nmn.native-benchmark.v1"):
        for field in ("sample_ids", "inputs"):
            compare(
                snapshot[field],
                actual["model_snapshot"][field],
                "/model_snapshot/" + field,
            )
        compare(
            snapshot["observations"]["baseline"],
            actual["model_snapshot"]["observations"]["baseline"],
            "/model_snapshot/observations/baseline",
        )
    if schema == "nmn.response-space.v1":
        saved_basis = torch.tensor(record["basis"], dtype=torch.float64)
        actual_basis = torch.tensor(actual["basis"], dtype=torch.float64)
        compare(
            (saved_basis @ saved_basis.T).tolist(),
            (actual_basis @ actual_basis.T).tolist(),
            "/subspace_projector",
        )
        for field in ("sample_ids", "inputs", "observations"):
            compare(
                record["evaluation_snapshot"][field],
                actual["evaluation_snapshot"][field],
                "/evaluation_snapshot/" + field,
            )
    if schema == "nmn.sampled-contract-evidence.v1":
        compare(
            record["model_snapshot"]["observations"],
            actual["model_snapshot"]["observations"],
            "/model_snapshot/observations",
        )
    if schema == "nmn.alignment-study.v1":

        def measurements(value):
            return {
                "baseline": value["baseline"],
                "cases": value["cases"],
                "donor_rows": value["donor_execution"]["rows"],
                "observations": value["model_snapshot"]["observations"],
            }

        compare(
            [measurements(v) for v in record["selection_executions"]],
            [measurements(v) for v in actual["selection_executions"]],
            "/selection_executions/measurements",
        )
        compare(
            measurements(record["evaluation_execution"]),
            measurements(actual["evaluation_execution"]),
            "/evaluation_execution/measurements",
        )
    if schema == "nmn.probe-study.v1":
        compare(
            snapshot["observations"],
            actual["model_snapshot"]["observations"],
            "/model_snapshot/observations",
        )
        for field in ("model_sha256", "sample_ids", "inputs", "observations"):
            compare(
                record["evaluation_snapshot"][field],
                actual["evaluation_snapshot"][field],
                "/evaluation_snapshot/" + field,
            )
    if schema == "nmn.edit-selection.v1":
        saved_validation = record["validation_snapshot"]
        current_validation = actual["validation_snapshot"]
        if saved_validation is None or current_validation is None:
            compare(saved_validation, current_validation, "/validation_snapshot")
        else:
            for field in ("model_sha256", "sample_ids", "inputs", "observations"):
                compare(
                    saved_validation[field],
                    current_validation[field],
                    "/validation_snapshot/" + field,
                )
    return {
        "schema": "nmn.native-replay.v1",
        "status": "matched" if not mismatches else "mismatch",
        "source_schema": schema,
        "record_sha256": hashlib.sha256(
            json.dumps(record, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "model_sha256": (
            None if schema == "nmn.native-benchmark.v1" else snapshot["model_sha256"]
        ),
        "model_identities": (
            {
                name: row["model_snapshot"]["model_sha256"]
                for name, row in record["methods"].items()
            }
            if schema == "nmn.native-benchmark.v1"
            else {}
        ),
        "tolerances": {
            "atol": atol,
            "rtol": rtol,
            "rule": "abs(saved-replayed) <= atol + rtol*abs(saved)",
        },
        "compared_fields": keys
        + (
            ["model_snapshot/observations"]
            if schema == "nmn.sampled-contract-evidence.v1"
            else []
        )
        + (
            ["selection_executions/measurements", "evaluation_execution/measurements"]
            if schema == "nmn.alignment-study.v1"
            else []
        )
        + (
            ["validation_snapshot/{model_sha256,sample_ids,inputs,observations}"]
            if schema == "nmn.edit-selection.v1"
            else (
                ["evaluation_snapshot/{model_sha256,sample_ids,inputs,observations}"]
                if schema == "nmn.probe-study.v1"
                else []
            )
        ),
        "additional_comparisons": (
            [
                "subspace_projector",
                "evaluation_snapshot/{sample_ids,inputs,observations}",
            ]
            if schema == "nmn.response-space.v1"
            else []
        ),
        "ignored_paths": sorted(ignored_paths),
        "checked_nodes": checked,
        "maximum_absolute_error": maximum_error,
        "mismatches": mismatches,
        "execution": actual,
        "limitations": [
            "Numerical agreement with saved observations is not scientific validation or a certificate.",
            "CPU replay may differ from original hardware or runtime versions.",
            "Benchmark replay uses one forward per condition; timings and historical failed methods are not reproduced.",
            "Response-space replay compares projection operators, not basis signs or coordinate orientation.",
            "Probe replay refits the saved affine ridge classifier on its declared fit population, then evaluates it; this is not a neural-network training run.",
            "Donor reference expectations are reused as supplied data; reference callbacks are not rerun.",
        ],
    }
