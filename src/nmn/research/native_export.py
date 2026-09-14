"""Backend-independent Obsidian notes for native numerical research records."""

import hashlib
import html
import json
import math
import shutil
from pathlib import Path

SCHEMAS = {
    "nmn.preimage-study.v1": "Dataset-linked native preimage proposals",
    "nmn.preimage-search.v1": "Bounded finite-kernel preimage search",
    "nmn.nnx-suffix-study.v1": "NNX downstream boundary-state study",
    "nmn.nnx-model.v1": "NNX native model definition",
    "nmn.nnx-research.v1": "NNX native kernel observations",
    "nmn.gate-search.v1": "Native gate proposal search and frozen selection",
    "nmn.donor-plan.v1": "Declared donor selection plan",
    "nmn.edge-study.v1": "Producer-specific residual edge study",
    "nmn.probe-study.v1": "Frozen internal-state classification probe",
    "nmn.fitted-reduction.v1": "Fitted state summary and held-out residuals",
    "nmn.reduction-study.v1": "State summaries and reduced dynamics",
    "nmn.rational-difference.v1": "Rational intervention-difference enclosure",
    "nmn.interval-certificate.v1": "Rational box range certificate",
    "nmn.interval-check.v1": "Rational box certificate check",
    "nmn.rational-enclosure.v1": "Rational real-function enclosure",
    "nmn.response-space.v1": "Finite edit-response subspace",
    "nmn.edit-selection.v1": "Frozen native edit selection",
    "nmn.semantic-study.v1": "Supplied semantic correspondence study",
    "nmn.suffix-study.v1": "Native suffix-state study",
    "nmn.native-replay.v1": "Native numerical replay",
    "nmn.coalition-study.v1": "Native coalition study",
    "nmn.protection-study.v1": "Native classification protection study",
    "nmn.native-model.v1": "Native model",
    "nmn.native-research.v1": "Native model observations",
    "nmn.donor-study.v1": "Native donor study",
    "nmn.gate-path-study.v1": "Joint gate-path comparison",
    "nmn.kernel-diagnostics.v1": "Kernel and sensor diagnostics",
    "nmn.native-benchmark.v1": "Native model comparison",
    "nmn.native-training.v1": "Native training record",
}


def _text(value):
    return (
        html.escape(str(value))
        .replace("|", "&#124;")
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _maximum(value):
    values = []

    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
        ):
            values.append(abs(item))
        else:
            raise ValueError("not a finite numeric measurement")

    try:
        visit(value)
    except ValueError:
        return "unavailable/nonfinite"
    return f"{max(values):.8g}" if values else "not measured"


def _table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(_text(cell) for cell in row) + " |" for row in rows],
        ]
    )


def _check_identities(value):
    if isinstance(value, dict):
        if value.get("schema") == "nmn.preimage-search.v1":
            digest = hashlib.sha256(
                json.dumps(
                    value["module_snapshot"], sort_keys=True, allow_nan=False
                ).encode()
            ).hexdigest()
            if value.get("module_sha256") != digest:
                raise ValueError("preimage module content identity mismatch")
        if value.get("schema") in (
            "nmn.native-model.v1",
            "nmn.native-research.v1",
            "nmn.nnx-research.v1",
            "nmn.nnx-model.v1",
        ):
            identity = {key: value[key] for key in ("configuration", "parameters")}
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if value.get("model_sha256") != digest:
                raise ValueError("embedded model content identity mismatch")
        for child in value.values():
            _check_identities(child)
    elif isinstance(value, list):
        for child in value:
            _check_identities(child)


def render_native_note(record):
    """Summarize recorded observations without rerunning or upgrading assurance."""
    schema = record.get("schema")
    if schema not in SCHEMAS:
        raise ValueError("unsupported native record schema")
    _check_identities(record)
    lines = [
        "---",
        "type: experiment-report",
        "status: recorded-observations",
        "tags: [research/nmn, research/intervention]",
        "---",
        f"# {SCHEMAS[schema]}",
        "",
        "[Complete data](data.json) · [File hashes](manifest.json)",
        "",
        f"Record schema: `{schema}`.",
        "",
        "This note summarizes saved data. Export does not execute the model, certify a claim, "
        "or establish semantic meaning. Model identity checks cover stored configuration and parameters.",
        "",
    ]
    if "selection_plan" in record:
        lines += [
            "## Executed donor selection plan",
            "",
            f"Plan identity: `{_text(record['selection_plan_sha256'])}`.",
            "",
            f"Selection status: {_text(record['selection_plan']['status'])}; coverage: {_text(record['selection_plan']['coverage'])}.",
            "",
            "The complete plan is embedded in data.json. Export does not recheck or execute it.",
            "",
        ]
    snapshot = (
        record
        if schema in ("nmn.native-model.v1", "nmn.native-research.v1")
        else record.get("model_snapshot")
    )
    if snapshot:
        lines += [
            f"Model identity: `{_text(snapshot['model_sha256'])}`.",
            "",
            f"Architecture: {_text(snapshot['configuration'].get('class', snapshot['configuration'].get('architecture', 'unknown')))}.",
            "",
        ]
        if "sample_ids" in snapshot:
            lines += [f"Recorded samples: {len(snapshot['sample_ids'])}.", ""]
    if record.get("dataset"):
        ds = record["dataset"]
        lines += [
            f"Dataset: {_text(ds.get('name', 'not named'))}.",
            "",
            f"Data/semantic provenance: {_text(ds.get('provenance', 'not recorded'))}.",
            "",
        ]
    if schema == "nmn.preimage-study.v1":
        search = record["search"]
        lines += [
            "## Dataset-linked preimage proposals",
            "",
            f"Module: {_text(record['module'])}; selected samples: {len(record['sample_ids'])}.",
            "",
            f"Status: {_text(search['status'])}; selected step: {_text(search['selected_step'])}.",
            "",
            f"Maximum absolute feature residual: {_maximum(search['feature_residuals'])}.",
            "",
            "Proposed module inputs are keyed by sample ID. This study does not install them as native edits.",
            "",
        ]
    elif schema == "nmn.preimage-search.v1":
        lines += [
            "## Native preimage search",
            "",
            f"Search status: {_text(record['status'])}; selected step: {_text(record['selected_step'])}.",
            "",
            f"Maximum absolute feature residual: {_maximum(record['feature_residuals'])}.",
            "",
            "Inputs were optimized per example against a fixed finite feature bank.",
            "Residuals do not prove preimage existence, impossibility, erasure, or downstream protection.",
            "",
        ]
    elif schema == "nmn.nnx-model.v1":
        lines += [
            "## NNX model definition",
            "",
            f"Model identity: `{_text(record['model_sha256'])}`.",
            "",
            f"Configuration: {_text(record['configuration'])}.",
            "",
            "Parameters are stored in data.json. This definition contains no measured observations.",
            "",
        ]
    elif schema == "nmn.nnx-research.v1":
        lines += [
            "## NNX observations",
            "",
            f"Samples: {len(record.get('sample_ids', []))}; split: {_text(record.get('split'))}.",
            "Direct squared-distance arithmetic; signed center contributions, local Gram matrices,",
            "executed edits and optional input/gate derivatives are retained in data.json.",
            "This adapter supports the strict three-neuron model only. It does not supply",
            "suffix replay, external-model conversion, or interval/statistical certificates.",
            "",
        ]
    elif schema == "nmn.rational-difference.v1":
        lines += [
            "## Edited minus reference outputs",
            "",
            f"Changed modules: {_text(record['changed_modules'])}.",
            "",
            f"Structurally zero differences: {_text(record['structural_zero_outputs'])}.",
            "",
            _table(
                ["Output", "Difference lower", "Difference upper"],
                [[name, *bounds] for name, bounds in record["output_bounds"].items()],
            ),
            "",
            _text(record["assurance"]),
            "",
        ]
    elif schema == "nmn.interval-certificate.v1":
        lines += [
            "## Saved range-contract outcome",
            "",
            f"Saved status: {_text(record['status'])}.",
            f"Quantity: {_text(record.get('quantity', 'output-difference' if 'reference_controls' in record['contract'] else 'output'))}.",
            "",
            _text(record["assurance"]),
            "",
            f"Budget: {_text(record['budget'])}.",
            "",
            "Use native check-box to verify the complete saved partition, enclosures and any claimed point witness. Export alone does not check this certificate.",
            "",
        ]
    elif schema == "nmn.interval-check.v1":
        lines += [
            "## Checked partition outcome",
            "",
            f"Outcome: {_text(record['outcome'])}.",
            f"Quantity: {_text(record.get('quantity', 'output'))}.",
            "",
            f"Leaf/node counts: {_text(record['counts'])}.",
            "",
            f"Certificate identity: {_text(record['certificate_sha256'])}.",
            "",
            _text(record["assurance"]),
            "",
        ]
    elif schema == "nmn.rational-enclosure.v1":
        lines += [
            "## Exact rational enclosure",
            "",
            _text(record["assurance"]),
            "",
            f"Input box: {_text(record['input_box'])}.",
            "",
            _table(
                ["Output", "Lower", "Upper"],
                [[name, *bounds] for name, bounds in record["output_bounds"].items()],
            ),
            "",
            "These are real-function bounds, not a contract verdict or floating-runtime certificate.",
            "",
        ]
    elif schema == "nmn.response-space.v1":
        lines += [
            "## Observed response subspace",
            "",
            f"Protocol: {_text(record['protocol'])}.",
            "",
            f"Numerical rank: {_text(record['numerical_rank'])}.",
            "",
            _table(
                ["Population", "Residual norm", "Relative residual"],
                [
                    [
                        name,
                        record[name]["residual_norm"],
                        record[name]["relative_residual"],
                    ]
                    for name in ("fit", "evaluation")
                ],
            ),
            "",
            "Evaluation uses full measured responses; this is not an unseen-edit prediction or uniform rank certificate.",
            "",
        ]
    elif schema == "nmn.edit-selection.v1":
        lines += [
            "## Frozen edit selection",
            "",
            f"Selected: {_text(record['selected'])}. Unexecuted candidates: {_text(record['unexecuted'])}.",
            "",
            _table(
                [
                    "Candidate",
                    "Status",
                    "Selection MSE",
                    "Selection protection satisfied",
                ],
                [
                    [
                        name,
                        row["status"],
                        row.get("target_mse"),
                        row.get("protection_satisfied"),
                    ]
                    for name, row in record["selection"].items()
                ],
            ),
            "",
            f"Validation: {_text({k: v for k, v in (record['validation'] or {}).items() if k not in ('trace', 'outputs')})}.",
            "",
            "The frozen winner is never changed using validation outcomes. This is finite empirical selection, not a statistical certificate.",
            "",
        ]
    elif schema == "nmn.semantic-study.v1":
        lines += [
            "## Declared semantic correspondence",
            "",
            f"Origin: {_text(record['correspondence']['origin'])}. Status: {_text(record['status'])}.",
            "",
            f"Coverage: {_text(record['coverage'])}.",
            "",
            _table(
                ["Case", "Variables", "Native modules", "Agrees"],
                [
                    [
                        row["case_id"],
                        row["variables"],
                        row["native_modules"],
                        row["agrees"],
                    ]
                    for row in record["cases"]
                ],
            ),
            "",
            "Agreement checks a supplied table and does not identify a unique mechanism.",
            "",
        ]
    elif schema == "nmn.gate-search.v1":
        lines += [
            "## Gate proposals and measured selection",
            "",
            f"Generation: {_text(record['status'])}; proposals: {len(record['proposals'])}.",
            "",
            f"Protocol: {_text(record['protocol'])}.",
            "",
            f"Selection outcome: {_text(record['selection']['status'])}; selected: {_text(record['selection']['selected'])}.",
            "",
            "All proposals and the full selection/validation study are embedded in data.json. Proposal completion is not global feasibility or a validation-success claim.",
            "",
        ]
    elif schema == "nmn.donor-plan.v1":
        lines += [
            "## Donor selection coverage",
            "",
            f"Protocol: {_text(record['protocol'])}.",
            "",
            f"Coverage: {_text(record['coverage'])}.",
            "",
            "Every inspected decision and selected pair is retained. No model has been executed; uninspected candidates are not ineligible.",
            "",
        ]
    elif schema == "nmn.edge-study.v1":
        lines += [
            "## Residual edge effects",
            "",
            f"Protocol: {_text(record['protocol'])}.",
            "",
            _table(
                ["Condition", "Maximum absolute output change"],
                [
                    [name, _maximum(row["delta"])]
                    for name, row in record["results"].items()
                ],
            ),
            "",
            "Full per-example outputs, receiver corrections and traces remain in data.json. This does not establish semantic causality.",
            "",
        ]
    elif schema == "nmn.probe-study.v1":
        lines += [
            "## Fit-only classifier and held-out decoding",
            "",
            f"Protocol: {_text(record['protocol'])}.",
            "",
            _table(
                ["Population/condition", "Accuracy"],
                [
                    ["fit", record["fit"]["accuracy"]],
                    [
                        "evaluation baseline",
                        record["evaluation"]["baseline"]["accuracy"],
                    ],
                ]
                + [
                    ["evaluation frozen edit: " + name, row["accuracy"]]
                    for name, row in record["evaluation"]["edits"].items()
                ]
                + [
                    [
                        "evaluation refitted edit: " + name,
                        row["refitted"]["evaluation"]["accuracy"],
                    ]
                    for name, row in record["evaluation"]["edits"].items()
                    if "refitted" in row
                ],
            ),
            "",
            "All features, scores, confusion counts and individual predictions are saved. Probe failure does not establish concept erasure.",
            "",
        ]
    elif schema == "nmn.fitted-reduction.v1":
        lines += [
            "## Fit-only summary and frozen evaluation",
            "",
            f"Protocol: {_text(record['protocol'])}.",
            "",
            _table(
                [
                    "Population",
                    "Samples",
                    "State residual",
                    "Summary transition residual",
                    "Output prediction residual",
                ],
                [
                    [
                        name,
                        len(record[name]["sample_ids"]),
                        _maximum(
                            record[name]["observations"][
                                "state_reconstruction_residual"
                            ]
                        ),
                        _maximum(
                            record[name]["observations"]["summary_transition_residual"]
                        ),
                        _maximum(
                            record[name]["observations"]["prediction_output_residual"]
                        ),
                    ]
                    for name in ("fit", "evaluation")
                ],
            ),
            "",
            "Maps are fitted on the declared fit population and frozen before evaluation. Embedded records retain every sample and support frozen-map numerical replay; fitting itself is not replayed.",
            "",
        ]
    elif schema == "nmn.reduction-study.v1":
        lines += [
            "## Supplied summary maps and observed residuals",
            "",
            f"Protocol: {_text(record['protocol'])}.",
            "",
            _table(
                ["Residual", "Maximum absolute value"],
                [
                    [name, _maximum(value)]
                    for name, value in record["observations"].items()
                    if name.endswith("residual")
                ],
            ),
            "",
            "Per-example states, summaries, maps and downstream traces are retained in the data. These observations do not certify closure through depth.",
            "",
        ]
    elif schema in ("nmn.suffix-study.v1", "nmn.nnx-suffix-study.v1"):
        lines += [
            "## State boundary and downstream effects",
            "",
            f"Protocol: {_text(record['protocol'])}.",
            "",
            _table(
                [
                    "Variant",
                    "Maximum absolute state change",
                    "Maximum absolute output change",
                ],
                [
                    [name, _maximum(row["state_delta"]), _maximum(row["output_delta"])]
                    for name, row in record["variants"].items()
                ],
            ),
            "",
            "Full states and downstream traces remain in the data. Supplied states need not be reachable from model inputs.",
            "",
        ]
    elif schema == "nmn.native-replay.v1":
        lines += [
            "## Numerical replay",
            "",
            f"Saved replay status: {_text(record['status'])}.",
            "",
            f"Source record: `{_text(record['record_sha256'])}`.",
            "",
            f"Compared fields: {_text(record['compared_fields'])}.",
            "",
            f"Tolerances: {_text(record['tolerances'])}.",
            "",
            f"Mismatch count: {len(record['mismatches'])}.",
            "",
            "The complete data retains mismatch paths and the new execution. Agreement is not a scientific certificate.",
            "",
        ]
    elif schema == "nmn.coalition-study.v1":
        lines += [
            "## Coalition coverage",
            "",
            f"Saved status: {_text(record['status'])}; {_text(record['reason'])}.",
            "",
            f"Evaluated {record['coverage']['evaluated']} of {record['coverage']['total']} coalitions.",
            "",
            f"Module order: {_text(record['protocol']['modules'])}.",
            "",
            f"Background gates: {_text(record['protocol']['background_gates'])}.",
            "",
            "Per-example responses, subset coefficients (when complete), reconstruction residuals and costs remain in the data. Missing coalitions are not zero effects.",
            "",
        ]
    elif schema == "nmn.protection-study.v1":
        rows = []
        for task, edits in record["results"].items():
            for edit, result in edits.items():
                for group in result["strata"]:
                    m = group["metrics"]
                    rows.append(
                        [
                            task,
                            edit,
                            (
                                "all"
                                if group["field"] is None
                                else str(
                                    {k: group[k] for k in ("field", "value", "missing")}
                                )
                            ),
                            m["count"],
                            m["originally_correct_count"],
                            m["accuracy_after"],
                            m["conditional_damage_rate"],
                            m["disagreement_rate"],
                        ]
                    )
        lines += [
            "## Protection measurements",
            "",
            _table(
                [
                    "Task",
                    "Edit",
                    "Stratum",
                    "Count",
                    "Eligible",
                    "Accuracy after",
                    "Conditional damage",
                    "Disagreement",
                ],
                rows,
            ),
            "",
            "An undefined conditional damage rate means no originally correct examples; it is not zero damage.",
            "",
        ]
    elif schema == "nmn.native-research.v1":
        rows = [
            [name, _maximum(row["delta"])]
            for name, row in record["observations"]["edits"].items()
        ]
        lines += [
            "## Executed edits",
            "",
            _table(["Edit", "Maximum absolute output change"], rows),
            "",
            "Per-output/per-example values and raw/effective states remain in the complete data. "
            "An output change alone does not indicate target success or a protection violation.",
            "",
        ]
    elif schema == "nmn.donor-study.v1":
        donor_execution = record.get("protocol", {}).get(
            "donor_execution", "not declared"
        )
        lines += [
            f"Donor execution: {_text(donor_execution)}.",
            "",
        ]
        rows = [
            [
                row["pair"]["pair_id"],
                row["pair"]["base_id"],
                row["pair"]["donor_id"],
                _maximum(row["absolute_reference_error"]),
                _maximum(row["protected_delta"]),
            ]
            for row in record["rows"]
        ]
        lines += [
            "## Donor effects",
            "",
            _table(
                [
                    "Pair",
                    "Base",
                    "Donor",
                    "Max reference error",
                    "Max protected change",
                ],
                rows,
            ),
            "",
            "Reference labels and donor eligibility are supplied by the study. These errors have no implicit acceptance tolerance.",
            "",
        ]
    elif schema == "nmn.gate-path-study.v1":
        path = record["path"]
        rows = [
            [name, _maximum(residual)] for name, residual in path["residuals"].items()
        ]
        lines += [
            "## Prediction residuals",
            "",
            _table(["Method", "Maximum absolute endpoint residual"], rows),
            "",
            f"Quadrature intervals: {_text(path['cost']['intervals'])}; model calls: {_text(path['cost']['model_forward_calls'])}.",
            "",
            "Residuals compare numerical predictions with actual endpoint effects; they are not uniform error bounds.",
            "",
        ]
    elif schema == "nmn.kernel-diagnostics.v1":
        layer, sensors = record["layer"], record["sensors"]
        lines += [
            f"Module: {_text(record['module'])}.",
            "",
            f"Function-space scope: {_text(layer['scope'])}.",
            "",
            f"Empirical minimum separation ratio: {_text(sensors['empirical_minimum_separation_ratio'])}.",
            "",
            f"Supplied observation noise radius: {_text(sensors['noise_radius'])}.",
            "",
            "Finite spectra and separation observations do not establish global PSD or stable state recovery.",
            "",
        ]
    elif schema == "nmn.native-benchmark.v1":
        rows = []
        for name, method in record["methods"].items():
            rows.append(
                [
                    name,
                    method["status"],
                    method.get("parameter_count", "unavailable"),
                    (
                        method.get("baseline_mse")
                        if method.get("baseline_mse") is not None
                        else "not measured"
                    ),
                    method.get("error", ""),
                ]
            )
        lines += [
            "## All methods",
            "",
            _table(["Method", "Status", "Parameters", "Baseline MSE", "Error"], rows),
            "",
            f"Timing budget: {_text(record['contract']['warmup'])} warmup and {_text(record['contract']['repeats'])} measured batches per case.",
            "",
            "Different parameter counts, competence and fitting histories remain relevant. No winner is selected by this note.",
            "",
        ]
    elif schema == "nmn.native-training.v1":
        rows = [
            [
                run["seed"],
                run["status"],
                run["steps_completed"],
                run["best_step"],
                run.get("best_validation_mse"),
                run.get("error") or run.get("checkpoint_error") or "",
            ]
            for run in record["runs"]
        ]
        lines += [
            f"Training protocol: {_text(record['protocol'])}.",
            "",
            f"Target provenance: {_text(record['target_provenance'])}.",
            "",
            "## All seed outcomes",
            "",
            _table(
                ["Seed", "Status", "Steps", "Selected step", "Selection MSE", "Error"],
                rows,
            ),
            "",
            f"Selection rule: {_text(record['selection_rule'])}.",
            "",
            f"Seed scope: {_text(record['seed_scope'])}.",
            "",
            "Validation selects checkpoints; it is not untouched final evaluation. Routing and supplied semantics are imposed, not identified by low loss.",
            "",
        ]
    limitations = list(record.get("limitations", []))
    if schema == "nmn.gate-path-study.v1":
        limitations += record["path"].get("limitations", [])
    if limitations:
        lines += (
            ["## Recorded limitations", ""]
            + [f"- {_text(item)}" for item in limitations]
            + [""]
        )
    return "\n".join(lines)


def export_native_record(source, destination):
    """Copy exact JSON bytes and a note into a new directory, with file hashes.

    The manifest supports integrity checking, not model replay or authorship.
    No frameworks are imported and no vault indexes or existing files are changed.
    """
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    record = json.loads(data)
    note = render_native_note(record).encode("utf-8")
    payload = {"data.json": data, "Report.md": note}
    manifest = {
        "schema": "nmn.native-export.v1",
        "record_schema": record["schema"],
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in payload.items()
        },
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assurance": "stored file integrity; no computational replay",
    }
    payload["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        for name, content in payload.items():
            (destination / name).write_bytes(content)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {
        "status": "exported",
        "output": str(destination),
        "schema": record["schema"],
    }


def verify_native_export(directory):
    """Check the exported files and embedded model identities without execution."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if manifest.get("schema") != "nmn.native-export.v1" or set(manifest["files"]) != {
        "data.json",
        "Report.md",
    }:
        raise ValueError("unsupported native export manifest")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"file content mismatch: {name}")
    record = json.loads((directory / "data.json").read_bytes())
    if (
        record.get("schema") not in SCHEMAS
        or record["schema"] != manifest["record_schema"]
    ):
        raise ValueError("record schema mismatch")
    _check_identities(record)
    return {
        "status": "integrity-checked",
        "output": str(directory),
        "schema": record["schema"],
        "recomputed": False,
    }
