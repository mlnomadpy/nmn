"""Portable offline evidence index; no backend imports or remote assets."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .native_export import SCHEMAS, export_native_record, verify_native_export

FINITE = "nmn.finite-contract-evidence.v1"


def load_dashboard_sources(path):
    """Read explicit local evidence paths relative to a portable source-list file."""
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(record, dict)
        or record.get("schema") != "nmn.evidence-sources.v1"
    ):
        raise ValueError("unsupported dashboard source-list schema")
    sources = record.get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(p, str) or not p.strip() for p in sources)
    ):
        raise ValueError("source list requires nonempty local path strings")
    if any("://" in p for p in sources):
        raise ValueError("source lists support local paths only")
    return [path.parent / source for source in sources]


def _summary(record, source, now):
    schema = record["schema"]
    execution = record.get("execution", record)
    snapshot = execution.get("model_snapshot", execution)
    configuration = snapshot.get("configuration", {})
    label = execution.get("dataset", {}).get("name") or (
        source.parent.name
        if source.name in ("data.json", "evidence.json")
        else source.stem
    )
    status = record.get("status", "observed")
    if schema in ("nmn.native-model.v1", "nmn.nnx-model.v1"):
        status = "definition"
    elif schema == "nmn.native-training.v1":
        statuses = {run["status"] for run in record["runs"]}
        status = (
            "failed"
            if "failed" in statuses
            else "inconclusive" if "budget-stopped" in statuses else "completed"
        )
    elif schema == "nmn.native-benchmark.v1":
        status = (
            "contains-failures"
            if any(m["status"] == "failed" for m in record["methods"].values())
            else "observed"
        )
    scope = (
        "finite exact contract"
        if schema == FINITE
        else (
            "model definition"
            if status == "definition"
            else "floating-point observations"
        )
    )
    if schema == "nmn.donor-plan.v1":
        scope = "metadata selection; no model execution"
    if schema == "nmn.interval-check.v1":
        status = record["outcome"]
    if schema in ("nmn.interval-certificate.v1", "nmn.interval-check.v1"):
        scope = "rational real-function range contract"
    if schema in ("nmn.rational-enclosure.v1", "nmn.rational-difference.v1"):
        scope = "exact rational real-function enclosure"
        status = "enclosed"
    if schema == "nmn.native-replay.v1":
        scope = "numerical replay comparison"
    return {
        "title": str(label),
        "schema": schema,
        "status": status,
        "scope": scope,
        "intervention": (
            execution.get("protocol", {}).get("donor_execution", "not declared")
            if execution.get("schema") == "nmn.donor-study.v1"
            else (
                "explicit producer-specific residual edge replacement"
                if execution.get("schema") == "nmn.edge-study.v1"
                else "see recorded protocol"
            )
        ),
        "evidence_handling": (
            "Saved checker outcome; checker was not rerun by this dashboard"
            if schema == "nmn.interval-check.v1"
            else (
                "Saved certificate; not checked by this dashboard"
                if schema == "nmn.interval-certificate.v1"
                else "Saved evidence; no computation was replayed by this dashboard"
            )
        ),
        "architecture": (
            "not applicable (selection plan)"
            if schema == "nmn.donor-plan.v1"
            else configuration.get(
                "class",
                (
                    "three-neuron rational reference"
                    if schema == FINITE
                    else configuration.get("architecture", "multiple / embedded")
                ),
            )
        ),
        "backend": (
            "none (metadata only)"
            if schema == "nmn.donor-plan.v1"
            else (
                "exact rational"
                if schema
                in (
                    FINITE,
                    "nmn.rational-enclosure.v1",
                    "nmn.rational-difference.v1",
                    "nmn.interval-certificate.v1",
                    "nmn.interval-check.v1",
                )
                else (
                    "JAX " + snapshot.get("runtime", {}).get("jax", "unknown")
                    if execution.get("schema")
                    in ("nmn.nnx-research.v1", "nmn.nnx-model.v1")
                    else snapshot.get("runtime", {}).get("torch", "PyTorch record")
                )
            )
        ),
        "contract": record.get(
            "contract_sha256",
            execution.get("contract", execution.get("protocol", "not declared")),
        ),
        "model": snapshot.get(
            "model_sha256", record.get("initial_model_sha256", "multiple / embedded")
        ),
        "samples": len(snapshot.get("sample_ids", record.get("sample_ids", []))),
        "source_name": source.name,
        "modified": datetime.fromtimestamp(
            source.stat().st_mtime, timezone.utc
        ).isoformat(),
        "age_days": max(
            0, round((now.timestamp() - source.stat().st_mtime) / 86400, 1)
        ),
        "limitations": record.get("limitations", []),
    }


def build_dashboard(sources, destination):
    """Index recognized JSON records under explicit source files/directories.

    Each recognized native record is exported with its note and byte-preserved
    data; finite exact-contract records retain their saved status and raw data.
    Invalid or missing supplied inputs become visible unavailable entries. Native
    export manifests are checked when present. No result is rerun or promoted.
    Relative evidence paths remain portable when the entire output folder moves.
    """
    destination = Path(destination)
    if destination.exists():
        raise ValueError("dashboard output already exists")
    if not sources:
        raise ValueError("supply at least one evidence source")
    files = []
    explicit = set()
    for item in sources:
        path = Path(item)
        if not path.is_dir():
            explicit.add(path)
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    now = datetime.now(timezone.utc)
    entries: list[dict[str, Any]] = []
    seen = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        for source in files:
            raw = None
            output = None
            try:
                raw = source.read_bytes()
                record = json.loads(raw)
                if not isinstance(record, dict) or record.get("schema") not in (
                    *SCHEMAS,
                    FINITE,
                ):
                    if source in explicit:
                        raise ValueError("unsupported evidence schema")
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                if digest in seen:
                    continue
                summary = _summary(record, source, now)
                index = len(entries)
                relative = f"evidence/{index:04d}"
                output = destination / relative
                manifest_path = source.parent / "manifest.json"
                if source.name == "data.json" and manifest_path.exists():
                    manifest = json.loads(manifest_path.read_bytes())
                    if manifest.get("schema") == "nmn.native-export.v1":
                        verify_native_export(source.parent)
                if record["schema"] in SCHEMAS:
                    export_native_record(source, output)
                    summary["note"] = relative + "/Report.md"
                else:
                    output.mkdir(parents=True)
                    (output / "data.json").write_bytes(raw)
                    outcome = record["status"]
                    if outcome not in (
                        "certified-under-assumptions",
                        "counterexample-found",
                        "inconclusive",
                    ):
                        raise ValueError("unknown finite-contract outcome")
                    (output / "Report.md").write_text(
                        "# Exact finite-contract record\n\n"
                        f"Saved outcome: **{outcome}**.\n\n"
                        f"Cases checked: {record['cases_checked']} / {record['cases_total']}.\n\n"
                        "[Complete evidence](data.json)\n\n"
                        "This is a copy of the saved finite result. Dashboard generation did not replay it "
                        "or establish a new certificate.\n"
                    )
                    summary["note"] = relative + "/Report.md"
                summary.update(
                    {
                        "data": relative + "/data.json",
                        "sha256": digest,
                        "integrity": (
                            "copied data; any embedded identities checked"
                            if record["schema"] in SCHEMAS
                            else "saved record; not replayed"
                        ),
                    }
                )
                # Keep a compact structured preview, with full traces in the copied data.
                if record["schema"] == "nmn.native-replay.v1":
                    summary["details"] = {
                        key: record[key]
                        for key in (
                            "source_schema",
                            "record_sha256",
                            "tolerances",
                            "compared_fields",
                            "mismatches",
                        )
                    }
                elif record["schema"] in (
                    "nmn.interval-certificate.v1",
                    "nmn.interval-check.v1",
                ):
                    summary["details"] = {
                        key: record[key]
                        for key in (
                            "status",
                            "outcome",
                            "contract",
                            "budget",
                            "counts",
                            "certificate_sha256",
                            "assurance",
                        )
                        if key in record
                    }
                elif record["schema"] in (
                    "nmn.rational-enclosure.v1",
                    "nmn.rational-difference.v1",
                ):
                    summary["details"] = {
                        key: record[key]
                        for key in (
                            "input_box",
                            "controls",
                            "output_bounds",
                            "assurance",
                        )
                    }
                elif record["schema"] == "nmn.edit-selection.v1":
                    summary["details"] = {
                        "selected": record["selected"],
                        "unexecuted": record["unexecuted"],
                        "selection": {
                            name: {
                                key: value
                                for key, value in row.items()
                                if key not in ("trace", "outputs")
                            }
                            for name, row in record["selection"].items()
                        },
                        "validation": {
                            key: value
                            for key, value in (record["validation"] or {}).items()
                            if key not in ("trace", "outputs")
                        },
                        "freeze": record["ledger"]["selected"],
                        "interpretation": "Selection status is not a validation-success or population-risk guarantee.",
                    }
                elif record["schema"] == "nmn.probe-study.v1":
                    rows = []
                    for name, row in record["evaluation"]["edits"].items():
                        rows.append(
                            {
                                "edit": name,
                                "frozen_accuracy": row["accuracy"],
                                "refitted_accuracy": row.get("refitted", {})
                                .get("evaluation", {})
                                .get("accuracy"),
                                "conditional_damage": row["conditional_damage"],
                                "baseline_correct_count": row["baseline_correct_count"],
                                "damaged_count": row["damaged_count"],
                                "disagreement": row["disagreement"],
                            }
                        )
                    summary["details"] = {
                        "protocol": record["protocol"],
                        "fit_accuracy": record["fit"]["accuracy"],
                        "evaluation_baseline_accuracy": record["evaluation"][
                            "baseline"
                        ]["accuracy"],
                        "edits_total": len(rows),
                        "first_25_edits": rows[:25],
                        "interpretation": "Null refitted accuracy means no refitted comparison. Null conditional damage means no baseline-correct samples. Neither probe establishes erasure.",
                    }
                elif record["schema"] == "nmn.coalition-study.v1":
                    summary["details"] = {
                        key: record[key]
                        for key in ("coverage", "reason", "protocol", "cost")
                    }
                    summary["details"]["coefficient_table_available"] = (
                        record["subset_coefficients"] is not None
                    )
                elif record["schema"] == "nmn.protection-study.v1":
                    rows = []
                    for task, edits in record["results"].items():
                        for edit, result in edits.items():
                            for group in result["strata"]:
                                rows.append(
                                    {
                                        "task": task,
                                        "edit": edit,
                                        "stratum": {
                                            key: group[key]
                                            for key in ("field", "value", "missing")
                                        },
                                        **{
                                            key: group["metrics"][key]
                                            for key in (
                                                "count",
                                                "originally_correct_count",
                                                "accuracy_before",
                                                "accuracy_after",
                                                "conditional_damage_rate",
                                                "disagreement_rate",
                                            )
                                        },
                                    }
                                )
                    summary["details"] = {
                        "rows_total": len(rows),
                        "first_25_rows": rows[:25],
                        "null_damage_rate": "No originally correct examples; not zero damage.",
                    }
                elif record["schema"] == "nmn.donor-plan.v1":
                    summary["details"] = {
                        "protocol": record["protocol"],
                        "coverage": record["coverage"],
                        "selected_pairs_total": len(record["pairs"]),
                        "first_25_selected_pairs": record["pairs"][:25],
                        "decisions_total": len(record["decisions"]),
                        "first_25_decisions": record["decisions"][:25],
                        "interpretation": "Saved selection only; no model execution. Uninspected pairs may be eligible and complete selection does not imply scientific validity.",
                    }
                elif record["schema"] == "nmn.edge-study.v1":
                    summary["details"] = {
                        "protocol": record["protocol"],
                        "sample_ids": record["sample_ids"],
                        "conditions_total": len(record["results"]),
                        "first_25_conditions": [
                            {
                                "condition": name,
                                "patches": record["patches"][name],
                                "samples_total": len(row["outputs"]),
                                "first_25_outputs": row["outputs"][:25],
                                "first_25_output_deltas": row["delta"][:25],
                            }
                            for name, row in list(record["results"].items())[:25]
                        ],
                        "interpretation": "Explicit producer replacements; donor origins and protection predicates are not inferred.",
                    }
                elif record["schema"] == "nmn.donor-study.v1":
                    summary["details"] = {
                        "protocol": record["protocol"],
                        "pairs_total": len(record["rows"]),
                        "first_25_pairs": [
                            {
                                "pair_id": row["pair"]["pair_id"],
                                "base_id": row["pair"]["base_id"],
                                "donor_id": row["pair"]["donor_id"],
                                "donor_reads": row.get("donor_reads", {}),
                                "donor_writes": row.get("donor_writes", {}),
                                "donor_edges": row.get("donor_edges", {}),
                                **{
                                    key: row[key]
                                    for key in (
                                        "self_donor",
                                        "expected",
                                        "edited_outputs",
                                        "absolute_reference_error",
                                        "protected_delta",
                                    )
                                },
                            }
                            for row in record["rows"][:25]
                        ],
                    }
                    if "selection_plan" in record:
                        summary["details"]["selection_plan"] = {
                            "sha256": record["selection_plan_sha256"],
                            "status": record["selection_plan"]["status"],
                            "coverage": record["selection_plan"]["coverage"],
                            "handling": "Saved plan linkage; not rechecked by the dashboard",
                        }
                elif record["schema"] == FINITE:
                    summary["details"] = {
                        key: record.get(key)
                        for key in (
                            "cases_checked",
                            "cases_total",
                            "case_budget",
                            "protected_violations_observed",
                            "counterexample",
                        )
                    }
                elif record["schema"] == "nmn.native-training.v1":
                    summary["details"] = {
                        "runs": [
                            {
                                key: run.get(key)
                                for key in (
                                    "seed",
                                    "status",
                                    "steps_completed",
                                    "best_step",
                                    "best_validation_mse",
                                    "error",
                                )
                            }
                            for run in record["runs"]
                        ]
                    }
                elif record["schema"] == "nmn.native-benchmark.v1":
                    summary["details"] = {
                        name: {
                            key: value.get(key)
                            for key in (
                                "status",
                                "parameter_count",
                                "baseline_mse",
                                "error",
                            )
                        }
                        for name, value in record["methods"].items()
                    }
                else:
                    summary["details"] = {
                        "schema": record["schema"],
                        "full_record": "Open data for all traces and measurements.",
                    }
                entries.append(summary)
                seen.add(digest)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                if output is not None and output.exists():
                    shutil.rmtree(output)
                entries.append(
                    {
                        "title": source.parent.name + "/" + source.name,
                        "status": "unavailable",
                        "scope": "unverified input",
                        "architecture": "unknown",
                        "backend": "unknown",
                        "contract": "unknown",
                        "schema": "unavailable",
                        "error": str(exc),
                        "data": None,
                        "note": None,
                        "details": {"error": str(exc)},
                    }
                )
        catalog = {
            "schema": "nmn.dashboard.v1",
            "generated": now.isoformat(),
            "records": entries,
            "assurance": "saved observations; no replay or publication",
        }
        payload = (
            json.dumps(catalog, ensure_ascii=True)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        template = (Path(__file__).parent / "assets" / "dashboard.html").read_text()
        (destination / "index.html").write_text(
            template.replace("__NMN_DATA__", payload), encoding="utf-8"
        )
        (destination / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
        return {
            "status": "created",
            "output": str(destination),
            "records": len(entries),
            "unavailable": sum(entry["status"] == "unavailable" for entry in entries),
        }
    except BaseException:
        shutil.rmtree(destination)
        raise
