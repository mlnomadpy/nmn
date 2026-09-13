"""Portable offline evidence index; no backend imports or remote assets."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .native_export import SCHEMAS, export_native_record, verify_native_export

FINITE = "nmn.finite-contract-evidence.v1"


def _summary(record, source, now):
    schema = record["schema"]
    snapshot = record.get("model_snapshot", record)
    configuration = snapshot.get("configuration", {})
    label = record.get("dataset", {}).get("name") or (
        source.parent.name
        if source.name in ("data.json", "evidence.json")
        else source.stem
    )
    status = record.get("status", "observed")
    if schema == "nmn.native-model.v1":
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
    return {
        "title": str(label),
        "schema": schema,
        "status": status,
        "scope": scope,
        "architecture": configuration.get(
            "class",
            (
                "three-neuron rational reference"
                if schema == FINITE
                else "multiple / embedded"
            ),
        ),
        "backend": (
            "exact rational"
            if schema == FINITE
            else snapshot.get("runtime", {}).get("torch", "PyTorch record")
        ),
        "contract": record.get(
            "contract_sha256", record.get("protocol", "not declared")
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
                            "embedded identity checked"
                            if record["schema"] in SCHEMAS
                            else "saved record; not replayed"
                        ),
                    }
                )
                # Keep a compact structured preview, with full traces in the copied data.
                if record["schema"] == FINITE:
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
