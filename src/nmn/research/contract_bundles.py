"""Replayable explicit-contract runs, including incomplete finite checks."""

from __future__ import annotations

import platform
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

from .bundles import _json, _read, export_any, reproduce_any
from .contracts import check
from .contracts import validate as validate_contract
from .model import validate as validate_model
from .reference import digest, encoded

SCHEMA = "nmn.contract-bundle.v1"
FILES = {"model.json", "contract.json", "run.json", "evidence.json", "Report.md"}


def _implementation() -> Dict[str, str]:
    root = Path(__file__).parent
    return {
        name: digest((root / name).read_bytes())
        for name in (
            "contract_bundles.py",
            "contracts.py",
            "model.py",
            "reference.py",
            "bundles.py",
        )
    }


def _report(data: Dict[str, Any]) -> str:
    witness = data["counterexample"]
    detail = "No violation observed in the checked cases."
    if witness:
        detail = "Counterexample: " + "; ".join(witness["violations"]) + "."
    return (
        "---\ntype: experiment\nstatus: evidence-recorded\n"
        "tags: [research/intervention, nmn/reference]\n---\n"
        "# Explicit-contract run\n\n"
        f"**Outcome: {data['status']}**\n\n"
        f"Cases checked: **{data['cases_checked']}/{data['cases_total']}**. "
        f"Case budget: {data['case_budget']}.\n\n"
        f"Target: **{data['target_status']}**. Observed protection violations: "
        f"{data['protected_violations_observed']}.\n\n{detail}\n\n"
        "[Model](model.json) · [Contract](contract.json) · [Run budget](run.json) · "
        "[All checked cases](evidence.json) · [Manifest](manifest.json)\n\n"
        "The contract's finite input grid, shared gate edit, protected tolerance "
        "and optional target witness govern this result. Incomplete coverage without "
        "a failure is inconclusive; a failure can disprove the contract before "
        "coverage completes. Replay preserves the original budget and outcome; it "
        "does not complete the remaining cases.\n\n"
        "Exact rational arithmetic only. No training, semantic discovery, continuous "
        "or floating-point guarantee. Replay uses the same implementation. Hashes "
        "check integrity, not authorship.\n"
    )


def _publish(destination: Path, payload: Dict[str, bytes]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise ValueError("output already exists; choose a new directory") from exc
    try:
        for name in (*sorted(FILES), "manifest.json"):
            (destination / name).write_bytes(payload[name])
    except BaseException:
        shutil.rmtree(destination)
        raise


def create_contract_bundle(
    model: Dict[str, Any],
    contract: Dict[str, Any],
    destination: Path,
    max_cases: int = 4096,
) -> Dict[str, Any]:
    model, contract = validate_model(model), validate_contract(contract)
    evidence = check(model, contract, max_cases)
    payload = {
        "model.json": encoded(model),
        "contract.json": encoded(contract),
        "run.json": encoded({"max_cases": max_cases}),
        "evidence.json": encoded(evidence),
        "Report.md": _report(evidence).encode(),
    }
    manifest = {
        "schema": SCHEMA,
        "python": platform.python_version(),
        "implementation": _implementation(),
        "files": {name: digest(data) for name, data in payload.items()},
    }
    payload["manifest.json"] = encoded(manifest)
    _publish(destination, payload)
    return {
        "status": "created",
        "bundle": str(destination),
        "verification_status": evidence["status"],
        "cases_checked": evidence["cases_checked"],
        "cases_total": evidence["cases_total"],
    }


def _checked(bundle: Path) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    manifest_bytes = _read(bundle / "manifest.json", 65536)
    manifest = _json(manifest_bytes)
    if (
        set(manifest) != {"schema", "python", "implementation", "files"}
        or manifest["schema"] != SCHEMA
    ):
        raise ValueError("unsupported contract-bundle manifest")
    if not isinstance(manifest["python"], str):
        raise ValueError("invalid recorded Python version")
    if manifest["implementation"] != _implementation():
        raise ValueError("implementation differs; use the recorded source version")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != FILES:
        raise ValueError("unexpected contract-bundle inventory")
    payload = {
        name: _read(
            bundle / name, 64 * 1024 * 1024 if name == "evidence.json" else 65536
        )
        for name in FILES
    }
    for name, content in payload.items():
        if digest(content) != manifest["files"][name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    model = validate_model(_json(payload["model.json"]))
    contract = validate_contract(_json(payload["contract.json"]))
    run = _json(payload["run.json"])
    if set(run) != {"max_cases"}:
        raise ValueError("invalid run configuration")
    evidence = check(model, contract, run["max_cases"])
    expected = {
        "model.json": encoded(model),
        "contract.json": encoded(contract),
        "run.json": encoded(run),
        "evidence.json": encoded(evidence),
        "Report.md": _report(evidence).encode(),
    }
    for name, content in expected.items():
        if payload[name] != content:
            raise ValueError(f"{name} differs from recomputed contract")
    payload["manifest.json"] = manifest_bytes
    return {
        "status": "reproduced",
        "verification_status": evidence["status"],
        "cases_checked": evidence["cases_checked"],
        "cases_total": evidence["cases_total"],
        "case_budget": run["max_cases"],
        "target_status": evidence["target_status"],
        "python_original": manifest["python"],
        "python_current": platform.python_version(),
    }, payload


def reproduce(bundle: Path) -> Dict[str, Any]:
    if _json(_read(bundle / "manifest.json", 65536)).get("schema") == SCHEMA:
        result, _ = _checked(bundle)
        return result
    return reproduce_any(bundle)


def export(bundle: Path, destination: Path) -> Dict[str, Any]:
    if _json(_read(bundle / "manifest.json", 65536)).get("schema") == SCHEMA:
        _, payload = _checked(bundle)
        _publish(destination, payload)
        return {"status": "exported", "report": str(destination / "Report.md")}
    return export_any(bundle, destination)
