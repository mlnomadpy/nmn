"""Portable configured-model evidence with checked, byte-preserving exports."""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

from .model import _unique_pairs, model_verify, validate
from .reference import SCHEMA as LEGACY_SCHEMA
from .reference import digest, encoded, replay, write_bundle

SCHEMA = "nmn.configured-bundle.v1"
FILES = {"model.json", "evidence.json", "Report.md"}


def _read(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular, non-symlink file: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"bundle file exceeds size limit: {path.name}")
    return data


def _json(data: bytes) -> Dict[str, Any]:
    result = json.loads(data, object_pairs_hook=_unique_pairs)
    if not isinstance(result, dict):
        raise ValueError("bundle JSON must be an object")
    return result


def _implementation() -> Dict[str, str]:
    root = Path(__file__).parent
    return {
        name: digest((root / name).read_bytes())
        for name in ("model.py", "reference.py", "bundles.py")
    }


def render_report(evidence: Dict[str, Any]) -> str:
    witness = evidence["counterexample"]
    details = "No counterexample in the declared finite domain."
    if witness is not None:
        u, v = witness["baseline"]["input"].values()
        before = witness["baseline"]["outputs"]["protected"]
        after = witness["edited"]["outputs"]["protected"]
        details = (
            f"First counterexample: u={u}, v={v}; protected output {before} → {after}."
        )
    return (
        "---\ntype: experiment\nstatus: evidence-recorded\n"
        "tags: [research/intervention, nmn/reference]\n---\n"
        "# Configured three-neuron verification\n\n"
        f"**Outcome: {evidence['status']}**\n\n"
        f"Model SHA-256: `{evidence['model_sha256']}`\n\n"
        "Shared gate edit 1 → 0 on the 25 quarter-grid input pairs. "
        "Protected readout: p + protected_leak*y; required to stay exactly equal.\n\n"
        f"Coverage: {evidence['coverage']}. Protection violations: {evidence['protected_violations']}. "
        f"Target changes: {evidence['target_changes']} (descriptive, not target success).\n\n"
        f"{details}\n\n"
        "[Model](model.json) · [Full evidence](evidence.json) · [Manifest](manifest.json)\n\n"
        "## Scope and limitations\n\n"
        "Exact rational evaluation on the specified finite domain only. No training, "
        "learned semantic recovery, continuous-domain or floating-point certification "
        "is claimed. Replay uses the same implementation, not an independent proof "
        "checker. Hashes detect changes but do not authenticate authorship.\n"
    )


def _publish(destination: Path, payload: Dict[str, bytes]) -> None:
    """Reserve a new directory exclusively; write completion manifest last."""
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


def create_configured(model: Dict[str, Any], destination: Path) -> Dict[str, Any]:
    model = validate(model)
    evidence = model_verify(model)
    payload = {
        "model.json": encoded(model),
        "evidence.json": encoded(evidence),
        "Report.md": render_report(evidence).encode(),
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
        "protected_violations": evidence["protected_violations"],
    }


def _checked(bundle: Path) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    manifest_bytes = _read(bundle / "manifest.json", 65536)
    manifest = _json(manifest_bytes)
    if (
        set(manifest) != {"schema", "python", "implementation", "files"}
        or manifest["schema"] != SCHEMA
    ):
        raise ValueError("unsupported configured manifest")
    if not isinstance(manifest["python"], str):
        raise ValueError("manifest python version must be a string")
    if manifest["implementation"] != _implementation():
        raise ValueError(
            "implementation differs; replay with the recorded source version"
        )
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != FILES:
        raise ValueError("unexpected configured bundle inventory")
    payload = {
        name: _read(bundle / name, 65536 if name == "model.json" else 16 * 1024 * 1024)
        for name in FILES
    }
    for name, data in payload.items():
        if digest(data) != manifest["files"][name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    model = validate(_json(payload["model.json"]))
    evidence = model_verify(model)
    if payload["model.json"] != encoded(model):
        raise ValueError("model is not canonically encoded")
    if payload["evidence.json"] != encoded(evidence):
        raise ValueError("evidence differs from recomputed contract")
    if payload["Report.md"] != render_report(evidence).encode():
        raise ValueError("report differs from recomputed evidence")
    payload["manifest.json"] = manifest_bytes
    result = {
        "status": "reproduced",
        "verification_status": evidence["status"],
        "scope": evidence["contract"]["scope"],
        "python_original": manifest["python"],
        "python_current": platform.python_version(),
    }
    return result, payload


def reproduce_any(bundle: Path) -> Dict[str, Any]:
    schema = _json(_read(bundle / "manifest.json", 65536)).get("schema")
    if schema == LEGACY_SCHEMA:
        return replay(bundle)
    if schema == SCHEMA:
        result, _ = _checked(bundle)
        return result
    raise ValueError("unsupported bundle schema")


def export_any(bundle: Path, destination: Path) -> Dict[str, Any]:
    schema = _json(_read(bundle / "manifest.json", 65536)).get("schema")
    if schema == LEGACY_SCHEMA:
        replay(bundle)
        write_bundle(destination)
    elif schema == SCHEMA:
        _, payload = _checked(bundle)
        _publish(destination, payload)
    else:
        raise ValueError("unsupported bundle schema")
    return {"status": "exported", "report": str(destination / "Report.md")}
