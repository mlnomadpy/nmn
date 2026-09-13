"""Three-neuron reference: exact arithmetic, explicit interventions and scope.

This is a designed computation, not a trained model or semantic-discovery result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import tempfile
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA = "nmn.three-neuron.v1"


def rational(value: str) -> Fraction:
    """Parse bounded textual rational inputs without floating-point conversion."""
    if (
        not isinstance(value, str)
        or len(value) > 64
        or not re.fullmatch(r"[+-]?(?:\d+(?:/\d+|\.\d*)?|\.\d+)", value)
    ):
        raise ValueError(
            "expected a rational/decimal string up to 64 characters, without exponents"
        )
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("expected a finite rational such as 1/2") from exc
    if not 0 <= result <= 1:
        raise ValueError("inputs, gates and replacement h must lie in [0, 1]")
    return result


def kernel(center: tuple, value: tuple) -> Fraction:
    """Unbiased yat section with epsilon=1 and exact rational arithmetic."""
    if len(center) != len(value) or not center:
        raise ValueError("kernel dimensions must be equal and nonempty")
    dot = sum((c * x for c, x in zip(center, value)), Fraction(0))
    distance = sum(((c - x) ** 2 for c, x in zip(center, value)), Fraction(0))
    return dot**2 / (1 + distance)


def trace(
    u: Fraction,
    v: Fraction,
    gate: Fraction = Fraction(1),
    replacement_h: Optional[Fraction] = None,
) -> Dict[str, Any]:
    """Recompute y after gating h or replacing it at the layer-one cut."""
    for value in (u, v, gate):
        if not isinstance(value, Fraction) or not 0 <= value <= 1:
            raise ValueError("use Fraction values in [0, 1]")
    if replacement_h is not None and (
        not isinstance(replacement_h, Fraction) or not 0 <= replacement_h <= 1
    ):
        raise ValueError("replacement h must be a Fraction in [0, 1]")
    h_section = kernel((Fraction(1),), (u,))
    h_gated = gate * h_section
    h = h_gated if replacement_h is None else replacement_h
    p = kernel((Fraction(1),), (v,))
    y = kernel((Fraction(1), Fraction(1)), (h, v))
    return {
        "input": {"u": str(u), "v": str(v)},
        "action": {
            "gate": str(gate),
            "replacement_h": None if replacement_h is None else str(replacement_h),
            "order": "gate first; optional h replacement after layer one",
        },
        "layer1": {
            "h_section": str(h_section),
            "h_gated": str(h_gated),
            "h": str(h),
            "p": str(p),
        },
        "layer2": {"reads": {"h": str(h), "v": str(v)}, "y": str(y)},
        "outputs": {"target": str(y), "protected": str(p), "leaky": str(p + y)},
    }


def architecture() -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "arithmetic": "exact rational; epsilon=1; fixed centers and coefficients",
        "state": ["u", "v", "h", "p", "y"],
        "encoder": "(u,v,0,0,0)",
        "layer1": ["h=gamma*k((1),(u))", "p=k((1),(v))"],
        "layer2": ["y=k((1,1),(h,v))"],
        "readouts": {"target": "y", "protected": "p", "failure_variant": "p+y"},
        "semantics": "supplied by construction; no learned mechanism identified",
    }


def experiment() -> Dict[str, Any]:
    """Exhaustive evidence for one fixed finite input/action contract."""
    grid = [Fraction(i, 4) for i in range(5)]
    cases = []
    for u, v in product(grid, repeat=2):
        baseline, edited = trace(u, v), trace(u, v, Fraction(0))
        cases.append({"baseline": baseline, "edited": edited})
    protected_failures = [
        c
        for c in cases
        if c["baseline"]["outputs"]["protected"] != c["edited"]["outputs"]["protected"]
    ]
    leaky_failures = [
        c
        for c in cases
        if c["baseline"]["outputs"]["leaky"] != c["edited"]["outputs"]["leaky"]
    ]
    witness = cases[-1]
    target_pass = Fraction(witness["edited"]["outputs"]["target"]) - Fraction(
        witness["baseline"]["outputs"]["target"]
    ) == Fraction(-7, 2)
    return {
        "schema": SCHEMA,
        "architecture": architecture(),
        "contract": {
            "input_grid": [str(x) for x in grid],
            "action": "shared gate change gamma=1 -> gamma=0",
            "protected": "p; exact equality at every grid input",
            "target": "at u=v=1, y decreases by exactly 7/2",
            "failure_variant": "require p+y unchanged on the same grid",
            "scope": "25 explicit inputs; one shared action; exact rational execution",
        },
        "results": {
            "coverage": len(cases),
            "protected_status": (
                "certified-under-assumptions"
                if not protected_failures
                else "counterexample-found"
            ),
            "protected_violations": len(protected_failures),
            "target_witness_pass": target_pass,
            "target_changes": len(leaky_failures),
            "failure_variant_status": (
                "counterexample-found"
                if leaky_failures
                else "certified-under-assumptions"
            ),
            "failure_variant_violations": len(leaky_failures),
        },
        "witness": witness,
        "cases": cases,
        "limitations": [
            "Finite enumeration is not a continuous-domain or floating-point certificate.",
            "Protection is architectural and is not specific to the yat kernel.",
            "No training, semantic recovery or comparison advantage is established.",
            "Bundle replay uses the same evaluator, not an independent verifier.",
        ],
    }


def encoded(value: Dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def report(data: Dict[str, Any]) -> str:
    result = data["results"]
    return (
        "---\ntype: experiment\nstatus: evidence-recorded\n"
        "tags: [research/intervention, nmn/reference]\n---\n"
        "# Three-neuron exact reference\n\n"
        "Designed, fixed-parameter model; no training or semantic discovery.\n\n"
        "h = gamma k(1,u); p = k(1,v); y = k((1,1),(h,v)); epsilon = 1.\n\n"
        f"Checked **{result['coverage']} inputs** with the shared gate edit 1 → 0.\n"
        f"Protected-output violations: **{result['protected_violations']}**.\n"
        f"Target changes: **{result['target_changes']}**; zero change is allowed outside the target witness.\n\n"
        "At u=v=1: h changes 1 → 0, y changes 4 → 1/2, and p stays 1.\n"
        "The failure variant p+y changes 5 → 3/2; its protection contract fails.\n\n"
        "## Evidence\n\n"
        "[Exact cases and contract](evidence.json) · [Artifact manifest](manifest.json)\n\n"
        "## Limits\n\n" + "\n".join("- " + x for x in data["limitations"]) + "\n"
    )


def write_bundle(destination: Path) -> None:
    """Write a new bundle; never overwrite a user's existing directory."""
    if destination.exists() or destination.is_symlink():
        raise ValueError("output already exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nmn-reference-", dir=destination.parent))
    try:
        evidence = experiment()
        files = {
            "evidence.json": encoded(evidence),
            "Report.md": report(evidence).encode(),
        }
        manifest = {
            "schema": SCHEMA,
            "python": platform.python_version(),
            "evaluator_sha256": digest(Path(__file__).read_bytes()),
            "files": {name: digest(content) for name, content in files.items()},
        }
        for name, content in files.items():
            (staging / name).write_bytes(content)
        (staging / "manifest.json").write_bytes(encoded(manifest))
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def replay(bundle: Path) -> Dict[str, Any]:
    """Check known files and regenerate evidence; never execute bundled code."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported bundle schema")
    if manifest.get("evaluator_sha256") != digest(Path(__file__).read_bytes()):
        raise ValueError("evaluator differs; replay with the recorded implementation")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("manifest files must be an object")
    if set(manifest["files"]) != {"evidence.json", "Report.md"}:
        raise ValueError("unexpected bundle file inventory")
    for name, expected in manifest["files"].items():
        if digest((bundle / name).read_bytes()) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    data = experiment()
    if (bundle / "evidence.json").read_bytes() != encoded(data):
        raise ValueError("evidence differs from recomputed contract")
    if (bundle / "Report.md").read_text() != report(data):
        raise ValueError("report differs from recomputed evidence")
    return {
        "status": "reproduced",
        "scope": data["contract"]["scope"],
        "python_original": manifest["python"],
        "python_current": platform.python_version(),
    }
