"""Backend-free extraction of reusable native research components."""

import hashlib
import json
import shutil
from pathlib import Path

from .native_export import SCHEMAS, _check_identities

_FIELDS = {
    "model": "model_snapshot",
    "dataset": "dataset",
    "maps": "maps",
    "fit": "fit",
    "evaluation": "evaluation",
    "evaluation-model": "evaluation_snapshot",
}


def _read_components(source):
    raw = Path(source).read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict) or record.get("schema") not in SCHEMAS:
        raise ValueError("unsupported native research schema")
    _check_identities(record)
    # Only components with a documented consumer are exposed. In particular,
    # probe result tables are not standalone replay records or saved classifiers.
    available = {}
    if record["schema"] in (
        "nmn.native-model.v1",
        "nmn.native-research.v1",
        "nmn.nnx-model.v1",
        "nmn.nnx-research.v1",
    ):
        available["model"] = record
    for name in ("model", "dataset", "evaluation-model"):
        key = _FIELDS[name]
        if isinstance(record.get(key), dict):
            available[name] = record[key]
    if record["schema"] in ("nmn.fitted-reduction.v1", "nmn.reduction-study.v1"):
        available["maps"] = record["maps"]
    if record["schema"] == "nmn.fitted-reduction.v1":
        for name in ("fit", "evaluation"):
            if record[name].get("schema") != "nmn.reduction-study.v1":
                raise ValueError("fitted summary contains an unsupported child record")
            available[name] = record[name]
    if record["schema"] == "nmn.gate-search.v1":
        available["selection"] = record["selection"]
    if record["schema"] in ("nmn.gate-search.v1", "nmn.edit-selection.v1"):
        selection = (
            record["selection"] if record["schema"] == "nmn.gate-search.v1" else record
        )
        if selection.get("schema") != "nmn.edit-selection.v1":
            raise ValueError("search contains an unsupported selection record")
        selected = selection.get("selected")
        if selected is not None:
            if (
                not isinstance(selected, str)
                or selection.get("status") != "selected"
                or selected not in selection.get("candidates", {})
                or selection.get("ledger", {}).get("selected", {}).get("candidate_id")
                != selected
            ):
                raise ValueError(
                    "selected edit does not match its saved candidate and freeze ledger"
                )
            available["selected-edit"] = {selected: selection["candidates"][selected]}
    if record["schema"] == "nmn.donor-plan.v1":
        available["pairs"] = record["pairs"]
    return raw, record, available


def list_native_components(source):
    """List reusable local JSON components without executing a model."""
    _, record, available = _read_components(source)
    return {"source_schema": record["schema"], "components": sorted(available)}


def extract_native_component(source, component, destination):
    """Write data.json and an extraction receipt into a new directory.

    The receipt links exact source bytes to serialized selected JSON, without
    claiming numerical replay, fitting reproduction, or semantic validation.
    The data file is directly consumable by native --model/--dataset/--maps or
    replay, according to the selected component.
    """
    raw, record, available = _read_components(source)
    if component not in available:
        raise ValueError(
            "component is unavailable; list this record's components first"
        )
    data = json.dumps(available[component], indent=2, allow_nan=False).encode() + b"\n"
    receipt = {
        "schema": "nmn.component-extraction.v1",
        "source_schema": record["schema"],
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "component": component,
        "data_file": "data.json",
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "recomputed": False,
        "assurance": "JSON selection; stored model identities checked; no model execution or scientific validation",
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "data.json").write_bytes(data)
        (destination / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(destination)
        raise
    return {"status": "extracted", "output": str(destination), **receipt}
