"""Recompute supported native records, separating replay from file integrity."""

import hashlib
import json
import math

import torch

from ..research.datasets import DonorPair, ResearchDataset
from ..research.native_export import _check_identities
from .coalitions import coalition_study
from .interpretable import Intervention
from .protection import protection_study
from .research import collect_research_data, model_from_snapshot
from .studies import donor_study


def replay_native_record(record, *, atol=1e-10, rtol=1e-8):
    """Recompute saved measurements on CPU without training or executable loaders.

    Supports native observations, donor studies (saved reference expectations),
    classification protection and coalition studies. Match means agreement with
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
        "nmn.native-research.v1",
        "nmn.donor-study.v1",
        "nmn.protection-study.v1",
        "nmn.coalition-study.v1",
    }
    schema = record.get("schema")
    if schema not in supported:
        raise ValueError("unsupported native replay schema")
    _check_identities(record)
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

    if schema == "nmn.native-research.v1":
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
    else:
        dataset = ResearchDataset.from_dict(record["dataset"])
        if dataset.sha256 != record["dataset_sha256"]:
            raise ValueError("dataset content hash mismatch")
        protocol = record["protocol"]
        if schema == "nmn.donor-study.v1":
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
            )
            # Replay supplied expectations, never execute a saved callback name.
            for current, saved in zip(actual["rows"], record["rows"]):
                current["pair"]["expected"] = saved["pair"]["expected"]
                if "donor_reads" not in saved:
                    current.pop("donor_reads", None)
            keys = ["rows"]
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
    mismatches = []
    checked = 0
    maximum_error = 0.0

    def compare(saved, current, path):
        nonlocal checked, maximum_error
        checked += 1
        if isinstance(saved, dict) and isinstance(current, dict):
            if set(saved) != set(current):
                mismatches.append({"path": path, "reason": "mapping keys differ"})
            for key in sorted(set(saved) & set(current)):
                compare(saved[key], current[key], path + "/" + str(key))
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
    if schema != "nmn.native-research.v1":
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
    return {
        "schema": "nmn.native-replay.v1",
        "status": "matched" if not mismatches else "mismatch",
        "source_schema": schema,
        "record_sha256": hashlib.sha256(
            json.dumps(record, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "model_sha256": snapshot["model_sha256"],
        "tolerances": {
            "atol": atol,
            "rtol": rtol,
            "rule": "abs(saved-replayed) <= atol + rtol*abs(saved)",
        },
        "compared_fields": keys,
        "checked_nodes": checked,
        "maximum_absolute_error": maximum_error,
        "mismatches": mismatches,
        "execution": actual,
        "limitations": [
            "Numerical agreement with saved observations is not scientific validation or a certificate.",
            "CPU replay may differ from original hardware or runtime versions.",
            "Donor reference expectations are reused as supplied data; reference callbacks are not rerun.",
        ],
    }
