"""A finite-study candidate ledger with explicit selection/validation separation."""

import copy
import json
from typing import Any, Dict

from .datasets import ResearchDataset


class SelectionLedger:
    """Freeze a candidate before recording held-out validation measurements.

    Records caller-supplied controls, costs and measured results; it neither fits
    nor assigns statistical guarantees. The dataset defines sample/group splits.
    Results cannot be registered as selection evidence after freeze. Validation
    can only concern the frozen candidate and declared validation splits.
    """

    def __init__(
        self,
        dataset: ResearchDataset,
        *,
        model_sha256: str,
        selection_splits=("train", "tuning"),
        validation_splits=("validation",),
    ):
        if (
            not isinstance(model_sha256, str)
            or len(model_sha256) != 64
            or any(c not in "0123456789abcdef" for c in model_sha256)
        ):
            raise ValueError("model_sha256 must be a lowercase SHA256 digest")
        self.dataset = dataset
        self.selection_splits = tuple(selection_splits)
        self.validation_splits = tuple(validation_splits)
        if (
            not self.selection_splits
            or not self.validation_splits
            or set(self.selection_splits) & set(self.validation_splits)
        ):
            raise ValueError(
                "selection and validation splits must be nonempty and disjoint"
            )
        self._record: Dict[str, Any] = {
            "schema": "nmn.selection-ledger.v1",
            "model_sha256": model_sha256,
            "dataset_sha256": dataset.sha256,
            "selection_splits": list(selection_splits),
            "validation_splits": list(validation_splits),
            "candidates": {},
            "events": [],
            "selected": None,
        }

    @staticmethod
    def _json(value):
        return json.loads(json.dumps(value, allow_nan=False))

    def register(self, candidate_id: str, controls: dict):
        if self._record["selected"] is not None:
            raise ValueError("candidate set is frozen")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in self._record["candidates"]
        ):
            raise ValueError("candidate ID must be nonempty and unique")
        self._record["candidates"][candidate_id] = self._json(controls)

    def record(self, candidate_id, *, phase, sample_ids, measurements, costs=None):
        if candidate_id not in self._record["candidates"]:
            raise ValueError("unknown candidate")
        if phase not in ("selection", "validation"):
            raise ValueError("phase must be selection or validation")
        selected = self._record["selected"]
        if phase == "selection" and selected is not None:
            raise ValueError("selection evidence is frozen")
        if phase == "validation" and (
            selected is None or selected["candidate_id"] != candidate_id
        ):
            raise ValueError("validation requires the frozen candidate")
        ids = tuple(sample_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("provide nonempty unique sample IDs")
        splits = (
            self.selection_splits if phase == "selection" else self.validation_splits
        )
        if any(self.dataset.sample(sid).split not in splits for sid in ids):
            raise ValueError(f"sample access violates {phase} splits")
        self._record["events"].append(
            self._json(
                {
                    "index": len(self._record["events"]),
                    "phase": phase,
                    "candidate_id": candidate_id,
                    "sample_ids": ids,
                    "measurements": measurements,
                    "costs": {} if costs is None else costs,
                }
            )
        )

    def freeze(self, candidate_id: str, *, rule: str):
        if self._record["selected"] is not None:
            raise ValueError("selection is already frozen")
        if (
            candidate_id not in self._record["candidates"]
            or not isinstance(rule, str)
            or not rule
        ):
            raise ValueError("provide a registered candidate and selection rule")
        self._record["selected"] = {
            "candidate_id": candidate_id,
            "rule": rule,
            "after_event": len(self._record["events"]) - 1,
        }

    def to_dict(self):
        return copy.deepcopy(self._record)
