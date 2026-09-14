"""Framework-independent sample, split, and donor-pair research contracts."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ResearchSample:
    """One observation with an explicit leakage group and supplied semantics.

    A group denotes the unit that must not cross splits (for example entity or
    document). Declaring groups does not establish statistical independence.
    Inputs/semantic values must be JSON-compatible; numerical model inputs are
    finite vectors. Semantic labels and their provenance are supplied by users.
    """

    sample_id: str
    inputs: Tuple[float, ...]
    split: str
    group_id: str
    semantics: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class DonorPair:
    """Replace complete named module outputs using one unchanged donor run."""

    pair_id: str
    base_id: str
    donor_id: str
    modules: Tuple[str, ...]
    expected: Mapping[str, float] = field(default_factory=dict)


class ResearchDataset:
    """Validated finite study population, serializable without ML dependencies.

    Cross-split groups, duplicate IDs and nonfinite/ragged inputs are rejected.
    Donor protocols require base and donor in the same split by default. Explicit
    cross-split access is recorded as an override rather than silently allowed.
    Data is deep-copied on ingress and access, keeping its content identity stable.
    """

    def __init__(
        self, samples: Sequence[ResearchSample], *, name: str, provenance: str
    ):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(provenance, str)
            or not provenance
        ):
            raise ValueError("dataset name and semantic/data provenance are required")
        if not samples:
            raise ValueError("dataset must contain samples")
        records = json.loads(json.dumps([asdict(s) for s in samples], allow_nan=False))
        ids = set()
        groups: Dict[str, str] = {}
        width = None
        for sample in records:
            for key in ("sample_id", "split", "group_id"):
                if not isinstance(sample[key], str) or not sample[key]:
                    raise ValueError(f"{key} must be a nonempty string")
            sid = sample["sample_id"]
            if sid in ids:
                raise ValueError(f"duplicate sample ID: {sid}")
            ids.add(sid)
            if not isinstance(sample["semantics"], dict):
                raise ValueError("semantics must be a mapping")
            values = sample["inputs"]
            if not values or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in values
            ):
                raise ValueError("inputs must be nonempty finite numeric vectors")
            width = len(values) if width is None else width
            if len(values) != width:
                raise ValueError("all sample inputs must have equal width")
            group, split = sample["group_id"], sample["split"]
            if group in groups and groups[group] != split:
                raise ValueError(f"group {group} crosses splits")
            groups[group] = split
        self._record = {
            "schema": "nmn.research-dataset.v1",
            "name": name,
            "provenance": provenance,
            "samples": records,
        }
        self._by_id = {s["sample_id"]: s for s in records}
        self.input_width = width

    @property
    def sha256(self):
        return hashlib.sha256(
            json.dumps(self._record, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def to_dict(self):
        return json.loads(json.dumps(self._record, allow_nan=False))

    @classmethod
    def from_dict(cls, record):
        if record.get("schema") != "nmn.research-dataset.v1":
            raise ValueError("unsupported dataset schema")
        return cls(
            [ResearchSample(**s) for s in record["samples"]],
            name=record["name"],
            provenance=record["provenance"],
        )

    def sample(self, sample_id):
        if sample_id not in self._by_id:
            raise ValueError(f"unknown sample ID: {sample_id}")
        record = json.loads(json.dumps(self._by_id[sample_id]))
        record["inputs"] = tuple(record["inputs"])
        return ResearchSample(**record)

    def sample_ids(self, *, split: Optional[str] = None):
        return tuple(
            sid
            for sid, s in self._by_id.items()
            if split is None or s["split"] == split
        )

    def validate_pairs(
        self,
        pairs: Sequence[DonorPair],
        *,
        allow_cross_split=False,
        match_semantics: Sequence[str] = (),
    ):
        """Check identities, split access and explicitly required semantic matches.

        Matching keys specify donor eligibility; this is not causal scrubbing or
        a learned semantic equivalence test. No donor is chosen automatically.
        """
        seen = set()
        if not pairs:
            raise ValueError("provide at least one donor pair")
        for pair in pairs:
            if (
                not isinstance(pair.pair_id, str)
                or not pair.pair_id
                or pair.pair_id in seen
            ):
                raise ValueError("pair IDs must be nonempty and unique")
            seen.add(pair.pair_id)
            base, donor = self.sample(pair.base_id), self.sample(pair.donor_id)
            if base.split != donor.split and not allow_cross_split:
                raise ValueError(f"pair {pair.pair_id} crosses splits")
            if (
                not pair.modules
                or len(set(pair.modules)) != len(pair.modules)
                or any(not isinstance(m, str) or not m for m in pair.modules)
            ):
                raise ValueError("pair modules must be nonempty unique names")
            for key in match_semantics:
                if key not in base.semantics or key not in donor.semantics:
                    raise ValueError(f"missing semantic key: {key}")
                if base.semantics[key] != donor.semantics[key]:
                    raise ValueError(
                        f"pair {pair.pair_id} does not match semantic key {key}"
                    )
            for key, value in pair.expected.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or isinstance(value, bool)
                    or (not isinstance(value, (int, float)) or not math.isfinite(value))
                ):
                    raise ValueError(
                        "expected measurements need named finite numeric values"
                    )
