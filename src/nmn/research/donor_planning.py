"""Deterministic, budgeted donor eligibility enumeration without model execution."""

from dataclasses import asdict

from .datasets import DonorPair


def plan_donors(
    dataset, *, split, modules, match_semantics, max_comparisons, max_pairs, provenance
):
    """Select a lexicographic prefix of eligible within-split donor pairs.

    Exclude self and same-group pairs, require supplied semantic keys to exist
    and match, and preserve every inspected decision. This is not randomized
    acquisition, causal equivalence discovery, or model-based edit selection.
    """
    if (
        not isinstance(split, str)
        or not split
        or not isinstance(provenance, str)
        or not provenance.strip()
    ):
        raise ValueError("named split and selection provenance are required")
    for name, values, required in (
        ("modules", modules, True),
        ("match_semantics", match_semantics, False),
    ):
        if (
            not isinstance(values, (list, tuple))
            or (required and not values)
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(name + " must contain unique nonempty names")
    if any(type(v) is not int or v < 1 for v in (max_comparisons, max_pairs)):
        raise ValueError("comparison and pair budgets must be positive integers")
    ids = sorted(dataset.sample_ids(split=split))
    if not ids:
        raise ValueError("selected split has no samples")
    samples = {sid: dataset.sample(sid) for sid in ids}
    decisions = []
    pairs: list[DonorPair] = []
    possible = len(ids) ** 2
    for index in range(min(possible, max_comparisons)):
        if len(pairs) == max_pairs:
            break
        base_id, donor_id = ids[index // len(ids)], ids[index % len(ids)]
        base, donor = samples[base_id], samples[donor_id]
        missing = [
            key
            for key in match_semantics
            if key not in base.semantics or key not in donor.semantics
        ]
        mismatched = [
            key
            for key in match_semantics
            if key not in missing and base.semantics[key] != donor.semantics[key]
        ]
        reason = (
            "self"
            if base_id == donor_id
            else (
                "same-group"
                if base.group_id == donor.group_id
                else (
                    "missing-semantics"
                    if missing
                    else "semantic-mismatch" if mismatched else "selected"
                )
            )
        )
        decision = dict(
            base_id=base_id,
            donor_id=donor_id,
            outcome=reason,
            missing_keys=missing,
            mismatched_keys=mismatched,
        )
        if reason == "selected":
            pair = DonorPair(
                "pair-" + str(len(pairs)), base_id, donor_id, tuple(modules)
            )
            pairs.append(pair)
            decision["pair_id"] = pair.pair_id
        decisions.append(decision)
    if pairs:
        dataset.validate_pairs(pairs, match_semantics=match_semantics)
    complete = len(decisions) == possible
    return dict(
        schema="nmn.donor-plan.v1",
        status="complete" if complete else "budget-stopped",
        dataset=dataset.to_dict(),
        dataset_sha256=dataset.sha256,
        protocol=dict(
            split=split,
            modules=list(modules),
            match_semantics=list(match_semantics),
            provenance=provenance,
            ordering="base ID then donor ID, lexicographic",
            self_pairs="excluded",
            same_group_pairs="excluded",
            max_comparisons=max_comparisons,
            max_pairs=max_pairs,
        ),
        pairs=[asdict(pair) for pair in pairs],
        decisions=decisions,
        coverage=dict(
            possible=possible,
            inspected=len(decisions),
            uninspected=possible - len(decisions),
            selected=len(pairs),
            exclusions={
                reason: sum(row["outcome"] == reason for row in decisions)
                for reason in (
                    "self",
                    "same-group",
                    "missing-semantics",
                    "semantic-mismatch",
                )
            },
        ),
        limitations=[
            "Selection uses declared metadata and deterministic ID order, not model outputs or learned semantic equivalence.",
            "A budget-stopped prefix may omit eligible donors and may favor earlier base IDs; no sampling or coverage guarantee is made.",
            "Same-split/different-group checks do not prove statistical independence; model module names are validated only when executing the plan.",
        ],
    )
