"""Bounded supervised search over native whole-write semantic assignments."""

import hashlib
import itertools
import math
from pathlib import Path

from ..research.semantics import TabulatedReference
from .semantics import semantic_study


def alignment_study(
    model,
    dataset,
    *,
    reference,
    module_pool,
    max_candidates,
    provenance,
    selection_split="tuning",
    evaluation_split="validation",
    tolerance=1e-8,
):
    """Select an injective variable-to-module mapping before held-out replay.

    The reference variables, native module pool and candidate order are supplied.
    Only selection counterfactual squared errors choose a mapping. Exact ties use
    the first permutation in pool order. This fits a finite correspondence, not
    a distributed representation or uniquely identified causal mechanism.
    """
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("alignment provenance is required")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if (
        not all(isinstance(s, str) and s for s in (selection_split, evaluation_split))
        or selection_split == evaluation_split
    ):
        raise ValueError("selection and evaluation splits must be distinct")
    table = reference.to_dict()
    variables = table["variables"]
    if (
        not isinstance(module_pool, (list, tuple))
        or any(not isinstance(m, str) or not m for m in module_pool)
        or len(set(module_pool)) != len(module_pool)
        or not set(module_pool) <= set(model.state_names)
        or len(module_pool) < len(variables)
    ):
        raise ValueError("supply unique native modules, at least one per variable")
    module_pool = list(module_pool)
    split_cases: dict[str, list[dict]] = {selection_split: [], evaluation_split: []}
    for case in table["cases"]:
        base = dataset.sample(case["base_id"])
        donor = dataset.sample(case["donor_id"])
        if base.split != donor.split or base.split not in split_cases:
            raise ValueError("reference pairs must stay within declared study splits")
        split_cases[base.split].append(case)
    if any(not cases for cases in split_cases.values()):
        raise ValueError("both splits require reference counterfactual cases")
    for cases in split_cases.values():
        if {v for c in cases for v in c["variables"]} != set(variables):
            raise ValueError("each split must exercise every reference variable")

    def subset(split):
        cases = split_cases[split]
        ids = {c[k] for c in cases for k in ("base_id", "donor_id")}
        return TabulatedReference(
            dict(table, cases=cases, baseline={s: table["baseline"][s] for s in ids})
        )

    def correspondence(mapping):
        return dict(
            mapping=mapping,
            origin="supervised",
            provenance=provenance,
            anchors=[],
            ambiguities=[
                "Finite injective whole-module family; equivalent mappings may exist."
            ],
        )

    def score(study):
        errors = [e for row in study["cases"] for e in row["absolute_error"].values()]
        return math.fsum(e * e / len(errors) for e in errors)

    selection_reference = subset(selection_split)
    candidates, executions = [], []
    winner = None
    for index, modules in enumerate(
        itertools.islice(
            itertools.permutations(module_pool, len(variables)), max_candidates
        )
    ):
        mapping = {v: [m] for v, m in zip(variables, modules)}
        study = semantic_study(
            model,
            dataset,
            reference=selection_reference,
            correspondence=correspondence(mapping),
            tolerance=tolerance,
        )
        loss = score(study)
        if not math.isfinite(loss):
            raise ValueError("alignment score is nonfinite")
        row = dict(candidate=index, mapping=mapping, selection_mse=loss)
        candidates.append(row)
        executions.append(study)
        if winner is None or loss < winner["selection_mse"]:
            winner = row
    assert winner is not None
    # The winner is frozen before any evaluation reference is executed.
    frozen = dict(winner)
    evaluation = semantic_study(
        model,
        dataset,
        reference=subset(evaluation_split),
        correspondence=correspondence(frozen["mapping"]),
        tolerance=tolerance,
    )
    total = math.prod(
        range(len(module_pool) - len(variables) + 1, len(module_pool) + 1)
    )
    return dict(
        schema="nmn.alignment-study.v1",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status="complete" if len(candidates) == total else "candidate-budget-stopped",
        model_snapshot=executions[0]["model_snapshot"],
        dataset=dataset.to_dict(),
        dataset_sha256=dataset.sha256,
        reference=table,
        protocol=dict(
            module_pool=module_pool,
            max_candidates=max_candidates,
            provenance=provenance,
            selection_split=selection_split,
            evaluation_split=evaluation_split,
            tolerance=tolerance,
            objective="mean squared counterfactual error over named output values",
            tie_break="first exact minimum in module-pool permutation order",
        ),
        coverage=dict(
            candidates_total=total,
            candidates_executed=len(candidates),
            selection_cases=len(split_cases[selection_split]),
            evaluation_cases=len(split_cases[evaluation_split]),
        ),
        candidates=candidates,
        selected=frozen,
        exact_ties=[
            r["candidate"]
            for r in candidates
            if r["selection_mse"] == frozen["selection_mse"]
        ],
        evaluation=dict(
            mse=score(evaluation),
            status=evaluation["status"],
            baseline=evaluation["baseline"],
            cases=evaluation["cases"],
        ),
        selection_executions=executions,
        evaluation_execution=evaluation,
        limitations=[
            "Supervised finite correspondence search; no distributed alignment or unique causal identification.",
            "Only injective single-module assignments are searched; other native actions remain outside the family.",
            "Budget-stopped winners are best among executed candidates only.",
            "Separate declared splits do not establish population independence or generalization guarantees.",
            "Checkpoint or architecture selection using these evaluation results requires a fresh final population.",
        ],
    )
