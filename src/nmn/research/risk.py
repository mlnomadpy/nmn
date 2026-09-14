"""Exact Bernoulli clause tests for a fixed, predeclared edit family."""

import hashlib
import json
from fractions import Fraction


def validate_risk(plan, observations):
    """Compute binomial lower-tail tests and Bonferroni across candidate edits.

    Each edit must pass ALL clauses, so its p-value is their maximum. Dependence
    across edits/clauses is permitted; IID trials within each clause, a frozen
    family and fixed sample sizes are external assumptions, never inferred here.
    """
    plan = json.loads(json.dumps(plan, allow_nan=False))
    observations = json.loads(json.dumps(observations, allow_nan=False))
    if not isinstance(plan, dict) or not isinstance(observations, dict):
        raise ValueError("risk plan and observations must be objects")
    if (
        plan.get("schema") != "nmn.risk-plan.v1"
        or observations.get("schema") != "nmn.risk-observations.v1"
    ):
        raise ValueError("unsupported risk plan or observation schema")
    for key in ("provenance", "sampling_law", "freeze_provenance"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"{key} is required")
    assumptions = plan.get("assumptions")
    required = {
        "iid_within_clause",
        "family_independent_of_validation",
        "fixed_sample_counts",
    }
    if (
        not isinstance(assumptions, dict)
        or set(assumptions) != required
        or any(type(v) is not bool for v in assumptions.values())
    ):
        raise ValueError(
            "declare each required assumption as a boolean; false requests diagnostic calculation only"
        )

    def probability(value):
        if not isinstance(value, str):
            raise ValueError("probabilities must be exact rational strings")
        try:
            p = Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("invalid rational probability") from exc
        if not 0 < p < 1 or p.denominator.bit_length() > 32:
            raise ValueError(
                "probabilities must lie in (0,1), denominator at most 32 bits"
            )
        return p

    delta = probability(plan["delta"])
    candidates = plan["candidates"]
    clauses = plan["clauses"]
    if (
        not isinstance(candidates, dict)
        or not 1 <= len(candidates) <= 256
        or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in candidates.items()
        )
    ):
        raise ValueError("declare 1 to 256 candidate identities")
    if not isinstance(clauses, dict) or not 1 <= len(clauses) <= 16:
        raise ValueError("declare 1 to 16 clauses")
    for name, spec in clauses.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(spec, dict)
            or set(spec) != {"alpha", "sample_count"}
        ):
            raise ValueError("clauses require named alpha and sample_count")
        probability(spec["alpha"])
        if (
            type(spec["sample_count"]) is not int
            or not 1 <= spec["sample_count"] <= 2048
        ):
            raise ValueError("sample counts must be fixed integers from 1 to 2048")
    records = observations["candidates"]
    if not isinstance(records, dict) or set(records) != set(candidates):
        raise ValueError("observations must cover exactly the frozen candidate family")
    threshold = delta / len(candidates)

    def encode(value):
        return dict(
            numerator_hex=hex(value.numerator),
            denominator_hex=hex(value.denominator),
            approximate=float(value),
        )

    results = {}
    passes = []
    for name, record in records.items():
        if record["identity"] != candidates[name] or set(record["clauses"]) != set(
            clauses
        ):
            raise ValueError("candidate identity or clause coverage mismatch")
        rows = {}
        pvalues = []
        for clause, spec in clauses.items():
            trials = record["clauses"][clause]
            n = spec["sample_count"]
            if not isinstance(trials, list) or len(trials) != n:
                raise ValueError("observed trial count differs from frozen plan")
            ids = []
            groups = []
            for trial in trials:
                if (
                    not isinstance(trial, dict)
                    or set(trial) != {"sample_id", "group_id", "failed"}
                    or any(
                        not isinstance(trial[k], str) or not trial[k]
                        for k in ("sample_id", "group_id")
                    )
                    or type(trial["failed"]) is not bool
                ):
                    raise ValueError("trials require IDs and a boolean failure")
                ids.append(trial["sample_id"])
                groups.append(trial["group_id"])
            if len(set(ids)) != n or len(set(groups)) != n:
                raise ValueError(
                    "repeated samples/groups cannot be counted as IID trials within a clause"
                )
            x = sum(t["failed"] for t in trials)
            alpha = probability(spec["alpha"])
            a, b = alpha.numerator, alpha.denominator
            term = (b - a) ** n
            total = term
            for k in range(x):
                term = term * (n - k) * a // ((k + 1) * (b - a))
                total += term
            q = Fraction(total, b**n)
            pvalues.append(q)
            rows[clause] = dict(
                samples=n, failures=x, empirical_failure_rate=x / n, p_value=encode(q)
            )
        joint = max(pvalues)
        if joint <= threshold:
            passes.append(name)
        results[name] = dict(
            clauses=rows,
            all_clause_p_value=encode(joint),
            passes_rule=joint <= threshold,
        )
    declared = all(assumptions.values())
    return dict(
        schema="nmn.risk-validation.v1",
        status=(
            "calculated-under-declared-assumptions" if declared else "diagnostic-only"
        ),
        plan=plan,
        observations=observations,
        plan_sha256=hashlib.sha256(
            json.dumps(plan, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        results=results,
        bonferroni_threshold=encode(threshold),
        rule_passes=sorted(passes),
        accepted_under_declared_assumptions=sorted(passes) if declared else [],
        method="exact binomial lower tail; maximum across clauses; Bonferroni across fixed edits",
        arithmetic="Exact integer/rational decisions; approximate floats are display only",
        limitations=[
            "Assumption declarations are not verified by this calculation.",
            "Repeated/shared-pool pairs are not automatically independent trials.",
            "No optional stopping, adaptive family expansion, distribution shift or uniform-input certificate.",
            "Passing loose semantic criteria does not establish useful task accuracy.",
        ],
        source="https://arxiv.org/html/2110.01052v5",
    )
