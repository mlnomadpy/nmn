"""Exact Bernoulli decision thresholds and dependence declarations."""

from fractions import Fraction
from math import comb

import pytest

from nmn.research.risk import validate_risk


def fixture(n=59, failures=0):
    plan = dict(
        schema="nmn.risk-plan.v1",
        delta="1/20",
        candidates={"edit": "fixed-v1"},
        clauses={"target": dict(alpha="1/20", sample_count=n)},
        provenance="fixture",
        sampling_law="Bernoulli trials",
        freeze_provenance="fixed before trials",
        assumptions=dict(
            iid_within_clause=True,
            family_independent_of_validation=True,
            fixed_sample_counts=True,
        ),
    )
    trials = [
        dict(sample_id=str(i), group_id=str(i), failed=i < failures) for i in range(n)
    ]
    obs = dict(
        schema="nmn.risk-observations.v1",
        candidates={"edit": dict(identity="fixed-v1", clauses={"target": trials})},
    )
    return plan, obs


def test_exact_zero_failure_threshold_and_binomial_sum():
    assert validate_risk(*fixture(58))["rule_passes"] == []
    assert validate_risk(*fixture(59))["rule_passes"] == ["edit"]
    p, o = fixture(12, 3)
    r = validate_risk(p, o)["results"]["edit"]["all_clause_p_value"]
    q = Fraction(int(r["numerator_hex"], 16), int(r["denominator_hex"], 16))
    assert q == sum(Fraction(comb(12, k) * 19 ** (12 - k), 20**12) for k in range(4))


def test_failed_clause_controls_joint_test_and_unverified_assumptions_abstain():
    p, o = fixture(59)
    p["clauses"]["protected"] = dict(alpha="1/20", sample_count=59)
    o["candidates"]["edit"]["clauses"]["protected"] = [
        dict(sample_id=str(i), group_id=str(i), failed=True) for i in range(59)
    ]
    assert validate_risk(p, o)["rule_passes"] == []
    p, o = fixture(59)
    p["assumptions"]["family_independent_of_validation"] = False
    r = validate_risk(p, o)
    assert (
        r["rule_passes"] == ["edit"] and r["accepted_under_declared_assumptions"] == []
    )
    o["candidates"]["edit"]["clauses"]["target"][1]["group_id"] = "0"
    with pytest.raises(ValueError, match="IID"):
        validate_risk(p, o)
