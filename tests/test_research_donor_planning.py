"""Budgeted eligibility preserves excluded, missing and uninspected pairs."""

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.donor_planning import plan_donors


def test_metadata_selection_budget_and_stable_order():
    samples = [
        ResearchSample(s, [i], "evaluation", s, tags)
        for i, (s, tags) in enumerate(
            [("c", {}), ("b", {"kind": 1}), ("a", {"kind": 1})]
        )
    ]
    data = ResearchDataset(samples, name="eligibility", provenance="supplied metadata")
    options = dict(
        split="evaluation",
        modules=["b"],
        match_semantics=["kind"],
        max_comparisons=9,
        max_pairs=9,
        provenance="declared selection",
    )
    complete = plan_donors(data, **options)
    assert complete["status"] == "complete"
    assert [(p["base_id"], p["donor_id"]) for p in complete["pairs"]] == [
        ("a", "b"),
        ("b", "a"),
    ]
    assert complete["coverage"]["exclusions"]["missing-semantics"] == 4
    limited = plan_donors(data, **{**options, "max_pairs": 1})
    assert limited["status"] == "budget-stopped"
    assert limited["coverage"]["inspected"] == 2
    assert limited["coverage"]["uninspected"] == 7
    assert limited["pairs"] == complete["pairs"][:1]
