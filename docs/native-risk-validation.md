# Fixed-family Bernoulli risk calculations

`nmn research risk --plan plan.json --observations observations.json` emits a
backend-free `nmn.risk-validation.v1` report. This implements the elementary
Bernoulli all-clauses baseline in the vault's decision-sufficient acquisition note.
[Learn then Test, Theorem 1 and Proposition 2](https://arxiv.org/html/2110.01052v5)
provide the multiple-testing framework. This is not its general bounded-loss
Hoeffding–Bentkus implementation, fixed-sequence procedure or later multi-risk algorithm.

Freeze M edit identities and clause definitions before independent validation.
For a clause with n IID Bernoulli failures and X observed failures, compute
q = Σ(k=0..X) C(n,k) α^k(1−α)^(n−k). Its lower-tail p-value is valid for failure
probability greater than α. A candidate passes only when the maximum clause
p-value is at most δ/M. If any clause is truly bad, this maximum dominates that
clause's valid p-value. A union bound over M edits controls acceptance of any bad
candidate at δ. Dependence across edits and clauses is allowed; independence within
a clause and independence from the frozen family are not optional.

`nmn.risk-plan.v1` requires rational-string `delta`, `candidates` mapping names to
fixed identity strings, and `clauses` mapping names to rational-string `alpha` and
integer `sample_count`. Supply `provenance`, `sampling_law`, `freeze_provenance` and
three boolean `assumptions`: `iid_within_clause`,
`family_independent_of_validation`, `fixed_sample_counts`.
`nmn.risk-observations.v1` contains matching `candidates`; each has `identity` and
`clauses`, whose values are trial lists with `sample_id`, `group_id`, boolean
`failed`. Trials must cover exactly the fixed counts and all declared clauses.
Duplicate samples or groups within a clause are rejected; unique IDs do not prove IID.

If all assumptions are declared true, status is
`calculated-under-declared-assumptions`. The declarations are not independently
verified. If any is false, status is `diagnostic-only` and
`accepted_under_declared_assumptions` is empty even if `rule_passes` is nonempty.
This permits honest retrospective analysis without claiming fresh validation.
An empty accepted set means abstention, not proof that all candidates are bad.

Decisions use exact integers and rational probabilities, including comparison at
the Bonferroni threshold. Numerator/denominator hex strings retain exact values;
approximate floats are display-only. Limits are 256 candidates, 16 clauses,
1–2048 trials per clause, and probability denominators at most 32 bits.
Raw trials, declarations, counts and plan identity remain in the report. Native
Obsidian export recomputes the calculation and rejects inconsistent reports;
the dashboard labels exact calculation separately from verified assumptions.

No optional stopping, growing candidate family, reused-pool independence,
distribution-shift assurance, semantic correctness or all-input guarantee is supplied.
Changing model, reference, criteria or sample counts after looking at outcomes
requires a new valid protocol. Loose behavioral criteria may pass while being useless.
