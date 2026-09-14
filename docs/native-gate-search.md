# Native gate proposal search

`nmn.torch.gate_search.search_gates` and `native search-gates` optimize gate values
in [0,1] while leaving the native network parameters unchanged. They supply
candidate-generation data for KG05 protected editing, followed by the existing
[finite selector](native-selection.md). This is a local numerical search, not a
global feasibility solver or protection certificate.

```bash
nmn research native search-gates --model model.json --dataset dataset.json \
  --targets targets.json --config search.json --output search-result.json
nmn research native extract search-result.json --component selection --output selected-study
nmn research native replay selected-study/data.json --output replay.json
nmn research native export search-result.json --output vault/search
```

The config supplies `modules`, `target_outputs`, `protected_outputs`,
`protection_tolerance`, `provenance`, `max_steps`, `max_seconds`, `learning_rate`,
and `protection_weight`. Optional split names default to `tuning` and `validation`.
See [the reference config](../examples/research/native-gate-search.json). Targets
use the selector's sample-ID → output-name → finite value format and cover exactly
both populations. Target and protected outputs must be disjoint.

Starting from all-one gates, projected Adam minimizes selection-population target
MSE plus `protection_weight` times protected-output-change MSE. Each update is
clamped to [0,1]. The penalty guides proposals; it does not determine final
feasibility. The initial candidate and each finite iterate are retained, for at
most `max_steps + 1` candidates. Proposal losses and elapsed times are saved.
A nonfinite proposal loss/gradient stops generation, retaining previous candidates;
if none can be generated the command fails. A zero gradient is retained without
inventing a direction or a feasibility conclusion.

`max_seconds` is checked between proposal iterations after the first candidate.
It is a soft generation budget, not a timeout for an individual forward, backward,
optimizer step, initial baseline, or the subsequent selector/validation. Generation
status distinguishes step completion, time stopping and nonfinite stopping.

After generation, the existing selector reexecutes every retained candidate,
checks per-coordinate absolute protection changes against `protection_tolerance`,
and minimizes target MSE among measured feasible candidates. It freezes the choice
before validation; validation does not change the winner. No target-MSE threshold
is implied by selection success. A candidate can preserve the protected output
and still have a large target error or fail to generalize.

The `nmn.gate-search.v1` record embeds candidates, proposal history, protocol,
model/data identities, and the complete `selection` record with validation and
its ledger. Extracting `selection` produces an ordinary replayable study. Replaying
that component checks measured candidates and the frozen choice, **not** the
proposal optimizer or its timing history. The outer search has no replay adapter.

The arithmetic fixture lowers selection target error without changing native
parameters or their gradient buffers. Changing only validation targets leaves
proposals and the selected candidate unchanged. These checks establish the data
flow of this example, not optimality, semantic meaning, statistical independence,
or uniform protection. Repeated evaluation-guided runs can still leak information.
