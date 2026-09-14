# Supplied semantic references and counterfactual correspondence

`native semantics` compares a finite reference table with actual whole-write
donor interventions. Ordinary predictions and counterfactual agreement are
reported separately. The machine-readable `nmn.semantic-study.v1` record retains
full donor traces and all supplied assumptions.

```bash
python examples/research/native_semantics.py --output semantic-fixture
nmn research native semantics --model semantic-fixture/model.json \
  --dataset semantic-fixture/dataset.json --reference semantic-fixture/reference.json \
  --correspondence semantic-fixture/correspondence.json --output semantic-study.json
nmn research native replay semantic-study.json --output semantic-replay.json
nmn research native export semantic-study.json --output semantic-note
```

The arithmetic fixture copies two Boolean-valued coordinates. Ordinary predictions
are correct under both the correct and deliberately swapped module labels; only
the counterfactual check rejects the swapped correspondence. This is an interface
fixture, not a selected research architecture or evidence of discovered semantics.

`TabulatedReference` in `nmn.research.semantics` is backend-independent. Its
`nmn.semantic-reference.v1` JSON declares `reference_id`, `provenance`, `variables`,
a `baseline` map of sample IDs to named output values, and `cases`. Each case names
`case_id`, `base_id`, `donor_id`, transferred `variables` and expected `outputs`.
Only listed outputs and contexts are tested. Baseline coverage counts distinguish
touched samples from all supplied reference rows; unused rows are not evaluated.

A correspondence declares `mapping` (every semantic variable to one or more
native module names), `origin` (`supplied`, `supervised`, or `inferred`),
`provenance`, `anchors` and `ambiguities` lists. Origin is retained as a caller
claim, never established by this command. Counterfactual modules are the union
of mapped modules for the case's variables. Each pair must use the same dataset
split. No module alignment or reference callback is loaded as executable JSON.

Absolute per-output error must be at most `--tolerance` (default `1e-8`) to count
as finite observed agreement. This is not statistical validation, semantic
identification or a unique-mechanism claim. Reference labels themselves may be
incorrect. Symbolic reference evaluators, recursive scrubbing, read-slot semantic
correspondence and automatically inferred alignment remain unsupported.
