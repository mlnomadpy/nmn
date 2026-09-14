# Select an edit, freeze it, then validate

```bash
nmn research native init --output model.json
nmn research native select --model model.json --dataset examples/research/selection-dataset.json \
  --candidates examples/research/selection-candidates.json --targets examples/research/selection-targets.json \
  --target-outputs target --protected-outputs protected \
  --protection-tolerance 0 --max-candidates 20 \
  --provenance 'target labels and protocol v1' --output selection.json
```

`nmn.torch.selection.select_edit` implements the same protocol. Candidates use the
named edit JSON format: `{"identity": {}, "remove-h": {"h": {"gate": 0}}}`.
Targets map sample IDs to values for exactly the declared target outputs. They
must cover exactly the nonempty selection and validation populations. Defaults
are `tuning` and `validation`; override with `--selection-split` and
`--validation-split`. Dataset group/split checks remain in force.

Candidates run in lexicographic ID order up to the required budget. Selection
feasibility requires every protected output change on selection examples to stay
within the supplied absolute tolerance. Among feasible measured candidates, choose
minimum target mean squared error, breaking ties by ID. Empty protected outputs
mean no protection constraint. Invalid execution and nonfinite results are retained
as failed candidates; unexecuted candidates are listed, not presumed infeasible.

The ledger freezes the winner before validation forward execution. Only the frozen
candidate is evaluated on validation; its result cannot change the winner. If no
measured candidate is feasible, validation is not executed. The record preserves
separate selection and validation snapshots, predictions, traces, per-example
errors/protection changes, candidate controls, labels and ledger events. The
`selected` status means a choice was made, not that validation succeeded.

The budget bounds attempted candidate forwards on selection, not diagnostic
preparation, samples, memory or time. Validation adds an unchanged preparation
and one candidate attempt when a winner exists. No optimization or training is
performed. This is finite empirical selection; it does not establish independence,
population risk, semantic label validity or a multiple-comparison guarantee.

The shipped arithmetic fixture deliberately gives different targets to the two splits. The selected edit remains frozen even though validation favors another candidate. Selection records support Obsidian export and `native replay`. Replay repeats the declared selection and frozen validation, checking the winner, measured candidate set, ledger event order and validation observations. A matched replay does not mean the selected edit passed validation.
