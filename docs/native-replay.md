# Replay native research measurements

```bash
nmn research native replay observations.json --atol 1e-10 --rtol 1e-8 \
  --output replay.json
nmn research native export replay.json --output replay-note
```

Unlike `verify-export`, this command executes the saved model. It supports
native observation collections, donor studies (including read-slot patches),
classification protection, coalition, supplied-semantic, edit-selection, suffix-state, response-space, measured benchmark, gate-path and kernel diagnostic records. Other schemas are rejected.
CPU float32/float64 execution follows the saved dtype. No training, pickle
loading or executable reference callbacks are invoked.

Replay checks embedded model identities and, for composite studies, dataset
identity. It recomputes the declared measurements and compares their structure,
values and statuses. Observation records compare traces, geometry and recorded
derivatives. Composite studies compare their result tables and the unchanged
population inputs, IDs and outputs. Gate-path replay also compares derivatives, predictions, residuals and deterministic call counts; the report lists ignored source-hash and timing paths. Kernel replay reconstructs the selected module inputs and repeats layer/sensor measurements at the saved noise radius. Runtime/source versions, metadata and timing
are not equality targets. Older trace schemas can produce a mismatch when the
current implementation exposes additional fields; these are not silently ignored.

Numbers agree when `abs(saved-replayed) <= atol + rtol*abs(saved)`. Both tolerances
must be finite and nonnegative. Choose them for the original precision and study;
the defaults do not promise cross-hardware bitwise equivalence. Nonfinite saved
measurements cannot count as a successful numerical match.

The `nmn.native-replay.v1` report saves the canonical source-record SHA256, fields
compared, tolerances, mismatch paths, maximum finite numeric difference and full
new execution. CLI exit 0 means matched, 1 means a written mismatch record, and 2
means invalid input or execution failure. The original record is never modified.

Donor reference expectations are read from the saved rows. This recomputes the
model's discrepancy from those numbers, not the semantic reference calculation.
Replaying an inconclusive coalition record repeats its declared budget and can
match without completing the lattice. Agreement verifies reproducibility of saved
measurements; it does not certify a mathematical or population claim.

Path records without a separate dataset can replay their saved snapshot inputs and IDs. If a path/diagnostic dataset is present, replay checks its identity when recorded and verifies selected inputs after conversion to the saved precision. Training records remain unsupported; replay never silently launches optimization.

Edit-selection replay recomputes the declared candidate budget and split sequence. It checks the selected ID, failed/unexecuted outcomes, ledger freeze/event ordering, per-candidate measurements and validation inputs/traces. Validation still cannot select a new winner.

Response-space replay compares projection operators, singular values, responses and reconstruction residuals. It ignores raw basis coordinates, which change under harmless sign/rotation conventions. A rank cutoff through a repeated singular-value group can still produce a different subspace and a mismatch.

Benchmark replay requires at least two measured methods with saved snapshots. It restores all methods and compares parameter counts, predictions, deltas and target errors using one forward per condition. Timing, warmup/repetition costs and diagnostic snapshots are explicitly excluded. Historical failed rows without reconstructible snapshots are rejected, never reported as reproduced.
