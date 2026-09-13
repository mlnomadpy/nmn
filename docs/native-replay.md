# Replay native research measurements

```bash
nmn research native replay observations.json --atol 1e-10 --rtol 1e-8 \
  --output replay.json
nmn research native export replay.json --output replay-note
```

Unlike `verify-export`, this command executes the saved model. It supports
native observation collections, donor studies (including read-slot patches),
classification protection and coalition records. Other schemas are rejected.
CPU float32/float64 execution follows the saved dtype. No training, pickle
loading or executable reference callbacks are invoked.

Replay checks embedded model identities and, for composite studies, dataset
identity. It recomputes the declared measurements and compares their structure,
values and statuses. Observation records compare traces, geometry and recorded
derivatives. Composite studies compare their result tables and the unchanged
population inputs, IDs and outputs. Runtime/source versions, metadata and timing
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
