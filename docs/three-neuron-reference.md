# Three-neuron exact reference

This research CLI slice supports a fixed three-neuron topology with configurable
parameters. It is not a general architecture loader. It requires only Python's standard library. No model is
trained, downloaded or fit. Parameters and meanings are supplied by construction.

## Model and contract

With the unbiased yat kernel and epsilon=1:

- Layer 1 simultaneously computes h = gamma k(1,u) and p = k(1,v).
- Optional native replacement of h occurs after layer 1.
- Layer 2 recomputes y = k((1,1),(h,v)).
- Readouts are target y and protected p. The deliberate failure variant uses p+y.

The finite contract enumerates u,v in {0,1/4,1/2,3/4,1}, applies the same gate
change 1 → 0 to every input, and requires p unchanged everywhere. The target
condition is scoped to u=v=1: y decreases by exactly 7/2. It is not required to
change everywhere. All 25 pairs are checked with exact rational arithmetic.

At (1,1), (h,p,y) changes from (1,1,4) to (0,1,1/2). The failure readout p+y
changes from 5 to 3/2. Protection follows from routing, not a special property
of yat. This is not evidence of learned semantics or a kernel-specific advantage.

## Run

From an installed version containing this feature:

```bash
nmn research inspect
nmn research trace --u 1 --v 1 --gate 0
nmn research trace --u 1 --v 1 --gate 0 --replace-h 1
nmn research compare --u 1 --v 1 --gate 1/2
nmn research verify
nmn research verify --leaky  # expected exit 1: protection contract fails
nmn research demo --output runs/three-neuron
nmn research reproduce runs/three-neuron
nmn research export runs/three-neuron --output /path/to/vault/Generated/NMN-reference
```

A checkout can use `PYTHONPATH=src python -m nmn` instead of `nmn`.
Trace values accept rational/decimal strings in [0,1]. Gate and replacement
order is explicit in each trace. Output directories must be new: existing notes
and bundles are never overwritten. Export creates a standalone report plus its
linked evidence and manifest; it does not scan or modify other vault notes.

Exit codes: 0 successful command or passing selected contract, 1 failed selected
contract, 2 invalid input/artifact or filesystem error. Demo returns 0 when it
successfully records both expected outcomes, including the intentional failure.

## Evidence and limits

Bundles contain evidence.json (contract, architecture, all baseline/edited traces
and counts), Report.md and manifest.json (artifact hashes, evaluator source hash,
and Python version). Replay checks exact inventory and hashes, then recomputes
the fixed experiment. It never runs bundled code. Hashes detect accidental
changes but are not signatures or proof of who created a bundle. A changed
evaluator is rejected, and different Python versions are reported explicitly.
Replay uses the same evaluator; it is not independent formal verification.

Enumeration certifies only the declared finite domain in exact arithmetic. It
does not certify floating-point backends or a continuous domain. The architectural
reason p is independent of the gate can be inspected in the explicit equations,
but the CLI is not a general structural proof engine.

General YAML architecture schemas, arbitrary checkpoints, learned semantics,
interval certification, training and backend adapters remain separate roadmap
items. See https://github.com/mlnomadpy/nmn/issues/28.

## Compare a chosen edit

`nmn research compare` uses the unedited gate-one model as baseline. It returns
both complete traces, signed edited-minus-baseline deltas, and separate flags
for target change, protected-p equality and leaky-readout equality. Its default
action disables h; pass `--gate` or `--replace-h` for a different native action.
At (1,1), gate 1/2 produces y=9/5, a target delta of -11/5, while p stays 1.

A comparison is pointwise evidence, not the exhaustive contract implemented by
`verify`. A changed target is not automatically a successful target intervention.
The command exits 0 when the comparison completes, including when collateral
effects are reported, and exits 2 on invalid input.

## Configurable three-neuron models

Create the default JSON model and validate or execute it:

```bash
nmn research model > model.json
nmn research model model.json
nmn research inspect --model model.json
nmn research trace --model model.json --u 1 --v 1 --gate 0
nmn research compare --model model.json --gate 1/2
nmn research verify --model model.json
```

The schema `nmn.three-neuron-model.v1` requires exactly `schema`, `epsilon`,
`neurons` and `protected_leak`. Each of h, p and y requires a center array and
a coefficient. Their input dimensions remain 1, 1 and 2. All numbers must be
rational strings, epsilon must be positive, and duplicate/unknown keys are
rejected. Model files are limited to 64 KiB and individual numbers to 64
characters. Signed centers and coefficients are supported; intermediate states
need not lie in [0,1]. Inputs, gates and optional h replacements remain in [0,1].

The protected readout is `p + protected_leak*y`. Set `protected_leak` to `"1"`
to expose the downstream protection failure. Runnable examples live in
`examples/research/three-neuron.json` and `three-neuron-leaky.json`.

Configured verification enumerates the same 25 inputs for the shared gate edit
1 → 0, testing exact equality of the configured protected output. It records
all traces and the first counterexample, with a hash of normalized model
parameters. There is **no target-success requirement** in this configured
contract. Target-change counts are descriptive. Exit 1 denotes a protection
failure; exit 2 denotes invalid configuration. `--leaky` cannot be combined
with `--model`: specify the readout in the model file instead.

Configured verification emits standalone JSON. Use `demo --model` to create a
portable bundle with the model, evidence and report:

```bash
nmn research demo --model model.json --output runs/configured
nmn research reproduce runs/configured
nmn research export runs/configured --output /path/to/vault/Generated/Configured-NMN
```

The configured schema `nmn.configured-bundle.v1` contains model.json,
evidence.json, Report.md and manifest.json. The manifest records Python version,
hashes of all three artifacts and the model/reference/bundle implementation
files. Replay checks the schema, file inventory and hashes, validates the saved
model, and recomputes every case and report. Export copies the validated bytes,
preserving original provenance. No source model file outside the bundle is needed.

Failed contracts are valid evidence: successful replay returns exit 0 with
`verification_status: counterexample-found` when it reproduces a recorded
failure. `verify --model` still returns exit 1 for that protection failure.
This distinction separates command/reproduction success from scientific outcome.

Output directories must not exist. The configured writer reserves a new
directory exclusively and writes its completion manifest last; failed writes
are cleaned up. Do not consume a directory without a valid manifest. Readers
reject missing files, symlinked artifacts, duplicate JSON keys, unexpected
manifest paths, source-version mismatch and changed/rehashed false evidence.
Hashes are not signatures. An attacker replacing all data and recomputing a
consistent experiment can create a different valid bundle; this is not proof
of the original author's identity.

Existing fixed-reference bundles remain supported by schema dispatch. Changing
evaluator source requires its matching implementation for strict replay; this
release does not provide automatic source installation or schema migration.

## Explicit finite intervention contracts

```bash
nmn research contract > contract.json
nmn research contract contract.json
nmn research verify --model model.json --contract contract.json
nmn research verify --model model.json --contract contract.json --max-cases 10
```

The last command returns exit **3 (inconclusive)** if ten checked cases contain
no violation and cases remain. A real counterexample already found still
justifies exit 1, even if other cases remain unchecked. Reports retain checked
and total counts and never label incomplete coverage as a certificate.

The `nmn.finite-intervention-contract.v1` schema defines:

- Separate finite u/v grids, each containing 1–64 unique rational values in [0,1].
- A shared baseline and edited gate in [0,1].
- Nonnegative absolute protected-output tolerance (inclusive boundary).
- Optional `target`: a declared grid witness (u,v) and nonnegative
  `minimum_decrease`, interpreted as baseline y minus edited y. Set target to
  null to request only protection.

The default contract requires exact protection everywhere on the quarter grid
and a target decrease of at least 7/2 at (1,1). Target success is required only
at that witness; other target changes remain unconstrained. Rational values
must be strings without exponent notation. Grids are sorted canonically;
numerically duplicate values are rejected. Input files are capped at 64 KiB.

Case budgets must be in [1,4096]. Case order is lexicographic in normalized u/v.
The output contains the normalized model and contract, hashes, all checked
traces, first failure, coverage and a distinct target status (not requested,
not evaluated, passed or failed). Case budgets change execution coverage, not
the identity of the declared contract. `--max-cases` requires `--contract`.
Without `--model`, explicit-contract verification uses the default model.

Explicit-contract evidence currently emits standalone JSON. The older bundle
schemas still cover their own prescribed quarter-grid gate-1-to-0 contracts;
they do not store or replay these custom contracts. No custom-contract bundle
support is implied by `demo` or `export`.
