# Three-neuron exact reference

This first research CLI slice implements one fixed model, not a general
architecture loader. It requires only Python's standard library. No model is
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
