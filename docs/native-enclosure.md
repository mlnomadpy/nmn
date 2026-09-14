# Exact rational enclosures of native real-valued functions

```bash
nmn research native init --output model.json
nmn research native enclose --model model.json --box box.json --output bounds.json
nmn research native export bounds.json --output bounds-note
```

For the reference network, `box.json` can be
`{"u": ["0", "1"], "v": ["0", "1"]}`. Box keys must match input names.
Use rational strings for intended exact decimal/fraction endpoints. JSON floats
and stored model weights are interpreted as their exact binary rational values,
not nearby ideal decimals. `--controls controls.json` accepts module gates or
constant write replacements, scalar or one value per write coordinate.

`nmn.torch.enclosure.enclose_native` supports reconstructed `ThreeNeuronYat` and
`YatGraph` snapshots with fixed unbiased Yat or reciprocal-distance IMQ expansions,
signed coefficients, fixed positive epsilon and residual addition. Tanh, parameter
edits, receiving-slot patches and input-dependent controls are rejected or absent
from the supported interface. Snapshot construction validates shapes and routing.

The `nmn.rational-enclosure.v1` record contains rational-string output bounds,
intermediate interval traces, denominator bounds, controls and source identities.
It is an enclosure of the real-arithmetic expression defined by the stored
parameters. It does **not** bound PyTorch roundoff, prove a contract, search the
input domain or independently check a branch-cover certificate.

## Enclosure claim and derivation

For every real input in the supplied closed box, every supported intermediate
quantity and output lies in its recorded interval. Assume finite rational
parameters, positive epsilon, fixed routing and the supported constant controls.

Addition uses endpoint sums; multiplication takes the extrema of the four endpoint
products. Squaring has lower endpoint zero when the interval crosses zero, otherwise
the smaller endpoint square; its upper endpoint is the larger square. Each formula
contains all possible real results on its operand intervals. A strictly positive
interval `[a,b]` has reciprocal `[1/b,1/a]`. All endpoints are computed using
Python `Fraction`, so these operations introduce no floating rounding error.

For a Yat section, enclose the dot product and square it. Enclose each squared
coordinate difference, add them and epsilon, then multiply the numerator enclosure
by the denominator's positive reciprocal. The denominator lower endpoint is at
least epsilon, so division is defined throughout the box. IMQ uses numerator one.
Signed coefficient sums and constant controls preserve containment by the same
rules. Induction over the fixed graph layers establishes containment: all modules
in a layer read the same enclosed incoming state before residual writes are added.

This proof concerns inclusion, not tightness. Repeated use of a variable loses
correlation, so bounds can be loose. A bound that crosses a desired threshold is
not a pointwise counterexample. The [range-contract checker](native-interval-contract.md) adds bounded subdivision and search-independent partition checking for closed output ranges. General relational predicates and runtime-roundoff coverage remain outside this subset.

Validation includes exact reference-point evaluation using a separate rational
kernel implementation, zero-crossing arithmetic, exact binary-float conversion,
positive denominators and an IMQ→Yat residual example with output `1/5`. These
checks exercise the implementation; they are not peer review or a runtime-roundoff
certificate. Large rational denominators and wide/deep graphs can be expensive.
