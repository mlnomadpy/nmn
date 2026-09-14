# Bounded real-function range verification

```bash
nmn research native verify-box --model model.json --contract contract.json \
  --max-boxes 100 --output certificate.json
nmn research native check-box certificate.json --output checked.json
nmn research native export certificate.json --output certificate-note
```

A reference-network contract is:

```json
{
  "schema": "nmn.interval-contract.v1",
  "provenance": "Declared arithmetic output ranges v1",
  "input_box": {"u": ["0", "1"], "v": ["0", "1"]},
  "outputs": {"target": ["0", "4"], "protected": ["0", "1"]},
  "controls": {}
}
```

The proposition is: for **every real input in the closed input box**, each named
controlled-model output lies in its declared closed interval. Unnamed outputs are
not constrained. Parameters are the exact binary rationals stored in the snapshot.
Supported operations and the containment proof are in [native enclosures](native-enclosure.md).
Fixed positive epsilon keeps these rational expressions defined on all real
states; no additional hidden-state invariant or runtime-roundoff guarantee is asserted.

The search encloses boxes using exact rational arithmetic. A contained enclosure
closes a leaf. Otherwise it evaluates the exact rational midpoint. Only a point
that violates an output range becomes a counterexample. If the midpoint satisfies
the ranges, the widest coordinate is bisected (ties by coordinate name). The
required budget counts region enclosures; at most one extra midpoint enclosure
is made per processed region. It does not cap runtime or rational integer sizes.

The saved binary partition retains every covered, split, pending and witness node.
Budget exhaustion leaves all remaining boxes as `pending` and returns `inconclusive`.
A loose enclosure alone never establishes a counterexample. Search can return
`certified-under-assumptions`, `counterexample-found` or `inconclusive`; CLI exit
zero means the record was produced, not that the proposition was verified.

`check-box` reconstructs each pair of child boxes from the recorded interior split,
rejects gaps/inconsistent children/unreachable nodes, recomputes covered-leaf bounds,
and independently evaluates witness points inside their leaves. It checks budget
accounting and derives the outcome from checked nodes. It accepts arbitrary valid
split choices rather than rerunning the search strategy. It shares the rational
enclosure primitives with the search; it is not a second arithmetic implementation,
an external proof assistant or peer review.

Why coverage suffices: each split replaces a closed parent box by two closed children
whose union equals the parent (overlap only at the split boundary). Induction over
the checked tree proves the leaves cover the root. The enclosure containment result
then proves the range predicate everywhere when all leaves are covered. An exact
violating point proves failure regardless of pending leaves. Pending leaves without
a witness prove neither outcome. Source identities and the checked certificate hash
identify the evaluated artifacts, not their authorship.

The cancellation fixture needs subdivision even though its real output is zero:
independent interval occurrences lose correlation. Near-boundary or narrow margins
may require many boxes; no convergence-rate or practical completeness claim is made.
Output-range predicates are the supported contract language. Relations between distinct parameter snapshots, tanh layers and floating-runtime certification remain unsupported. The output-difference extension below supports two control settings on one fixed model. Malformed certificates cause an error rather than a checked result.

## Continuous output-difference protection

Including `reference_controls` switches the constrained quantity to
`f_controls(x) - f_reference_controls(x)` with the same stored parameters and
same real input `x`. For example:

```json
{
  "schema": "nmn.interval-contract.v1",
  "provenance": "Declared protected-branch contract",
  "input_box": {"u": ["0", "1"], "v": ["0", "1"]},
  "controls": {"h": {"gate": "0"}},
  "reference_controls": {},
  "outputs": {"protected": ["0", "0"]}
}
```

This certifies zero change in the reference network's protected output when `h`
is disabled. The certificate and checked report explicitly label the quantity
as `output-difference`. Omitting `reference_controls` retains absolute-output
range semantics. A reference control may itself be nontrivial.

`enclose_native_difference` first encloses both executions. Subtracting their
intervals contains their pointwise difference, although it discards correlation.
For an output whose conservative dependency set contains no changed module,
it instead returns exactly `[0,0]`. This follows by induction on fixed routing:
unchanged inputs and unchanged ancestor computations give the same output under
both controls. Residual dependencies include every overlapping writer; the
criterion does not rely on sampled equality, parameter cancellation or learned
semantic independence. Unsupported native operations remain rejected.

Thus unaffected branches can satisfy zero-tolerance protection on a whole box.
Affected branches may still produce loose difference bounds and inconclusive
searches. Distinct parameter snapshots, read-slot/path interventions, general
relations beyond output differences and floating-runtime error remain unsupported.
