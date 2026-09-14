# Backend-free graph schema and validation

`nmn research architecture` prints a JSON Schema for the existing
`nmn.torch.YatGraph` configuration format. Supply a JSON file to validate and
normalize topology without importing a deep-learning backend:

```bash
nmn research architecture > graph-schema.json
nmn research architecture graph.json > validation.json
```

The `nmn.architecture-validation.v1` report contains normalized `configuration`,
a deterministic configuration hash, module dimensions, and encoder/layer/readout
semantics. The configuration field can be saved as a graph JSON and supplied to
`nmn research native init --graph graph.json --output model.json`.
The validation report itself is not an initialized model or an `--graph` input.

Validation checks unique declared slots, input/output membership, nonempty layers,
globally unique module identifiers, read/write membership, positive integer widths,
finite positive epsilon and supported Yat/IMQ/tanh/linear families. Optional module
fields receive explicit defaults. Unknown fields and implicit sequential update
semantics are rejected with a field path. JSON is parsed as data, never Python.
Existing constructor APIs remain unchanged; this validator is an explicit preflight.

All modules in a layer read the same incoming state; overlapping writes add.
Inputs populate named slots and other slots start at zero. Readout selects named
slots. No extra encoder or learned readout is inferred. Epsilon is required to be
positive in this strict research schema even for families whose computation does
not use it. Tanh modules have additional hidden biases, recorded in dimensions.

JSON Schema covers shape and local numeric constraints. Global module uniqueness,
slot references, finite arithmetic and Python identifier rules require the NMN
validator. `valid-topology` does not check checkpoint parameters, dtype/device,
backend-reserved module attribute names, native gate ranges or a scientific
contract. Backend-specific construction still performs its own checks. This
schema does not establish task competence, interpretability or protected behavior.

The validation report supports `nmn research native export` and `report` without
an ML backend. Export checks its contents against recomputed topology validation;
the dashboard labels it as an architecture definition with no numerical execution.

`native extract validation.json --component configuration --output graph-input`
writes a `data.json` directly consumable by `native init --graph`.
