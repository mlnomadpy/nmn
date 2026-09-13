# Budgeted coalition measurements

```bash
nmn research native coalitions --model model.json --dataset dataset.json \
  --modules h p y --max-evaluations 8 --output coalitions.json
```

`nmn.torch.coalitions.coalition_study` supports `ThreeNeuronYat` and `YatGraph`.
Choose any nonempty set of unique module names. Bit `i` of each integer mask
sets module `modules[i]`'s write gate to zero. Unselected modules retain their
background gates. Background defaults to one everywhere; `--background gates.json`
accepts a module-to-finite-scalar mapping. Mask zero is the declared background,
which may differ from the unchanged model snapshot stored for provenance.

The record retains sample IDs, dataset and model identity, module ordering,
background, evaluated masks, every output, changes from mask zero, and costs.
Masks run in ascending integer order. The required evaluation budget bounds
coalition forward calls, not preparation, samples, memory or wall time. With
`m` modules a full lattice requires `2**m` calls. Outputs occupy space proportional
to evaluated masks × samples × outputs. Complete transforms add a factor of `m`
to arithmetic cost. Use small declared module sets for exhaustive analysis.

A complete finite lattice includes subset coefficients and reconstruction
residuals. Their sum over all subsets of a mask reconstructs that mask's response
up to floating-point error. Coefficients depend on the chosen background and
module order. They do not establish global interaction sparsity or bounds over
unqueried inputs/backgrounds.

Budget exhaustion produces `inconclusive`, retains measured responses and leaves
coefficients `null`. Nonfinite arithmetic produces `failed`. Neither condition
is interpreted as zero interaction. Other execution errors fail the command.
Per-coalition internal traces are not retained; the snapshot stores the unchanged
trace and the record defines the gate controls for replay. Timing is host wall
time, including numerical transforms; it is not a matched hardware benchmark.

Use `nmn research native export` for an Obsidian note or `native report` to index
complete, partial and failed records without upgrading their assurance.
