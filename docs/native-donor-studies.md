# Native donor studies and semantic reference data

`nmn.research.datasets` provides framework-independent study records.
`nmn.torch.studies.donor_study` executes them on `ThreeNeuronYat` or `YatGraph`.
These APIs make the input population, donor access and expected counterfactual
explicit, alongside actual model execution.

```python
from nmn.research.datasets import ResearchSample, ResearchDataset, DonorPair
from nmn.torch import ThreeNeuronYat
from nmn.torch.studies import donor_study

population = ResearchDataset([
    ResearchSample("base", (1., 1.), "evaluation", "entity-a", {"u": 1}),
    ResearchSample("donor", (0., 1.), "evaluation", "entity-b", {"u": 0}),
], name="scalar transfer", provenance="supplied arithmetic labels, version 1")

result = donor_study(
    ThreeNeuronYat.reference(), population,
    [DonorPair("transfer-h", "base", "donor", ("h",), expected={"target": .5})],
    protected_outputs=["protected"],
)
```

Each sample has an immutable content identity through the dataset's SHA256,
a unique ID, finite input vector, split, leakage group and supplied semantic
values. The dataset deep-copies records on ingress and access. A group may not
cross splits. Choose the group according to the claim: entity transfer and
context/template transfer need different grouping units. Group declarations do
not prove independence, and missing real-world dependencies are not discoverable
from these records. JSON round-trip uses `to_dict()` / `from_dict()`.

A donor pair names the base, donor and one or more module outputs to replace.
By default both samples must belong to the same split. Explicit cross-split use
requires `allow_cross_split=True` and is recorded in the protocol. Optional
`match_semantics=["task", ...]` checks that the requested labels exist and agree.
It does not choose donors, implement recursive causal scrubbing or infer an
appropriate semantic equivalence relation. Self-donor controls are allowed and
flagged in the output.

All donor values come from the unchanged network. Joint replacements use that
same fixed donor execution; base descendants then recompute normally. On a
residual graph, this replaces a module write vector, not the complete state slot.
Raw baseline/donor states are retained in the population snapshot; edited traces
and actual donor writes are retained per pair. No cached reference result is
substituted for model execution.

## Separate reference computation

Instead of pair-supplied expected values, pass a callable
`reference(base_sample, donor_sample, module_names)` returning a dictionary of
expected numerical output coordinates, together with `reference_id="name-version"`.
The callback receives dataset records, not model outputs. Its semantics and
version are supplied by the researcher; NMN does not infer its correctness.
Callbacks and pair expectations cannot be mixed in one run. Reference-source
provenance belongs in the dataset provenance/reference ID and experiment record.

The result preserves each expected value, signed/absolute reference error and
signed change to each specified protected output. It supplies no implicit
acceptance tolerance or population guarantee. Use a task contract to interpret
these measurements. Source hashes identify the runner/data validation files;
the nested model snapshot contains model/configuration/input identities.

```bash
python examples/research/native_donor_study.py --output donor-study.json
```

This runnable example includes a donor transfer and a self-donor control against
a separate arithmetic reference, without training. Save returned records with
`nmn.torch.research.save_research_data`. Remaining work includes coordinate/edge
patching, donor resampling policies, semantic alignment search, candidate-selection
history, population risk validation and source-faithful benchmark adapters.

## Patch one receiving module's read

`YatGraph.forward_with_trace(x, read_patches={"b": {"h": values}})` replaces
only `b`'s read of slot `h`. The shared residual state and other readers remain
unchanged. `b` and downstream modules recompute from that patched read. Values
broadcast to the selected coordinate's batch shape and retain tensor gradients.
Write gates/replacements, if supplied, apply after this computation.

The trace distinguishes `b.input_original` from `b.input`. This patches the
receiving slot value, not a single producer's contribution to a residual sum.
It does not implement arbitrary edge/path isolation or recursive scrubbing.

For unchanged-donor values, supply `read_slots={"b": ["h"]}` to `donor_study`,
or pass `--read-slots routes.json` to `nmn research native donor`. The routes file
maps receiving module names to lists of their read slots. Routes must cover
exactly the modules referenced across donor pairs. Omit this option to retain
whole-write replacement. Read-slot mode requires `YatGraph`.

Each pair saves the injected `donor_reads`, original and edited traces, expected
outputs, reference errors and protected changes. In read mode `donor_writes` is
empty, and `protocol.donor_execution` identifies receiving-slot replacement.
Existing split, group, semantic matching and reference-provenance rules apply.

Run the arithmetic example:

```bash
python examples/research/native_read_patches.py --output read-patches.json
nmn research native export read-patches.json --output read-patch-note
```

Two modules share `h`; transferring a zero donor value into only one reader
changes its output from one to zero while the other reader still outputs one.
The self-donor control preserves both outputs. These are finite numerical
observations, not a general protected-edit guarantee.

## Donor contributions at individual residual edges

Use `--edge-routes` instead of `--read-slots` to transfer selected producer writes
at a receiver, preserving other contributions to its read coordinate:

```json
{"b": {"h": ["a"]}}
```

```bash
nmn research native donor --model graph.json --dataset dataset.json \
  --pairs pairs.json --edge-routes edge-routes.json --protected p --output donor-edges.json
nmn research native replay donor-edges.json --output replay.json
```

In this mode each pair's `modules` names receiving modules. Routes must cover
exactly their union. Each receiver maps read slots to nonempty, unique producer
name lists; producers must write that slot in an earlier layer. The Python API
is `donor_study(..., edge_routes=routes)` and requires `YatGraph`. Whole-slot and
edge routes are mutually exclusive.

All donor replacement values come from the unchanged donor execution and remain
fixed during the base replay. At each receiver the graph subtracts the named
producer's current effective base write and adds the saved donor value. Other
producers' writes and other receivers remain unchanged except for downstream
consequences of recomputation. Indirect effects through other writers are not
removed. This is not recursive causal scrubbing.

The existing pair split/semantic-match rules and supplied reference expectations
still apply. Records retain `protocol.edge_routes`, an explicit donor-execution
mode, and `rows[*].donor_edges`, together with pair identities, complete donor/base
population traces, edited traces, reference errors and protected deltas. Numerical
replay recomputes the donor writes and checks every saved row. Self-donor pairs
provide a useful identity comparison; semantic labels and expectations remain
supplied rather than inferred.
