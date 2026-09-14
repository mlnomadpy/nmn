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


## Plan metadata-matched pairs without running a model

```bash
nmn research native plan-donors --dataset dataset.json --split evaluation \
  --modules b --match-semantics category --max-comparisons 100 --max-pairs 10 \
  --provenance "Declared matching rule before model evaluation" --output plan.json
nmn research native extract plan.json --component pairs --output selected-pairs
```

Use `selected-pairs/data.json` as `donor --pairs`, retaining the same dataset and
`--match-semantics category` in that execution. The Python API is
`nmn.research.donor_planning.plan_donors`. The planner and extraction need no ML
backend. It does not validate module names against a model; execution does.

Within one declared split, pairs are inspected in lexicographic base-ID then
donor-ID order. Self pairs and same-group pairs are excluded. Missing required
semantic fields and unequal values are excluded, not guessed. Every inspected
pair retains its reason and missing/mismatched keys. Exclusion counts use the
first applicable reason in that order. With no matching keys, only identity/group
restrictions apply. Opposite directions are distinct candidate pairs.

The `nmn.donor-plan.v1` record includes dataset identity, complete protocol,
selected pairs, inspected decisions, and total/inspected/uninspected counts.
Either budget can stop inspection; `budget-stopped` means eligible pairs may
remain. `complete` means the finite declared Cartesian population was inspected,
not that any semantic or scientific claim passed. Zero selected pairs are retained
in the plan; donor execution still requires a nonempty pair list.

This policy selects an ID-ordered prefix and can favor earlier base IDs. It is
not randomized sampling, model-based acquisition, causal-scrubbing equivalence,
or a population-coverage guarantee. Keep the plan beside its executed study;
extracting a pair list does not enforce the original dataset identity in later
commands. The ordinary donor executor still checks its supplied dataset, splits,
module names and explicit semantic matching rules.


### Execute a plan with identity checks

Prefer `donor --plan plan.json` over manually extracting pairs when executing a
saved plan. `--plan` and `--pairs` are mutually exclusive. The executor requires
the exact recorded dataset, recomputes the deterministic eligibility decisions,
and rejects modified coverage, pairs or protocol fields that do not reproduce.
It inherits the plan's semantic matching rules; omit `--match-semantics` and
`--allow-cross-split` overrides. Whole-write, `--read-slots`, and `--edge-routes`
execution modes remain available.

The Python entry point is `donor_study_from_plan(model, dataset, plan, ...)`.
A nonempty budget-stopped plan is executable as the recorded partial selection;
it is never relabeled complete. The resulting study embeds the complete
`selection_plan` and its canonical JSON hash. Numerical replay rechecks the plan
before execution and compares its linkage and all donor rows. This checks
consistency, not authenticity or independence: replacing an entire plan and its
dataset is not prevented. Empty plans are retained by planning but cannot execute.
