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
