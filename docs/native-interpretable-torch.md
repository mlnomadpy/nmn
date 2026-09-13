# Native interpretable ⵟ networks in PyTorch

`nmn.torch` now provides actual trainable neural modules using `YatNMN`:

- `YatExpansion(in_features, out_features, num_centers)` learns prototype centers
  and signed output coefficients. `contributions(x)` exposes each center's
  contribution, with shape `(..., out_features, num_centers)`.
- `ThreeNeuronYat(num_centers=1)` connects three expansions through fixed, named
  paths: `h = H(u)`, `p = P(v)`, `y = Y(h, v)`. It returns `(y, p)`.
- `Intervention(gate=..., replacement=...)` scales or replaces a named state,
  then recomputes downstream modules during the same forward pass.

```python
import torch
from nmn.torch import Intervention, ThreeNeuronYat

model = ThreeNeuronYat.reference()
x = torch.tensor([[1., 1.]])
output, trace = model.forward_with_trace(x)   # [[4., 1.]]
gated = model(x, {"h": Intervention(gate=0)}) # [[0.5, 1.]]
restored = model(x, {"h": Intervention(replacement=trace["h"])})

# Normal torch parameters, checkpoints, device migration, and autograd.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = (output[:, 0] - 2.).square().mean()
loss.backward()
# optimizer.step() updates the centers and coefficients when training is desired.
```

The minimum network has seven trainable scalar parameters. Increase
`num_centers` to expand each module's capacity; routing stays the same. Default
construction uses random centers; `reference()` sets centers and coefficients
all to one with epsilon one. Inputs can have any leading batch dimensions.
Use floating tensors; for per-example gates or replacements use `(batch, 1)`.
A replacement takes precedence over its gate. Tensor controls preserve autograd.

The trace contains `h`, `p`, `y`, plus `.input`, `.raw`, and `.contributions` for
all three modules. Contributions reconstruct the raw output before intervention.
Trace tensors remain attached to the graph; detach them when retaining reports.

Changing `h` cannot change `p` because the protected path reads only `v`. This
is a routing property for fixed parameters, not a semantic guarantee, a security
claim, or protection against subsequent parameter updates. Named states have
roles supplied by the designer. This release does not discover semantic roles,
certify continuous input regions, or supply arbitrary graph architectures.

Run `python examples/research/native_three_neuron.py` with `nmn[torch]` installed
for forward, gate, restore and gradient output. Save `model.state_dict()` with
`torch.save`; restore into a model constructed with the same center count and
epsilon. Epsilon and routing are constructor configuration, not learned weights.

## Research data API

```python
from nmn.torch.research import (
    collect_research_data, save_research_data, expansion_geometry,
    input_jacobian, gate_derivatives, intervention_table,
    coalition_effects, protection_metrics, yat_gram,
)

data = collect_research_data(
    model, x, sample_ids=["sample-1"],
    edits={"remove_h": {"h": Intervention(gate=0)}},
    metadata={"split": "diagnostic", "semantic_status": "supplied by design"},
)
save_research_data(data, "observations.json")  # refuses overwrite
```

The collector supports `ThreeNeuronYat` and `YatGraph`. Geometry works for any
`YatExpansion`; the standalone Jacobian helper supports sample-independent
PyTorch modules. Other NMN architectures and backends need explicit adapters.
This is not yet a universal NMN model recorder.

| API | Data provided | Scope |
|---|---|---|
| `expansion_geometry` | centers, coefficients, dots, squared distances, kernel values, center Gram, eigenvalues, numerical rank/condition, RKHS inner products | Fixed unbiased shared-epsilon module; float64 diagnostics |
| `input_jacobian` | per-example output/input derivatives | Singleton execution; no batch-coupled guarantee |
| `gate_derivatives` | output derivatives and full mixed Hessian in h,p,y gates | Local shared module gates, no finite-path bound |
| `intervention_table` | baseline/edited outputs, signed and absolute effects, raw/effective traces | Actual independent replays of caller-specified edits |
| `coalition_effects` | all eight deletion responses and subset coefficients | Exhaustive three-state family; no sparse recovery claim |
| `protection_metrics` | per-example correctness, break/fix/disagreement, both accuracies, conditional damage | Caller-supplied class labels; run separately per declared stratum |
| `collect_research_data` | parameters, configuration, inputs/IDs, controls, traces, geometry, derivatives, runtime and source hashes | Small finite input banks; JSON-ready detached snapshot |

`save_research_data` writes strict JSON. Singular condition numbers are encoded
as the string `"infinity"`; numerical NaN/negative infinity, if encountered, are
explicit strings, never silently omitted. Consumers must treat nonfinite
observations as unresolved diagnostics. The snapshot model hash covers
configuration and parameters; source hashes identify the three main implementation
files. It is neither a signature nor a complete environment lock. Collection
cost includes diagnostics and serialization preparation; it is not a matched
performance benchmark. Full Gram matrices and Hessians are optional research
costs: use individual APIs or `derivatives=False` when appropriate.

Run `python examples/research/collect_native_data.py --output observations.json`
to collect the reference grid. Dataset IDs, source/donor relationships, labels,
semantic reference outputs, population/stratum definitions, splits and selection
history must come from the experiment. A package must not invent these from
weights or assume the diagnostic grid is a held-out validation population.

The vault-facing implementation matrix is in
[research data requirements](research-data-requirements.md).

## General explicit-state graphs

`YatGraph` implements the vault's residual update with fixed coordinate read/write
maps. It supports arbitrary state width, layer depth, module count and per-module
center counts/epsilon. Modules within a layer read the same incoming state;
all writes add simultaneously. Two writes to the same slot add rather than
silently overwrite. The encoder places inputs in named slots and initializes
other slots to zero; readout selects slots without a hidden learned head.

```python
from nmn.torch import YatGraph, YatModuleSpec

model = YatGraph(
    slots=["u", "v", "h", "p", "y"],
    input_names=["u", "v"], output_names=["y", "p"],
    layers=[
        [YatModuleSpec("hidden", ["u"], ["h"], num_centers=8),
         YatModuleSpec("protected", ["v"], ["p"], num_centers=4)],
        [YatModuleSpec("target", ["h", "v"], ["y"], num_centers=8)],
    ],
)
outputs, trace = model.forward_with_trace(x)
edited = model(x, {"hidden": Intervention(gate=0)})
print(model.dependencies())
```

An intervention addresses a **module write**, not a whole state coordinate:
replacing a write does not erase the previous residual state or other writes to
the same slot. `trace["state.0"]` is the encoded input; subsequent state snapshots
follow each complete layer. Module traces use the same keys as `ThreeNeuronYat`.
`dependencies()` reports conservative structural input/module ancestors of each
output, including residual paths; it makes no semantic or numeric pruning claim.

`configuration()` exports the complete graph specification, including per-module
epsilon. Reconstruct with `YatGraph.from_configuration(config)` before loading
weights. The graph's `state_dict()` includes configuration and rejects mismatched
routing on load. These are fixed coordinate maps; learned routing, normalization,
attention and arbitrary user-defined block types are not implemented here.

`collect_research_data`, `intervention_table`, `gate_derivatives` and
`input_jacobian` support these graphs. Gate derivatives now have shapes
`(N, outputs, modules)` and `(N, outputs, modules, modules)`; full Hessian cost grows
with graph size. The exhaustive `coalition_effects` helper remains specific to
`ThreeNeuronYat`. Run `examples/research/native_graph.py --output graph.json` for
an executable graph and complete observation snapshot.


[Native donor studies](native-donor-studies.md) add explicit populations, split/group
checks, supplied semantic references and actual donor replacement protocols.
