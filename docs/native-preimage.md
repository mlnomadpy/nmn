# Native finite-bank preimage search

`nmn.torch.preimage.search_preimages` searches bounded input coordinates for a
supplied feature target while holding an existing `YatExpansion` fixed. It
minimizes mean squared residual between unweighted kernel-bank evaluations and
the requested features, using projected Adam with explicit step and time budgets.
Initialization is eligible for selection; the lowest aggregate loss wins and
an earlier iterate wins ties. Raw residuals remain available per sample.

This addresses the vault's preimage interface gap. It does not implement the
kernelized-erasure paper's learned mapper, Nyström projection or minimax algorithm.
A finite vector of bank evaluations is not an arbitrary RKHS representation.
A large residual does not prove nonexistence, and a small residual does not
establish erasure or downstream protection. Native execution must be measured
separately after mapping the selected coordinates to a valid intervention.

```python
from nmn.torch.preimage import search_preimages

record = search_preimages(
    module, inputs, target_features,
    lower=lower_bounds, upper=upper_bounds,
    sample_ids=sample_ids, provenance="origin of supplied target features",
    max_steps=100, max_seconds=10.0, learning_rate=0.05,
)
```

Targets have shape `(samples, centers)` and inputs `(samples, input_features)`.
Finite ordered bounds must contain the initial inputs; bounds may broadcast to
that exact shape. Model parameters and existing parameter-gradient buffers are
unchanged. Inputs are optimized individually within a jointly averaged objective;
no generalizing map is learned. The record retains the fixed bank, content hash,
precision, bounds, initial/selected inputs, target/actual features, per-sample
residuals and all finite iteration losses. Time is checked between iterations;
initialization and final reporting are outside the stopping budget.

Run `python examples/research/native_preimage.py /tmp/native-preimage` for an
end-to-end example: solve for the input `(h,v)` of Y while fixing v, replace h in
the actual three-neuron network, then save native target/protected measurements.
The resulting search and execution records support Obsidian export and dashboard
ingestion. Native replay supports the execution record; optimizer replay is not
implemented for the outer search record.

## Dataset-linked CLI studies

```bash
nmn research native preimage --model model.json --dataset dataset.json --split evaluation --module y --targets targets.json --bounds bounds.json --max-steps 100 --max-seconds 10 --learning-rate 0.1 --provenance "supplied finite-bank target" --output preimage.json
nmn research native export preimage.json --output obsidian-preimage
nmn research native report preimage.json --output preimage-dashboard
```

`targets.json` maps exactly the selected sample IDs to feature vectors, with one
entry per center of the chosen module. `bounds.json` contains exactly `lower` and
`upper`, each broadcastable to the selected module-input matrix. Bounds constrain
module coordinates, not necessarily raw model inputs. Initial coordinates come
from the model's actual baseline trace. Omit `--split` to optimize all samples.

The `nmn.preimage-study.v1` record includes the original model execution, dataset
and content identity, chosen module, target vectors, complete bounded search and
`proposed_inputs` keyed by sample ID. The API is
`preimage_study(model, dataset, module_name=..., targets=..., lower=..., upper=...,
provenance=..., max_steps=..., max_seconds=..., learning_rate=..., split=...)`.

This command supports strict YatExpansion modules inside native PyTorch graphs
and the three-neuron architecture. IMQ/tanh modules are rejected. It does not
install the proposed coordinates as an intervention: that mapping depends on
which module reads, inputs or state slots the experiment author intends to edit.
Every selected sample participates in optimization; there is no held-out
validation population or trained mapper in this study. The outer search is not
yet supported by the numerical replay command.
