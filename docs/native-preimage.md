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
