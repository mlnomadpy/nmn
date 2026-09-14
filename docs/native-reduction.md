# State summaries and reduced dynamics

`nmn.torch.reduction.reduction_study` measures whether a supplied affine summary
preserves a graph's next state and downstream outputs. This supplies observations
for KG02 (closure through depth) and predictive-state/reduction research. It does
not learn a summary, establish invariance or reproduce a literature theorem.

```bash
nmn research native reduce --model graph.json --dataset dataset.json \
  --maps maps.json --start-layer 1 --split evaluation \
  --provenance "Supplied maps; selected before this evaluation" --output reduction.json
nmn research native replay reduction.json --output reduction-replay.json
nmn research native export reduction.json --output vault/reduction
```

The model must be a native `YatGraph`. Boundary `k` is `state.k` before layer `k`;
`start-layer` must have a successor, so the final readout boundary is not valid.
The command executes the unchanged model with no optimizer or interventions.

## Supplied map contract

JSON contains exactly `encoder`, `decoder`, and `transition`. Each has finite
`weight` and `bias` arrays. All maps use row vectors: `x @ weight + bias`.
For `d` graph slots and summary width `r`, shapes are:

| Map | Weight | Bias | Operation |
|---|---|---|---|
| Encoder | d × r | r | Full state to summary |
| Decoder | r × d | d | Summary to reconstructed state |
| Transition | r × r | r | Summary to predicted next summary |

The same encoder and decoder apply at both boundaries. All arrays are cast to the
model's dtype/device; the record stores those executed values. The summary need
not be smaller than the state. Provenance describes how maps were obtained; the
package cannot verify it or prevent prior fitting on the evaluation population.

[Example maps](../examples/native-reduction-maps.json) retain only the first slot
of a four-slot graph. They require a matching slot order; they are not a pretrained
summary or a recommendation for an arbitrary graph.

## Data retained

The `nmn.reduction-study.v1` record retains dataset and model identities, sample
IDs, slot order, maps, provenance, source identity and full execution traces.
All residuals are **predicted/reconstructed minus actual**, with rows in sample-ID
order. The following are measured separately:

- Reconstruction of the current full state and its downstream output.
- Predicted next summary versus the summary of the actual next state.
- Decoding the actual next summary versus the actual full next state and output.
- Decoding the predicted next summary versus the actual full next state and output.

Separating these matters: a summary can predict its own evolution exactly while
omitting information required by the output. In the arithmetic fixture, retaining
only an unchanged input coordinate gives zero summary-transition residual, yet
loses the hidden value and changes the final output from 1 to 0.

The workflow makes two full forward calls (including the evidence snapshot) and
three suffix calls, each batched over the selected samples. Snapshot collection
also computes local kernel geometry; these counts are not a runtime cost model.
Nonfinite executed tensors produce `nonfinite-observation`, not an agreement
claim. There is no tolerance-based success verdict.

Numerical replay reconstructs the saved model, reexecutes the saved maps and
compares every observation and trace at explicit tolerances. Export and dashboard
commands only summarize saved data. Neither establishes semantic equivalence,
realizability of decoded states, continuous-domain closure, controlled dynamics,
nor propagation bounds through further layers.
