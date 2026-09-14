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

## Fit a summary before evaluating it

`fit_reduction_study` and `nmn research native fit-reduction` learn affine maps
from a declared fit population. They leave the native model parameters unchanged.
Rank and a strictly positive ridge penalty are explicit inputs, not selected using
evaluation results.

```bash
nmn research native fit-reduction --model graph.json --dataset dataset.json \
  --start-layer 1 --rank 1 --ridge 0.01 --fit-split tuning \
  --evaluation-split validation --output fitted-summary.json
nmn research native export fitted-summary.json --output vault/fitted-summary
```

The algorithm executes only fit inputs to collect the states immediately before
and after the selected layer. It centers their concatenation, computes an SVD,
and uses the requested leading directions as the encoder. The decoder is the
transpose basis plus the joint mean. Rank exceeding the fit matrix's numerical
rank is rejected; the saved threshold is `max(matrix.shape) * dtype_eps * s_max`.
Columns have a canonical sign, but repeated singular values do not define unique
subspaces. This implementation requires float32 or float64 parameters.

The reduced transition solves affine ridge regression on paired fit summaries:
`mean_i ||z_i A + b - z_next_i||² + ridge * ||A||_F²`. The intercept is not
penalized. All maps are frozen into JSON before executing evaluation inputs.
Neither evaluation values nor their boundary states enter the mean, PCA, or solve.
At least two fit samples and one evaluation sample are required, with distinct
split names. Dataset group checks prevent declared groups crossing splits; they
cannot establish independence or prevent evaluation-guided reruns by callers.

The `nmn.fitted-reduction.v1` record contains fitted maps, singular values, rank
threshold, source/model/dataset identities, split IDs, and complete `fit` and
`evaluation` reduction records. Those embedded records can be extracted for
ordinary numerical replay of the **frozen maps**:

```bash
nmn research native extract fitted-summary.json
nmn research native extract fitted-summary.json --component evaluation --output extracted-evaluation
nmn research native extract fitted-summary.json --component maps --output extracted-maps
```

`nmn research native replay extracted-evaluation/data.json --output replay.json` checks
execution of those maps. Replaying the outer fitted record is unsupported: no
claim is made that this reruns fitting or validates its provenance. The saved
maps also work with `native reduce` on further populations.

This is a finite linear baseline for learned summaries. It does not fit neural
summary encoders, controlled transitions, semantic correspondences or invariant
domains. It provides the raw data needed to compare these later. The workflow
uses one fitting forward, four measurement forwards and six suffix forwards,
plus one SVD and one linear solve; geometry collection is additional work and
these operation counts are not a wall-time guarantee.
