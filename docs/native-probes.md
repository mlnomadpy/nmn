# Internal-state classification probes

`nmn.torch.probes.probe_study` and `nmn research native probe` fit a linear
classifier to one recorded trace tensor and measure decoding on a separate
population, including named native edits. This supplies raw activation/label data
for KG03 observability and future erasure comparisons. It is not an implementation
of LEACE or a claim that a semantic concept has been localized.

```bash
nmn research native probe --model graph.json --dataset dataset.json \
  --feature a --labels labels.json --classes zero one --ridge 0.01 \
  --provenance "Supplied labels; feature chosen before evaluation" \
  --fit-split tuning --evaluation-split validation --edits edits.json \
  --output probe.json
nmn research native replay probe.json --output probe-replay.json
nmn research native export probe.json --output vault/probe
```

`--feature` is an actual trace key with shape samples × features. For a graph,
`state.1` is the full state after layer zero, `a` is module a's effective write,
`a.raw` is its unedited raw write, and `a.input` is its received input. A gate
changes the effective write, not the raw write. Module contribution tensors have
three dimensions and are rejected; no implicit flattening or pooling is applied.

Labels are a JSON object mapping exactly the fit and evaluation sample IDs to
class-name strings. Every declared class must occur in the fit population. There
must be at least two fit samples and one evaluation sample, with different split
names. Other dataset splits are not executed. Class order is explicit: score ties
choose the first class. `--edits` is optional and uses the native module gate/
replacement format. By default, edits are applied only to the evaluation population. The explicit refit option below also applies them to fit samples.

The classifier minimizes `mean_i ||x_i W + b - one_hot(y_i)||² + ridge * ||W||_F²`.
Feature and target centering use only baseline fit samples. The intercept is
unpenalized; ridge must be strictly positive and representable in float32/float64
model arithmetic. Features are not standardized. The native model is unchanged.
The probe is frozen before evaluation execution; it is not refitted for edits.
Scores are uncalibrated regression outputs, **not probabilities**.

The `nmn.probe-study.v1` record preserves:

- Model and dataset identities, complete fit/evaluation snapshots and traces.
- Label provenance, split IDs, selected trace key, class order and ridge penalty.
- Fitted coefficients, intercept, and fit-only feature/target means.
- Raw features, scores, labels, predictions and correctness for every example.
- Accuracy and confusion counts (rows=true class, columns=predicted class).
- Per-edit feature/score changes, disagreement, and damage conditional on the
  unchanged probe being correct. Conditional damage is null with zero eligible
  samples.

Numerical replay reconstructs the model and refits the declared affine ridge
classifier before comparing the recorded probe, predictions, and evaluation
traces. This explicit linear solve is distinct from the unsupported neural
training-record replay. It does not validate labels or their provenance.

In the arithmetic fixture, a probe decodes a module write perfectly. Disabling
that write reduces frozen-probe accuracy to 50%, while a probe reading the preserved module input still decodes perfectly. This is why
frozen-probe failure does not prove information erasure. The package records both
conditions without inferring a semantic or causal guarantee.

No stable inverse, statistical confidence bound, nonlinear adversary, semantic
causality or erasure certificate is provided.
Repeated evaluation-guided feature/ridge choices can leak evaluation information;
structural split/group validation cannot prevent that. Cost records count model
forwards and the linear solve; snapshot kernel-geometry work is additional.


## Compare frozen decoding with a refitted linear probe

Add `--refit-edits` (Python: `refit_edits=True`) to fit a separate affine ridge
classifier for each edited representation. All edited fits use only the fit
population, and every classifier is frozen before any evaluation forward. The
baseline classifier and its frozen edited predictions remain in the same record.
The model parameters are unchanged. This is additional fitting of small linear
probes, not training the native neural network.

This option requires at least one edit and population-independent scalar gate or
replacement controls. Batch-dependent and vector controls are rejected rather
than reused ambiguously across fit/evaluation populations. Use the same feature,
class order and ridge penalty for all conditions; no evaluation-driven tuning or
best-probe selection is performed.

Each evaluation edit includes `refitted.probe`, `refitted.fit`, and
`refitted.evaluation`, with full features, scores, predictions and confusion
counts. The fit snapshot additionally retains all edited traces. Replay refits
all declared probes and compares their coefficients, predictions and both
populations' observations. The export note lists frozen and refitted accuracies
separately. With E edits, the workflow makes 2 + 2E full model forwards and 1 + E
linear solves, plus snapshot geometry work.

In the binary arithmetic fixture, negating the module write reduces frozen-probe
accuracy to 50%, but a refitted probe reaches 100%. Replacing the write with zero
leaves the refitted probe at 50%. These measurements distinguish two kinds of
failed decoding in this finite linear comparison. They do not prove information
erasure across other features, nonlinear decoders or unseen populations.
