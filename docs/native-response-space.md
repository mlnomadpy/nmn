# Finite edit-response subspaces

```bash
nmn research native response-space --model model.json --dataset dataset.json \
  --edits edits.json --rank 1 --fit-split tuning \
  --evaluation-split validation --output response-space.json
nmn research native export response-space.json --output response-space-note
```

The Python interface is `nmn.torch.response_space.response_space_study`. It runs
all declared edits on both populations. A response is an edit's output change
relative to the unchanged network. Matrix rows index sample/output pairs and
columns index named edits in sorted order. Uncentered SVD on the fit population
supplies the requested number of right singular vectors; that basis is fixed
before the evaluation population executes. Model parameters are not trained.

The record retains full fit/evaluation snapshots and traces, raw responses,
singular values, the basis, projected coordinates, reconstructed responses,
residuals and per-sample residual norms. Numerical rank uses the recorded
threshold `max(matrix.shape) * float64_eps * largest_singular_value`. Requested
projection rank and measured numerical rank are distinct. Relative residual is
undefined (`null`) when the original response norm is zero.

This measures a finite response table, not a whole-function RKHS norm or uniform
response-rank bound. Evaluation requires every declared edit response: no claim
of predicting unknown edits from fewer queries is made. Each output coordinate
has equal weight; choose comparable output units or interpret this metric with
care. Repeated singular values permit basis rotations, and a cutoff through a
repeated group can produce nonunique projections. Low fit error alone says
nothing about unobserved response directions.

Storage includes all per-edit traces and response arrays. Collection performs
`2 * (1 + number_of_edits)` batched model forwards plus kernel diagnostics;
SVD costs depend on fit rows and edit count. Dataset group/split checks do not
establish statistical independence. Nonfinite responses are rejected. Native
export/dashboard support this schema; its replay adapter remains unsupported.
