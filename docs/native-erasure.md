# Learned finite-bank erasure targets

`nmn.torch.erasure.erasure_study` fits a covariance-removal projection in the
unweighted finite kernel features of a strict native Yat module. It supplies
learned targets for the existing preimage interface. It does not implement LEACE,
Nyström whitening, a minimax eraser, or an intrinsic RKHS projection.

For fit feature matrix X and label matrix Y, center both at their fit means and
compute C=XcᵀYc/n. Take its thin SVD and retain left singular vectors U whose
singular values exceed `rtol * largest_singular_value`. The frozen map is
`x -> mean_X + (x - mean_X)(I - UUᵀ)`. Zero covariance selects rank zero. Rank
truncation and numerical arithmetic can leave a nonzero residual, which is
recorded. Removing covariance does not establish statistical independence or
prevent nonlinear decoding.

```sh
nmn research native erase --model model.json --dataset dataset.json --module y --labels labels.json --rtol 1e-10 --fit-split fit --evaluation-split evaluation --provenance "declared numeric concept labels" --output erasure.json
nmn research native extract erasure.json --component erasure-targets --output targets
nmn research native preimage --model model.json --dataset dataset.json --module y --split evaluation --targets targets/data.json --bounds bounds.json --max-steps 100 --max-seconds 30 --learning-rate 0.05 --provenance "frozen projection targets" --output proposal.json
nmn research native apply-preimage proposal.json --output execution.json
nmn research native replay execution.json --output replay.json
```

Labels map exactly the fit and evaluation sample IDs to finite vectors of a
consistent positive width; one-hot class labels can be supplied explicitly.
The fit population has at least two samples. Evaluation labels enter only the
reported covariance, never the fitted map or target construction. Choosing the
bank, labels or threshold using evaluation results still compromises the study.

The `nmn.erasure-study.v1` record retains both native snapshots, dataset identity,
labels, population order, projection/mean/singular spectrum, threshold, rank,
idempotence residual, raw original/projected features, covariance matrices,
distortion and evaluation targets. Export and dashboard ingestion preserve it;
projection fitting replay is not implemented. Native execution replay is
available after the separate preimage step.

Projected bank features may not have a native preimage. Preimage optimization
uses evaluation inputs and projected targets individually; although the feature
projection is fit-only, this is not a generalizing learned native mapper. Compare
projected versus executed covariance, feature residuals and task/protection losses.
Holding protected coordinates fixed in a receiver does not prove every protected
output remains fixed in an arbitrary graph.

`python examples/research/kernel_erasure.py CHECKPOINT OUTPUT` runs this pipeline
on fresh synthetic data seed20260920 with a compatible two-input target/protected
graph and a Yat receiver named y. It records projection and execution separately,
including negative results and the downstream error relative to target=0.5v.
