# Joint gate paths and finite-effect comparisons

`nmn.torch.paths.gate_path(model, inputs, start, end, steps=32)` evaluates
`g(t) = start + t * (end - start)` through actual native model execution.
Gate vectors follow `model.state_names`; outputs follow `model.output_names`.
Both `ThreeNeuronYat` and `YatGraph` are supported.

The result retains sampled outputs, all gate gradients, joint directional
curvature, per-module numerical integrated contributions, actual endpoint change
and four effect predictions: first order, endpoint second order, integrated
gradient and integrated weighted curvature. Residuals are prediction minus actual
change, per sample and output. Directional curvature uses scalar path
differentiation, including mixed module effects without allocating a full gate
Hessian. Model forward calls and instrumented wall time are recorded.

```python
import torch
from nmn.torch import ThreeNeuronYat
from nmn.torch.paths import gate_path

model = ThreeNeuronYat.reference(dtype=torch.float64)
x = torch.tensor([[1., 1.]], dtype=torch.float64)
result = gate_path(model, x, [1, 1, 1], [0, 1, 1], steps=64)
```

For this reference the actual target change is −3.5. First order predicts −4;
endpoint second order predicts −7 and is worse. With 64 uniform intervals,
trapezoidal integrated-gradient residual is approximately 0.0001933 and the
weighted-curvature residual approximately −0.0006816. Protected change is zero.
These are measured numerical results on one path. A finite-path certificate
requires a separate sound quadrature/smoothness argument.

This path is an actual shared gate path. It is not EAP-IG input interpolation,
a reproduction of Integrated Hessians, or an attribution to isolated component
edits. Numerical completeness does not establish semantic correctness or a
uniform error bound. Run `examples/research/native_gate_path.py --output path.json`
for a model snapshot and complete path observations.

## Direct distance mode for curvature

`YatNMN(..., distance_mode="direct")` computes squared distances using explicit
coordinate differences. It avoids expanded-distance cancellation and the clamp
boundary at exact prototype matches. The default `"expanded"` mode remains
unchanged for existing callers; `YatExpansion` uses `"direct"` so research
curvature follows the smooth kernel formula. Direct mode allocates an intermediate
with shape `(..., out_features, in_features)` and can cost substantially more
memory than expanded computation. Spherical direct mode uses the actual
normalized vectors rather than substituting unit norms.

The earlier research modules inherited the expanded mode. At the all-ones
reference prototype match, the observed target path curvature was +2 rather than
the analytic −6. Earlier exported curvature observations at such points must not
be used as smooth-kernel Hessians. Current snapshot configurations record direct
mode, and source hashes distinguish the implementations. Forward predictions
alone would not have revealed this problem.

## Candidate selection ledger

`nmn.research.selection.SelectionLedger` records supplied candidate controls,
measured results, sample IDs and costs against a model/dataset identity.

```python
from nmn.research.selection import SelectionLedger
ledger = SelectionLedger(dataset, model_sha256=snapshot["model_sha256"])
ledger.register("remove_h", {"h": {"gate": 0}})
ledger.record("remove_h", phase="selection", sample_ids=tuning_ids,
              measurements=tuning_measurements, costs={"forwards": 5})
ledger.freeze("remove_h", rule="lowest target loss among protection-feasible candidates")
ledger.record("remove_h", phase="validation", sample_ids=validation_ids,
              measurements=validation_measurements)
record = ledger.to_dict()
```

Selection and validation splits must be disjoint. Validation requires a frozen
candidate; freezing prevents adding candidates or selection results. Group/split
validity comes from `ResearchDataset`. The ledger records explicit researcher
choices; it is not an optimizer, tamper-proof log, statistical testing procedure,
or evidence that repeated validation observations are independent. No population
claim follows from a low recorded finite-sample error.
