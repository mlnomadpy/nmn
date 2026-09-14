# Kernel and sensor diagnostics

```bash
nmn research native diagnose --model native-model.json \
  --dataset examples/research/native-dataset.json \
  --module y --noise-radius 0.01 --output kernel-diagnostics.json
```

The command measures the selected module at its **actual incoming states** on
the selected dataset. For example, the target module reads `(h, v)`, not the
original `(u, v)`. `--split` restricts the population. The output contains a model
snapshot, dataset/sample identities, observed layer settings and sensor geometry.

`nmn.torch.diagnostics.diagnose_layer(layer, inputs)` also accepts existing
`YatNMN` layers directly. It reports actual outputs and zero-input behavior,
parameter values/trainability, bias/alpha/epsilon settings, lazy mode and distance
mode before assigning any function-space quantities. The implemented unbiased
snapshot diagnostic excludes biased, normalized, tied-bank, learnable-epsilon and
hidden-alpha configurations. Those layers remain runnable but return explicit
unsupported reasons. This is a limitation of the diagnostic, not a claim that
those mathematical constructions are invalid.

Supported `YatExpansion` snapshots provide sampled Gram matrices/symmetry residuals,
eigenvalues, numerical rank, center conditioning and coordinatewise RKHS inner
products `A K_centers Aᵀ`. These are local expansion norms. The roundoff scale used
to contextualize small negative eigenvalues is a floating-point heuristic, not a
certified error bound. A sampled Gram spectrum does not certify global PSD,
universality or semantic meaning. Fixed-snapshot interpretation does not assert a
common RKHS throughout training; lazy mode freezes only kernel directions.

## What the finite bank can distinguish

`sensor_diagnostics(centers, points, epsilon=1, noise_radius=0)` measures the
unbiased shared-epsilon sensor map `Psi(x) = [k(x, c_i)]`. It returns:

- Raw sensor responses, zero response, sample mean and covariance.
- Sensor/input Jacobians and singular values, including zero input directions
  omitted by a thin SVD when the number of sensors is smaller than input width.
- All sampled source/observation pair distances, the minimum separation ratio
  over distinct sampled pairs and the corresponding pair mask.
- Whether each sampled pair's observation distance exceeds twice the supplied
  Euclidean noise radius. This concerns disjoint observation-space noise balls.

No noise is synthesized and no sensor inverse is fitted. A positive finite-pair
ratio is not a uniform inverse-Lipschitz bound; positive local Jacobian gain is
not global injectivity. Covariance describes the supplied finite population.
Duplicate inputs are excluded from ratio denominators; a population with no
distinct pair returns no minimum ratio. A single sample returns no covariance.
All-pairs arrays cost quadratic storage in sample count. Geometry uses float64;
choose a backend/device that supports it.

A one-sensor bank in two input dimensions always has a zero local input gain in
at least one direction. This can coexist with well-separated observations on a
particular finite input list. At zero, the unbiased sensor map also has zero
first derivative. These diagnostics expose the distinction between recovering
full state and observing a particular task response.

This supplies numerical data for issue #45 and vault gap KG03. Stable recovery
bounds, probe fitting, sensor acquisition and semantic identification remain
separate research obligations.
