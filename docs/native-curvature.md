# Directional gate curvature and finite edits

`nmn.torch.curvature.curvature_study` measures supplied gate Hessian-vector
products without allocating a dense module-by-module Hessian. One forward gradient
graph per sample is reused across output coordinates and directions. Models and
parameter gradient buffers are preserved.

```bash
nmn research native curvature --model model.json --dataset dataset.json \
  --directions directions.json --split evaluation --max-directions 16 \
  --provenance "Declared native gate directions" --output curvature.json
nmn research native replay curvature.json --output curvature-replay.json
nmn research native export curvature.json --output curvature-notes
```

`directions.json` maps unique names to finite gate displacement vectors in the
model's `state_names` order. `--background` optionally supplies a JSON gate vector;
the default is all ones. An endpoint is background plus displacement. Gates are
unrestricted finite scalars, so this interface does not enforce an attenuation
budget. Both direction count and population are explicit; too many directions
are rejected before derivative execution.

For every sample/output, the API records gradient g, products H v, directional
curvature vᵀH v, and cross-direction curvature uᵀH v. It executes each endpoint
through the original model, then retains first-order gᵀv and second-order
 gᵀv+½vᵀH v predictions and their signed errors relative to the actual output
change. A second-order prediction need not be more accurate for a finite edit.

`nmn.curvature-study.v1` contains axes, named directions, full per-example tensors,
model/dataset identity, source hash, baseline traces and cost counters. Derivative
and endpoint timing excludes the initial observation snapshot. HVP storage scales
with samples × outputs × directions × modules; the small cross-direction table
scales quadratically in direction count. Graph reuse is not a measured speedup.
The implementation loops over samples, outputs and directions rather than using
a backend-specific vectorized HVP API.

Numerical replay checks all saved derivative and endpoint measurements; timing is
not compared. Obsidian export and dashboard previews retain raw data and residuals.
This supplies local interaction data, not path-integrated attributions, sparse
interaction recovery, source-faithful HVP patching, uniform residual bounds or
floating-point certification. Use the separate `path` command for numerical gate
path integration and explicit domain certificates for continuous guarantees.
