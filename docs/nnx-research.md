# NNX research observations

`nmn.nnx.YatExpansion` and `ThreeNeuronYat` are trainable Flax NNX modules.
The width-one architecture has seven parameters and routes `h=H(u)`, `p=P(v)`,
`y=Y(h,v)`. It exposes live inputs, signed center contributions, raw outputs and
effective edited states. Gate and replacement dictionaries work under NNX JIT;
replacement takes precedence. Parameter, input and control gradients remain live.

```python
from flax import nnx
import jax.numpy as jnp
from nmn.nnx import ThreeNeuronYat

model = ThreeNeuronYat.reference()
x = jnp.ones((1, 2), dtype=jnp.float32)
outputs, trace = model.forward_with_trace(x, {"h": {"gate": 0.0}})
# outputs: [[0.5, 1.0]]
compiled = nnx.jit(lambda model, x: model(x))
```

The strict expansion uses an NMN NNX kernel parameter bank, with kernel storage
input-by-center and exported centers center-by-input. Its execution explicitly
uses direct squared coordinate differences in float32 or float64. This differs
from the general `YatNMN` forward's FP32 expanded-distance policy. Bias, alpha,
learnable epsilon, spherical normalization, dropconnect and tied/frozen banks
are not options on the strict expansion. Existing general layers are unchanged.
Do not pass a general NNX layer to this collector as if it met this contract.
Float64 requires the caller to enable JAX x64 before construction; the package
never changes global JAX settings. Inputs must match parameter dtype.

`nmn.nnx.research.collect_research_data(model, dataset, split=..., edits=...)`
accepts the shared `ResearchDataset`. The `nmn.nnx-research.v1` JSON record retains:

- Exact parameter/configuration and dataset identities, sample/split/group IDs,
  supplied semantics and their provenance.
- Baseline and edited per-example outputs, deltas and full executed traces.
- Center Gram matrices, eigenvalues, features and local expansion RKHS products.
- Optional baseline input Jacobians, gate Jacobians and gate Hessians, with gate order.
- Runtime/device/precision, synchronized elapsed time and capability limitations.

Local RKHS products describe individual expansions, not the composed network.
Complete graph means this fixed architecture's graph is known; it does not make
its floating-point observations a structural or continuous-domain certificate.
The collector rejects nonfinite serialized results and has no cached execution.
Suffix replay, arbitrary external-model conversion, statistical guarantees and
interval certification are unsupported. The native replay command dispatches this schema to JAX without importing Torch.

Run the self-contained four-point fixture, then use the backend-independent
export/dashboard commands with its observations:

```bash
JAX_ENABLE_X64=1 python examples/research/nnx_observability.py /tmp/nnx-evidence
nmn research native report /tmp/nnx-evidence/observations.json --output /tmp/nnx-dashboard
```

The example writes `observations.json` and an Obsidian directory containing the
exact data, a Markdown note and integrity manifest. Existing destinations fail.
The fixture is an arithmetic check, not a trained task or literature reproduction.


## Restore and replay saved evidence

```bash
JAX_ENABLE_X64=1 nmn research native replay /tmp/nnx-evidence/observations.json --output /tmp/nnx-replay.json
nmn research native export /tmp/nnx-replay.json --output /tmp/nnx-replay-note
```

`nmn.nnx.replay.model_from_snapshot(record)` restores saved parameters without
pickle or executable loaders. It validates content identity, the exact supported
routing/arithmetic, parameter shape and representability in the saved dtype.
`replay_native_record` additionally checks dataset identity and reruns the saved
population, edits, geometry and optional derivatives on CPU. It compares complete
measurement trees, retaining mismatch paths and a fresh execution. Source hashes,
runtime versions and timings are retained but excluded from numerical comparison.
Defaults are atol=1e-10 and rtol=1e-8; callers can declare different tolerances.
The CLI exits 0 for agreement, 1 for a written mismatch, and 2 for an invalid or
unsupported record. Float64 records require caller-enabled JAX x64; no silent
precision downgrade occurs. Replay establishes numerical agreement only.
