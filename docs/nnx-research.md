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
Arbitrary external-model conversion, statistical guarantees and
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

## CLI-only dataset workflow

A model definition is now a separate `nmn.nnx-model.v1` record: no placeholder
observation is executed or exported when initializing a model.

```bash
JAX_ENABLE_X64=1 nmn research native init --backend nnx --output nnx-model.json
JAX_ENABLE_X64=1 nmn research native inspect --model nnx-model.json
JAX_ENABLE_X64=1 nmn research native collect --model nnx-model.json --dataset dataset.json --split evaluation --edits edits.json --output observations.json
JAX_ENABLE_X64=1 nmn research native replay observations.json --output replay.json
nmn research native export observations.json --output obsidian-observations
nmn research native report nnx-model.json observations.json replay.json --output dashboard
```

Use a shared `nmn.research-dataset.v1` dataset and named edit dictionaries as in
the Python API. `--split` selects one population; omitting it explicitly collects
all samples in the dataset. `--no-derivatives` omits local derivatives.
`init --dtype float32` works without enabling JAX x64; initialization defaults to
float64 for both backends. The NNX reference is deterministically all ones, so
`--seed` does not change its parameters. `--graph` is rejected for NNX rather than
silently changing the requested architecture. Torch remains the default backend.

`collect` and `inspect` choose the backend from the saved schema; they accept
both NNX definitions and observation records. `extract --component model` exposes
the reusable model record from either schema. Restoring a model still validates
its identity and strict execution configuration. Unsupported Torch research
commands have not been ported to JAX by this dispatch.

## Complete-state suffix studies

`model.forward_from_state(state, start_layer=..., interventions=...)` exposes
three boundaries: 0 uses `(u,v)` inputs, 1 uses `(h,v,p)` after H/P execution,
and 2 uses `(target,protected)` outputs. Boundary 1 executes only Y; boundary 2
is identity readout. Controls on skipped modules are rejected. Supplied states
retain gradients and work under NNX JIT. Their reachability is not inferred.

```bash
JAX_ENABLE_X64=1 nmn research native suffix --model nnx-model.json --dataset dataset.json --split evaluation --states states.json --start-layer 1 --provenance "supplied decoder outputs" --output suffix.json
JAX_ENABLE_X64=1 nmn research native replay suffix.json --output suffix-replay.json
nmn research native export suffix.json --output obsidian-suffix
```

`states.json` maps variant names to sample-ID vectors covering exactly the selected
population. The `nmn.nnx-suffix-study.v1` record retains original states, baseline
and replayed outputs, reconstruction residuals, variant states/deltas, downstream
outputs/deltas and executed suffix traces. Numeric replay checks these fields.
No decoder is fitted and no closure or protected-domain certificate is inferred.

New observation records advertise suffix capability. Earlier observations retain
their historical `suffix_replay: false`; replay reports this API availability change
in `availability_changes`, separately from numerical mismatches. It does not alter
the saved full-forward computation. Graph completeness and cached-execution flags
remain compared; malformed capability values and missing fields still fail comparison.
