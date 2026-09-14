# Matched-routing module comparisons

`YatModuleSpec(..., family="yat" | "imq" | "tanh")` chooses a native block within
the same explicit-state graph. Routing, residual writes, gates, replacements,
traces and JSON model restoration remain shared.

| Family | Implemented block | Interpretation |
|---|---|---|
| `yat` | `(c·x)² / (||c-x||² + epsilon)` with signed coefficients | Local fixed-epsilon ⵟ expansion |
| `imq` | `1 / (||c-x||² + epsilon)` with signed coefficients | Generalized IMQ exponent 1, matching the rational denominator comparison; not exponent 1/2 |
| `tanh` | affine hidden layer → tanh → bias-free linear readout | Conventional smooth neural block; no RKHS norm assigned |

`num_centers` sets kernel-bank width or tanh hidden width. Tanh has hidden biases,
so equal widths can have different parameter counts. Epsilon is irrelevant to
tanh. IMQ and tanh blocks are also exported as `IMQExpansion` and `TanhMLPBlock`.
Trace contributions mean center contributions for kernel blocks and hidden-unit
contributions for tanh. None supplies semantic labels by itself.

## Run a comparison

```bash
python examples/research/native_baselines.py --output comparison.json

nmn research native benchmark --models models.json --dataset dataset.json \
  --edits edits.json --protected p --repeats 5 --warmup 1 --output benchmark.json
```

`models.json` maps method names to native model/snapshot JSON paths, resolved
relative to that file. Dataset and edits use the existing native CLI schemas.
`--expected targets.json` optionally supplies a finite `(samples, outputs)` target
matrix aligned with selected sample IDs; `--split` restricts that population.
Invalid declarations/model files are input errors. Once models are loaded, every
method is retained as measured or failed, including incompatible routing and
execution failures. The runner does not select only successful methods.

The Python API is `nmn.torch.benchmark.benchmark_models`. It checks identical
routing and common output/module names, while allowing different families and
widths. Records include parameters/configuration, actual outputs, per-example
baseline squared errors when labels are supplied, edit deltas, each protected
output's change, parameter counts, runtime/hardware, execution order, warmup and
repeat budgets, timing samples and failures. Each method uses its own baseline
when measuring its protected changes. Tracing/geometry collection is timed
separately from repeated batched forward calls.

This initial runner measures CPU direct replay only. It does not fit models or
compare acquisition algorithms, certified verification or prediction-fallback
policies. Training cost is explicitly unmeasured for supplied models. Timing
includes Python instrumentation and is not a hardware-independent ranking.
Method order is recorded but not randomized; this is not a controlled throughput
benchmark. Equal routing or width does not establish equal capacity, competence
or information acquisition cost.

The runnable fixture supplies untrained parameters. ⵟ and IMQ have seven scalars
and tanh has ten. Their outputs differ; no task competence is claimed without
reference labels, and no ⵟ-specific advantage follows from this demonstration.
The comparison API supplies data needed for a subsequently specified study.
