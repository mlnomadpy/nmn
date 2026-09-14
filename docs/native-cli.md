# Native NMN research workflow

Install `nmn[torch]` for model execution. Native CLI execution uses CPU float64;
replay follows saved float32/float64 precision. Help, export, integrity checking
and offline reports work without a PyTorch installation. All output paths must
be new so saved evidence is never overwritten.

From the repository, run a first observation/export workflow:

```bash
nmn research native init --output native-model.json
nmn research native inspect --model native-model.json
nmn research native collect --model native-model.json \
  --dataset examples/research/native-dataset.json \
  --edits examples/research/native-edits.json --output observations.json
nmn research native replay observations.json --output replay.json
nmn research native export observations.json --output observation-note
nmn research native report observations.json replay.json --output dashboard
```

`init` constructs the all-ones reference. `--graph configuration.json --seed 0`
constructs a general graph. It does not train. Saved models use checked JSON
configuration and parameters, without pickle or executable model loaders.
`--model` accepts both a model definition and a complete native model snapshot.
Historical expanded-distance snapshots are rejected rather than changing their
curvature semantics silently.

| Command | Purpose and detailed guide |
|---|---|
| `init`, `inspect` | [Native models, routing and data](native-interpretable-torch.md) |
| `collect` | Full traces, kernel geometry, derivatives and executed edit effects |
| `donor` | [Declared donor pairs and receiving-module read patches](native-donor-studies.md) |
| `path` | [Joint gate paths, numerical curvature and quadrature residuals](native-gate-paths.md) |
| `verify-box`, `check-box` | [Bounded range contracts and partition checking](native-interval-contract.md) |
| `enclose` | [Exact rational real-function bounds](native-enclosure.md) |
| `diagnose` | [Kernel and sensor measurements](native-kernel-diagnostics.md) |
| `protect` | [Classification accuracy, damage, repairs and eligibility](native-protection.md) |
| `coalitions` | [Budgeted module-subset interaction measurements](native-coalitions.md) |
| `suffix` | [Supplied intermediate states and downstream responses](native-suffix.md) |
| `semantics` | [Supplied references and counterfactual correspondence checks](native-semantics.md) |
| `select` | [Candidate selection followed by frozen-winner validation](native-selection.md) |
| `response-space` | [Fit-only response bases and evaluation reconstruction errors](native-response-space.md) |
| `benchmark` | [Saved-model comparison under a shared replay contract](native-baseline-comparisons.md) |
| `train`, `checkpoint` | [Explicit bounded optimization and selected checkpoint extraction](native-training.md) |
| `replay` | [Numerical reproduction with declared tolerances](native-replay.md) |
| `export`, `verify-export` | [Obsidian notes, raw data and integrity checks](native-vault-export.md) |
| `report` | [Portable offline evidence dashboard](native-dashboard.md) |

Run `nmn research native COMMAND --help` for required inputs. Research JSON
examples live under `examples/research/`; the table links to supported formats
and scope limits. `collect --no-derivatives` avoids costly Jacobian/Hessian data.
`path` gate order follows the module order returned by `inspect`. Dataset split,
group and semantic metadata are declarations, not evidence of independence or
semantic discovery. See [the vault-driven requirements map](research-data-requirements.md)
for capabilities that remain unimplemented.

Exit status 0 normally means an artifact was produced, not a scientific hypothesis
passed. Read the record's outcome: selection can fail validation, semantic tests
can disagree, and budgeted coalitions can be inconclusive. `replay` is the exception:
exit 0 means a numerical match and exit 1 means a written mismatch report. Status 2
means invalid input, unavailable backend or execution/write failure.

File integrity, numerical reproduction and scientific validity are separate.
Hashes detect accidental content changes; replacing both data and hashes is not
prevented. Replay never silently launches training. Only `train` performs model
optimization; response-space fitting is explicitly a numerical SVD on observations.
