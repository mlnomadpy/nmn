# Native NMN research from the CLI

Install `nmn[torch]`. All native commands run on CPU in float64. The ordinary NMN
CLI and `nmn research native --help` remain available without PyTorch.

The commands below use the JSON examples shipped in `examples/research/`.
Each output path must be new; existing evidence is never overwritten.

```bash
nmn research native init --output native-model.json
nmn research native inspect --model native-model.json

nmn research native collect --model native-model.json \
  --dataset examples/research/native-dataset.json \
  --edits examples/research/native-edits.json --output observations.json

nmn research native donor --model native-model.json \
  --dataset examples/research/native-dataset.json \
  --pairs examples/research/native-pairs.json \
  --protected protected --output donor-study.json

nmn research native path --model native-model.json \
  --dataset examples/research/native-dataset.json \
  --start 1,1,1 --end 0,1,1 --steps 64 --output gate-path.json
```

`init` creates the all-ones reference. Pass `--graph configuration.json --seed 0`
to initialize a general graph from `YatGraph.configuration()` JSON. This is
initialization, not training. Model files contain parameters, configuration,
identity and source hashes, without a pickle payload or arbitrary Python loader.
Both model files and full native research snapshots can be supplied to `--model`.

`collect` runs every named edit on the chosen population and exports model data,
module geometry, traces, derivatives and per-example effects. Use `--split NAME`
to restrict samples and `--no-derivatives` to omit costly Jacobian/Hessian data.

`donor` consumes a JSON array of `DonorPair` records with optional expected output
values. It retains the full dataset, donor/base identities and native responses.
`--match-semantics KEY ...` checks supplied donor eligibility labels. Cross-split
access requires `--allow-cross-split` and is recorded. Python reference callbacks
are available through the library API, not executable code in these JSON files.

`path` takes one comma-separated scalar gate per module, in the order shown by
`inspect`. `--split` restricts the measured population. It exports actual endpoint
effects and numerical path predictions; none is presented as a sound certificate.

Exit status 0 means the requested artifact was produced; it does not mean a
research hypothesis passed. Status 2 indicates invalid input, missing PyTorch or
an execution/write error. Model content hashes detect accidental parameter or
configuration changes, not malicious replacement of a whole file and its hash.
Historical expanded-distance snapshots are rejected instead of silently changing
their curvature semantics.

For Python integrations, `nmn.torch.research.model_from_snapshot` restores a native
model from validated JSON parameter/configuration identity. The CLI uses this same
API. See [datasets and donor studies](native-donor-studies.md),
[gate-path analysis](native-gate-paths.md), and
[native models and data](native-interpretable-torch.md).


`nmn research native diagnose --model native-model.json --dataset dataset.json
--module y --output diagnostics.json` exposes scoped kernel and sensor measurements.
See [kernel diagnostics](native-kernel-diagnostics.md) for the supported layer
settings and finite-data interpretation.


`nmn research native benchmark` compares saved models under one declared dataset
and edit contract. See [baseline comparisons](native-baseline-comparisons.md).


`nmn research native train` is an explicit bounded optimization command;
`native checkpoint` extracts a selected model for later research commands. See
[training protocols](native-training.md). Read-only commands never invoke training.


`nmn research native export record.json --output vault/new-run` writes an Obsidian
note beside the untouched JSON and file hashes. `native verify-export vault/new-run`
checks stored integrity without model replay; neither requires PyTorch. See
[native vault export](native-vault-export.md).

`native protect` executes declared classification tasks and preserves per-example
damage, repairs and eligible counts; see [protection studies](native-protection.md).
`native coalitions` measures module-subset interactions under explicit budgets;
see [coalition studies](native-coalitions.md). `native donor --read-slots` patches
selected reads of receiving graph modules; see [donor studies](native-donor-studies.md).

`native replay record.json --output replay.json` recomputes supported native
measurements. Exit 1 denotes a saved numerical mismatch; see
[native replay](native-replay.md) for fields compared and tolerance semantics.
`native report SOURCES... --output dashboard` creates a portable offline index;
see [the dashboard](native-dashboard.md).

`native suffix` compares supplied intermediate graph states with the unchanged boundary state and executes the remaining network. See [suffix studies](native-suffix.md). These records support native replay and Obsidian export.

`native semantics` checks supplied finite reference tables and variable-to-module correspondences. See [semantic studies](native-semantics.md); successful ordinary predictions do not imply counterfactual agreement.
