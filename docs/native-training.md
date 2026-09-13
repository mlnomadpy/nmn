# Explicit native training and checkpoint selection

`nmn.torch.training.train_native` fits independent CPU copies of a supplied
native model snapshot. Ordinary task MSE and optional donor-supervised MSE are
separate terms. Graph routing is imposed by the architecture; fitting parameters
does not establish recovery of semantic organization.

```bash
nmn research native train --model initial-model.json --dataset dataset.json \
  --targets targets.json --config training.json --contract architecture-contract.json \
  --target-provenance "task reference, version 1" --seeds 0,1,2 --output training-run.json

nmn research native checkpoint --run training-run.json --seed 0 --output selected-model.json
nmn research native collect --model selected-model.json --dataset evaluation.json \
  --output selected-observations.json
```

Training occurs only on explicit invocation of `train` or the Python function.
Inspection, collection, donor analysis, paths and benchmarking never call it.
There is no automatic cloud execution or dataset download. Outputs must use new
paths. Artifact creation is not itself training success: inspect each run's
`status`, errors and checkpoint-selection history.

`training.json` configures `max_steps` (required), `max_seconds` (default 60 per
seed), `batch_size`, `learning_rate`, `evaluate_every`, `train_split`,
`validation_split`, and `intervention_weight`. Adam and CPU float64 are currently
fixed. The wall-time limit is checked between steps and does not preempt a slow
step. Optimizer timing excludes initialization and checkpoint instrumentation.
`targets.json` maps exactly the training and checkpoint-selection sample IDs to
finite vectors in model output order. Final-test labels are not accepted there.

The architecture contract requires nonempty `architecture_id`,
`semantic_specification`, `intervention_specification`, `guarantee_scope`, and
`worked_example` references/descriptions. These are researcher declarations, not
an automated proof of the architecture obligations. Dataset provenance, supplied
targets and the full declarations are retained in the training record.

## Protocol distinctions

All seeds start from the **same supplied weights**. Seeds control minibatch and
pair sampling, not independent initialization. Use separately initialized model
files to study initialization variation. Validation selects the checkpoint with
the lowest observed MSE; ties retain the earlier checkpoint, including the initial
model. It is checkpoint-selection data, not untouched final evaluation. No final
study is selected or launched by this implementation work.

For intervention-supervised training, supply `--pairs pairs.json` and a positive
`intervention_weight`. Every pair must use training samples and provide expected
named outputs. Donor writes come from the current model with donor gradients
stopped; native base descendants recompute and carry gradients. This specific
protocol is labeled separately from ordinary training and is not claimed to
reproduce a published interchange-intervention-training algorithm. Correspondence
and counterfactual labels are supplied, not inferred.

Records retain all seeds, completed/failed/budget-stopped outcomes, completed step
counts, task/intervention losses at evaluation checkpoints, validation history,
selected step and selected model snapshots. The selected snapshot can be extracted
and used by the other native tools. Frozen parameter flags are preserved. Older
snapshots without trainability declarations default to trainable parameters;
parameter/configuration content hashes identify the function snapshot, not the
training policy. Initial trainability is recorded separately in each run record.

Checkpoints are weight/configuration snapshots, not full optimizer-resume files.
Restarting from one creates a new optimization run. No learned interpretability,
population risk guarantee or training success-rate theorem is implied by a small
loss or a few successful seeds.

## Runnable implementation fixture

`python examples/research/native_training_fixture.py --output fixture.json`
explicitly runs two optimizer steps for each of two sampling seeds, with the
protected branch frozen. It is a synthetic implementation fixture, not a research
training result. The package checks also exercise task/donor loss separation,
frozen-weight preservation and rejection of validation donor access.
