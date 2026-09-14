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
`validation_split`, `intervention_weight`, `separate_pair_rng`, and `detach_donor`. Adam and CPU float64 are currently
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
named outputs. Donor writes come from the current model. By default
`detach_donor=true` stops donor gradients; set it to `false` to train through the
donor representation too. Native base descendants recompute and carry gradients
in both modes. The saved protocol distinguishes these objectives from ordinary
training. Neither is claimed to
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

For controlled task-only versus donor-supervised comparisons, set
`"separate_pair_rng": true` in the training config. This preserves the same
minibatch stream when pair sampling is enabled; pair seeds are derived by the
recorded offset policy. The default shared-stream behavior is unchanged.
See [the runnable comparison](intervention-training.md) for a fresh-data study
that retains a negative donor-supervision result and distinguishes update from
compute budgets.

## Supervise a fixed native edit

Pass `fixed_edit=objective` to `train_native`, or `--fixed-edit objective.json`
to `nmn research native train`. Set `fixed_edit_weight` to a positive number
and optionally `protection_weight` in `TrainingConfig`. The objective is:

```json
{
  "schema": "nmn.fixed-edit-objective.v1",
  "controls": {"h": {"gate": 0.0}},
  "output_names": ["target"],
  "protected_outputs": ["protected"],
  "targets": {"train-0": [0.5], "validation-0": [0.25]},
  "provenance": "Declared analytic counterfactual target"
}
```

Replace the example target map with exactly every training and checkpoint-selection
sample ID, excluding evaluation IDs. Controls are fixed, shared scalar gates in
[0,1]. Outputs name native output coordinates. This version does not optimize
replacement vectors or input-dependent controls.

The loss adds edited-output target MSE and protected edited-versus-baseline MSE
to ordinary task MSE, with separately configured weights. Protection gradients
flow through both executions. A protected output can still be inaccurate; ordinary
supervision and held-out measurements remain necessary. Donor supervision can be
used concurrently, with its separate existing gradient policy.

`checkpoint_objective="task"` retains ordinary validation selection. Explicitly
choose `"task-plus-fixed-edit"` to select by the weighted task, edit and protection
validation losses. Records retain each component, the selection score, supplied
objective and selected checkpoint. They do not assert semantic correctness.

Run the matched example with:

```sh
python examples/research/intervention_training.py results --fixed-edit-comparison
```

It compares task-only against fixed h-deletion supervision on fresh seed 20260919,
with 128 training, 32 selection and 128 evaluation samples. Three minibatch seeds
share the same 28-parameter initialization, 300-update budget and data. The
protocol changes both the training loss and checkpoint criterion; it is not an
isolated estimate of the effect of the loss alone. Raw evaluation observations
include the actual h-deleted execution, alongside donor studies and replay.
