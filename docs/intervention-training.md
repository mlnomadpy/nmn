# Detached-donor supervision comparison

Run `python examples/research/intervention_training.py OUTPUT` to compare task-only
training against task plus detached-donor supervision on a fixed hybrid graph.
The script writes its protocol before fitting and retains complete checkpoints,
held-out observations, donor transfer rows and six numerical donor replays.

The task is target=u²+0.5v, protected=v. H and Y use four-factor Yat expansions;
P uses four-factor linear expansion. Fresh dataset seed 20260915 generates 128
training, 32 checkpoint-selection and 128 evaluation samples. Initial factors
use seed 17. Each training sample's donor is the cyclic offset-17 sample in its
own split. Replacing h with donor h is assigned target donor_u²+0.5base_v. Final
evaluation pairs use only evaluation samples, with no training donor access.

Both conditions use 300 Adam updates, batch size 16, learning rate 0.01, validation
every 20 steps, and a 60-second optimization cap per run, across seeds 0,1,2.
The donor condition adds weight-one mean squared transfer loss; the donor write
is detached from autograd. Task-only gets weight zero and no donor pairs. Ordinary
validation MSE chooses checkpoints in both conditions. Extra donor forwards make
this an equal-update comparison, not an equal-compute comparison.

`TrainingConfig(separate_pair_rng=True)` makes minibatches use `seed` and donor
pair sampling use `(seed+2**32) mod 2**63`. This prevents donor sampling from
advancing the task minibatch stream. The flag defaults to false to preserve the
previous shared-stream behavior. The selected policy is recorded in the config
and seed-scope field. Separate deterministic streams do not themselves establish
statistical independence of research populations.

The first executed run of this protocol increased held-out donor-target error
for all three seeds when donor supervision was added. It should be retained as a
negative result under this budget, optimizer, detached-gradient convention and
checkpoint-selection rule. This does not reproduce a published IIT algorithm or
show that intervention supervision generally fails. A revised objective or
selection strategy needs a fresh evaluation protocol rather than tuning against
these held-out outcomes.

## Donor gradient policy

`TrainingConfig(detach_donor=False)` enables gradients through both the donor
representation and the recomputed base suffix. The default `True` preserves
detached donor writes. The saved configuration and protocol distinguish these
objectives, which can produce different learning signals even with identical
forward losses. Neither setting establishes semantic correspondence by itself.

Run `python examples/research/intervention_training.py OUTPUT --gradient-comparison`
to compare detached and joint donor gradients on fresh dataset seed 20260916.
Both conditions use donor loss weight one, the same initialization, task and pair
minibatches, and the same budgets above. The protocol is saved before fitting;
checkpoint selection still uses ordinary validation loss only.
