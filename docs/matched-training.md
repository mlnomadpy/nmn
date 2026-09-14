# Matched synthetic training study

`python examples/research/matched_training.py OUTPUT` performs actual bounded CPU
training of Yat, exponent-one IMQ and tanh models, followed by evaluation on a
separate untouched population. No network downloads or accelerator jobs are used.
The output directory must be new; partial records remain if execution fails.

The fixed task is `target=u²+0.5v`, `protected=v` on synthetic inputs in `[-1,1]²`.
There are 128 training, 32 checkpoint-selection and 128 final evaluation samples.
A saved deterministic dataset seed generates all inputs before training. Labels
come from the analytic task, not a teacher model. Evaluation labels are excluded
from the training API's target mapping.

All methods share graph routing H(u), P(v), Y(h,v), four units per module and
matching initial center/hidden-weight arrays. Readout coefficients start at 1/4;
tanh biases start at zero. Yat and IMQ have 28 parameters; tanh has 40. This is an
equal-width/routing comparison, not an equal-capacity or tuned-performance claim.

Each method runs minibatch seeds 0, 1 and 2 from its fixed initialization, with
300 Adam updates, batch size 32, learning rate 0.01, checkpoint evaluation every
20 steps and a 20-second optimization budget per run. Validation MSE chooses the
checkpoint; final evaluation never changes it. The protocol file is written
before fitting. Seeds vary minibatches only, not dataset or initialization.

Raw training histories/checkpoints, per-example final outputs, errors, native
h-deletion effects, model identities and numerical replay are retained. Summary
rows include per-output task MSE and error against the intended `u=0`
counterfactual. Deleting h is a native action; it need not realize that semantic
counterfactual for a learned model. Protected-output independence is imposed by
the shared routing, so observing unchanged protected outputs cannot establish a
kernel-specific advantage. No population confidence interval or erasure claim is
made from these three runs.
