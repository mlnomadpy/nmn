# Supervised native module alignment

`nmn.torch.alignment.alignment_study` fits a finite semantic correspondence in a
frozen native model. It searches injective assignments of supplied semantic
variables to single module writes. Each candidate is executed through the original
model using donor replacements. No probe decoder or surrogate suffix is fitted.

```bash
nmn research native align --model model.json --dataset dataset.json \
  --reference reference.json --module-pool p y h --max-candidates 3 \
  --selection-split tuning --evaluation-split validation \
  --provenance "Declared variable and candidate-module family" --output alignment.json
nmn research native replay alignment.json --output alignment-replay.json
nmn research native export alignment.json --output alignment-notes
```

The reference uses `nmn.semantic-reference.v1`: named variables, baseline outputs
by sample ID, and cases with base/donor IDs, transferred variables and expected
named outputs. Both populations must contain cases for every declared variable.
Pairs may not cross splits. Reference labels supply the semantic meaning.

Variables follow reference order; modules follow the explicit pool order. Candidate
permutations are generated lazily up to `max_candidates`. Selection minimizes the
mean squared error over all named counterfactual output values, with exact ties
resolved by the first candidate. Including protected outputs changes this objective;
record that choice in the reference. Baseline accuracy is measured separately.
The winner is frozen before evaluation cases execute. Evaluation labels cannot
change the chosen assignment. Do not reuse evaluation results for another selection
step while describing them as untouched final evidence.

`nmn.alignment-study.v1` retains the complete candidate count, executed prefix,
candidate scores, exact ties, winner, model/dataset identities, raw selection and
evaluation donor traces, baseline errors, protocol and limitations. A truncated
search has status `candidate-budget-stopped`; its winner is best only among the
executed candidates. `complete` means the finite family was exhausted, not that
the semantic hypothesis was verified. Replay recomputes assignment selection and
compares candidate scores, evaluation outcomes and underlying intervention traces.
Obsidian export and the offline dashboard preserve the record.

This is supervised finite mapping selection. It is not distributed alignment
search, automatic semantic discovery, causal scrubbing, a uniqueness result or a
statistical generalization guarantee. Multi-module, noninjective and read-edge
correspondences are outside this candidate family.

`python examples/research/semantic_alignment.py MODEL OUTPUT` supplies a runnable
polynomial-task experiment with fresh seed-20260917 selection/evaluation data.
Use a compatible saved hybrid graph. Model fitting and correspondence fitting are
separate stages; the example never updates model parameters.
