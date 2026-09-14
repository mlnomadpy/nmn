# Native classification protection study

`nmn.torch.protection.protection_study` executes each named edit independently
from the same model, retaining baseline/edited outputs, internal traces, controls,
model identity, sample IDs and dataset provenance. It evaluates declared
classification tasks without inferring their meaning from a neuron name.

```bash
nmn research native protect --model model.json --dataset dataset.json \
  --edits edits.json --tasks tasks.json --provenance 'label source and version' \
  --split evaluation --strata kind --output protection.json
nmn research native export protection.json --output protection-note
nmn research native report protection.json --output protection-dashboard
```

A tasks file maps task names to explicit decoding and labels:

```json
{
  "protected": {
    "outputs": ["protected"],
    "rule": "threshold",
    "threshold": 0.5,
    "labels": {"base": 1, "donor": 0}
  }
}
```

Labels must cover exactly the selected sample IDs. A threshold task has one
output and labels 0/1, with equality decoded as 1. An argmax task has at least
two named outputs and integer labels indexing that declared order; ties select
the first. Nonfinite task scores are rejected rather than decoded as classes.

The `nmn.protection-study.v1` record contains per-example original and edited
correctness, broken/fixed flags, predictions, disagreement, accuracy and the
conditional damage rate among originally correct examples. A zero eligible
count gives `null`, never a fabricated zero rate. Each requested semantic field
has separate strata, including a distinct missing-value group; these are not
intersectional groups. All rows remain in the overall population.

Unchanged aggregate accuracy can conceal damage offset by repairs. Interpret
accuracy, conditional damage and disagreement separately. This collector does
not select edits, establish label validity, supply population confidence bounds,
correct multiple comparisons or implement sequence generation protocols.
