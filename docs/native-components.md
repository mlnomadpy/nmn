# Reuse research inputs from saved evidence

`nmn research native extract RECORD` lists reusable JSON components. To extract
one, supply `--component NAME --output NEW_DIRECTORY`. The directory contains
`data.json` and a receipt hashing the exact source bytes and extracted bytes.
Extraction requires no ML backend and never executes or fits a model.

| Component | Source | Consumer |
|---|---|---|
| `model`, `dataset` | Records embedding those fields | Native `--model`, `--dataset` |
| `contract` | Sampled contract evidence | `evaluate-contract --contract` |
| `contract-edit` | Sampled contract evidence | `collect --edits` |
| `configuration` | Architecture validation report | `init --graph` |
| `directions`, `background` | Curvature study | `curvature --directions`, `--background` |
| `selected-edit` | Successful gate search or edit selection | `collect --edits` |
| `maps` | Fitted or supplied reduction study | `reduce --maps` |
| `fit`, `evaluation` | Fitted reduction record | Native `replay` |
| `pairs` | Donor plan | Native `donor --pairs` |

The dynamic listing is authoritative for a particular record. Unavailable components
are rejected. Curvature extraction preserves the recorded direction-name order even
when the containing JSON object's keys were sorted. Architecture records and stored
model/contract identities are checked before extraction. These integrity checks do
not replace numerical replay or establish scientific validity.

```bash
nmn research native extract evidence.json --component contract --output reused-contract
nmn research native extract evidence.json --component contract-edit --output reused-edit
nmn research native evaluate-contract --model model.json --dataset dataset.json \
  --contract reused-contract/data.json --output checked-again.json
nmn research native collect --model model.json --dataset dataset.json --split validation \
  --edits reused-edit/data.json --output new-observations.json
```

A contract remains bound to its original model and dataset identities. Extracting
it does not rebind it to other data. A `contract-edit` is only a named native control:
collecting it on other data does not apply the original success criteria or imply
protection. A saved contract with violations remains a valid source of reproducible
inputs; extraction does not turn it into a successful result.
