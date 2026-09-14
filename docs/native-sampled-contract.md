# Bound sampled target and protection contracts

`nmn research native evaluate-contract` executes a declared native edit under a
model/data-bound sampled contract. Edited target accuracy, baseline accuracy and
protection are separate measurements. A sampled observation never becomes a
uniform certificate.

```bash
nmn research native evaluate-contract --model model.json --dataset dataset.json \
  --contract contract.json --output evidence.json
nmn research native replay evidence.json --output replay.json
nmn research native export evidence.json --output notes
```

The `nmn.sampled-contract.v1` JSON requires:

- `scope: "sampled"`, `arithmetic: "floating-point"`, model and dataset SHA-256
  identities, ordered unique `sample_ids`, and nonempty `provenance`.
- `action_domain` with `kind: "shared-module-writes"`, finite ordered
  `gate_bounds: [lower, upper]`, and boolean `allow_replacements`.
- `controls`: module names mapped to optional scalar `gate` and optional shared
  scalar/output-width-vector `replacement`. Omitted gates are one, including
  unedited modules; every effective gate must satisfy the declared range.
- `targets`: named `output_names`, coordinate `absolute_tolerance`, and `expected`
  vectors keyed by exactly the declared sample IDs. These compare edited outputs
  with supplied semantic references.
- `protected`: `output_names` and `absolute_tolerance`, comparing edited outputs
  with the corresponding unedited baseline coordinates.
- Optional `baseline_targets`, structured like `targets`, to separately require
  ordinary baseline correctness. Protection alone only means unchanged behavior.

All tolerances must be finite and nonnegative. Model/dataset mismatches, undeclared
outputs, forbidden replacements, unsupported scopes and per-example policies are
rejected. Replacement takes precedence over gating according to native execution;
even an overridden gate must satisfy the declared action-domain bounds. The current
adapter supports float32/float64 Torch reference and explicit-state graph models.
Input-domain scope is exactly the listed samples; split labels are retained and no
population independence is inferred.

`nmn.sampled-contract-evidence.v1` retains the complete contract and content hash,
raw baseline/edited traces, per-sample references/errors, and every violation.
Status remains `observed`; `assessment` is `observed-within-tolerances` or
`observed-violations`. Changing the model, data, controls, references or tolerances
changes contract identity. Replay verifies the binding and recomputes numerical
measurements and raw traces. A replay can match a study containing violations.

The CLI saves evidence and exits 0 for sampled compliance, 1 for observed violations,
and 2 for invalid input or execution errors. Neither exit 0 nor a matching replay
certifies an unlisted population. Use the separate finite exact or continuous
interval contract interfaces for their explicitly supported arithmetic scopes.
Read/edge edits, parameter edits and input-dependent policies remain unsupported
by this adapter even where other native APIs can execute them.

Use [component extraction](native-components.md) to recover the bound `contract`
or reusable `contract-edit` directly from evidence. The former preserves its
model/data binding; the latter does not carry success or protection guarantees.
