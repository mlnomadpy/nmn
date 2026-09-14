# Producer-specific residual read patches

`YatGraph.forward`, `forward_with_trace`, and `forward_from_state` accept
`edge_patches` to replace a particular producer's contribution at one receiving
module. This differs from replacing the whole residual read slot or disabling
a producer everywhere.

```python
output, trace = model.forward_with_trace(
    inputs,
    edge_patches={"receiver": {"shared_slot": {"producer": replacement}}},
)
```

The nesting is receiver → read slot → producer → replacement write value. For
that reader, the executed coordinate becomes:

`current residual read + sum(replacement - producer's current effective write)`.

Other writers' contributions, the input encoding, shared state and other readers
are retained. The receiver and its descendants recompute normally. A producer's
current effective write includes existing gates/replacements and any earlier
edge changes in this execution, rather than assuming its baseline output.
Multiple named producer terms can be replaced in one read coordinate.

Every producer must write the selected slot and execute in a strictly earlier
layer than the receiver. Same-layer producers are invalid because layers read
simultaneously. In suffix execution, producers must execute inside that suffix;
the incoming boundary state alone does not identify each skipped producer's
contribution. Unknown receivers/producers and a whole-slot plus edge patch on
the same read coordinate are rejected.

Replacement values use the graph's existing scalar/batch broadcasting rules and
retain autograd connectivity. No state is modified in place. Trace fields include
`receiver.input_original` (shared-state read), `receiver.input` (executed read),
and `receiver.edge_delta.slot` (the signed correction). Effective producer writes
and all later states remain available in the trace. Without edge patches, trace
keys and execution retain their previous behavior.

For two producers writing 1 and 2 to the same slot, replacing the first
producer's contribution with zero at one reader changes that read from 3 to 2.
The shared slot remains 3, and another reader still receives 3. A whole-slot
replacement with zero instead gives the selected reader 0; globally disabling
the first producer also changes the other reader. These are distinct experiments.

This API isolates explicit additive residual contributions. It does not discover
causal edges or implement recursive causal scrubbing. If a producer influences
another writer upstream of the receiver, that indirect effect remains in the
other writer's current contribution. Donor selection and recursive path protocols remain separate work. The existing
donor `--read-slots` option still replaces whole read coordinates. No
continuous-domain or semantic guarantee is inferred.

## Save and replay an edge study

```bash
nmn research native edges --model graph.json --dataset dataset.json \
  --patches edge-patches.json --split evaluation \
  --provenance "Supplied producer replacements" --output edges.json
nmn research native replay edges.json --output replay.json
nmn research native export edges.json --output vault/edges
```

The patch file adds an outer condition name to the API mapping:

```json
{"remove-a-at-b": {"b": {"h": {"a": 0.0}}}}
```

`nmn.torch.edges.edge_study` executes all named conditions against the unchanged
graph. Each replacement is a finite scalar or array following the graph's
broadcast rules. Arrays follow the selected sample-ID order, which is saved in
the record; they do not trigger a donor lookup. The study does not currently
combine background interventions or whole-slot patches with edge patches.

`nmn.edge-study.v1` retains model/data identities, sample IDs, normalized numeric
patch values, baseline outputs, per-condition signed and absolute output deltas,
and complete traces including receiver edge corrections. Output columns follow
`protocol.output_order`. The provenance is supplied, not independently verified.
Nonfinite executed tensors are labeled `nonfinite-observation` when a record can
be produced; invalid arithmetic or snapshot construction can instead fail.

Replay reconstructs the model and executes the saved edge conditions, comparing
all outputs, deltas and trace values with declared tolerances. Export creates an
Obsidian note and raw data copy; the report command recognizes the schema. No
semantic reference, threshold, protected-output predicate, or population guarantee
is inferred from zero observed deltas. Cost counts one baseline forward and one
forward per condition; kernel geometry in snapshot collection is additional.
