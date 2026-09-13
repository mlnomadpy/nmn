# Intermediate-state and suffix measurements

`YatGraph.forward_from_state(state, start_layer=k)` executes layers `k` onward
and returns outputs plus downstream traces. Boundary `k` is `state.k` in a full
trace. At `k == len(layers)` only the fixed readout executes. The state tensor
has one coordinate per graph slot; it is neither detached nor mutated. Gradients
reach it and the executed parameters. Write interventions and read patches are
supported, but controls for modules before the boundary are rejected.

```bash
nmn research native suffix --model graph.json --dataset dataset.json \
  --start-layer 1 --states states.json --provenance 'state construction v1' \
  --output suffix.json
nmn research native replay suffix.json --output suffix-replay.json
nmn research native export suffix.json --output suffix-note
```

The states file maps variant names to sample-ID/state-vector mappings. Each must
cover exactly the selected population (`--split` is optional), in the model's
slot order. For slots `["x", "h", "y", "p"]`, one example is:

```json
{"remove-hidden": {"base": [1, 0, 0, 0], "donor": [0, 0, 0, 0]}}
```

`nmn.torch.suffix.suffix_study` exposes the same workflow. It retains the unchanged
model snapshot, original boundary state, baseline suffix reconstruction error,
supplied states, state/output differences and complete downstream traces.
This supplies data for investigating state reduction and accumulated downstream
error. It does not learn an encoder/decoder, establish state realizability or
prove invariant-domain closure. A supplied full-state change affects every
subsequent reader of changed slots, unlike a receiving-module read patch.
