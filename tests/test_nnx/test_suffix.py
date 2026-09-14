"""Complete-state replay and derivatives through the native NNX suffix."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nmn.nnx import ThreeNeuronYat
from nmn.nnx.replay import replay_native_record
from nmn.nnx.suffix import suffix_study
from nmn.research.datasets import ResearchDataset, ResearchSample


def test_boundary_reconstruction_jit_and_gradients():
    model = ThreeNeuronYat.reference()
    state = jnp.array([[1.0, 1.0, 1.0]])
    output, _ = nnx.jit(lambda m, s: m.forward_from_state(s, start_layer=1))(
        model, state
    )
    np.testing.assert_allclose(output, [[4.0, 1.0]])
    derivative = jax.jacfwd(lambda s: model.forward_from_state(s, start_layer=1)[0])(
        state[0]
    )
    np.testing.assert_allclose(derivative, [[4.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    for boundary, value in [(0, state[:, :2]), (2, output)]:
        np.testing.assert_allclose(
            model.forward_from_state(value, start_layer=boundary)[0], output
        )
    with pytest.raises(ValueError, match="executed"):
        model.forward_from_state(
            state, start_layer=1, interventions={"h": {"gate": 0.0}}
        )
    with pytest.raises(ValueError, match="width"):
        model.forward_from_state(state[:, :2], start_layer=1)


def test_suffix_study_replays_and_detects_altered_effects():
    dataset = ResearchDataset(
        [ResearchSample("a", (1.0, 1.0), "evaluation", "a")],
        name="boundary",
        provenance="synthetic",
    )
    record = suffix_study(
        ThreeNeuronYat.reference(),
        dataset,
        start_layer=1,
        states={"zero": {"a": [0.0, 1.0, 1.0]}},
        provenance="supplied hidden-state deletion",
    )
    assert record["baseline_reconstruction_error"] == [[0.0, 0.0]]
    assert record["variants"]["zero"]["outputs"] == [[0.5, 1.0]]
    assert replay_native_record(record)["status"] == "matched"
    record["variants"]["zero"]["output_delta"][0][1] = 1.0
    result = replay_native_record(record)
    assert result["status"] == "mismatch"
    assert any(
        m["path"] == "/variants/zero/output_delta/0/1" for m in result["mismatches"]
    )
