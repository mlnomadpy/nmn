"""Research arithmetic, native edits and trainable NNX state."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nmn.nnx import ThreeNeuronYat, YatExpansion
from nmn.nnx.research import collect_research_data
from nmn.research.datasets import ResearchDataset, ResearchSample


def test_live_edits_jit_and_parameter_gradients():
    model = ThreeNeuronYat.reference()
    x = jnp.ones((1, 2))
    np.testing.assert_allclose(nnx.jit(lambda m, p: m(p))(model, x), [[4.0, 1.0]])
    np.testing.assert_allclose(model(x, {"h": {"gate": 0.0}}), [[0.5, 1.0]])
    np.testing.assert_allclose(
        model(x, {"h": {"gate": 0.0, "replacement": 1.0}}), [[4.0, 1.0]]
    )
    state = nnx.state(model, nnx.Param)
    assert sum(a.size for a in jax.tree.leaves(state)) == 7
    grads = nnx.grad(lambda m: jnp.sum(m(x)))(model)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves(grads))
    assert any(np.any(a != 0) for a in jax.tree.leaves(grads))
    derivative = jax.jacfwd(lambda g: model(x, {"h": {"gate": g}}))(jnp.array(1.0))
    np.testing.assert_allclose(derivative, [[4.0, 0.0]])


def test_direct_curvature_and_validation():
    model = ThreeNeuronYat.reference()
    curvature = jax.hessian(lambda x: model.y(x)[0])(jnp.ones(2))
    np.testing.assert_allclose(curvature, [[-6.0, 2.0], [2.0, -6.0]])
    with pytest.raises(ValueError):
        model(jnp.ones(2), {"unknown": {"gate": 0.0}})
    with pytest.raises(ValueError):
        model(jnp.ones(2), {"h": {"ignored": 1.0}})
    with pytest.raises(ValueError):
        YatExpansion(1, 1, epsilon=0.0, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        YatExpansion(1, 1, dtype=jnp.float16, rngs=nnx.Rngs(0))


def test_collector_preserves_population_and_finite_evidence():
    dataset = ResearchDataset(
        [
            ResearchSample("a", (1.0, 1.0), "evaluation", "a"),
            ResearchSample("b", (0.0, 0.0), "training", "b"),
        ],
        name="fixture",
        provenance="synthetic",
    )
    record = collect_research_data(
        ThreeNeuronYat.reference(),
        dataset,
        split="evaluation",
        derivatives=False,
        edits={"delete": {"h": {"gate": 0.0}}},
    )
    assert record["sample_ids"] == ["a"]
    assert record["dataset_sha256"] == dataset.sha256
    assert record["edits"]["delete"]["outputs"] == [[0.5, 1.0]]
    assert record["capabilities"]["interval_certificates"] is False
    with pytest.raises(ValueError):
        collect_research_data(
            ThreeNeuronYat.reference(),
            dataset,
            split="evaluation",
            edits={"bad": {"h": {"gate": float("nan")}}},
        )
