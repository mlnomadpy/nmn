import importlib
import sys
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


@pytest.fixture(scope="module")
def m3za_modules():
    """Import example-only dependencies without polluting the full NNX suite."""
    mocked_dependencies = {
        name: MagicMock() for name in ("mteb", "datasets", "wandb", "tokenizers")
    }
    module_names = (
        "nmn.nnx.examples.language.m3za",
        "nmn.nnx.examples.language.m3za_perf",
    )
    parent = importlib.import_module("nmn.nnx.examples.language")
    with patch.dict(sys.modules, mocked_dependencies):
        modules = tuple(importlib.import_module(name) for name in module_names)
        for module in modules:
            module.mesh = None
        yield modules

    for name, module in zip(module_names, modules):
        sys.modules.pop(name, None)
        child_name = name.rsplit(".", 1)[-1]
        if getattr(parent, child_name, None) is module:
            delattr(parent, child_name)


def test_m3za_forward(m3za_modules):
    m3za, _ = m3za_modules
    # Config
    config = {
        "maxlen": 128,
        "vocab_size": 1000,
        "embed_dim": 64,
        "num_heads": 4,
        "feed_forward_dim": 256,
        "num_transformer_blocks": 2,
    }

    rngs = nnx.Rngs(42)
    model = m3za.MiniBERT(
        maxlen=config["maxlen"],
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        feed_forward_dim=config["feed_forward_dim"],
        num_transformer_blocks=config["num_transformer_blocks"],
        rngs=rngs,
    )

    # Dummy input
    batch_size = 2
    inputs = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, config["maxlen"]), 0, config["vocab_size"]
    )

    # Forward pass
    logits = model(inputs, training=False)

    assert logits.shape == (batch_size, config["maxlen"], config["vocab_size"])

    def loss_fn(model, x):
        logits = model(x, training=True)
        return jnp.mean(logits**2)

    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(model, inputs)
    gradient_leaves = jax.tree.leaves(grads)
    assert gradient_leaves
    assert all(jnp.isfinite(leaf).all() for leaf in gradient_leaves)


def test_m3za_performer_forward_and_optimizer_step(m3za_modules):
    """Keep the performer example executable against the current NNX API."""
    import optax

    _, m3za_perf = m3za_modules
    model = m3za_perf.MiniBERT(
        maxlen=8,
        vocab_size=32,
        embed_dim=8,
        num_heads=2,
        feed_forward_dim=16,
        num_transformer_blocks=1,
        rngs=nnx.Rngs(7),
    )
    inputs = jnp.ones((1, 8), dtype=jnp.int32)
    labels = jnp.ones((1, 8), dtype=jnp.int32)
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)

    logits = model(inputs, training=False)
    assert logits.shape == (1, 8, 32)

    loss = m3za_perf.train_step_mlm(
        model, optimizer, {"input_ids": inputs, "labels": labels}
    )[0]
    assert jnp.isfinite(loss)
