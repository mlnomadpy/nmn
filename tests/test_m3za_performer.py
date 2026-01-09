
import sys
from unittest.mock import MagicMock

# Mock heavy/problematic dependencies
sys.modules['mteb'] = MagicMock()
sys.modules['datasets'] = MagicMock()

import jax
import jax.numpy as jnp
from flax import nnx
from nmn.nnx.examples.language import m3za
# Disable mesh partitioning for simple local test
m3za.mesh = None

from nmn.nnx.examples.language.m3za import MiniBERT

def test_m3za_forward():
    print("Testing M3ZA MiniBERT with Performer...")
    
    # Config
    config = {
        'maxlen': 128,
        'vocab_size': 1000,
        'embed_dim': 64,
        'num_heads': 4,
        'feed_forward_dim': 256,
        'num_transformer_blocks': 2,
    }
    
    rngs = nnx.Rngs(42)
    model = MiniBERT(
        maxlen=config['maxlen'],
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        feed_forward_dim=config['feed_forward_dim'],
        num_transformer_blocks=config['num_transformer_blocks'],
        rngs=rngs
    )
    
    # Dummy input
    batch_size = 2
    inputs = jax.random.randint(jax.random.PRNGKey(0), (batch_size, config['maxlen']), 0, config['vocab_size'])
    
    # Forward pass
    print("Running forward pass...")
    logits = model(inputs, training=False)
    
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, config['maxlen'], config['vocab_size'])
    print("Forward pass successful!")
    
    # Check gradients
    print("Checking gradients...")
    def loss_fn(model, x):
        logits = model(x, training=True)
        return jnp.mean(logits ** 2)
        
    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(model, inputs)
    print("Gradient computation successful!")

if __name__ == "__main__":
    test_m3za_forward()
