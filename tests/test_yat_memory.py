
import os
import jax
import jax.numpy as jnp
from jax import random
import time
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
from nmn.nnx.attention.spherical_yat_performer import create_yat_tp_projection, yat_tp_attention

def test_memory_and_shapes():
    print("Initializing params...")
    key = random.PRNGKey(0)
    
    # Dimensions from GPT2Config (Large ones that caused OOM)
    # B=16, L=1024, H=12, D=64
    B, L, H, D = 8, 512, 12, 64 # Larger scale
    
    # Original config causing 22TB
    # num_features_per_scale = 32
    # num_scales = 8
    # poly_sketch_dim = 64
    
    print(f"Testing with Large Config: B={B}, L={L}, H={H}, D={D}")
    
    # Optimized Config from gpt.py (Safe)
    # num_features=8, scales=2, poly=8 => Total 128
    
    params = create_yat_tp_projection(
        key, head_dim=D, 
        num_prf_features=8, 
        num_quad_nodes=2, 
        epsilon=1e-5, 
        poly_sketch_dim=8
    )
    
    print("\nParams created.")
    print(f"Sketch H shape: {params['sketch_params']['h'].shape}")
    print(f"Sketch S shape: {params['sketch_params']['s'].shape}")
    
    key, subkey = random.split(key)
    q = random.normal(subkey, (B, L, H, D))
    k = random.normal(subkey, (B, L, H, D))
    v = random.normal(subkey, (B, L, H, D))
    
    print(f"\nInput shape: {q.shape}")
    
    # JIT compile to check XLA allocations
    @jax.jit
    def forward(q, k, v):
        return yat_tp_attention(q, k, v, params, causal=True)
    
    print("Compiling...")
    start = time.time()
    out = forward(q, k, v)
    jax.block_until_ready(out)
    print(f"Compilation + Run took {time.time() - start:.2f}s")
    print(f"Output shape: {out.shape}")
    
    # Check gradients memory
    print("\nChecking Gradients...")
    @jax.jit
    def loss_fn(q, k, v):
        out = yat_tp_attention(q, k, v, params, causal=True)
        return jnp.sum(out)
        
    grad_fn = jax.grad(loss_fn)
    
    print("Compiling Grad...")
    start = time.time()
    grads = grad_fn(q, k, v)
    jax.block_until_ready(grads[0])
    print(f"Grad Run took {time.time() - start:.2f}s")

if __name__ == "__main__":
    test_memory_and_shapes()
