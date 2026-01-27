"""Test gradient scaling fix."""
import jax
import jax.numpy as jnp
from jax import random
from nmn.nnx.attention.spherical_yat_performer import yat_tp_attention, create_yat_tp_projection

print('Testing gradient scaling fix...')
key = random.PRNGKey(42)
batch, heads, head_dim = 2, 8, 64

params = create_yat_tp_projection(key, head_dim, num_prf_features=64, num_quad_nodes=8)

print()
print('WITHOUT gradient scaling:')
print(f"{'Seq Len':>8} | {'grad_Q max':>12} | {'grad_V max':>12}")
print('-' * 38)
for seq_len in [64, 256, 1024]:
    q = random.normal(key, (batch, seq_len, heads, head_dim))
    k = random.normal(random.split(key)[0], (batch, seq_len, heads, head_dim))
    v = random.normal(random.split(key)[1], (batch, seq_len, heads, head_dim))
    
    def loss_fn(q, k, v):
        out = yat_tp_attention(q, k, v, params, gradient_scaling=False)
        return jnp.mean(out ** 2)
    
    grad_q, _, grad_v = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    print(f"{seq_len:>8} | {float(jnp.max(jnp.abs(grad_q))):>12.2e} | {float(jnp.max(jnp.abs(grad_v))):>12.2e}")

print()
print('WITH gradient scaling (default):')
print(f"{'Seq Len':>8} | {'grad_Q max':>12} | {'grad_V max':>12}")
print('-' * 38)
for seq_len in [64, 256, 1024]:
    q = random.normal(key, (batch, seq_len, heads, head_dim))
    k = random.normal(random.split(key)[0], (batch, seq_len, heads, head_dim))
    v = random.normal(random.split(key)[1], (batch, seq_len, heads, head_dim))
    
    def loss_fn(q, k, v):
        out = yat_tp_attention(q, k, v, params, gradient_scaling=True)
        return jnp.mean(out ** 2)
    
    grad_q, _, grad_v = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    print(f"{seq_len:>8} | {float(jnp.max(jnp.abs(grad_q))):>12.2e} | {float(jnp.max(jnp.abs(grad_v))):>12.2e}")

print()
print('Done!')
