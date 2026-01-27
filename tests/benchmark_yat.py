
import os
import sys
import time
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import psutil
import gc

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from nmn.nnx.attention.yat_attention import yat_attention
from nmn.nnx.attention.spherical_yat_performer import create_yat_tp_projection, yat_tp_attention, yat_tp_features

def benchmark():
    print("="*60)
    print("BENCHMARK: Standard YAT vs YAT Performer")
    print("="*60)
    
    key = random.PRNGKey(42)
    B = 2
    H = 4
    D = 64
    
    # Configs
    # 1. Standard (Exact)
    # 2. Performer (Current Optimized: M=128)
    
    # Create Performer Params (Optimized)
    # 8 feats, 2 scales, 8 sketch -> Total 128
    perf_params = create_yat_tp_projection(
        key, head_dim=D, 
        num_prf_features=8, 
        num_quad_nodes=2, 
        epsilon=1e-5, 
        poly_sketch_dim=8
    )
    
    seq_lengths = [256, 1024, 2048, 4096]
    
    results = {
        'L': seq_lengths,
        'std_time': [], 'perf_time': [],
        'std_mem': [], 'perf_mem': [], # RSS Delta (MB)
        'error': []
    }
    
    process = psutil.Process()
    
    def get_mem():
        gc.collect()
        return process.memory_info().rss / 1024 / 1024

    for L in seq_lengths:
        print(f"\nTesting Sequence Length L={L}...")
        
        # Clear previous JIT cache/garbage to get clean baseline
        jax.clear_caches()
        gc.collect()
        base_mem = get_mem()
        print(f"  Base Mem: {base_mem:.2f} MB")
        
        # Init vars
        out_std = None
        out_std_ref = None
        out_perf = None
        out_exact = None
        
        key, subkey = random.split(key)
        q = random.normal(subkey, (B, L, H, D))
        k = random.normal(subkey, (B, L, H, D))
        v = random.normal(subkey, (B, L, H, D))
        
        # --- Standard YAT ---
        # Need mask? causal=True
        # Standard yat_attention expects inputs, but usually handles causal via mask argument or loop?
        # Checking yat_attention signature... it has `bias` for mask.
        # We'll create causal mask.
        # mask[i, j] = 0 if i >= j else -inf
        mask = jnp.tril(jnp.ones((L, L))) 
        mask = (1.0 - mask) * -1e9
        mask = mask[None, None, :, :] # Broadcast B, H
        
        # --- Exact Linear YAT (The True Target) ---
        @jax.jit
        def run_exact_linear(q, k, v, mask):
             # 1. Compute Matrix: W = (q.k)^2 / (||q-k||^2 + eps)
             # q: [B, L, H, D]
             # dot: [B, H, L, L] via einsum
             dot = jnp.einsum('...qhd,...khd->...hqk', q, k)
             # norms
             q_norm_sq = jnp.sum(q**2, axis=-1, keepdims=True) # [... L, H, 1]
             k_norm_sq = jnp.sum(k**2, axis=-1, keepdims=True)
             
             # distance
             # q_norm: broadcast to [B, H, L, 1]
             # k_norm: broadcast to [B, H, 1, L]
             q_sq = jnp.transpose(q_norm_sq, (0, 2, 1, 3)) # B, H, L, 1
             k_sq = jnp.transpose(k_norm_sq, (0, 2, 1, 3)).transpose((0, 1, 3, 2)) # B, H, 1, L
             
             dist = q_sq + k_sq - 2*dot
             
             weights = (dot**2) / (dist + 1e-5)
             
             # Causal Mask (Zero out future)
             # mask has -1e9. For linear attention (sum), we want 0.
             causal_mask = jnp.tril(jnp.ones((L, L)))
             causal_mask = causal_mask[None, None, :, :]
             weights = weights * causal_mask
             
             # 2. Normalize: Out = (W @ V) / (W @ 1)
             den = jnp.sum(weights, axis=-1, keepdims=True) + 1e-5
             # den shape: [B, H, L, 1]. Num shape: [B, L, H, D]
             # Transpose den to [B, L, H, 1]
             den = den.transpose(0, 2, 1, 3)
             
             num = jnp.einsum('...hqk,...khd->...qhd', weights, v)
             
             return num / den

             return num / den

        # --- Standard Softmax YAT ---
        @jax.jit
        def run_std(q, k, v, mask):
             # yat_attention signature: q, k, v, bias, ...
             return yat_attention(q, k, v, bias=mask)

        # Warmup and Measure Mem
        try:
            # Measure Peak roughly by snapshotting after run (imperfect for JAX, but indicative of persistent buffers)
            # Actually JAX keeps buffers alive if 'out_std' is alive.
            
            # 1. Warmup
            _ = run_std(q, k, v, mask)
            jax.block_until_ready(_)
            
            # 2. Timing
            start = time.time()
            out_std = run_std(q, k, v, mask)
            out_std.block_until_ready()
            dt_std = time.time() - start
            print(f"  Standard Time: {dt_std:.4f}s")
            results['std_time'].append(dt_std)
            
            # 3. Memory (Hold reference to output)
            curr_mem = get_mem()
            mem_delta = curr_mem - base_mem
            print(f"  Standard Mem (Est): {mem_delta:.2f} MB")
            results['std_mem'].append(mem_delta)
            
        except Exception as e:
            print(f"  Standard: OOM/Fail ({e})")
            results['std_time'].append(None)
            results['std_mem'].append(None)
            out_std = None
            out_std_ref = None
            
        # Cleanup
        out_std_ref = out_std
        del out_std, _
        jax.clear_caches()
        gc.collect()
        base_mem_2 = get_mem()

        # --- Performer YAT ---
        @jax.jit
        def run_perf(q, k, v):
            return yat_tp_attention(q, k, v, perf_params, causal=True)
            
        try:
             # 1. Warmup
             _ = run_perf(q, k, v)
             jax.block_until_ready(_)
             
             # 2. Timing
             start = time.time()
             out_perf = run_perf(q, k, v)
             out_perf.block_until_ready()
             dt_perf = time.time() - start
             print(f"  Performer Time: {dt_perf:.4f}s")
             results['perf_time'].append(dt_perf)
             
             # 3. Memory
             curr_mem = get_mem()
             mem_delta = curr_mem - base_mem_2
             print(f"  Performer Mem (Est): {mem_delta:.2f} MB")
             results['perf_mem'].append(mem_delta)

        except Exception as e:
             print(f"  Performer: OOM/Fail ({e})")
             results['perf_time'].append(None)
             results['perf_mem'].append(None)
             out_perf = None
             
        # --- Exact Linear YAT Run ---
        try:
             _ = run_exact_linear(q, k, v, mask)
             jax.block_until_ready(_)
             out_exact = run_exact_linear(q, k, v, mask)
             out_exact.block_until_ready()
        except Exception as e:
             print(f"  Exact Linear Failed: {e}")  # Optional debug
             out_exact = None

        # --- Accuracy ---
        # Compare Performer vs Exact Linear (Fair comparison)
        if out_exact is not None and out_perf is not None:
            diff = jnp.linalg.norm(out_exact - out_perf)
            norm = jnp.linalg.norm(out_exact)
            rel_err_exact = float(diff / (norm + 1e-6))
            print(f"  Rel Error (vs Exact Linear): {rel_err_exact:.4f}")
            results['error'].append(rel_err_exact)
        elif out_std_ref is not None and out_perf is not None:
            # Fallback to Std if Exact failed (e.g. OOM)
            diff = jnp.linalg.norm(out_std_ref - out_perf)
            norm = jnp.linalg.norm(out_std_ref)
            rel_err = float(diff / (norm + 1e-6))
            print(f"  Rel Error (vs Softmax): {rel_err:.4f}")
            results['error'].append(rel_err)
        else:
            results['error'].append(None)

    print("\nSUMMARY:")
    print(f"{'L':<6} | {'Std Time':<10} | {'Perf Time':<10} | {'Std Mem':<10} | {'Perf Mem':<10} | {'Rel Err':<8}")
    for i, L in enumerate(seq_lengths):
        st = f"{results['std_time'][i]:.4f}" if results['std_time'][i] else "Fail"
        pt = f"{results['perf_time'][i]:.4f}" if results['perf_time'][i] else "Fail"
        sm = f"{results['std_mem'][i]:.2f}" if results['std_mem'][i] else "-"
        pm = f"{results['perf_mem'][i]:.2f}" if results['perf_mem'][i] else "-"
        err = f"{results['error'][i]:.4f}" if results['error'][i] else "-"
        print(f"{L:<6} | {st:<10} | {pt:<10} | {sm:<10} | {pm:<10} | {err:<8}")

if __name__ == "__main__":
    benchmark()
