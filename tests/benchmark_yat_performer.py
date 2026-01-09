"""Comprehensive Benchmark for Spherical Yat-Performer.

Tests:
1. Approximation quality vs standard YAT attention
2. Inference speed at various context lengths
3. Memory consumption
4. Gradient magnitude analysis (vanishing/exploding)
"""

import time
import gc
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from flax import nnx

# For memory tracking
try:
    from jax.lib import xla_bridge
    HAS_GPU = xla_bridge.get_backend().platform == 'gpu'
except:
    HAS_GPU = False

from nmn.nnx.attention import RotaryYatAttention
from nmn.nnx.attention.yat_attention import yat_attention, normalize_qk
from nmn.nnx.attention.spherical_yat_performer import (
    yat_tp_attention,
    create_yat_tp_projection,
    yat_tp_features,
)
from nmn.nnx.attention.yat_attention import yat_attention_weights


def print_header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)


def get_memory_mb():
    """Get current memory usage in MB."""
    try:
        # For GPU
        if HAS_GPU:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            return float(result.stdout.strip().split('\n')[0])
    except:
        pass
    return 0.0


def benchmark_approximation_quality():
    """Compare Performer approximation to exact YAT attention."""
    print_header("1. APPROXIMATION QUALITY")
    
    key = random.PRNGKey(42)
    batch, heads, head_dim = 2, 8, 64
    
    print("\nComparing Performer vs Exact YAT attention outputs:")
    print(f"{'Seq Len':>8} | {'Cosine Sim':>12} | {'L2 Error':>12} | {'Max Diff':>12}")
    print("-" * 52)
    
    results = []
    for seq_len in [32, 64, 128, 256, 512]:
        k1, k2, k3, k4 = random.split(key, 4)
        
        q = random.normal(k1, (batch, seq_len, heads, head_dim))
        k = random.normal(k2, (batch, seq_len, heads, head_dim))
        v = random.normal(k3, (batch, seq_len, heads, head_dim))
        
        # Exact YAT attention (quadratic)
        q_norm, k_norm = normalize_qk(q, k)
        exact_out = yat_attention(q_norm, k_norm, v, epsilon=1e-5)
        
        # Performer approximation
        params = create_yat_tp_projection(k4, head_dim, num_prf_features=64, num_quad_nodes=8)
        
        # Normalize inputs for performer too
        q_p, k_p = normalize_qk(q, k)
        approx_out = yat_tp_attention(q_p, k_p, v, params, causal=False, gradient_scaling=True)
        
        # Metrics
        exact_flat = exact_out.reshape(-1)
        approx_flat = approx_out.reshape(-1)
        
        # Avoid division by zero
        norm_exact = jnp.linalg.norm(exact_flat)
        norm_approx = jnp.linalg.norm(approx_flat)
        
        if norm_exact > 0 and norm_approx > 0:
            cosine_sim = float(jnp.sum(exact_flat * approx_flat) / (norm_exact * norm_approx))
            l2_error = float(jnp.linalg.norm(exact_flat - approx_flat) / norm_exact)
        else:
            cosine_sim = 0.0
            l2_error = 0.0
            
        max_diff = float(jnp.max(jnp.abs(exact_out - approx_out)))
        
        print(f"{seq_len:>8} | {cosine_sim:>12.4f} | {l2_error:>12.4f} | {max_diff:>12.4f}")
        results.append((seq_len, cosine_sim, l2_error))
    
    return results


def benchmark_inference_speed():
    """Measure inference speed at various context lengths."""
    print_header("2. INFERENCE SPEED")
    
    key = random.PRNGKey(42)
    batch, heads, head_dim = 1, 8, 64
    embed_dim = heads * head_dim
    num_warmup = 3
    num_runs = 10
    
    print("\nLatency comparison (ms):")
    print(f"{'Seq Len':>8} | {'Exact YAT':>12} | {'Performer':>12} | {'Speedup':>10}")
    print("-" * 48)
    
    results = []
    # Reduced sequence lengths for CPU testing stability
    for seq_len in [128, 256, 512, 1024]:
        k1, k2, k3, k4 = random.split(key, 4)
        
        q = random.normal(k1, (batch, seq_len, heads, head_dim))
        k = random.normal(k2, (batch, seq_len, heads, head_dim))
        v = random.normal(k3, (batch, seq_len, heads, head_dim))
        
        params = create_yat_tp_projection(k4, head_dim, num_prf_features=64, num_quad_nodes=8)
        
        # JIT compile
        exact_fn = jax.jit(lambda q, k, v: yat_attention(*normalize_qk(q, k), v, epsilon=1e-5))
        perf_fn = jax.jit(lambda q, k, v: yat_tp_attention(*normalize_qk(q, k), v, params, causal=False))
        
        # Warmup
        for _ in range(num_warmup):
            _ = exact_fn(q, k, v).block_until_ready()
            _ = perf_fn(q, k, v).block_until_ready()
        
        # Benchmark exact
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = exact_fn(q, k, v).block_until_ready()
        exact_time = (time.perf_counter() - start) / num_runs * 1000
        
        # Benchmark performer
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = perf_fn(q, k, v).block_until_ready()
        perf_time = (time.perf_counter() - start) / num_runs * 1000
        
        speedup = exact_time / perf_time if perf_time > 0 else 0
        
        print(f"{seq_len:>8} | {exact_time:>12.2f} | {perf_time:>12.2f} | {speedup:>10.2f}x")
        results.append((seq_len, exact_time, perf_time))
        
        gc.collect()
    
    return results


def benchmark_memory():
    """Measure memory consumption at various context lengths."""
    print_header("3. MEMORY CONSUMPTION")
    
    # Simple calculation due to environment limitations
    print("\nTheoretical memory usage (activations only):")
    print(f"{'Seq Len':>8} | {'Exact (L²)':>12} | {'Performer':>12} | {'Ratio':>10}")
    print("-" * 48)
    
    batch, heads = 1, 8
    
    for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192]:
        # Exact: L^2 maps per head
        exact_feats = seq_len * seq_len * heads
        
        # Performer: L * M * H maps
        # M = num_scales (8) * num_prf (64) = 512
        m = 512
        perf_feats = seq_len * m * heads
        
        # Assuming float32 (4 bytes)
        exact_mb = exact_feats * 4 / (1024*1024)
        perf_mb = perf_feats * 4 / (1024*1024)
        
        ratio = exact_mb / perf_mb if perf_mb > 0 else 0
        
        print(f"{seq_len:>8} | {exact_mb:>12.2f} | {perf_mb:>12.2f} | {ratio:>10.2f}x")


def benchmark_gradient_analysis():
    """Analyze gradient magnitudes across layers and sequence lengths."""
    print_header("4. GRADIENT ANALYSIS")
    
    key = random.PRNGKey(42)
    batch, heads, head_dim = 2, 8, 64
    
    print("\nGradient magnitude analysis:")
    print(f"{'Seq Len':>8} | {'grad_Q max':>12} | {'grad_V max':>12} | {'Status':>12}")
    print("-" * 52)
    
    for seq_len in [64, 128, 256, 512, 1024]:
        k1, k2, k3, k4 = random.split(key, 4)
        
        q = random.normal(k1, (batch, seq_len, heads, head_dim))
        k = random.normal(k2, (batch, seq_len, heads, head_dim))
        v = random.normal(k3, (batch, seq_len, heads, head_dim))
        
        params = create_yat_tp_projection(k4, head_dim, num_prf_features=64, num_quad_nodes=8)
        
        def loss_fn(q, k, v):
            # Normalize inside loss to capture normalization gradients too
            q_n, k_n = normalize_qk(q, k)
            out = yat_tp_attention(q_n, k_n, v, params, causal=False)
            return jnp.mean(out ** 2)
        
        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
        grad_q, grad_k, grad_v = grad_fn(q, k, v)
        
        max_q = float(jnp.max(jnp.abs(grad_q)))
        max_v = float(jnp.max(jnp.abs(grad_v)))
        
        if max_q > 100: status = "⚠️ EXPLODE"
        elif max_q < 1e-8: status = "⚠️ VANISH"
        else: status = "✅ HEALTHY"
        
        print(f"{seq_len:>8} | {max_q:>12.2e} | {max_v:>12.2e} | {status:>12}")


def benchmark_gradient_flow_depth():
    """Test gradient flow through multiple attention layers."""
    print_header("5. GRADIENT FLOW THROUGH DEPTH")
    
    key = random.PRNGKey(42)
    batch, seq_len, heads, head_dim = 2, 128, 8, 64
    embed_dim = heads * head_dim
    
    print("\nGradient through stacked RotaryYatAttention layers:")
    print(f"{'Layers':>8} | {'Input Grad':>12} | {'Output':>12} | {'Ratio':>10}")
    print("-" * 48)
    
    for num_layers in [1, 2, 4, 8]:
        rngs = nnx.Rngs(42)
        
        layers = []
        for i in range(num_layers):
            layer = RotaryYatAttention(
                embed_dim=embed_dim,
                num_heads=heads,
                max_seq_len=1024,
                use_performer=True,
                num_features=64, # This maps to prf_features
                rngs=rngs,
            )
            layers.append(layer)
        
        x = random.normal(key, (batch, seq_len, embed_dim))
        
        def forward(x, layers):
            for layer in layers:
                x = x + layer(x, deterministic=True)
            return jnp.mean(x ** 2)
        
        grad_fn = jax.grad(forward)
        grad_x = grad_fn(x, layers)
        
        input_grad = float(jnp.max(jnp.abs(grad_x)))
        
        # Check output mag
        def forward_out(x, layers):
            for layer in layers:
                x = x + layer(x, deterministic=True)
            return x
        out = forward_out(x, layers)
        out_mag = float(jnp.std(out))
        
        ratio = out_mag / float(jnp.std(x))
        
        print(f"{num_layers:>8} | {input_grad:>12.2e} | {out_mag:>12.2f} | {ratio:>10.2f}")


def main():
    print("="*60)
    print("MULTI-SCALE YAT PERFORMER BENCHMARK")
    print("="*60)
    
    benchmark_approximation_quality()
    benchmark_inference_speed()
    benchmark_memory()
    benchmark_gradient_analysis()
    benchmark_gradient_flow_depth()
    
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    main()
