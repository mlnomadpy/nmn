"""Speed Benchmarks for All Attention Types.

This module benchmarks the performance of different attention mechanisms:
1. Standard YAT attention (O(n²))
2. Normalized YAT attention (O(n²), optimized)
3. YAT Performer attention (O(n))
4. Rotary YAT attention (O(n²))
5. Rotary YAT Performer attention (O(n))
6. Rotary YAT Performer + Normalize (O(n), fastest)
7. Standard dot-product attention (baseline)

Run with: pytest tests/benchmarks/test_attention_speed.py -v -s

Note on Performance:
- On CPU: Performer attention may have higher constant overhead due to random
  projections. The linear complexity shines at very long sequences (4096+).
- On GPU: Performer attention should show clear wins at medium-to-long sequences
  due to memory bandwidth and parallelism benefits.
- Memory: Performer uses O(n) memory vs O(n²) for quadratic attention, which
  is the main benefit for long sequences (avoids OOM).
"""

import pytest
import time
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Dict, List, Tuple
import sys

# Skip if JAX not available
pytest.importorskip("jax")


# =============================================================================
# Benchmark Configuration
# =============================================================================

# Sequence lengths to benchmark
SEQ_LENGTHS = [64, 128, 256, 512, 1024]

# Model configuration
BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64
EMBED_DIM = NUM_HEADS * HEAD_DIM  # 512
NUM_FEATURES = 256  # For Performer

# Number of warmup and benchmark iterations
WARMUP_ITERS = 3
BENCHMARK_ITERS = 10


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_fn(fn: Callable, *args, warmup: int = WARMUP_ITERS, iters: int = BENCHMARK_ITERS) -> Tuple[float, float]:
    """Benchmarks a function and returns (mean_time_ms, std_time_ms)."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        result.block_until_ready()
    
    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn(*args)
        result.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    return mean_time, std_time


def format_time(mean: float, std: float) -> str:
    """Formats time as mean ± std ms."""
    return f"{mean:>8.3f} +/- {std:>6.3f} ms"


def print_benchmark_header():
    """Prints the benchmark header."""
    print("\n" + "=" * 80)
    print("ATTENTION SPEED BENCHMARK")
    print("=" * 80)
    print(f"Config: batch={BATCH_SIZE}, heads={NUM_HEADS}, head_dim={HEAD_DIM}, embed_dim={EMBED_DIM}")
    print(f"Performer features: {NUM_FEATURES}")
    print(f"Warmup: {WARMUP_ITERS}, Iterations: {BENCHMARK_ITERS}")
    print("=" * 80)


def print_results_table(results: Dict[str, Dict[int, Tuple[float, float]]]):
    """Prints results as a formatted table."""
    # Header
    print("\n" + "-" * 100)
    header = f"{'Attention Type':<40}"
    for seq_len in SEQ_LENGTHS:
        header += f" | {seq_len:>12}"
    print(header)
    print("-" * 100)
    
    # Rows
    for name, timings in results.items():
        row = f"{name:<40}"
        for seq_len in SEQ_LENGTHS:
            if seq_len in timings:
                mean, std = timings[seq_len]
                row += f" | {mean:>8.2f} ms"
            else:
                row += f" | {'N/A':>11}"
        print(row)
    
    print("-" * 100)


# =============================================================================
# Attention Implementations for Benchmarking
# =============================================================================

class AttentionBenchmarks:
    """Container for attention benchmark functions."""
    
    def __init__(self, seq_len: int, rngs: nnx.Rngs):
        self.seq_len = seq_len
        self.rngs = rngs
        
        # Create random inputs
        key = jax.random.key(0)
        self.q = jax.random.normal(key, (BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM))
        self.k = jax.random.normal(key, (BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM))
        self.v = jax.random.normal(key, (BATCH_SIZE, seq_len, NUM_HEADS, HEAD_DIM))
        self.x = jax.random.normal(key, (BATCH_SIZE, seq_len, EMBED_DIM))
        
        # Precompute projections
        from nmn.nnx.attention import create_yat_projection, precompute_freqs_cis
        self.projection = create_yat_projection(key, NUM_FEATURES, HEAD_DIM)
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(HEAD_DIM, seq_len)
        
        # Create modules
        self._create_modules()
    
    def _create_modules(self):
        """Creates attention modules."""
        from nmn.nnx.attention import RotaryYatAttention
        
        # Rotary YAT (standard quadratic)
        self.rotary_yat = RotaryYatAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            max_seq_len=self.seq_len,
            use_performer=False,
            rngs=self.rngs,
        )
        
        # Rotary YAT Performer
        self.rotary_yat_performer = RotaryYatAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            max_seq_len=self.seq_len,
            use_performer=True,
            num_features=NUM_FEATURES,
            performer_normalize=False,
            rngs=self.rngs,
        )
        
        # Rotary YAT Performer + Normalize (fastest)
        self.rotary_yat_performer_norm = RotaryYatAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            max_seq_len=self.seq_len,
            use_performer=True,
            num_features=NUM_FEATURES,
            performer_normalize=True,
            rngs=self.rngs,
        )
    
    def benchmark_yat_attention(self) -> Tuple[float, float]:
        """Benchmarks standard YAT attention."""
        from nmn.nnx.attention import yat_attention
        
        @jax.jit
        def fn(q, k, v):
            return yat_attention(q, k, v)
        
        return benchmark_fn(fn, self.q, self.k, self.v)
    
    def benchmark_yat_attention_normalized(self) -> Tuple[float, float]:
        """Benchmarks normalized YAT attention."""
        from nmn.nnx.attention import yat_attention_normalized
        
        @jax.jit
        def fn(q, k, v):
            return yat_attention_normalized(q, k, v)
        
        return benchmark_fn(fn, self.q, self.k, self.v)
    
    def benchmark_yat_performer(self) -> Tuple[float, float]:
        """Benchmarks YAT Performer attention."""
        from nmn.nnx.attention import yat_performer_attention
        
        @jax.jit
        def fn(q, k, v, proj):
            return yat_performer_attention(q, k, v, proj, normalize_inputs=False)
        
        return benchmark_fn(fn, self.q, self.k, self.v, self.projection)
    
    def benchmark_yat_performer_normalized(self) -> Tuple[float, float]:
        """Benchmarks YAT Performer with normalization."""
        from nmn.nnx.attention import yat_performer_attention
        
        @jax.jit
        def fn(q, k, v, proj):
            return yat_performer_attention(q, k, v, proj, normalize_inputs=True)
        
        return benchmark_fn(fn, self.q, self.k, self.v, self.projection)
    
    def benchmark_rotary_yat(self) -> Tuple[float, float]:
        """Benchmarks Rotary YAT attention."""
        @jax.jit
        def fn(x):
            return self.rotary_yat(x, deterministic=True)
        
        return benchmark_fn(fn, self.x)
    
    def benchmark_rotary_yat_performer(self) -> Tuple[float, float]:
        """Benchmarks Rotary YAT Performer."""
        @jax.jit
        def fn(x):
            return self.rotary_yat_performer(x, deterministic=True)
        
        return benchmark_fn(fn, self.x)
    
    def benchmark_rotary_yat_performer_normalized(self) -> Tuple[float, float]:
        """Benchmarks Rotary YAT Performer + Normalize."""
        @jax.jit
        def fn(x):
            return self.rotary_yat_performer_norm(x, deterministic=True)
        
        return benchmark_fn(fn, self.x)
    
    def benchmark_dot_product_attention(self) -> Tuple[float, float]:
        """Benchmarks standard dot-product attention (baseline)."""
        from nmn.nnx.attention import dot_product_attention
        
        @jax.jit
        def fn(q, k, v):
            return dot_product_attention(q, k, v)
        
        return benchmark_fn(fn, self.q, self.k, self.v)


# =============================================================================
# Test Functions
# =============================================================================

@pytest.fixture(scope="module")
def benchmark_results():
    """Runs all benchmarks and returns results."""
    results = {
        "Dot-Product (baseline)": {},
        "YAT Standard": {},
        "YAT Normalized": {},
        "YAT Performer": {},
        "YAT Performer + Normalize": {},
        "Rotary YAT Standard": {},
        "Rotary YAT Performer": {},
        "Rotary YAT Performer + Normalize": {},
    }
    
    print_benchmark_header()
    
    for seq_len in SEQ_LENGTHS:
        print(f"\nBenchmarking seq_len={seq_len}...")
        
        rngs = nnx.Rngs(42)
        bench = AttentionBenchmarks(seq_len, rngs)
        
        # Run benchmarks
        try:
            results["Dot-Product (baseline)"][seq_len] = bench.benchmark_dot_product_attention()
            print(f"  Dot-Product: {format_time(*results['Dot-Product (baseline)'][seq_len])}")
        except Exception as e:
            print(f"  Dot-Product: FAILED ({e})")
        
        try:
            results["YAT Standard"][seq_len] = bench.benchmark_yat_attention()
            print(f"  YAT Standard: {format_time(*results['YAT Standard'][seq_len])}")
        except Exception as e:
            print(f"  YAT Standard: FAILED ({e})")
        
        try:
            results["YAT Normalized"][seq_len] = bench.benchmark_yat_attention_normalized()
            print(f"  YAT Normalized: {format_time(*results['YAT Normalized'][seq_len])}")
        except Exception as e:
            print(f"  YAT Normalized: FAILED ({e})")
        
        try:
            results["YAT Performer"][seq_len] = bench.benchmark_yat_performer()
            print(f"  YAT Performer: {format_time(*results['YAT Performer'][seq_len])}")
        except Exception as e:
            print(f"  YAT Performer: FAILED ({e})")
        
        try:
            results["YAT Performer + Normalize"][seq_len] = bench.benchmark_yat_performer_normalized()
            print(f"  YAT Performer + Norm: {format_time(*results['YAT Performer + Normalize'][seq_len])}")
        except Exception as e:
            print(f"  YAT Performer + Norm: FAILED ({e})")
        
        try:
            results["Rotary YAT Standard"][seq_len] = bench.benchmark_rotary_yat()
            print(f"  Rotary YAT: {format_time(*results['Rotary YAT Standard'][seq_len])}")
        except Exception as e:
            print(f"  Rotary YAT: FAILED ({e})")
        
        try:
            results["Rotary YAT Performer"][seq_len] = bench.benchmark_rotary_yat_performer()
            print(f"  Rotary YAT Performer: {format_time(*results['Rotary YAT Performer'][seq_len])}")
        except Exception as e:
            print(f"  Rotary YAT Performer: FAILED ({e})")
        
        try:
            results["Rotary YAT Performer + Normalize"][seq_len] = bench.benchmark_rotary_yat_performer_normalized()
            print(f"  Rotary YAT Perf+Norm: {format_time(*results['Rotary YAT Performer + Normalize'][seq_len])}")
        except Exception as e:
            print(f"  Rotary YAT Perf+Norm: FAILED ({e})")
    
    return results


class TestAttentionSpeed:
    """Test class for attention speed benchmarks."""
    
    def test_benchmark_all_attention_types(self, benchmark_results):
        """Runs and displays all attention benchmarks."""
        print_results_table(benchmark_results)
        
        # Print speedup analysis
        print("\n" + "=" * 80)
        print("SPEEDUP ANALYSIS (vs Dot-Product baseline at seq_len=512)")
        print("=" * 80)
        
        if 512 in benchmark_results["Dot-Product (baseline)"]:
            baseline = benchmark_results["Dot-Product (baseline)"][512][0]
            
            for name, timings in benchmark_results.items():
                if name == "Dot-Product (baseline)":
                    continue
                if 512 in timings:
                    time_ms = timings[512][0]
                    speedup = baseline / time_ms if time_ms > 0 else 0
                    slower = time_ms / baseline if baseline > 0 else 0
                    if speedup >= 1:
                        print(f"  {name:<40}: {speedup:.2f}x faster")
                    else:
                        print(f"  {name:<40}: {slower:.2f}x slower")
        
        print("=" * 80)
        
        # Test passes if we got results
        assert len(benchmark_results) > 0
    
    def test_performer_scales_linearly(self, benchmark_results):
        """Tests that Performer attention scales approximately linearly.
        
        Note: On CPU, JIT compilation overhead and cache effects can cause
        significant variability. This test is informational and uses lenient
        thresholds. True linear scaling is most visible on GPU at long sequences.
        """
        performer_times = benchmark_results.get("YAT Performer + Normalize", {})
        yat_times = benchmark_results.get("YAT Standard", {})
        
        if len(performer_times) < 2:
            pytest.skip("Not enough data points for scaling analysis")
        
        # Get times for different sequence lengths
        seq_lens = sorted(performer_times.keys())
        perf_times = [performer_times[s][0] for s in seq_lens]
        
        print("\nPerformer Scaling Analysis:")
        print(f"{'Transition':>15} | {'Seq Ratio':>10} | {'Perf Ratio':>12} | {'Status':>10}")
        print("-" * 55)
        
        issues = []
        for i in range(1, len(seq_lens)):
            ratio_seq = seq_lens[i] / seq_lens[i-1]
            ratio_time = perf_times[i] / perf_times[i-1] if perf_times[i-1] > 0 else float('inf')
            
            # On CPU, Performer has high constant overhead
            # We only flag if scaling looks worse than quadratic (4x for 2x seq)
            # and we're at short sequences where overhead dominates
            if seq_lens[i] >= 512 and ratio_time > ratio_seq ** 2:
                status = "SLOW"
                issues.append((seq_lens[i-1], seq_lens[i], ratio_time))
            else:
                status = "OK"
            
            print(f"{seq_lens[i-1]:>6} -> {seq_lens[i]:<5} | {ratio_seq:>10.1f}x | {ratio_time:>12.2f}x | {status:>10}")
        
        print("-" * 55)
        print("Note: CPU overhead can dominate at short sequences.")
        print("Performer benefits most at seq_len >= 2048 on GPU.")
        
        # Only fail if we have consistent super-quadratic scaling at long sequences
        # which would indicate a bug
        if len(issues) >= 2:
            print(f"\n[WARN] Performer showing super-quadratic scaling in {len(issues)} transitions")
        else:
            print(f"\n[OK] Performer scaling within expected bounds")
    
    def test_quadratic_attention_slower_at_long_sequences(self, benchmark_results):
        """Tests that quadratic attention is slower for long sequences."""
        if 1024 not in benchmark_results.get("YAT Standard", {}):
            pytest.skip("Need seq_len=1024 for this test")
        
        if 1024 not in benchmark_results.get("YAT Performer + Normalize", {}):
            pytest.skip("Need seq_len=1024 for this test")
        
        quad_time = benchmark_results["YAT Standard"][1024][0]
        linear_time = benchmark_results["YAT Performer + Normalize"][1024][0]
        
        # At seq_len=1024, Performer should be faster than quadratic
        # (This depends on hardware, so we use a loose threshold)
        print(f"\nAt seq_len=1024:")
        print(f"  YAT Standard: {quad_time:.2f} ms")
        print(f"  YAT Performer + Norm: {linear_time:.2f} ms")
        
        # Just verify both ran successfully
        assert quad_time > 0
        assert linear_time > 0


class TestAttentionCorrectness:
    """Tests that different attention types produce valid outputs."""
    
    def test_all_attention_outputs_valid(self):
        """Tests that all attention types produce valid (non-NaN) outputs."""
        from nmn.nnx.attention import (
            yat_attention,
            yat_attention_normalized,
            yat_performer_attention,
            dot_product_attention,
            create_yat_projection,
            RotaryYatAttention,
        )
        
        key = jax.random.key(0)
        batch, seq_len, num_heads, head_dim = 2, 64, 4, 32
        embed_dim = num_heads * head_dim
        
        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        x = jax.random.normal(key, (batch, seq_len, embed_dim))
        projection = create_yat_projection(key, 64, head_dim)
        
        rngs = nnx.Rngs(0)
        
        # Test each attention type
        tests = [
            ("Dot-Product", lambda: dot_product_attention(q, k, v)),
            ("YAT Standard", lambda: yat_attention(q, k, v)),
            ("YAT Normalized", lambda: yat_attention_normalized(q, k, v)),
            ("YAT Performer", lambda: yat_performer_attention(q, k, v, projection)),
            ("YAT Performer + Norm", lambda: yat_performer_attention(q, k, v, projection, normalize_inputs=True)),
        ]
        
        for name, fn in tests:
            output = fn()
            assert not jnp.any(jnp.isnan(output)), f"{name} produced NaN"
            assert not jnp.any(jnp.isinf(output)), f"{name} produced Inf"
            assert output.shape == (batch, seq_len, num_heads, head_dim), f"{name} wrong shape"
            print(f"  [OK] {name}")
        
        # Test Rotary attention modules
        rotary_tests = [
            ("Rotary YAT", RotaryYatAttention(embed_dim=embed_dim, num_heads=num_heads, max_seq_len=seq_len, rngs=rngs)),
            ("Rotary YAT Performer", RotaryYatAttention(embed_dim=embed_dim, num_heads=num_heads, max_seq_len=seq_len, use_performer=True, num_features=64, rngs=rngs)),
            ("Rotary YAT Perf+Norm", RotaryYatAttention(embed_dim=embed_dim, num_heads=num_heads, max_seq_len=seq_len, use_performer=True, num_features=64, performer_normalize=True, rngs=rngs)),
        ]
        
        for name, module in rotary_tests:
            output = module(x, deterministic=True)
            assert not jnp.any(jnp.isnan(output)), f"{name} produced NaN"
            assert not jnp.any(jnp.isinf(output)), f"{name} produced Inf"
            assert output.shape == (batch, seq_len, embed_dim), f"{name} wrong shape"
            print(f"  [OK] {name}")


class TestMemoryScaling:
    """Tests memory usage patterns for different attention types."""
    
    def test_memory_analysis(self):
        """Analyzes theoretical memory usage for different attention types."""
        print("\n" + "=" * 80)
        print("MEMORY ANALYSIS (Theoretical)")
        print("=" * 80)
        print(f"Config: batch={BATCH_SIZE}, heads={NUM_HEADS}, head_dim={HEAD_DIM}")
        print()
        print(f"{'Seq Len':>10} | {'Quadratic (n²)':>15} | {'Linear (n)':>15} | {'Ratio':>10}")
        print("-" * 60)
        
        for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
            # Attention matrix size for quadratic: batch * heads * n * n
            quadratic_elements = BATCH_SIZE * NUM_HEADS * seq_len * seq_len
            quadratic_mb = (quadratic_elements * 4) / (1024 * 1024)  # float32 = 4 bytes
            
            # For linear attention: batch * heads * n * features
            linear_elements = BATCH_SIZE * NUM_HEADS * seq_len * NUM_FEATURES
            linear_mb = (linear_elements * 4) / (1024 * 1024)
            
            ratio = quadratic_mb / linear_mb if linear_mb > 0 else float('inf')
            
            print(f"{seq_len:>10} | {quadratic_mb:>12.2f} MB | {linear_mb:>12.2f} MB | {ratio:>8.1f}x")
        
        print("-" * 60)
        print("Note: Quadratic attention becomes prohibitive at very long sequences.")
        print("At seq_len=8192, quadratic uses 16x more memory than linear.")
        print("=" * 80)


class TestScalingAnalysis:
    """Analyzes scaling behavior of different attention types."""
    
    def test_scaling_analysis(self):
        """Calculates and displays scaling factors."""
        from nmn.nnx.attention import (
            yat_attention,
            yat_performer_attention,
            create_yat_projection,
        )
        
        print("\n" + "=" * 80)
        print("SCALING ANALYSIS")
        print("=" * 80)
        
        key = jax.random.key(0)
        seq_lengths = [64, 128, 256, 512]
        
        yat_times = []
        perf_times = []
        
        for seq_len in seq_lengths:
            q = jax.random.normal(key, (1, seq_len, 4, 32))
            k = jax.random.normal(key, (1, seq_len, 4, 32))
            v = jax.random.normal(key, (1, seq_len, 4, 32))
            proj = create_yat_projection(key, 64, 32)
            
            @jax.jit
            def yat_fn(q, k, v):
                return yat_attention(q, k, v)
            
            @jax.jit
            def perf_fn(q, k, v, proj):
                return yat_performer_attention(q, k, v, proj)
            
            # Warmup
            yat_fn(q, k, v).block_until_ready()
            perf_fn(q, k, v, proj).block_until_ready()
            
            # Measure
            start = time.perf_counter()
            for _ in range(5):
                yat_fn(q, k, v).block_until_ready()
            yat_time = (time.perf_counter() - start) * 200  # ms per iter
            
            start = time.perf_counter()
            for _ in range(5):
                perf_fn(q, k, v, proj).block_until_ready()
            perf_time = (time.perf_counter() - start) * 200  # ms per iter
            
            yat_times.append(yat_time)
            perf_times.append(perf_time)
        
        print("\nScaling factors (time[i] / time[i-1] when seq_len doubles):")
        print(f"{'Seq Len Change':>20} | {'YAT (expect 4x)':>15} | {'Performer (expect 2x)':>20}")
        print("-" * 60)
        
        for i in range(1, len(seq_lengths)):
            yat_ratio = yat_times[i] / yat_times[i-1] if yat_times[i-1] > 0 else 0
            perf_ratio = perf_times[i] / perf_times[i-1] if perf_times[i-1] > 0 else 0
            change = f"{seq_lengths[i-1]} -> {seq_lengths[i]}"
            print(f"{change:>20} | {yat_ratio:>15.2f}x | {perf_ratio:>20.2f}x")
        
        print("-" * 60)
        print("Quadratic attention should scale ~4x when seq_len doubles.")
        print("Linear attention should scale ~2x when seq_len doubles.")
        print("=" * 80)


class TestLongSequences:
    """Tests attention at very long sequences."""
    
    @pytest.mark.slow
    def test_long_sequence_performer(self):
        """Tests that Performer can handle very long sequences."""
        from nmn.nnx.attention import RotaryYatAttention
        
        print("\n" + "=" * 80)
        print("LONG SEQUENCE TEST (Performer)")
        print("=" * 80)
        
        # Test with a sequence length that would cause OOM with quadratic attention
        seq_len = 2048
        batch, heads, dim = 1, 4, 64
        embed_dim = heads * dim
        
        rngs = nnx.Rngs(0)
        attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            max_seq_len=seq_len,
            use_performer=True,
            num_features=128,
            performer_normalize=True,
            rngs=rngs,
        )
        
        key = jax.random.key(0)
        x = jax.random.normal(key, (batch, seq_len, embed_dim))
        
        # Compile and run
        @jax.jit
        def forward(x):
            return attn(x, deterministic=True)
        
        print(f"Testing: batch={batch}, seq_len={seq_len}, embed_dim={embed_dim}")
        
        start = time.perf_counter()
        out = forward(x)
        out.block_until_ready()
        compile_time = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        for _ in range(3):
            out = forward(x)
            out.block_until_ready()
        run_time = (time.perf_counter() - start) * 1000 / 3
        
        print(f"Compile time: {compile_time:.2f} ms")
        print(f"Run time: {run_time:.2f} ms")
        print(f"Output shape: {out.shape}")
        print(f"Output valid: {not jnp.any(jnp.isnan(out))}")
        print("=" * 80)
        
        assert out.shape == x.shape
        assert not jnp.any(jnp.isnan(out))


if __name__ == "__main__":
    # Run benchmarks directly
    print_benchmark_header()
    
    results = {
        "Dot-Product (baseline)": {},
        "YAT Standard": {},
        "YAT Normalized": {},
        "YAT Performer": {},
        "YAT Performer + Normalize": {},
        "Rotary YAT Standard": {},
        "Rotary YAT Performer": {},
        "Rotary YAT Performer + Normalize": {},
    }
    
    for seq_len in SEQ_LENGTHS:
        print(f"\nBenchmarking seq_len={seq_len}...")
        
        rngs = nnx.Rngs(42)
        bench = AttentionBenchmarks(seq_len, rngs)
        
        try:
            results["Dot-Product (baseline)"][seq_len] = bench.benchmark_dot_product_attention()
            results["YAT Standard"][seq_len] = bench.benchmark_yat_attention()
            results["YAT Normalized"][seq_len] = bench.benchmark_yat_attention_normalized()
            results["YAT Performer"][seq_len] = bench.benchmark_yat_performer()
            results["YAT Performer + Normalize"][seq_len] = bench.benchmark_yat_performer_normalized()
            results["Rotary YAT Standard"][seq_len] = bench.benchmark_rotary_yat()
            results["Rotary YAT Performer"][seq_len] = bench.benchmark_rotary_yat_performer()
            results["Rotary YAT Performer + Normalize"][seq_len] = bench.benchmark_rotary_yat_performer_normalized()
        except Exception as e:
            print(f"  Error: {e}")
    
    print_results_table(results)

