"""
Cross-Framework Consistency Tests for YAT Layers.

This test suite verifies that the YAT formula produces identical outputs
across all frameworks when given the same inputs and weights.

The YAT formula:
    y = (dot_product)^2 / (distance^2 + epsilon)
    where distance^2 = ||input||^2 + ||kernel||^2 - 2 * dot_product

This geometric computation should produce identical results regardless
of the deep learning framework used.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# Framework Availability
# ============================================================================

FRAMEWORKS = {}

try:
    import torch
    FRAMEWORKS['torch'] = True
except ImportError:
    FRAMEWORKS['torch'] = False

try:
    import tensorflow as tf
    FRAMEWORKS['tensorflow'] = True
except ImportError:
    FRAMEWORKS['tensorflow'] = False

try:
    import keras
    if keras.backend.backend() == 'tensorflow':
        FRAMEWORKS['keras'] = True
    else:
        FRAMEWORKS['keras'] = False
except ImportError:
    FRAMEWORKS['keras'] = False

try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    FRAMEWORKS['linen'] = True
except ImportError:
    FRAMEWORKS['linen'] = False

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    FRAMEWORKS['nnx'] = True
except ImportError:
    FRAMEWORKS['nnx'] = False


def get_available_frameworks() -> List[str]:
    """Get list of available frameworks."""
    return [name for name, available in FRAMEWORKS.items() if available]


AVAILABLE = get_available_frameworks()


# ============================================================================
# Result Container
# ============================================================================

@dataclass
class FrameworkOutput:
    """Container for framework output and metadata."""
    name: str
    output: np.ndarray
    
    
@dataclass  
class ComparisonResult:
    """Result of comparing two framework outputs."""
    framework_a: str
    framework_b: str
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    matching: bool  # Within tolerance


# ============================================================================
# Framework-Specific Implementations
# ============================================================================

def run_torch_dense(inputs: np.ndarray, weights: np.ndarray, 
                    bias: Optional[np.ndarray] = None,
                    alpha: Optional[float] = None,
                    epsilon: float = 1e-6) -> np.ndarray:
    """Run YAT dense layer in PyTorch."""
    import torch
    from nmn.torch.nmn import YatNMN
    
    in_features, out_features = weights.shape
    layer = YatNMN(
        in_features=in_features,
        out_features=out_features,
        bias=(bias is not None),
        alpha=(alpha is not None),
        epsilon=epsilon
    )
    
    with torch.no_grad():
        # PyTorch stores weight transposed: (out_features, in_features)
        layer.weight.data = torch.tensor(weights.T, dtype=torch.float32)
        if bias is not None:
            layer.bias.data = torch.tensor(bias, dtype=torch.float32)
        if alpha is not None:
            layer.alpha.data = torch.tensor([alpha], dtype=torch.float32)
        
        x = torch.tensor(inputs, dtype=torch.float32)
        output = layer(x)
    
    return output.numpy()


def run_linen_dense(inputs: np.ndarray, weights: np.ndarray,
                    bias: Optional[np.ndarray] = None,
                    alpha: Optional[float] = None,
                    epsilon: float = 1e-6) -> np.ndarray:
    """Run YAT dense layer in Flax Linen."""
    import jax
    import jax.numpy as jnp
    from nmn.linen import YatNMN
    
    in_features, out_features = weights.shape
    layer = YatNMN(
        features=out_features,
        use_bias=(bias is not None),
        use_alpha=(alpha is not None),
        epsilon=epsilon
    )
    
    key = jax.random.PRNGKey(0)
    x = jnp.array(inputs)
    _ = layer.init(key, x)  # Initialize
    
    # Linen kernel is (features, in_features)
    params = {'params': {'kernel': jnp.array(weights.T)}}
    if bias is not None:
        params['params']['bias'] = jnp.array(bias)
    if alpha is not None:
        params['params']['alpha'] = jnp.array([alpha])
    
    output = layer.apply(params, x)
    return np.array(output)


def run_nnx_dense(inputs: np.ndarray, weights: np.ndarray,
                  bias: Optional[np.ndarray] = None,
                  alpha: Optional[float] = None,
                  epsilon: float = 1e-6) -> np.ndarray:
    """Run YAT dense layer in Flax NNX."""
    import jax.numpy as jnp
    from flax import nnx
    from nmn.nnx.nmn import YatNMN
    
    in_features, out_features = weights.shape
    rngs = nnx.Rngs(0)
    layer = YatNMN(
        in_features=in_features,
        out_features=out_features,
        use_bias=(bias is not None),
        use_alpha=(alpha is not None),
        epsilon=epsilon,
        rngs=rngs
    )
    
    # NNX kernel is (in_features, out_features)
    layer.kernel.value = jnp.array(weights)
    if bias is not None:
        layer.bias.value = jnp.array(bias)
    if alpha is not None:
        layer.alpha.value = jnp.array([alpha])
    
    x = jnp.array(inputs)
    output = layer(x)
    return np.array(output)


# ============================================================================
# Reference NumPy Implementation
# ============================================================================

def run_numpy_yat(inputs: np.ndarray, weights: np.ndarray,
                  bias: Optional[np.ndarray] = None,
                  alpha: Optional[float] = None,
                  epsilon: float = 1e-6) -> np.ndarray:
    """Reference YAT implementation in pure NumPy."""
    # Dot product
    dot_prod = np.matmul(inputs, weights)
    
    # Squared norms
    inputs_sq_sum = np.sum(inputs**2, axis=-1, keepdims=True)
    weights_sq_sum = np.sum(weights**2, axis=0, keepdims=True)
    
    # Squared distance
    distance_sq = inputs_sq_sum + weights_sq_sum - 2 * dot_prod
    
    # Bias
    if bias is not None:
        y = y + bias

    # YAT transformation
    y = dot_prod**2 / (distance_sq + epsilon)
    
    
    # Alpha scaling
    if alpha is not None:
        out_features = weights.shape[1]
        scale = (np.sqrt(out_features) / np.log(1 + out_features)) ** alpha
        y = y * scale
    
    return y.astype(np.float32)


# ============================================================================
# Comparison Utilities
# ============================================================================

def compare_outputs(a: np.ndarray, b: np.ndarray, 
                    name_a: str, name_b: str,
                    rtol: float = 1e-4, atol: float = 1e-4) -> ComparisonResult:
    """Compare two framework outputs."""
    abs_diff = np.abs(a - b)
    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)
    
    # Relative error (avoid division by zero)
    denominator = np.maximum(np.abs(a), np.abs(b))
    denominator = np.where(denominator < 1e-10, 1.0, denominator)
    rel_diff = abs_diff / denominator
    max_rel = np.max(rel_diff)
    mean_rel = np.mean(rel_diff)
    
    matching = np.allclose(a, b, rtol=rtol, atol=atol)
    
    return ComparisonResult(
        framework_a=name_a,
        framework_b=name_b,
        max_abs_error=float(max_abs),
        mean_abs_error=float(mean_abs),
        max_rel_error=float(max_rel),
        mean_rel_error=float(mean_rel),
        matching=matching
    )


def print_comparison_table(results: List[ComparisonResult], title: str = ""):
    """Print a formatted comparison table."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    print(f"\n{'Comparison':<25} {'Max Abs':>12} {'Mean Abs':>12} {'Max Rel':>12} {'Match':>8}")
    print("-" * 75)
    
    for r in results:
        match_str = "PASS" if r.matching else "FAIL"
        print(f"{r.framework_a} vs {r.framework_b:<15} {r.max_abs_error:>12.2e} {r.mean_abs_error:>12.2e} {r.max_rel_error:>12.2e} {match_str:>8}")


# ============================================================================
# Main Test Class
# ============================================================================

@pytest.mark.skipif(len(AVAILABLE) < 2, reason="Need at least 2 frameworks for comparison")
class TestCrossFrameworkYATConsistency:
    """
    Test that YAT produces identical outputs across all frameworks.
    
    This validates the geometric consistency of the YAT formula:
        y = (dot_product)^2 / (distance^2 + epsilon)
    
    The formula is framework-agnostic and should produce identical
    results given identical inputs and weights.
    """
    
    @pytest.fixture
    def dense_test_data(self):
        """Generate deterministic test data for dense layer."""
        np.random.seed(42)
        return {
            'inputs': np.random.randn(8, 32).astype(np.float32),
            'weights': np.random.randn(32, 64).astype(np.float32),
            'bias': np.random.randn(64).astype(np.float32),
            'alpha': 1.0,
            'epsilon': 1e-6
        }
    
    def run_all_frameworks(self, inputs: np.ndarray, weights: np.ndarray,
                           bias: Optional[np.ndarray] = None,
                           alpha: Optional[float] = None,
                           epsilon: float = 1e-6) -> Dict[str, np.ndarray]:
        """Run YAT dense layer on all available frameworks."""
        outputs = {}
        
        # Reference NumPy implementation
        outputs['NumPy (ref)'] = run_numpy_yat(inputs, weights, bias, alpha, epsilon)
        
        if FRAMEWORKS.get('torch'):
            outputs['PyTorch'] = run_torch_dense(inputs, weights, bias, alpha, epsilon)
        
        if FRAMEWORKS.get('linen'):
            outputs['Linen'] = run_linen_dense(inputs, weights, bias, alpha, epsilon)
        
        if FRAMEWORKS.get('nnx'):
            outputs['NNX'] = run_nnx_dense(inputs, weights, bias, alpha, epsilon)
        
        return outputs
    
    def compare_all_pairs(self, outputs: Dict[str, np.ndarray],
                          rtol: float = 1e-4, atol: float = 1e-4) -> List[ComparisonResult]:
        """Compare all pairs of framework outputs."""
        results = []
        names = list(outputs.keys())
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                result = compare_outputs(
                    outputs[names[i]], outputs[names[j]],
                    names[i], names[j], rtol, atol
                )
                results.append(result)
        
        return results
    
    def test_basic_yat_consistency(self, dense_test_data):
        """Test basic YAT (no bias, no alpha) produces identical outputs."""
        outputs = self.run_all_frameworks(
            dense_test_data['inputs'],
            dense_test_data['weights'],
            epsilon=dense_test_data['epsilon']
        )
        
        results = self.compare_all_pairs(outputs)
        print_comparison_table(results, "Basic YAT (no bias, no alpha)")
        
        for r in results:
            assert r.matching, f"{r.framework_a} vs {r.framework_b}: max_abs={r.max_abs_error:.2e}"
    
    def test_yat_with_bias_consistency(self, dense_test_data):
        """Test YAT with bias produces identical outputs."""
        outputs = self.run_all_frameworks(
            dense_test_data['inputs'],
            dense_test_data['weights'],
            bias=dense_test_data['bias'],
            epsilon=dense_test_data['epsilon']
        )
        
        results = self.compare_all_pairs(outputs)
        print_comparison_table(results, "YAT with Bias")
        
        for r in results:
            assert r.matching, f"{r.framework_a} vs {r.framework_b}: max_abs={r.max_abs_error:.2e}"
    
    def test_yat_with_alpha_consistency(self, dense_test_data):
        """Test YAT with alpha scaling produces identical outputs."""
        outputs = self.run_all_frameworks(
            dense_test_data['inputs'],
            dense_test_data['weights'],
            alpha=dense_test_data['alpha'],
            epsilon=dense_test_data['epsilon']
        )
        
        results = self.compare_all_pairs(outputs)
        print_comparison_table(results, "YAT with Alpha Scaling")
        
        for r in results:
            assert r.matching, f"{r.framework_a} vs {r.framework_b}: max_abs={r.max_abs_error:.2e}"
    
    def test_full_yat_consistency(self, dense_test_data):
        """Test full YAT (bias + alpha) produces identical outputs."""
        outputs = self.run_all_frameworks(
            dense_test_data['inputs'],
            dense_test_data['weights'],
            bias=dense_test_data['bias'],
            alpha=dense_test_data['alpha'],
            epsilon=dense_test_data['epsilon']
        )
        
        results = self.compare_all_pairs(outputs)
        print_comparison_table(results, "Full YAT (Bias + Alpha)")
        
        for r in results:
            assert r.matching, f"{r.framework_a} vs {r.framework_b}: max_abs={r.max_abs_error:.2e}"
    
    def test_various_input_sizes(self):
        """Test consistency across various input sizes."""
        np.random.seed(42)
        
        sizes = [
            (1, 4, 8),    # Single sample
            (16, 8, 16),  # Small batch
            (32, 64, 128), # Medium
        ]
        
        all_results = []
        
        for batch, in_feat, out_feat in sizes:
            inputs = np.random.randn(batch, in_feat).astype(np.float32)
            weights = np.random.randn(in_feat, out_feat).astype(np.float32)
            
            outputs = self.run_all_frameworks(inputs, weights)
            results = self.compare_all_pairs(outputs)
            
            print(f"\nSize: batch={batch}, in={in_feat}, out={out_feat}")
            for r in results:
                print(f"  {r.framework_a} vs {r.framework_b}: max_abs={r.max_abs_error:.2e}")
                assert r.matching, f"Mismatch for size ({batch}, {in_feat}, {out_feat})"
            
            all_results.extend(results)
        
        # Summary
        max_error = max(r.max_abs_error for r in all_results)
        mean_error = np.mean([r.mean_abs_error for r in all_results])
        print(f"\n=== Summary ===")
        print(f"Max absolute error across all sizes: {max_error:.2e}")
        print(f"Mean absolute error across all sizes: {mean_error:.2e}")
    
    def test_numerical_edge_cases(self):
        """Test consistency with edge case inputs."""
        np.random.seed(42)
        
        in_feat, out_feat = 16, 32
        weights = np.random.randn(in_feat, out_feat).astype(np.float32)
        
        edge_cases = [
            ("ones", np.ones((4, in_feat), dtype=np.float32)),
            ("large", np.random.randn(4, in_feat).astype(np.float32) * 100),
            ("small", np.random.randn(4, in_feat).astype(np.float32) * 0.001),
        ]
        
        print("\n" + "="*60)
        print("  Numerical Edge Cases")
        print("="*60)
        
        for name, inputs in edge_cases:
            outputs = self.run_all_frameworks(inputs, weights)
            results = self.compare_all_pairs(outputs)
            
            max_err = max(r.max_abs_error for r in results)
            all_match = all(r.matching for r in results)
            status = "PASS" if all_match else "FAIL"
            
            print(f"\n{name}: max_error={max_err:.2e} {status}")
            
            for r in results:
                assert r.matching, f"Edge case '{name}': {r.framework_a} vs {r.framework_b}"


@pytest.mark.skipif(len(AVAILABLE) < 2, reason="Need at least 2 frameworks for comparison")
class TestYATGeometricProperties:
    """
    Test the geometric properties of YAT that should be consistent
    across all frameworks.
    """
    
    def test_positive_outputs(self):
        """YAT should always produce non-negative outputs (squared ratio)."""
        np.random.seed(42)
        
        inputs = np.random.randn(8, 16).astype(np.float32)
        weights = np.random.randn(16, 32).astype(np.float32)
        
        print("\n" + "="*60)
        print("  Positive Output Test (YAT = squared ratio >= 0)")
        print("="*60)
        
        if FRAMEWORKS.get('torch'):
            out = run_torch_dense(inputs, weights)
            assert np.all(out >= 0), "PyTorch produced negative values"
            print(f"PyTorch: min={out.min():.6f}, all positive [PASS]")
        
        if FRAMEWORKS.get('linen'):
            out = run_linen_dense(inputs, weights)
            assert np.all(out >= 0), "Linen produced negative values"
            print(f"Linen: min={out.min():.6f}, all positive [PASS]")
        
        if FRAMEWORKS.get('nnx'):
            out = run_nnx_dense(inputs, weights)
            assert np.all(out >= 0), "NNX produced negative values"
            print(f"NNX: min={out.min():.6f}, all positive [PASS]")
    
    def test_epsilon_effect(self):
        """Epsilon should only affect outputs where distance is near 0."""
        np.random.seed(42)
        
        inputs = np.random.randn(4, 8).astype(np.float32)
        weights = np.random.randn(8, 16).astype(np.float32)
        
        print("\n" + "="*60)
        print("  Epsilon Sensitivity Test")
        print("="*60)
        
        epsilons = [1e-3, 1e-5, 1e-7, 1e-9]
        
        for fw_name, fw_func in [
            ('PyTorch', run_torch_dense) if FRAMEWORKS.get('torch') else (None, None),
            ('Linen', run_linen_dense) if FRAMEWORKS.get('linen') else (None, None),
        ]:
            if fw_name is None:
                continue
                
            outputs = [fw_func(inputs, weights, epsilon=eps) for eps in epsilons]
            
            print(f"\n{fw_name}:")
            for i in range(len(epsilons) - 1):
                diff = np.max(np.abs(outputs[i] - outputs[i+1]))
                print(f"  eps={epsilons[i]:.0e} vs eps={epsilons[i+1]:.0e}: max_diff={diff:.2e}")
            
            # Larger epsilon differences should be small for non-degenerate inputs
            assert np.max(np.abs(outputs[0] - outputs[-1])) < 0.1, f"{fw_name} is too epsilon-sensitive"
    
    def test_matching_vector_behavior(self):
        """When input matches a weight vector, distance=0, output=dot^2/epsilon."""
        np.random.seed(42)
        
        weights = np.random.randn(8, 4).astype(np.float32)
        # Use first weight column as input (exact match)
        inputs = weights[:, 0:1].T  # Shape: (1, 8)
        
        epsilon = 1e-6
        
        # For matching vectors: dot = ||w||^2, dist^2 = 0
        # YAT = dot^2 / (dist^2 + eps) = (||w||^2)^2 / eps
        w_norm_sq = np.sum(weights[:, 0]**2)
        expected_first = (w_norm_sq**2) / epsilon
        
        print("\n" + "="*60)
        print("  Matching Vector Test (distance = 0)")
        print("="*60)
        print(f"Expected first output: {expected_first:.6f}")
        
        if FRAMEWORKS.get('torch'):
            out = run_torch_dense(inputs, weights, epsilon=epsilon)
            print(f"PyTorch first output: {out[0, 0]:.6f}")
            np.testing.assert_allclose(out[0, 0], expected_first, rtol=1e-3)
        
        if FRAMEWORKS.get('linen'):
            out = run_linen_dense(inputs, weights, epsilon=epsilon)
            print(f"Linen first output: {out[0, 0]:.6f}")
            np.testing.assert_allclose(out[0, 0], expected_first, rtol=1e-3)
        
        if FRAMEWORKS.get('nnx'):
            out = run_nnx_dense(inputs, weights, epsilon=epsilon)
            print(f"NNX first output: {out[0, 0]:.6f}")
            np.testing.assert_allclose(out[0, 0], expected_first, rtol=1e-3)


# ============================================================================
# Summary Report Generator
# ============================================================================

@pytest.mark.skipif(len(AVAILABLE) < 2, reason="Need at least 2 frameworks")
def test_generate_consistency_report():
    """Generate a comprehensive consistency report."""
    np.random.seed(42)
    
    print("\n")
    print("="*70)
    print("       YAT CROSS-FRAMEWORK CONSISTENCY REPORT")
    print("="*70)
    print(f"\nAvailable frameworks: {', '.join(AVAILABLE)}")
    
    # Test data
    inputs = np.random.randn(16, 32).astype(np.float32)
    weights = np.random.randn(32, 64).astype(np.float32)
    epsilon = 1e-6
    
    # Collect outputs
    outputs = {}
    outputs['NumPy (ref)'] = run_numpy_yat(inputs, weights, epsilon=epsilon)
    
    if FRAMEWORKS.get('torch'):
        outputs['PyTorch'] = run_torch_dense(inputs, weights, epsilon=epsilon)
    if FRAMEWORKS.get('linen'):
        outputs['Linen'] = run_linen_dense(inputs, weights, epsilon=epsilon)
    if FRAMEWORKS.get('nnx'):
        outputs['NNX'] = run_nnx_dense(inputs, weights, epsilon=epsilon)
    
    # Compare all pairs
    names = list(outputs.keys())
    all_errors = []
    
    print("\n" + "-"*70)
    print("Framework Pair Comparison (Basic YAT)")
    print("-"*70)
    print(f"{'Comparison':<30} {'Max Abs Error':>15} {'Mean Abs Error':>15}")
    print("-"*70)
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result = compare_outputs(outputs[names[i]], outputs[names[j]], names[i], names[j])
            all_errors.append(result.max_abs_error)
            print(f"{names[i]} vs {names[j]:<20} {result.max_abs_error:>15.2e} {result.mean_abs_error:>15.2e}")
    
    # Summary statistics
    print("\n" + "-"*70)
    print("SUMMARY STATISTICS")
    print("-"*70)
    print(f"Maximum error across all pairs: {max(all_errors):.2e}")
    print(f"Mean error across all pairs:    {np.mean(all_errors):.2e}")
    print(f"All pairs within tolerance:     {'YES [PASS]' if max(all_errors) < 1e-3 else 'NO [FAIL]'}")
    
    # Output statistics
    print("\n" + "-"*70)
    print("OUTPUT STATISTICS PER FRAMEWORK")
    print("-"*70)
    print(f"{'Framework':<15} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-"*70)
    
    for name, out in outputs.items():
        print(f"{name:<15} {out.min():>12.4f} {out.max():>12.4f} {out.mean():>12.4f} {out.std():>12.4f}")
    
    print("\n" + "="*70)
    print("                    END OF REPORT")
    print("="*70)
    
    # Final assertion
    assert max(all_errors) < 1e-3, f"Cross-framework error too high: {max(all_errors):.2e}"
