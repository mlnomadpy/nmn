"""Analyze the YAT kernel and test different feature map approaches.

The YAT kernel for normalized vectors is:
    K(q, k) = (q·k)² / (2(1 - q·k) + ε)

This is NOT the softmax kernel exp(q·k), so we need a different approach.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nmn.nnx.attention.yat_attention import create_yat_projection


def yat_kernel(dot_product: float, epsilon: float = 1e-5) -> float:
    """Exact YAT kernel for normalized vectors."""
    return (dot_product ** 2) / (2 * (1 - dot_product) + epsilon)


def analyze_yat_kernel():
    """Analyze the properties of the YAT kernel."""
    print("="*70)
    print("ANALYSIS: YAT Kernel Properties")
    print("="*70)
    
    # Range of dot products
    dots = np.linspace(-0.99, 0.99, 100)
    yat_values = [yat_kernel(d) for d in dots]
    
    print("\nYAT kernel values at key dot products:")
    for d in [-1.0, -0.5, 0.0, 0.5, 0.7, 0.9, 0.95, 0.99]:
        val = yat_kernel(d)
        print(f"  dot={d:+.2f}: K={val:.4f}")
    
    # Plot for visual analysis
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dots, yat_values, 'b-', linewidth=2)
    plt.xlabel('Dot product (q·k)')
    plt.ylabel('YAT Kernel K(q,k)')
    plt.title('YAT Kernel: (q·k)² / (2(1-q·k) + ε)')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Log scale
    plt.subplot(1, 2, 2)
    yat_positive = [max(v, 1e-10) for v in yat_values]
    plt.semilogy(dots, yat_positive, 'b-', linewidth=2)
    plt.xlabel('Dot product (q·k)')
    plt.ylabel('YAT Kernel (log scale)')
    plt.title('YAT Kernel (log scale)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('yat_kernel_analysis.png', dpi=150)
    print("\nSaved plot to yat_kernel_analysis.png")


def analyze_softmax_kernel():
    """Compare YAT kernel to softmax kernel."""
    print("\n" + "="*70)
    print("COMPARISON: YAT vs Softmax Kernel")
    print("="*70)
    
    dots = np.linspace(-0.99, 0.99, 100)
    
    head_dim = 64
    yat_values = [yat_kernel(d) for d in dots]
    softmax_values = [np.exp(d * np.sqrt(head_dim)) for d in dots]  # scaled softmax
    
    print("\nComparison at key dot products:")
    print(f"{'dot':>6} | {'YAT':>12} | {'Softmax':>12} | {'Ratio':>12}")
    print("-" * 50)
    for d in [-0.5, 0.0, 0.5, 0.7, 0.9]:
        yat_v = yat_kernel(d)
        soft_v = np.exp(d * np.sqrt(head_dim))
        ratio = yat_v / soft_v if soft_v > 0 else 0
        print(f"{d:+.2f}   | {yat_v:12.4f} | {soft_v:12.4f} | {ratio:12.4f}")


def test_feature_map_alternatives():
    """Test alternative feature map approaches for YAT."""
    print("\n" + "="*70)
    print("TESTING: Alternative Feature Map Approaches")
    print("="*70)
    
    key = jax.random.PRNGKey(42)
    head_dim = 64
    num_features = 256
    
    projection = create_yat_projection(key, num_features, head_dim)
    
    # Create a fixed query
    q = jax.random.normal(key, (head_dim,))
    q = q / jnp.linalg.norm(q)
    
    # Test various approaches
    def current_feature_map(x, proj):
        """Current softplus-based approach."""
        x_proj = jnp.einsum("d,md->m", x, proj)
        features = jax.nn.softplus(x_proj)
        features = features / jnp.sqrt(head_dim)
        features = features / jnp.sqrt(num_features)
        return features
    
    def exp_feature_map(x, proj):
        """Exponential feature map (standard Performer)."""
        x_proj = jnp.einsum("d,md->m", x, proj)
        x_norm_sq = jnp.sum(x ** 2) / 2.0
        features = jnp.exp(x_proj - x_norm_sq)
        features = features / jnp.sqrt(num_features)
        return features
    
    def relu_feature_map(x, proj):
        """ReLU feature map."""
        x_proj = jnp.einsum("d,md->m", x, proj)
        features = jax.nn.relu(x_proj)
        features = features / jnp.sqrt(num_features)
        return features
    
    def squared_feature_map(x, proj):
        """Squared feature map (polynomial kernel)."""
        x_proj = jnp.einsum("d,md->m", x, proj)
        features = x_proj ** 2
        features = features / jnp.sqrt(num_features)
        return features
    
    approaches = [
        ("Current (softplus)", current_feature_map),
        ("Exponential", exp_feature_map),
        ("ReLU", relu_feature_map),
        ("Squared", squared_feature_map),
    ]
    
    print("\nApproximation quality at different angles:")
    print(f"{'Approach':<25} | {'dot=-0.5':>10} | {'dot=0.0':>10} | {'dot=0.5':>10} | {'dot=0.9':>10}")
    print("-" * 75)
    
    # Exact YAT values for reference
    exact_vals = {-0.5: yat_kernel(-0.5), 0.0: yat_kernel(0.0), 0.5: yat_kernel(0.5), 0.9: yat_kernel(0.9)}
    print(f"{'Exact YAT':<25} | {exact_vals[-0.5]:>10.4f} | {exact_vals[0.0]:>10.4f} | {exact_vals[0.5]:>10.4f} | {exact_vals[0.9]:>10.4f}")
    
    for name, feature_fn in approaches:
        approx_vals = {}
        for target_dot in [-0.5, 0.0, 0.5, 0.9]:
            # Create k with desired dot product
            k_orth = jax.random.normal(jax.random.split(key)[0], (head_dim,))
            k_orth = k_orth - jnp.dot(k_orth, q) * q
            k_orth = k_orth / jnp.linalg.norm(k_orth)
            
            # k = target_dot * q + sqrt(1 - target_dot^2) * k_orth
            k = target_dot * q + jnp.sqrt(1 - target_dot**2) * k_orth
            k = k / jnp.linalg.norm(k)
            
            # Compute approximation
            q_feat = feature_fn(q, projection)
            k_feat = feature_fn(k, projection)
            approx = float(jnp.sum(q_feat * k_feat))
            approx_vals[target_dot] = approx
        
        print(f"{name:<25} | {approx_vals[-0.5]:>10.4f} | {approx_vals[0.0]:>10.4f} | {approx_vals[0.5]:>10.4f} | {approx_vals[0.9]:>10.4f}")


def propose_yat_specific_feature_map():
    """Propose a feature map specifically designed for the YAT kernel."""
    print("\n" + "="*70)
    print("PROPOSAL: YAT-Specific Feature Map")
    print("="*70)
    
    print("""
The YAT kernel K(q,k) = (q·k)² / (2(1 - q·k) + ε) is problematic for Performer because:

1. It's NOT positive definite (can be 0 at orthogonal vectors)
2. It blows up as dot product approaches 1
3. It's not a simple exponential form like softmax

PROPOSED SOLUTIONS:

A. Modified Performer (recommended):
   - Use the standard softmax approximation but scale differently
   - This means Performer mode doesn't compute exact YAT but a reasonable approximation
   - The key is to ensure attention still works well for gradients

B. Quadratic Attention (polynomial approach):
   - Approximate (q·k)² using squared random features
   - φ(x) = [x @ W]² where W is random projection
   - Then φ(q)·φ(k) ≈ (q·k)²
   - This handles the numerator but not the denominator

C. Hybrid Approach:
   - Use normal softmax Performer for attention weights
   - Apply YAT-specific scaling as a post-processing step
   - This preserves the O(n) complexity while getting YAT-like behavior

D. Fall back to standard attention:
   - For sequence lengths < 2048, use standard O(n²) YAT attention
   - Only use Performer for very long sequences where approximation is acceptable
""")


def main():
    analyze_yat_kernel()
    analyze_softmax_kernel()
    test_feature_map_alternatives()
    propose_yat_specific_feature_map()


if __name__ == "__main__":
    main()
