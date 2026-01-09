
import jax
import jax.numpy as jnp
from jax import random

def verify_feature_maps():
    print("="*60)
    print("COMPARING FEATURE MAP STRATEGIES")
    print("="*60)
    
    key = random.PRNGKey(42)
    d = 64
    m = 256  # Num features
    C = 2.00001
    
    # Create random projections
    # Orthogonal projection matrix
    def create_orth_proj(key, m, d):
        blocks = []
        for _ in range((m + d - 1) // d):
            key, subkey = random.split(key)
            mat = random.normal(subkey, (d, d))
            q, _ = jnp.linalg.qr(mat)
            blocks.append(q)
        W = jnp.concatenate(blocks, axis=0)[:m]
        return W * jnp.sqrt(d) # Scale so rows have norm sqrt(d)

    W = create_orth_proj(key, m, d)
    
    # Heuristic 1: My previous "stable" implementation
    # phi = |Wx| * exp(Wx - 0.5)
    def feature_map_heuristic(x, W):
        # x is unit vector
        # Wx has variance 1
        wx = jnp.einsum('d,md->m', x, W) / jnp.sqrt(d)
        
        linear = jnp.abs(wx)
        exp = jnp.exp(wx - 0.5)
        return linear * exp / jnp.sqrt(m)
    
    # Strategy 2: Squared linear
    # phi = (Wx)^2 * exp(...)
    def feature_map_squared(x, W):
        wx = jnp.einsum('d,md->m', x, W) / jnp.sqrt(d)
        
        linear = wx ** 2
        exp = jnp.exp(wx - 0.5)
        return linear * exp / jnp.sqrt(m)
        
    # Strategy 3: Modified "Tensor Product" approximation
    # Attempt to approximate (q·k)^2 * e^{2s qk}
    # using sum((w_i q)^2 * e^{w_i q})? 
    # E[(w^T q)^2] = |q|^2 = 1. E[(w^T k)^2] = 1.
    # Cov((w^T q)^2, (w^T k)^2) = 2(q^T k)^2.
    # So E[(w^T q)^2 (w^T k)^2] = 1 + 2(q^T k)^2.
    # This has an offset of 1! That's bad.
    
    # Strategy 4: "Linear * Exp" without abs?
    # E[ (wT q)(wT k) exp(wT q + wT k) ]
    # Hard to analyze closed form.
    
    # Strategy 5: Random Maclaurin for polynomial * FAVOR+
    # Explicitly compute 2nd degree polynomial?
    # poly = [x_i x_j]. dim d^2.
    # We want (xTy)^2 / (C - 2xTy).
    # Since we can't do d^2, let's stick to O(d) or O(m) methods.
    
    print("\nEvaluating Strategies:")
    print(f"{'dot':>6} | {'Exact':>10} | {'Heuristic':>10} | {'Squared':>10}")
    print("-" * 52)
    
    q = random.normal(key, (d,))
    q = q / jnp.linalg.norm(q)
    
    for dot_target in [-0.8, -0.5, 0.0, 0.5, 0.8, 0.95]:
        # Construct k
        k_orth = random.normal(random.split(key)[0], (d,))
        k_orth = k_orth - jnp.dot(k_orth, q) * q
        k_orth = k_orth / jnp.linalg.norm(k_orth)
        k = dot_target * q + jnp.sqrt(max(0, 1 - dot_target**2)) * k_orth
        k = k / jnp.linalg.norm(k)
        
        dot = float(jnp.dot(q, k))
        exact = (dot**2) / (C - 2*dot)
        
        # We need to sum over quadrature nodes formally, but let's just test
        # the core kernel approximation: x^2 * exp(2sx) vs feature product
        # Assume single quadrature node s=1, w=1 for simplicity kernel x^2 e^{2x}
        
        # But we want to approximate (q.k)^2 / (C - 2q.k) directly?
        # Let's stick to the full feature map logic.
        
        # Test 1: Heuristic (abs(wx) * exp) 
        # This was approximating roughly x * e^x.
        # Its square is x^2 e^{2x}.
        # YAT kernel is effectively that form.
        f1_q = feature_map_heuristic(q, W)
        f1_k = feature_map_heuristic(k, W)
        approx1 = float(jnp.sum(f1_q * f1_k))
        
        # Test 2: Squared ((wx)^2 * exp)
        f2_q = feature_map_squared(q, W)
        f2_k = feature_map_squared(k, W)
        approx2 = float(jnp.sum(f2_q * f2_k))

        print(f"{dot:6.2f} | {exact:10.4f} | {approx1:10.4f} | {approx2:10.4f}")

if __name__ == "__main__":
    verify_feature_maps()
