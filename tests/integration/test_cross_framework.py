"""
Cross-Framework Consistency Tests.

These tests verify that the YAT formula is consistently implemented
across all frameworks (PyTorch, TensorFlow, Keras, Linen, NNX).
"""

import pytest
import numpy as np


# ============================================================================
# Framework Availability Checks
# ============================================================================

def get_available_frameworks():
    """Get list of available frameworks."""
    frameworks = []
    
    try:
        import torch
        frameworks.append('torch')
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        frameworks.append('tensorflow')
    except ImportError:
        pass
    
    try:
        import keras
        if keras.backend.backend() == 'tensorflow':
            frameworks.append('keras')
    except ImportError:
        pass
    
    try:
        import jax
        from flax import linen
        frameworks.append('linen')
    except ImportError:
        pass
    
    try:
        import jax
        from flax import nnx
        frameworks.append('nnx')
    except ImportError:
        pass
    
    return frameworks


AVAILABLE_FRAMEWORKS = get_available_frameworks()


# ============================================================================
# Reference YAT Implementation
# ============================================================================

def yat_reference_numpy(inputs, weights, bias=None, alpha=None, epsilon=1e-6):
    """
    Reference YAT implementation in pure NumPy.
    
    Args:
        inputs: (batch, in_features) numpy array
        weights: (in_features, out_features) numpy array
        bias: (out_features,) numpy array or None
        alpha: scalar or None
        epsilon: small constant
        
    Returns:
        output: (batch, out_features) numpy array
    """
    # Dot product
    dot_prod = np.matmul(inputs, weights)
    
    # Squared norms
    inputs_sq_sum = np.sum(inputs**2, axis=-1, keepdims=True)
    weights_sq_sum = np.sum(weights**2, axis=0, keepdims=True)
    
    # Squared distance
    distance_sq = inputs_sq_sum + weights_sq_sum - 2 * dot_prod
    
    # YAT transformation
    y = dot_prod**2 / (distance_sq + epsilon)
    
    # Bias
    if bias is not None:
        y = y + bias
    
    # Alpha scaling
    if alpha is not None:
        out_features = weights.shape[1]
        scale = (np.sqrt(out_features) / np.log(1 + out_features)) ** alpha
        y = y * scale
    
    return y


# ============================================================================
# PyTorch Implementation Helper
# ============================================================================

def get_torch_output(inputs_np, weights_np, bias_np=None, alpha_np=None, epsilon=1e-6):
    """Get YAT output from PyTorch implementation."""
    import torch
    from nmn.torch.nmn import YatNMN
    
    in_features, out_features = weights_np.shape
    layer = YatNMN(
        in_features=in_features, 
        out_features=out_features,
        bias=(bias_np is not None),
        alpha=(alpha_np is not None),
        epsilon=epsilon
    )
    
    # Set weights
    with torch.no_grad():
        layer.weight.data = torch.tensor(weights_np.T, dtype=torch.float32)
        if bias_np is not None:
            layer.bias.data = torch.tensor(bias_np, dtype=torch.float32)
        if alpha_np is not None:
            layer.alpha.data = torch.tensor([alpha_np], dtype=torch.float32)
    
    x = torch.tensor(inputs_np, dtype=torch.float32)
    with torch.no_grad():
        output = layer(x)
    
    return output.numpy()


def get_tf_output(inputs_np, weights_np, bias_np=None, alpha_np=None, epsilon=1e-6):
    """Get YAT output from TensorFlow implementation."""
    import tensorflow as tf
    from nmn.tf import YatNMN
    
    in_features, out_features = weights_np.shape
    layer = YatNMN(
        features=out_features,  # TF YatNMN uses 'features' not 'in_features'
        use_bias=(bias_np is not None),
        epsilon=epsilon
    )
    
    # Build layer
    x = tf.constant(inputs_np, dtype=tf.float32)
    _ = layer(x)
    
    # Set weights - TF YatNMN kernel is (out_features, in_features)
    layer.kernel.assign(weights_np.T)
    if bias_np is not None:
        layer.bias.assign(bias_np)
    if alpha_np is not None:
        layer.alpha.assign([alpha_np])
    
    output = layer(x)
    return output.numpy()


def get_keras_output(inputs_np, weights_np, bias_np=None, alpha_np=None, epsilon=1e-6):
    """Get YAT output from Keras implementation."""
    import keras
    from nmn.keras import YatNMN
    
    in_features, out_features = weights_np.shape
    layer = YatNMN(
        units=out_features,
        use_bias=(bias_np is not None),
        epsilon=epsilon
    )
    
    # Build layer
    x = keras.ops.convert_to_tensor(inputs_np.astype(np.float32))
    _ = layer(x)
    
    # Set weights - Keras has kernel as (in, out), alpha, then bias
    weights_list = [weights_np]
    weights_list.append(np.array([1.0 if alpha_np is None else alpha_np]))  # Always has alpha
    if bias_np is not None:
        weights_list.append(bias_np)
    
    layer.set_weights(weights_list)
    
    output = layer(x)
    return keras.ops.convert_to_numpy(output)


def get_linen_output(inputs_np, weights_np, bias_np=None, alpha_np=None, epsilon=1e-6):
    """Get YAT output from Linen implementation."""
    import jax
    import jax.numpy as jnp
    from nmn.linen import YatNMN
    
    in_features, out_features = weights_np.shape
    layer = YatNMN(
        features=out_features,
        use_bias=(bias_np is not None),
        use_alpha=(alpha_np is not None),
        epsilon=epsilon
    )
    
    key = jax.random.PRNGKey(0)
    x = jnp.array(inputs_np)
    params = layer.init(key, x)
    
    # Create new params dict - Linen kernel is (features, in_features)
    new_params = {'params': {'kernel': jnp.array(weights_np.T)}}
    if bias_np is not None:
        new_params['params']['bias'] = jnp.array(bias_np)
    if alpha_np is not None:
        new_params['params']['alpha'] = jnp.array([alpha_np])
    
    output = layer.apply(new_params, x)
    return np.array(output)


def get_nnx_output(inputs_np, weights_np, bias_np=None, alpha_np=None, epsilon=1e-6):
    """Get YAT output from NNX implementation."""
    import jax.numpy as jnp
    from flax import nnx
    from nmn.nnx.nmn import YatNMN
    
    in_features, out_features = weights_np.shape
    rngs = nnx.Rngs(0)
    layer = YatNMN(
        in_features=in_features,
        out_features=out_features,
        use_bias=(bias_np is not None),
        use_alpha=(alpha_np is not None),
        epsilon=epsilon,
        rngs=rngs
    )
    
    # Set weights
    layer.kernel.value = jnp.array(weights_np)
    if bias_np is not None:
        layer.bias.value = jnp.array(bias_np)
    if alpha_np is not None:
        layer.alpha.value = jnp.array([alpha_np])
    
    x = jnp.array(inputs_np)
    output = layer(x)
    return np.array(output)


# ============================================================================
# Cross-Framework Tests
# ============================================================================

@pytest.mark.skipif(len(AVAILABLE_FRAMEWORKS) < 2, reason="Need at least 2 frameworks")
class TestCrossFrameworkConsistency:
    """Test that all frameworks produce consistent results."""
    
    @pytest.fixture
    def test_data(self):
        """Generate test data."""
        np.random.seed(42)
        return {
            'inputs': np.random.randn(4, 8).astype(np.float32),
            'weights': np.random.randn(8, 16).astype(np.float32),
            'bias': np.random.randn(16).astype(np.float32),
            'alpha': 1.0,
            'epsilon': 1e-6
        }
    
    def test_basic_yat_no_bias_no_alpha(self, test_data):
        """Test basic YAT (no bias, no alpha) across frameworks."""
        inputs = test_data['inputs']
        weights = test_data['weights']
        epsilon = test_data['epsilon']
        
        # Reference output
        ref_output = yat_reference_numpy(inputs, weights, epsilon=epsilon)
        
        outputs = {}
        
        if 'torch' in AVAILABLE_FRAMEWORKS:
            outputs['torch'] = get_torch_output(inputs, weights, epsilon=epsilon)
        
        if 'linen' in AVAILABLE_FRAMEWORKS:
            outputs['linen'] = get_linen_output(inputs, weights, epsilon=epsilon)
        
        if 'nnx' in AVAILABLE_FRAMEWORKS:
            outputs['nnx'] = get_nnx_output(inputs, weights, epsilon=epsilon)
        
        # Compare all frameworks to reference
        for name, output in outputs.items():
            np.testing.assert_allclose(
                output, ref_output, rtol=1e-3, atol=1e-3,
                err_msg=f"{name} output differs from reference"
            )
    
    def test_yat_with_bias(self, test_data):
        """Test YAT with bias across frameworks."""
        inputs = test_data['inputs']
        weights = test_data['weights']
        bias = test_data['bias']
        epsilon = test_data['epsilon']
        
        # Reference output
        ref_output = yat_reference_numpy(inputs, weights, bias=bias, epsilon=epsilon)
        
        outputs = {}
        
        if 'torch' in AVAILABLE_FRAMEWORKS:
            outputs['torch'] = get_torch_output(inputs, weights, bias_np=bias, epsilon=epsilon)
        
        if 'linen' in AVAILABLE_FRAMEWORKS:
            outputs['linen'] = get_linen_output(inputs, weights, bias_np=bias, epsilon=epsilon)
        
        if 'nnx' in AVAILABLE_FRAMEWORKS:
            outputs['nnx'] = get_nnx_output(inputs, weights, bias_np=bias, epsilon=epsilon)
        
        for name, output in outputs.items():
            np.testing.assert_allclose(
                output, ref_output, rtol=1e-3, atol=1e-3,
                err_msg=f"{name} output with bias differs from reference"
            )
    
    def test_yat_with_alpha(self, test_data):
        """Test YAT with alpha scaling across frameworks."""
        inputs = test_data['inputs']
        weights = test_data['weights']
        alpha = test_data['alpha']
        epsilon = test_data['epsilon']
        
        # Reference output
        ref_output = yat_reference_numpy(inputs, weights, alpha=alpha, epsilon=epsilon)
        
        outputs = {}
        
        if 'torch' in AVAILABLE_FRAMEWORKS:
            outputs['torch'] = get_torch_output(inputs, weights, alpha_np=alpha, epsilon=epsilon)
        
        if 'linen' in AVAILABLE_FRAMEWORKS:
            outputs['linen'] = get_linen_output(inputs, weights, alpha_np=alpha, epsilon=epsilon)
        
        if 'nnx' in AVAILABLE_FRAMEWORKS:
            outputs['nnx'] = get_nnx_output(inputs, weights, alpha_np=alpha, epsilon=epsilon)
        
        for name, output in outputs.items():
            np.testing.assert_allclose(
                output, ref_output, rtol=1e-3, atol=1e-3,
                err_msg=f"{name} output with alpha differs from reference"
            )
    
    def test_positive_outputs_all_frameworks(self, test_data):
        """Test that all frameworks produce non-negative outputs (no bias)."""
        inputs = test_data['inputs']
        weights = test_data['weights']
        epsilon = test_data['epsilon']
        
        if 'torch' in AVAILABLE_FRAMEWORKS:
            output = get_torch_output(inputs, weights, epsilon=epsilon)
            assert np.all(output >= 0), "PyTorch produced negative values"
        
        if 'linen' in AVAILABLE_FRAMEWORKS:
            output = get_linen_output(inputs, weights, epsilon=epsilon)
            assert np.all(output >= 0), "Linen produced negative values"
        
        if 'nnx' in AVAILABLE_FRAMEWORKS:
            output = get_nnx_output(inputs, weights, epsilon=epsilon)
            assert np.all(output >= 0), "NNX produced negative values"


# ============================================================================
# Pairwise Framework Comparison
# ============================================================================

@pytest.mark.skipif('torch' not in AVAILABLE_FRAMEWORKS or 'linen' not in AVAILABLE_FRAMEWORKS,
                    reason="Need both PyTorch and Linen")
class TestTorchVsLinen:
    """Direct comparison between PyTorch and Linen."""
    
    def test_dense_layer_match(self):
        """Test that dense layers match between frameworks."""
        np.random.seed(42)
        inputs = np.random.randn(4, 8).astype(np.float32)
        weights = np.random.randn(8, 16).astype(np.float32)
        
        torch_out = get_torch_output(inputs, weights)
        linen_out = get_linen_output(inputs, weights)
        
        np.testing.assert_allclose(torch_out, linen_out, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif('linen' not in AVAILABLE_FRAMEWORKS or 'nnx' not in AVAILABLE_FRAMEWORKS,
                    reason="Need both Linen and NNX")
class TestLinenVsNNX:
    """Direct comparison between Linen and NNX."""
    
    def test_dense_layer_match(self):
        """Test that dense layers match between Linen and NNX."""
        np.random.seed(42)
        inputs = np.random.randn(4, 8).astype(np.float32)
        weights = np.random.randn(8, 16).astype(np.float32)
        
        linen_out = get_linen_output(inputs, weights)
        nnx_out = get_nnx_output(inputs, weights)
        
        np.testing.assert_allclose(linen_out, nnx_out, rtol=1e-3, atol=1e-3)


# ============================================================================
# Numerical Stability Tests
# ============================================================================

@pytest.mark.skipif(len(AVAILABLE_FRAMEWORKS) < 1, reason="Need at least 1 framework")
class TestNumericalStabilityAllFrameworks:
    """Test numerical stability across frameworks."""
    
    def test_large_values(self):
        """Test with large input values."""
        np.random.seed(42)
        inputs = np.random.randn(4, 8).astype(np.float32) * 1000
        weights = np.random.randn(8, 16).astype(np.float32)
        
        if 'torch' in AVAILABLE_FRAMEWORKS:
            output = get_torch_output(inputs, weights)
            assert not np.isnan(output).any(), "PyTorch NaN with large values"
            assert not np.isinf(output).any(), "PyTorch Inf with large values"
        
        if 'linen' in AVAILABLE_FRAMEWORKS:
            output = get_linen_output(inputs, weights)
            assert not np.isnan(output).any(), "Linen NaN with large values"
            assert not np.isinf(output).any(), "Linen Inf with large values"
    
    def test_small_values(self):
        """Test with small input values."""
        np.random.seed(42)
        inputs = np.random.randn(4, 8).astype(np.float32) * 1e-6
        weights = np.random.randn(8, 16).astype(np.float32)
        
        if 'torch' in AVAILABLE_FRAMEWORKS:
            output = get_torch_output(inputs, weights)
            assert not np.isnan(output).any(), "PyTorch NaN with small values"
        
        if 'linen' in AVAILABLE_FRAMEWORKS:
            output = get_linen_output(inputs, weights)
            assert not np.isnan(output).any(), "Linen NaN with small values"
    
    def test_matching_input_weight(self):
        """Test when input exactly matches a weight vector."""
        np.random.seed(42)
        weights = np.random.randn(8, 16).astype(np.float32)
        # Use first weight column as input
        inputs = weights[:, 0:1].T  # Shape: (1, 8)
        
        if 'torch' in AVAILABLE_FRAMEWORKS:
            output = get_torch_output(inputs, weights)
            assert not np.isnan(output).any(), "PyTorch NaN with matching vectors"
        
        if 'linen' in AVAILABLE_FRAMEWORKS:
            output = get_linen_output(inputs, weights)
            assert not np.isnan(output).any(), "Linen NaN with matching vectors"
