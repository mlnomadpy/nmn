import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Any

class YatDense(nn.Module):
    """A custom transformation applied over the last dimension of the input using squared Euclidean distance.
    
    Args:
        in_features: size of input features
        out_features: size of output features
        use_bias: whether to add a bias to the output (default: True)
        dtype: the dtype of the computation (default: None)
        epsilon: small constant added to avoid division by zero (default: 1e-6)
        return_weights: whether to return the weight matrix along with output (default: False)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        epsilon: float = 1e-6,
        return_weights: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.epsilon = epsilon
        self.return_weights = return_weights
        
        # Initialize weights using orthogonal initialization
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )
        init.orthogonal_(self.weight)
        
        if use_bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter('bias', None)
            
        # Initialize alpha parameter to 1.0
        self.alpha = nn.Parameter(torch.ones(1, dtype=dtype, device=device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the layer.
        
        Args:
            inputs: Input tensor of shape (..., in_features)
            
        Returns:
            Transformed tensor of shape (..., out_features) or
            tuple of (transformed tensor, weight matrix) if return_weights is True
        """
        # Compute dot product between input and weight
        y = F.linear(inputs, self.weight)  # equivalent to inputs @ weight.T
        
        # Compute squared Euclidean distances
        inputs_squared_sum = torch.sum(inputs**2, dim=-1, keepdim=True)
        weight_squared_sum = torch.sum(self.weight**2, dim=-1)
        distances = inputs_squared_sum + weight_squared_sum - 2 * y
        
        # Apply transformation
        y = y**2 / (distances + self.epsilon)
        
        # Apply scaling factor
        scale = (math.sqrt(self.out_features) / math.log(1 + self.out_features)) ** self.alpha
        y = y * scale
        
        # Add bias if used
        if self.bias is not None:
            # Reshape bias to match output dimensions
            bias_shape = (1,) * (y.dim() - 1) + (-1,)
            y = y + self.bias.view(*bias_shape)
            
        if self.return_weights:
            return y, self.weight
        return y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'