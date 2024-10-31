import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Any, Tuple

class YatDense(nn.Module):
    """A custom transformation applied over the last dimension of the input using squared Euclidean distance.

    Args:
        features (int): the number of output features
        use_bias (bool): whether to add a bias to the output (default: True)
        dtype (Optional[torch.dtype]): the dtype of the computation (default: None)
        epsilon (float): small constant added to avoid division by zero (default: 1e-6)
        return_weights (bool): whether to return the weight matrix along with output (default: False)
    """
    def __init__(
        self,
        features: int,
        use_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        epsilon: float = 1e-6,
        return_weights: bool = False
    ):
        super().__init__()
        self.features = features
        self.use_bias = use_bias
        self.dtype = dtype
        self.epsilon = epsilon
        self.return_weights = return_weights

        # Initialize weights using orthogonal initialization
        self.weight = nn.Parameter(torch.empty(features, 0))  # Will be properly sized in forward
        nn.init.orthogonal_(self.weight)

        # Initialize alpha parameter to 1.0
        self.alpha = nn.Parameter(torch.ones(1))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Applies a transformation to the inputs along the last dimension using squared Euclidean distance.

        Args:
            inputs (torch.Tensor): The input tensor to be transformed

        Returns:
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: The transformed input, 
            optionally with the weight matrix if return_weights is True
        """
        # Ensure weight matrix is properly sized for the input
        if self.weight.size(1) != inputs.size(-1):
            new_weight = torch.empty(self.features, inputs.size(-1), 
                                   dtype=self.dtype if self.dtype else inputs.dtype,
                                   device=inputs.device)
            nn.init.orthogonal_(new_weight)
            self.weight.data = new_weight

        # Cast inputs to the correct dtype if specified
        if self.dtype is not None:
            inputs = inputs.to(self.dtype)

        # Compute dot product between input and transposed kernel
        y = torch.matmul(inputs, self.weight.t())

        # Compute squared Euclidean distances
        inputs_squared_sum = torch.sum(inputs**2, dim=-1, keepdim=True)
        kernel_squared_sum = torch.sum(self.weight**2, dim=-1)
        distances = inputs_squared_sum + kernel_squared_sum - 2 * y

        # Apply the transformation
        y = y**2 / (distances + self.epsilon)
        
        # Apply scaling factor
        scale = (math.sqrt(self.features) / math.log(1 + self.features)) ** self.alpha
        y = y * scale

        # Add bias if used
        if self.bias is not None:
            y = y + self.bias.view(*([1] * (y.dim() - 1)), -1)

        if self.return_weights:
            return y, self.weight
        return y