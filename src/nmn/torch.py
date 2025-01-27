import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class YatDense(nn.Module):
    """
    A PyTorch implementation of the Yat neuron with squared Euclidean distance transformation.

    Attributes:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        use_bias (bool): Whether to add a bias to the output
        dtype (torch.dtype): Data type for computation
        epsilon (float): Small constant to avoid division by zero
        kernel_init (callable): Initializer for the weight matrix
        bias_init (callable): Initializer for the bias
        alpha_init (callable): Initializer for the scaling parameter
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        epsilon: float = 1e-4, # 1/epsilon is the maximum score per neuron, setting it low increase the precision but the scores explode 
        kernel_init: callable = None,
        bias_init: callable = None,
        alpha_init: callable = None
    ):
        super().__init__()

        # Store attributes
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.epsilon = epsilon
        # Weight initialization
        if kernel_init is None:
            kernel_init = nn.init.xavier_normal_

        # Create weight parameter
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features),
            dtype=dtype
        ))

        # Alpha scaling parameter
        self.alpha = nn.Parameter(torch.ones(
            (1,),
            dtype=dtype
        ))

        # Bias parameter
        if bias:
            self.bias = nn.Parameter(torch.empty(
                (out_features,),
                dtype=dtype
            ))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters(kernel_init, bias_init, alpha_init)

    def reset_parameters(
        self,
        kernel_init: callable = None,
        bias_init: callable = None,
        alpha_init: callable = None
    ):
        """
        Initialize network parameters with specified or default initializers.
        """
        # Kernel (weight) initialization
        if kernel_init is None:
            kernel_init = nn.init.orthogonal_
        kernel_init(self.weight)

        # Bias initialization
        if self.bias is not None:
            if bias_init is None:
                # Default: uniform initialization
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                bias_init(self.bias)

        # Alpha initialization (default to 1.0)
        if alpha_init is None:
            self.alpha.data.fill_(1.0)
        else:
            alpha_init(self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with squared Euclidean distance transformation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed output
        """
        # Ensure input and weight are in the same dtype
        x = x.to(self.dtype)

        # Compute dot product
        y = torch.matmul(x, self.weight.t())

        # Compute squared distances
        inputs_squared_sum = torch.sum(x**2, dim=-1, keepdim=True)
        kernel_squared_sum = torch.sum(self.weight**2, dim=-1)
        distances = inputs_squared_sum + kernel_squared_sum - 2 * y

        # Apply squared Euclidean distance transformation
        y = y ** 2 / (distances + self.epsilon)

        # Add bias if used
        if self.bias is not None:
            y += self.bias
            
        # Dynamic scaling
        scale = (math.sqrt(self.out_features) / math.log(1 + self.out_features)) ** self.alpha
        y = y * scale


        return y

    def extra_repr(self) -> str:
        """
        Extra representation of the module for print formatting.
        """
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.use_bias}")
