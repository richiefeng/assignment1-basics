import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        
        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        
        # Create weight parameter with shape (out_features, in_features)
        # This is W (not W^T) for memory ordering reasons
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights using trunc_normal_
        init.trunc_normal_(self.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x (torch.Tensor): Input tensor with shape (..., in_features)
            
        Returns:
            torch.Tensor: Output tensor with shape (..., out_features)
        """
        # Compute x @ W^T using einsum: '...i,ji->...j'
        # This means: for each batch dimension (...), multiply input (i) with weight transpose (ji) to get output (j)
        return torch.einsum('...i,ji->...j', x, self.weight)