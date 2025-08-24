import torch
import torch.nn as nn
import torch.nn.init as init


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct a SwiGLU feed-forward network.
        
        Args:
            d_model (int): Hidden dimension of the model
            d_ff (int): Dimension of the feed-forward inner layer
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        
        # Use the provided d_ff value directly
        self.d_ff = d_ff
        
        # Create the three linear transformations
        # W1: d_model → d_ff
        self.w1 = nn.Parameter(
            torch.empty(self.d_ff, d_model, device=device, dtype=dtype)
        )
        
        # W2: d_ff → d_model  
        self.w2 = nn.Parameter(
            torch.empty(d_model, self.d_ff, device=device, dtype=dtype)
        )
        
        # W3: d_model → d_ff (for GLU)
        self.w3 = nn.Parameter(
            torch.empty(self.d_ff, d_model, device=device, dtype=dtype)
        )
        
        # Initialize weights using trunc_normal_
        init.trunc_normal_(self.w1, std=0.02)
        init.trunc_normal_(self.w2, std=0.02)
        init.trunc_normal_(self.w3, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SwiGLU feed-forward network to the input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., d_model)
        """
        # Apply W1 transformation: x @ W1^T
        # x: (..., d_model), W1: (d_ff, d_model) → (..., d_ff)
        h1 = torch.einsum('...i,ji->...j', x, self.w1)
        
        # Apply SiLU activation to h1
        h1_silu = torch.nn.functional.silu(h1)
        
        # Apply W3 transformation: x @ W3^T  
        # x: (..., d_model), W3: (d_ff, d_model) → (..., d_ff)
        h3 = torch.einsum('...i,ji->...j', x, self.w3)
        
        # Element-wise multiplication: SiLU(W1x) ⊙ W3x
        # This is the gating mechanism
        gated = torch.einsum('...i,...i->...i', h1_silu, h3)
        
        # Apply W2 transformation: gated @ W2^T
        # gated: (..., d_ff), W2: (d_model, d_ff) → (..., d_model)
        output = torch.einsum('...i,ji->...j', gated, self.w2)
        
        return output