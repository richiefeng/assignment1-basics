import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability (default: 1e-5)
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        # Create weight parameter for affine transformation
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and return RMSNorm output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: Normalized tensor of the same shape
        """
        # Calculate RMS (Root Mean Square) along the last dimension (d_model)
        # RMS = sqrt(mean(x^2))
        # rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))

        # # Normalize: x / (rms + eps)
        # normalized = x / (rms + self.eps)
        # Calculate variance and normalize with eps inside sqrt for stability
        variance = torch.mean(x * x, dim=-1, keepdim=True)
        normalized = x / torch.sqrt(variance + self.eps)

        # Apply affine transformation: weight * normalized
        # weight has shape (d_model,) and normalized has shape (..., d_model)
        # Use einsum for explicit dimension handling: 'i,...i->...i'
        output = torch.einsum("i,...i->...i", self.weight, normalized)

        return output
