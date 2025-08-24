import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Pre-compute cos and sin values for all positions and dimensions
        self.register_buffer("cos_cached", torch.zeros((max_seq_len, d_k // 2)))
        self.register_buffer("sin_cached", torch.zeros((max_seq_len, d_k // 2)))

        # Initialize the cached values
        self._init_cache()

    def _init_cache(self):
        """Initialize the cached cos and sin values."""
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(self.max_seq_len, device=self.device).unsqueeze(1)

        # Create dimension indices: [0, 1, 2, ..., d_k//2-1]
        dims = torch.arange(self.d_k // 2, device=self.device).unsqueeze(0)

        # Compute frequency: theta^(-2i/d_k) for i in [0, d_k//2-1]
        freqs = self.theta ** (-2.0 * dims / self.d_k)

        # Compute angles: position * freq
        angles = positions * freqs

        # Cache cos and sin values
        self.cos_cached = torch.cos(angles)
        self.sin_cached = torch.sin(angles)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., sequence_length, d_k)
            token_positions (torch.Tensor): Token positions of shape (..., sequence_length)

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input
        """
        # Get the sequence length from the input
        seq_len = x.shape[-2]

        # Ensure we don't exceed max_seq_len
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        # Get cos and sin values for the current sequence length
        cos = self.cos_cached[:seq_len]  # (seq_len, d_k//2)
        sin = self.sin_cached[:seq_len]  # (seq_len, d_k//2)

        # Reshape input to separate even and odd dimensions
        # x: (..., seq_len, d_k) -> (..., seq_len, d_k//2, 2)
        x_reshaped = x.view(*x.shape[:-1], -1, 2)

        # Extract even and odd dimensions
        x_even = x_reshaped[..., 0]  # (..., seq_len, d_k//2)
        x_odd = x_reshaped[..., 1]  # (..., seq_len, d_k//2)

        # Apply RoPE rotation
        # For even dimensions: x_even * cos - x_odd * sin
        # For odd dimensions: x_even * sin + x_odd * cos

        # Reshape cos and sin to broadcast with x
        cos = cos.unsqueeze(0)  # (1, seq_len, d_k//2)
        sin = sin.unsqueeze(0)  # (1, seq_len, d_k//2)

        # Apply rotation
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        # Stack back together
        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)

        # Reshape back to original shape
        output = x_rotated.view(*x.shape)

        return output
