import torch
import torch.nn as nn
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # First sublayer: Multi-head self-attention with RMSNorm
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)

        # Second sublayer: Position-wise feed-forward network with RMSNorm
        self.ln2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: Output tensor of the same shape
        """
        # First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))
        # Apply RMSNorm first
        x_norm1 = self.ln1(x)
        # Apply multi-head self-attention
        attn_output = self.attn(x_norm1)
        # Add residual connection
        x = x + attn_output

        # Second sublayer: y = x + PositionwiseFeedForward(RMSNorm(x))
        # Apply RMSNorm first
        x_norm2 = self.ln2(x)
        # Apply feed-forward network
        ffn_output = self.ffn(x_norm2)
        # Add residual connection
        x = x + ffn_output

        return x
