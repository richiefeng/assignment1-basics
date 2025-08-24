import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project the input features to the query, key, and value tensors
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # (batch_size, seq_len, d_model)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        # Now q, k, v have shape: (batch_size, num_heads, seq_len, d_head)

        # Create causal mask to prevent attention to future tokens
        # mask[i, j] = 1 if i >= j (can attend), 0 if i < j (cannot attend)
        # Use the same dtype as the input for consistency
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype)
        )
        # Expand mask for batch and head dimensions: (1, 1, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Compute attention scores
        # Use a scalar to avoid dtype conversion issues
        scale_factor = self.d_head**0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale_factor
        # scores shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply causal mask by setting masked positions to -inf
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        # attention_weights shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, v)
        # attended_values shape: (batch_size, num_heads, seq_len, d_head)

        # Reshape back to combine heads
        attended_values = (
            attended_values.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        # attended_values shape: (batch_size, seq_len, d_model)

        # Apply output projection
        output = self.o_proj(attended_values)
        # output shape: (batch_size, seq_len, d_model)

        return output
