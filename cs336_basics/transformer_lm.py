import torch
import torch.nn as nn
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.linear import Linear


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
                for _ in range(num_layers)
            ]
        )

        # Final RMSNorm
        self.ln_final = RMSNorm(d_model)

        # Language model head (output projection to vocabulary)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer language model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing token indices

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size) containing logits
        """
        batch_size, seq_len = x.shape

        # Check sequence length doesn't exceed context length
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context length {self.context_length}"
            )

        # Get token embeddings: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.token_embeddings(x)

        # Pass through transformer blocks with manual RoPE application
        for layer in self.layers:
            # Apply RMSNorm first
            x_norm = layer.ln1(x)

            # Apply multi-head self-attention with RoPE manually
            # Project to Q, K, V
            q = layer.attn.q_proj(x_norm)
            k = layer.attn.k_proj(x_norm)
            v = layer.attn.v_proj(x_norm)

            # Reshape to separate heads
            d_head = self.d_model // self.num_heads
            q = q.view(batch_size, seq_len, self.num_heads, d_head)
            k = k.view(batch_size, seq_len, self.num_heads, d_head)
            v = v.view(batch_size, seq_len, self.num_heads, d_head)

            # Apply RoPE to Q and K (not V)
            from cs336_basics.rope import RotaryPositionalEmbedding

            rope = RotaryPositionalEmbedding(self.theta, d_head, self.max_seq_len, device=x.device)
            token_positions = torch.arange(seq_len, device=x.device)

            for head in range(self.num_heads):
                q[:, :, head, :] = rope.forward(q[:, :, head, :], token_positions)
                k[:, :, head, :] = rope.forward(k[:, :, head, :], token_positions)

            # Transpose to get (batch_size, num_heads, seq_len, d_head)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Create causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            # Apply scaled dot product attention
            from cs336_basics.utils import scaled_dot_product_attention

            q_batched = q.reshape(batch_size * self.num_heads, seq_len, d_head)
            k_batched = k.reshape(batch_size * self.num_heads, seq_len, d_head)
            v_batched = v.reshape(batch_size * self.num_heads, seq_len, d_head)
            mask_batched = causal_mask.expand(
                batch_size * self.num_heads, 1, seq_len, seq_len
            ).squeeze(1)

            attention_output = scaled_dot_product_attention(
                q_batched, k_batched, v_batched, mask_batched
            )

            # Reshape back and apply output projection
            attention_output = attention_output.view(
                batch_size, self.num_heads, seq_len, d_head
            )
            attention_output = (
                attention_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, self.d_model)
            )
            attn_output = layer.attn.o_proj(attention_output)

            # Add residual connection
            x = x + attn_output

            # Apply RMSNorm and feed-forward network
            x_norm2 = layer.ln2(x)
            ffn_output = layer.ffn(x_norm2)
            x = x + ffn_output

        # Apply final RMSNorm
        x = self.ln_final(x)

        # Apply language model head to get logits: (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits
