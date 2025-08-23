from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
import time
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
from collections import Counter, defaultdict
import regex as re
import multiprocessing as mp

from .common import gpt2_bytes_to_unicode


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    from cs336_basics.linear import Linear

    # Create Linear module
    linear = Linear(d_in, d_out)

    # Load the weights into the module
    linear.load_state_dict({"weight": weights})

    # Apply the linear transformation
    return linear.forward(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    from cs336_basics.embedding import Embedding

    # Create Embedding module
    embedding = Embedding(vocab_size, d_model)

    # Load the weights into the module
    embedding.load_state_dict({"embedding_matrix": weights})

    # Get embeddings for the token IDs
    return embedding.forward(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    from cs336_basics.positionwise_feedforward import PositionwiseFeedForward

    # Create PositionwiseFeedForward module
    swiglu = PositionwiseFeedForward(d_model, d_ff)

    # Load the weights into the module
    swiglu.load_state_dict(
        {
            "w1": w1_weight,  # (d_ff, d_model)
            "w2": w2_weight,  # (d_model, d_ff)
            "w3": w3_weight,  # (d_ff, d_model)
        }
    )

    # Apply SwiGLU to input features
    return swiglu.forward(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    from cs336_basics.utils import scaled_dot_product_attention

    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from cs336_basics.multihead_self_attention import MultiHeadSelfAttention

    # Create the multi-head attention module
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    # Load the provided weights
    state_dict = {
        "q_proj.weight": q_proj_weight,
        "k_proj.weight": k_proj_weight,
        "v_proj.weight": v_proj_weight,
        "o_proj.weight": o_proj_weight,
    }
    mha.load_state_dict(state_dict)

    # Run the forward pass
    return mha(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    import torch
    from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
    from cs336_basics.rope import RotaryPositionalEmbedding
    from cs336_basics.utils import scaled_dot_product_attention

    # Create the multi-head attention module
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    # Load the provided weights
    state_dict = {
        "q_proj.weight": q_proj_weight,
        "k_proj.weight": k_proj_weight,
        "v_proj.weight": v_proj_weight,
        "o_proj.weight": o_proj_weight,
    }
    mha.load_state_dict(state_dict)

    # Create RoPE module for positional encoding
    d_head = d_model // num_heads
    rope = RotaryPositionalEmbedding(theta, d_head, max_seq_len)

    # We need to modify the forward pass to apply RoPE
    # Get the intermediate representations
    batch_size, seq_len, _ = in_features.shape

    # Project the input features to the query, key, and value tensors
    q = mha.q_proj(in_features)  # (batch_size, seq_len, d_model)
    k = mha.k_proj(in_features)  # (batch_size, seq_len, d_model)
    v = mha.v_proj(in_features)  # (batch_size, seq_len, d_model)

    # Reshape to separate heads: (batch_size, seq_len, num_heads, d_head)
    q = q.view(batch_size, seq_len, num_heads, d_head)
    k = k.view(batch_size, seq_len, num_heads, d_head)
    v = v.view(batch_size, seq_len, num_heads, d_head)

    # Apply RoPE to query and key vectors (not values)
    # RoPE should be applied per head
    for head in range(num_heads):
        q[:, :, head, :] = rope.forward(q[:, :, head, :], token_positions)
        k[:, :, head, :] = rope.forward(k[:, :, head, :], token_positions)

    # Transpose to get (batch_size, num_heads, seq_len, d_head)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Create causal mask to prevent attention to future tokens
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Apply scaled dot product attention with causal masking
    q_batched = q.reshape(batch_size * num_heads, seq_len, d_head)
    k_batched = k.reshape(batch_size * num_heads, seq_len, d_head)
    v_batched = v.reshape(batch_size * num_heads, seq_len, d_head)

    # Expand causal mask for all batch*head combinations
    mask_batched = causal_mask.expand(
        batch_size * num_heads, 1, seq_len, seq_len
    ).squeeze(1)

    # Apply attention using scaled_dot_product_attention
    attention_output = scaled_dot_product_attention(
        q_batched, k_batched, v_batched, mask_batched
    )

    # Reshape back to (batch_size, num_heads, seq_len, d_head)
    attention_output = attention_output.view(batch_size, num_heads, seq_len, d_head)

    # Transpose and reshape to concatenate heads: (batch_size, seq_len, d_model)
    output = (
        attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    )

    # Apply output projection
    output = mha.o_proj(output)

    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    from cs336_basics.rope import RotaryPositionalEmbedding

    # Create RoPE module
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)

    # Apply RoPE to input
    return rope.forward(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    from cs336_basics.transformer_block import TransformerBlock
    from cs336_basics.rope import RotaryPositionalEmbedding
    from cs336_basics.utils import scaled_dot_product_attention

    # Create the Transformer block
    transformer_block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)

    # Load the weights into the module
    # Map the weights dictionary keys to the TransformerBlock state dict keys
    state_dict = {
        # Attention weights
        "attn.q_proj.weight": weights["attn.q_proj.weight"],
        "attn.k_proj.weight": weights["attn.k_proj.weight"],
        "attn.v_proj.weight": weights["attn.v_proj.weight"],
        "attn.o_proj.weight": weights["attn.output_proj.weight"],
        # RMSNorm weights
        "ln1.weight": weights["ln1.weight"],
        "ln2.weight": weights["ln2.weight"],
        # Feed-forward network weights
        "ffn.w1": weights["ffn.w1.weight"],
        "ffn.w2": weights["ffn.w2.weight"],
        "ffn.w3": weights["ffn.w3.weight"],
    }

    transformer_block.load_state_dict(state_dict)

    # We need to manually implement the forward pass with RoPE since TransformerBlock doesn't have it built-in
    batch_size, seq_len, _ = in_features.shape

    # First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))
    # Apply RMSNorm first
    x_norm1 = transformer_block.ln1(in_features)

    # Apply multi-head self-attention with RoPE via the same adapter to match numerics
    attn_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=x_norm1,
        token_positions=torch.arange(seq_len, device=in_features.device),
    )

    # Add residual connection
    x = in_features + attn_output

    # Second sublayer: y = x + PositionwiseFeedForward(RMSNorm(x))
    # Apply RMSNorm first
    x_norm2 = transformer_block.ln2(x)
    # Apply feed-forward network
    ffn_output = transformer_block.ffn(x_norm2)
    # Add residual connection
    x = x + ffn_output

    return x


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    from cs336_basics.transformer_lm import TransformerLM

    # Create the Transformer language model
    transformer_lm = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=context_length,  # Use context_length as max_seq_len
        theta=rope_theta,
    )

    # Load the weights into the module
    state_dict = {}

    # Token embeddings
    state_dict["token_embeddings.embedding_matrix"] = weights["token_embeddings.weight"]

    # Transformer layers
    for layer_idx in range(num_layers):
        layer_prefix = f"layers.{layer_idx}."

        # Attention weights
        state_dict[f"layers.{layer_idx}.attn.q_proj.weight"] = weights[
            f"{layer_prefix}attn.q_proj.weight"
        ]
        state_dict[f"layers.{layer_idx}.attn.k_proj.weight"] = weights[
            f"{layer_prefix}attn.k_proj.weight"
        ]
        state_dict[f"layers.{layer_idx}.attn.v_proj.weight"] = weights[
            f"{layer_prefix}attn.v_proj.weight"
        ]
        state_dict[f"layers.{layer_idx}.attn.o_proj.weight"] = weights[
            f"{layer_prefix}attn.output_proj.weight"
        ]

        # RMSNorm weights
        state_dict[f"layers.{layer_idx}.ln1.weight"] = weights[
            f"{layer_prefix}ln1.weight"
        ]
        state_dict[f"layers.{layer_idx}.ln2.weight"] = weights[
            f"{layer_prefix}ln2.weight"
        ]

        # Feed-forward network weights
        state_dict[f"layers.{layer_idx}.ffn.w1"] = weights[
            f"{layer_prefix}ffn.w1.weight"
        ]
        state_dict[f"layers.{layer_idx}.ffn.w2"] = weights[
            f"{layer_prefix}ffn.w2.weight"
        ]
        state_dict[f"layers.{layer_idx}.ffn.w3"] = weights[
            f"{layer_prefix}ffn.w3.weight"
        ]

    # Final RMSNorm
    state_dict["ln_final.weight"] = weights["ln_final.weight"]

    # Language model head
    state_dict["lm_head.weight"] = weights["lm_head.weight"]

    transformer_lm.load_state_dict(state_dict)

    # Run the forward pass
    return transformer_lm(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    from cs336_basics.rms_norm import RMSNorm

    # Create RMSNorm module
    rmsnorm = RMSNorm(d_model, eps)

    # Load the weights into the module
    rmsnorm.load_state_dict({"weight": weights})

    # Apply RMSNorm to input features
    return rmsnorm.forward(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    from cs336_basics.utils import get_batch

    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    from cs336_basics.utils import softmax

    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    from cs336_basics.utils import cross_entropy

    return cross_entropy(inputs, targets)


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    from cs336_basics.utils import gradient_clipping

    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    from cs336_basics.adamw import AdamW

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    from cs336_basics.utils import learning_rate_schedule

    return learning_rate_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    from cs336_basics.utils import save_checkpoint

    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    from cs336_basics.utils import load_checkpoint

    return load_checkpoint(src, model, optimizer)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Create reverse mapping from bytes to token ID
        self.token_to_id = {
            token_bytes: token_id for token_id, token_bytes in vocab.items()
        }

        # Add special tokens to vocab if they don't exist
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.token_to_id:
                # Find the next available ID
                next_id = max(vocab.keys()) + 1 if vocab else 0
                self.vocab[next_id] = special_bytes
                self.token_to_id[special_bytes] = next_id

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if not text:
            return []

        # First, split text by special tokens to preserve them
        segments = self._split_by_special_tokens(text)

        token_ids = []
        for segment in segments:
            if segment in self.special_tokens:
                # This is a special token
                special_bytes = segment.encode("utf-8")
                if special_bytes in self.token_to_id:
                    token_ids.append(self.token_to_id[special_bytes])
            else:
                # This is regular text, apply GPT-2 regex pattern
                import regex as re

                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                pattern = re.compile(PAT)

                for match in pattern.finditer(segment):
                    pretoken = match.group()
                    if not pretoken:
                        continue

                    # Encode the pretoken using BPE
                    pretoken_ids = self._encode_pretoken(pretoken)
                    token_ids.extend(pretoken_ids)

        return token_ids

    def _split_by_special_tokens(self, text: str) -> list[str]:
        """Split text by special tokens while preserving the special tokens."""
        if not self.special_tokens:
            return [text]

        # Sort special tokens by length (longest first) to handle overlapping tokens
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        # Find all occurrences of special tokens in the text
        matches = []
        for special_token in sorted_special_tokens:
            start = 0
            while True:
                pos = text.find(special_token, start)
                if pos == -1:
                    break
                matches.append((pos, pos + len(special_token), special_token))
                start = pos + 1

        # Sort matches by position
        matches.sort(key=lambda x: x[0])

        # Remove overlapping matches (keep the longest one when there's overlap)
        filtered_matches = []
        for match in matches:
            start, end, token = match
            # Check if this match overlaps with any existing match
            overlaps = False
            for existing_start, existing_end, _ in filtered_matches:
                if not (end <= existing_start or start >= existing_end):
                    overlaps = True
                    break
            if not overlaps:
                filtered_matches.append(match)

        # Sort matches by position again
        filtered_matches.sort(key=lambda x: x[0])

        # Split text based on matches
        segments = []
        last_end = 0

        for start, end, token in filtered_matches:
            # Add text before the special token
            if start > last_end:
                segments.append(text[last_end:start])
            # Add the special token
            segments.append(token)
            last_end = end

        # Add remaining text after the last special token
        if last_end < len(text):
            segments.append(text[last_end:])

        return segments

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""

        # Convert token IDs to bytes
        token_bytes_list = []
        for token_id in token_ids:
            if token_id in self.vocab:
                token_bytes_list.append(self.vocab[token_id])
            else:
                # Handle unknown token IDs
                token_bytes_list.append(b"")

        # Concatenate all bytes
        all_bytes = b"".join(token_bytes_list)

        # Decode to string, handling invalid UTF-8
        try:
            return all_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Encode text from an iterable, yielding token IDs one at a time."""
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """Encode a single pretoken using BPE."""
        # Convert pretoken to bytes
        pretoken_bytes = pretoken.encode("utf-8")

        # Start with individual bytes
        current_tokens = [bytes([b]) for b in pretoken_bytes]

        # Apply BPE merges in order
        for token1, token2 in self.merges:
            merged = token1 + token2

            # Keep applying this merge until no more can be applied
            while True:
                new_tokens = []
                i = 0
                merged_this_round = False

                while i < len(current_tokens):
                    if (
                        i < len(current_tokens) - 1
                        and current_tokens[i] == token1
                        and current_tokens[i + 1] == token2
                    ):
                        new_tokens.append(merged)
                        i += 2
                        merged_this_round = True
                    else:
                        new_tokens.append(current_tokens[i])
                        i += 1

                current_tokens = new_tokens

                # If no merges were applied this round, move to next merge
                if not merged_this_round:
                    break

        # Convert final tokens to IDs
        token_ids = []
        for token_bytes in current_tokens:
            if token_bytes in self.token_to_id:
                token_ids.append(self.token_to_id[token_bytes])
            else:
                # Handle unknown tokens by encoding as individual bytes
                for b in token_bytes:
                    byte_token = bytes([b])
                    if byte_token in self.token_to_id:
                        token_ids.append(self.token_to_id[byte_token])

        return token_ids


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe_slow(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Read the input file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Vocabulary initialization
    vocab = {}
    next_id = 0

    # Add special tokens first
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    # Add all 256 bytes
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    # 2. Pre-tokenization using the GPT-2 regex pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Count pre-tokens and their frequencies, and convert to tokens in one pass
    pretoken_counts = Counter()
    pretoken_tokens = {}

    # Create a set of all substrings of special tokens to prevent merging
    # Only if the special token actually appears in the text
    special_token_substrings = set()
    for special_token in special_tokens:
        if special_token in text:
            special_bytes = special_token.encode("utf-8")
            for i in range(len(special_bytes)):
                for j in range(i + 1, len(special_bytes) + 1):
                    special_token_substrings.add(special_bytes[i:j])

    # Pre-compile the regex for better performance
    pattern = re.compile(PAT)
    for match in pattern.finditer(text):
        pretoken = match.group()
        if pretoken in special_tokens or len(pretoken) == 0:
            continue
        pretoken_counts[pretoken] += 1
        if pretoken not in pretoken_tokens:
            pretoken_tokens[pretoken] = [bytes([b]) for b in pretoken.encode("utf-8")]

    # 3. Compute BPE merges
    merges = []
    target_vocab_size = vocab_size

    while len(vocab) < target_vocab_size:
        # Count pairs within each pre-token (not crossing boundaries)
        pair_counts = Counter()
        for pretoken, count in pretoken_counts.items():
            tokens = pretoken_tokens[pretoken]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += count

        if not pair_counts:
            break

        # Find most frequent pair with lexicographic tie-breaking
        # Skip pairs that would create tokens that are parts of special tokens
        valid_pairs = [
            (pair, count)
            for pair, count in pair_counts.items()
            if pair[0] + pair[1] not in special_token_substrings
        ]

        if not valid_pairs:
            break

        most_common_pair = max(valid_pairs, key=lambda x: (x[1], x[0]))[0]

        # Add to vocabulary and merges
        vocab[next_id] = most_common_pair[0] + most_common_pair[1]
        next_id += 1
        merges.append(most_common_pair)

        # Update pre-tokens by merging the pair within each pre-token
        new_pretoken_counts = Counter()
        new_pretoken_tokens = {}

        # Pre-compute the merged token to avoid repeated concatenation
        merged_token = most_common_pair[0] + most_common_pair[1]

        for pretoken, count in pretoken_counts.items():
            tokens = pretoken_tokens[pretoken]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == most_common_pair[0]
                    and tokens[i + 1] == most_common_pair[1]
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # Convert back to string for deduplication
            new_pretoken = b"".join(new_tokens).decode("utf-8", errors="ignore")
            new_pretoken_counts[new_pretoken] += count
            if new_pretoken not in new_pretoken_tokens:
                new_pretoken_tokens[new_pretoken] = new_tokens

        pretoken_counts = new_pretoken_counts
        pretoken_tokens = new_pretoken_tokens

    return vocab, merges


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Highly optimized BPE tokenizer training implementation with cached pair counts."""

    # Read the input file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Vocabulary initialization
    vocab = {}
    next_id = 0

    # Add special tokens first
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    # Add all 256 bytes
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    # 2. Pre-tokenization using the GPT-2 regex pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)

    # Create a set of all substrings of special tokens to prevent merging
    # Only if the special token actually appears in the text
    special_token_substrings = set()
    for special_token in special_tokens:
        if special_token in text:
            special_bytes = special_token.encode("utf-8")
            for i in range(len(special_bytes)):
                for j in range(i + 1, len(special_bytes) + 1):
                    special_token_substrings.add(special_bytes[i:j])

    # Process text to get initial tokens - use Counter for efficiency
    pretoken_counts = Counter()
    pretoken_tokens = {}

    start_time = time.time()
    for match in pattern.finditer(text):
        pretoken = match.group()
        if pretoken in special_tokens or len(pretoken) == 0:
            continue
        pretoken_counts[pretoken] += 1
        if pretoken not in pretoken_tokens:
            pretoken_tokens[pretoken] = [bytes([b]) for b in pretoken.encode("utf-8")]
    end_time = time.time()
    print(f"Time taken to pre-tokenize: {end_time - start_time} seconds")

    # 3. Initialize pair count cache
    # This is the key optimization: maintain a cache of all pair counts
    pair_counts = Counter()
    pair_to_pretokens = defaultdict(
        set
    )  # Maps (token1, token2) -> set of pretokens containing this pair

    # Build initial pair counts and mapping
    for pretoken, count in pretoken_counts.items():
        tokens = pretoken_tokens[pretoken]
        if len(tokens) < 2:
            continue
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
            pair_to_pretokens[pair].add(pretoken)

    # 4. Compute BPE merges with incremental updates
    merges = []
    target_vocab_size = vocab_size

    while len(vocab) < target_vocab_size:
        if not pair_counts:
            break

        # Find most frequent pair with lexicographic tie-breaking
        # Skip pairs that would create tokens that are parts of special tokens
        valid_pairs = [
            (pair, count)
            for pair, count in pair_counts.items()
            if pair[0] + pair[1] not in special_token_substrings
        ]

        if not valid_pairs:
            break

        most_common_pair = max(valid_pairs, key=lambda x: (x[1], x[0]))[0]

        # Add to vocabulary and merges
        merged_token = most_common_pair[0] + most_common_pair[1]
        vocab[next_id] = merged_token
        next_id += 1
        merges.append(most_common_pair)

        # Get affected pretokens
        affected_pretokens = pair_to_pretokens[most_common_pair].copy()

        # Remove the merged pair from counts
        del pair_counts[most_common_pair]
        del pair_to_pretokens[most_common_pair]

        # Process each affected pretoken
        for pretoken in affected_pretokens:
            if pretoken not in pretoken_counts:
                continue

            tokens = pretoken_tokens[pretoken]
            count = pretoken_counts[pretoken]

            # Remove old pairs that are no longer present
            for i in range(len(tokens) - 1):
                old_pair = (tokens[i], tokens[i + 1])
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                pair_to_pretokens[old_pair].discard(pretoken)
                if not pair_to_pretokens[old_pair]:
                    del pair_to_pretokens[old_pair]

            # Apply merge to create new tokens
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == most_common_pair[0]
                    and tokens[i + 1] == most_common_pair[1]
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # Update the pretoken
            pretoken_tokens[pretoken] = new_tokens

            # Add new pairs to counts
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                pair_counts[new_pair] += count
                pair_to_pretokens[new_pair].add(pretoken)

    return vocab, merges
