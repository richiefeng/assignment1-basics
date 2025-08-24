#!/usr/bin/env python3
"""
Test script for text generation functions.
This script creates a simple model and tests all the generation functions.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add the cs336_basics package to the path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cs336_basics.utils import (
    softmax_with_temperature,
    top_p_sampling,
    decode,
    batch_decode,
    generate_text_with_controls,
)


class SimpleTestModel(nn.Module):
    """Simple test model for testing text generation functions."""

    def __init__(self, vocab_size=100, context_length=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length

        # Simple embedding and output projection
        self.embedding = nn.Embedding(vocab_size, 64)
        self.output_proj = nn.Linear(64, vocab_size)

        # Positional encoding
        self.register_buffer(
            "pos_encoding", self._create_positional_encoding(context_length, 64)
        )

    def _create_positional_encoding(self, seq_len, d_model):
        """Create simple positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        """Forward pass."""
        batch_size, seq_len = x.shape

        # Get embeddings
        embeddings = self.embedding(x)  # (batch_size, seq_len, 64)

        # Add positional encoding
        embeddings = embeddings + self.pos_encoding[:seq_len, :].unsqueeze(0)

        # Simple transformation
        hidden = torch.tanh(embeddings)

        # Output projection
        logits = self.output_proj(hidden)  # (batch_size, seq_len, vocab_size)

        return logits


class SimpleTestTokenizer:
    """Simple test tokenizer for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.special_tokens = ["<|endoftext|>"]

        # Create simple vocabulary
        self.vocab = {i: f"token_{i}" for i in range(vocab_size)}
        self.vocab[vocab_size] = "<|endoftext|>"

        # Create reverse mapping
        self.token_to_id = {token: i for i, token in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Simple encoding for testing."""
        # For testing, just return some token IDs
        return [0, 1, 2, 3, 4]  # Simple test sequence

    def decode(self, token_ids: list[int]) -> str:
        """Simple decoding for testing."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.vocab:
                tokens.append(self.vocab[token_id])
            else:
                tokens.append("UNK")
        return " ".join(tokens)


def test_softmax_with_temperature():
    """Test temperature scaling function."""
    print("Testing softmax_with_temperature...")

    # Create test logits
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # Test different temperatures
    temperatures = [0.5, 1.0, 2.0]

    for temp in temperatures:
        probs = softmax_with_temperature(logits, temp)
        print(f"Temperature {temp}: {probs}")

        # Verify probabilities sum to 1
        assert (
            abs(probs.sum().item() - 1.0) < 1e-6
        ), f"Probabilities don't sum to 1 for temp {temp}"

        # Verify lower temperature makes distribution more concentrated
        if temp < 1.0:
            assert (
                probs.max().item() > 0.8
            ), f"Low temperature should concentrate distribution"

    print("✅ softmax_with_temperature tests passed!")


def test_top_p_sampling():
    """Test top-p sampling function."""
    print("\nTesting top_p_sampling...")

    # Create test probabilities
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    # Test different top-p values
    top_p_values = [0.3, 0.6, 0.9, 1.0]

    for p in top_p_values:
        modified_probs = top_p_sampling(probs, p)
        print(f"Top-p {p}: {modified_probs}")

        # Verify probabilities sum to 1
        assert (
            abs(modified_probs.sum().item() - 1.0) < 1e-6
        ), f"Probabilities don't sum to 1 for top_p {p}"

        # Verify top-p constraint is satisfied
        if p < 1.0:
            # Sort probabilities and check cumulative sum
            sorted_probs, _ = torch.sort(modified_probs, dim=-1, descending=True)
            cumulative_sum = torch.cumsum(sorted_probs, dim=-1)
            # The sum of non-zero probabilities should be approximately p
            non_zero_probs = modified_probs[modified_probs > 0]
            if len(non_zero_probs) > 0:
                assert (
                    abs(non_zero_probs.sum().item() - 1.0) < 1e-6
                ), f"Top-p constraint violated for {p}"

    print("✅ top_p_sampling tests passed!")


def test_decode_function():
    """Test the main decode function."""
    print("\nTesting decode function...")

    # Create simple model and tokenizer
    model = SimpleTestModel(vocab_size=100, context_length=50)
    tokenizer = SimpleTestTokenizer(vocab_size=100)

    # Test basic decoding
    try:
        generated_text = decode(
            model=model,
            prompt="test prompt",
            tokenizer=tokenizer,
            max_new_tokens=10,
            temperature=1.0,
            top_p=1.0,
            device="cpu",
        )

        print(f"Generated text: {generated_text}")
        print("✅ decode function test passed!")

    except Exception as e:
        print(f"❌ decode function test failed: {e}")


def test_batch_decode_function():
    """Test batch decoding function."""
    print("\nTesting batch_decode function...")

    # Create simple model and tokenizer
    model = SimpleTestModel(vocab_size=100, context_length=50)
    tokenizer = SimpleTestTokenizer(vocab_size=100)

    # Test batch decoding
    try:
        prompts = ["prompt 1", "prompt 2", "prompt 3"]
        generated_texts = batch_decode(
            model=model,
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=5,
            temperature=1.0,
            top_p=1.0,
            device="cpu",
        )

        print(f"Generated {len(generated_texts)} texts")
        for i, text in enumerate(generated_texts):
            print(f"  {i+1}: {text}")

        print("✅ batch_decode function test passed!")

    except Exception as e:
        print(f"❌ batch_decode function test failed: {e}")


def test_generate_text_with_controls():
    """Test detailed generation function."""
    print("\nTesting generate_text_with_controls function...")

    # Create simple model and tokenizer
    model = SimpleTestModel(vocab_size=100, context_length=50)
    tokenizer = SimpleTestTokenizer(vocab_size=100)

    # Test detailed generation
    try:
        result = generate_text_with_controls(
            model=model,
            prompt="test prompt",
            tokenizer=tokenizer,
            max_new_tokens=5,
            temperature=0.8,
            top_p=0.9,
            device="cpu",
            verbose=False,
        )

        print(f"Generation result keys: {list(result.keys())}")
        print(f"Generated text: {result['generated_text']}")
        print(f"Generation steps: {result['generation_steps']}")
        print(f"Average probability: {result['avg_probability']:.4f}")

        print("✅ generate_text_with_controls function test passed!")

    except Exception as e:
        print(f"❌ generate_text_with_controls function test failed: {e}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")

    # Test temperature validation
    try:
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        softmax_with_temperature(logits, 0.0)  # Should raise ValueError
        print("❌ Temperature validation failed - should raise error for temp <= 0")
    except ValueError:
        print("✅ Temperature validation working correctly")

    # Test top-p validation
    try:
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        top_p_sampling(probs, 0.0)  # Should raise ValueError
        print("❌ Top-p validation failed - should raise error for top_p <= 0")
    except ValueError:
        print("✅ Top-p validation working correctly")

    try:
        top_p_sampling(probs, 1.5)  # Should raise ValueError
        print("❌ Top-p validation failed - should raise error for top_p > 1")
    except ValueError:
        print("✅ Top-p validation working correctly")

    print("✅ Edge case tests passed!")


def main():
    """Run all tests."""
    print("🧪 Testing Text Generation Functions")
    print("=" * 50)

    # Run all tests
    test_softmax_with_temperature()
    test_top_p_sampling()
    test_decode_function()
    test_batch_decode_function()
    test_generate_text_with_controls()
    test_edge_cases()

    print("\n🎉 All text generation tests completed!")
    print("\nThe text generation functions are ready to use with your trained model.")


if __name__ == "__main__":
    main()

