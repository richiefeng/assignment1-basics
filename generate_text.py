#!/usr/bin/env python3
"""
Text Generation Script for Transformer Language Model

This script demonstrates the text generation capabilities of our trained model,
including temperature scaling, top-p sampling, and various generation controls.

Usage:
    python generate_text.py --model path/to/model.pt --prompt "Your prompt here"
    python generate_text.py --help
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add the cs336_basics package to the path
sys.path.insert(0, str(Path(__file__).parent))

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.utils import (
    decode,
    batch_decode,
    generate_text_with_controls,
    softmax_with_temperature,
    top_p_sampling,
)


def load_model_and_tokenizer(model_path: str, config: dict, device: str):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {model_path}...")

    # Create model with the same architecture as training
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_seq_len=config["context_length"],
        theta=config["rope_theta"],
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(
        f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create tokenizer (you may need to provide vocab and merges files)
    try:
        vocab_path = "data/gpt2_vocab.json"
        merges_path = "data/gpt2_merges.txt"

        if os.path.exists(vocab_path) and os.path.exists(merges_path):
            import json

            with open(vocab_path, "r") as f:
                vocab = json.load(f)

            with open(merges_path, "r") as f:
                merges = []
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        merges.append(
                            (bytes(parts[0], "utf-8"), bytes(parts[1], "utf-8"))
                        )

            tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
            print("BPE tokenizer loaded successfully")
        else:
            print("BPE tokenizer files not found, using simple character tokenizer")
            tokenizer = create_simple_tokenizer()
    except Exception as e:
        print(f"Failed to load BPE tokenizer: {e}")
        print("Using simple character tokenizer")
        tokenizer = create_simple_tokenizer()

    return model, tokenizer


def create_simple_tokenizer():
    """Create a simple character-level tokenizer as fallback."""

    class SimpleTokenizer:
        def __init__(self):
            # Create a simple vocabulary with common characters
            self.chars = list(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}-_+=@#$%^&*|\\/<>~`"
            )
            self.vocab = {i: char for i, char in enumerate(self.chars)}
            self.vocab[len(self.vocab)] = "<|endoftext|>"
            self.token_to_id = {char: i for i, char in enumerate(self.chars)}
            self.token_to_id["<|endoftext|>"] = len(self.chars)

        def encode(self, text: str) -> list[int]:
            """Encode text to token IDs."""
            token_ids = []
            for char in text:
                if char in self.token_to_id:
                    token_ids.append(self.token_to_id[char])
                else:
                    # Handle unknown characters
                    token_ids.append(0)
            return token_ids

        def decode(self, token_ids: list[int]) -> str:
            """Decode token IDs back to text."""
            text = ""
            for token_id in token_ids:
                if token_id in self.vocab:
                    text += self.vocab[token_id]
                else:
                    text += "?"
            return text

    return SimpleTokenizer()


def demonstrate_temperature_scaling(model, tokenizer, prompt: str, device: str):
    """Demonstrate the effect of temperature on text generation."""
    print("\n" + "=" * 60)
    print("TEMPERATURE SCALING DEMONSTRATION")
    print("=" * 60)

    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 40)

        try:
            generated_text = decode(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=temp,
                top_p=1.0,
                device=device,
            )

            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")

        except Exception as e:
            print(f"Error with temperature {temp}: {e}")


def demonstrate_top_p_sampling(model, tokenizer, prompt: str, device: str):
    """Demonstrate the effect of top-p sampling on text generation."""
    print("\n" + "=" * 60)
    print("TOP-P SAMPLING DEMONSTRATION")
    print("=" * 60)

    top_p_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    for p in top_p_values:
        print(f"\nTop-p: {p}")
        print("-" * 40)

        try:
            generated_text = decode(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=1.0,
                top_p=p,
                device=device,
            )

            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")

        except Exception as e:
            print(f"Error with top-p {p}: {e}")


def demonstrate_combined_controls(model, tokenizer, prompt: str, device: str):
    """Demonstrate combining temperature and top-p sampling."""
    print("\n" + "=" * 60)
    print("COMBINED CONTROLS DEMONSTRATION")
    print("=" * 60)

    # Different combinations of temperature and top-p
    combinations = [
        (0.5, 0.3),  # Low temperature, low top-p (focused, deterministic)
        (0.5, 0.9),  # Low temperature, high top-p (focused, diverse)
        (1.5, 0.3),  # High temperature, low top-p (creative, focused)
        (1.5, 0.9),  # High temperature, high top-p (creative, diverse)
    ]

    for temp, top_p in combinations:
        print(f"\nTemperature: {temp}, Top-p: {top_p}")
        print("-" * 50)

        try:
            generated_text = decode(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=temp,
                top_p=top_p,
                device=device,
            )

            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")

        except Exception as e:
            print(f"Error with temp={temp}, top_p={top_p}: {e}")


def demonstrate_batch_generation(model, tokenizer, prompts: list[str], device: str):
    """Demonstrate batch text generation."""
    print("\n" + "=" * 60)
    print("BATCH GENERATION DEMONSTRATION")
    print("=" * 60)

    try:
        generated_texts = batch_decode(
            model=model,
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=30,
            temperature=1.0,
            top_p=0.9,
            device=device,
        )

        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            print(f"\nBatch {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")

    except Exception as e:
        print(f"Error in batch generation: {e}")


def demonstrate_detailed_generation(model, tokenizer, prompt: str, device: str):
    """Demonstrate detailed generation with statistics."""
    print("\n" + "=" * 60)
    print("DETAILED GENERATION WITH STATISTICS")
    print("=" * 60)

    try:
        result = generate_text_with_controls(
            model=model,
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=40,
            temperature=0.8,
            top_p=0.7,
            device=device,
            verbose=True,
        )

        print(f"\nGeneration Statistics:")
        print(f"Total steps: {result['generation_steps']}")
        print(f"Average probability: {result['avg_probability']:.4f}")
        print(
            f"Probability range: [{result['min_probability']:.4f}, {result['max_probability']:.4f}]"
        )
        print(f"Full text: {result['full_text']}")

    except Exception as e:
        print(f"Error in detailed generation: {e}")


def main():
    """Main function for text generation."""
    parser = argparse.ArgumentParser(
        description="Generate text using trained Transformer Language Model"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Text prompt to complete",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (0.1-2.0)",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="Top-p sampling threshold (0.1-1.0)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto/cpu/cuda/mps)"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run demonstration of all features"
    )

    args = parser.parse_args()

    # Device setup
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Default configuration (you can modify this or load from a config file)
    config = {
        "vocab_size": 1000,  # Adjust based on your model
        "context_length": 512,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 512,
        "rope_theta": 10000.0,
    }

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, config, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if args.demo:
        # Run comprehensive demonstration
        print("\n🚀 RUNNING COMPREHENSIVE TEXT GENERATION DEMONSTRATION")

        # Test prompts
        test_prompts = [
            "The future of artificial intelligence",
            "Once upon a time",
            "The best way to learn programming is",
        ]

        # Demonstrate temperature scaling
        demonstrate_temperature_scaling(model, tokenizer, test_prompts[0], device)

        # Demonstrate top-p sampling
        demonstrate_top_p_sampling(model, tokenizer, test_prompts[1], device)

        # Demonstrate combined controls
        demonstrate_combined_controls(model, tokenizer, test_prompts[2], device)

        # Demonstrate batch generation
        demonstrate_batch_generation(model, tokenizer, test_prompts, device)

        # Demonstrate detailed generation
        demonstrate_detailed_generation(model, tokenizer, test_prompts[0], device)

    else:
        # Single generation with specified parameters
        print(f"\nGenerating text with prompt: '{args.prompt}'")
        print(
            f"Parameters: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}"
        )

        try:
            generated_text = decode(
                model=model,
                prompt=args.prompt,
                tokenizer=tokenizer,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )

            print(f"\nGenerated text:")
            print(f"Prompt: {args.prompt}")
            print(f"Completion: {generated_text}")
            print(f"Full text: {args.prompt + generated_text}")

        except Exception as e:
            print(f"Error during text generation: {e}")


if __name__ == "__main__":
    main()

