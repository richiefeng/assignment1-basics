#!/usr/bin/env python3
"""
Test script to demonstrate the training functionality.
This script creates a small model and runs a few training steps to verify everything works.
"""

import os
import tempfile
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add the cs336_basics package to the path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.utils import (
    get_batch,
    learning_rate_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
)


def create_test_data(num_tokens=1000, vocab_size=1000):
    """Create synthetic test data."""
    # Create random token IDs
    data = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.int64)
    return data


def test_training_components():
    """Test all training components work together."""
    print("Testing training components...")

    # Configuration
    config = {
        "vocab_size": 1000,
        "context_length": 64,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 512,
        "rope_theta": 10000.0,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "use_lr_schedule": True,
        "max_learning_rate": 1e-3,
        "min_learning_rate": 1e-5,
        "warmup_iters": 10,
        "cosine_cycle_iters": 100,
    }

    # Create test data
    print("Creating test data...")
    train_data = create_test_data(1000, config["vocab_size"])
    val_data = create_test_data(500, config["vocab_size"])

    # Create model
    print("Creating model...")
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Create optimizer
    print("Creating optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )

    # Test data loading
    print("Testing data loading...")
    device = "cpu"
    input_ids, target_ids = get_batch(
        train_data, config["batch_size"], config["context_length"], device
    )
    print(f"Batch shape: {input_ids.shape}, {target_ids.shape}")

    # Test forward pass
    print("Testing forward pass...")
    model.train()
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")

    # Test loss calculation
    print("Testing loss calculation...")
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        ignore_index=-1,
    )
    print(f"Initial loss: {loss.item():.4f}")

    # Test training step
    print("Testing training step...")
    optimizer.zero_grad()
    loss.backward()

    # Test gradient clipping
    print("Testing gradient clipping...")
    gradient_clipping(model.parameters(), config["grad_clip_norm"])

    # Test optimizer step
    optimizer.step()
    print("Training step completed successfully!")

    # Test learning rate scheduling
    print("Testing learning rate scheduling...")
    for step in range(5):
        lr = learning_rate_schedule(
            step,
            config["max_learning_rate"],
            config["min_learning_rate"],
            config["warmup_iters"],
            config["cosine_cycle_iters"],
        )
        print(f"Step {step}: LR = {lr:.6f}")

    # Test checkpointing
    print("Testing checkpointing...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        checkpoint_path = tmp_file.name

    # Save checkpoint
    save_checkpoint(model, optimizer, 1, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Load checkpoint
    loaded_iteration = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"Checkpoint loaded, iteration: {loaded_iteration}")

    # Clean up
    os.unlink(checkpoint_path)
    print("Checkpoint file cleaned up.")

    # Test evaluation
    print("Testing evaluation...")
    model.eval()
    with torch.no_grad():
        val_input_ids, val_target_ids = get_batch(
            val_data, config["batch_size"], config["context_length"], device
        )
        val_logits = model(val_input_ids)
        val_loss = nn.functional.cross_entropy(
            val_logits.view(-1, val_logits.size(-1)),
            val_target_ids.view(-1),
            ignore_index=-1,
        )
        print(f"Validation loss: {val_loss.item():.4f}")

    print("\n✅ All training components tested successfully!")
    return True


def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting configuration loading...")

    # Create a simple config
    config_content = """
vocab_size: 1000
context_length: 64
d_model: 128
num_layers: 2
num_heads: 4
d_ff: 512
rope_theta: 10000.0
num_epochs: 10
batch_size: 4
learning_rate: 1e-3
betas: [0.9, 0.999]
eps: 1e-8
weight_decay: 0.01
grad_clip_norm: 1.0
use_lr_schedule: true
max_learning_rate: 1e-3
min_learning_rate: 1e-5
warmup_iters: 10
cosine_cycle_iters: 100
train_data_path: "data/train.txt"
val_data_path: "data/val.txt"
checkpoint_dir: "checkpoints"
checkpoint_interval: 5
log_level: "INFO"
log_file: "test_training.log"
log_interval: 10
"""

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp_file:
        tmp_file.write(config_content)
        config_path = tmp_file.name

    try:
        # Test loading config
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print("Configuration loaded successfully!")
        print(
            f"Model parameters: {config['vocab_size']} vocab, {config['d_model']} dim, {config['num_layers']} layers"
        )
        print(
            f"Training parameters: {config['num_epochs']} epochs, {config['batch_size']} batch size"
        )

        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    finally:
        os.unlink(config_path)


if __name__ == "__main__":
    print("🚀 Testing Training Script Components")
    print("=" * 50)

    # Test training components
    success1 = test_training_components()

    # Test configuration loading
    success2 = test_config_loading()

    if success1 and success2:
        print("\n🎉 All tests passed! The training script is ready to use.")
        print("\nTo start training:")
        print("1. Create a configuration file (see config_simple.yaml)")
        print("2. Prepare your training data")
        print("3. Run: python train.py --config your_config.yaml")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

