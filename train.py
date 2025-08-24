#!/usr/bin/env python3
"""
Training script for Transformer Language Model

This script integrates all the components we've implemented:
- BPE Tokenizer
- Transformer Language Model
- AdamW Optimizer
- Learning Rate Scheduling
- Gradient Clipping
- Data Loading
- Checkpointing
- Logging

Usage:
    python train.py --config config.yaml
    python train.py --help
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add the cs336_basics package to the path
sys.path.insert(0, str(Path(__file__).parent))

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.utils import (
    get_batch,
    learning_rate_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
)


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("train")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_dataset(data_path: str, vocab_size: int) -> np.ndarray:
    """Load dataset using memory mapping for efficiency."""
    logger = logging.getLogger("train")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Try to load as numpy array first
    try:
        if data_path.endswith(".npy"):
            dataset = np.load(data_path, mmap_mode="r")
        elif data_path.endswith(".npz"):
            # For .npz files, assume the first array is the dataset
            with np.load(data_path, mmap_mode="r") as data:
                dataset = data[data.files[0]]
        else:
            # Try to load as text and tokenize
            logger.info(f"Loading text file and tokenizing: {data_path}")
            dataset = load_and_tokenize_text(data_path, vocab_size)

        logger.info(f"Dataset loaded: {len(dataset)} tokens")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def load_and_tokenize_text(text_path: str, vocab_size: int) -> np.ndarray:
    """Load text file and tokenize it using BPE tokenizer."""
    logger = logging.getLogger("train")

    # Load pre-trained tokenizer (you would need to provide these files)
    vocab_path = "data/gpt2_vocab.json"
    merges_path = "data/gpt2_merges.txt"

    if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
        logger.warning(
            "Pre-trained tokenizer files not found, using simple tokenization"
        )
        return simple_tokenize_text(text_path)

    # Load vocabulary and merges
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    with open(merges_path, "r") as f:
        merges = []
        for line in f:
            if line.strip():
                parts = line.strip().split()
                merges.append((bytes(parts[0], "utf-8"), bytes(parts[1], "utf-8")))

    # Create tokenizer
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    # Tokenize the text
    logger.info("Tokenizing text file...")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text)
    logger.info(f"Text tokenized: {len(token_ids)} tokens")

    return np.array(token_ids, dtype=np.int64)


def simple_tokenize_text(text_path: str) -> np.ndarray:
    """Simple character-level tokenization as fallback."""
    logger = logging.getLogger("train")
    logger.info("Using simple character-level tokenization")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Create simple vocabulary (characters + special tokens)
    chars = sorted(list(set(text)))
    vocab = {i: char for i, char in enumerate(chars)}
    vocab[len(vocab)] = "<|endoftext|>"

    # Tokenize
    token_ids = []
    for char in text:
        if char in vocab:
            token_ids.append(list(vocab.keys())[list(vocab.values()).index(char)])
        else:
            token_ids.append(0)  # Unknown character

    return np.array(token_ids, dtype=np.int64)


def create_model(config: Dict[str, Any], device: str) -> TransformerLM:
    """Create and initialize the Transformer Language Model."""
    logger = logging.getLogger("train")

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

    # Move to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> AdamW:
    """Create and configure the AdamW optimizer."""
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
    )
    return optimizer


def evaluate_model(
    model: nn.Module,
    dataset: np.ndarray,
    config: Dict[str, Any],
    device: str,
    num_eval_batches: int = 10,
) -> Dict[str, float]:
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(num_eval_batches):
            # Get batch
            input_ids, target_ids = get_batch(
                dataset,
                config["batch_size"],
                config["context_length"],
                device,
            )

            # Forward pass
            logits = model(input_ids)

            # Calculate loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-1,
            )

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
    }


def train_epoch(
    model: nn.Module,
    optimizer: AdamW,
    dataset: np.ndarray,
    config: Dict[str, Any],
    device: str,
    epoch: int,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    # Calculate steps per epoch
    steps_per_epoch = len(dataset) // (config["batch_size"] * config["context_length"])

    start_time = time.time()

    for step in range(steps_per_epoch):
        # Get batch
        input_ids, target_ids = get_batch(
            dataset,
            config["batch_size"],
            config["context_length"],
            device,
        )

        # Forward pass
        logits = model(input_ids)

        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-1,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config["grad_clip_norm"] > 0:
            gradient_clipping(model.parameters(), config["grad_clip_norm"])

        # Optimizer step
        optimizer.step()

        # Update learning rate
        if config["use_lr_schedule"]:
            current_lr = learning_rate_schedule(
                step + epoch * steps_per_epoch,
                config["max_learning_rate"],
                config["min_learning_rate"],
                config["warmup_iters"],
                config["cosine_cycle_iters"],
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        # Accumulate statistics
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        num_batches += 1

        # Logging
        if step % config["log_interval"] == 0:
            avg_loss = total_loss / total_tokens
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time

            logger.info(
                f"Epoch {epoch}, Step {step}/{steps_per_epoch}, "
                f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, "
                f"Time: {elapsed:.2f}s"
            )

    epoch_loss = total_loss / total_tokens
    epoch_time = time.time() - start_time

    return {
        "loss": epoch_loss,
        "time": epoch_time,
        "steps": steps_per_epoch,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto/cpu/cuda/mps)"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(args.log_level, config.get("log_file"))
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")

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

    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading training dataset...")
    train_dataset = load_dataset(config["train_data_path"], config["vocab_size"])

    if "val_data_path" in config:
        logger.info("Loading validation dataset...")
        val_dataset = load_dataset(config["val_data_path"], config["vocab_size"])
    else:
        val_dataset = None

    # Create model
    logger.info("Creating model...")
    model = create_model(config, device)

    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(model, config)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info("Starting training loop...")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")

        # Train
        train_metrics = train_epoch(
            model, optimizer, train_dataset, config, device, epoch, logger
        )

        # Evaluate
        if val_dataset is not None:
            val_metrics = evaluate_model(model, val_dataset, config, device)

            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val PPL: {val_metrics['perplexity']:.2f}"
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_model_path = os.path.join(
                    config["checkpoint_dir"], "best_model.pt"
                )
                save_checkpoint(model, optimizer, epoch + 1, best_model_path)
                logger.info(f"New best model saved: {best_model_path}")
        else:
            logger.info(
                f"Epoch {epoch + 1} - " f"Train Loss: {train_metrics['loss']:.4f}"
            )

        # Save checkpoint
        if (epoch + 1) % config["checkpoint_interval"] == 0:
            checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"checkpoint_epoch_{epoch + 1}.pt"
            )
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_checkpoint_path = os.path.join(config["checkpoint_dir"], "final_model.pt")
    save_checkpoint(model, optimizer, config["num_epochs"], final_checkpoint_path)
    logger.info(f"Final model saved: {final_checkpoint_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()

