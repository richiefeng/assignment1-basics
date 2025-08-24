import torch
import numpy as np
from typing import Optional


# Write a function to apply the softmax operation on a tensor
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Use PyTorch's built-in softmax for efficiency and numerical stability
    return torch.softmax(x, dim=dim)


# Implement the scaled dot-product attention function. Your implementation should
# handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
# (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
# dimensions (if provided)
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Get the dimension of the key vectors
    d_k = K.size(-1)

    # Compute attention scores: Q @ K^T
    # Q: (..., seq_len, d_k), K: (..., seq_len, d_k) -> (..., seq_len, seq_len)
    scores = torch.einsum("...ij,...kj->...ik", Q, K)

    # Scale the scores by sqrt(d_k)
    scores = scores / torch.sqrt(
        torch.tensor(d_k, dtype=scores.dtype, device=scores.device)
    )

    # Apply mask if provided (set masked positions to -inf)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Apply softmax along the last dimension (keys dimension)
    attention_weights = softmax(scores, dim=-1)

    # Apply attention weights to values: attention_weights @ V
    # attention_weights: (..., seq_len, seq_len), V: (..., seq_len, d_v) -> (..., seq_len, d_v)
    output = torch.einsum("...ij,...jk->...ik", attention_weights, V)

    return output


def cross_entropy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(x, y)


def learning_rate_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Implements cosine annealing learning rate schedule with warmup as used in LLaMA training.

    Args:
        t (int): Current iteration number
        alpha_max (float): Maximum learning rate (used during warmup and start of cosine annealing)
        alpha_min (float): Minimum/final learning rate (used after cosine annealing)
        warmup_iters (int): Number of warmup iterations (Tw)
        cosine_cycle_iters (int): Number of cosine annealing iterations (Tc)

    Returns:
        float: Learning rate at iteration t (αt)

    The schedule follows these phases:
    1. Warm-up (t < Tw): αt = (t/Tw) * αmax
    2. Cosine annealing (Tw ≤ t ≤ Tc): αt = αmin + 0.5 * (1 + cos((t-Tw)/(Tc-Tw) * π)) * (αmax - αmin)
    3. Post-annealing (t > Tc): αt = αmin
    """
    import math

    # Warm-up phase: linear increase from 0 to alpha_max
    if t < warmup_iters:
        return (t / warmup_iters) * alpha_max

    # Cosine annealing phase: cosine decay from alpha_max to alpha_min
    elif t <= cosine_cycle_iters:
        # Calculate progress through cosine phase (0 to 1)
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # Apply cosine function: cos(π * progress) goes from 1 to -1
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine_factor * (alpha_max - alpha_min)

    # Post-annealing phase: constant at alpha_min
    else:
        return alpha_min


def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6):
    """
    Implements gradient clipping to prevent exploding gradients during training.

    Args:
        parameters: Iterable of parameters (nn.Parameter) whose gradients will be clipped
        max_norm (float): Maximum L2 norm allowed for the combined gradients
        eps (float): Small value added for numerical stability (default: 1e-6)

    The function computes the L2 norm of all parameter gradients combined. If this norm
    exceeds max_norm, it scales down all gradients by a factor of max_norm / (norm + eps)
    so that the resulting norm is just under max_norm.

    This modifies the gradients in-place for all parameters.
    """
    # Collect all gradients into a single list
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad.data.view(-1))

    # If no gradients, nothing to do
    if not gradients:
        return

    # Concatenate all gradients into a single vector
    all_gradients = torch.cat(gradients)

    # Compute the L2 norm of the combined gradients
    total_norm = torch.norm(all_gradients, p=2)

    # If norm exceeds max_norm, scale down all gradients
    if total_norm > max_norm:
        # Compute scaling factor: max_norm / (total_norm + eps)
        scale_factor = max_norm / (total_norm + eps)

        # Apply scaling to all parameter gradients
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(scale_factor)


def get_batch(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates training batches from a sequence of token IDs.

    Args:
        x (np.ndarray): Integer array with token IDs (single sequence)
        batch_size (int): Number of sequences per batch (B)
        context_length (int): Length of each sequence (m)
        device (str): PyTorch device string ('cpu', 'cuda:0', 'mps', etc.)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A pair of tensors:
            - input_sequences: Shape (batch_size, context_length) containing input token sequences
            - target_sequences: Shape (batch_size, context_length) containing next-token targets

    The function samples random starting positions and creates sequences where:
    - input_sequences[i] = [x[start_i], x[start_i+1], ..., x[start_i+context_length-1]]
    - target_sequences[i] = [x[start_i+1], x[start_i+2], ..., x[start_i+context_length]]

    This creates a language modeling task where the model predicts the next token given the previous tokens.
    """
    # Calculate the maximum valid starting index
    # We need at least context_length + 1 tokens to create a sequence with its target
    max_start_idx = len(x) - context_length - 1

    if max_start_idx < 0:
        raise ValueError(
            f"Dataset too short: need at least {context_length + 1} tokens, but only have {len(x)}"
        )

    # Randomly sample starting indices for each batch
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Pre-allocate numpy arrays for better performance
    input_sequences = np.zeros((batch_size, context_length), dtype=x.dtype)
    target_sequences = np.zeros((batch_size, context_length), dtype=x.dtype)

    for i, start_idx in enumerate(start_indices):
        # Input sequence: tokens from start_idx to start_idx + context_length - 1
        input_sequences[i] = x[start_idx : start_idx + context_length]
        # Target sequence: tokens from start_idx + 1 to start_idx + context_length
        target_sequences[i] = x[start_idx + 1 : start_idx + context_length + 1]

    # Convert to PyTorch tensors and move to specified device
    input_tensor = torch.from_numpy(input_sequences).to(device=device, dtype=torch.long)
    target_tensor = torch.from_numpy(target_sequences).to(
        device=device, dtype=torch.long
    )

    return input_tensor, target_tensor


def save_checkpoint(model, optimizer, iteration, out):
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        iteration (int): Current training iteration number
        out (str | os.PathLike | BinaryIO | IO[bytes]): Output path or file-like object

    The checkpoint contains:
    - model_state_dict: Model parameters and buffers
    - optimizer_state_dict: Optimizer state (momentum, learning rate, etc.)
    - iteration: Training iteration number
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """
    Load model and optimizer state from a checkpoint file.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Checkpoint file path or file-like object
        model (torch.nn.Module): The model to restore
        optimizer (torch.optim.Optimizer): The optimizer to restore

    Returns:
        int: The iteration number that was saved in the checkpoint

    The function restores:
    - Model parameters and buffers from model_state_dict
    - Optimizer state from optimizer_state_dict
    - Returns the saved iteration number
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def softmax_with_temperature(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply softmax with temperature scaling to logits.

    Args:
        logits (torch.Tensor): Input logits of shape (..., vocab_size)
        temperature (float): Temperature parameter. Lower values make distribution more concentrated.
                           Higher values make it more uniform.

    Returns:
        torch.Tensor: Temperature-scaled probability distribution
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    # Apply temperature scaling: logits / temperature
    scaled_logits = logits / temperature

    # Apply softmax
    return torch.softmax(scaled_logits, dim=-1)


def top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution of shape (..., vocab_size)
        top_p (float): Cumulative probability threshold (0 < top_p <= 1)

    Returns:
        torch.Tensor: Modified probability distribution with top-p sampling applied
    """
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1]")

    # Handle 1D tensors by adding a batch dimension
    original_shape = probs.shape
    if len(original_shape) == 1:
        probs = probs.unsqueeze(0)
        was_1d = True
    else:
        was_1d = False

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff index where cumulative probability >= top_p
    # We want the smallest set of indices such that sum >= top_p
    cutoff_mask = cumulative_probs < top_p

    # For the last dimension, we need to handle the case where we might not reach top_p
    # In that case, we include at least one token
    cutoff_mask[..., -1] = True

    # Create the modified probability distribution
    modified_probs = torch.zeros_like(probs)

    # For each batch item, apply top-p sampling
    for batch_idx in range(probs.shape[0]):
        # Get the cutoff mask for this batch item
        item_cutoff_mask = cutoff_mask[batch_idx]

        # Get the sorted probabilities and indices for this item
        item_sorted_probs = sorted_probs[batch_idx]
        item_sorted_indices = sorted_indices[batch_idx]

        # Find which tokens to keep based on cutoff mask
        keep_indices = item_sorted_indices[item_cutoff_mask]

        # Set probabilities for kept tokens
        for keep_idx in keep_indices:
            modified_probs[batch_idx, keep_idx] = probs[batch_idx, keep_idx]

    # Renormalize the probabilities
    modified_probs = modified_probs / (modified_probs.sum(dim=-1, keepdim=True) + 1e-8)

    # Remove the batch dimension if the input was 1D
    if was_1d:
        modified_probs = modified_probs.squeeze(0)

    return modified_probs


def decode(
    model: torch.nn.Module,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> str:
    """
    Generate text completion from a language model given a prompt.

    Args:
        model (torch.nn.Module): The trained language model
        prompt (str): Input text prompt
        tokenizer: Tokenizer with encode/decode methods
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for softmax scaling (higher = more random)
        top_p (float): Top-p sampling threshold (0 < top_p <= 1)
        end_token (str): Token that signals end of generation
        device (str): Device to run the model on

    Returns:
        str: Generated text completion
    """
    model.eval()

    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("Failed to encode prompt")

    # Convert to tensor and move to device
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Track generated tokens
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass through the model
            logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

            # Get the last token's logits (next token prediction)
            next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

            # Apply temperature scaling
            if temperature != 1.0:
                probs = softmax_with_temperature(next_token_logits, temperature)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)

            # Apply top-p sampling
            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)

            # Sample from the probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # Check if we hit the end token
            if tokenizer.decode([next_token_id]) == end_token:
                break

            # Add the generated token to our sequence
            generated_tokens.append(next_token_id)

            # Append to input for next iteration
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=device)], dim=1
            )

            # Check if we've exceeded the model's context length
            if input_ids.shape[1] > getattr(model, "context_length", float("inf")):
                break

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def batch_decode(
    model: torch.nn.Module,
    prompts: list[str],
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> list[str]:
    """
    Generate text completions for multiple prompts in batch.

    Args:
        model (torch.nn.Module): The trained language model
        prompts (list[str]): List of input text prompts
        tokenizer: Tokenizer with encode/decode methods
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for softmax scaling
        top_p (float): Top-p sampling threshold
        end_token (str): Token that signals end of generation
        device (str): Device to run the model on

    Returns:
        list[str]: List of generated text completions
    """
    model.eval()

    # Encode all prompts
    prompt_tokens_list = [tokenizer.encode(prompt) for prompt in prompts]

    # Find the maximum prompt length for padding
    max_prompt_len = max(len(tokens) for tokens in prompt_tokens_list)

    # Pad all prompts to the same length
    padded_prompts = []
    for tokens in prompt_tokens_list:
        # Pad with the first token or a special padding token
        padding_length = max_prompt_len - len(tokens)
        if padding_length > 0:
            padded_tokens = (
                tokens + [tokens[0]] * padding_length
            )  # Simple padding strategy
        else:
            padded_tokens = tokens
        padded_prompts.append(padded_tokens)

    # Convert to tensor and move to device
    input_ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)
    batch_size = input_ids.shape[0]

    # Track generated tokens for each prompt
    generated_tokens = [[] for _ in range(batch_size)]
    active_prompts = [True] * batch_size  # Track which prompts are still generating

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Check if all prompts are done
            if not any(active_prompts):
                break

            # Forward pass through the model
            logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

            # Get the last token's logits for each active prompt
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

            # Apply temperature scaling
            if temperature != 1.0:
                probs = softmax_with_temperature(next_token_logits, temperature)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)

            # Apply top-p sampling
            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)

            # Sample from the probability distribution for each prompt
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(
                -1
            )  # Shape: (batch_size,)

            # Process each prompt
            for i in range(batch_size):
                if not active_prompts[i]:
                    continue

                next_token_id = next_token_ids[i].item()

                # Check if we hit the end token
                if tokenizer.decode([next_token_id]) == end_token:
                    active_prompts[i] = False
                    continue

                # Add the generated token to our sequence
                generated_tokens[i].append(next_token_id)

                # Check if we've exceeded the model's context length
                if input_ids.shape[1] > getattr(model, "context_length", float("inf")):
                    active_prompts[i] = False

            # Append to input for next iteration
            input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(1)], dim=1)

    # Decode the generated tokens for each prompt
    generated_texts = []
    for tokens in generated_tokens:
        generated_text = tokenizer.decode(tokens)
        generated_texts.append(generated_text)

    return generated_texts


def generate_text_with_controls(
    model: torch.nn.Module,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token: str = "<|endoftext|>",
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """
    Generate text with detailed control and return generation statistics.

    Args:
        model (torch.nn.Module): The trained language model
        prompt (str): Input text prompt
        tokenizer: Tokenizer with encode/decode methods
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for softmax scaling
        top_p (float): Top-p sampling threshold
        end_token (str): Token that signals end of generation
        device (str): Device to run the model on
        verbose (bool): Whether to print generation progress

    Returns:
        dict: Dictionary containing generated text and statistics
    """
    model.eval()

    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        raise ValueError("Failed to encode prompt")

    # Convert to tensor and move to device
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Track generated tokens and statistics
    generated_tokens = []
    token_probabilities = []
    generation_steps = 0

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Generating up to {max_new_tokens} tokens...")

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass through the model
            logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

            # Get the last token's logits (next token prediction)
            next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

            # Apply temperature scaling
            if temperature != 1.0:
                probs = softmax_with_temperature(next_token_logits, temperature)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)

            # Apply top-p sampling
            if top_p < 1.0:
                probs = top_p_sampling(probs, top_p)

            # Sample from the probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # Get the probability of the chosen token
            token_prob = probs[next_token_id].item()
            token_probabilities.append(token_prob)

            # Check if we hit the end token
            decoded_token = tokenizer.decode([next_token_id])
            if decoded_token == end_token:
                if verbose:
                    print(f"Step {step + 1}: Generated end token")
                break

            # Add the generated token to our sequence
            generated_tokens.append(next_token_id)

            if verbose:
                print(
                    f"Step {step + 1}: Generated '{decoded_token}' (prob: {token_prob:.4f})"
                )

            # Append to input for next iteration
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token_id]], device=device)], dim=1
            )

            # Check if we've exceeded the model's context length
            if input_ids.shape[1] > getattr(model, "context_length", float("inf")):
                if verbose:
                    print(f"Reached maximum context length at step {step + 1}")
                break

            generation_steps = step + 1

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens)

    # Calculate statistics
    avg_probability = (
        sum(token_probabilities) / len(token_probabilities)
        if token_probabilities
        else 0.0
    )
    min_probability = min(token_probabilities) if token_probabilities else 0.0
    max_probability = max(token_probabilities) if token_probabilities else 0.0

    if verbose:
        print(f"\nGeneration completed in {generation_steps} steps")
        print(f"Generated text: {generated_text}")
        print(f"Average token probability: {avg_probability:.4f}")
        print(
            f"Token probability range: [{min_probability:.4f}, {max_probability:.4f}]"
        )

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": prompt + generated_text,
        "generated_tokens": generated_tokens,
        "token_probabilities": token_probabilities,
        "generation_steps": generation_steps,
        "avg_probability": avg_probability,
        "min_probability": min_probability,
        "max_probability": max_probability,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
