# Training Script for Transformer Language Model

This training script integrates all the components we've implemented in the CS336 Basics assignment:

## Features

### 🚀 **Complete Integration**
- **BPE Tokenizer**: Handles text tokenization with support for special tokens
- **Transformer Language Model**: Full transformer architecture with RoPE positional encoding
- **AdamW Optimizer**: Modern optimizer with decoupled weight decay
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Gradient Clipping**: Prevents exploding gradients
- **Memory-Efficient Data Loading**: Uses `np.memmap` for large datasets
- **Checkpointing**: Save/resume training state
- **Comprehensive Logging**: Console and file logging

### ⚙️ **Configurable Hyperparameters**
- Model architecture (layers, dimensions, heads)
- Training parameters (batch size, learning rate, epochs)
- Learning rate schedule (warmup, decay)
- Optimization settings (betas, weight decay, gradient clipping)
- Data paths and checkpointing intervals

### 💾 **Memory Efficiency**
- **Memory Mapping**: Loads large datasets without loading everything into RAM
- **Batch Processing**: Efficient data loading with configurable batch sizes
- **Gradient Accumulation**: Support for large effective batch sizes

### 🔄 **Training Features**
- **Resumable Training**: Continue from any checkpoint
- **Validation**: Optional validation during training
- **Best Model Saving**: Automatically save the best model based on validation loss
- **Progress Tracking**: Detailed logging of training progress

## Installation

1. **Install Dependencies**:
   ```bash
   pip install torch numpy pyyaml
   ```

2. **Ensure all components are implemented**:
   - `cs336_basics/tokenizer.py` - BPE Tokenizer
   - `cs336_basics/transformer_lm.py` - Transformer Language Model
   - `cs336_basics/adamw.py` - AdamW Optimizer
   - `cs336_basics/utils.py` - Utility functions

## Usage

### Basic Training

```bash
# Train with default configuration
python train.py --config config_simple.yaml

# Train with custom device
python train.py --config config_simple.yaml --device cuda

# Resume from checkpoint
python train.py --config config_simple.yaml --resume checkpoints/checkpoint_epoch_50.pt
```

### Configuration

The training script uses YAML configuration files. See `config_simple.yaml` for a complete example:

```yaml
# Model Architecture
vocab_size: 50257          # Vocabulary size
context_length: 512        # Maximum sequence length
d_model: 768              # Model dimension
num_layers: 12            # Number of transformer layers
num_heads: 12             # Number of attention heads
d_ff: 3072               # Feed-forward dimension
rope_theta: 10000.0      # RoPE parameter

# Training Parameters
num_epochs: 100           # Total epochs
batch_size: 32            # Batch size
learning_rate: 1e-4       # Initial learning rate
grad_clip_norm: 1.0      # Gradient clipping

# Learning Rate Schedule
use_lr_schedule: true     # Enable LR scheduling
max_learning_rate: 1e-3   # Max LR during warmup
min_learning_rate: 1e-5   # Min LR after decay
warmup_iters: 1000        # Warmup iterations
cosine_cycle_iters: 10000 # Total schedule iterations

# Data Paths
train_data_path: "data/train.txt"
val_data_path: "data/val.txt"

# Checkpointing
checkpoint_dir: "checkpoints"
checkpoint_interval: 10

# Logging
log_level: "INFO"
log_file: "training.log"
log_interval: 100
```

### Data Format

The script supports multiple data formats:

1. **Pre-tokenized Data** (`.npy` or `.npz`):
   - Loaded directly with memory mapping
   - Most efficient for large datasets

2. **Text Files**:
   - Automatically tokenized using BPE tokenizer
   - Requires `data/gpt2_vocab.json` and `data/gpt2_merges.txt`
   - Falls back to simple character-level tokenization if BPE files not found

### Directory Structure

```
project/
├── train.py                 # Training script
├── config_simple.yaml       # Configuration file
├── data/                    # Data directory
│   ├── train.txt           # Training data
│   ├── val.txt             # Validation data
│   ├── gpt2_vocab.json     # BPE vocabulary (optional)
│   └── gpt2_merges.txt     # BPE merges (optional)
├── checkpoints/             # Checkpoint directory (created automatically)
└── training.log             # Training log (created automatically)
```

## Advanced Features

### Learning Rate Scheduling

The script implements cosine annealing with warmup:

1. **Warmup Phase**: Linear increase from 0 to `max_learning_rate`
2. **Cosine Annealing**: Smooth decay from `max_learning_rate` to `min_learning_rate`
3. **Post-Annealing**: Constant at `min_learning_rate`

### Gradient Clipping

Prevents exploding gradients by clipping the L2 norm of all gradients to `grad_clip_norm`.

### Checkpointing

- **Regular Checkpoints**: Saved every `checkpoint_interval` epochs
- **Best Model**: Automatically saved when validation loss improves
- **Final Model**: Saved at the end of training
- **Resumable**: Training can resume from any checkpoint

### Device Support

- **Auto-detection**: Automatically selects best available device
- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2) GPUs
- **CPU**: Fallback option

## Monitoring Training

### Console Output

```
2024-01-15 10:30:15 - train - INFO - Starting training...
2024-01-15 10:30:15 - train - INFO - Model created with 124,439,808 total parameters
2024-01-15 10:30:16 - train - INFO - Epoch 1/100
2024-01-15 10:30:16 - train - INFO - Epoch 1, Step 0/156, Loss: 10.8234, LR: 0.000100, Time: 0.12s
2024-01-15 10:30:18 - train - INFO - Epoch 1, Step 100/156, Loss: 8.4567, LR: 0.000100, Time: 2.34s
2024-01-15 10:30:20 - train - INFO - Epoch 1 - Train Loss: 7.2345, Val Loss: 7.1234, Val PPL: 1234.56
2024-01-15 10:30:20 - train - INFO - New best model saved: checkpoints/best_model.pt
```

### Log File

All training information is also saved to `training.log` for later analysis.

## Customization

### Adding New Features

The script is modular and easy to extend:

1. **New Optimizers**: Modify `create_optimizer()` function
2. **New Schedulers**: Extend learning rate scheduling logic
3. **New Metrics**: Add evaluation metrics in `evaluate_model()`
4. **Data Augmentation**: Modify `get_batch()` calls

### Hyperparameter Tuning

Use the configuration file to experiment with different settings:

```yaml
# Try different model sizes
d_model: 512              # Smaller model
num_layers: 6             # Fewer layers

# Experiment with learning rates
learning_rate: 5e-5       # Lower learning rate
max_learning_rate: 5e-4   # Lower max LR

# Adjust training schedule
warmup_iters: 500         # Shorter warmup
cosine_cycle_iters: 5000  # Shorter schedule
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `batch_size`
   - Reduce `context_length`
   - Use gradient accumulation

2. **Slow Training**:
   - Check device selection
   - Verify data loading efficiency
   - Monitor GPU utilization

3. **Poor Convergence**:
   - Adjust learning rate schedule
   - Check gradient clipping
   - Verify data quality

### Performance Tips

1. **Use Memory Mapping**: For datasets > 1GB
2. **Optimize Batch Size**: Balance memory usage and training speed
3. **Monitor Logs**: Check for warnings or errors
4. **Regular Checkpoints**: Save progress frequently

## Example Training Runs

### Small Model (Fast Training)

```yaml
# config_small.yaml
vocab_size: 10000
context_length: 256
d_model: 256
num_layers: 4
num_heads: 8
d_ff: 1024
batch_size: 64
num_epochs: 50
```

### Large Model (Production Training)

```yaml
# config_large.yaml
vocab_size: 50257
context_length: 1024
d_model: 1024
num_layers: 24
num_heads: 16
d_ff: 4096
batch_size: 16
num_epochs: 200
grad_clip_norm: 0.5
```

## Integration with External Tools

### Weights & Biases

To integrate with Weights & Biases, add to the training loop:

```python
import wandb

wandb.init(project="transformer-lm", config=config)

# In training loop
wandb.log({
    "train_loss": train_metrics["loss"],
    "val_loss": val_metrics["loss"],
    "learning_rate": current_lr,
    "epoch": epoch
})
```

### TensorBoard

For TensorBoard logging, add:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/transformer_training")

# In training loop
writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
writer.add_scalar("Loss/Validation", val_metrics["loss"], epoch)
```

## Conclusion

This training script provides a complete, production-ready solution for training transformer language models. It integrates all the components from the CS336 Basics assignment and provides extensive customization options for different training scenarios.

For questions or issues, refer to the individual component implementations or modify the script as needed for your specific use case.

