# Experiment Logging with Weights & Biases

This document explains how to use the comprehensive experiment logging system implemented for the CS336 Basics assignment. The system provides systematic tracking of all experiments, hyperparameters, and results using Weights & Biases (wandb).

## 🎯 **Overview**

The experiment logging system provides:

- **Comprehensive Tracking**: All training metrics, hyperparameters, and system information
- **Wandb Integration**: Professional experiment tracking and visualization
- **Systematic Organization**: Consistent logging across all experiments
- **Reproducible Research**: Complete experiment records for future reference

## 🚀 **Quick Start**

### **1. Install Dependencies**

```bash
# Install wandb
pip install wandb

# Or using uv
uv add wandb
```

### **2. Setup Wandb Account**

```bash
# Login to wandb (first time only)
wandb login

# Or set environment variable
export WANDB_API_KEY="your_api_key_here"
```

### **3. Run Training with Logging**

```bash
# Basic training with wandb
python train_with_wandb.py --config config_with_wandb.yaml

# Training without wandb (fallback)
python train_with_wandb.py --config config_with_wandb.yaml --no-wandb
```

## 🔧 **Configuration**

### **Wandb Configuration in YAML**

```yaml
# Weights & Biases Integration
logging:
  project_name: "cs336-basics"         # Wandb project name
  experiment_name: "transformer-training"  # Experiment name
  tags: ["transformer", "language-model", "cs336"]  # Organization tags
  notes: "Training transformer language model on custom dataset"  # Description
  entity: null                         # Wandb entity/username (null = use default)
  log_model: true                      # Log model checkpoints
  log_code: true                       # Log code changes
  log_artifacts: true                  # Log artifacts
  
  # Logging frequency
  log_training_every: 10               # Log training metrics every N steps
  log_validation_every: 100            # Log validation metrics every N steps
  
  # Custom metrics
  log_gradient_norms: true             # Log gradient norms
  log_learning_rate: true              # Log learning rate changes
  log_memory_usage: true               # Log memory usage
  log_compute_time: true               # Log compute time per step
```

### **Environment Variables**

```bash
# Wandb configuration
export WANDB_PROJECT="cs336-basics"
export WANDB_ENTITY="your_username"
export WANDB_MODE="online"  # or "offline" for local logging

# Disable wandb if needed
export WANDB_DISABLED=true
```

## 📊 **What Gets Logged**

### **Training Metrics**
- **Loss Curves**: Training and validation loss over time
- **Learning Rate**: Current learning rate value
- **Gradient Norms**: L2 norm of gradients
- **Memory Usage**: GPU memory consumption
- **Compute Time**: Time per training step

### **Hyperparameters**
- **Model Architecture**: Layers, dimensions, attention heads
- **Training Parameters**: Batch size, learning rate, optimizer settings
- **Data Configuration**: Dataset paths, vocabulary size, context length
- **System Information**: Hardware, PyTorch version, CUDA info

### **Model Information**
- **Parameter Counts**: Total and trainable parameters
- **Architecture Details**: Model class, module structure
- **Checkpoints**: Model and optimizer state saves
- **Performance Metrics**: Training speed, convergence

### **System Information**
- **Hardware**: CPU, GPU, memory specifications
- **Software**: Python, PyTorch, CUDA versions
- **Environment**: Working directory, platform information

## 🎛️ **Usage Examples**

### **Basic Experiment Logging**

```python
from cs336_basics.experiment_log import ExperimentLogger

# Create logger
logger = ExperimentLogger(
    project_name="cs336-basics",
    experiment_name="my-experiment",
    config={"learning_rate": 1e-4, "batch_size": 32},
    tags=["experiment", "transformer"],
    notes="Testing different learning rates"
)

# Log training step
logger.log_training_step(
    loss=0.5,
    learning_rate=1e-4,
    gradient_norm=1.2,
    step=100,
    epoch=1
)

# Log validation
logger.log_validation_step(
    loss=0.4,
    perplexity=1.5,
    step=100,
    epoch=1
)

# Finish experiment
logger.finish()
```

### **Training Integration**

```python
from cs336_basics.experiment_log import log_training_experiment

# Create training logger
logger = log_training_experiment(
    config=config,
    model=model,
    experiment_name="transformer-training",
    tags=["training", "transformer"]
)

# Training loop
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        # ... training code ...
        
        # Log training metrics
        logger.log_training_step(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]['lr'],
            step=step + epoch * steps_per_epoch,
            epoch=epoch
        )
    
    # Log epoch summary
    logger.log_epoch(
        epoch=epoch,
        train_loss=epoch_loss,
        val_loss=val_loss
    )

# Finish training
logger.finish()
```

### **Hyperparameter Ablation**

```python
# Test different learning rates
learning_rates = [1e-5, 1e-4, 1e-3]

for lr in learning_rates:
    # Update config
    config["learning_rate"] = lr
    
    # Create experiment logger
    logger = ExperimentLogger(
        project_name="cs336-basics",
        experiment_name=f"lr-ablation-{lr}",
        config=config,
        tags=["ablation", "learning-rate"]
    )
    
    # Run training
    # ... training code ...
    
    # Log results
    logger.log_experiment_summary(
        final_train_loss=train_loss,
        final_val_loss=val_loss,
        total_training_time=training_time
    )
    
    logger.finish()
```

## 📈 **Visualization and Analysis**

### **Wandb Dashboard**

Once you run experiments, you can view results in the wandb dashboard:

1. **Go to**: https://wandb.ai/your_username/cs336-basics
2. **View Runs**: All experiments are organized by tags and names
3. **Compare Runs**: Side-by-side comparison of different configurations
4. **Custom Charts**: Create custom visualizations and reports

### **Key Charts Available**

- **Loss Curves**: Training and validation loss over time
- **Learning Rate**: Learning rate schedule visualization
- **Gradient Norms**: Gradient stability monitoring
- **Memory Usage**: Resource utilization tracking
- **Parameter Histograms**: Weight distribution analysis

### **Run Comparison**

```python
import wandb

# Compare multiple runs
api = wandb.Api()
runs = api.runs("your_username/cs336-basics")

# Find best run
best_run = min(runs, key=lambda r: r.summary.get('val_loss', float('inf')))
print(f"Best run: {best_run.name} with val_loss: {best_run.summary['val_loss']}")
```

## 🔍 **Advanced Features**

### **Custom Metrics**

```python
# Log custom metrics
logger.log_custom_metric(
    metric_name="custom_loss",
    value=0.123,
    step=100,
    additional_info="special_metric"
)
```

### **Model Checkpointing**

```python
# Log model checkpoint
logger.log_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=step,
    loss=loss,
    filepath="checkpoint.pt",
    is_best=True,
    additional_metadata={"val_perplexity": 1.5}
)
```

### **Text Generation Logging**

```python
# Log text generation results
logger.log_text_generation(
    prompt="The future of AI",
    generated_text="is bright and promising...",
    temperature=0.8,
    top_p=0.9,
    max_tokens=50,
    step=100
)
```

### **Ablation Studies**

```python
# Log ablation study results
logger.log_hyperparameter_ablation(
    ablation_name="attention_heads",
    baseline_metrics={"loss": 0.5, "perplexity": 1.5},
    ablated_metrics={"loss": 0.6, "perplexity": 1.8},
    ablation_config={"num_heads": 4},
    step=100
)
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Wandb Login Failed**
   ```bash
   # Check API key
   echo $WANDB_API_KEY
   
   # Re-login
   wandb login --relogin
   ```

2. **Offline Mode**
   ```bash
   # Force offline mode
   export WANDB_MODE=offline
   
   # Or disable completely
   export WANDB_DISABLED=true
   ```

3. **Memory Issues**
   ```yaml
   # Reduce logging frequency
   logging:
     log_training_every: 100  # Log less frequently
     log_model: false         # Don't log model checkpoints
   ```

### **Fallback Options**

```python
# Graceful fallback if wandb fails
try:
    logger = ExperimentLogger(...)
except Exception as e:
    print(f"Wandb failed: {e}")
    logger = None

# Use logger if available
if logger:
    logger.log_training_step(...)
else:
    print("Training step logged locally")
```

## 📝 **Best Practices**

### **1. Experiment Naming**

```python
# Use descriptive names
experiment_name = "transformer-12L-768D-lr1e4-bs32"

# Include key parameters
experiment_name = f"transformer-{num_layers}L-{d_model}D-lr{lr}-bs{batch_size}"
```

### **2. Tagging Strategy**

```python
# Use consistent tags for organization
tags = [
    "transformer",           # Component type
    "language-model",        # Application
    "cs336",                # Course/project
    "experiment",           # Experiment type
    "hyperparameter-tuning" # Purpose
]
```

### **3. Configuration Logging**

```python
# Log complete configuration
logger.log_hyperparameters({
    "model": {
        "num_layers": 12,
        "d_model": 768,
        "num_heads": 12
    },
    "training": {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 100
    }
})
```

### **4. Regular Logging**

```python
# Log at regular intervals
if step % log_interval == 0:
    logger.log_training_step(...)

# Log at epoch boundaries
logger.log_epoch(...)
```

## 🔮 **Future Enhancements**

### **Planned Features**

1. **Advanced Visualizations**: Custom charts and dashboards
2. **Hyperparameter Optimization**: Integration with Optuna/Hyperopt
3. **Experiment Templates**: Pre-defined experiment configurations
4. **Collaborative Features**: Team experiment sharing

### **Integration Possibilities**

1. **MLflow**: Alternative experiment tracking
2. **TensorBoard**: Local visualization
3. **Comet ML**: Alternative cloud platform
4. **Custom Backends**: Local database storage

## 📚 **Resources**

### **Documentation**

- **Wandb Docs**: https://docs.wandb.ai/
- **PyTorch Integration**: https://docs.wandb.ai/guides/integrations/pytorch
- **API Reference**: https://docs.wandb.ai/ref/python/public-api

### **Examples**

- **Training Scripts**: `train_with_wandb.py`
- **Configuration**: `config_with_wandb.yaml`
- **Experiment Log**: `EXPERIMENT_LOG.md`

### **Support**

- **Wandb Community**: https://community.wandb.ai/
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive guides and tutorials

## 🎉 **Conclusion**

The experiment logging system provides a professional-grade infrastructure for tracking all your CS336 experiments. Key benefits include:

1. **Systematic Organization**: Consistent logging across all experiments
2. **Professional Visualization**: Beautiful charts and dashboards
3. **Easy Comparison**: Side-by-side analysis of different configurations
4. **Reproducible Research**: Complete experiment records
5. **Collaboration**: Easy sharing and discussion of results

By using this system, you'll be able to:
- Track all your experiments systematically
- Compare different approaches easily
- Share results with instructors and peers
- Build a portfolio of reproducible research
- Learn best practices for ML experimentation

Start logging your experiments today and build a comprehensive understanding of transformer architectures through systematic experimentation!

