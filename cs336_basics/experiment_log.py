"""
Experiment logging infrastructure using Weights & Biases (wandb).

This module provides comprehensive experiment tracking for training and evaluation,
including loss curves, hyperparameters, and experiment metadata.
"""

import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import wandb
import torch
import torch.nn as nn


class ExperimentLogger:
    """
    Comprehensive experiment logger using Weights & Biases.

    This class provides methods to track:
    - Training and validation metrics
    - Hyperparameters and configuration
    - Model architecture and parameters
    - System information and resources
    - Experiment metadata and notes
    """

    def __init__(
        self,
        project_name: str = "cs336-basics",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        entity: Optional[str] = None,
        resume: Optional[str] = None,
        log_model: bool = True,
        log_code: bool = True,
        log_artifacts: bool = True,
    ):
        """
        Initialize the experiment logger.

        Args:
            project_name: Name of the wandb project
            experiment_name: Name for this specific experiment
            config: Configuration dictionary to log
            tags: List of tags for organizing experiments
            notes: Additional notes about the experiment
            entity: Wandb entity/username (optional)
            resume: Resume from existing run ID (optional)
            log_model: Whether to log model checkpoints
            log_code: Whether to log code changes
            log_artifacts: Whether to log artifacts
        """
        self.project_name = project_name
        self.experiment_name = (
            experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.entity = entity
        self.resume = resume
        self.log_model = log_model
        self.log_code = log_code
        self.log_artifacts = log_artifacts

        # Initialize wandb run
        self.run = None
        self.start_time = None
        self.step = 0

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize the experiment
        self._init_experiment()

    def _init_experiment(self):
        """Initialize the wandb experiment."""
        try:
            # Initialize wandb with basic parameters (compatible with older versions)
            self.run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config,
                tags=self.tags,
                notes=self.notes,
                entity=self.entity,
                resume=self.resume,
            )

            self.start_time = time.time()
            self.logger.info(
                f"Experiment '{self.experiment_name}' initialized successfully"
            )
            self.logger.info(f"Wandb run ID: {self.run.id}")

            # Log system information
            self._log_system_info()

        except Exception as e:
            self.logger.error(f"Failed to initialize wandb experiment: {e}")
            self.run = None

    def _log_system_info(self):
        """Log system information and resources."""
        if not self.run:
            return

        try:
            # System information
            system_info = {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "platform": os.sys.platform,
                "cpu_count": os.cpu_count(),
                "cwd": os.getcwd(),
            }

            # PyTorch information
            torch_info = {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else None
                ),
                "mps_available": (
                    torch.backends.mps.is_available()
                    if hasattr(torch.backends, "mps")
                    else False
                ),
            }

            if torch.cuda.is_available():
                torch_info.update(
                    {
                        "cuda_device_count": torch.cuda.device_count(),
                        "cuda_device_name": torch.cuda.get_device_name(0),
                        "cuda_memory_allocated": torch.cuda.memory_allocated(0),
                        "cuda_memory_reserved": torch.cuda.memory_reserved(0),
                    }
                )

            # Log system info
            wandb.config.update(
                {
                    "system_info": system_info,
                    "torch_info": torch_info,
                }
            )

        except Exception as e:
            self.logger.warning(f"Failed to log system info: {e}")

    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters for the experiment.

        Args:
            hyperparams: Dictionary of hyperparameters to log
        """
        if not self.run:
            return

        try:
            # Update config with hyperparameters
            self.run.config.update(hyperparams)
            self.logger.info(f"Logged {len(hyperparams)} hyperparameters")

        except Exception as e:
            self.logger.error(f"Failed to log hyperparameters: {e}")

    def log_model_architecture(self, model: nn.Module, input_shape: tuple = None):
        """
        Log model architecture information.

        Args:
            model: PyTorch model to analyze
            input_shape: Input tensor shape for parameter counting
        """
        if not self.run:
            return

        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            non_trainable_params = total_params - trainable_params

            # Model architecture info
            model_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                "model_class": model.__class__.__name__,
                "model_module": model.__class__.__module__,
            }

            # Log model info
            wandb.config.update({"model_architecture": model_info})

            # Log model summary if input_shape is provided
            if input_shape:
                try:
                    # Create dummy input
                    dummy_input = torch.randn(input_shape)

                    # Log model graph (wandb will automatically handle this)
                    wandb.watch(model, log="all", log_freq=100)

                except Exception as e:
                    self.logger.warning(f"Failed to log model graph: {e}")

            self.logger.info(
                f"Logged model architecture: {total_params:,} total parameters"
            )

        except Exception as e:
            self.logger.error(f"Failed to log model architecture: {e}")

    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        **kwargs,
    ):
        """
        Log training step metrics.

        Args:
            loss: Training loss value
            learning_rate: Current learning rate
            gradient_norm: Gradient norm (optional)
            step: Current step number (optional)
            epoch: Current epoch number (optional)
            **kwargs: Additional metrics to log
        """
        if not self.run:
            return

        # Use provided step or increment internal counter
        if step is not None:
            self.step = step
        else:
            self.step += 1

        try:
            # Prepare metrics
            metrics = {
                "train/loss": loss,
                "train/learning_rate": learning_rate,
                "train/step": self.step,
                "train/epoch": epoch,
                "train/wallclock_time": time.time() - self.start_time,
            }

            # Add optional metrics
            if gradient_norm is not None:
                metrics["train/gradient_norm"] = gradient_norm

            # Add additional metrics
            metrics.update({f"train/{k}": v for k, v in kwargs.items()})

            # Log to wandb
            wandb.log(metrics, step=self.step)

        except Exception as e:
            self.logger.error(f"Failed to log training step: {e}")

    def log_validation_step(
        self,
        loss: float,
        perplexity: Optional[float] = None,
        accuracy: Optional[float] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        **kwargs,
    ):
        """
        Log validation step metrics.

        Args:
            loss: Validation loss value
            perplexity: Perplexity (optional)
            accuracy: Accuracy (optional)
            step: Current step number (optional)
            epoch: Current epoch number (optional)
            **kwargs: Additional metrics to log
        """
        if not self.run:
            return

        # Use provided step or current internal step
        current_step = step if step is not None else self.step

        try:
            # Prepare metrics
            metrics = {
                "val/loss": loss,
                "val/step": current_step,
                "val/epoch": epoch,
                "val/wallclock_time": time.time() - self.start_time,
            }

            # Add optional metrics
            if perplexity is not None:
                metrics["val/perplexity"] = perplexity
            if accuracy is not None:
                metrics["val/accuracy"] = accuracy

            # Add additional metrics
            metrics.update({f"val/{k}": v for k, v in kwargs.items()})

            # Log to wandb
            wandb.log(metrics, step=current_step)

        except Exception as e:
            self.logger.error(f"Failed to log validation step: {e}")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Log epoch-level metrics.

        Args:
            epoch: Current epoch number
            train_loss: Average training loss for the epoch
            val_loss: Average validation loss for the epoch (optional)
            train_metrics: Additional training metrics (optional)
            val_metrics: Additional validation metrics (optional)
            **kwargs: Additional metrics to log
        """
        if not self.run:
            return

        try:
            # Prepare metrics
            metrics = {
                "epoch": epoch,
                "epoch/train_loss": train_loss,
                "epoch/wallclock_time": time.time() - self.start_time,
            }

            # Add validation loss
            if val_loss is not None:
                metrics["epoch/val_loss"] = val_loss

            # Add training metrics
            if train_metrics:
                metrics.update(
                    {f"epoch/train_{k}": v for k, v in train_metrics.items()}
                )

            # Add validation metrics
            if val_metrics:
                metrics.update({f"epoch/val_{k}": v for k, v in val_metrics.items()})

            # Add additional metrics
            metrics.update({f"epoch/{k}": v for k, v in kwargs.items()})

            # Log to wandb
            wandb.log(metrics, step=self.step)

        except Exception as e:
            self.logger.error(f"Failed to log epoch: {e}")

    def log_checkpoint(
        self,
        model: nn.Module,
        optimizer,
        epoch: int,
        step: int,
        loss: float,
        filepath: str,
        is_best: bool = False,
        **kwargs,
    ):
        """
        Log model checkpoint.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            step: Current step
            loss: Current loss
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
            **kwargs: Additional checkpoint metadata
        """
        if not self.run or not self.log_model:
            return

        try:
            # Save checkpoint locally
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "is_best": is_best,
                **kwargs,
            }

            torch.save(checkpoint, filepath)

            # Log checkpoint to wandb
            artifact = wandb.Artifact(
                name=f"model-checkpoint-{epoch:03d}",
                type="model",
                description=f"Model checkpoint from epoch {epoch}, step {step}",
                metadata={
                    "epoch": epoch,
                    "step": step,
                    "loss": loss,
                    "is_best": is_best,
                    **kwargs,
                },
            )

            artifact.add_file(filepath)
            wandb.log_artifact(artifact)

            self.logger.info(f"Logged checkpoint: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to log checkpoint: {e}")

    def log_text_generation(
        self,
        prompt: str,
        generated_text: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        step: Optional[int] = None,
        **kwargs,
    ):
        """
        Log text generation results.

        Args:
            prompt: Input prompt
            generated_text: Generated text
            temperature: Temperature used for generation
            top_p: Top-p value used for generation
            max_tokens: Maximum tokens generated
            step: Current step number (optional)
            **kwargs: Additional generation metadata
        """
        if not self.run:
            return

        current_step = step if step is not None else self.step

        try:
            # Log generation parameters
            generation_config = {
                "generation/temperature": temperature,
                "generation/top_p": top_p,
                "generation/max_tokens": max_tokens,
                "generation/step": current_step,
            }

            # Log to wandb
            wandb.log(generation_config, step=current_step)

            # Log text samples (wandb will handle text logging)
            wandb.log(
                {
                    "generation/prompt": prompt,
                    "generation/generated_text": generated_text,
                    "generation/full_text": prompt + generated_text,
                },
                step=current_step,
            )

            # Log additional metadata
            if kwargs:
                additional_metrics = {f"generation/{k}": v for k, v in kwargs.items()}
                wandb.log(additional_metrics, step=current_step)

        except Exception as e:
            self.logger.error(f"Failed to log text generation: {e}")

    def log_experiment_summary(
        self,
        final_train_loss: float,
        final_val_loss: Optional[float] = None,
        best_val_loss: Optional[float] = None,
        total_training_time: Optional[float] = None,
        **kwargs,
    ):
        """
        Log experiment summary and final results.

        Args:
            final_train_loss: Final training loss
            final_val_loss: Final validation loss (optional)
            best_val_loss: Best validation loss achieved (optional)
            total_training_time: Total training time in seconds (optional)
            **kwargs: Additional summary metrics
        """
        if not self.run:
            return

        try:
            # Prepare summary metrics
            summary = {
                "summary/final_train_loss": final_train_loss,
                "summary/total_steps": self.step,
                "summary/total_wallclock_time": time.time() - self.start_time,
            }

            # Add optional metrics
            if final_val_loss is not None:
                summary["summary/final_val_loss"] = final_val_loss
            if best_val_loss is not None:
                summary["summary/best_val_loss"] = best_val_loss
            if total_training_time is not None:
                summary["summary/total_training_time"] = total_training_time

            # Add additional metrics
            summary.update({f"summary/{k}": v for k, v in kwargs.items()})

            # Log to wandb
            wandb.log(summary, step=self.step)

            # Update run summary
            wandb.run.summary.update(summary)

            self.logger.info("Logged experiment summary")

        except Exception as e:
            self.logger.error(f"Failed to log experiment summary: {e}")

    def log_hyperparameter_ablation(
        self,
        ablation_name: str,
        baseline_metrics: Dict[str, float],
        ablated_metrics: Dict[str, float],
        ablation_config: Dict[str, Any],
        step: Optional[int] = None,
    ):
        """
        Log hyperparameter ablation results.

        Args:
            ablation_name: Name of the ablation study
            baseline_metrics: Metrics from baseline configuration
            ablated_metrics: Metrics from ablated configuration
            ablation_config: Configuration changes made
            step: Current step number (optional)
        """
        if not self.run:
            return

        current_step = step if step is not None else self.step

        try:
            # Log ablation configuration
            ablation_info = {
                f"ablation/{ablation_name}/config": ablation_config,
                f"ablation/{ablation_name}/step": current_step,
            }

            # Log baseline metrics
            for metric_name, value in baseline_metrics.items():
                ablation_info[f"ablation/{ablation_name}/baseline_{metric_name}"] = (
                    value
                )

            # Log ablated metrics
            for metric_name, value in ablated_metrics.items():
                ablation_info[f"ablation/{ablation_name}/ablated_{metric_name}"] = value

            # Log to wandb
            wandb.log(ablation_info, step=current_step)

            self.logger.info(f"Logged ablation study: {ablation_name}")

        except Exception as e:
            self.logger.error(f"Failed to log ablation study: {e}")

    def log_custom_metric(
        self, metric_name: str, value: Any, step: Optional[int] = None, **kwargs
    ):
        """
        Log custom metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Current step number (optional)
            **kwargs: Additional metadata
        """
        if not self.run:
            return

        current_step = step if step is not None else self.step

        try:
            # Prepare metric data
            metric_data = {
                f"custom/{metric_name}": value,
                f"custom/{metric_name}_step": current_step,
            }

            # Add additional metadata
            metric_data.update(
                {f"custom/{metric_name}_{k}": v for k, v in kwargs.items()}
            )

            # Log to wandb
            wandb.log(metric_data, step=current_step)

        except Exception as e:
            self.logger.error(f"Failed to log custom metric: {e}")

    def finish(self, exit_code: int = 0):
        """
        Finish the experiment and close wandb run.

        Args:
            exit_code: Exit code for the experiment
        """
        if self.run:
            try:
                # Set exit code
                wandb.run.summary.update({"exit_code": exit_code})

                # Finish the run
                wandb.finish(exit_code=exit_code)

                self.logger.info(
                    f"Experiment '{self.experiment_name}' finished successfully"
                )

            except Exception as e:
                self.logger.error(f"Failed to finish experiment: {e}")

        self.run = None

    def get_run_id(self) -> Optional[str]:
        """Get the current wandb run ID."""
        return self.run.id if self.run else None

    def get_run_url(self) -> Optional[str]:
        """Get the URL to view the current run in wandb."""
        if self.run:
            return f"https://wandb.ai/{self.run.entity}/{self.project_name}/runs/{self.run.id}"
        return None


def log_experiment(
    experiment_name: str,
    experiment_description: str,
    experiment_results: Dict[str, Any],
    project_name: str = "cs336-basics",
    tags: Optional[list] = None,
    entity: Optional[str] = None,
) -> str:
    """
    Convenience function to log a single experiment.

    Args:
        experiment_name: Name of the experiment
        experiment_description: Description of what was tested
        experiment_results: Results and metrics from the experiment
        project_name: Wandb project name
        tags: List of tags
        entity: Wandb entity/username

    Returns:
        Run ID of the logged experiment
    """
    logger = ExperimentLogger(
        project_name=project_name,
        experiment_name=experiment_name,
        config={"description": experiment_description},
        tags=tags,
        entity=entity,
    )

    try:
        # Log experiment results
        for metric_name, value in experiment_results.items():
            logger.log_custom_metric(metric_name, value)

        # Get run ID
        run_id = logger.get_run_id()

        # Finish the experiment
        logger.finish()

        return run_id

    except Exception as e:
        logger.logger.error(f"Failed to log experiment: {e}")
        logger.finish(exit_code=1)
        return None


# Example usage and integration functions
def create_experiment_logger_from_config(
    config: Dict[str, Any],
    experiment_name: Optional[str] = None,
    tags: Optional[list] = None,
) -> ExperimentLogger:
    """
    Create an experiment logger from a configuration dictionary.

    Args:
        config: Configuration dictionary
        experiment_name: Optional experiment name override
        tags: Optional tags override

    Returns:
        Configured ExperimentLogger instance
    """
    # Extract logging configuration
    logging_config = config.get("logging", {})

    # Create logger
    logger = ExperimentLogger(
        project_name=logging_config.get("project_name", "cs336-basics"),
        experiment_name=experiment_name or logging_config.get("experiment_name"),
        config=config,
        tags=tags or logging_config.get("tags", []),
        notes=logging_config.get("notes"),
        entity=logging_config.get("entity"),
        log_model=logging_config.get("log_model", True),
        log_code=logging_config.get("log_code", True),
        log_artifacts=logging_config.get("log_artifacts", True),
    )

    return logger


def log_training_experiment(
    config: Dict[str, Any],
    model: Optional[nn.Module] = None,
    experiment_name: str = None,
    tags: Optional[list] = None,
) -> ExperimentLogger:
    """
    Create and configure an experiment logger for training experiments.

    Args:
        config: Training configuration
        model: Model to train (optional, can be logged later)
        experiment_name: Name for this experiment
        tags: Optional tags for organization

    Returns:
        Configured ExperimentLogger instance
    """
    # Create logger
    logger = create_experiment_logger_from_config(config, experiment_name, tags)

    # Log hyperparameters
    logger.log_hyperparameters(config)

    # Log model architecture if model is provided
    if model is not None:
        input_shape = (config.get("batch_size", 1), config.get("context_length", 512))
        logger.log_model_architecture(model, input_shape)

    return logger
