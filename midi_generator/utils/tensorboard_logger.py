"""
TensorBoard Logger Utility for Domain Encoder Training

Provides a simple, unified interface for logging training metrics to TensorBoard.

Usage:
    from midi_generator.utils.tensorboard_logger import TensorBoardLogger

    # Initialize logger
    logger = TensorBoardLogger(
        log_dir='output/semantic_discovery/logs',
        experiment_name='harmony_encoder_v2'
    )

    # Log metrics during training
    for epoch in range(num_epochs):
        # Training step
        train_loss = train_step()
        logger.log_scalar('Loss/train', train_loss, epoch)

        # Validation step
        val_loss = validate()
        logger.log_scalar('Loss/val', val_loss, epoch)

        # Multiple metrics at once
        logger.log_scalars({
            'Loss/reconstruction': recon_loss,
            'Loss/sparsity': sparsity_loss,
            'Loss/locality': locality_loss,
            'Learning_Rate': current_lr
        }, epoch)

    # Close logger
    logger.close()

Author: v2.0 Training Pipeline
"""

from pathlib import Path
from typing import Dict, Optional, Union
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ Warning: TensorBoard not available. Install with: pip install tensorboard")


class TensorBoardLogger:
    """
    Unified TensorBoard logger for domain encoder training.

    Features:
    - Simple scalar logging
    - Batch logging of multiple metrics
    - Histogram logging for weights/gradients
    - Automatic experiment naming
    - Graceful fallback if TensorBoard unavailable
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = 'output/semantic_discovery/logs',
        experiment_name: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name of experiment (creates subdirectory)
            enabled: Whether to enable logging (False = dry run)
        """
        self.enabled = enabled and TENSORBOARD_AVAILABLE

        if not TENSORBOARD_AVAILABLE and enabled:
            print("⚠️ TensorBoard not available. Logging disabled.")
            print("   Install with: pip install tensorboard")
            return

        if not enabled:
            print("ℹ️  TensorBoard logging disabled")
            return

        # Setup log directory
        log_dir = Path(log_dir)
        if experiment_name:
            log_dir = log_dir / experiment_name

        log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize writer
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.log_dir = log_dir

        print(f"✅ TensorBoard logging enabled")
        print(f"   Log directory: {log_dir}")
        print(f"   To view: tensorboard --logdir={log_dir.parent}")

    def log_scalar(
        self,
        tag: str,
        value: Union[float, torch.Tensor],
        step: int
    ):
        """
        Log a scalar value.

        Args:
            tag: Metric name (e.g., 'Loss/train', 'Accuracy')
            value: Scalar value to log
            step: Training step/epoch number
        """
        if not self.enabled:
            return

        if isinstance(value, torch.Tensor):
            value = value.item()

        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        metrics: Dict[str, Union[float, torch.Tensor]],
        step: int
    ):
        """
        Log multiple scalar values at once.

        Args:
            metrics: Dictionary of {tag: value} pairs
            step: Training step/epoch number
        """
        if not self.enabled:
            return

        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int
    ):
        """
        Log a histogram of values (useful for weights/gradients).

        Args:
            tag: Histogram name (e.g., 'Weights/encoder_fc1')
            values: Tensor of values
            step: Training step/epoch number
        """
        if not self.enabled:
            return

        self.writer.add_histogram(tag, values, step)

    def log_model_weights(
        self,
        model: torch.nn.Module,
        step: int,
        prefix: str = 'Weights'
    ):
        """
        Log histograms of all model weights.

        Args:
            model: PyTorch model
            step: Training step/epoch number
            prefix: Prefix for histogram tags
        """
        if not self.enabled:
            return

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(
                    f'{prefix}/{name}',
                    param.data,
                    step
                )

    def log_model_gradients(
        self,
        model: torch.nn.Module,
        step: int,
        prefix: str = 'Gradients'
    ):
        """
        Log histograms of all model gradients.

        Args:
            model: PyTorch model
            step: Training step/epoch number
            prefix: Prefix for histogram tags
        """
        if not self.enabled:
            return

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(
                    f'{prefix}/{name}',
                    param.grad.data,
                    step
                )

    def log_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        step: int,
        tag: str = 'Learning_Rate'
    ):
        """
        Log current learning rate from optimizer.

        Args:
            optimizer: PyTorch optimizer
            step: Training step/epoch number
            tag: Tag for learning rate metric
        """
        if not self.enabled:
            return

        lr = optimizer.param_groups[0]['lr']
        self.log_scalar(tag, lr, step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: int = 0
    ):
        """
        Log text (useful for hyperparameters, notes).

        Args:
            tag: Text tag
            text: Text content
            step: Training step/epoch number
        """
        if not self.enabled:
            return

        self.writer.add_text(tag, text, step)

    def log_hparams(
        self,
        hparams: Dict,
        metrics: Optional[Dict] = None
    ):
        """
        Log hyperparameters and optional metrics.

        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional dictionary of metrics to associate
        """
        if not self.enabled:
            return

        if metrics is None:
            metrics = {}

        self.writer.add_hparams(hparams, metrics)

    def flush(self):
        """Flush pending writes to disk"""
        if self.enabled:
            self.writer.flush()

    def close(self):
        """Close the logger"""
        if self.enabled:
            self.writer.close()
            print(f"✅ TensorBoard logger closed")

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()


# Convenience function for quick setup
def create_logger(
    experiment_name: str,
    log_dir: Union[str, Path] = 'output/semantic_discovery/logs',
    enabled: bool = True
) -> TensorBoardLogger:
    """
    Create a TensorBoard logger with sensible defaults.

    Args:
        experiment_name: Name of experiment
        log_dir: Base directory for logs
        enabled: Whether to enable logging

    Returns:
        TensorBoardLogger instance
    """
    return TensorBoardLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        enabled=enabled
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TensorBoard Logger - Example Usage")
    print("="*70)

    # Create logger
    logger = create_logger(
        experiment_name='harmony_encoder_test',
        enabled=True
    )

    # Simulate training
    print("\nSimulating training loop...")
    for epoch in range(10):
        # Simulate losses
        train_loss = 100 / (epoch + 1)  # Decreasing loss
        val_loss = 120 / (epoch + 1)
        recon_loss = 80 / (epoch + 1)
        sparsity_loss = 0.01 * (epoch + 1)

        # Log metrics
        logger.log_scalars({
            'Loss/train': train_loss,
            'Loss/val': val_loss,
            'Loss/reconstruction': recon_loss,
            'Loss/sparsity': sparsity_loss,
            'Learning_Rate': 0.01 * (0.95 ** epoch)
        }, epoch)

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.2f}, val_loss={val_loss:.2f}")

    # Log hyperparameters
    logger.log_hparams({
        'hidden_dim': 1024,
        'learning_rate': 0.01,
        'batch_size': 32,
        'dropout': 0.2
    }, {
        'final_loss': train_loss
    })

    # Close logger
    logger.close()

    print("\n✅ Example completed!")
    print(f"   View logs: tensorboard --logdir={logger.log_dir.parent}")
    print("="*70)
