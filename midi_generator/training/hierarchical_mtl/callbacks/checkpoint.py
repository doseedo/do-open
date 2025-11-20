"""
Model Checkpointing Callback for Hierarchical MTL Training.

Author: Agent 06
Date: November 20, 2025
"""

import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import shutil


class ModelCheckpoint:
    """
    Model checkpointing callback to save model during training.

    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor: Quantity to monitor (e.g., 'val_loss')
        mode: One of 'min', 'max'. Determines whether to minimize or maximize monitored quantity
        save_best_only: If True, only saves the best model based on monitored quantity
        save_every_n_epochs: Save checkpoint every N epochs (in addition to best)
        keep_n_checkpoints: Maximum number of checkpoints to keep (older ones are deleted)
        verbose: If True, print messages when saving checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_every_n_epochs: int = 5,
        keep_n_checkpoints: int = 3,
        verbose: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_n_checkpoints = keep_n_checkpoints
        self.verbose = verbose

        if mode == 'min':
            self.best_score = float('inf')
            self.is_better = lambda new, old: new < old
        elif mode == 'max':
            self.best_score = float('-inf')
            self.is_better = lambda new, old: new > old
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        # Track saved checkpoints
        self.saved_checkpoints = []

    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the end of each epoch to potentially save checkpoint.

        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: LR scheduler state to save
            metrics: Dictionary of metrics for this epoch
            config: Training configuration to save with checkpoint
        """
        current_score = metrics.get(self.monitor, None)

        if current_score is None:
            if self.verbose:
                print(f"Warning: Monitor metric '{self.monitor}' not found in metrics")
            return

        # Determine if we should save
        save_best = False
        save_periodic = False

        # Check if this is the best model
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            save_best = True

        # Check if we should save periodic checkpoint
        if (epoch + 1) % self.save_every_n_epochs == 0:
            save_periodic = True

        # Save checkpoint if needed
        if save_best or (save_periodic and not self.save_best_only):
            checkpoint_type = "best" if save_best else "periodic"
            self._save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                config=config,
                is_best=save_best
            )

            if self.verbose:
                print(f"Saved {checkpoint_type} checkpoint at epoch {epoch} "
                      f"({self.monitor}={current_score:.6f})")

    def _save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]],
        is_best: bool
    ):
        """Save a checkpoint to disk."""
        # Create checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score,
            'timestamp': datetime.now().isoformat()
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if config is not None:
            checkpoint['config'] = config

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, checkpoint_path)

        # Track saved checkpoint
        if not is_best:
            self.saved_checkpoints.append({
                'path': checkpoint_path,
                'epoch': epoch,
                'score': metrics.get(self.monitor, 0)
            })

            # Remove old checkpoints if we have too many
            if len(self.saved_checkpoints) > self.keep_n_checkpoints:
                # Sort by score and keep the best N
                self.saved_checkpoints.sort(
                    key=lambda x: x['score'],
                    reverse=(self.mode == 'max')
                )

                # Remove excess checkpoints
                for ckpt in self.saved_checkpoints[self.keep_n_checkpoints:]:
                    if ckpt['path'].exists():
                        ckpt['path'].unlink()

                self.saved_checkpoints = self.saved_checkpoints[:self.keep_n_checkpoints]

        # Save metadata
        self._save_metadata(epoch, metrics)

    def _save_metadata(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint metadata to JSON."""
        metadata_path = self.checkpoint_dir / 'checkpoint_metadata.json'

        metadata = {
            'last_epoch': epoch,
            'best_score': float(self.best_score),
            'monitor': self.monitor,
            'mode': self.mode,
            'latest_metrics': {k: float(v) for k, v in metrics.items()},
            'saved_checkpoints': [
                {
                    'path': str(ckpt['path']),
                    'epoch': ckpt['epoch'],
                    'score': float(ckpt['score'])
                }
                for ckpt in self.saved_checkpoints
            ]
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_checkpoint(
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint to

        Returns:
            Dictionary with checkpoint information (epoch, metrics, etc.)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

        return checkpoint

    def get_best_checkpoint_path(self) -> Path:
        """Get path to best model checkpoint."""
        return self.checkpoint_dir / 'best_model.pt'

    def __repr__(self) -> str:
        return (f"ModelCheckpoint(checkpoint_dir='{self.checkpoint_dir}', "
                f"monitor='{self.monitor}', mode='{self.mode}', "
                f"save_best_only={self.save_best_only})")
