"""
Hierarchical Multi-Task Learning Trainer.

Main training loop for the hierarchical MTL system.

Author: Agent 06
Date: November 20, 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import time

from midi_generator.training.hierarchical_mtl.config.training_config import (
    HierarchicalMTLConfig
)
from midi_generator.training.hierarchical_mtl.callbacks.early_stopping import EarlyStopping
from midi_generator.training.hierarchical_mtl.callbacks.checkpoint import ModelCheckpoint
from midi_generator.training.hierarchical_mtl.callbacks.logging_callback import LoggingCallback
from midi_generator.training.hierarchical_mtl.optimizers.optimizer_factory import (
    create_optimizer,
    create_scheduler
)


class HierarchicalMTLTrainer:
    """
    Trainer for Hierarchical Multi-Task Learning.

    Handles end-to-end training of the hierarchical MTL model for predicting
    50 parameters across 3 levels from MIDI features.

    Args:
        model: Hierarchical MTL model (from Agent 05)
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        device: Device to train on ('cuda', 'cpu', or 'mps')
    """

    def __init__(
        self,
        model: nn.Module,
        config: HierarchicalMTLConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Set device
        if device is None:
            device = config.device
        self.device = torch.device(device)
        self.model.to(self.device)

        print(f"Using device: {self.device}")

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(model, config.optimizer)
        self.scheduler = create_scheduler(
            self.optimizer,
            config.scheduler,
            config.num_epochs
        )

        # Mixed precision training
        self.use_amp = config.use_amp
        self.scaler = GradScaler() if self.use_amp else None

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.min_delta,
            mode='min',  # Minimize validation loss
            verbose=True
        )

        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir,
            monitor='val_loss',
            mode='min',
            save_best_only=config.save_best_only,
            save_every_n_epochs=config.save_every_n_epochs,
            keep_n_checkpoints=config.keep_n_checkpoints
        )

        self.logger = LoggingCallback(
            log_dir=config.log_dir,
            use_wandb=config.use_wandb,
            use_mlflow=config.use_mlflow,
            experiment_name=config.experiment_name,
            run_name=config.run_name,
            log_every_n_steps=config.log_every_n_steps
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Distributed training setup
        self.distributed = config.distributed
        if self.distributed:
            self._setup_distributed()

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dictionary with training results and metrics
        """
        print("\n" + "=" * 80)
        print("STARTING HIERARCHICAL MTL TRAINING")
        print("=" * 80)

        # Log config
        self.logger.on_train_begin(self.config.to_dict())

        # Validate before training if requested
        if self.config.validate_on_start:
            print("\nInitial validation...")
            val_metrics = self.validate()
            print(f"Initial val_loss: {val_metrics['loss']:.4f}")

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'=' * 80}")

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate epoch
            val_metrics = self.validate()

            # Log metrics
            self.logger.on_epoch_end(epoch, train_metrics, val_metrics)

            # Update learning rate
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()

            # Check early stopping
            self.early_stopping.on_epoch_end(
                epoch,
                val_metrics['loss'],
                self.model.state_dict()
            )

            # Save checkpoint
            self.checkpoint.on_epoch_end(
                epoch,
                self.model,
                self.optimizer,
                self.scheduler,
                {'val_loss': val_metrics['loss'], **val_metrics},
                self.config.to_dict()
            )

            # Check if we should stop
            if self.early_stopping.should_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        # Restore best weights if early stopping was used
        if self.early_stopping.restore_best_weights:
            best_weights = self.early_stopping.get_best_weights()
            if best_weights is not None:
                self.model.load_state_dict(best_weights)
                print("\nRestored best model weights")

        # Final test evaluation
        test_metrics = None
        if self.test_loader is not None:
            print("\nRunning final test evaluation...")
            test_metrics = self.test()

        # Log training end
        self.logger.on_train_end(test_metrics)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch,
            'test_metrics': test_metrics
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_level1_loss = 0.0
        total_level2_loss = 0.0
        total_level3_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            features = batch['features'].to(self.device)
            level1_labels = {k: v.to(self.device) for k, v in batch['level1'].items()}
            level2_labels = {k: v.to(self.device) for k, v in batch['level2'].items()}
            level3_labels = {k: v.to(self.device) for k, v in batch['level3'].items()}

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(dtype=torch.float16):
                    outputs = self.model(features)
                    loss, loss_dict = self._compute_loss(
                        outputs,
                        level1_labels,
                        level2_labels,
                        level3_labels
                    )

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.optimizer.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimizer.clip_grad_norm
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(features)
                loss, loss_dict = self._compute_loss(
                    outputs,
                    level1_labels,
                    level2_labels,
                    level3_labels
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.optimizer.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimizer.clip_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_level1_loss += loss_dict.get('level1_loss', 0.0)
            total_level2_loss += loss_dict.get('level2_loss', 0.0)
            total_level3_loss += loss_dict.get('level3_loss', 0.0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })

            self.global_step += 1

        # Compute average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'level1_loss': total_level1_loss / num_batches,
            'level2_loss': total_level2_loss / num_batches,
            'level3_loss': total_level3_loss / num_batches
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_level1_loss = 0.0
        total_level2_loss = 0.0
        total_level3_loss = 0.0
        num_batches = len(self.val_loader)

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False
        )

        for batch in progress_bar:
            # Move batch to device
            features = batch['features'].to(self.device)
            level1_labels = {k: v.to(self.device) for k, v in batch['level1'].items()}
            level2_labels = {k: v.to(self.device) for k, v in batch['level2'].items()}
            level3_labels = {k: v.to(self.device) for k, v in batch['level3'].items()}

            # Forward pass
            if self.use_amp:
                with autocast(dtype=torch.float16):
                    outputs = self.model(features)
                    loss, loss_dict = self._compute_loss(
                        outputs,
                        level1_labels,
                        level2_labels,
                        level3_labels
                    )
            else:
                outputs = self.model(features)
                loss, loss_dict = self._compute_loss(
                    outputs,
                    level1_labels,
                    level2_labels,
                    level3_labels
                )

            # Update metrics
            total_loss += loss.item()
            total_level1_loss += loss_dict.get('level1_loss', 0.0)
            total_level2_loss += loss_dict.get('level2_loss', 0.0)
            total_level3_loss += loss_dict.get('level3_loss', 0.0)

        # Compute average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'level1_loss': total_level1_loss / num_batches,
            'level2_loss': total_level2_loss / num_batches,
            'level3_loss': total_level3_loss / num_batches
        }

        # Update best validation loss
        if metrics['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['loss']

        return metrics

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """
        Test the model on test set.

        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            raise ValueError("Test loader not provided")

        self.model.eval()

        total_loss = 0.0
        total_level1_loss = 0.0
        total_level2_loss = 0.0
        total_level3_loss = 0.0
        num_batches = len(self.test_loader)

        progress_bar = tqdm(
            self.test_loader,
            desc="Testing",
            leave=False
        )

        for batch in progress_bar:
            # Move batch to device
            features = batch['features'].to(self.device)
            level1_labels = {k: v.to(self.device) for k, v in batch['level1'].items()}
            level2_labels = {k: v.to(self.device) for k, v in batch['level2'].items()}
            level3_labels = {k: v.to(self.device) for k, v in batch['level3'].items()}

            # Forward pass
            outputs = self.model(features)
            loss, loss_dict = self._compute_loss(
                outputs,
                level1_labels,
                level2_labels,
                level3_labels
            )

            # Update metrics
            total_loss += loss.item()
            total_level1_loss += loss_dict.get('level1_loss', 0.0)
            total_level2_loss += loss_dict.get('level2_loss', 0.0)
            total_level3_loss += loss_dict.get('level3_loss', 0.0)

        # Compute average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'level1_loss': total_level1_loss / num_batches,
            'level2_loss': total_level2_loss / num_batches,
            'level3_loss': total_level3_loss / num_batches
        }

        return metrics

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        level1_labels: Dict[str, torch.Tensor],
        level2_labels: Dict[str, torch.Tensor],
        level3_labels: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hierarchical multi-task loss.

        This is a placeholder implementation. The actual loss computation
        will depend on the model architecture from Agent 05.

        Args:
            outputs: Model outputs for all levels
            level1_labels: Level 1 ground truth labels
            level2_labels: Level 2 ground truth labels
            level3_labels: Level 3 ground truth labels

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Level 1 loss (Global Context)
        level1_loss = 0.0
        for param_name in level1_labels.keys():
            if param_name in outputs.get('level1', {}):
                pred = outputs['level1'][param_name]
                target = level1_labels[param_name]

                # Use appropriate loss based on parameter type
                if param_name in ['genre.primary', 'time_signature', 'key.tonic',
                                  'key.mode', 'structure.form']:
                    # Categorical - use cross entropy
                    loss = nn.functional.cross_entropy(pred, target)
                else:
                    # Continuous - use MSE
                    loss = nn.functional.mse_loss(pred, target)

                level1_loss += loss

        # Level 2 loss (Universal Dimensions)
        level2_loss = 0.0
        for param_name in level2_labels.keys():
            if param_name in outputs.get('level2', {}):
                pred = outputs['level2'][param_name]
                target = level2_labels[param_name]

                # All level 2 parameters are continuous
                loss = nn.functional.mse_loss(pred, target)
                level2_loss += loss

        # Level 3 loss (Genre-Specific)
        level3_loss = 0.0
        for param_name in level3_labels.keys():
            if param_name in outputs.get('level3', {}):
                pred = outputs['level3'][param_name]
                target = level3_labels[param_name]

                # Skip NaN targets (genre-specific params not applicable)
                if not torch.isnan(target).any():
                    loss = nn.functional.mse_loss(pred, target)
                    level3_loss += loss

        # Weighted combination
        total_loss = (
            self.config.loss.level1_weight * level1_loss +
            self.config.loss.level2_weight * level2_loss +
            self.config.loss.level3_weight * level3_loss
        )

        loss_dict = {
            'level1_loss': level1_loss.item() if isinstance(level1_loss, torch.Tensor) else level1_loss,
            'level2_loss': level2_loss.item() if isinstance(level2_loss, torch.Tensor) else level2_loss,
            'level3_loss': level3_loss.item() if isinstance(level3_loss, torch.Tensor) else level3_loss
        }

        return total_loss, loss_dict

    def _setup_distributed(self):
        """Setup distributed training (DDP)."""
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                init_method='env://'
            )

        # Wrap model in DDP
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.config.local_rank] if self.device.type == 'cuda' else None
        )

        print(f"Initialized distributed training: rank {self.config.local_rank}/{self.config.world_size}")

    def save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        self.checkpoint.on_epoch_end(
            self.current_epoch,
            self.model,
            self.optimizer,
            self.scheduler,
            {'val_loss': self.best_val_loss},
            self.config.to_dict()
        )

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint_data = ModelCheckpoint.load_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            device=str(self.device)
        )

        self.current_epoch = checkpoint_data.get('epoch', 0)
        self.best_val_loss = checkpoint_data.get('best_score', float('inf'))

        print(f"Resumed from epoch {self.current_epoch}")
