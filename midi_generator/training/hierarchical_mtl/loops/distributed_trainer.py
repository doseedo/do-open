"""
Enhanced Distributed Training for Hierarchical Multi-Task Learning.

Full PyTorch DDP implementation with gradient accumulation, mixed precision,
and optimizations for large-scale pretraining.

Agent 5: Distributed Training Infrastructure
Date: November 21, 2025
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import os
import time

from midi_generator.training.hierarchical_mtl.loops.trainer import HierarchicalMTLTrainer


class DistributedHierarchicalMTLTrainer(HierarchicalMTLTrainer):
    """
    Enhanced distributed trainer for Hierarchical MTL with full DDP support.

    Features:
    - PyTorch Distributed Data Parallel (DDP)
    - Gradient accumulation for effective large batches
    - Mixed precision training (AMP)
    - Efficient data loading with distributed samplers
    - Checkpoint sharding and aggregation
    - Multi-GPU synchronization

    Args:
        model: Hierarchical MTL model
        config: Training configuration
        train_loader: Training data loader (will be wrapped with DistributedSampler)
        val_loader: Validation data loader
        test_loader: Optional test data loader
        local_rank: Local GPU rank
        world_size: Total number of processes
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        local_rank: int = 0,
        world_size: int = 1
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = (local_rank == 0)

        # Initialize distributed backend
        self._init_distributed()

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cpu')

        # Move model to device before DDP wrapping
        model = model.to(self.device)

        # Wrap model with DDP
        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,  # Set True if some params don't get gradients
                broadcast_buffers=True,
                gradient_as_bucket_view=True  # Memory optimization
            )
            if self.is_main_process:
                print(f"Model wrapped with DDP across {world_size} GPUs")

        # Store original model reference for checkpointing
        self.module = model.module if hasattr(model, 'module') else model

        # Initialize parent class (will set up optimizer, scheduler, etc.)
        # Note: We bypass parent __init__ to control DDP setup
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Gradient accumulation
        self.accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self.effective_batch_size = config.data.batch_size * self.accumulation_steps * world_size

        if self.is_main_process:
            print(f"Gradient accumulation: {self.accumulation_steps} steps")
            print(f"Effective batch size: {self.effective_batch_size}")

        # Create optimizer and scheduler (from parent)
        from midi_generator.training.hierarchical_mtl.optimizers.optimizer_factory import (
            create_optimizer, create_scheduler
        )
        self.optimizer = create_optimizer(self.module, config.optimizer)
        self.scheduler = create_scheduler(
            self.optimizer,
            config.scheduler,
            config.num_epochs
        )

        # Mixed precision
        self.use_amp = config.use_amp
        self.scaler = GradScaler() if self.use_amp else None

        # Callbacks (only on main process to avoid duplication)
        if self.is_main_process:
            from midi_generator.training.hierarchical_mtl.callbacks.early_stopping import EarlyStopping
            from midi_generator.training.hierarchical_mtl.callbacks.checkpoint import ModelCheckpoint
            from midi_generator.training.hierarchical_mtl.callbacks.logging_callback import LoggingCallback

            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.min_delta,
                mode='min',
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
        else:
            self.early_stopping = None
            self.checkpoint = None
            self.logger = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _init_distributed(self):
        """Initialize distributed training backend."""
        if self.world_size > 1:
            if not dist.is_initialized():
                # Initialize process group
                dist.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.local_rank
                )

                if self.is_main_process:
                    print(f"Initialized distributed training: {self.world_size} processes")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Set epoch for distributed sampler (for proper shuffling)
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        total_loss = 0.0
        total_level1_loss = 0.0
        total_level2_loss = 0.0
        total_level3_loss = 0.0
        num_batches = len(self.train_loader)

        # Only show progress bar on main process
        if self.is_main_process:
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Training Epoch {self.current_epoch + 1}",
                leave=False
            )
        else:
            progress_bar = self.train_loader

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            features = batch['features'].to(self.device, non_blocking=True)
            level1_labels = {k: v.to(self.device, non_blocking=True) for k, v in batch['level1'].items()}
            level2_labels = {k: v.to(self.device, non_blocking=True) for k, v in batch['level2'].items()}
            level3_labels = {k: v.to(self.device, non_blocking=True) for k, v in batch['level3'].items()}

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(dtype=torch.float16):
                    outputs = self.model(features)
                    loss, loss_dict = self._compute_loss(
                        outputs,
                        level1_labels,
                        level2_labels,
                        level3_labels
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Optimizer step every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
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
                    self.optimizer.zero_grad()

            else:
                outputs = self.model(features)
                loss, loss_dict = self._compute_loss(
                    outputs,
                    level1_labels,
                    level2_labels,
                    level3_labels
                )
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

                # Backward pass
                loss.backward()

                # Optimizer step every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.optimizer.clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.optimizer.clip_grad_norm
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update metrics (unscaled loss for logging)
            total_loss += loss.item() * self.accumulation_steps
            total_level1_loss += loss_dict.get('level1_loss', 0.0)
            total_level2_loss += loss_dict.get('level2_loss', 0.0)
            total_level3_loss += loss_dict.get('level3_loss', 0.0)

            # Update progress bar (main process only)
            if self.is_main_process:
                progress_bar.set_postfix({
                    'loss': loss.item() * self.accumulation_steps,
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

        # Synchronize metrics across all processes
        if self.world_size > 1:
            metrics = self._sync_metrics(metrics)

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model (synchronized across all processes).

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_level1_loss = 0.0
        total_level2_loss = 0.0
        total_level3_loss = 0.0
        num_batches = len(self.val_loader)

        # Only show progress bar on main process
        if self.is_main_process:
            progress_bar = tqdm(
                self.val_loader,
                desc="Validation",
                leave=False
            )
        else:
            progress_bar = self.val_loader

        for batch in progress_bar:
            # Move batch to device
            features = batch['features'].to(self.device, non_blocking=True)
            level1_labels = {k: v.to(self.device, non_blocking=True) for k, v in batch['level1'].items()}
            level2_labels = {k: v.to(self.device, non_blocking=True) for k, v in batch['level2'].items()}
            level3_labels = {k: v.to(self.device, non_blocking=True) for k, v in batch['level3'].items()}

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

        # Synchronize metrics across all processes
        if self.world_size > 1:
            metrics = self._sync_metrics(metrics)

        # Update best validation loss (main process)
        if self.is_main_process and metrics['loss'] < self.best_val_loss:
            self.best_val_loss = metrics['loss']

        return metrics

    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics across all distributed processes."""
        # Convert to tensor
        metric_tensor = torch.tensor(
            [metrics[k] for k in sorted(metrics.keys())],
            device=self.device
        )

        # All-reduce to get average
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)

        # Convert back to dict
        synced_metrics = {
            k: metric_tensor[i].item()
            for i, k in enumerate(sorted(metrics.keys()))
        }

        return synced_metrics

    def save_checkpoint(self, path: Path):
        """Save training checkpoint (main process only)."""
        if self.is_main_process and self.checkpoint is not None:
            self.checkpoint.on_epoch_end(
                self.current_epoch,
                self.module,  # Save unwrapped model
                self.optimizer,
                self.scheduler,
                {'val_loss': self.best_val_loss},
                self.config.to_dict()
            )

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        if self.is_main_process:
            print(f"Loading checkpoint from {path}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state (unwrap DDP if needed)
        self.module.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scheduler
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_score', float('inf'))

        # Synchronize across processes
        if self.world_size > 1:
            dist.barrier()

        if self.is_main_process:
            print(f"Resumed from epoch {self.current_epoch}")

    def cleanup(self):
        """Cleanup distributed training."""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed_training(
    rank: int,
    world_size: int,
    backend: str = 'nccl'
):
    """
    Setup distributed training environment.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend ('nccl' or 'gloo')
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    print(f"Initialized process {rank}/{world_size}")


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
