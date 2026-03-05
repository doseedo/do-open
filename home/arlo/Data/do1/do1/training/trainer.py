# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
PyTorch Lightning training module for DO1.

Implements the training loop with:
- Rectified flow matching loss
- Task-weighted training
- Per-task logging
- Gradient checkpointing
- Mixed precision training
"""

import math
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from ..models import DO1Transformer2DModel
from ..data import DO1Dataset, do1_collate_fn
from .flow_matching import RectifiedFlowMatching


class DO1Pipeline(LightningModule):
    """
    PyTorch Lightning module for DO1 training.

    Handles:
    - Model initialization
    - Flow matching training step
    - Task-weighted loss computation
    - Learning rate scheduling
    - Per-task metric logging

    Args:
        model_config: Configuration for DO1Transformer2DModel
        training_config: Training hyperparameters
        data_config: Dataset configuration
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        data_config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        # Training config
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.weight_decay = training_config.get('weight_decay', 0.01)
        self.warmup_steps = training_config.get('warmup_steps', 5000)
        self.max_steps = training_config.get('max_steps', 500000)
        self.grad_clip = training_config.get('grad_clip', 1.0)
        self.batch_size = training_config.get('batch_size', 8)
        self.num_workers = training_config.get('num_workers', 8)
        self.cfg_dropout_rate = training_config.get('cfg_dropout', 0.3)

        # Flow matching config
        timestep_dist = training_config.get('timestep_distribution', 'logit_normal')
        logit_mean = training_config.get('logit_mean', 0.0)
        logit_std = training_config.get('logit_std', 1.0)

        # Initialize model
        self.model = DO1Transformer2DModel(**model_config)
        self.model.enable_gradient_checkpointing()

        # Initialize flow matching
        self.flow_matching = RectifiedFlowMatching(
            timestep_distribution=timestep_dist,
            logit_mean=logit_mean,
            logit_std=logit_std,
        )

        # Data config (stored for dataloader)
        self.data_config = data_config

        # Task loss accumulators for logging
        self.task_losses: Dict[str, List[float]] = {}

    def forward(
        self,
        x_noisy: torch.Tensor,
        x_cond: torch.Tensor,
        x_ref: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_noisy: Noisy latent [B, 8, 16, T]
            x_cond: Conditioning latent [B, 8, 16, T]
            x_ref: Reference latent [B, 8, 16, T']
            mask: Task mask [B, 1, 16, T]
            timestep: Flow matching timestep [B]
            attention_mask: Padding mask [B, T]
            ref_mask: Reference padding mask [B, T']

        Returns:
            Predicted velocity field [B, 8, 16, T]
        """
        output = self.model(
            x_noisy=x_noisy,
            x_cond=x_cond,
            x_ref=x_ref,
            mask=mask,
            timestep=timestep,
            attention_mask=attention_mask,
            ref_mask=ref_mask,
            return_dict=True,
        )
        return output.sample

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Collated batch from dataloader
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        x_cond = batch['x_cond']
        x_ref = batch['x_ref']
        z_target = batch['z_target']
        mask = batch['mask']
        attention_mask = batch['attention_mask']
        ref_mask = batch['ref_mask']
        tasks = batch['tasks']
        loss_weights = batch['loss_weights'].to(self.device)

        B = z_target.shape[0]

        # Sample timesteps
        t = self.flow_matching.sample_timesteps(B, self.device)

        # Add noise to get x_noisy and velocity target
        x_noisy, noise, v_target = self.flow_matching.add_noise(z_target, t)

        # Forward pass
        v_pred = self(
            x_noisy=x_noisy,
            x_cond=x_cond,
            x_ref=x_ref,
            mask=mask,
            timestep=t,
            attention_mask=attention_mask,
            ref_mask=ref_mask,
        )

        # Compute loss with masking for padding
        loss = self.flow_matching.compute_loss_masked(
            v_pred=v_pred,
            v_target=v_target,
            attention_mask=attention_mask,
            loss_weight=loss_weights,
        )

        # Log overall metrics
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], sync_dist=True)

        # Log per-task losses
        with torch.no_grad():
            mse_per_sample = F.mse_loss(v_pred, v_target, reduction='none')
            mse_per_sample = mse_per_sample.mean(dim=(1, 2, 3))

            for task_name in set(tasks):
                task_mask = torch.tensor([t == task_name for t in tasks], device=self.device)
                if task_mask.sum() > 0:
                    task_loss = mse_per_sample[task_mask].mean()
                    self.log(f'train/loss_{task_name}', task_loss, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: Collated batch from dataloader
            batch_idx: Batch index

        Returns:
            Dict with validation metrics
        """
        x_cond = batch['x_cond']
        x_ref = batch['x_ref']
        z_target = batch['z_target']
        mask = batch['mask']
        attention_mask = batch['attention_mask']
        ref_mask = batch['ref_mask']
        loss_weights = batch['loss_weights'].to(self.device)

        B = z_target.shape[0]

        # Sample timesteps
        t = self.flow_matching.sample_timesteps(B, self.device)

        # Add noise
        x_noisy, noise, v_target = self.flow_matching.add_noise(z_target, t)

        # Forward pass
        v_pred = self(
            x_noisy=x_noisy,
            x_cond=x_cond,
            x_ref=x_ref,
            mask=mask,
            timestep=t,
            attention_mask=attention_mask,
            ref_mask=ref_mask,
        )

        # Compute loss
        loss = self.flow_matching.compute_loss_masked(
            v_pred=v_pred,
            v_target=v_target,
            attention_mask=attention_mask,
            loss_weight=loss_weights,
        )

        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

        return {'val_loss': loss}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        # Linear warmup + cosine decay
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                # Linear warmup
                return step / max(1, self.warmup_steps)
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                progress = min(1.0, progress)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = DO1Dataset(
            latents_dir=self.data_config['latents_dir'],
            fx_pairs_dir=self.data_config.get('fx_pairs_dir'),
            vst_synths_dir=self.data_config.get('vst_synths_dir'),
            labels_path=self.data_config.get('labels_path'),
            cfg_dropout_rate=self.cfg_dropout_rate,
            max_time_frames=self.data_config.get('max_time_frames', 4096),
            samples_per_epoch=self.data_config.get('samples_per_epoch', 100000),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=do1_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        val_latents_dir = self.data_config.get('val_latents_dir')
        if val_latents_dir is None:
            return None

        dataset = DO1Dataset(
            latents_dir=val_latents_dir,
            fx_pairs_dir=self.data_config.get('fx_pairs_dir'),
            vst_synths_dir=self.data_config.get('vst_synths_dir'),
            labels_path=self.data_config.get('labels_path'),
            cfg_dropout_rate=0.0,  # No dropout during validation
            max_time_frames=self.data_config.get('max_time_frames', 4096),
            samples_per_epoch=self.data_config.get('val_samples', 1000),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=do1_collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def on_train_start(self):
        """Log model info at training start."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.print(f"Total parameters: {total_params:,}")
        self.print(f"Trainable parameters: {trainable_params:,}")
        self.print(f"Model size: {total_params * 4 / 1e9:.2f} GB (fp32)")
