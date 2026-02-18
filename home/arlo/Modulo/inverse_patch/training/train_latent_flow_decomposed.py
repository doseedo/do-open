"""
Decomposed Flow Training: Direction + Magnitude

Key insight: The model predicts DIRECTION well but not MAGNITUDE.
Solution: Decompose and supervise each explicitly.

Direction: Which way to move in latent space (cosine loss)
Magnitude: How far to move (explicit MSE supervision)

At inference: velocity = direction * magnitude
"""

import sys
if "/home/arlo/Data/ACE-Step" not in sys.path:
    sys.path.insert(0, "/home/arlo/Data/ACE-Step")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional

from inverse_afx.data.precomputed_latent_dataset import PrecomputedLatentDataModule


class MagnitudePredictor(nn.Module):
    """
    Predicts ||dry - wet|| from (wet_latent, effect_types, effect_params).

    The magnitude IS in the wet latent - heavily processed audio looks different
    from lightly processed audio. This network learns to extract that signal.
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']
    NUM_EFFECTS = len(EFFECT_TYPES)

    def __init__(
        self,
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 256,
        max_chain_length: int = 4,
        max_params: int = 20,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.hidden_dim = hidden_dim

        # Effect embedding (same structure as velocity net)
        self.effect_embed = nn.Embedding(
            self.NUM_EFFECTS + 1,
            hidden_dim // 4,
            padding_idx=self.NUM_EFFECTS,
        )

        self.param_encoder = nn.Sequential(
            nn.Linear(max_params, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

        self.effect_combiner = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        self.chain_aggregator = nn.Sequential(
            nn.Linear(hidden_dim // 2 * max_chain_length, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Latent analyzer - extracts features that correlate with magnitude
        # Key: reverb tail, harmonic content, transient changes, etc.
        self.latent_encoder = nn.Sequential(
            # Spatial analysis
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Preserve some spatial info
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.SiLU(),
        )

        # Temporal analysis - captures time-varying features
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(latent_channels * latent_height, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Magnitude predictor head
        # Input: effect_cond (hidden) + latent_spatial (hidden) + latent_temporal (hidden)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Magnitude is always positive
        )

    def encode_effects(
        self,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """Encode effect chain to conditioning vector."""
        B = effect_types.shape[0]

        effect_types_safe = effect_types.clone()
        effect_types_safe[effect_types_safe < 0] = self.NUM_EFFECTS

        type_emb = self.effect_embed(effect_types_safe)
        param_emb = self.param_encoder(effect_params)

        combined = torch.cat([type_emb, param_emb], dim=-1)
        combined = self.effect_combiner(combined)

        combined_flat = combined.view(B, -1)
        effect_cond = self.chain_aggregator(combined_flat)

        return effect_cond

    def forward(
        self,
        wet_latent: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict transformation magnitude.

        Args:
            wet_latent: [B, C, H, T] wet latent
            effect_types: [B, max_chain] effect indices
            effect_params: [B, max_chain, max_params] parameters

        Returns:
            magnitude: [B] predicted ||dry - wet||
        """
        B, C, H, T = wet_latent.shape

        # Effect conditioning
        effect_cond = self.encode_effects(effect_types, effect_params)  # [B, hidden]

        # Spatial features from latent
        latent_spatial = self.latent_encoder(wet_latent)  # [B, hidden]

        # Temporal features from latent
        latent_flat = wet_latent.reshape(B, C * H, T)  # [B, C*H, T]
        latent_temporal = self.temporal_encoder(latent_flat)  # [B, hidden]

        # Combine and predict magnitude
        combined = torch.cat([effect_cond, latent_spatial, latent_temporal], dim=-1)
        magnitude = self.magnitude_head(combined).squeeze(-1)  # [B]

        return magnitude


class DirectionPredictor(nn.Module):
    """
    Predicts direction of transformation (unit velocity).

    Based on EffectConditionedVelocityNet but outputs normalized direction.
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']
    NUM_EFFECTS = len(EFFECT_TYPES)

    def __init__(
        self,
        latent_channels: int = 8,
        latent_height: int = 16,
        hidden_dim: int = 512,
        max_chain_length: int = 4,
        max_params: int = 20,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.hidden_dim = hidden_dim

        # Effect embedding
        self.effect_embed = nn.Embedding(
            self.NUM_EFFECTS + 1,
            hidden_dim // 4,
            padding_idx=self.NUM_EFFECTS,
        )

        self.param_encoder = nn.Sequential(
            nn.Linear(max_params, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

        self.effect_combiner = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        self.chain_aggregator = nn.Sequential(
            nn.Linear(hidden_dim // 2 * max_chain_length, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main direction network
        latent_flat_dim = latent_channels * latent_height
        self.input_proj = nn.Linear(latent_flat_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(4)
        ])

        self.output_proj = nn.Linear(hidden_dim, latent_flat_dim)

    def encode_effects(
        self,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """Encode effect chain."""
        B = effect_types.shape[0]

        effect_types_safe = effect_types.clone()
        effect_types_safe[effect_types_safe < 0] = self.NUM_EFFECTS

        type_emb = self.effect_embed(effect_types_safe)
        param_emb = self.param_encoder(effect_params)

        combined = torch.cat([type_emb, param_emb], dim=-1)
        combined = self.effect_combiner(combined)

        combined_flat = combined.view(B, -1)
        return self.chain_aggregator(combined_flat)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict normalized direction.

        Returns:
            direction: [B, C, H, T] unit vector in latent space
        """
        B, C, H, T = z.shape

        z_flat = z.reshape(B, T, C * H)
        z_proj = self.input_proj(z_flat)

        t_emb = self.time_embed(t.view(-1, 1))
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)

        effect_cond = self.encode_effects(effect_types, effect_params)
        effect_cond = effect_cond.unsqueeze(1).expand(-1, T, -1)

        h = z_proj
        for block in self.blocks:
            h_in = torch.cat([h, t_emb, effect_cond], dim=-1)
            h = h + block(h_in)

        velocity_flat = self.output_proj(h)
        velocity = velocity_flat.reshape(B, C, H, T)

        # Normalize to unit vector
        norm = velocity.reshape(B, -1).norm(dim=1, keepdim=True).view(B, 1, 1, 1) + 1e-8
        direction = velocity / norm

        return direction


class DecomposedFlowSystem(pl.LightningModule):
    """
    Training system with decomposed direction + magnitude.

    Key losses:
    - Direction: Cosine similarity (angle matters)
    - Magnitude: MSE (scale matters, explicit supervision)
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        direction_weight: float = 1.0,
        magnitude_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.direction_net = DirectionPredictor(
            latent_channels=latent_channels,
            hidden_dim=hidden_dim,
        )

        self.magnitude_net = MagnitudePredictor(
            latent_channels=latent_channels,
            hidden_dim=hidden_dim // 2,  # Smaller, simpler task
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight

    def forward(
        self,
        wet_latent: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity = direction * magnitude.

        For inference, use t=0 (start from wet).
        """
        B = wet_latent.shape[0]
        t_zero = torch.zeros(B, device=wet_latent.device)

        # Predict direction (unit vector)
        direction = self.direction_net(wet_latent, t_zero, effect_types, effect_params)

        # Predict magnitude (scalar)
        magnitude = self.magnitude_net(wet_latent, effect_types, effect_params)

        # Combine: velocity = direction * magnitude
        velocity = direction * magnitude.view(-1, 1, 1, 1)

        return velocity, direction, magnitude

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        wet_latent = batch['wet_latent']
        dry_latent = batch['dry_latent']
        effect_types = batch['effect_types']
        effect_params = batch['effect_params']

        B = wet_latent.shape[0]
        device = wet_latent.device

        # Target velocity and its decomposition
        target_velocity = dry_latent - wet_latent
        target_magnitude = target_velocity.reshape(B, -1).norm(dim=1)  # [B]
        target_direction = target_velocity / (target_magnitude.view(-1, 1, 1, 1) + 1e-8)

        # Random timestep for direction training (flow matching)
        t = torch.rand(B, device=device)
        t_expand = t.view(-1, 1, 1, 1)
        z_t = t_expand * dry_latent + (1 - t_expand) * wet_latent

        # Predict direction at z_t
        pred_direction = self.direction_net(z_t, t, effect_types, effect_params)

        # Predict magnitude from wet latent (always at t=0)
        pred_magnitude = self.magnitude_net(wet_latent, effect_types, effect_params)

        # Direction loss: cosine similarity (1 - cos_sim to minimize)
        # Flatten for cosine computation
        pred_dir_flat = pred_direction.reshape(B, -1)
        target_dir_flat = target_direction.reshape(B, -1)
        cos_sim = F.cosine_similarity(pred_dir_flat, target_dir_flat, dim=-1)
        direction_loss = (1 - cos_sim).mean()

        # Magnitude loss: MSE with log scale for stability across magnitudes
        # Use log(1 + x) to handle zero magnitudes and compress range
        log_pred_mag = torch.log1p(pred_magnitude)
        log_target_mag = torch.log1p(target_magnitude)
        magnitude_loss = F.mse_loss(log_pred_mag, log_target_mag)

        # Total loss
        total_loss = (
            self.direction_weight * direction_loss +
            self.magnitude_weight * magnitude_loss
        )

        # Logging
        self.log('train/direction_loss', direction_loss, prog_bar=True)
        self.log('train/magnitude_loss', magnitude_loss, prog_bar=True)
        self.log('train/total_loss', total_loss)
        self.log('train/cos_sim_mean', cos_sim.mean())

        with torch.no_grad():
            # Log magnitude prediction accuracy
            mag_ratio = pred_magnitude / (target_magnitude + 1e-8)
            self.log('train/mag_ratio_mean', mag_ratio.mean())
            self.log('train/mag_ratio_std', mag_ratio.std())

            # Log target magnitude distribution
            self.log('train/target_mag_mean', target_magnitude.mean())
            self.log('train/target_mag_std', target_magnitude.std())
            self.log('train/pred_mag_mean', pred_magnitude.mean())

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        wet_latent = batch['wet_latent']
        dry_latent = batch['dry_latent']
        effect_types = batch['effect_types']
        effect_params = batch['effect_params']

        B = wet_latent.shape[0]
        device = wet_latent.device

        # Targets
        target_velocity = dry_latent - wet_latent
        target_magnitude = target_velocity.reshape(B, -1).norm(dim=1)
        target_direction = target_velocity / (target_magnitude.view(-1, 1, 1, 1) + 1e-8)

        # Predictions (at t=0 for fair eval)
        t_zero = torch.zeros(B, device=device)
        pred_direction = self.direction_net(wet_latent, t_zero, effect_types, effect_params)
        pred_magnitude = self.magnitude_net(wet_latent, effect_types, effect_params)

        # Reconstruct velocity
        pred_velocity = pred_direction * pred_magnitude.view(-1, 1, 1, 1)

        # Losses
        pred_dir_flat = pred_direction.reshape(B, -1)
        target_dir_flat = target_direction.reshape(B, -1)
        cos_sim = F.cosine_similarity(pred_dir_flat, target_dir_flat, dim=-1)
        direction_loss = (1 - cos_sim).mean()

        log_pred_mag = torch.log1p(pred_magnitude)
        log_target_mag = torch.log1p(target_magnitude)
        magnitude_loss = F.mse_loss(log_pred_mag, log_target_mag)

        # Also compute flow reconstruction loss (the actual metric)
        flow_loss = F.mse_loss(pred_velocity, target_velocity)

        self.log('val/direction_loss', direction_loss, prog_bar=True)
        self.log('val/magnitude_loss', magnitude_loss, prog_bar=True)
        self.log('val/flow_loss', flow_loss, prog_bar=True)
        self.log('val/cos_sim_mean', cos_sim.mean())

        # Stratified logging by magnitude
        with torch.no_grad():
            subtle_mask = target_magnitude < 0.1
            strong_mask = target_magnitude >= 0.1

            if subtle_mask.any():
                subtle_mag_error = (pred_magnitude[subtle_mask] - target_magnitude[subtle_mask]).abs().mean()
                self.log('val/subtle_mag_error', subtle_mag_error)

            if strong_mask.any():
                strong_mag_error = (pred_magnitude[strong_mask] - target_magnitude[strong_mask]).abs().mean()
                self.log('val/strong_mag_error', strong_mag_error)

        return flow_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


def create_trainer(
    max_epochs: int = 100,
    devices: int = 1,
    precision: str = "32",
    accumulate_grad_batches: int = 4,
    checkpoint_dir: str = "checkpoints/latent_flow_decomposed",
) -> pl.Trainer:
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="decomposed-{epoch:02d}-{val_flow_loss:.4f}",
            monitor="val/flow_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    logger = TensorBoardLogger("logs/latent_flow_decomposed", name="decomposed")

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=devices,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )


def main():
    parser = argparse.ArgumentParser(description='Train decomposed flow model')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--direction-weight', type=float, default=1.0)
    parser.add_argument('--magnitude-weight', type=float, default=1.0)
    parser.add_argument('--checkpoint-dir', type=str,
                        default='checkpoints/latent_flow_decomposed')
    args = parser.parse_args()

    system = DecomposedFlowSystem(
        learning_rate=args.lr,
        direction_weight=args.direction_weight,
        magnitude_weight=args.magnitude_weight,
    )

    total_params = sum(p.numel() for p in system.parameters())
    dir_params = sum(p.numel() for p in system.direction_net.parameters())
    mag_params = sum(p.numel() for p in system.magnitude_net.parameters())

    print(f"Total params: {total_params:,}")
    print(f"  Direction net: {dir_params:,}")
    print(f"  Magnitude net: {mag_params:,}")
    print(f"Direction weight: {args.direction_weight}")
    print(f"Magnitude weight: {args.magnitude_weight}")

    data_module = PrecomputedLatentDataModule(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = create_trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.fit(system, data_module)


if __name__ == '__main__':
    main()
