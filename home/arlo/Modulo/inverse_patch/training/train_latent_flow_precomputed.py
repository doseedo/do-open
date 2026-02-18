"""
Training with precomputed DCAE latents.

Skips audio encoding for much faster training iteration.
Now with effect conditioning for better inversion quality.
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


class EffectConditionedVelocityNet(nn.Module):
    """
    Velocity network for flow matching with effect conditioning.

    Key insight: The model needs to know WHAT effect was applied and HOW STRONG
    to decide whether/how much to transform.
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
        self.max_chain_length = max_chain_length
        self.max_params = max_params

        # Effect type embedding
        self.effect_embed = nn.Embedding(
            self.NUM_EFFECTS + 1,  # +1 for padding (-1 mapped to NUM_EFFECTS)
            hidden_dim // 4,
            padding_idx=self.NUM_EFFECTS,
        )

        # Effect parameters encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(max_params, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

        # Combine effect type + params for each effect in chain
        self.effect_combiner = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Aggregate across chain (attention-like pooling)
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

        # Main velocity network
        latent_flat_dim = latent_channels * latent_height
        self.input_proj = nn.Linear(latent_flat_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # z + t + effect
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(4)
        ])

        self.output_proj = nn.Linear(hidden_dim, latent_flat_dim)

        # Latent analyzer: extract features from wet latent for gating
        self.latent_analyzer = nn.Sequential(
            nn.Conv2d(latent_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, hidden_dim // 4),
        )

        # Magnitude gate: learns when to apply transformation
        # Now sees BOTH effect params AND wet latent characteristics
        self.magnitude_gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def encode_effects(
        self,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode effect chain to conditioning vector.

        Args:
            effect_types: [B, max_chain] effect type indices (-1 for padding)
            effect_params: [B, max_chain, max_params] normalized parameters

        Returns:
            effect_cond: [B, hidden_dim] effect conditioning vector
        """
        B = effect_types.shape[0]

        # Map -1 (padding) to NUM_EFFECTS (padding_idx)
        effect_types_safe = effect_types.clone()
        effect_types_safe[effect_types_safe < 0] = self.NUM_EFFECTS

        # Embed effect types: [B, max_chain, hidden//4]
        type_emb = self.effect_embed(effect_types_safe)

        # Encode parameters: [B, max_chain, hidden//4]
        param_emb = self.param_encoder(effect_params)

        # Combine type + params: [B, max_chain, hidden//2]
        combined = torch.cat([type_emb, param_emb], dim=-1)
        combined = self.effect_combiner(combined)

        # Flatten and aggregate: [B, hidden]
        combined_flat = combined.view(B, -1)
        effect_cond = self.chain_aggregator(combined_flat)

        return effect_cond

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field.

        Args:
            z: [B, C, H, T] noisy latent
            t: [B] timestep in [0, 1]
            effect_types: [B, max_chain] effect type indices
            effect_params: [B, max_chain, max_params] effect parameters

        Returns:
            velocity: [B, C, H, T] predicted velocity
        """
        B, C, H, T = z.shape

        # Flatten spatial dims, keep time: [B, T, C*H]
        z_flat = z.reshape(B, T, C * H)

        # Project latent: [B, T, hidden]
        z_proj = self.input_proj(z_flat)

        # Time embedding: [B, hidden] -> [B, T, hidden]
        t_emb = self.time_embed(t.view(-1, 1))
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)

        # Effect conditioning: [B, hidden] -> [B, T, hidden]
        effect_cond = self.encode_effects(effect_types, effect_params)
        effect_cond = effect_cond.unsqueeze(1).expand(-1, T, -1)

        # Process through blocks with residual connections
        h = z_proj
        for block in self.blocks:
            h_in = torch.cat([h, t_emb, effect_cond], dim=-1)
            h = h + block(h_in)

        # Output velocity: [B, T, C*H]
        velocity_flat = self.output_proj(h)

        # Analyze wet latent for gating (z contains info about input)
        latent_stats = self.latent_analyzer(z)  # [B, hidden//4]

        # Compute magnitude gate from effect conditioning + latent stats
        gate_input = torch.cat([effect_cond[:, 0, :], latent_stats], dim=-1)
        gate = self.magnitude_gate(gate_input)  # [B, 1]

        # Scale velocity by gate (learns when NOT to transform)
        velocity_flat = velocity_flat * gate.unsqueeze(1)

        # Reshape: [B, C, H, T]
        velocity = velocity_flat.reshape(B, C, H, T)

        return velocity, gate.squeeze(-1)  # Also return gate for supervision


class SimpleVelocityNet(nn.Module):
    """Simple velocity network for flow matching (no conditioning, for backwards compat)."""

    def __init__(self, latent_channels: int = 8, hidden_dim: int = 256):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main network - operates on flattened latent
        self.net = nn.Sequential(
            nn.Linear(latent_channels * 16 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_channels * 16),
        )

        self.latent_channels = latent_channels

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, T = z.shape
        z_flat = z.reshape(B, T, C * H)
        t_emb = self.time_embed(t.view(-1, 1))
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([z_flat, t_emb], dim=-1)
        out = self.net(x)
        return out.reshape(B, C, H, T)


class PrecomputedLatentFlowSystem(pl.LightningModule):
    """Training system for precomputed latents with effect conditioning."""

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        use_effect_conditioning: bool = True,
        max_chain_length: int = 4,
        max_params: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.use_effect_conditioning = use_effect_conditioning

        if use_effect_conditioning:
            self.velocity_net = EffectConditionedVelocityNet(
                latent_channels=latent_channels,
                hidden_dim=hidden_dim,
                max_chain_length=max_chain_length,
                max_params=max_params,
            )
        else:
            self.velocity_net = SimpleVelocityNet(
                latent_channels=latent_channels,
                hidden_dim=hidden_dim,
            )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        effect_types: Optional[torch.Tensor] = None,
        effect_params: Optional[torch.Tensor] = None,
        return_gate: bool = False,
    ) -> torch.Tensor:
        if self.use_effect_conditioning and effect_types is not None:
            velocity, gate = self.velocity_net(z, t, effect_types, effect_params)
            if return_gate:
                return velocity, gate
            return velocity
        else:
            return self.velocity_net(z, t)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        wet_latent = batch['wet_latent']  # [B, C, H, T]
        dry_latent = batch['dry_latent']  # [B, C, H, T]

        B = wet_latent.shape[0]
        device = wet_latent.device

        # Random timestep
        t = torch.rand(B, device=device)

        # Interpolate between wet and dry
        t_expand = t.view(-1, 1, 1, 1)
        z_t = t_expand * dry_latent + (1 - t_expand) * wet_latent

        # Target velocity (constant field)
        target_velocity = dry_latent - wet_latent

        # Predict velocity
        if self.use_effect_conditioning and 'effect_types' in batch:
            effect_types = batch['effect_types']
            effect_params = batch['effect_params']
            pred_velocity, _ = self.velocity_net(z_t, t, effect_types, effect_params)
        else:
            pred_velocity = self.velocity_net(z_t, t)

        # MAGNITUDE-WEIGHTED LOSS: Makes subtle effects matter equally
        # Without this, MSE is biased toward large targets (subtle effects ignored)
        target_mag = (target_velocity ** 2).mean(dim=(1, 2, 3)).sqrt() + 1e-6  # [B]
        weight = 1.0 / (target_mag + 0.1)  # Inverse magnitude weighting
        weight = weight / weight.mean()  # Normalize so mean weight = 1

        # Weighted MSE loss
        diff = (pred_velocity - target_velocity) ** 2
        loss = (diff * weight.view(-1, 1, 1, 1)).mean()

        self.log('train/flow_loss', loss, prog_bar=True)

        # Log metrics
        with torch.no_grad():
            target_magnitude = target_mag.mean()
            pred_magnitude = (pred_velocity ** 2).mean(dim=(1, 2, 3)).sqrt().mean()
            self.log('train/target_mag', target_magnitude)
            self.log('train/pred_mag', pred_magnitude)
            self.log('train/weight_mean', weight.mean())
            self.log('train/weight_std', weight.std())

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        wet_latent = batch['wet_latent']
        dry_latent = batch['dry_latent']

        B = wet_latent.shape[0]
        device = wet_latent.device

        t = torch.rand(B, device=device)
        t_expand = t.view(-1, 1, 1, 1)
        z_t = t_expand * dry_latent + (1 - t_expand) * wet_latent
        target_velocity = dry_latent - wet_latent

        if self.use_effect_conditioning and 'effect_types' in batch:
            effect_types = batch['effect_types']
            effect_params = batch['effect_params']
            pred_velocity, _ = self.velocity_net(z_t, t, effect_types, effect_params)
        else:
            pred_velocity = self.velocity_net(z_t, t)

        # Same magnitude-weighted loss as training
        target_mag = (target_velocity ** 2).mean(dim=(1, 2, 3)).sqrt() + 1e-6
        weight = 1.0 / (target_mag + 0.1)
        weight = weight / weight.mean()

        diff = (pred_velocity - target_velocity) ** 2
        loss = (diff * weight.view(-1, 1, 1, 1)).mean()

        # Also log unweighted for comparison
        unweighted_loss = F.mse_loss(pred_velocity, target_velocity)

        self.log('val/flow_loss', loss, prog_bar=True)
        self.log('val/flow_loss_unweighted', unweighted_loss)
        return loss

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
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def create_trainer(
    max_epochs: int = 50,
    devices: int = 1,
    precision: str = "32",
    accumulate_grad_batches: int = 4,
    log_dir: str = "logs/latent_flow_precomputed",
    checkpoint_dir: str = "checkpoints/latent_flow_precomputed",
) -> pl.Trainer:
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="latent_flow-{epoch:02d}-{val_flow_loss:.4f}",
            monitor="val/flow_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    logger = TensorBoardLogger(log_dir, name="precomputed")

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
    parser = argparse.ArgumentParser(description='Train latent flow with precomputed latents')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to latent manifest JSON (with effect info)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--precision', type=str, default='32',
                        choices=['32', '16-mixed', 'bf16-mixed'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--no-effect-conditioning', action='store_true',
                        help='Disable effect conditioning (for comparison)')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='checkpoints/latent_flow_conditioned',
                        help='Directory for saving checkpoints')
    args = parser.parse_args()

    # Create system
    system = PrecomputedLatentFlowSystem(
        latent_channels=8,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        use_effect_conditioning=not args.no_effect_conditioning,
    )

    print(f"Model params: {sum(p.numel() for p in system.parameters()):,}")
    print(f"Effect conditioning: {not args.no_effect_conditioning}")

    # Create data module
    data_module = PrecomputedLatentDataModule(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create trainer
    trainer = create_trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        precision=args.precision,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    trainer.fit(system, data_module)


if __name__ == '__main__':
    main()
