"""
Training system for Latent Flow Matching Inverter.

Operates in ACE-Step DCAE latent space for efficiency.
Key differences from audio-domain training:
1. Flow matching loss computed in latent space
2. Optional reconstruction loss in audio space
3. Temporal encoders operate on audio (preserve timing resolution)
"""

import sys
# Add ACE-Step to path
if "/home/arlo/Data/ACE-Step" not in sys.path:
    sys.path.insert(0, "/home/arlo/Data/ACE-Step")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from inverse_afx.models.latent_flow_inverter import (
    LatentFlowMatchingInverter,
    LightweightLatentFlowInverter,
    create_latent_flow_inverter,
)
from inverse_afx.training.losses import MultiResolutionSTFTLoss


@dataclass
class LatentFlowTrainingConfig:
    """Training configuration for latent flow inverter."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Loss weights
    flow_loss_weight: float = 1.0
    recon_loss_weight: float = 0.1  # Audio reconstruction loss (optional)

    # Model configuration
    model_size: str = 'base'  # tiny, small, base, large
    use_mamba: bool = False

    # DCAE checkpoint paths (None = use defaults from ACE-Step)
    dcae_checkpoint_path: Optional[str] = None
    vocoder_checkpoint_path: Optional[str] = None

    # Training settings
    n_flow_steps_train: int = 1  # During training, we sample random t
    compute_audio_loss: bool = True  # Whether to compute audio-domain loss


class LatentFlowInverterSystem(pl.LightningModule):
    """
    PyTorch Lightning module for training latent flow inverter.

    Training process:
    1. Encode wet/dry audio to latent space (frozen DCAE)
    2. Extract temporal features from wet audio (audio domain)
    3. Flow matching loss in latent space
    4. Optional: decode and compute audio reconstruction loss
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    def __init__(
        self,
        config: LatentFlowTrainingConfig,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.sample_rate = sample_rate

        # Model
        self.model = create_latent_flow_inverter(
            size=config.model_size,
            sample_rate=sample_rate,
            use_mamba=config.use_mamba,
            dcae_checkpoint_path=config.dcae_checkpoint_path,
            vocoder_checkpoint_path=config.vocoder_checkpoint_path,
        )

        # Audio-domain losses (optional)
        if config.compute_audio_loss:
            self.stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=[512, 1024, 2048],
                hop_sizes=[128, 256, 512],
                win_lengths=[512, 1024, 2048],
            )
        else:
            self.stft_loss = None

    def forward(
        self,
        wet_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        return self.model(wet_audio, effect_types, effect_params, n_steps)

    def compute_flow_loss(
        self,
        wet_audio: torch.Tensor,
        dry_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute flow matching loss in latent space."""
        return self.model.training_step(wet_audio, dry_audio, effect_types, effect_params)

    def compute_audio_recon_loss(
        self,
        dry_pred: torch.Tensor,
        dry_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute audio reconstruction loss."""
        losses = {}

        if self.stft_loss is not None:
            stft_loss = self.stft_loss(dry_pred, dry_target)
            losses['stft'] = stft_loss

            l1_loss = F.l1_loss(dry_pred, dry_target)
            losses['l1'] = l1_loss

            losses['recon_total'] = stft_loss + 0.1 * l1_loss

        return losses

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        wet_audio = batch['wet_audio']
        dry_audio = batch['dry_audio']
        effect_types = batch['effect_types']
        effect_params = batch['effect_params']

        # Flow matching loss in latent space
        flow_losses = self.compute_flow_loss(wet_audio, dry_audio, effect_types, effect_params)
        total_loss = flow_losses['flow_loss'] * self.config.flow_loss_weight

        # Log flow loss
        self.log('train/flow_loss', flow_losses['flow_loss'], prog_bar=True)

        # Optional: audio reconstruction loss
        if self.config.compute_audio_loss and self.config.recon_loss_weight > 0:
            # Generate prediction (with few steps for efficiency during training)
            with torch.no_grad():
                # Use 2 steps during training for speed
                dry_pred = self.model(wet_audio, effect_types, effect_params, n_steps=2)

            recon_losses = self.compute_audio_recon_loss(dry_pred, dry_audio)

            if 'recon_total' in recon_losses:
                total_loss = total_loss + recon_losses['recon_total'] * self.config.recon_loss_weight
                self.log('train/recon_loss', recon_losses['recon_total'])
                self.log('train/stft_loss', recon_losses['stft'])
                self.log('train/l1_loss', recon_losses['l1'])

        self.log('train/total_loss', total_loss, prog_bar=True)

        # Log per-effect losses
        self._log_per_effect_loss(effect_types, total_loss, prefix='train')

        return total_loss

    def _log_per_effect_loss(
        self,
        effect_types: torch.Tensor,
        loss: torch.Tensor,
        prefix: str = 'train',
    ):
        """Log loss broken down by effect type."""
        for idx, effect_name in enumerate(self.EFFECT_TYPES):
            mask = (effect_types == idx).any(dim=1)
            if mask.any():
                self.log(f'{prefix}/loss_{effect_name}', loss, on_step=False, on_epoch=True)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        wet_audio = batch['wet_audio']
        dry_audio = batch['dry_audio']
        effect_types = batch['effect_types']
        effect_params = batch['effect_params']

        # Flow loss
        flow_losses = self.compute_flow_loss(wet_audio, dry_audio, effect_types, effect_params)

        # Generate prediction with proper number of steps
        n_steps = self.model.estimate_steps(effect_types) if hasattr(self.model, 'estimate_steps') else 8
        dry_pred = self.model(wet_audio, effect_types, effect_params, n_steps=n_steps)

        # Audio metrics
        si_sdr = self._compute_si_sdr(dry_pred, dry_audio)
        si_sdr_baseline = self._compute_si_sdr(wet_audio, dry_audio)

        # Log
        self.log('val/flow_loss', flow_losses['flow_loss'], prog_bar=True, sync_dist=True)
        self.log('val/si_sdr', si_sdr.mean(), prog_bar=True, sync_dist=True)
        self.log('val/si_sdr_improvement', (si_sdr - si_sdr_baseline).mean(), sync_dist=True)

        # Optional recon loss
        if self.config.compute_audio_loss:
            recon_losses = self.compute_audio_recon_loss(dry_pred, dry_audio)
            if 'recon_total' in recon_losses:
                self.log('val/recon_loss', recon_losses['recon_total'], sync_dist=True)

        return flow_losses

    def _compute_si_sdr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Scale-Invariant SDR."""
        eps = 1e-8

        # Remove mean
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        # Compute SI-SDR
        dot = (pred * target).sum(dim=-1, keepdim=True)
        s_target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
        proj = dot * target / s_target_energy

        noise = pred - proj

        si_sdr = 10 * torch.log10(
            (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
        )

        return si_sdr.mean(dim=1)

    def configure_optimizers(self):
        # Only train the flow network, not the frozen codec
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_steps,
            pct_start=self.config.warmup_steps / self.config.max_steps,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


# Re-export dataset classes from train_unified for convenience
from inverse_afx.training.train_unified import UnifiedDataset, UnifiedDataModule


class LatentFlowDataModule(UnifiedDataModule):
    """
    DataModule for latent flow inverter training.

    Same as UnifiedDataModule but can include additional options
    specific to latent flow training.
    """

    def __init__(
        self,
        manifest_path: str,
        batch_size: int = 4,  # Smaller batch due to latent processing
        num_workers: int = 8,
        sample_rate: int = 44100,  # Match DAC sample rate
        segment_length: int = 88200,  # 2 seconds at 44.1kHz
        val_split: float = 0.1,
        max_chain_length: Optional[int] = None,
        effect_types: Optional[List[str]] = None,
    ):
        super().__init__(
            manifest_path=manifest_path,
            batch_size=batch_size,
            num_workers=num_workers,
            sample_rate=sample_rate,
            segment_length=segment_length,
            val_split=val_split,
            max_chain_length=max_chain_length,
            effect_types=effect_types,
        )


def create_trainer(
    max_epochs: int = 50,
    devices: int = 1,
    precision: str = "16-mixed",
    accumulate_grad_batches: int = 4,
    log_dir: str = "logs/latent_flow",
    checkpoint_dir: str = "checkpoints/latent_flow",
) -> pl.Trainer:
    """Create PyTorch Lightning trainer with good defaults."""
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="latent_flow-{epoch:02d}-{val_si_sdr:.2f}",
            monitor="val/si_sdr",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    logger = TensorBoardLogger(log_dir, name="latent_flow_inverter")

    trainer = pl.Trainer(
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

    return trainer


if __name__ == '__main__':
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Latent Flow Inverter')
    parser.add_argument('--manifest', type=str, required=True, help='Path to manifest JSON')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--model_size', type=str, default='base', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--use_mamba', action='store_true')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['32', '16-mixed', 'bf16-mixed'])
    args = parser.parse_args()

    # Create config
    config = LatentFlowTrainingConfig(
        model_size=args.model_size,
        use_mamba=args.use_mamba,
        learning_rate=args.lr,
    )

    # Create system
    system = LatentFlowInverterSystem(config, sample_rate=44100)

    # Create data module
    data_module = LatentFlowDataModule(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        sample_rate=44100,
    )

    # Create trainer
    trainer = create_trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        precision=args.precision,
    )

    # Train
    trainer.fit(system, data_module)
