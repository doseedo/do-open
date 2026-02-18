"""
Training with forward consistency loss.

Key insight: The model can't know transformation magnitude from inputs alone.
Solution: Verify predictions by re-applying effects.

wet → [Flow] → pred_dry → [Effects] → rewet
                                        ↓
                                Compare to wet
                                        ↓
                            If wrong, adjust

The forward model (effects) validates the inverse model.
"""

import sys
if "/home/arlo/Data/ACE-Step" not in sys.path:
    sys.path.insert(0, "/home/arlo/Data/ACE-Step")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, List, Tuple

from inverse_afx.data.precomputed_latent_dataset import PrecomputedLatentDataModule
from inverse_afx.models.differentiable_chain import DifferentiableFXChain
from inverse_afx.training.train_latent_flow_precomputed import (
    EffectConditionedVelocityNet,
    SimpleVelocityNet,
)


class LatentFlowWithConsistency(pl.LightningModule):
    """
    Flow matching with forward consistency loss.

    The consistency loss teaches correct magnitude by verifying:
    apply_effects(pred_dry) ≈ wet
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_dim: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        use_effect_conditioning: bool = True,
        consistency_weight: float = 0.5,
        dcae_checkpoint_path: str = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path: str = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder",
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.use_effect_conditioning = use_effect_conditioning
        self.consistency_weight = consistency_weight
        self.sample_rate = sample_rate

        # Velocity network
        if use_effect_conditioning:
            self.velocity_net = EffectConditionedVelocityNet(
                latent_channels=latent_channels,
                hidden_dim=hidden_dim,
            )
        else:
            self.velocity_net = SimpleVelocityNet(
                latent_channels=latent_channels,
                hidden_dim=hidden_dim,
            )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # DCAE codec (for decoding latents to audio)
        self.dcae = None  # Lazy load to avoid memory issues
        self.dcae_checkpoint_path = dcae_checkpoint_path
        self.vocoder_checkpoint_path = vocoder_checkpoint_path

        # Differentiable effects chain
        self.fx_chain = DifferentiableFXChain(sample_rate=sample_rate)

    def _load_dcae(self):
        """Lazy load DCAE codec."""
        if self.dcae is None:
            from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
            self.dcae = MusicDCAE(
                source_sample_rate=self.sample_rate,
                dcae_checkpoint_path=self.dcae_checkpoint_path,
                vocoder_checkpoint_path=self.vocoder_checkpoint_path,
            )
            self.dcae.to(self.device)
            self.dcae.eval()
            for param in self.dcae.parameters():
                param.requires_grad = False

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to audio."""
        self._load_dcae()

        # latent: [B, C, H, T] -> audio: [B, 2, T_audio]
        B = latent.shape[0]
        T_latent = latent.shape[-1]

        # Estimate audio length (DCAE uses 8x compression in time)
        audio_length = T_latent * 8 * 512  # Approximate
        audio_lengths = torch.full((B,), audio_length, device=latent.device)

        with torch.no_grad():
            audio = self.dcae.decode(latent, audio_lengths)

        return audio

    def build_chain_spec(
        self,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> List[List[Tuple[str, torch.Tensor]]]:
        """
        Convert batch tensors to chain specs for DifferentiableFXChain.

        Args:
            effect_types: [B, max_chain] effect type indices
            effect_params: [B, max_chain, max_params] normalized parameters

        Returns:
            List of chain_specs, one per batch item
        """
        B = effect_types.shape[0]
        batch_specs = []

        for b in range(B):
            chain_spec = []
            for i in range(effect_types.shape[1]):
                etype_idx = effect_types[b, i].item()
                if etype_idx < 0:  # Padding
                    continue
                if etype_idx >= len(self.EFFECT_TYPES):
                    continue

                etype = self.EFFECT_TYPES[etype_idx]
                params = effect_params[b, i]  # [max_params]
                chain_spec.append((etype, params))

            batch_specs.append(chain_spec)

        return batch_specs

    def apply_effects_batch(
        self,
        audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply effects to a batch of audio.

        Note: Currently processes one at a time due to varying chain specs.
        """
        B = audio.shape[0]
        chain_specs = self.build_chain_spec(effect_types, effect_params)

        outputs = []
        for b in range(B):
            # Get single sample
            single_audio = audio[b:b+1]  # [1, C, T]

            # Convert stereo to mono if needed
            if single_audio.shape[1] == 2:
                single_audio = single_audio.mean(dim=1, keepdim=True)

            # Apply effects
            if chain_specs[b]:
                wet_audio = self.fx_chain(single_audio, chain_specs[b])
            else:
                wet_audio = single_audio

            outputs.append(wet_audio)

        # Stack and convert back to stereo
        output = torch.cat(outputs, dim=0)  # [B, 1, T]
        output = output.expand(-1, 2, -1)  # [B, 2, T]

        return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        wet_latent = batch['wet_latent']
        dry_latent = batch['dry_latent']

        B = wet_latent.shape[0]
        device = wet_latent.device

        # Random timestep
        t = torch.rand(B, device=device)
        t_expand = t.view(-1, 1, 1, 1)
        z_t = t_expand * dry_latent + (1 - t_expand) * wet_latent

        # Target velocity
        target_velocity = dry_latent - wet_latent

        # Predict velocity
        if self.use_effect_conditioning and 'effect_types' in batch:
            effect_types = batch['effect_types']
            effect_params = batch['effect_params']
            pred_velocity, _ = self.velocity_net(z_t, t, effect_types, effect_params)
        else:
            pred_velocity = self.velocity_net(z_t, t)
            effect_types = None
            effect_params = None

        # Standard flow loss (magnitude-weighted)
        target_mag = (target_velocity ** 2).mean(dim=(1, 2, 3)).sqrt() + 1e-6
        weight = 1.0 / (target_mag + 0.1)
        weight = weight / weight.mean()

        diff = (pred_velocity - target_velocity) ** 2
        flow_loss = (diff * weight.view(-1, 1, 1, 1)).mean()

        # Forward consistency loss (every N batches to save compute)
        consistency_loss = torch.tensor(0.0, device=device)

        if self.consistency_weight > 0 and batch_idx % 10 == 0 and effect_types is not None:
            try:
                # Single-step prediction for efficiency
                pred_dry_latent = wet_latent + pred_velocity

                # Decode to audio
                pred_dry_audio = self.decode_latent(pred_dry_latent)
                wet_audio = self.decode_latent(wet_latent)

                # Re-apply effects
                rewet_audio = self.apply_effects_batch(
                    pred_dry_audio, effect_types, effect_params
                )

                # Consistency loss: rewet should match original wet
                # Use only overlapping length
                min_len = min(rewet_audio.shape[-1], wet_audio.shape[-1])
                consistency_loss = F.mse_loss(
                    rewet_audio[..., :min_len],
                    wet_audio[..., :min_len]
                )

                self.log('train/consistency_loss', consistency_loss)
            except Exception as e:
                # Skip consistency if it fails (shape mismatch, etc.)
                print(f"Consistency loss failed: {e}")

        # Total loss
        total_loss = flow_loss + self.consistency_weight * consistency_loss

        self.log('train/flow_loss', flow_loss, prog_bar=True)
        self.log('train/total_loss', total_loss)

        return total_loss

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

        loss = F.mse_loss(pred_velocity, target_velocity)
        self.log('val/flow_loss', loss, prog_bar=True)
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
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


def create_trainer(
    max_epochs: int = 100,
    devices: int = 1,
    precision: str = "32",
    accumulate_grad_batches: int = 4,
    checkpoint_dir: str = "checkpoints/latent_flow_consistency",
) -> pl.Trainer:
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="flow-{epoch:02d}-{val_flow_loss:.4f}",
            monitor="val/flow_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    logger = TensorBoardLogger("logs/latent_flow_consistency", name="consistency")

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
    parser = argparse.ArgumentParser(description='Train with forward consistency')
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)  # Smaller due to DCAE
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--consistency-weight', type=float, default=0.5)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-dir', type=str,
                        default='checkpoints/latent_flow_consistency')
    args = parser.parse_args()

    system = LatentFlowWithConsistency(
        learning_rate=args.lr,
        consistency_weight=args.consistency_weight,
    )

    print(f"Model params: {sum(p.numel() for p in system.parameters()):,}")
    print(f"Consistency weight: {args.consistency_weight}")

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
