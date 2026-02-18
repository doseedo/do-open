"""
Training system for Unified HDemucs Inverter.

Single model, single forward pass, any effect chain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from inverse_afx.models.unified_inverter import HDemucsInverter, create_unified_inverter
from inverse_afx.training.losses import MultiResolutionSTFTLoss


@dataclass
class UnifiedTrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Loss weights
    stft_loss_weight: float = 1.0
    l1_loss_weight: float = 0.1

    # Model size
    model_size: str = 'base'  # small, base, large


class UnifiedInverterSystem(pl.LightningModule):
    """
    PyTorch Lightning module for training unified inverter.
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    def __init__(
        self,
        config: UnifiedTrainingConfig,
        sample_rate: int = 48000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.sample_rate = sample_rate

        # Model
        self.model = create_unified_inverter(
            size=config.model_size,
            sample_rate=sample_rate,
        )

        # Losses
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048],
        )

    def forward(
        self,
        wet_audio: torch.Tensor,
        effect_types: torch.Tensor,
        effect_params: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(wet_audio, effect_types, effect_params)

    def compute_loss(
        self,
        dry_pred: torch.Tensor,
        dry_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        # Multi-resolution STFT loss
        stft_loss = self.stft_loss(dry_pred, dry_target)
        losses['stft'] = stft_loss * self.config.stft_loss_weight

        # L1 loss for waveform
        l1_loss = F.l1_loss(dry_pred, dry_target)
        losses['l1'] = l1_loss * self.config.l1_loss_weight

        # Total
        losses['total'] = losses['stft'] + losses['l1']

        return losses

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        wet_audio = batch['wet_audio']
        dry_audio = batch['dry_audio']
        effect_types = batch['effect_types']
        effect_params = batch['effect_params']

        # Forward pass
        dry_pred = self.model(wet_audio, effect_types, effect_params)

        # Compute loss
        losses = self.compute_loss(dry_pred, dry_audio)

        # Log overall
        self.log('train/loss', losses['total'], prog_bar=True)
        self.log('train/stft_loss', losses['stft'])
        self.log('train/l1_loss', losses['l1'])

        # Log per-effect losses (for monitoring which effects are learning)
        self._log_per_effect_loss(effect_types, losses['total'], prefix='train')

        # Log chain length distribution
        chain_lengths = (effect_types >= 0).sum(dim=1).float()
        self.log('train/avg_chain_len', chain_lengths.mean())

        return losses['total']

    def _log_per_effect_loss(
        self,
        effect_types: torch.Tensor,
        loss: torch.Tensor,
        prefix: str = 'train',
    ):
        """Log loss broken down by effect type."""
        # Count samples per effect type
        for idx, effect_name in enumerate(self.EFFECT_TYPES):
            # Check if this effect appears in any position in the chain
            mask = (effect_types == idx).any(dim=1)
            if mask.any():
                self.log(f'{prefix}/loss_{effect_name}', loss, on_step=False, on_epoch=True)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        wet_audio = batch['wet_audio']
        dry_audio = batch['dry_audio']
        effect_types = batch['effect_types']
        effect_params = batch['effect_params']

        # Forward pass
        dry_pred = self.model(wet_audio, effect_types, effect_params)

        # Compute loss
        losses = self.compute_loss(dry_pred, dry_audio)

        # Compute SI-SDR
        si_sdr = self._compute_si_sdr(dry_pred, dry_audio)
        si_sdr_baseline = self._compute_si_sdr(wet_audio, dry_audio)

        # Log
        self.log('val/loss', losses['total'], prog_bar=True, sync_dist=True)
        self.log('val/si_sdr', si_sdr.mean(), prog_bar=True, sync_dist=True)
        self.log('val/si_sdr_improvement', (si_sdr - si_sdr_baseline).mean(), sync_dist=True)

        return losses

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

        return si_sdr.mean(dim=1)  # Average over channels

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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


# ===== Dataset Adapter =====

class UnifiedDataset(torch.utils.data.Dataset):
    """
    Dataset adapter for unified inverter training.

    Converts existing manifest format to unified format with
    effect_types and effect_params tensors.
    """

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']
    MAX_CHAIN_LENGTH = 6
    MAX_PARAMS = 15

    # Parameter counts per effect (for padding)
    PARAM_COUNTS = {
        'eq': 5,        # 5-band EQ gains
        'compressor': 6,  # threshold, ratio, attack, release, knee, makeup
        'reverb': 4,     # decay, predelay, wet, damping
        'distortion': 4,  # drive, tone, mix, output
        'chorus': 4,      # rate, depth, mix, feedback
        'delay': 3,       # delay_ms, feedback, mix
    }

    def __init__(
        self,
        manifest: List[Dict],
        sample_rate: int = 48000,
        segment_length: int = 144000,
    ):
        self.manifest = manifest
        self.sample_rate = sample_rate
        self.segment_length = segment_length

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.manifest[idx]

        # Load audio
        import torchaudio
        wet_audio, sr = torchaudio.load(item['wet_path'])
        dry_audio, _ = torchaudio.load(item['dry_path'])

        # Ensure correct length
        if wet_audio.shape[-1] > self.segment_length:
            wet_audio = wet_audio[..., :self.segment_length]
            dry_audio = dry_audio[..., :self.segment_length]
        elif wet_audio.shape[-1] < self.segment_length:
            pad = self.segment_length - wet_audio.shape[-1]
            wet_audio = F.pad(wet_audio, (0, pad))
            dry_audio = F.pad(dry_audio, (0, pad))

        # Parse chain spec
        chain_spec = item.get('chain_spec', [])
        effect_types, effect_params = self._encode_chain(chain_spec)

        return {
            'wet_audio': wet_audio,
            'dry_audio': dry_audio,
            'effect_types': effect_types,
            'effect_params': effect_params,
        }

    def _encode_chain(
        self,
        chain_spec: List,
    ) -> tuple:
        """
        Encode chain spec into tensors.

        Args:
            chain_spec: List of [effect_name, params_dict] pairs

        Returns:
            effect_types: [max_chain] tensor of effect indices (-1 for padding)
            effect_params: [max_chain, max_params] tensor of normalized params
        """
        effect_types = torch.full((self.MAX_CHAIN_LENGTH,), -1, dtype=torch.long)
        effect_params = torch.zeros((self.MAX_CHAIN_LENGTH, self.MAX_PARAMS))

        for i, (effect_name, params) in enumerate(chain_spec):
            if i >= self.MAX_CHAIN_LENGTH:
                break

            # Effect type index
            if effect_name in self.EFFECT_TYPES:
                effect_types[i] = self.EFFECT_TYPES.index(effect_name)
            else:
                continue

            # Normalize and encode params
            normalized = self._normalize_params(effect_name, params)
            effect_params[i, :len(normalized)] = torch.tensor(normalized)

        return effect_types, effect_params

    def _normalize_params(
        self,
        effect_name: str,
        params: Dict,
    ) -> List[float]:
        """Normalize parameters to [0, 1] range."""

        if effect_name == 'eq':
            # 5-band EQ: gains in dB, normalize from [-12, 12] to [0, 1]
            gains = []
            for band in ['low', 'low_mid', 'mid', 'high_mid', 'high']:
                g = params.get(f'{band}_gain_db', params.get(f'{band}_gain', 0))
                gains.append((g + 12) / 24)  # [-12, 12] -> [0, 1]
            return gains

        elif effect_name == 'compressor':
            return [
                (params.get('threshold_db', params.get('threshold', -20)) + 60) / 60,  # [-60, 0] -> [0, 1]
                (params.get('ratio', 4) - 1) / 19,  # [1, 20] -> [0, 1]
                params.get('attack_ms', params.get('attack', 10)) / 100,  # [0, 100] -> [0, 1]
                params.get('release_ms', params.get('release', 100)) / 1000,  # [0, 1000] -> [0, 1]
                params.get('knee_db', params.get('knee', 0)) / 12,  # [0, 12] -> [0, 1]
                params.get('makeup_db', params.get('makeup', 0)) / 24,  # [0, 24] -> [0, 1]
            ]

        elif effect_name == 'reverb':
            return [
                params.get('decay_time', 1) / 10,  # [0, 10] -> [0, 1]
                params.get('pre_delay_ms', params.get('pre_delay', 0)) / 100,  # [0, 100] -> [0, 1]
                params.get('wet_mix', params.get('mix', 0.5)),  # Already [0, 1]
                params.get('damping', 0.5),  # Already [0, 1]
            ]

        elif effect_name == 'distortion':
            return [
                params.get('drive', 0.5),  # [0, 1]
                params.get('tone', 0.5),  # [0, 1]
                params.get('mix', 1.0),  # [0, 1]
                (params.get('output_gain_db', params.get('output', 0)) + 12) / 24,  # [-12, 12] -> [0, 1]
            ]

        elif effect_name == 'chorus':
            return [
                params.get('rate_hz', params.get('rate', 1)) / 10,  # [0, 10] -> [0, 1]
                params.get('depth', 0.5),  # [0, 1]
                params.get('mix', 0.5),  # [0, 1]
                params.get('feedback', 0),  # [0, 1]
            ]

        elif effect_name == 'delay':
            return [
                params.get('delay_ms', params.get('delay', 250)) / 2000,  # [0, 2000] -> [0, 1]
                params.get('feedback', 0.3),  # [0, 1]
                params.get('mix', 0.5),  # [0, 1]
            ]

        return []


class UnifiedDataModule(pl.LightningDataModule):
    """DataModule for unified inverter training."""

    def __init__(
        self,
        manifest_path: str,
        batch_size: int = 8,
        num_workers: int = 8,
        sample_rate: int = 48000,
        segment_length: int = 144000,
        val_split: float = 0.1,
        max_chain_length: Optional[int] = None,  # Filter by chain length
        effect_types: Optional[List[str]] = None,  # Filter by effect types
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.val_split = val_split
        self.max_chain_length = max_chain_length
        self.effect_types = effect_types

    def setup(self, stage: Optional[str] = None):
        import json

        with open(self.manifest_path) as f:
            manifest = json.load(f)

        # Filter by chain length if specified
        if self.max_chain_length is not None:
            manifest = [
                m for m in manifest
                if len(m.get('chain_spec', [])) <= self.max_chain_length
            ]
            print(f"Filtered to chains <= {self.max_chain_length}: {len(manifest)} samples")

        # Filter by effect types if specified
        if self.effect_types is not None:
            def has_allowed_effects(m):
                chain = m.get('chain_spec', [])
                return all(fx[0] in self.effect_types for fx in chain)
            manifest = [m for m in manifest if has_allowed_effects(m)]
            print(f"Filtered to effects {self.effect_types}: {len(manifest)} samples")

        # Split
        n_val = int(len(manifest) * self.val_split)
        self.train_manifest = manifest[n_val:]
        self.val_manifest = manifest[:n_val]

        self.train_dataset = UnifiedDataset(
            self.train_manifest,
            sample_rate=self.sample_rate,
            segment_length=self.segment_length,
        )

        self.val_dataset = UnifiedDataset(
            self.val_manifest,
            sample_rate=self.sample_rate,
            segment_length=self.segment_length,
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
