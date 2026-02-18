#!/usr/bin/env python3
"""
Decoder-Guided Sparse Mapper

Instead of learning temporal patterns from scratch, leverage the DCAE decoder's
already-learned attention/temporal processing.

Architecture:
  z → [frozen decoder conv_in] → [frozen decoder attention block] → [trainable sine head] → sines

The decoder already learned how to temporally mix z frames. We just change the output
from dense audio to sparse sines.
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import orjson

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# DECODER-GUIDED MAPPER
# ============================================================

class DecoderGuidedMapper(nn.Module):
    """
    Use DCAE decoder's learned temporal processing, replace output with sine params.

    Frozen:
      - conv_in: expands z from 8 to 1024 channels
      - up_blocks[3]: EfficientViT attention (temporal mixing)

    Trainable:
      - sine_head: converts 1024-channel features to sine parameters
    """

    def __init__(
        self,
        dcae_decoder: nn.Module,
        n_sines: int = 64,
        hidden_dim: int = 256,
        freq_min: float = 20.0,
        freq_max: float = 8000.0,
    ):
        super().__init__()
        self.n_sines = n_sines
        self.freq_min = freq_min
        self.freq_max = freq_max

        # Steal layers from decoder
        self.conv_in = dcae_decoder.conv_in
        self.attention_block = dcae_decoder.up_blocks[3]

        # Freeze decoder layers - use their learned patterns
        for p in self.conv_in.parameters():
            p.requires_grad = False
        for p in self.attention_block.parameters():
            p.requires_grad = False

        # Feature dim from decoder (1024 channels, 16 height)
        feature_dim = 1024 * 16

        # Trainable sine head
        self.sine_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        # Separate heads for each parameter
        self.freq_head = nn.Linear(hidden_dim, n_sines)
        self.amp_head = nn.Linear(hidden_dim, n_sines)
        self.phase_head = nn.Linear(hidden_dim, n_sines)

        # Log frequency range
        self.log_freq_min = np.log(freq_min)
        self.log_freq_max = np.log(freq_max)

        # Count params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"DecoderGuidedMapper:")
        print(f"  Frozen params (from decoder): {frozen:,}")
        print(f"  Trainable params (sine head): {trainable:,}")

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, 8, 16, T] DCAE latent

        Returns:
            freqs: [B, T, n_sines]
            amps: [B, T, n_sines]
            phases: [B, T, n_sines]
        """
        B, C, H, T = z.shape

        # Decoder expects [B, C, T, H]
        z = z.permute(0, 1, 3, 2)  # [B, 8, T, 16]

        # Use decoder's temporal processing (frozen)
        with torch.no_grad():
            h = self.conv_in(z)           # [B, 1024, T, 16]
            h = self.attention_block(h)   # [B, 1024, T, 16] - attention mixed!

        # Flatten spatial dims for our head
        # [B, 1024, T, 16] -> [B, T, 1024*16]
        h = h.permute(0, 2, 1, 3).reshape(B, T, -1)

        # Trainable sine head
        h = self.sine_head(h)  # [B, T, hidden_dim]

        # Output heads
        freq_logits = self.freq_head(h)
        freq_norm = torch.sigmoid(freq_logits)
        log_freqs = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freqs)

        amps = torch.sigmoid(self.amp_head(h))
        phases = torch.tanh(self.phase_head(h)) * np.pi

        return {
            'freqs': freqs,
            'amps': amps,
            'phases': phases,
        }


# ============================================================
# DATASET (same as before)
# ============================================================

class SMSDataset(Dataset):
    """Dataset for SMS training."""

    DRUM_KEYWORDS = ['drum', 'kick', 'snare', 'hat', 'tom', 'perc', 'cymbal',
                     'overhead', ' oh ', '_oh_', 'hihat', 'hh_', '_hh']

    def __init__(
        self,
        sms_manifest_path: str,
        max_samples: Optional[int] = None,
        target_frames: int = 22,
        skip_drums: bool = True,
        n_sines: Optional[int] = None,
        amp_scale: float = 10.0,
    ):
        self.target_frames = target_frames
        self.amp_scale = amp_scale

        print(f"Loading SMS manifest from {sms_manifest_path}...")
        with open(sms_manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        entries = manifest['entries']
        extracted_n_sines = manifest.get('n_sines', 64)

        if n_sines is not None and n_sines < extracted_n_sines:
            self.n_sines = n_sines
        else:
            self.n_sines = extracted_n_sines

        if max_samples:
            entries = entries[:max_samples]

        print(f"  Found {len(entries)} entries, using {self.n_sines} sines")

        self.data = []
        skipped = 0

        for entry in entries:
            try:
                sms_data = torch.load(entry['path'], weights_only=True, map_location='cpu')

                if skip_drums:
                    audio_path = sms_data.get('audio_path', '').lower()
                    if any(kw in audio_path for kw in self.DRUM_KEYWORDS):
                        skipped += 1
                        continue

                lat_data = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
                if 'latents' in lat_data:
                    latent = lat_data['latents']
                elif 'latent' in lat_data:
                    latent = lat_data['latent']
                else:
                    continue

                C, H, T = latent.shape
                freqs = sms_data['freqs']
                amps = sms_data['amps'] * self.amp_scale
                phases = sms_data['phases']

                if freqs.shape[1] > self.n_sines:
                    freqs = freqs[:, :self.n_sines]
                    amps = amps[:, :self.n_sines]
                    phases = phases[:, :self.n_sines]

                if T < target_frames:
                    latent = F.pad(latent, (0, target_frames - T))
                    freqs = F.pad(freqs, (0, 0, 0, target_frames - freqs.shape[0]))
                    amps = F.pad(amps, (0, 0, 0, target_frames - amps.shape[0]))
                    phases = F.pad(phases, (0, 0, 0, target_frames - phases.shape[0]))
                elif T > target_frames:
                    activity = amps.sum(dim=1)
                    cumsum = torch.cumsum(activity, dim=0)
                    padded = F.pad(cumsum, (1, 0))
                    window_sums = padded[target_frames:] - padded[:-target_frames]
                    start = min(window_sums.argmax().item(), T - target_frames)
                    latent = latent[:, :, start:start + target_frames]
                    freqs = freqs[start:start + target_frames]
                    amps = amps[start:start + target_frames]
                    phases = phases[start:start + target_frames]

                self.data.append({
                    'latent': latent,
                    'freqs': freqs,
                    'amps': amps,
                    'phases': phases,
                })

            except Exception:
                continue

        print(f"  Loaded {len(self.data)} samples (skipped {skipped} drums)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return {
        'latent': torch.stack([b['latent'] for b in batch]),
        'freqs': torch.stack([b['freqs'] for b in batch]),
        'amps': torch.stack([b['amps'] for b in batch]),
        'phases': torch.stack([b['phases'] for b in batch]),
    }


# ============================================================
# LOSS (Sinkhorn matching)
# ============================================================

def sinkhorn_matching_loss(
    pred_freqs: torch.Tensor,
    pred_amps: torch.Tensor,
    target_freqs: torch.Tensor,
    target_amps: torch.Tensor,
    pred_phases: Optional[torch.Tensor] = None,
    target_phases: Optional[torch.Tensor] = None,
    n_iters: int = 50,
    tau: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sinkhorn optimal transport matching."""
    B, T, N_pred = pred_freqs.shape
    N_target = target_freqs.shape[-1]
    device = pred_freqs.device

    log_pred = torch.log(pred_freqs.clamp(min=20))
    log_target = torch.log(target_freqs.clamp(min=20))

    freq_cost = (log_pred.unsqueeze(-1) - log_target.unsqueeze(-2)).pow(2)
    active_mask = (target_amps > 0.01).float()
    cost = freq_cost + (1 - active_mask.unsqueeze(-2)) * 100.0

    log_P = -cost / tau
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    P = torch.exp(log_P)

    freq_loss = (P * freq_cost * active_mask.unsqueeze(-2)).sum() / active_mask.sum().clamp(min=1)

    matched_target_amp = (P * target_amps.unsqueeze(-2)).sum(dim=-1)
    amp_loss = F.mse_loss(pred_amps, matched_target_amp)

    phase_loss = torch.tensor(0.0, device=device)
    if pred_phases is not None and target_phases is not None:
        phase_diff = (pred_phases.unsqueeze(-1) - target_phases.unsqueeze(-2)).abs()
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
        phase_cost = phase_diff.pow(2)
        phase_loss = (P * phase_cost * active_mask.unsqueeze(-2)).sum() / active_mask.sum().clamp(min=1)

    pred_active = (pred_amps > 0.01).float().sum(dim=-1)
    target_active = active_mask.sum(dim=-1)
    count_loss = F.mse_loss(pred_active, target_active)

    total_loss = freq_loss + amp_loss + 0.1 * phase_loss + 0.1 * count_loss

    metrics = {
        'freq_loss': freq_loss.item(),
        'amp_loss': amp_loss.item(),
        'phase_loss': phase_loss.item(),
        'count_loss': count_loss.item(),
        'n_active_pred': pred_amps.gt(0.01).float().sum(dim=-1).mean().item(),
        'n_active_target': target_active.mean().item(),
    }

    return total_loss, metrics


# ============================================================
# TRAINER
# ============================================================

class DecoderGuidedTrainer:
    """Train decoder-guided sparse mapper."""

    def __init__(
        self,
        n_sines: int = 64,
        hidden_dim: int = 256,
        device: str = 'cuda',
    ):
        self.device = device

        # Load DCAE
        print("Loading DCAE decoder...")
        DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
        VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

        dcae = MusicDCAE(
            dcae_checkpoint_path=DCAE_PATH,
            vocoder_checkpoint_path=VOCODER_PATH,
        )

        # Create mapper using decoder layers
        self.mapper = DecoderGuidedMapper(
            dcae_decoder=dcae.dcae.decoder,
            n_sines=n_sines,
            hidden_dim=hidden_dim,
        ).to(device)

        self.n_sines = n_sines
        self.scaler = torch.amp.GradScaler('cuda')

        print(f"\nDecoderGuidedTrainer ready")
        print(f"  N sines: {n_sines}")
        print(f"  Hidden dim: {hidden_dim}")

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()

        latent = batch['latent'].to(self.device)
        target_freqs = batch['freqs'].to(self.device)
        target_amps = batch['amps'].to(self.device)
        target_phases = batch['phases'].to(self.device)

        with torch.amp.autocast('cuda'):
            pred = self.mapper(latent)

            loss, metrics = sinkhorn_matching_loss(
                pred['freqs'], pred['amps'],
                target_freqs, target_amps,
                pred['phases'], target_phases,
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.mapper.parameters() if p.requires_grad],
            1.0
        )
        self.scaler.step(optimizer)
        self.scaler.update()

        metrics['loss'] = loss.item()
        return metrics

    def train(self, dataloader, n_epochs: int = 100, lr: float = 1e-3,
              save_dir: Optional[str] = None):
        # Only optimize trainable params
        trainable_params = [p for p in self.mapper.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print("\n" + "=" * 70)
        print("Decoder-Guided Training")
        print("=" * 70)
        print("Using DCAE decoder's learned temporal attention (frozen)")
        print("Training only the sine output head")
        print("=" * 70)

        for epoch in range(n_epochs):
            self.mapper.train()
            metrics_sum = {}
            n_batches = 0

            for batch in dataloader:
                m = self.train_step(batch, optimizer)
                for k, v in m.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
                n_batches += 1

            scheduler.step()

            metrics_avg = {k: v / n_batches for k, v in metrics_sum.items()}

            print(f"Epoch {epoch:4d}: loss={metrics_avg['loss']:.4f} "
                  f"freq={metrics_avg['freq_loss']:.4f} "
                  f"amp={metrics_avg['amp_loss']:.4f} "
                  f"sines={metrics_avg['n_active_pred']:.0f}/{metrics_avg['n_active_target']:.0f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

            if save_dir and metrics_avg['loss'] < best_loss:
                best_loss = metrics_avg['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.mapper.state_dict(),
                    'loss': best_loss,
                    'config': {
                        'n_sines': self.n_sines,
                    }
                }, str(save_path / "best_model.pt"))

            if epoch % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if save_dir:
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': self.mapper.state_dict(),
                'loss': metrics_avg['loss'],
                'config': {
                    'n_sines': self.n_sines,
                }
            }, str(save_path / "final_model.pt"))
            print(f"\nSaved to {save_dir}")

        print(f"Training complete! Best loss: {best_loss:.4f}")
        return self.mapper


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sms_manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_sines', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--skip_drums', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("Decoder-Guided Sparse Mapper")
    print("=" * 70)
    print("\nKey insight: Use DCAE decoder's learned temporal attention,")
    print("             just replace the output with sine parameters.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load dataset
    dataset = SMSDataset(
        sms_manifest_path=args.sms_manifest,
        max_samples=args.max_samples,
        skip_drums=args.skip_drums,
        n_sines=args.n_sines,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Dataloader: {len(dataloader)} batches")

    # Train
    trainer = DecoderGuidedTrainer(
        n_sines=args.n_sines,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    save_dir = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/decoder_guided"

    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
