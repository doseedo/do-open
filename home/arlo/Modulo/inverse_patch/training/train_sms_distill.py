#!/usr/bin/env python3
"""
SMS Distillation Training: Learn to extract sine params from DCAE latents.

Training (fast - no audio synthesis!):
  z_dcae → Mapper → (freqs, amps, phases)_pred
                         ↓ MSE loss
  SMS.analyze(audio) → (freqs, amps, phases)_target

The mapper learns to read DCAE's latent representation and output
what classical SMS analysis would find in the audio.

Inference:
  z_dcae → Mapper → (freqs, amps, phases) → SineSynth → Audio
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import orjson

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# SAMI-INFORMED MAPPER
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SAMIInformedMapper(nn.Module):
    """
    Maps DCAE latents to sine parameters, informed by SAMI structure.

    SAMI found:
    - Channels 0-3: Coarse features (energy, rough pitch)
    - Channels 4-7: Fine features (harmonics, timbre)
    - Height dim (16): Frequency band information

    Architecture mirrors this: coarse path for fundamentals,
    fine path for harmonics/detail.
    """

    def __init__(
        self,
        n_sines: int = 64,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        freq_min: float = 20.0,
        freq_max: float = 16000.0,
    ):
        super().__init__()
        self.n_sines = n_sines
        self.freq_min = freq_min
        self.freq_max = freq_max

        # SAMI-informed: process coarse and fine channels separately
        # Channels 0-3 (coarse): 4 * 16 = 64 dims
        # Channels 4-7 (fine): 4 * 16 = 64 dims

        self.coarse_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),
            nn.GELU(),
            ResBlock(hidden_dim // 2),
        )

        self.fine_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),
            nn.GELU(),
            ResBlock(hidden_dim // 2),
        )

        # Combine coarse + fine
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            *[ResBlock(hidden_dim) for _ in range(n_blocks - 2)],
        )

        # Separate heads for each param type
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_sines),
        )

        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_sines),
        )

        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_sines),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, 8, 16, T] DCAE latent

        Returns:
            freqs: [B, T, n_sines] in Hz
            amps: [B, T, n_sines] in [0, 1]
            phases: [B, T, n_sines] in [-π, π]
        """
        B, C, H, T = z.shape

        # Split into coarse (0-3) and fine (4-7)
        z_coarse = z[:, :4, :, :].permute(0, 3, 1, 2).reshape(B, T, 64)  # [B, T, 64]
        z_fine = z[:, 4:, :, :].permute(0, 3, 1, 2).reshape(B, T, 64)    # [B, T, 64]

        # Encode separately
        h_coarse = self.coarse_encoder(z_coarse)  # [B, T, hidden/2]
        h_fine = self.fine_encoder(z_fine)        # [B, T, hidden/2]

        # Combine
        h = torch.cat([h_coarse, h_fine], dim=-1)  # [B, T, hidden]
        h = self.combiner(h)

        # Output heads
        # Frequencies: sigmoid to [0,1], then scale to [freq_min, freq_max] in log space
        freq_logits = self.freq_head(h)
        log_freq_min = np.log(self.freq_min)
        log_freq_max = np.log(self.freq_max)
        log_freqs = log_freq_min + torch.sigmoid(freq_logits) * (log_freq_max - log_freq_min)
        freqs = torch.exp(log_freqs)

        # Amplitudes: sigmoid to [0, 1]
        amps = torch.sigmoid(self.amp_head(h))

        # Phases: tanh to [-1, 1], scale to [-π, π]
        phases = torch.tanh(self.phase_head(h)) * np.pi

        return {
            'freqs': freqs,
            'amps': amps,
            'phases': phases,
        }


# ============================================================
# DATASET
# ============================================================

class SMSDistillDataset(Dataset):
    """Dataset with precomputed SMS parameters."""

    def __init__(
        self,
        sms_manifest_path: str,
        max_samples: Optional[int] = None,
        target_frames: int = 22,
    ):
        self.target_frames = target_frames

        print(f"Loading SMS manifest from {sms_manifest_path}...")
        with open(sms_manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        entries = manifest['entries']
        if max_samples:
            entries = entries[:max_samples]

        print(f"  Found {len(entries)} entries, loading...")

        self.data = []
        loaded = 0

        for entry in entries:
            try:
                # Load SMS params
                sms_data = torch.load(entry['path'], weights_only=True, map_location='cpu')

                # Load latent
                lat_data = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
                if 'latents' in lat_data:
                    latent = lat_data['latents']
                elif 'latent' in lat_data:
                    latent = lat_data['latent']
                else:
                    continue

                # Align frames
                C, H, T = latent.shape
                freqs = sms_data['freqs']  # [T_sms, n_sines]
                amps = sms_data['amps']
                phases = sms_data['phases']

                # Crop/pad to target frames
                if T < target_frames:
                    latent = F.pad(latent, (0, target_frames - T))
                    freqs = F.pad(freqs, (0, 0, 0, target_frames - freqs.shape[0]))
                    amps = F.pad(amps, (0, 0, 0, target_frames - amps.shape[0]))
                    phases = F.pad(phases, (0, 0, 0, target_frames - phases.shape[0]))
                elif T > target_frames:
                    start = (T - target_frames) // 2
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

                loaded += 1
                if loaded % 500 == 0:
                    print(f"\r    Loaded {loaded}...", end="", flush=True)

            except Exception:
                continue

        print(f"\r    Loaded {len(self.data)} samples")

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
# LOSS FUNCTIONS
# ============================================================

def frequency_sorted_loss(pred_freqs, pred_amps, target_freqs, target_amps,
                          pred_phases=None, target_phases=None):
    """
    Compute loss with frequency-sorted matching.

    Both pred and target are sorted by frequency before comparison.
    This handles the permutation invariance of sine representation.
    """
    B, T, N = pred_freqs.shape

    # Sort by frequency (ascending)
    pred_order = pred_freqs.argsort(dim=-1)
    target_order = target_freqs.argsort(dim=-1)

    pred_f = pred_freqs.gather(-1, pred_order)
    pred_a = pred_amps.gather(-1, pred_order)
    target_f = target_freqs.gather(-1, target_order)
    target_a = target_amps.gather(-1, target_order)

    # Mask for active sines (freq > 0 in target)
    active_mask = (target_f > 20).float()

    # Frequency loss (log scale for perceptual weighting)
    log_pred_f = torch.log(pred_f.clamp(min=20))
    log_target_f = torch.log(target_f.clamp(min=20))
    freq_loss = (active_mask * (log_pred_f - log_target_f).pow(2)).sum() / (active_mask.sum() + 1e-8)

    # Amplitude loss
    amp_loss = (active_mask * (pred_a - target_a).pow(2)).sum() / (active_mask.sum() + 1e-8)

    # Phase loss (circular)
    phase_loss = 0.0
    if pred_phases is not None and target_phases is not None:
        pred_ph = pred_phases.gather(-1, pred_order)
        target_ph = target_phases.gather(-1, target_order)
        # Circular distance
        phase_diff = torch.abs(pred_ph - target_ph)
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
        phase_loss = (active_mask * phase_diff.pow(2)).sum() / (active_mask.sum() + 1e-8)

    return freq_loss, amp_loss, phase_loss


# ============================================================
# TRAINER
# ============================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SMSDistillTrainer:
    """Train mapper to predict SMS params from DCAE latents."""

    def __init__(
        self,
        n_sines: int = 64,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        sparsity_weight: float = 0.001,
        phase_weight: float = 0.1,
        device: str = 'cuda',
    ):
        self.device = device
        self.sparsity_weight = sparsity_weight
        self.phase_weight = phase_weight

        self.mapper = SAMIInformedMapper(
            n_sines=n_sines,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
        ).to(device)

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')

        params = sum(p.numel() for p in self.mapper.parameters())
        print(f"\nSMSDistillTrainer:")
        print(f"  Mapper: SAMI-informed (coarse/fine split)")
        print(f"  N sines: {n_sines}")
        print(f"  Hidden dim: {hidden_dim}, Blocks: {n_blocks}")
        print(f"  Params: {params:,}")
        print(f"  Sparsity weight: {sparsity_weight}")

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()

        latent = batch['latent'].to(self.device)      # [B, 8, 16, T]
        target_freqs = batch['freqs'].to(self.device)  # [B, T, n_sines]
        target_amps = batch['amps'].to(self.device)
        target_phases = batch['phases'].to(self.device)

        with torch.amp.autocast('cuda'):
            # Forward
            pred = self.mapper(latent)

            # Sorted loss
            freq_loss, amp_loss, phase_loss = frequency_sorted_loss(
                pred['freqs'], pred['amps'],
                target_freqs, target_amps,
                pred['phases'], target_phases
            )

            # Sparsity on amplitudes
            sparsity_loss = pred['amps'].mean()

            loss = freq_loss + amp_loss + self.phase_weight * phase_loss + self.sparsity_weight * sparsity_loss

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        # Count active sines
        n_active_pred = (pred['amps'] > 0.1).float().sum(dim=-1).mean().item()
        n_active_target = (target_amps > 0.01).float().sum(dim=-1).mean().item()

        return {
            'loss': loss.item(),
            'freq_loss': freq_loss.item(),
            'amp_loss': amp_loss.item(),
            'phase_loss': phase_loss.item(),
            'sparsity': sparsity_loss.item(),
            'n_active_pred': n_active_pred,
            'n_active_target': n_active_target,
        }

    def train(self, dataloader, n_epochs: int = 100, lr: float = 1e-3,
              save_dir: Optional[str] = None):
        optimizer = torch.optim.AdamW(self.mapper.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print("\n" + "=" * 60)
        print("SMS Distillation Training")
        print("=" * 60)

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

            # Average metrics
            metrics_avg = {k: v / n_batches for k, v in metrics_sum.items()}

            print(f"Epoch {epoch:4d}: loss={metrics_avg['loss']:.4f} "
                  f"freq={metrics_avg['freq_loss']:.4f} "
                  f"amp={metrics_avg['amp_loss']:.4f} "
                  f"active={metrics_avg['n_active_pred']:.0f}/{metrics_avg['n_active_target']:.0f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

            if save_dir and metrics_avg['loss'] < best_loss:
                best_loss = metrics_avg['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.mapper.state_dict(),
                    'loss': best_loss,
                }, str(save_path / "best_model.pt"))

            if epoch % 20 == 0:
                clear_memory()

        if save_dir:
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': self.mapper.state_dict(),
                'loss': metrics_avg['loss'],
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
                        default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_params/sms_manifest.json')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_sines', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--sparsity', type=float, default=0.001)
    args = parser.parse_args()

    print("=" * 60)
    print("SMS Distillation: z_dcae → Sine Params")
    print("=" * 60)
    print("\nLearning to extract what SMS analysis finds,")
    print("directly from DCAE's latent representation.")
    sys.stdout.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load dataset
    dataset = SMSDistillDataset(
        sms_manifest_path=args.sms_manifest,
        max_samples=args.max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    print(f"Dataloader ready: {len(dataloader)} batches")

    # Train
    trainer = SMSDistillTrainer(
        n_sines=args.n_sines,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        sparsity_weight=args.sparsity,
        device=device,
    )

    save_dir = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/sms_distill"

    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
