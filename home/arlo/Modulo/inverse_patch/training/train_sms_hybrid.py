#!/usr/bin/env python3
"""
Hybrid SMS Distillation Training: Sines + Noise Bands.

Training (fast - no audio synthesis!):
  z_dcae → Mapper → (freqs, amps, phases, noise_amps)_pred
                         ↓ MSE loss
  SMS.analyze(audio) → (freqs, amps, phases, noise_amps)_target

The mapper learns to predict both:
- Deterministic component: Sinusoidal tracks (freqs, amps, phases)
- Stochastic component: Noise band amplitudes (mel-spaced)

Inference:
  z_dcae → Mapper → (sines, noise) → Hybrid Synth → Audio
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import orjson

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

# Synthesis constants - computed from extracted data (mean across 4713 samples)
NOISE_MIX_RATIO = 0.43  # Noise ratio for synthesis (noise_energy / total_energy)


# ============================================================
# HYBRID SAMI MAPPER
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


class HybridSAMIMapper(nn.Module):
    """
    Maps DCAE latents to hybrid sine + noise parameters.

    Outputs:
    - freqs: [B, T, n_sines] - Frequencies in Hz
    - amps: [B, T, n_sines] - Sine amplitudes [0, 1]
    - phases: [B, T, n_sines] - Phases [-π, π]
    - noise_amps: [B, T, n_noise_bands] - Noise band amplitudes [0, 1]

    SAMI-informed architecture:
    - Channels 0-3: Coarse features (energy, rough pitch)
    - Channels 4-7: Fine features (harmonics, timbre, noise)
    """

    def __init__(
        self,
        n_sines: int = 64,
        n_noise_bands: int = 8,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        freq_min: float = 20.0,
        freq_max: float = 16000.0,
    ):
        super().__init__()
        self.n_sines = n_sines
        self.n_noise_bands = n_noise_bands
        self.freq_min = freq_min
        self.freq_max = freq_max

        # SAMI-informed: process coarse and fine channels separately
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

        # Sine parameter heads
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

        # Noise band head - uses fine features more heavily
        # Noise is typically in high-frequency fine details
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, n_noise_bands),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, 8, 16, T] DCAE latent

        Returns:
            freqs: [B, T, n_sines] in Hz
            amps: [B, T, n_sines] in [0, 1]
            phases: [B, T, n_sines] in [-π, π]
            noise_amps: [B, T, n_noise_bands] in [0, 1]
        """
        B, C, H, T = z.shape

        # Split into coarse (0-3) and fine (4-7)
        z_coarse = z[:, :4, :, :].permute(0, 3, 1, 2).reshape(B, T, 64)
        z_fine = z[:, 4:, :, :].permute(0, 3, 1, 2).reshape(B, T, 64)

        # Encode separately
        h_coarse = self.coarse_encoder(z_coarse)
        h_fine = self.fine_encoder(z_fine)

        # Combine
        h = torch.cat([h_coarse, h_fine], dim=-1)
        h = self.combiner(h)

        # Frequency output (log scale)
        freq_logits = self.freq_head(h)
        log_freq_min = np.log(self.freq_min)
        log_freq_max = np.log(self.freq_max)
        log_freqs = log_freq_min + torch.sigmoid(freq_logits) * (log_freq_max - log_freq_min)
        freqs = torch.exp(log_freqs)

        # Amplitude output
        amps = torch.sigmoid(self.amp_head(h))

        # Phase output
        phases = torch.tanh(self.phase_head(h)) * np.pi

        # Noise band output
        noise_amps = torch.sigmoid(self.noise_head(h))

        return {
            'freqs': freqs,
            'amps': amps,
            'phases': phases,
            'noise_amps': noise_amps,
        }


# ============================================================
# DATASET
# ============================================================

class HybridSMSDataset(Dataset):
    """Dataset with hybrid SMS parameters (sines + noise bands)."""

    # Keywords to filter out drum/percussion samples
    DRUM_KEYWORDS = ['drum', 'kick', 'snare', 'hat', 'tom', 'perc', 'cymbal', 'overhead', ' oh ', '_oh_', 'hihat', 'hh_', '_hh']

    def __init__(
        self,
        sms_manifest_path: str,
        max_samples: Optional[int] = None,
        target_frames: int = 22,
        skip_drums: bool = True,
        noise_scale: float = 0.4,  # Calibration fix for over-scaled noise
        amp_scale: float = 10.0,   # Calibration fix for under-scaled amps
        n_sines: Optional[int] = None,  # If provided, slice to this many sines
    ):
        self.target_frames = target_frames
        self.noise_scale = noise_scale
        self.amp_scale = amp_scale

        print(f"Loading hybrid SMS manifest from {sms_manifest_path}...")
        with open(sms_manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        entries = manifest['entries']
        extracted_n_sines = manifest.get('n_sines', 64)
        self.n_noise_bands = manifest.get('n_noise_bands', 8)

        # Use requested n_sines if provided and smaller than extracted
        if n_sines is not None and n_sines < extracted_n_sines:
            self.n_sines = n_sines
            print(f"  Slicing from {extracted_n_sines} to {n_sines} sines")
        else:
            self.n_sines = extracted_n_sines

        if max_samples:
            entries = entries[:max_samples]

        print(f"  Found {len(entries)} entries (n_sines={self.n_sines}, n_noise_bands={self.n_noise_bands})")
        print(f"  Skip drums: {skip_drums}")

        self.data = []
        loaded = 0
        skipped_drums = 0

        for entry in entries:
            try:
                # Load SMS params
                sms_data = torch.load(entry['path'], weights_only=True, map_location='cpu')

                # Filter out drums if requested
                if skip_drums:
                    audio_path = sms_data.get('audio_path', '').lower()
                    if any(kw in audio_path for kw in self.DRUM_KEYWORDS):
                        skipped_drums += 1
                        continue

                # Load latent
                lat_data = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
                if 'latents' in lat_data:
                    latent = lat_data['latents']
                elif 'latent' in lat_data:
                    latent = lat_data['latent']
                else:
                    continue

                # Get params
                C, H, T = latent.shape
                freqs = sms_data['freqs']
                amps = sms_data['amps']
                phases = sms_data['phases']
                noise_amps = sms_data.get('noise_amps', torch.zeros(freqs.shape[0], self.n_noise_bands))

                # Slice to requested n_sines if needed
                if freqs.shape[1] > self.n_sines:
                    freqs = freqs[:, :self.n_sines]
                    amps = amps[:, :self.n_sines]
                    phases = phases[:, :self.n_sines]

                # Calibration fix: amps were normalized too aggressively
                # Scale up so more sines are "active" (above 0.01 threshold)
                amps = amps * self.amp_scale

                # Calibration fix: noise was over-scaled during extraction
                noise_amps = noise_amps * self.noise_scale

                # Crop/pad to target frames - find region with most activity
                if T < target_frames:
                    latent = F.pad(latent, (0, target_frames - T))
                    freqs = F.pad(freqs, (0, 0, 0, target_frames - freqs.shape[0]))
                    amps = F.pad(amps, (0, 0, 0, target_frames - amps.shape[0]))
                    phases = F.pad(phases, (0, 0, 0, target_frames - phases.shape[0]))
                    noise_amps = F.pad(noise_amps, (0, 0, 0, target_frames - noise_amps.shape[0]))
                elif T > target_frames:
                    # Find window with most activity using conv1d (fast)
                    activity = amps.sum(dim=1)  # [T]
                    # Use cumsum for fast window sum
                    cumsum = torch.cumsum(activity, dim=0)
                    # Window sums: cumsum[i+w] - cumsum[i]
                    padded = F.pad(cumsum, (1, 0))
                    window_sums = padded[target_frames:] - padded[:-target_frames]
                    best_start = window_sums.argmax().item()
                    start = min(best_start, T - target_frames)
                    latent = latent[:, :, start:start + target_frames]
                    freqs = freqs[start:start + target_frames]
                    amps = amps[start:start + target_frames]
                    phases = phases[start:start + target_frames]
                    noise_amps = noise_amps[start:start + target_frames]

                self.data.append({
                    'latent': latent,
                    'freqs': freqs,
                    'amps': amps,
                    'phases': phases,
                    'noise_amps': noise_amps,
                })

                loaded += 1
                if loaded % 500 == 0:
                    print(f"\r    Loaded {loaded}...", end="", flush=True)

            except Exception:
                continue

        print(f"\r    Loaded {len(self.data)} samples (skipped {skipped_drums} drum samples)")

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
        'noise_amps': torch.stack([b['noise_amps'] for b in batch]),
    }


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def hungarian_matching_loss(pred_freqs, pred_amps, target_freqs, target_amps,
                            pred_phases=None, target_phases=None,
                            freq_weight=1.0, amp_weight=10.0,
                            frame_weights=None):
    """
    Fast amplitude-weighted frequency matching loss.

    Instead of Hungarian, use amplitude-sorted matching:
    - Sort both pred and target by amplitude (descending)
    - Match in order (loudest to loudest)
    - This ensures high-amplitude predictions match high-amplitude targets

    Args:
        frame_weights: Optional [B, T] weights to prioritize active frames over silent ones

    IMPORTANT: Amp loss is computed on ALL slots, not just active targets.
    This forces the model to predict low amps for inactive slots.
    """
    B, T, N = pred_freqs.shape
    device = pred_freqs.device

    # Sort by amplitude (descending)
    pred_order = pred_amps.argsort(dim=-1, descending=True)
    target_order = target_amps.argsort(dim=-1, descending=True)

    # Gather sorted values
    pred_f = pred_freqs.gather(-1, pred_order)
    pred_a = pred_amps.gather(-1, pred_order)
    target_f = target_freqs.gather(-1, target_order)
    target_a = target_amps.gather(-1, target_order)

    # Mask for active targets (after sorting, active ones come first)
    active_mask = (target_a > 0.01).float()

    # Frequency loss (log scale) - only on active targets
    log_pred_f = torch.log(pred_f.clamp(min=20))
    log_target_f = torch.log(target_f.clamp(min=20))
    freq_err = active_mask * (log_pred_f - log_target_f).pow(2)  # [B, T, N]

    # Amplitude loss - ON ALL SLOTS
    amp_err = (pred_a - target_a).pow(2)  # [B, T, N]

    # Phase loss
    phase_err = torch.zeros_like(freq_err)
    if pred_phases is not None and target_phases is not None:
        pred_ph = pred_phases.gather(-1, pred_order)
        target_ph = target_phases.gather(-1, target_order)
        phase_diff = torch.abs(pred_ph - target_ph)
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
        phase_err = active_mask * phase_diff.pow(2)

    # Apply frame weights if provided
    if frame_weights is not None:
        # frame_weights: [B, T] -> [B, T, 1] for broadcasting
        fw = frame_weights.unsqueeze(-1)
        # Weight and reduce: sum over N, weighted mean over B*T
        freq_loss = (fw * freq_err).sum() / (fw * active_mask).sum().clamp(min=1e-8)
        amp_loss = (fw * amp_err).mean()
        phase_loss = (fw * phase_err).sum() / (fw * active_mask).sum().clamp(min=1e-8)
    else:
        # Original reduction
        freq_loss = freq_err.sum() / active_mask.sum().clamp(min=1e-8)
        amp_loss = amp_err.mean()
        phase_loss = phase_err.sum() / active_mask.sum().clamp(min=1e-8)

    return freq_loss, amp_loss, phase_loss


def frequency_sorted_loss(pred_freqs, pred_amps, target_freqs, target_amps,
                          pred_phases=None, target_phases=None):
    """OLD: Compute loss with frequency-sorted matching (has amplitude assignment bug)."""
    B, T, N = pred_freqs.shape

    # Sort by frequency
    pred_order = pred_freqs.argsort(dim=-1)
    target_order = target_freqs.argsort(dim=-1)

    pred_f = pred_freqs.gather(-1, pred_order)
    pred_a = pred_amps.gather(-1, pred_order)
    target_f = target_freqs.gather(-1, target_order)
    target_a = target_amps.gather(-1, target_order)

    # Mask for active sines
    active_mask = (target_f > 20).float()

    # Frequency loss (log scale)
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
        phase_diff = torch.abs(pred_ph - target_ph)
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
        phase_loss = (active_mask * phase_diff.pow(2)).sum() / (active_mask.sum() + 1e-8)

    return freq_loss, amp_loss, phase_loss


def noise_band_loss(pred_noise, target_noise):
    """MSE loss for noise band amplitudes."""
    return F.mse_loss(pred_noise, target_noise)


# ============================================================
# TRAINER
# ============================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HybridSMSTrainer:
    """Train mapper to predict hybrid (sines + noise) params from DCAE latents."""

    def __init__(
        self,
        n_sines: int = 64,
        n_noise_bands: int = 8,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        sparsity_weight: float = 0.001,
        phase_weight: float = 0.1,
        noise_weight: float = 1.0,
        use_noise: bool = True,
        device: str = 'cuda',
    ):
        self.device = device
        self.sparsity_weight = sparsity_weight
        self.phase_weight = phase_weight
        self.noise_weight = noise_weight
        self.use_noise = use_noise

        self.mapper = HybridSAMIMapper(
            n_sines=n_sines,
            n_noise_bands=n_noise_bands,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
        ).to(device)

        self.scaler = torch.amp.GradScaler('cuda')

        params = sum(p.numel() for p in self.mapper.parameters())
        print(f"\nHybridSMSTrainer:")
        print(f"  Mapper: SAMI-informed hybrid")
        print(f"  N sines: {n_sines}, N noise bands: {n_noise_bands}")
        print(f"  Hidden dim: {hidden_dim}, Blocks: {n_blocks}")
        print(f"  Params: {params:,}")
        print(f"  Weights: sparsity={sparsity_weight}, phase={phase_weight}, noise={noise_weight}")
        print(f"  Use noise: {use_noise}")

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()

        latent = batch['latent'].to(self.device)
        target_freqs = batch['freqs'].to(self.device)
        target_amps = batch['amps'].to(self.device)
        target_phases = batch['phases'].to(self.device)
        target_noise = batch['noise_amps'].to(self.device)

        with torch.amp.autocast('cuda'):
            pred = self.mapper(latent)

            # === FRAME WEIGHTING ===
            # Weight active frames higher than silent frames so model focuses on learning
            # actual content, not just predicting silence
            ACTIVE_THRESH = 0.01  # Threshold for "active" sine
            SILENT_WEIGHT = 0.1   # Silent frames contribute 10x less
            ACTIVE_WEIGHT = 1.0   # Active frames get full weight

            # Per-frame activity: number of active sines in target [B, T]
            frame_activity = (target_amps > ACTIVE_THRESH).float().sum(dim=-1)
            # Frame is "active" if it has at least 1 active sine
            frame_is_active = (frame_activity >= 1.0).float()
            # Compute per-frame weights
            frame_weights = frame_is_active * ACTIVE_WEIGHT + (1 - frame_is_active) * SILENT_WEIGHT
            # Normalize so weights sum to T (preserve loss scale)
            frame_weights = frame_weights / frame_weights.mean(dim=-1, keepdim=True).clamp(min=1e-6)
            # [B, T] -> [B, T, 1] for broadcasting
            frame_weights_3d = frame_weights.unsqueeze(-1)

            # Sine losses with Hungarian matching (proper assignment)
            freq_loss, amp_loss, phase_loss = hungarian_matching_loss(
                pred['freqs'], pred['amps'],
                target_freqs, target_amps,
                pred['phases'], target_phases,
                frame_weights=frame_weights  # Pass weights
            )

            # Noise loss (weighted by frame activity) - skip if use_noise=False
            if self.use_noise:
                noise_err = (pred['noise_amps'] - target_noise).pow(2)
                noise_loss = (frame_weights_3d * noise_err).mean()
            else:
                noise_loss = torch.tensor(0.0, device=self.device)

            # === ANTI-COLLAPSE LOSSES (also weighted) ===

            # 1. Hard count loss - penalize wrong NUMBER of active sines
            steepness = 50.0
            pred_active_soft = torch.sigmoid((pred['amps'] - ACTIVE_THRESH) * steepness).sum(dim=-1)
            target_active_soft = torch.sigmoid((target_amps - ACTIVE_THRESH) * steepness).sum(dim=-1)
            count_err = (pred_active_soft - target_active_soft).pow(2)
            count_loss = (frame_weights * count_err).mean()

            # 2. Top-K amplitude matching
            pred_amps_sorted = pred['amps'].sort(dim=-1, descending=True)[0]
            target_amps_sorted = target_amps.sort(dim=-1, descending=True)[0]
            K = 16
            pos_weights = torch.exp(-torch.arange(K, device=self.device).float() * 0.1)
            pos_weights = pos_weights / pos_weights.sum()
            topk_err = (pos_weights * (pred_amps_sorted[:, :, :K] - target_amps_sorted[:, :, :K]).pow(2)).sum(dim=-1)
            topk_loss = (frame_weights * topk_err).mean()

            # 3. Active sine coverage
            target_active_mask = (target_amps > ACTIVE_THRESH).float()
            target_active_energy = (target_amps * target_active_mask).sum(dim=-1)
            pred_active_energy = (pred_amps_sorted[:, :, :K] * (target_amps_sorted[:, :, :K] > ACTIVE_THRESH).float()).sum(dim=-1)
            coverage_err = ((target_active_energy - pred_active_energy).clamp(min=0) / target_active_energy.clamp(min=1e-6)).pow(2)
            coverage_loss = (frame_weights * coverage_err).mean()

            # 4. Anti-sparsity: penalize fewer active than target
            pred_count = (pred['amps'] > ACTIVE_THRESH).float().sum(dim=-1)
            target_count = (target_amps > ACTIVE_THRESH).float().sum(dim=-1)
            underpred_err = F.relu(target_count - pred_count).pow(2)
            underpred_loss = (frame_weights * underpred_err).mean()

            # 5. FREQUENCY SMOOTHNESS - match TARGET's smoothness, not arbitrary smoothness
            # Sort by amplitude to track the "same" sines across frames
            pred_freq_sorted = pred['freqs'].gather(-1, pred['amps'].argsort(dim=-1, descending=True))
            target_freq_sorted = target_freqs.gather(-1, target_amps.argsort(dim=-1, descending=True))

            # Frame-to-frame frequency difference (log scale for perceptual relevance)
            pred_log_freq = torch.log(pred_freq_sorted.clamp(min=20))
            target_log_freq = torch.log(target_freq_sorted.clamp(min=20))

            # Compute frame-to-frame changes for BOTH pred and target
            pred_freq_diff = pred_log_freq[:, 1:, :] - pred_log_freq[:, :-1, :]  # [B, T-1, N]
            target_freq_diff = target_log_freq[:, 1:, :] - target_log_freq[:, :-1, :]

            # Penalize difference between pred's changes and target's changes
            # This allows genuine jumps (attacks) but penalizes extra wobble
            diff_err = (pred_freq_diff - target_freq_diff).pow(2)  # [B, T-1, N]

            # Weight by target amplitude - focus on active sines
            target_amp_sorted = target_amps_sorted
            amp_weight = (target_amp_sorted[:, :-1, :] + target_amp_sorted[:, 1:, :]) / 2
            amp_weight = amp_weight / amp_weight.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            # Smoothness loss on top K sines
            K_smooth = 8
            smooth_err = (amp_weight[:, :, :K_smooth] * diff_err[:, :, :K_smooth]).sum(dim=-1)
            freq_smooth_loss = smooth_err.mean()

            # Combined loss - focus on reconstruction, minimal auxiliary penalties
            loss = (freq_loss +
                    amp_loss +
                    self.phase_weight * phase_loss +
                    self.noise_weight * noise_loss +
                    0.1 * count_loss +      # Reduced from 5.0
                    1.0 * topk_loss +        # Reduced from 10.0
                    0.1 * coverage_loss +    # Reduced from 5.0
                    0.1 * underpred_loss +   # Reduced from 10.0
                    1.0 * freq_smooth_loss)  # Reduced from 20.0

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        # Count active sines (use same threshold for fair comparison)
        n_active_pred = (pred['amps'] > 0.01).float().sum(dim=-1).mean().item()
        n_active_target = (target_amps > 0.01).float().sum(dim=-1).mean().item()

        # Noise activity
        noise_active_pred = pred['noise_amps'].mean().item()
        noise_active_target = target_noise.mean().item()

        return {
            'loss': loss.item(),
            'freq_loss': freq_loss.item(),
            'amp_loss': amp_loss.item(),
            'phase_loss': phase_loss.item(),
            'noise_loss': noise_loss.item(),
            'count_loss': count_loss.item(),
            'topk_loss': topk_loss.item(),
            'coverage_loss': coverage_loss.item(),
            'underpred_loss': underpred_loss.item(),
            'freq_smooth_loss': freq_smooth_loss.item(),
            'n_active_pred': n_active_pred,
            'n_active_target': n_active_target,
            'noise_pred': noise_active_pred,
            'noise_target': noise_active_target,
        }

    def train(self, dataloader, n_epochs: int = 100, lr: float = 1e-3,
              save_dir: Optional[str] = None):
        optimizer = torch.optim.AdamW(self.mapper.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print("\n" + "=" * 70)
        print("Hybrid SMS Distillation Training (Sines + Noise)")
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
                  f"smooth={metrics_avg['freq_smooth_loss']:.4f} "
                  f"under={metrics_avg['underpred_loss']:.4f} "
                  f"sines={metrics_avg['n_active_pred']:.0f}/{metrics_avg['n_active_target']:.0f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

            if save_dir and metrics_avg['loss'] < best_loss:
                best_loss = metrics_avg['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.mapper.state_dict(),
                    'loss': best_loss,
                    'n_sines': self.mapper.n_sines,
                    'n_noise_bands': self.mapper.n_noise_bands,
                }, str(save_path / "best_model.pt"))

            if epoch % 20 == 0:
                clear_memory()

        if save_dir:
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': self.mapper.state_dict(),
                'loss': metrics_avg['loss'],
                'n_sines': self.mapper.n_sines,
                'n_noise_bands': self.mapper.n_noise_bands,
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
                        default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_hybrid/sms_manifest.json')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_sines', type=int, default=64)
    parser.add_argument('--n_noise_bands', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--sparsity', type=float, default=0.001)
    parser.add_argument('--noise_weight', type=float, default=1.0)
    parser.add_argument('--skip_drums', action='store_true', help='Filter out drum/percussion samples')
    parser.add_argument('--no_noise', action='store_true', help='Train on sines only, skip noise loss')
    parser.add_argument('--noise_scale', type=float, default=0.4, help='Scale factor for noise_amps (calibration fix)')
    parser.add_argument('--amp_scale', type=float, default=10.0, help='Scale factor for sine amps (calibration fix)')
    args = parser.parse_args()

    print("=" * 70)
    print("Hybrid SMS Distillation: z_dcae → Sine + Noise Params")
    print("=" * 70)
    print("\nLearning to extract:")
    print("  - Deterministic: Sinusoidal tracks (freqs, amps, phases)")
    print("  - Stochastic: Noise band amplitudes")
    sys.stdout.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load dataset
    dataset = HybridSMSDataset(
        sms_manifest_path=args.sms_manifest,
        max_samples=args.max_samples,
        skip_drums=args.skip_drums,
        noise_scale=args.noise_scale,
        amp_scale=args.amp_scale,
        n_sines=args.n_sines,  # Pass requested n_sines for slicing
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    print(f"Dataloader ready: {len(dataloader)} batches")

    # Train
    trainer = HybridSMSTrainer(
        n_sines=args.n_sines,
        n_noise_bands=args.n_noise_bands,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        sparsity_weight=args.sparsity,
        noise_weight=args.noise_weight,
        use_noise=not args.no_noise,
        device=device,
    )

    save_dir = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/sms_hybrid"

    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
