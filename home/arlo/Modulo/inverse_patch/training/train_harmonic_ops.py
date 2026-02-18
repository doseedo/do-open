#!/usr/bin/env python3
"""
Harmonic Operator Mapper: Structured prediction of sine parameters.

HYBRID APPROACH:
  - Harmonic operators: Physics-constrained (f_n = f0 × n, proven math)
  - Free sines: Unconstrained (escape hatch for inharmonic content)

MATHEMATICALLY PROVEN (cannot hurt):
  - f_n = f0 × n        : Fourier series - this IS how pitched sound works
  - a_n = a0 × decay^n  : Energy dissipation across harmonics

PRESCRIBED (tunable):
  - n_ops, harmonics_per_op, n_free_sines: Hyperparameters to tune
  - Coarse→pitch routing: Based on SAMI analysis, may need adjustment

The harmonic expansion is DETERMINISTIC PHYSICS, not learned.
Network only learns f0, decay, amp (easy); physics does the expansion (hard).
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

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# HARMONIC OPERATOR MAPPER
# ============================================================

class HarmonicOperatorMapper(nn.Module):
    """
    Predict structured operators that expand to harmonic series.

    Each operator = one sound source with:
    - f0: fundamental frequency (20-8000 Hz)
    - decay: how harmonics roll off (0-1, lower = brighter)
    - amp: overall loudness (0-1)
    - inharmonicity: stretch factor for piano-like sounds

    Architecture uses SAMI insight:
    - Coarse channels (0-3) → f0, amp (pitch/energy)
    - Fine channels (4-7) → decay, inharmonicity (timbre)
    """

    def __init__(
        self,
        n_ops: int = 8,
        harmonics_per_op: int = 8,
        n_free_sines: int = 8,  # Non-harmonic residuals
        hidden_dim: int = 256,
        n_layers: int = 3,
        freq_min: float = 20.0,
        freq_max: float = 8000.0,
        temporal_kernel: int = 5,  # Temporal context window
        use_temporal_attention: bool = False,  # Use attention instead of conv
        n_attention_layers: int = 1,  # Number of stacked attention layers
        direct_mode: bool = False,  # Skip harmonic expansion, predict all sines directly
        n_direct_sines: int = 64,  # Number of sines to predict in direct mode
    ):
        super().__init__()
        self.n_ops = n_ops
        self.harmonics_per_op = harmonics_per_op
        self.n_free_sines = n_free_sines
        self.n_total_sines = n_ops * harmonics_per_op + n_free_sines
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.temporal_kernel = temporal_kernel
        self.use_temporal_attention = use_temporal_attention
        self.n_attention_layers = n_attention_layers
        self.direct_mode = direct_mode
        self.n_direct_sines = n_direct_sines

        if direct_mode:
            self.n_total_sines = n_direct_sines

        # Temporal context - blend neighboring frames like DCAE decoder does
        if use_temporal_attention:
            # Stacked self-attention layers
            self.temporal_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
                for _ in range(n_attention_layers)
            ])
            self.temporal_norms = nn.ModuleList([
                nn.LayerNorm(128) for _ in range(n_attention_layers)
            ])
            self.temporal_conv = None
        else:
            # Conv: local context window
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=temporal_kernel, padding=temporal_kernel // 2, groups=8),
                nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=temporal_kernel, padding=temporal_kernel // 2, groups=8),
            )
            self.temporal_attention = None

        # SAMI-informed: separate processing for coarse and fine
        self.coarse_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.fine_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Combined processing
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            ) for _ in range(n_layers - 1)]
        )

        if direct_mode:
            # DIRECT MODE: predict all sines independently, no harmonic constraint
            self.direct_freq_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_direct_sines),
            )
            self.direct_amp_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_direct_sines),
            )
            self.phase_head = nn.Linear(hidden_dim, n_direct_sines)
        else:
            # HARMONIC MODE: predict operator params, expand via physics
            # f0: from coarse (pitch info)
            self.f0_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_ops),
            )

            # decay: from fine (timbre info)
            self.decay_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_ops),
            )

            # amp: from combined (energy)
            self.amp_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_ops),
            )

            # inharmonicity: from fine (piano-like stretch)
            self.inharm_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_ops),
            )

            # Free sines for non-harmonic content
            if n_free_sines > 0:
                self.free_freq_head = nn.Linear(hidden_dim, n_free_sines)
                self.free_amp_head = nn.Linear(hidden_dim, n_free_sines)

            # Phase head (for all sines)
            self.phase_head = nn.Linear(hidden_dim, self.n_total_sines)

        # Noise bands (stochastic component)
        self.n_noise_bands = 8
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, self.n_noise_bands),
        )

        # Log frequency range
        self.log_freq_min = np.log(freq_min)
        self.log_freq_max = np.log(freq_max)

        # Pre-compute harmonic numbers (registered as buffer for device placement)
        harmonic_nums = torch.arange(1, harmonics_per_op + 1, dtype=torch.float32)
        self.register_buffer('harmonic_nums', harmonic_nums.view(1, 1, 1, -1))
        exponents = harmonic_nums - 1
        self.register_buffer('exponents', exponents.view(1, 1, 1, -1))

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, 8, 16, T] DCAE latent

        Returns:
            freqs: [B, T, n_total_sines]
            amps: [B, T, n_total_sines]
            phases: [B, T, n_total_sines]
            op_params: dict with f0, decay, amp, inharm for analysis
        """
        B, C, H, T = z.shape

        # === TEMPORAL CONTEXT ===
        # Blend neighboring frames like DCAE decoder does
        z_flat = z.reshape(B, C * H, T)  # [B, 128, T]

        if self.use_temporal_attention:
            # Stacked self-attention: each frame attends to all frames
            z_seq = z_flat.permute(0, 2, 1)  # [B, T, 128]
            for attn, norm in zip(self.temporal_attention_layers, self.temporal_norms):
                z_attended, _ = attn(z_seq, z_seq, z_seq)
                z_seq = norm(z_seq + z_attended)  # Residual + norm
            z_temporal = z_seq.permute(0, 2, 1)  # [B, 128, T]
        else:
            # Conv: local context window
            z_temporal = self.temporal_conv(z_flat)  # [B, 128, T]

        # Reshape back to [B, 8, 16, T]
        z = z_temporal.reshape(B, C, H, T)

        # Split coarse (0-3) and fine (4-7) per SAMI
        z_coarse = z[:, :4, :, :].permute(0, 3, 1, 2).reshape(B, T, 64)
        z_fine = z[:, 4:, :, :].permute(0, 3, 1, 2).reshape(B, T, 64)

        # Encode separately
        h_coarse = self.coarse_encoder(z_coarse)  # [B, T, hidden]
        h_fine = self.fine_encoder(z_fine)

        # Combine
        h = torch.cat([h_coarse, h_fine], dim=-1)
        h = self.combiner(h)  # [B, T, hidden]

        # === SINE PREDICTION ===

        if self.direct_mode:
            # DIRECT MODE: predict all sines independently, no harmonic constraint
            freq_logits = self.direct_freq_head(h)
            freq_normalized = torch.sigmoid(freq_logits)
            log_freq = self.log_freq_min + freq_normalized * (self.log_freq_max - self.log_freq_min)
            all_freqs = torch.exp(log_freq)  # [B, T, n_sines]

            all_amps = torch.sigmoid(self.direct_amp_head(h))  # [B, T, n_sines]

            # No operator params in direct mode
            op_params = None
        else:
            # HARMONIC MODE: predict operator params, expand via physics
            # f0: fundamental frequency (log scale for musical intervals)
            f0_logits = self.f0_head(h_coarse + h * 0.1)  # Mostly from coarse
            f0_normalized = torch.sigmoid(f0_logits)  # [0, 1]
            log_f0 = self.log_freq_min + f0_normalized * (self.log_freq_max - self.log_freq_min)
            f0 = torch.exp(log_f0)  # [B, T, n_ops]

            # decay: harmonic rolloff (lower = brighter, more harmonics)
            decay = torch.sigmoid(self.decay_head(h_fine + h * 0.1))  # [B, T, n_ops]
            decay = 0.3 + decay * 0.65  # Map to [0.3, 0.95] - realistic range

            # amp: overall amplitude per operator
            amp = torch.sigmoid(self.amp_head(h))  # [B, T, n_ops]

            # inharmonicity: stretch factor (1.0 = perfect harmonics)
            inharm_raw = self.inharm_head(h_fine)
            inharm = 1.0 + 0.001 * torch.tanh(inharm_raw)  # [0.999, 1.001] - subtle effect

            # === DETERMINISTIC HARMONIC EXPANSION (fully vectorized) ===
            f0_exp = f0.unsqueeze(-1)
            decay_exp = decay.unsqueeze(-1)
            amp_exp = amp.unsqueeze(-1)
            inharm_exp = inharm.unsqueeze(-1)

            harmonic_freqs = f0_exp * self.harmonic_nums * (inharm_exp ** self.exponents)
            harmonic_amps = amp_exp * (decay_exp ** self.exponents)

            harmonic_freqs = harmonic_freqs.reshape(B, T, -1)
            harmonic_amps = harmonic_amps.reshape(B, T, -1)

            # Free sines
            if self.n_free_sines > 0:
                free_freq_logits = self.free_freq_head(h)
                free_freq_norm = torch.sigmoid(free_freq_logits)
                log_free_freq = self.log_freq_min + free_freq_norm * (self.log_freq_max - self.log_freq_min)
                free_freqs = torch.exp(log_free_freq)
                free_amps = torch.sigmoid(self.free_amp_head(h)) * 0.5

                all_freqs = torch.cat([harmonic_freqs, free_freqs], dim=-1)
                all_amps = torch.cat([harmonic_amps, free_amps], dim=-1)
            else:
                all_freqs = harmonic_freqs
                all_amps = harmonic_amps

            op_params = {'f0': f0, 'decay': decay, 'amp': amp, 'inharm': inharm}

        # === PHASES ===
        phases = torch.tanh(self.phase_head(h)) * np.pi

        # === NOISE BANDS ===
        noise_amps = torch.sigmoid(self.noise_head(h)) * 0.2  # [B, T, 8], max 0.2

        return {
            'freqs': all_freqs,
            'amps': all_amps,
            'phases': phases,
            'noise_amps': noise_amps,
            'op_params': op_params,
        }


# ============================================================
# DATASET (reuse from train_sms_hybrid)
# ============================================================

class HarmonicSMSDataset(Dataset):
    """Dataset for harmonic operator training."""

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
                noise_amps = sms_data.get('noise_amps', torch.zeros(freqs.shape[0], 8))

                # Slice to requested sines
                if freqs.shape[1] > self.n_sines:
                    freqs = freqs[:, :self.n_sines]
                    amps = amps[:, :self.n_sines]
                    phases = phases[:, :self.n_sines]

                # Crop/pad to target frames
                if T < target_frames:
                    latent = F.pad(latent, (0, target_frames - T))
                    freqs = F.pad(freqs, (0, 0, 0, target_frames - freqs.shape[0]))
                    amps = F.pad(amps, (0, 0, 0, target_frames - amps.shape[0]))
                    phases = F.pad(phases, (0, 0, 0, target_frames - phases.shape[0]))
                    noise_amps = F.pad(noise_amps, (0, 0, 0, target_frames - noise_amps.shape[0]))
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
                    noise_amps = noise_amps[start:start + target_frames]

                self.data.append({
                    'latent': latent,
                    'freqs': freqs,
                    'amps': amps,
                    'phases': phases,
                    'noise_amps': noise_amps,
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
        'noise_amps': torch.stack([b['noise_amps'] for b in batch]),
    }


# ============================================================
# LOSS FUNCTIONS
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
    """
    Sinkhorn-based optimal transport matching.

    Fully vectorized, GPU-native, differentiable approximation to Hungarian.
    Finds optimal assignment between predicted and target sines.
    """
    B, T, N_pred = pred_freqs.shape
    N_target = target_freqs.shape[-1]
    device = pred_freqs.device

    # Log frequencies for perceptual distance
    log_pred = torch.log(pred_freqs.clamp(min=20))      # [B, T, N_pred]
    log_target = torch.log(target_freqs.clamp(min=20))  # [B, T, N_target]

    # Cost matrix: frequency distance
    # [B, T, N_pred, 1] - [B, T, 1, N_target] -> [B, T, N_pred, N_target]
    freq_cost = (log_pred.unsqueeze(-1) - log_target.unsqueeze(-2)).pow(2)

    # Mask inactive targets (don't match to silent sines)
    active_mask = (target_amps > 0.01).float()  # [B, T, N_target]

    # Add large cost for inactive targets (push assignment away)
    cost = freq_cost + (1 - active_mask.unsqueeze(-2)) * 100.0

    # Sinkhorn iterations (entropy-regularized optimal transport)
    log_P = -cost / tau
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)  # Row normalize
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)  # Col normalize

    P = torch.exp(log_P)  # [B, T, N_pred, N_target] soft assignment

    # Frequency loss (weighted by assignment probability and active mask)
    freq_loss = (P * freq_cost * active_mask.unsqueeze(-2)).sum() / active_mask.sum().clamp(min=1)

    # Amplitude loss: predicted amp should match assigned target amp
    matched_target_amp = (P * target_amps.unsqueeze(-2)).sum(dim=-1)  # [B, T, N_pred]
    amp_loss = F.mse_loss(pred_amps, matched_target_amp)

    # Phase loss (if provided)
    phase_loss = torch.tensor(0.0, device=device)
    if pred_phases is not None and target_phases is not None:
        # Phase cost matrix
        phase_diff = (pred_phases.unsqueeze(-1) - target_phases.unsqueeze(-2)).abs()
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
        phase_cost = phase_diff.pow(2)
        phase_loss = (P * phase_cost * active_mask.unsqueeze(-2)).sum() / active_mask.sum().clamp(min=1)

    # Count loss
    pred_active = (pred_amps > 0.01).float().sum(dim=-1)
    target_active = active_mask.sum(dim=-1)
    count_loss = F.mse_loss(pred_active, target_active)

    # Combined loss
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


def harmonic_aware_loss(
    pred_freqs: torch.Tensor,
    pred_amps: torch.Tensor,
    target_freqs: torch.Tensor,
    target_amps: torch.Tensor,
    pred_phases: Optional[torch.Tensor] = None,
    target_phases: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Fallback: amplitude-sorted matching (faster but less accurate).
    """
    B, T, N_pred = pred_freqs.shape
    N_target = target_freqs.shape[-1]

    # Sort both by amplitude (descending)
    pred_order = pred_amps.argsort(dim=-1, descending=True)
    target_order = target_amps.argsort(dim=-1, descending=True)

    pred_f = pred_freqs.gather(-1, pred_order)
    pred_a = pred_amps.gather(-1, pred_order)
    target_f = target_freqs.gather(-1, target_order)
    target_a = target_amps.gather(-1, target_order)

    # Match up to min(N_pred, N_target) sines
    N_match = min(N_pred, N_target)
    pred_f = pred_f[:, :, :N_match]
    pred_a = pred_a[:, :, :N_match]
    target_f = target_f[:, :, :N_match]
    target_a = target_a[:, :, :N_match]

    # Active mask (targets with significant amplitude)
    active_mask = (target_a > 0.01).float()

    # Frequency loss (log scale)
    log_pred_f = torch.log(pred_f.clamp(min=20))
    log_target_f = torch.log(target_f.clamp(min=20))
    freq_err = active_mask * (log_pred_f - log_target_f).pow(2)
    freq_loss = freq_err.sum() / active_mask.sum().clamp(min=1)

    # Amplitude loss (all slots)
    amp_loss = F.mse_loss(pred_a, target_a)

    # Phase loss
    phase_loss = torch.tensor(0.0, device=pred_freqs.device)
    if pred_phases is not None and target_phases is not None:
        pred_ph = pred_phases.gather(-1, pred_order)[:, :, :N_match]
        target_ph = target_phases.gather(-1, target_order)[:, :, :N_match]
        phase_diff = torch.abs(pred_ph - target_ph)
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
        phase_loss = (active_mask * phase_diff.pow(2)).sum() / active_mask.sum().clamp(min=1)

    # Count loss (match number of active sines)
    pred_active = (pred_amps > 0.01).float().sum(dim=-1)
    target_active = (target_amps > 0.01).float().sum(dim=-1)
    count_loss = F.mse_loss(pred_active, target_active)

    # Combined loss
    total_loss = freq_loss + amp_loss + 0.1 * phase_loss + 0.1 * count_loss

    metrics = {
        'freq_loss': freq_loss.item(),
        'amp_loss': amp_loss.item(),
        'phase_loss': phase_loss.item(),
        'count_loss': count_loss.item(),
        'n_active_pred': pred_active.mean().item(),
        'n_active_target': target_active.mean().item(),
    }

    return total_loss, metrics


def chroma_loss(
    pred_chroma_2d: torch.Tensor,
    target_freqs: torch.Tensor,
    target_amps: torch.Tensor,
    n_ops: int,
) -> torch.Tensor:
    """
    Compute chroma (pitch class) loss.

    pred_chroma_2d: [B, T, n_ops, 2] - predicted sin/cos of chroma
    target_freqs: [B, T, N_target] - target frequencies
    target_amps: [B, T, N_target] - target amplitudes

    For each operator, match to the closest active target frequency's chroma.
    """
    if pred_chroma_2d is None:
        return torch.tensor(0.0, device=target_freqs.device)

    B, T, N_target = target_freqs.shape
    device = target_freqs.device

    # Extract chroma from target frequencies
    # chroma = log2(freq) mod 1
    log2_freq = torch.log2(target_freqs.clamp(min=20))
    target_chroma = log2_freq % 1.0  # [B, T, N_target] in [0, 1)

    # Convert to sin/cos
    target_chroma_sin = torch.sin(2 * np.pi * target_chroma)
    target_chroma_cos = torch.cos(2 * np.pi * target_chroma)
    target_chroma_2d = torch.stack([target_chroma_sin, target_chroma_cos], dim=-1)  # [B, T, N_target, 2]

    # Weight by amplitude (focus on loud partials)
    weights = target_amps / target_amps.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, T, N_target]

    # For each operator, compute weighted average chroma of targets
    # This is a soft assignment - operator learns "average pitch class" of the frame
    # (In practice, strong fundamentals dominate)
    weighted_target_chroma = (target_chroma_2d * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, 2]

    # Normalize the weighted average (it should be on unit circle)
    weighted_target_chroma = F.normalize(weighted_target_chroma, dim=-1)

    # Compare each operator's chroma to the weighted target
    # All operators should capture the dominant pitch class
    pred_chroma_flat = pred_chroma_2d.mean(dim=2)  # Average across ops [B, T, 2]

    # Cosine similarity (1 = same, -1 = opposite)
    cos_sim = (pred_chroma_flat * weighted_target_chroma).sum(dim=-1)  # [B, T]

    # Loss: 1 - cos_sim (0 when perfect, 2 when opposite)
    loss = (1 - cos_sim).mean()

    return loss


def operator_regularization(op_params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Regularize operator parameters for stability.

    - Encourage diversity in f0 (don't collapse to same pitch)
    - Encourage some operators to be silent (sparsity)
    """
    f0 = op_params['f0']  # [B, T, n_ops]
    amp = op_params['amp']

    # f0 diversity: penalize if all f0s are similar
    f0_log = torch.log(f0.clamp(min=20))
    f0_std = f0_log.std(dim=-1).mean()  # Want this to be high
    diversity_loss = torch.exp(-f0_std)  # Low when diverse

    # Soft sparsity: encourage some ops to be silent
    # But don't force it - let the data decide
    sparsity_loss = amp.mean() * 0.01  # Very light penalty

    return diversity_loss + sparsity_loss


# ============================================================
# TRAINER
# ============================================================

class HarmonicOpTrainer:
    """Train harmonic operator mapper."""

    def __init__(
        self,
        n_ops: int = 8,
        harmonics_per_op: int = 8,
        n_free_sines: int = 8,
        hidden_dim: int = 256,
        use_sinkhorn: bool = True,
        temporal_kernel: int = 5,
        use_temporal_attention: bool = False,
        n_attention_layers: int = 1,
        direct_mode: bool = False,
        n_direct_sines: int = 64,
        device: str = 'cuda',
    ):
        self.device = device
        self.use_sinkhorn = use_sinkhorn
        self.direct_mode = direct_mode

        self.mapper = HarmonicOperatorMapper(
            n_ops=n_ops,
            harmonics_per_op=harmonics_per_op,
            n_free_sines=n_free_sines,
            hidden_dim=hidden_dim,
            temporal_kernel=temporal_kernel,
            use_temporal_attention=use_temporal_attention,
            n_attention_layers=n_attention_layers,
            direct_mode=direct_mode,
            n_direct_sines=n_direct_sines,
        ).to(device)

        self.n_total_sines = self.mapper.n_total_sines

        self.scaler = torch.amp.GradScaler('cuda')

        params = sum(p.numel() for p in self.mapper.parameters())
        print(f"\nHarmonicOpTrainer:")
        if direct_mode:
            print(f"  Mode: DIRECT (no harmonic constraint)")
            print(f"  Sines: {n_direct_sines}")
        else:
            print(f"  Mode: HARMONIC")
            print(f"  Operators: {n_ops}")
            print(f"  Harmonics per op: {harmonics_per_op}")
            print(f"  Free sines: {n_free_sines}")
        print(f"  Total sines: {self.n_total_sines}")
        print(f"  Hidden dim: {hidden_dim}")
        attn_str = f'attention x{n_attention_layers}' if use_temporal_attention else f'conv kernel={temporal_kernel}'
        print(f"  Temporal: {attn_str}")
        print(f"  Params: {params:,}")
        print(f"  Loss: {'Sinkhorn (optimal transport)' if use_sinkhorn else 'Amplitude-sorted'}")

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()

        latent = batch['latent'].to(self.device)
        target_freqs = batch['freqs'].to(self.device)
        target_amps = batch['amps'].to(self.device)
        target_phases = batch['phases'].to(self.device)
        target_noise = batch['noise_amps'].to(self.device)

        with torch.amp.autocast('cuda'):
            pred = self.mapper(latent)

            # Main reconstruction loss (Sinkhorn or amplitude-sorted)
            loss_fn = sinkhorn_matching_loss if self.use_sinkhorn else harmonic_aware_loss
            loss, metrics = loss_fn(
                pred['freqs'], pred['amps'],
                target_freqs, target_amps,
                pred['phases'], target_phases,
            )

            # Noise loss
            noise_loss = F.mse_loss(pred['noise_amps'], target_noise)
            loss = loss + 0.5 * noise_loss
            metrics['noise_loss'] = noise_loss.item()

            # Operator regularization (only in harmonic mode)
            if pred['op_params'] is not None:
                reg_loss = operator_regularization(pred['op_params'])
                loss = loss + 0.1 * reg_loss
                metrics['reg_loss'] = reg_loss.item()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        metrics['loss'] = loss.item()

        # Operator stats (only in harmonic mode)
        op_params = pred['op_params']
        if op_params is not None:
            metrics['mean_f0'] = op_params['f0'].mean().item()
            metrics['mean_decay'] = op_params['decay'].mean().item()
            metrics['mean_op_amp'] = op_params['amp'].mean().item()
            metrics['f0_std'] = torch.log(op_params['f0'].clamp(min=20)).std(dim=-1).mean().item()
        else:
            # Direct mode: use predicted freqs for stats
            metrics['mean_f0'] = pred['freqs'].mean().item()
            metrics['f0_std'] = torch.log(pred['freqs'].clamp(min=20)).std(dim=-1).mean().item()

        return metrics

    def train(self, dataloader, n_epochs: int = 100, lr: float = 1e-3,
              save_dir: Optional[str] = None):
        optimizer = torch.optim.AdamW(self.mapper.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print("\n" + "=" * 70)
        if self.direct_mode:
            print("Direct Sine Training (no harmonic constraint)")
            print("=" * 70)
            print("Network learns: all sine frequencies independently")
        else:
            print("Harmonic Operator Training")
            print("=" * 70)
            print("Physics: f0 → [f0, 2*f0, 3*f0, ...] (deterministic)")
            print("Network learns: f0, decay, amp (easy)")
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
                  f"noise={metrics_avg.get('noise_loss', 0):.4f} "
                  f"f0_std={metrics_avg['f0_std']:.2f} "
                  f"sines={metrics_avg['n_active_pred']:.0f}/{metrics_avg['n_active_target']:.0f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

            if save_dir and metrics_avg['loss'] < best_loss:
                best_loss = metrics_avg['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.mapper.state_dict(),
                    'loss': best_loss,
                    'config': {
                        'n_ops': self.mapper.n_ops,
                        'harmonics_per_op': self.mapper.harmonics_per_op,
                        'n_free_sines': self.mapper.n_free_sines,
                        'n_total_sines': self.n_total_sines,
                        'temporal_kernel': self.mapper.temporal_kernel,
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
                    'n_ops': self.mapper.n_ops,
                    'harmonics_per_op': self.mapper.harmonics_per_op,
                    'n_free_sines': self.mapper.n_free_sines,
                    'n_total_sines': self.n_total_sines,
                    'temporal_kernel': self.mapper.temporal_kernel,
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
    parser.add_argument('--n_ops', type=int, default=8, help='Number of harmonic operators')
    parser.add_argument('--harmonics', type=int, default=8, help='Harmonics per operator')
    parser.add_argument('--free_sines', type=int, default=8, help='Non-harmonic residual sines')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--temporal_kernel', type=int, default=5, help='Temporal conv kernel size (frames of context)')
    parser.add_argument('--temporal_attention', action='store_true', help='Use self-attention over full sequence instead of conv')
    parser.add_argument('--n_attention_layers', type=int, default=1, help='Number of stacked attention layers')
    parser.add_argument('--skip_drums', action='store_true')
    parser.add_argument('--n_target_sines', type=int, default=64, help='Target sines to match')
    parser.add_argument('--no_sinkhorn', action='store_true', help='Use amplitude-sorted matching instead of Sinkhorn')
    parser.add_argument('--direct', action='store_true', help='Direct mode: predict all sines independently, no harmonic constraint')
    args = parser.parse_args()

    print("=" * 70)
    if args.direct:
        print("DIRECT Sine Mapper (no harmonic constraint)")
        print("=" * 70)
        print(f"\nArchitecture:")
        print(f"  {args.n_target_sines} sines predicted independently")
        print(f"\nKey insight: No imposed structure - learns whatever is in the data.")
    else:
        print("Harmonic Operator Mapper")
        print("=" * 70)
        print(f"\nArchitecture:")
        print(f"  {args.n_ops} operators × {args.harmonics} harmonics = {args.n_ops * args.harmonics} harmonic sines")
        print(f"  + {args.free_sines} free sines = {args.n_ops * args.harmonics + args.free_sines} total")
        print(f"\nKey insight: Network predicts f0, decay, amp per operator.")
        print(f"             Expansion to harmonics is PHYSICS, not learned.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load dataset
    dataset = HarmonicSMSDataset(
        sms_manifest_path=args.sms_manifest,
        max_samples=args.max_samples,
        skip_drums=args.skip_drums,
        n_sines=args.n_target_sines,
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
    trainer = HarmonicOpTrainer(
        n_ops=args.n_ops,
        harmonics_per_op=args.harmonics,
        n_free_sines=args.free_sines,
        hidden_dim=args.hidden_dim,
        use_sinkhorn=not args.no_sinkhorn,
        temporal_kernel=args.temporal_kernel,
        use_temporal_attention=args.temporal_attention,
        n_attention_layers=args.n_attention_layers,
        direct_mode=args.direct,
        n_direct_sines=args.n_target_sines,
        device=device,
    )

    save_dir = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/harmonic_ops"

    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
