#!/usr/bin/env python3
"""
Train z_dcae → Sines with reconstruction loss.

Pipeline (respects temporal structure):
  z_dcae [B, 8, 16, T] → reshape [B, T, 128]
    → per-frame mapper → [B, T, n_sines, 3]
    → SineSynth → audio_pred
                      ↓
                 spectral loss
                      ↓
                   audio_gt (loaded from file)

NO SAMI flattening. DCAE already has temporal structure - use it.
SAMI's discovery (coarse/fine dims) informs analysis, not synthesis.
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
import torchaudio
import orjson
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# PER-FRAME SINE MAPPER (UNPRESCRIBED)
# ============================================================

class ResBlock(nn.Module):
    """Residual block for deeper network."""
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


class SparseSineMapper(nn.Module):
    """
    Deep network: [B, T, 128] → [B, T, max_sines, 3]

    Key ideas:
    - Large pool of potential sines (max_sines=512)
    - L1 sparsity on amplitudes → network learns to use minimum needed
    - Deeper network with residual blocks → can learn differentiated roles
    - Separate heads for freq/amp/phase → different input dims can specialize
    """

    def __init__(
        self,
        frame_dim: int = 128,
        max_sines: int = 512,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.frame_dim = frame_dim
        self.max_sines = max_sines
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.GELU(),
            *[ResBlock(hidden_dim) for _ in range(n_blocks)],
        )

        # Separate heads for each parameter type
        # This encourages the network to learn different roles
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_sines),
        )

        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_sines),
        )

        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, max_sines),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, T, 128] per-frame latent
        Returns:
            freqs: [B, T, max_sines] in Hz
            amps: [B, T, max_sines] in [0, 1] - sparse!
            phases: [B, T, max_sines] in [0, 2π]
        """
        h = self.encoder(z)  # [B, T, hidden_dim]

        # Separate projections
        # Min freq 20 Hz to prevent DC collapse
        min_freq = 20.0
        freqs = min_freq + torch.sigmoid(self.freq_head(h)) * (self.nyquist - min_freq)
        amps = torch.sigmoid(self.amp_head(h))  # Sparsity loss will push most to ~0
        phases = torch.sigmoid(self.phase_head(h)) * 2 * np.pi

        return {'freqs': freqs, 'amps': amps, 'phases': phases}

    def count_active_sines(self, amps: torch.Tensor, threshold: float = 0.1) -> float:
        """Count average number of sines with amplitude > threshold."""
        return (amps > threshold).float().sum(dim=-1).mean().item()

    def analyze_learned_structure(self) -> Dict:
        """Analyze which input dims each head uses."""
        # Get encoder output sensitivity to input
        # Use first linear layer as proxy
        W_enc = self.encoder[0].weight.data  # [hidden_dim, 128]

        # Get head input sensitivity
        W_freq = self.freq_head[0].weight.data  # [hidden/2, hidden]
        W_amp = self.amp_head[0].weight.data
        W_phase = self.phase_head[0].weight.data

        # Effective sensitivity to input
        freq_sens = (W_freq @ W_enc).abs().mean(dim=0)  # [128]
        amp_sens = (W_amp @ W_enc).abs().mean(dim=0)
        phase_sens = (W_phase @ W_enc).abs().mean(dim=0)

        _, freq_top = freq_sens.topk(20)
        _, amp_top = amp_sens.topk(20)
        _, phase_top = phase_sens.topk(20)

        return {
            'freq_importance': freq_sens,
            'amp_importance': amp_sens,
            'phase_importance': phase_sens,
            'freq_top_dims': freq_top.tolist(),
            'amp_top_dims': amp_top.tolist(),
            'phase_top_dims': phase_top.tolist(),
        }


# ============================================================
# SINE SYNTHESIZER
# ============================================================

class SineSynth(nn.Module):
    """Differentiable additive sine synthesis."""

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        freqs: torch.Tensor,
        amps: torch.Tensor,
        phases: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """
        Args:
            freqs: [B, T, n_sines] in Hz
            amps: [B, T, n_sines]
            phases: [B, T, n_sines]
            n_samples: output audio length

        Returns:
            audio: [B, n_samples]
        """
        B, T, N = freqs.shape
        device = freqs.device

        # Interpolate params to audio rate
        freqs = F.interpolate(freqs.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)
        amps = F.interpolate(amps.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)
        phases_interp = F.interpolate(phases.transpose(1, 2), size=n_samples, mode='linear').transpose(1, 2)

        # Cumulative phase for time-varying frequency
        dt = 1.0 / self.sample_rate
        inst_phase = 2 * np.pi * torch.cumsum(freqs * dt, dim=1) + phases_interp[:, :1, :]

        # Generate sines and sum
        sines = torch.sin(inst_phase)
        audio = (sines * amps).sum(dim=-1)

        # Normalize
        audio = audio / (audio.abs().max(dim=-1, keepdim=True)[0] + 1e-8)

        return audio


# ============================================================
# FULL PIPELINE
# ============================================================

class DCAESinePipeline(nn.Module):
    """
    z_dcae → sparse sines → audio

    Key: Large pool of sines, but sparsity loss encourages using minimum.
    Network learns both WHAT to represent and HOW MANY sines needed.
    """

    def __init__(
        self,
        max_sines: int = 512,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.mapper = SparseSineMapper(
            frame_dim=128,
            max_sines=max_sines,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            sample_rate=sample_rate,
        )
        self.synth = SineSynth(sample_rate)

        self.max_sines = max_sines
        self.sample_rate = sample_rate

    def forward(self, z_dcae: torch.Tensor, n_samples: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_dcae: [B, 8, 16, T] DCAE latent
            n_samples: output audio length

        Returns:
            audio: [B, n_samples]
            freqs, amps, phases: [B, T, max_sines]
        """
        B, C, H, T = z_dcae.shape

        # Reshape to [B, T, C*H] - preserve temporal structure
        z = z_dcae.permute(0, 3, 1, 2).reshape(B, T, C * H)  # [B, T, 128]

        # Per-frame mapping (sparse)
        params = self.mapper(z)

        # Synthesize
        audio = self.synth(
            params['freqs'],
            params['amps'],
            params['phases'],
            n_samples,
        )

        return {
            'audio': audio,
            'freqs': params['freqs'],
            'amps': params['amps'],
            'phases': params['phases'],
        }

    def count_active_sines(self, amps: torch.Tensor, threshold: float = 0.1) -> float:
        """Count average active sines."""
        return self.mapper.count_active_sines(amps, threshold)


# ============================================================
# SPECTRAL LOSS
# ============================================================

class AnalyticalSpectrumLoss(nn.Module):
    """
    Compute loss WITHOUT synthesizing audio.

    Key insight: We know what spectrum sines produce analytically.
    A sine at freq f with amp a → peak at bin f * n_fft / sr

    Instead of: synth → STFT → loss
    We do: predict_spectrum_from_params → loss

    ~100x faster than audio synthesis!
    """

    def __init__(
        self,
        n_fft: int = 2048,
        sample_rate: int = 44100,
        n_mels: int = 128,
        peak_width: float = 2.0,  # Width of frequency peaks in bins
    ):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.peak_width = peak_width
        self.n_bins = n_fft // 2 + 1

        # Frequency bins
        self.register_buffer('freq_bins', torch.linspace(0, sample_rate/2, self.n_bins))

        # Mel filterbank for comparing in perceptual space
        mel_fb = self._create_mel_filterbank(n_mels, n_fft, sample_rate)
        self.register_buffer('mel_fb', mel_fb)

    def _create_mel_filterbank(self, n_mels, n_fft, sr):
        """Create mel filterbank matrix."""
        n_bins = n_fft // 2 + 1

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_low = hz_to_mel(20)
        mel_high = hz_to_mel(sr / 2)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        fb = np.zeros((n_mels, n_bins))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return torch.from_numpy(fb).float()

    def freqs_to_spectrum(self, freqs: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
        """
        Convert (freqs, amps) to magnitude spectrum analytically.

        Args:
            freqs: [B, T, n_sines] frequencies in Hz
            amps: [B, T, n_sines] amplitudes

        Returns:
            spectrum: [B, T, n_bins] magnitude spectrum
        """
        B, T, N = freqs.shape
        device = freqs.device

        # Convert freqs to bin indices (continuous)
        bin_indices = freqs * self.n_fft / self.sample_rate  # [B, T, N]

        # Create spectrum by placing Gaussian peaks at each frequency
        # spectrum[bin] = sum over sines of: amp * exp(-(bin - bin_idx)^2 / (2*width^2))

        bins = torch.arange(self.n_bins, device=device).float()  # [n_bins]
        bins = bins.view(1, 1, 1, -1)  # [1, 1, 1, n_bins]

        bin_indices = bin_indices.unsqueeze(-1)  # [B, T, N, 1]
        amps = amps.unsqueeze(-1)  # [B, T, N, 1]

        # Gaussian peaks
        peaks = amps * torch.exp(-0.5 * ((bins - bin_indices) / self.peak_width) ** 2)

        # Sum across sines
        spectrum = peaks.sum(dim=2)  # [B, T, n_bins]

        return spectrum

    def audio_to_spectrum(self, audio: torch.Tensor, n_frames: int) -> torch.Tensor:
        """
        Compute target spectrum from audio.

        Args:
            audio: [B, n_samples]
            n_frames: number of frames to match

        Returns:
            spectrum: [B, n_frames, n_bins]
        """
        B = audio.shape[0]
        hop = audio.shape[-1] // n_frames

        window = torch.hann_window(self.n_fft, device=audio.device)

        stft = torch.stft(
            audio, self.n_fft, hop_length=hop, window=window,
            return_complex=True, pad_mode='reflect'
        )  # [B, n_bins, T_stft]

        mag = stft.abs().transpose(1, 2)  # [B, T_stft, n_bins]

        # Interpolate to match n_frames
        if mag.shape[1] != n_frames:
            mag = F.interpolate(
                mag.transpose(1, 2), size=n_frames, mode='linear'
            ).transpose(1, 2)

        return mag

    def forward(
        self,
        freqs: torch.Tensor,
        amps: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss between predicted sines and target audio.

        NO AUDIO SYNTHESIS - just analytical spectrum comparison!

        Args:
            freqs: [B, T, n_sines] predicted frequencies
            amps: [B, T, n_sines] predicted amplitudes
            target_audio: [B, n_samples] ground truth audio

        Returns:
            loss: scalar
        """
        B, T, N = freqs.shape

        # Predicted spectrum (analytical - no synthesis!)
        pred_spec = self.freqs_to_spectrum(freqs, amps)  # [B, T, n_bins]

        # Target spectrum (from audio)
        target_spec = self.audio_to_spectrum(target_audio, T)  # [B, T, n_bins]

        # Normalize both for fair comparison
        pred_spec = pred_spec / (pred_spec.max(dim=-1, keepdim=True)[0] + 1e-8)
        target_spec = target_spec / (target_spec.max(dim=-1, keepdim=True)[0] + 1e-8)

        # Linear domain loss
        lin_loss = F.l1_loss(pred_spec, target_spec)

        # Log domain loss (perceptually important)
        log_loss = F.l1_loss(
            torch.log1p(pred_spec * 100),
            torch.log1p(target_spec * 100)
        )

        # Mel domain loss (even more perceptual)
        pred_mel = torch.matmul(pred_spec, self.mel_fb.T)
        target_mel = torch.matmul(target_spec, self.mel_fb.T)
        mel_loss = F.l1_loss(
            torch.log1p(pred_mel),
            torch.log1p(target_mel)
        )

        return lin_loss + log_loss + mel_loss


class MultiScaleSpectralLoss(nn.Module):
    """Legacy audio-based spectral loss (slower but more accurate)."""
    def __init__(self, scales: List[int] = [2048, 1024, 512, 256]):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        loss = 0.0
        n_valid = 0
        for n_fft in self.scales:
            if min_len < n_fft:
                continue

            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=pred.device)

            pred_stft = torch.stft(pred, n_fft, hop_length=hop, window=window,
                                   return_complex=True, pad_mode='reflect')
            target_stft = torch.stft(target, n_fft, hop_length=hop, window=window,
                                     return_complex=True, pad_mode='reflect')

            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()

            loss += F.l1_loss(pred_mag, target_mag)
            loss += F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag))
            n_valid += 1

        return loss / max(n_valid, 1)


# ============================================================
# DATASET
# ============================================================

class UnifiedLatentDataset(Dataset):
    """Dataset that preloads everything into RAM to avoid GCS latency."""

    def __init__(
        self,
        manifest_path: str,
        max_samples: Optional[int] = None,
        filter_groups: Optional[List[str]] = None,
        sample_rate: int = 44100,
        target_frames: int = 22,
    ):
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.target_samples = int(sample_rate * 2.0)  # ~2 seconds

        print(f"Loading manifest from {manifest_path}...")
        sys.stdout.flush()
        with open(manifest_path, 'rb') as f:
            raw = f.read()
        print(f"  Read {len(raw) / 1e6:.1f}MB, parsing...")
        sys.stdout.flush()
        data = orjson.loads(raw)
        del raw

        entries = data.get('entries', data)
        if isinstance(entries, dict):
            entries = list(entries.values())
        print(f"  Found {len(entries)} total entries, filtering...")
        sys.stdout.flush()

        items = []
        for entry in entries:
            if not entry.get('has_latent', False):
                continue
            if entry.get('latent_path') is None:
                continue
            if entry.get('audio_path') is None:
                continue
            if filter_groups and entry.get('group') not in filter_groups:
                continue
            items.append(entry)

        if max_samples:
            items = items[:max_samples]

        # PRELOAD everything into RAM using parallel loading
        print(f"  Preloading {len(items)} samples into RAM (parallel)...")
        sys.stdout.flush()

        samples_per_frame = self.target_samples // self.target_frames

        def load_one(item):
            """Load a single sample - runs in thread pool."""
            try:
                # Load latent
                lat_data = torch.load(item['latent_path'], weights_only=True, map_location='cpu')
                if 'latents' in lat_data:
                    latent = lat_data['latents']
                elif 'latent' in lat_data:
                    latent = lat_data['latent']
                else:
                    return None

                # Load audio
                audio, sr = torchaudio.load(item['audio_path'])
                if sr != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0)
                else:
                    audio = audio.squeeze(0)

                # Crop/pad to target length
                C, H, T_lat = latent.shape

                if T_lat < self.target_frames:
                    latent = F.pad(latent, (0, self.target_frames - T_lat))
                    if audio.shape[-1] < self.target_samples:
                        audio = F.pad(audio, (0, self.target_samples - audio.shape[-1]))
                    else:
                        audio = audio[:self.target_samples]
                elif T_lat > self.target_frames:
                    start_frame = (T_lat - self.target_frames) // 2
                    latent = latent[:, :, start_frame:start_frame + self.target_frames]
                    start_sample = start_frame * samples_per_frame
                    end_sample = start_sample + self.target_samples
                    if end_sample <= audio.shape[-1]:
                        audio = audio[start_sample:end_sample]
                    else:
                        audio = audio[-self.target_samples:] if audio.shape[-1] >= self.target_samples else F.pad(audio, (0, self.target_samples - audio.shape[-1]))
                else:
                    if audio.shape[-1] < self.target_samples:
                        audio = F.pad(audio, (0, self.target_samples - audio.shape[-1]))
                    else:
                        audio = audio[:self.target_samples]

                audio = audio / (audio.abs().max() + 1e-8)

                return {
                    'latent': latent,
                    'audio': audio,
                    'group': item.get('group', 'unknown'),
                }
            except:
                return None

        # Parallel loading with 16 threads (GCS benefits from parallelism)
        self.data = []
        loaded = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(load_one, item): i for i, item in enumerate(items)}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    self.data.append(result)
                loaded += 1
                if loaded % 100 == 0:
                    print(f"\r    Loaded {loaded}/{len(items)}...", end="", flush=True)

        print(f"\r    Loaded {len(self.data)} samples into RAM (~{len(self.data) * 0.36:.0f}MB)  ")
        sys.stdout.flush()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch):
    # Data is pre-validated during preload, no errors expected
    return {
        'latent': torch.stack([b['latent'] for b in batch]),
        'audio': torch.stack([b['audio'] for b in batch]),
        'group': [b['group'] for b in batch],
    }


# ============================================================
# TRAINING
# ============================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SineTrainer:
    def __init__(
        self,
        max_sines: int = 512,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        sparsity_weight: float = 0.01,
        sample_rate: int = 44100,
        device: str = 'cuda',
        use_analytical: bool = False,  # Use audio synthesis loss (analytical has DC collapse bug)
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.sparsity_weight = sparsity_weight
        self.use_analytical = use_analytical

        self.pipeline = DCAESinePipeline(
            max_sines=max_sines,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            sample_rate=sample_rate,
        ).to(device)

        # Compile for speed (PyTorch 2.0+)
        try:
            self.pipeline = torch.compile(self.pipeline, mode='reduce-overhead')
            print("  torch.compile enabled")
        except Exception:
            pass  # Fallback if compile not available

        # Choose loss function
        if use_analytical:
            self.loss_fn = AnalyticalSpectrumLoss(sample_rate=sample_rate).to(device)
            loss_type = "ANALYTICAL (no audio synthesis!)"
        else:
            self.loss_fn = MultiScaleSpectralLoss().to(device)
            loss_type = "audio synthesis + STFT"

        params = sum(p.numel() for p in self.pipeline.parameters())
        print(f"\nSineTrainer (Sparse + Deep):")
        print(f"  Pipeline: z_dcae [8,16,T] → deep encoder → separate heads → sparse sines")
        print(f"  Loss: {loss_type}")
        print(f"  Max sines: {max_sines} (learns to use minimum needed)")
        print(f"  Sparsity weight: {sparsity_weight}")
        print(f"  Params: {params:,}")
        print(f"  Hidden dim: {hidden_dim}, Blocks: {n_blocks}")

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda')

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()

        latent = batch['latent'].to(self.device)
        audio_gt = batch['audio'].to(self.device)

        # Mixed precision forward pass
        with torch.amp.autocast('cuda'):
            B, C, H, T = latent.shape
            z = latent.permute(0, 3, 1, 2).reshape(B, T, C * H)  # [B, T, 128]
            params = self.pipeline.mapper(z)

            freqs = params['freqs']  # [B, T, max_sines]
            amps = params['amps']
            phases = params['phases']

            if self.use_analytical:
                # FAST: No audio synthesis, just spectrum comparison
                recon_loss = self.loss_fn(freqs, amps, audio_gt)
            else:
                # Full audio synthesis
                audio_pred = self.pipeline.synth(freqs, amps, phases, audio_gt.shape[-1])
                recon_loss = self.loss_fn(audio_pred, audio_gt)

            # Sparsity loss
            sparsity_loss = amps.mean()
            loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Mixed precision backward
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        # Count active sines
        n_active = (amps > 0.1).float().sum(dim=-1).mean().item()

        return {
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'n_active': n_active,
        }

    def train(self, dataloader, n_epochs: int = 100, lr: float = 1e-3, save_dir: Optional[str] = None):
        optimizer = torch.optim.AdamW(self.pipeline.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print("\n" + "=" * 60)
        print("Training: z_dcae → Sines (per-frame, temporal preserved)")
        print("=" * 60)

        for epoch in range(n_epochs):
            self.pipeline.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_sparsity = 0.0
            epoch_active = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                metrics = self.train_step(batch, optimizer)
                epoch_loss += metrics['loss']
                epoch_recon += metrics['recon']
                epoch_sparsity += metrics['sparsity']
                epoch_active += metrics['n_active']
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_recon = epoch_recon / max(n_batches, 1)
            avg_active = epoch_active / max(n_batches, 1)

            print(f"Epoch {epoch:4d}: loss={avg_loss:.4f} recon={avg_recon:.4f} active_sines={avg_active:.0f} | lr={scheduler.get_last_lr()[0]:.2e}")

            if save_dir and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.pipeline.state_dict(),
                    'loss': avg_loss,
                }, str(save_path / "best_model.pt"))

            if epoch % 20 == 0:
                clear_memory()

        if save_dir:
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': self.pipeline.state_dict(),
                'loss': avg_loss,
            }, str(save_path / "final_model.pt"))
            print(f"\nSaved to {save_dir}")

        print(f"Training complete! Best loss: {best_loss:.4f}")

        # Final sparsity analysis
        print("\n" + "=" * 60)
        print("SPARSITY ANALYSIS")
        print("=" * 60)

        self.pipeline.eval()
        with torch.no_grad():
            # Get a batch to analyze
            for batch in dataloader:
                latent = batch['latent'].to(self.device)
                out = self.pipeline(latent, n_samples=88200)
                amps = out['amps']

                for thresh in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    n_active = (amps > thresh).float().sum(dim=-1).mean().item()
                    print(f"  Active sines (amp > {thresh}): {n_active:.1f} / {self.pipeline.max_sines}")
                break

        # Analyze what the network learned
        print("\n" + "=" * 60)
        print("LEARNED STRUCTURE ANALYSIS")
        print("=" * 60)
        analysis = self.pipeline.mapper.analyze_learned_structure()

        # Check for differentiation
        freq_set = set(analysis['freq_top_dims'][:10])
        amp_set = set(analysis['amp_top_dims'][:10])
        phase_set = set(analysis['phase_top_dims'][:10])

        overlap_fa = len(freq_set & amp_set)
        overlap_fp = len(freq_set & phase_set)
        overlap_ap = len(amp_set & phase_set)

        print(f"\n  Role differentiation (lower = more specialized):")
        print(f"    Freq-Amp overlap: {overlap_fa}/10")
        print(f"    Freq-Phase overlap: {overlap_fp}/10")
        print(f"    Amp-Phase overlap: {overlap_ap}/10")

        print(f"\n  Top dims for FREQUENCY:")
        for d in analysis['freq_top_dims'][:10]:
            c, h = d // 16, d % 16
            print(f"    dim {d:3d} → channel {c}, height {h}")

        print(f"\n  Top dims for AMPLITUDE:")
        for d in analysis['amp_top_dims'][:10]:
            c, h = d // 16, d % 16
            print(f"    dim {d:3d} → channel {c}, height {h}")

        print(f"\n  Top dims for PHASE:")
        for d in analysis['phase_top_dims'][:10]:
            c, h = d // 16, d % 16
            print(f"    dim {d:3d} → channel {c}, height {h}")

        # Save analysis
        if save_dir:
            import json
            with open(save_path / "learned_structure.json", 'w') as f:
                json.dump({
                    'freq_top_dims': analysis['freq_top_dims'],
                    'amp_top_dims': analysis['amp_top_dims'],
                    'phase_top_dims': analysis['phase_top_dims'],
                    'overlap_freq_amp': overlap_fa,
                    'overlap_freq_phase': overlap_fp,
                    'overlap_amp_phase': overlap_ap,
                }, f, indent=2)

        return self.pipeline


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='/home/arlo/gcs-bucket/Manifests/unified_manifest.json')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_sines', type=int, default=512, help='Max sine pool (learns to use fewer)')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--sparsity', type=float, default=0.01, help='Sparsity weight (higher = fewer sines)')
    parser.add_argument('--groups', type=str, default=None)
    parser.add_argument('--analytical', action='store_true', help='Use fast analytical spectrum (has DC collapse bug, not recommended)')
    args = parser.parse_args()

    print("=" * 60)
    print("DCAE → Sparse Sines Training")
    print("Deep network learns minimum sines needed")
    print("=" * 60)
    sys.stdout.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    sys.stdout.flush()

    filter_groups = args.groups.split(',') if args.groups else None

    print(f"\nLoading dataset...")
    sys.stdout.flush()

    dataset = UnifiedLatentDataset(
        manifest_path=args.manifest,
        max_samples=args.max_samples,
        filter_groups=filter_groups,
    )

    print(f"Creating dataloader with {len(dataset)} samples...")
    sys.stdout.flush()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Dataloader ready: {len(dataloader)} batches")
    sys.stdout.flush()

    trainer = SineTrainer(
        max_sines=args.max_sines,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        sparsity_weight=args.sparsity,
        sample_rate=44100,
        device=device,
        use_analytical=args.analytical,
    )

    save_dir = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/dcae_sparse_sines"

    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
