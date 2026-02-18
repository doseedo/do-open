#!/usr/bin/env python3
"""
Probe SAMI's Learned Ops to Reverse Engineer Explicit DSP

SAMI discovered:
  - Op 10: coarse (noise-like, 0.993 activation at high noise)
  - Ops 0-5, 7, 8, 11: fine (tonal, high activation at low noise)
  - Ops 6, 9: medium (noise variants)

This script probes each op to understand:
  - What frequency content?
  - What envelope shape?
  - How does output change with z input?
  - What explicit DSP would replicate it?
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 44100
DURATION = 2.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# Add SAMI path
SAMI_PATH = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch"
if SAMI_PATH not in sys.path:
    sys.path.insert(0, SAMI_PATH)


# ============================================================
# AUDIO ANALYSIS FUNCTIONS
# ============================================================

def estimate_f0(audio: torch.Tensor, sr: int = SAMPLE_RATE,
                fmin: float = 50, fmax: float = 2000) -> Optional[float]:
    """Estimate fundamental frequency using autocorrelation."""
    if audio.dim() > 1:
        audio = audio.squeeze()
    audio = audio.cpu().numpy()

    # Autocorrelation
    n = len(audio)
    # Limit to first 0.5s for speed
    audio = audio[:min(n, sr // 2)]
    n = len(audio)

    corr = np.correlate(audio, audio, mode='full')
    corr = corr[n-1:]  # Keep positive lags

    # Find peaks in valid f0 range
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    if max_lag >= len(corr):
        max_lag = len(corr) - 1

    search_region = corr[min_lag:max_lag]
    if len(search_region) == 0:
        return None

    peak_idx = np.argmax(search_region) + min_lag

    # Check if it's a real peak (above threshold)
    if corr[peak_idx] < 0.1 * corr[0]:
        return None

    f0 = sr / peak_idx
    return float(f0)


def compute_spectral_flatness(audio: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """
    Spectral flatness (Wiener entropy).
    0 = pure tone, 1 = white noise.
    """
    if audio.dim() > 1:
        audio = audio.squeeze()

    n_fft = 2048
    hop = 512
    window = torch.hann_window(n_fft, device=audio.device)

    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = stft.abs() + 1e-10

    # Geometric mean / arithmetic mean per frame
    log_mag = torch.log(mag)
    geo_mean = torch.exp(log_mag.mean(dim=0))
    arith_mean = mag.mean(dim=0)

    flatness = geo_mean / (arith_mean + 1e-10)
    return float(flatness.mean())


def compute_spectral_centroid(audio: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """Spectral centroid - brightness measure."""
    if audio.dim() > 1:
        audio = audio.squeeze()

    n_fft = 2048
    hop = 512
    window = torch.hann_window(n_fft, device=audio.device)

    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = stft.abs()

    freqs = torch.fft.rfftfreq(n_fft, 1/sr).to(audio.device)

    # Weighted average frequency
    centroid = (freqs.unsqueeze(-1) * mag).sum(dim=0) / (mag.sum(dim=0) + 1e-10)
    return float(centroid.mean())


def analyze_harmonics(audio: torch.Tensor, f0: float, sr: int = SAMPLE_RATE,
                      n_harmonics: int = 16) -> np.ndarray:
    """Analyze harmonic amplitudes given f0."""
    if audio.dim() > 1:
        audio = audio.squeeze()

    n_fft = 4096
    window = torch.hann_window(n_fft, device=audio.device)

    # Zero-pad
    if len(audio) < n_fft:
        audio = F.pad(audio, (0, n_fft - len(audio)))

    stft = torch.stft(audio[:n_fft], n_fft=n_fft, hop_length=n_fft,
                      window=window, return_complex=True)
    mag = stft.abs().squeeze()

    freqs = torch.fft.rfftfreq(n_fft, 1/sr).to(audio.device)

    harmonic_amps = []
    for h in range(1, n_harmonics + 1):
        target_freq = f0 * h
        # Find nearest bin
        idx = (freqs - target_freq).abs().argmin()
        # Average over small window
        window_size = 3
        start = max(0, idx - window_size)
        end = min(len(mag), idx + window_size + 1)
        amp = mag[start:end].max()
        harmonic_amps.append(float(amp))

    return np.array(harmonic_amps)


def extract_envelope(audio: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Extract amplitude envelope."""
    if audio.dim() > 1:
        audio = audio.squeeze()

    # Hilbert-like envelope via low-pass filtered absolute value
    abs_audio = audio.abs()

    # Simple moving average
    kernel_size = int(sr * 0.01)  # 10ms window
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = torch.ones(1, 1, kernel_size, device=audio.device) / kernel_size

    envelope = F.conv1d(abs_audio.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2)
    return envelope.squeeze()


def classify_op_type(flatness: float, f0: Optional[float], centroid: float) -> str:
    """Classify op as noise, tonal, or transient."""
    if flatness > 0.7:
        return "noise"
    elif flatness < 0.3 and f0 is not None:
        return "tonal"
    elif flatness > 0.4 and flatness < 0.7:
        return "mixed"
    else:
        return "unknown"


# ============================================================
# SAMI MODEL LOADING (matches test_sami_op_discovery.py)
# ============================================================

class LearnableOperation(nn.Module):
    """Learnable audio generator conditioned on factorized z."""

    def __init__(self, z_dim: int = 64, hidden_dim: int = 128, n_samples: int = N_SAMPLES):
        super().__init__()
        self.n_samples = n_samples

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.to_freq = nn.Linear(hidden_dim, 32)
        self.to_amp = nn.Linear(hidden_dim, 32)
        self.to_phase = nn.Linear(hidden_dim, 32)

        t = torch.linspace(0, DURATION, n_samples)
        self.register_buffer('t', t)

        freqs = torch.linspace(50, 8000, 32)
        self.register_buffer('freq_basis', freqs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)

        B = z.shape[0]
        h = self.net(z)

        freq_weights = torch.softmax(self.to_freq(h), dim=-1)
        amp_envelope = torch.sigmoid(self.to_amp(h))
        phase_offset = self.to_phase(h)

        audio = torch.zeros(B, self.n_samples, device=z.device)

        for i in range(32):
            freq = self.freq_basis[i]
            wave = torch.sin(2 * np.pi * freq * self.t + phase_offset[:, i:i+1])
            audio += freq_weights[:, i:i+1] * amp_envelope[:, i:i+1] * wave

        return audio


class LearnableNoiseOperation(nn.Module):
    """Learnable noise generator."""

    def __init__(self, z_dim: int = 64, hidden_dim: int = 64, n_samples: int = N_SAMPLES):
        super().__init__()
        self.n_samples = n_samples

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.to_spectral_shape = nn.Linear(hidden_dim, 64)
        self.to_temporal_env = nn.Linear(hidden_dim, 32)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)

        B = z.shape[0]
        h = self.net(z)

        spectral_shape = torch.sigmoid(self.to_spectral_shape(h))
        temporal_env = torch.sigmoid(self.to_temporal_env(h))

        noise = torch.randn(B, self.n_samples, device=z.device)

        noise_fft = torch.fft.rfft(noise)
        shape_interp = F.interpolate(
            spectral_shape.unsqueeze(1),
            size=noise_fft.shape[-1],
            mode='linear',
            align_corners=True
        ).squeeze(1)
        noise_fft = noise_fft * shape_interp
        shaped_noise = torch.fft.irfft(noise_fft, n=self.n_samples)

        env_interp = F.interpolate(
            temporal_env.unsqueeze(1),
            size=self.n_samples,
            mode='linear',
            align_corners=True
        ).squeeze(1)

        return shaped_noise * env_interp


class SAMIEncoder(nn.Module):
    """SAMI encoder for noise-conditioned audio encoding."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=32),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=8, padding=16),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=8),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=4),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )

        self.noise_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 256),
        )

        self.to_mu = nn.Linear(256 * 16 + 256, latent_dim)
        self.to_logvar = nn.Linear(256 * 16 + 256, latent_dim)

    def forward(self, audio: torch.Tensor, noise_level: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        B = audio.shape[0]
        h = self.conv(audio)
        h = h.view(B, -1)
        t_embed = self.noise_embed(noise_level)
        h = torch.cat([h, t_embed], dim=-1)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class SAMIOpDiscoverer(nn.Module):
    """SAMI model with separate tonal and noise ops."""

    def __init__(self, latent_dim: int = 64, n_ops: int = 12, n_timesteps: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_ops = n_ops
        self.n_timesteps = n_timesteps

        self.encoder = SAMIEncoder(latent_dim=latent_dim)

        betas = torch.linspace(1e-4, 0.02, n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('alpha_bar', alpha_bar)

        self.tonal_ops = nn.ModuleList([
            LearnableOperation(latent_dim) for _ in range(n_ops // 2)
        ])
        self.noise_ops = nn.ModuleList([
            LearnableNoiseOperation(latent_dim) for _ in range(n_ops // 2)
        ])

        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_ops),
        )

        self.mix_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_ops),
        )


def load_sami_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained SAMI model."""

    # Create model with same architecture
    model = SAMIOpDiscoverer(latent_dim=64, n_ops=12, n_timesteps=100).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def decode_single_op(model, z: torch.Tensor, op_idx: int,
                     device: str = 'cuda') -> torch.Tensor:
    """Decode using only a single op."""
    model.eval()

    with torch.no_grad():
        # Get the specific op
        n_tonal = len(model.tonal_ops)
        if op_idx < n_tonal:
            op = model.tonal_ops[op_idx]
        else:
            op = model.noise_ops[op_idx - n_tonal]

        audio = op(z)

    return audio


# ============================================================
# PROBING
# ============================================================

def probe_op(model, op_idx: int, n_samples: int = 20,
             device: str = 'cuda') -> Dict:
    """Probe a single op with varied z inputs."""

    results = {
        'op_idx': op_idx,
        'f0s': [],
        'flatnesses': [],
        'centroids': [],
        'classifications': [],
        'harmonic_patterns': [],
    }

    for i in range(n_samples):
        # Generate varied z (use model's latent dim)
        z = torch.randn(1, model.latent_dim, device=device)

        # Decode single op
        audio = decode_single_op(model, z, op_idx, device)
        audio = audio.squeeze()

        # Normalize
        audio = audio / (audio.abs().max() + 1e-8)

        # Analyze
        f0 = estimate_f0(audio)
        flatness = compute_spectral_flatness(audio)
        centroid = compute_spectral_centroid(audio)
        classification = classify_op_type(flatness, f0, centroid)

        results['f0s'].append(f0)
        results['flatnesses'].append(flatness)
        results['centroids'].append(centroid)
        results['classifications'].append(classification)

        # If tonal, analyze harmonics
        if f0 is not None and flatness < 0.5:
            harmonics = analyze_harmonics(audio, f0)
            results['harmonic_patterns'].append(harmonics)

    return results


def summarize_op(results: Dict) -> Dict:
    """Summarize probing results for an op."""

    flatnesses = np.array(results['flatnesses'])
    centroids = np.array(results['centroids'])
    f0s = [f for f in results['f0s'] if f is not None]
    classifications = results['classifications']

    # Majority classification
    from collections import Counter
    class_counts = Counter(classifications)
    majority_class = class_counts.most_common(1)[0][0]

    summary = {
        'op_idx': results['op_idx'],
        'avg_flatness': float(flatnesses.mean()),
        'std_flatness': float(flatnesses.std()),
        'avg_centroid': float(centroids.mean()),
        'std_centroid': float(centroids.std()),
        'f0_detected_rate': len(f0s) / len(results['f0s']),
        'avg_f0': float(np.mean(f0s)) if f0s else None,
        'f0_std': float(np.std(f0s)) if len(f0s) > 1 else None,
        'majority_class': majority_class,
        'class_distribution': dict(class_counts),
    }

    # Harmonic pattern summary
    if results['harmonic_patterns']:
        patterns = np.array(results['harmonic_patterns'])
        avg_pattern = patterns.mean(axis=0)
        # Normalize
        avg_pattern = avg_pattern / (avg_pattern.max() + 1e-10)
        summary['avg_harmonic_pattern'] = avg_pattern.tolist()

        # Find dominant harmonics
        dominant = np.where(avg_pattern > 0.3)[0] + 1  # +1 for 1-indexed harmonics
        summary['dominant_harmonics'] = dominant.tolist()

    return summary


def infer_dsp_type(summary: Dict) -> str:
    """Infer what explicit DSP would replicate this op."""

    flatness = summary['avg_flatness']
    f0_rate = summary['f0_detected_rate']
    majority = summary['majority_class']

    if majority == 'noise' or flatness > 0.7:
        # Noise-like: filtered_noise
        centroid = summary['avg_centroid']
        if centroid < 500:
            return "filtered_noise(lowpass, cutoff~500Hz)"
        elif centroid < 2000:
            return "filtered_noise(bandpass, center~1kHz)"
        else:
            return "filtered_noise(highpass or white)"

    elif majority == 'tonal' and f0_rate > 0.5:
        # Tonal: harmonic stack
        f0 = summary.get('avg_f0', 200)
        f0_std = summary.get('f0_std', 0)

        if f0_std and f0_std > 50:
            f0_control = f"f0=z-controlled({f0:.0f}±{f0_std:.0f}Hz)"
        else:
            f0_control = f"f0~{f0:.0f}Hz"

        harmonics = summary.get('dominant_harmonics', [1])
        if len(harmonics) == 1 and harmonics[0] == 1:
            return f"sine_osc({f0_control})"
        else:
            return f"harmonic_stack({f0_control}, harmonics={harmonics})"

    elif majority == 'mixed':
        return "mixed: harmonic_stack + filtered_noise"

    else:
        return "unknown (needs more analysis)"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Probing SAMI's Learned Ops to Infer Explicit DSP")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Find SAMI checkpoint (correct path)
    checkpoint_path = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/sami_op_discovery/sami_op_discoverer.pt")

    if not checkpoint_path.exists():
        print(f"\nERROR: No checkpoint found at {checkpoint_path}")
        print("Run test_sami_op_discovery.py first to train SAMI.")
        return

    print(f"\nLoading SAMI from {checkpoint_path}...")
    model = load_sami_model(str(checkpoint_path), device)
    n_tonal = len(model.tonal_ops)
    n_noise = len(model.noise_ops)
    print(f"  Loaded model with {model.n_ops} ops ({n_tonal} tonal, {n_noise} noise)")

    # Output directory
    output_dir = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/sami_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe each op
    print("\n" + "=" * 60)
    print("Probing Each Op")
    print("=" * 60)

    all_summaries = []

    # Probe all ops in order (0-5 tonal, 6-11 noise)
    for op_idx in range(model.n_ops):
        op_type_str = "tonal" if op_idx < n_tonal else "noise"
        print(f"\n  Probing Op {op_idx} ({op_type_str})...")

        # Probe with 30 varied z inputs
        results = probe_op(model, op_idx, n_samples=30, device=device)
        summary = summarize_op(results)
        summary['op_type_design'] = op_type_str  # What it was designed as
        dsp_type = infer_dsp_type(summary)
        summary['inferred_dsp'] = dsp_type

        all_summaries.append(summary)

        # Print summary
        print(f"    Flatness: {summary['avg_flatness']:.3f} ± {summary['std_flatness']:.3f}")
        print(f"    Centroid: {summary['avg_centroid']:.0f} ± {summary['std_centroid']:.0f} Hz")
        print(f"    F0 detected: {summary['f0_detected_rate']:.0%}")
        if summary['avg_f0']:
            print(f"    Avg F0: {summary['avg_f0']:.1f} Hz")
        print(f"    Classification: {summary['majority_class']} ({summary['class_distribution']})")
        print(f"    → Inferred DSP: {dsp_type}")

        # Save sample audio
        z = torch.randn(1, model.latent_dim, device=device)
        audio = decode_single_op(model, z, op_idx, device)
        audio = audio / (audio.abs().max() + 1e-8) * 0.8
        torchaudio.save(str(output_dir / f"op_{op_idx:02d}_{op_type_str}_sample.wav"), audio.cpu(), SAMPLE_RATE)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: SAMI Ops → Explicit DSP")
    print("=" * 60)

    print("\n  Op | Design  | Learned | Flatness | F0 Rate | Inferred DSP")
    print("  " + "-" * 80)

    for s in all_summaries:
        op = s['op_idx']
        design = s.get('op_type_design', '?')[:7]
        learned = s['majority_class'][:7]
        flat = s['avg_flatness']
        f0r = s['f0_detected_rate']
        dsp = s['inferred_dsp'][:35]
        match = "✓" if design == learned else "✗"
        print(f"  {op:2d} | {design:7s} | {learned:7s} {match} | {flat:.3f}    | {f0r:.0%}     | {dsp}")

    # Group by learned type
    print("\n" + "=" * 60)
    print("GROUPED BY LEARNED TYPE")
    print("=" * 60)

    noise_ops = [s for s in all_summaries if s['majority_class'] == 'noise']
    tonal_ops = [s for s in all_summaries if s['majority_class'] == 'tonal']
    mixed_ops = [s for s in all_summaries if s['majority_class'] == 'mixed']
    unknown_ops = [s for s in all_summaries if s['majority_class'] == 'unknown']

    print(f"\n  NOISE ops ({len(noise_ops)}): {[s['op_idx'] for s in noise_ops]}")
    for s in noise_ops:
        print(f"    Op {s['op_idx']} (designed: {s.get('op_type_design', '?')}): {s['inferred_dsp']}")

    print(f"\n  TONAL ops ({len(tonal_ops)}): {[s['op_idx'] for s in tonal_ops]}")
    for s in tonal_ops:
        print(f"    Op {s['op_idx']} (designed: {s.get('op_type_design', '?')}): {s['inferred_dsp']}")
        if 'dominant_harmonics' in s:
            print(f"      Dominant harmonics: {s['dominant_harmonics']}")
        if s.get('avg_f0'):
            print(f"      Avg F0: {s['avg_f0']:.1f} Hz (std: {s.get('f0_std', 0):.1f})")

    print(f"\n  MIXED ops ({len(mixed_ops)}): {[s['op_idx'] for s in mixed_ops]}")
    for s in mixed_ops:
        print(f"    Op {s['op_idx']} (designed: {s.get('op_type_design', '?')}): {s['inferred_dsp']}")

    if unknown_ops:
        print(f"\n  UNKNOWN ops ({len(unknown_ops)}): {[s['op_idx'] for s in unknown_ops]}")

    # Generate explicit DSP specification
    print("\n" + "=" * 60)
    print("INFERRED EXPLICIT DSP SPECIFICATION")
    print("=" * 60)

    print("\n  Based on SAMI analysis, explicit DSP should have:")
    print(f"  - {len(noise_ops)} noise generators")
    print(f"  - {len(tonal_ops)} harmonic oscillators")
    print(f"  - {len(mixed_ops)} mixed (harmonic + noise) generators")

    # Save results
    import json
    with open(output_dir / "op_analysis.json", 'w') as f:
        # Convert numpy arrays to lists
        for s in all_summaries:
            if 'avg_harmonic_pattern' in s:
                s['avg_harmonic_pattern'] = list(s['avg_harmonic_pattern'])
        json.dump(all_summaries, f, indent=2)

    print(f"\n\nResults saved to: {output_dir}")
    print("\nNEXT STEP: Use these inferred DSP types to build explicit bridge")
    print("  - If tonal ops dominate: harmonic_stack(f0, harmonics)")
    print("  - If noise ops dominate: filtered_noise(spectral_shape)")
    print("  - If mixed: harmonic_stack + filtered_noise")


if __name__ == "__main__":
    main()
