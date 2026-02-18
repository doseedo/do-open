#!/usr/bin/env python3
"""
Analyze per-frame DCAE latent structure using SAMI-style noise stability.

Goal: Discover which of the 128 per-frame dims (8 channels × 16 height) are:
  - Coarse (stable across noise) → fundamental features like pitch
  - Fine (noise-sensitive) → texture, transients

This gives us structure that applies to EVERY frame, not mixed with time.

Method:
  1. Load z_dcae frames [8, 16] = 128 dims
  2. Add noise at different levels
  3. Measure which dims change vs stay stable
  4. Cluster into coarse/fine groups
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import orjson
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 44100
MANIFEST_PATH = "/home/arlo/gcs-bucket/Manifests/unified_manifest.json"
OUTPUT_DIR = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/perframe_structure")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# DATA LOADING
# ============================================================

def load_frames(manifest_path: str, max_files: int = 500, max_frames_per_file: int = 10):
    """Load individual frames from DCAE latents."""
    print(f"Loading frames from {manifest_path}...")

    with open(manifest_path, 'rb') as f:
        data = orjson.loads(f.read())

    entries = data.get('entries', data)

    frames = []
    files_loaded = 0

    for entry in entries:
        if not entry.get('has_latent', False):
            continue
        if entry.get('latent_path') is None:
            continue

        try:
            lat_data = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
            latent = lat_data.get('latents', lat_data.get('latent'))
            if latent is None:
                continue

            # latent: [C, H, T] = [8, 16, T]
            C, H, T = latent.shape

            # Sample frames from this file
            if T > max_frames_per_file:
                indices = torch.randperm(T)[:max_frames_per_file]
            else:
                indices = torch.arange(T)

            for t in indices:
                frame = latent[:, :, t].flatten()  # [128]
                frames.append(frame)

            files_loaded += 1
            if files_loaded >= max_files:
                break

        except Exception as e:
            continue

    frames = torch.stack(frames)  # [N, 128]
    print(f"  Loaded {len(frames)} frames from {files_loaded} files")
    return frames


# ============================================================
# NOISE STABILITY ANALYSIS
# ============================================================

def analyze_noise_stability(
    frames: torch.Tensor,
    noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    n_trials: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Analyze which dims are stable vs sensitive to noise.

    For each noise level:
      1. Add Gaussian noise to frames
      2. Measure change in each dim
      3. Dims that change little = coarse (stable)
      4. Dims that change a lot = fine (sensitive)
    """
    print("\nAnalyzing noise stability...")

    N, D = frames.shape  # [N, 128]
    device = frames.device

    # Normalize frames for fair comparison
    frame_mean = frames.mean(dim=0)
    frame_std = frames.std(dim=0) + 1e-8
    frames_norm = (frames - frame_mean) / frame_std

    # Track sensitivity per dim per noise level
    sensitivity = torch.zeros(len(noise_levels), D)

    for i, noise_level in enumerate(noise_levels):
        print(f"  Noise level {noise_level}...")

        dim_changes = torch.zeros(D)

        for trial in range(n_trials):
            # Add noise
            noise = torch.randn_like(frames_norm) * noise_level
            frames_noisy = frames_norm + noise

            # Measure change per dim (L2 across samples)
            change = (frames_noisy - frames_norm).pow(2).mean(dim=0).sqrt()
            dim_changes += change

        dim_changes /= n_trials
        sensitivity[i] = dim_changes

    # Average sensitivity across noise levels (weighted toward higher noise)
    weights = torch.tensor(noise_levels)
    weights = weights / weights.sum()
    avg_sensitivity = (sensitivity * weights.unsqueeze(1)).sum(dim=0)

    # Classify dims
    median_sens = avg_sensitivity.median()
    coarse_mask = avg_sensitivity < median_sens
    fine_mask = avg_sensitivity >= median_sens

    coarse_dims = torch.where(coarse_mask)[0].tolist()
    fine_dims = torch.where(fine_mask)[0].tolist()

    # Sort by sensitivity
    sorted_indices = torch.argsort(avg_sensitivity)
    most_stable = sorted_indices[:20].tolist()
    most_sensitive = sorted_indices[-20:].tolist()

    return {
        'sensitivity': avg_sensitivity,
        'sensitivity_per_level': sensitivity,
        'coarse_dims': coarse_dims,
        'fine_dims': fine_dims,
        'most_stable': most_stable,
        'most_sensitive': most_sensitive,
        'noise_levels': noise_levels,
    }


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def analyze_dim_correlations(frames: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Analyze correlations between dims to find clusters.

    Dims that are highly correlated might represent the same underlying feature.
    """
    print("\nAnalyzing dim correlations...")

    N, D = frames.shape

    # Normalize
    frames_norm = (frames - frames.mean(dim=0)) / (frames.std(dim=0) + 1e-8)

    # Correlation matrix
    corr = torch.mm(frames_norm.T, frames_norm) / N  # [D, D]

    # Find highly correlated dim pairs
    corr_upper = torch.triu(corr, diagonal=1)
    high_corr_threshold = 0.7
    high_corr_pairs = torch.where(corr_upper.abs() > high_corr_threshold)

    pairs = list(zip(high_corr_pairs[0].tolist(), high_corr_pairs[1].tolist()))

    # Find dim clusters (dims correlated with many others)
    n_correlations = (corr.abs() > high_corr_threshold).sum(dim=1) - 1  # -1 for self
    hub_dims = torch.argsort(n_correlations, descending=True)[:10].tolist()

    return {
        'correlation_matrix': corr,
        'high_corr_pairs': pairs[:50],  # Top 50 pairs
        'hub_dims': hub_dims,
        'n_correlations': n_correlations,
    }


# ============================================================
# VARIANCE ANALYSIS
# ============================================================

def analyze_variance(frames: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Analyze variance per dim across frames.

    High variance dims = encode variable features (pitch, dynamics)
    Low variance dims = encode stable features (instrument identity?)
    """
    print("\nAnalyzing variance...")

    variance = frames.var(dim=0)

    sorted_indices = torch.argsort(variance)
    low_var_dims = sorted_indices[:20].tolist()
    high_var_dims = sorted_indices[-20:].tolist()

    return {
        'variance': variance,
        'low_var_dims': low_var_dims,
        'high_var_dims': high_var_dims,
    }


# ============================================================
# SPATIAL STRUCTURE ANALYSIS
# ============================================================

def analyze_spatial_structure(frames: torch.Tensor) -> Dict:
    """
    Analyze structure in the [8, 16] spatial layout.

    The 128 dims come from [8 channels, 16 height].
    - Channels might encode different frequency bands
    - Height might encode different features within a band
    """
    print("\nAnalyzing spatial structure...")

    N = frames.shape[0]

    # Reshape to [N, 8, 16]
    frames_spatial = frames.view(N, 8, 16)

    # Variance per channel (average over height)
    channel_var = frames_spatial.var(dim=0).mean(dim=1)  # [8]

    # Variance per height (average over channel)
    height_var = frames_spatial.var(dim=0).mean(dim=0)  # [16]

    # Cross-channel correlation
    channel_means = frames_spatial.mean(dim=2)  # [N, 8]
    channel_corr = torch.corrcoef(channel_means.T)  # [8, 8]

    return {
        'channel_variance': channel_var,
        'height_variance': height_var,
        'channel_correlation': channel_corr,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Per-Frame DCAE Latent Structure Analysis")
    print("Discovering coarse vs fine dims in [8, 16] = 128 dims")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load frames
    frames = load_frames(MANIFEST_PATH, max_files=500, max_frames_per_file=10)

    # Run analyses
    noise_results = analyze_noise_stability(frames)
    corr_results = analyze_dim_correlations(frames)
    var_results = analyze_variance(frames)
    spatial_results = analyze_spatial_structure(frames)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  Total frames analyzed: {len(frames)}")
    print(f"  Frame dimension: {frames.shape[1]} (8 channels × 16 height)")

    print(f"\n  NOISE STABILITY:")
    print(f"    Coarse (stable) dims: {len(noise_results['coarse_dims'])}")
    print(f"    Fine (sensitive) dims: {len(noise_results['fine_dims'])}")
    print(f"    Most stable (top 10): {noise_results['most_stable'][:10]}")
    print(f"    Most sensitive (top 10): {noise_results['most_sensitive'][-10:]}")

    print(f"\n  VARIANCE:")
    print(f"    Low variance dims (stable features): {var_results['low_var_dims'][:10]}")
    print(f"    High variance dims (variable features): {var_results['high_var_dims'][-10:]}")

    print(f"\n  CORRELATIONS:")
    print(f"    Hub dims (correlated with many): {corr_results['hub_dims']}")
    print(f"    High-correlation pairs: {len(corr_results['high_corr_pairs'])}")

    print(f"\n  SPATIAL STRUCTURE [8 channels × 16 height]:")
    print(f"    Channel variance: {spatial_results['channel_variance'].tolist()}")
    print(f"    Height variance (first 8): {spatial_results['height_variance'][:8].tolist()}")

    # Map dims back to (channel, height)
    def dim_to_ch(dim):
        return (dim // 16, dim % 16)

    print(f"\n  MOST STABLE DIMS (spatial positions):")
    for d in noise_results['most_stable'][:10]:
        c, h = dim_to_ch(d)
        print(f"    dim {d:3d} → channel {c}, height {h}")

    print(f"\n  MOST SENSITIVE DIMS (spatial positions):")
    for d in noise_results['most_sensitive'][-10:]:
        c, h = dim_to_ch(d)
        print(f"    dim {d:3d} → channel {c}, height {h}")

    # Save results
    results = {
        'coarse_dims': noise_results['coarse_dims'],
        'fine_dims': noise_results['fine_dims'],
        'most_stable': noise_results['most_stable'],
        'most_sensitive': noise_results['most_sensitive'],
        'low_var_dims': var_results['low_var_dims'],
        'high_var_dims': var_results['high_var_dims'],
        'hub_dims': corr_results['hub_dims'],
        'channel_variance': spatial_results['channel_variance'].tolist(),
        'height_variance': spatial_results['height_variance'].tolist(),
    }

    import json
    with open(OUTPUT_DIR / "perframe_structure.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save tensors
    torch.save({
        'sensitivity': noise_results['sensitivity'],
        'variance': var_results['variance'],
        'correlation_matrix': corr_results['correlation_matrix'],
        'channel_correlation': spatial_results['channel_correlation'],
    }, OUTPUT_DIR / "perframe_analysis.pt")

    print(f"\n  Results saved to: {OUTPUT_DIR}")

    # Generate code snippet for StructuredPerFrameMapper
    print("\n" + "=" * 60)
    print("GENERATED CODE FOR STRUCTURED MAPPER")
    print("=" * 60)

    stable_dims = noise_results['most_stable'][:32]
    sensitive_dims = noise_results['most_sensitive'][-32:]

    print(f"""
class StructuredPerFrameMapper(nn.Module):
    '''
    Per-frame mapping using discovered structure.
    Stable dims → frequency (pitch)
    Sensitive dims → amplitude (dynamics, texture)
    '''

    # Discovered from noise stability analysis
    STABLE_DIMS = {stable_dims}
    SENSITIVE_DIMS = {sensitive_dims}

    def __init__(self, n_sines=128):
        super().__init__()
        self.freq_head = nn.Linear(len(self.STABLE_DIMS), n_sines)
        self.amp_head = nn.Linear(len(self.SENSITIVE_DIMS), n_sines)
        self.phase_head = nn.Linear(128, n_sines)  # Use all dims

    def forward(self, z_frame):  # [B, 128]
        z_stable = z_frame[:, self.STABLE_DIMS]
        z_sensitive = z_frame[:, self.SENSITIVE_DIMS]

        freqs = torch.sigmoid(self.freq_head(z_stable)) * 22050
        amps = torch.sigmoid(self.amp_head(z_sensitive))
        phases = torch.sigmoid(self.phase_head(z_frame)) * 2 * np.pi

        return freqs, amps, phases
""")

    print("\nDone!")


if __name__ == "__main__":
    main()
