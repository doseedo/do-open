#!/usr/bin/env python3
"""
Trajectory Discovery: find perceptual axes in trajectory space, not frame space.

The existing discovery scripts (contrastive, pitchbin, temporal) all collapse
temporal structure before running PCA:
  - contrastive_discovery: z_flat [T, 128] -> z_mean [128] (mean across time)
  - pitchbin_discovery: z frames pooled across tracks (temporal order discarded)
  - temporal_discovery: z frames concatenated (temporal order discarded)

Result: they only find per-frame axes (dynamics, one vague timbral direction).
The interesting perceptual qualities -- vibrato, growl, breathiness, articulation
-- are temporal patterns that don't exist at any single frame.

This script treats trajectories (z sequences over time) as the fundamental unit:
  1. Extract windowed trajectory features: modulation rate, depth, smoothness,
     velocity, spectral profile per z-dim
  2. Run PCA on trajectory-feature space (not frame-space)
  3. The discovered axes correspond to trajectory shapes, not frame values

Expected discoveries:
  - "Vibrato axis": high modulation rate (~5Hz), moderate depth, high autocorrelation
  - "Growl axis": high modulation depth, low autocorrelation (chaotic)
  - "Sustain smoothness axis": autocorrelation spectrum
  - "Attack shape axis": velocity profile during onset
  - "Breathiness axis": noise-like modulation pattern
"""

import sys
import math
import torch
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM

PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "trajectory_discovery"

Z_DIM = 128
MAX_FRAMES = 256
SAMPLE_RATE = 44100
DCAE_HOP = 4083
FPS = SAMPLE_RATE / DCAE_HOP  # ~10.8


# ============================================================
# Trajectory Feature Extraction
# ============================================================

def extract_trajectory_features(z_flat, window_frames=40, hop_frames=10):
    """
    Extract trajectory-level features from a z sequence.

    Args:
        z_flat: [T, 128] — z-space trajectory for one sample
        window_frames: int — window size in frames (~3.7s at 10.8fps for 40 frames)
        hop_frames: int — hop between windows

    Returns:
        features: [N_windows, feat_dim] — trajectory feature matrix
        feature_names: list of str — names for each feature dimension

    Feature groups (per window):
        Position:      z_mean [128]           — where in z-space
        Modulation:    z_std [128]            — modulation depth per dim
        Velocity:      vel_mean [128]         — drift direction
                       vel_std [128]          — speed variation per dim
        Smoothness:    autocorr_lag1 [128]    — temporal smoothness per dim
        Spectral:      spectral_centroid [128] — dominant modulation rate per dim
        Scalars:       overall_speed [1]      — total velocity magnitude
                       speed_variation [1]    — velocity variation
                       mean_autocorr [1]      — average smoothness
                       spectral_flatness [1]  — modulation regularity
    Total: 772 features per window
    """
    T, D = z_flat.shape
    features = []

    for t_start in range(0, T - window_frames, hop_frames):
        w = z_flat[t_start:t_start + window_frames]  # [W, 128]
        W = w.shape[0]

        # ---- Position: where in z-space ----
        z_mean = w.mean(dim=0)  # [128]

        # ---- Modulation depth: per-dim std ----
        z_std = w.std(dim=0)  # [128]

        # ---- Velocity: frame-to-frame differences ----
        vel = w[1:] - w[:-1]  # [W-1, 128]
        vel_mean = vel.mean(dim=0)  # [128] — net drift direction
        vel_std = vel.std(dim=0)  # [128] — speed variation per dim
        vel_mag = vel.norm(dim=1)  # [W-1] — scalar speed per frame

        overall_speed = vel_mag.mean().unsqueeze(0)  # [1]
        speed_var = vel_mag.std().unsqueeze(0)  # [1]

        # ---- Smoothness: lag-1 autocorrelation per dim ----
        w_centered = w - z_mean.unsqueeze(0)
        # Autocorr = E[z_t * z_{t+1}] / E[z_t^2]
        numerator = (w_centered[:-1] * w_centered[1:]).mean(dim=0)  # [128]
        denominator = (w_centered ** 2).mean(dim=0).clamp(min=1e-10)  # [128]
        autocorr = numerator / denominator  # [128]
        autocorr = autocorr.clamp(-1, 1)
        mean_autocorr = autocorr.mean().unsqueeze(0)  # [1]

        # ---- Spectral: modulation rate per dim via FFT ----
        w_fft = torch.fft.rfft(w_centered, dim=0).abs()  # [W//2+1, 128]
        freq_bins = torch.fft.rfftfreq(W, d=1.0 / FPS)  # [W//2+1] in Hz

        # Spectral centroid per dim: weighted average frequency
        total_power = w_fft.sum(dim=0).clamp(min=1e-10)  # [128]
        spectral_centroid = (
            freq_bins.unsqueeze(1) * w_fft
        ).sum(dim=0) / total_power  # [128]

        # Spectral flatness (geometric mean / arithmetic mean of power spectrum)
        # High = noise-like modulation, low = tonal modulation (like vibrato)
        log_power = torch.log(w_fft[1:] + 1e-10).mean(dim=0)  # exclude DC
        geo_mean = torch.exp(log_power)
        arith_mean = w_fft[1:].mean(dim=0).clamp(min=1e-10)
        flatness = (geo_mean / arith_mean).mean().unsqueeze(0)  # [1] scalar

        # ---- Assemble feature vector ----
        feat = torch.cat([
            z_mean,              # 128
            z_std,               # 128
            vel_mean,            # 128
            vel_std,             # 128
            autocorr,            # 128
            spectral_centroid,   # 128
            overall_speed,       # 1
            speed_var,           # 1
            mean_autocorr,       # 1
            flatness,            # 1
        ])  # total: 772
        features.append(feat)

    if not features:
        return None

    return torch.stack(features)  # [N_windows, 772]


def get_feature_names():
    """Return human-readable names for each feature dimension."""
    names = []
    for prefix in ['pos', 'mod_depth', 'vel_mean', 'vel_std', 'autocorr', 'spec_centroid']:
        for d in range(128):
            names.append(f"{prefix}_z{d}")
    names.extend(['overall_speed', 'speed_variation', 'mean_autocorr', 'spectral_flatness'])
    return names


# ============================================================
# Data Loading
# ============================================================

def load_z_sequences(max_samples=2000, min_frames=60):
    """Load z latent sequences, filtering for minimum length."""
    print(f"\nLoading z sequences from {LATENT_BASE}...")
    data = []
    skipped = 0

    for pt_file in LATENT_BASE.rglob("*.pt"):
        if len(data) >= max_samples:
            break
        try:
            loaded = torch.load(pt_file, weights_only=False, map_location='cpu')
            if isinstance(loaded, dict) and 'latents' in loaded:
                z = loaded['latents']
            elif isinstance(loaded, torch.Tensor):
                z = loaded
            else:
                skipped += 1
                continue

            if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16:
                skipped += 1
                continue

            T = z.shape[2]
            if T < min_frames:
                skipped += 1
                continue

            if T > MAX_FRAMES:
                z = z[:, :, :MAX_FRAMES]
                T = MAX_FRAMES

            # Flatten to [T, 128]
            z_flat = z.permute(2, 0, 1).reshape(T, Z_DIM).float()
            data.append({
                'z_flat': z_flat,
                'path': str(pt_file),
                'n_frames': T,
            })

            if len(data) % 200 == 0:
                print(f"    Loaded {len(data)} samples...")
        except Exception:
            skipped += 1
            continue

    print(f"  Loaded {len(data)} sequences (skipped {skipped})")
    return data


def compute_residuals(data, device):
    """Compute SMS residuals for each sequence (z_flat - G(F(z_flat)))."""
    if not PHASE1_PATH.exists():
        print(f"  Phase 1 codec not found at {PHASE1_PATH}")
        print("  Using raw z sequences instead of residuals")
        return data, False

    print("\nComputing SMS residuals...")
    codec = BidirectionalCodec(sms_dim=102, z_dim=Z_DIM, g_hidden=384, f_hidden=256)
    codec.load_state_dict(torch.load(PHASE1_PATH, weights_only=True, map_location='cpu'))
    codec = codec.to(device).eval()

    with torch.no_grad():
        for i, sample in enumerate(data):
            z_flat = sample['z_flat'].unsqueeze(0).to(device)  # [1, T, 128]
            sms_pred = codec.forward_F(z_flat)
            z_sms = codec.forward_G(sms_pred)
            residual = (z_flat - z_sms).squeeze(0).cpu()  # [T, 128]
            sample['residual'] = residual

            if i % 200 == 0:
                cos = TF.cosine_similarity(z_flat, z_sms, dim=-1).mean().item()
                print(f"    {i}/{len(data)}  cos_sim={cos:.4f}")

    del codec
    torch.cuda.empty_cache()
    return data, True


# ============================================================
# Discovery
# ============================================================

def discover_trajectory_axes(data, use_residual=True, window_frames=40,
                              hop_frames=10, n_components=16):
    """
    Main discovery: extract trajectory features, run PCA.

    Args:
        data: list of dicts with 'z_flat' and optionally 'residual'
        use_residual: if True and residuals available, analyze residual trajectories
        window_frames: trajectory window size
        hop_frames: hop between windows
        n_components: number of PCA axes to keep

    Returns:
        dict with axes, explained variance, feature importances
    """
    print("\n" + "=" * 60)
    print("TRAJECTORY AXIS DISCOVERY")
    print("=" * 60)

    key = 'residual' if (use_residual and 'residual' in data[0]) else 'z_flat'
    print(f"  Analyzing: {key}")
    print(f"  Window: {window_frames} frames ({window_frames/FPS:.1f}s)")
    print(f"  Hop: {hop_frames} frames ({hop_frames/FPS:.1f}s)")

    # Extract trajectory features from all samples
    print("\nExtracting trajectory features...")
    all_features = []
    sample_indices = []  # track which sample each window came from

    for i, sample in enumerate(tqdm(data, desc="Extracting")):
        z_seq = sample[key]  # [T, 128]
        feats = extract_trajectory_features(z_seq, window_frames, hop_frames)
        if feats is not None:
            all_features.append(feats)
            sample_indices.extend([i] * feats.shape[0])

    if not all_features:
        print("  No trajectory features extracted!")
        return None

    X = torch.cat(all_features, dim=0).numpy()  # [N_windows, 772]
    print(f"\n  Total trajectory windows: {X.shape[0]}")
    print(f"  Feature dimension: {X.shape[1]}")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    print(f"\nRunning PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n  Explained variance ratios:")
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        print(f"    PC{i:2d}: {var:6.3f}  (cumulative: {cumulative:6.3f})")

    # Analyze what each PC captures
    feature_names = get_feature_names()
    feature_groups = {
        'position': slice(0, 128),
        'mod_depth': slice(128, 256),
        'vel_mean': slice(256, 384),
        'vel_std': slice(384, 512),
        'autocorr': slice(512, 640),
        'spec_centroid': slice(640, 768),
        'overall_speed': 768,
        'speed_variation': 769,
        'mean_autocorr': 770,
        'spectral_flatness': 771,
    }

    print(f"\n  Per-axis feature group importances:")
    print(f"  {'PC':>4s}  {'Position':>8s}  {'ModDepth':>8s}  {'Velocity':>8s}  "
          f"{'VelStd':>8s}  {'Autocorr':>8s}  {'SpecCent':>8s}  {'Scalars':>8s}  Interpretation")

    axis_interpretations = []

    for pc_idx in range(min(n_components, 12)):
        loadings = pca.components_[pc_idx]  # [772]

        # Group energy: sum of squared loadings per feature group
        group_energy = {}
        for gname, gslice in feature_groups.items():
            if isinstance(gslice, slice):
                group_energy[gname] = np.sum(loadings[gslice] ** 2)
            else:
                group_energy[gname] = loadings[gslice] ** 2

        total_e = sum(group_energy.values()) + 1e-10

        # Normalize to percentages
        ge_pct = {k: v / total_e for k, v in group_energy.items()}

        # Interpret based on dominant feature groups
        dominant = max(ge_pct, key=ge_pct.get)
        interp = _interpret_axis(ge_pct, loadings, X_pca[:, pc_idx])
        axis_interpretations.append(interp)

        scalars_pct = (ge_pct.get('overall_speed', 0) +
                       ge_pct.get('speed_variation', 0) +
                       ge_pct.get('mean_autocorr', 0) +
                       ge_pct.get('spectral_flatness', 0))

        print(f"  PC{pc_idx:2d}  "
              f"{ge_pct['position']:8.1%}  "
              f"{ge_pct['mod_depth']:8.1%}  "
              f"{ge_pct['vel_mean']:8.1%}  "
              f"{ge_pct['vel_std']:8.1%}  "
              f"{ge_pct['autocorr']:8.1%}  "
              f"{ge_pct['spec_centroid']:8.1%}  "
              f"{scalars_pct:8.1%}  "
              f"{interp}")

    return {
        'pca': pca,
        'scaler': scaler,
        'X_pca': X_pca,
        'sample_indices': np.array(sample_indices),
        'feature_names': feature_names,
        'interpretations': axis_interpretations,
        'window_frames': window_frames,
        'hop_frames': hop_frames,
        'n_windows': X.shape[0],
        'use_residual': use_residual,
    }


def _interpret_axis(ge_pct, loadings, pc_values):
    """Heuristic interpretation of a trajectory PCA axis."""
    # Which feature group dominates?
    top_groups = sorted(ge_pct.items(), key=lambda x: -x[1])[:3]
    dom = top_groups[0][0]
    dom_pct = top_groups[0][1]

    if dom == 'position' and dom_pct > 0.5:
        return "Z-POSITION (instrument/register identity)"

    if dom == 'mod_depth' and dom_pct > 0.3:
        # Check if spectral centroid also high (fast modulation vs slow)
        sc_pct = ge_pct.get('spec_centroid', 0)
        if sc_pct > 0.15:
            return "VIBRATO/TREMOLO (depth + rate)"
        return "MODULATION DEPTH (dynamics/expression)"

    if dom == 'autocorr' and dom_pct > 0.3:
        return "TRAJECTORY SMOOTHNESS (clean vs chaotic)"

    if dom == 'spec_centroid' and dom_pct > 0.3:
        return "MODULATION RATE (slow swell vs fast vibrato)"

    if dom == 'vel_mean' and dom_pct > 0.3:
        return "DRIFT DIRECTION (pitch/timbre trajectory)"

    if dom == 'vel_std' and dom_pct > 0.3:
        return "SPEED VARIATION (steady vs accelerating)"

    # Mixed axes
    if ge_pct.get('mod_depth', 0) > 0.15 and ge_pct.get('autocorr', 0) > 0.15:
        return "ROUGHNESS (high depth + low smoothness = growl)"

    if ge_pct.get('mod_depth', 0) > 0.15 and ge_pct.get('spec_centroid', 0) > 0.15:
        return "VIBRATO CHARACTER (depth x rate)"

    return f"MIXED ({dom}={dom_pct:.0%})"


# ============================================================
# Detailed Axis Analysis
# ============================================================

def analyze_discovered_axes(discovery_result, data):
    """
    Detailed analysis of what each discovered trajectory axis captures.
    Projects all windows onto each axis and examines extremes.
    """
    if discovery_result is None:
        return

    print("\n" + "=" * 60)
    print("DETAILED AXIS ANALYSIS")
    print("=" * 60)

    X_pca = discovery_result['X_pca']
    sample_idx = discovery_result['sample_indices']

    n_axes = min(8, X_pca.shape[1])

    for ax in range(n_axes):
        scores = X_pca[:, ax]

        # Find extreme windows (high and low on this axis)
        sorted_idx = np.argsort(scores)
        n_extreme = min(10, len(sorted_idx) // 10)

        low_windows = sorted_idx[:n_extreme]
        high_windows = sorted_idx[-n_extreme:]

        # Which samples are at the extremes?
        low_samples = set(sample_idx[low_windows])
        high_samples = set(sample_idx[high_windows])

        print(f"\n  PC{ax}: {discovery_result['interpretations'][ax]}")
        print(f"    Score range: [{scores.min():.2f}, {scores.max():.2f}]  "
              f"std={scores.std():.2f}")
        print(f"    Low-end samples ({len(low_samples)} unique):")
        for s_idx in list(low_samples)[:3]:
            path = Path(data[s_idx]['path']).stem
            print(f"      {path}")
        print(f"    High-end samples ({len(high_samples)} unique):")
        for s_idx in list(high_samples)[:3]:
            path = Path(data[s_idx]['path']).stem
            print(f"      {path}")

        # Temporal distribution: where in the sample do extreme windows occur?
        # (onset vs sustain vs release)
        # This can reveal if the axis captures attack vs sustain differences


# ============================================================
# Multi-scale Analysis
# ============================================================

def multi_scale_discovery(data, use_residual=True):
    """
    Run trajectory discovery at multiple window sizes to capture
    different timescales of temporal structure.

    Short windows (~1s): attack transients, fast modulations
    Medium windows (~3.7s): vibrato, phrase-level dynamics
    Long windows (~7.4s): slow envelopes, overall shape
    """
    print("\n" + "=" * 60)
    print("MULTI-SCALE TRAJECTORY ANALYSIS")
    print("=" * 60)

    scales = [
        ('short', 12, 4),    # ~1.1s windows, ~0.4s hop
        ('medium', 40, 10),  # ~3.7s windows, ~0.9s hop
        ('long', 80, 20),    # ~7.4s windows, ~1.9s hop
    ]

    results = {}
    for name, window, hop in scales:
        print(f"\n--- Scale: {name} (window={window} frames = {window/FPS:.1f}s) ---")

        # Filter data for samples long enough
        valid_data = [d for d in data if d.get('residual', d['z_flat']).shape[0] >= window + hop]
        if len(valid_data) < 50:
            print(f"  Only {len(valid_data)} samples long enough, skipping")
            continue

        result = discover_trajectory_axes(
            valid_data,
            use_residual=use_residual,
            window_frames=window,
            hop_frames=hop,
            n_components=8,
        )
        results[name] = result

    # Compare scales: do the same axes appear across scales?
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("CROSS-SCALE COMPARISON")
        print("=" * 60)

        for name, result in results.items():
            if result is None:
                continue
            print(f"\n  {name} scale:")
            var = result['pca'].explained_variance_ratio_
            for i in range(min(4, len(var))):
                print(f"    PC{i}: {var[i]:.3f} — {result['interpretations'][i]}")

    return results


# ============================================================
# Save Results
# ============================================================

def save_discovery(discovery_result, multi_results, save_dir):
    """Save discovered trajectory axes for use in the Gradio UI."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save main (medium-scale) result
    if discovery_result is not None:
        pca = discovery_result['pca']

        # Convert to format similar to existing discovered_axes.pt
        axes_data = {
            'trajectory_pca': [],
            'scaler_mean': discovery_result['scaler'].mean_,
            'scaler_std': discovery_result['scaler'].scale_,
            'window_frames': discovery_result['window_frames'],
            'hop_frames': discovery_result['hop_frames'],
            'n_windows': discovery_result['n_windows'],
            'feature_dim': pca.components_.shape[1],
        }

        for i in range(pca.n_components_):
            axes_data['trajectory_pca'].append({
                'direction': pca.components_[i],  # [feat_dim]
                'variance_explained': float(pca.explained_variance_ratio_[i]),
                'interpretation': discovery_result['interpretations'][i]
                    if i < len(discovery_result['interpretations']) else 'unknown',
            })

        save_path = save_dir / "trajectory_axes.pt"
        torch.save(axes_data, save_path)
        print(f"\nSaved trajectory axes to {save_path}")

    # Save multi-scale results
    if multi_results:
        multi_data = {}
        for scale_name, result in multi_results.items():
            if result is None:
                continue
            pca = result['pca']
            multi_data[scale_name] = {
                'components': pca.components_,
                'explained_variance': pca.explained_variance_ratio_,
                'interpretations': result['interpretations'],
                'window_frames': result['window_frames'],
                'n_windows': result['n_windows'],
            }

        if multi_data:
            save_path = save_dir / "trajectory_axes_multiscale.pt"
            torch.save(multi_data, save_path)
            print(f"Saved multi-scale axes to {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("TRAJECTORY DISCOVERY")
    print("=" * 60)
    print()
    print("Finding perceptual axes in trajectory space (not frame space).")
    print("Fundamental unit: windowed z-trajectories with temporal features.")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load z sequences
    data = load_z_sequences(max_samples=2000, min_frames=60)

    # Compute SMS residuals (so we analyze what SMS can't explain)
    data, has_residual = compute_residuals(data, device)

    # Main discovery (medium scale)
    print("\n" + "=" * 60)
    print("MAIN DISCOVERY (medium scale)")
    print("=" * 60)
    discovery = discover_trajectory_axes(
        data,
        use_residual=has_residual,
        window_frames=40,    # ~3.7s at 10.8fps
        hop_frames=10,       # ~0.9s hop
        n_components=16,
    )

    # Detailed analysis of discovered axes
    if discovery is not None:
        analyze_discovered_axes(discovery, data)

    # Multi-scale analysis
    multi = multi_scale_discovery(data, use_residual=has_residual)

    # Save
    save_discovery(discovery, multi, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("TRAJECTORY DISCOVERY SUMMARY")
    print("=" * 60)

    if discovery is not None:
        var = discovery['pca'].explained_variance_ratio_
        print(f"\n  Windows analyzed: {discovery['n_windows']}")
        print(f"  Window size: {discovery['window_frames']} frames ({discovery['window_frames']/FPS:.1f}s)")
        print(f"  Top-3 variance: {var[0]:.3f}, {var[1]:.3f}, {var[2]:.3f}")
        print(f"  Total (16 PC): {var.sum():.3f}")

        print(f"\n  Discovered trajectory axes:")
        for i, interp in enumerate(discovery['interpretations'][:8]):
            print(f"    PC{i}: {var[i]:.3f}  {interp}")

    print(f"\n  vs frame-based discovery (for comparison):")
    print(f"    Frame PCA finds: position (instrument), dynamics (one axis)")
    print(f"    Trajectory PCA should find: vibrato, growl, smoothness, attack shape, ...")

    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
