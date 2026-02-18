#!/usr/bin/env python3
"""
Supervised Probing: Find perceptual editing axes in DCAE z-space.

Instead of unsupervised residual decomposition (which keeps finding frequency),
we measure known perceptual features from SMS/audio data and train linear probes
to find where those features live in z-space.

Each probe's weight vector = an editing direction in z-space.
    z_edited = z_real + delta * direction
    audio_edited = DCAE.decode(z_edited)

Features computed from SMS data (per-frame):
    1. HNR (breathiness proxy) — harmonic energy / noise energy
    2. Spectral centroid — brightness
    3. Spectral tilt — spectral slope across harmonics
    4. Noisiness — noise energy / total energy
    5. High-freq energy ratio — energy above 4kHz / total
    6. Total RMS — overall loudness

Features computed from f0 trajectory (per-sample, mapped to mean z):
    7. Vibrato rate — dominant modulation frequency of f0
    8. Vibrato depth — f0 modulation depth in cents
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
import os
import orjson
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

# Paths
DATA_CACHE_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "data_cache.pt"
MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
SMS_DATA_DIR = SCRIPT_DIR.parent / "data" / "sms_v4"
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "z_probes"

SAMPLE_RATE = 44100
Z_DIM = 128


def find_start_frame(sms_path):
    try:
        sms_data = torch.load(sms_path, weights_only=True, map_location='cpu')
        amps = sms_data['amps']
        frame_energy = amps.sum(dim=1)
        for t in range(len(frame_energy)):
            if frame_energy[t] > 0.001:
                return t
        return 0
    except Exception:
        return 0


def compute_per_frame_features(freqs, amps, noise_amps):
    """
    Compute perceptual features from raw SMS params.

    Args:
        freqs: [T, n_sines] Hz
        amps: [T, n_sines] linear amplitude
        noise_amps: [T, n_noise_bands] linear amplitude

    Returns:
        dict of feature_name → [T] numpy array
    """
    T = freqs.shape[0]
    eps = 1e-10

    # Energy metrics
    harmonic_energy = (amps ** 2).sum(dim=1)       # [T]
    noise_energy = (noise_amps ** 2).sum(dim=1)    # [T]
    total_energy = harmonic_energy + noise_energy   # [T]

    # 1. HNR — harmonics-to-noise ratio (breathiness: low HNR = breathy)
    hnr = torch.log10((harmonic_energy + eps) / (noise_energy + eps))

    # 2. Spectral centroid — brightness
    weighted_freqs = (freqs * amps).sum(dim=1)
    total_amps = amps.sum(dim=1) + eps
    spectral_centroid = weighted_freqs / total_amps

    # 3. Spectral tilt — slope of amplitude across frequency
    # Use log-spaced frequency bins, fit linear regression per frame
    spectral_tilt = torch.zeros(T)
    for t in range(T):
        active = amps[t] > 0.001
        if active.sum() < 3:
            continue
        f = freqs[t, active].numpy()
        a = amps[t, active].numpy()
        if f.max() <= f.min():
            continue
        # Normalize freq to [0, 1]
        f_norm = (np.log10(f + 1) - np.log10(f.min() + 1)) / (np.log10(f.max() + 1) - np.log10(f.min() + 1) + eps)
        a_log = np.log10(a + eps)
        # Linear fit
        if len(f_norm) >= 2:
            slope = np.polyfit(f_norm, a_log, 1)[0]
            spectral_tilt[t] = slope

    # 4. Noisiness — noise fraction of total energy
    noisiness = noise_energy / (total_energy + eps)

    # 5. High-freq energy ratio — energy above 4kHz / total
    high_mask = freqs > 4000
    high_energy = (amps * high_mask.float()).pow(2).sum(dim=1)
    brightness_ratio = high_energy / (harmonic_energy + eps)

    # 6. Total RMS
    total_rms = torch.sqrt(total_energy + eps)

    return {
        'hnr': hnr.numpy(),
        'spectral_centroid': spectral_centroid.numpy(),
        'spectral_tilt': spectral_tilt.numpy(),
        'noisiness': noisiness.numpy(),
        'brightness_ratio': brightness_ratio.numpy(),
        'total_rms': total_rms.numpy(),
    }


def compute_vibrato_features(freqs, amps):
    """
    Compute vibrato features from f0 trajectory.

    Returns:
        vibrato_rate: Hz (dominant modulation frequency, 0 if no vibrato)
        vibrato_depth: cents (std of f0 modulation)
    """
    # Find dominant f0 (highest-amplitude sine)
    mean_amps = amps.mean(dim=0)
    dominant_idx = mean_amps.argmax().item()
    f0_trajectory = freqs[:, dominant_idx].numpy()

    # Filter out silent frames
    active = f0_trajectory > 20
    if active.sum() < 10:
        return 0.0, 0.0

    f0_active = f0_trajectory[active]

    # Convert to cents relative to median
    median_f0 = np.median(f0_active)
    if median_f0 < 20:
        return 0.0, 0.0

    cents = 1200 * np.log2(f0_active / median_f0 + 1e-10)

    # Vibrato depth = std in cents
    vibrato_depth = np.std(cents)

    # Vibrato rate = dominant frequency of f0 modulation
    if len(cents) < 8:
        return 0.0, vibrato_depth

    # Remove DC
    cents_centered = cents - np.mean(cents)
    # FFT
    fft = np.fft.rfft(cents_centered)
    freqs_fft = np.fft.rfftfreq(len(cents_centered), d=1.0 / 10.8)  # DCAE frame rate

    # Look for peak in 3-10 Hz (typical vibrato range)
    vibrato_mask = (freqs_fft >= 3) & (freqs_fft <= 10)
    if not vibrato_mask.any():
        return 0.0, vibrato_depth

    magnitudes = np.abs(fft)
    vibrato_mags = magnitudes.copy()
    vibrato_mags[~vibrato_mask] = 0

    if vibrato_mags.max() > 0:
        peak_idx = vibrato_mags.argmax()
        vibrato_rate = freqs_fft[peak_idx]

        # Check if it's actually significant (peak should be > 2x median)
        if magnitudes[peak_idx] < 2 * np.median(magnitudes[1:]):
            vibrato_rate = 0.0
    else:
        vibrato_rate = 0.0

    return float(vibrato_rate), float(vibrato_depth)


def build_probe_dataset():
    """
    Build dataset of (z_real, perceptual_features) pairs.
    """
    print("=" * 60)
    print("BUILDING PROBE DATASET")
    print("=" * 60)

    # Load data cache
    print(f"\nLoading data cache from {DATA_CACHE_PATH}...")
    cache = torch.load(DATA_CACHE_PATH, weights_only=False, map_location='cpu')
    print(f"  {len(cache)} cached entries")

    # Load manifest
    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    sms_to_latent = {}
    for entry in manifest['entries']:
        sms_to_latent[entry['path']] = entry['latent_path']

    # Process each sample
    all_z_frames = []       # [N_total_frames, 128]
    all_features = {}       # feature_name → [N_total_frames]
    sample_vibrato = []     # per-sample vibrato features
    sample_mean_z = []      # per-sample mean z

    skipped = 0
    processed = 0

    for i, sample in enumerate(cache):
        sms_path = sample['path']

        # Find latent file
        latent_path = sms_to_latent.get(sms_path)
        if not latent_path or not os.path.exists(latent_path):
            skipped += 1
            continue

        # Load z_real
        try:
            loaded = torch.load(latent_path, weights_only=False, map_location='cpu')
            if isinstance(loaded, dict) and 'latents' in loaded:
                z_real_full = loaded['latents']
            elif isinstance(loaded, torch.Tensor):
                z_real_full = loaded
            else:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        if z_real_full.dim() != 3 or z_real_full.shape[0] != 8 or z_real_full.shape[1] != 16:
            skipped += 1
            continue

        # Load raw SMS file for features
        sms_file = sms_path
        if not os.path.exists(sms_file):
            sms_file = str(SCRIPT_DIR.parent / sms_path)
        if not os.path.exists(sms_file):
            sms_file = str(SMS_DATA_DIR / Path(sms_path).name)
        if not os.path.exists(sms_file):
            skipped += 1
            continue

        try:
            sms_data = torch.load(sms_file, weights_only=True, map_location='cpu')
            freqs = sms_data['freqs']       # [T_sms, n_sines]
            amps_raw = sms_data['amps']     # [T_sms, n_sines]
            noise = sms_data.get('noise_amps')
            if noise is None:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        # Find start frame and crop window (same as data cache)
        frame_energy = amps_raw.sum(dim=1)
        start_frame = 0
        for t in range(len(frame_energy)):
            if frame_energy[t] > 0.001:
                start_frame = t
                break

        # Crop SMS to same window as cache (3s = ~32 frames at 10.8fps)
        T_cache = sample['z_sms'].shape[2]
        end_sms = min(start_frame + T_cache, freqs.shape[0])
        T_sms = end_sms - start_frame
        if T_sms < 10:
            skipped += 1
            continue

        freqs_crop = freqs[start_frame:end_sms]
        amps_crop = amps_raw[start_frame:end_sms]
        noise_crop = noise[start_frame:end_sms]

        # Crop z_real to same window
        end_z = min(start_frame + T_sms, z_real_full.shape[2])
        T_actual = end_z - start_frame
        if T_actual < 10:
            skipped += 1
            continue

        T_use = min(T_sms, T_actual)
        freqs_use = freqs_crop[:T_use]
        amps_use = amps_crop[:T_use]
        noise_use = noise_crop[:T_use]

        z_real_crop = z_real_full[:, :, start_frame:start_frame + T_use]
        z_flat = z_real_crop.permute(2, 0, 1).reshape(T_use, Z_DIM)  # [T, 128]

        # Compute per-frame features
        features = compute_per_frame_features(freqs_use, amps_use, noise_use)

        # Store frames
        all_z_frames.append(z_flat)
        for fname, fvals in features.items():
            all_features.setdefault(fname, []).append(fvals[:T_use])

        # Compute per-sample vibrato features
        vib_rate, vib_depth = compute_vibrato_features(freqs_use, amps_use)
        sample_vibrato.append({
            'vibrato_rate': vib_rate,
            'vibrato_depth': vib_depth,
        })
        sample_mean_z.append(z_flat.mean(dim=0))

        processed += 1
        if (i + 1) % 500 == 0:
            print(f"    Processed {processed} samples ({skipped} skipped)...")

    print(f"\n  Total: {processed} samples ({skipped} skipped)")

    # Concatenate all frames
    z_all = torch.cat(all_z_frames, dim=0).numpy()  # [N, 128]
    features_all = {}
    for fname in all_features:
        features_all[fname] = np.concatenate(all_features[fname])

    print(f"  Total frames: {z_all.shape[0]:,}")
    print(f"  Features: {list(features_all.keys())}")

    # Per-sample vibrato
    vibrato_z = torch.stack(sample_mean_z).numpy()   # [n_samples, 128]
    vibrato_rates = np.array([v['vibrato_rate'] for v in sample_vibrato])
    vibrato_depths = np.array([v['vibrato_depth'] for v in sample_vibrato])

    return z_all, features_all, vibrato_z, vibrato_rates, vibrato_depths


def train_probe(z, feature, feature_name, alpha=1.0):
    """
    Train a linear probe: z → feature.
    Returns the probe direction, R² score, and the trained model.
    """
    # Remove NaN/Inf
    valid = np.isfinite(feature) & np.all(np.isfinite(z), axis=1)
    z_valid = z[valid]
    f_valid = feature[valid]

    if len(f_valid) < 100:
        return None

    # Standardize
    z_scaler = StandardScaler()
    f_scaler = StandardScaler()
    z_norm = z_scaler.fit_transform(z_valid)
    f_norm = f_scaler.fit_transform(f_valid.reshape(-1, 1)).ravel()

    # Split train/test
    n = len(z_norm)
    n_train = int(0.8 * n)
    idx = np.random.RandomState(42).permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # Ridge regression
    model = Ridge(alpha=alpha)
    model.fit(z_norm[train_idx], f_norm[train_idx])

    # Evaluate
    pred_train = model.predict(z_norm[train_idx])
    pred_test = model.predict(z_norm[test_idx])
    r2_train = r2_score(f_norm[train_idx], pred_train)
    r2_test = r2_score(f_norm[test_idx], pred_test)

    # The direction in z-space (accounting for z scaling)
    # weight is in normalized z-space, convert to raw z-space
    direction_raw = model.coef_ / (z_scaler.scale_ + 1e-10)
    direction_raw = direction_raw / (np.linalg.norm(direction_raw) + 1e-10)

    return {
        'name': feature_name,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'direction': direction_raw,   # [128] unit vector in z-space
        'coef': model.coef_,
        'z_scaler_mean': z_scaler.mean_,
        'z_scaler_scale': z_scaler.scale_,
        'f_scaler_mean': f_scaler.mean_[0],
        'f_scaler_scale': f_scaler.scale_[0],
        'feature_mean': np.mean(f_valid),
        'feature_std': np.std(f_valid),
        'n_samples': len(f_valid),
    }


def analyze_directions(probes):
    """Check orthogonality between probe directions."""
    print("\n  Direction similarity matrix (|cos_sim|):")
    names = [p['name'] for p in probes if p is not None]
    dirs = [p['direction'] for p in probes if p is not None]

    n = len(dirs)
    print(f"  {'':>20s}", end="")
    for name in names:
        print(f"  {name[:8]:>8s}", end="")
    print()

    for i in range(n):
        print(f"  {names[i]:>20s}", end="")
        for j in range(n):
            cos = abs(np.dot(dirs[i], dirs[j]))
            print(f"  {cos:8.3f}", end="")
        print()


def main():
    print("=" * 60)
    print("SUPERVISED PROBING OF Z-SPACE")
    print("Find perceptual editing axes in DCAE latent space")
    print("=" * 60)
    print()
    print("Approach: measure audio features → train linear probes on z")
    print("Each probe finds WHERE a feature lives in z-space")
    print("The probe's weight vector = editing direction")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build dataset
    z_all, features_all, vibrato_z, vibrato_rates, vibrato_depths = build_probe_dataset()

    # Train probes for per-frame features
    print("\n" + "=" * 60)
    print("TRAINING LINEAR PROBES")
    print("=" * 60)

    probes = []
    print(f"\n  {'Feature':>20s}  {'R² train':>8s}  {'R² test':>8s}  "
          f"{'N frames':>8s}  {'Verdict':>20s}")
    print("  " + "-" * 80)

    for fname, fvals in features_all.items():
        result = train_probe(z_all, fvals, fname)
        if result is None:
            print(f"  {fname:>20s}  {'SKIP':>8s}  (insufficient data)")
            continue

        verdict = "STRONG" if result['r2_test'] > 0.3 else \
                  "MODERATE" if result['r2_test'] > 0.1 else \
                  "WEAK" if result['r2_test'] > 0.03 else "NOT FOUND"

        print(f"  {fname:>20s}  {result['r2_train']:8.4f}  {result['r2_test']:8.4f}  "
              f"{result['n_samples']:8d}  {verdict:>20s}")
        probes.append(result)

    # Train probes for per-sample vibrato features
    print(f"\n  Per-sample features (using mean z per sample):")
    print(f"  {'Feature':>20s}  {'R² train':>8s}  {'R² test':>8s}  "
          f"{'N samples':>8s}  {'Verdict':>20s}")
    print("  " + "-" * 80)

    for fname, fvals in [('vibrato_rate', vibrato_rates),
                          ('vibrato_depth', vibrato_depths)]:
        # Only use samples with detected vibrato for rate
        if fname == 'vibrato_rate':
            has_vibrato = fvals > 0
            if has_vibrato.sum() < 50:
                print(f"  {fname:>20s}  {'SKIP':>8s}  (only {has_vibrato.sum()} samples with vibrato)")
                continue
            result = train_probe(vibrato_z[has_vibrato], fvals[has_vibrato], fname)
        else:
            result = train_probe(vibrato_z, fvals, fname)

        if result is None:
            print(f"  {fname:>20s}  {'SKIP':>8s}  (insufficient data)")
            continue

        verdict = "STRONG" if result['r2_test'] > 0.3 else \
                  "MODERATE" if result['r2_test'] > 0.1 else \
                  "WEAK" if result['r2_test'] > 0.03 else "NOT FOUND"

        print(f"  {fname:>20s}  {result['r2_train']:8.4f}  {result['r2_test']:8.4f}  "
              f"{result['n_samples']:8d}  {verdict:>20s}")
        probes.append(result)

    # Analyze direction similarity
    print("\n" + "=" * 60)
    print("DIRECTION ANALYSIS")
    print("=" * 60)
    analyze_directions(probes)

    # Top z-dimensions per direction
    print("\n  Top 5 z-dimensions per feature direction:")
    for p in probes:
        if p is None:
            continue
        d = p['direction']
        top_idx = np.argsort(np.abs(d))[::-1][:5]
        top_str = ", ".join([f"z[{i}]={d[i]:+.3f}" for i in top_idx])
        print(f"  {p['name']:>20s}: {top_str}")

    # Save probes
    print("\n" + "=" * 60)
    print("SAVING PROBES")
    print("=" * 60)

    probe_data = {}
    for p in probes:
        if p is None:
            continue
        probe_data[p['name']] = {
            'direction': p['direction'],
            'r2_test': p['r2_test'],
            'r2_train': p['r2_train'],
            'coef': p['coef'],
            'z_scaler_mean': p['z_scaler_mean'],
            'z_scaler_scale': p['z_scaler_scale'],
            'f_scaler_mean': p['f_scaler_mean'],
            'f_scaler_scale': p['f_scaler_scale'],
            'feature_mean': p['feature_mean'],
            'feature_std': p['feature_std'],
        }

    save_path = OUTPUT_DIR / "z_probes.pt"
    torch.save(probe_data, save_path)
    print(f"  Saved {len(probe_data)} probes to {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    strong = [p for p in probes if p and p['r2_test'] > 0.3]
    moderate = [p for p in probes if p and 0.1 < p['r2_test'] <= 0.3]
    weak = [p for p in probes if p and 0.03 < p['r2_test'] <= 0.1]

    print(f"\n  Strong axes (R² > 0.3):   {[p['name'] for p in strong]}")
    print(f"  Moderate axes (R² > 0.1): {[p['name'] for p in moderate]}")
    print(f"  Weak axes (R² > 0.03):    {[p['name'] for p in weak]}")

    if strong:
        print(f"\n  Strong axes are viable editing directions!")
        print(f"  To edit '{strong[0]['name']}' in z-space:")
        print(f"    direction = probes['{strong[0]['name']}']['direction']  # [128]")
        print(f"    z_edited = z_real + delta * direction")
        print(f"    audio = DCAE.decode(z_edited)")
        print(f"\n  delta > 0 increases the feature, delta < 0 decreases it")
        print(f"  Typical delta range: ±{strong[0]['feature_std']:.2f} * 0.5 to 2.0")
    else:
        print(f"\n  No strong axes found. Features may be nonlinearly encoded.")
        print(f"  Consider: nonlinear probes, different features, or more data.")

    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
