#!/usr/bin/env python3
"""
Investigate open questions from atom discovery.

1. Are dims 64-127 really inactive? Or do they control amplitude/temporal shape?
2. Can we fit freq = a*z² + b*z + c per band directly?
3. What do weak dims 0-47 control?
4. Is temporal spread convolution-like or attention-like?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from collections import defaultdict

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE


def load_dcae(device='cuda'):
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH)
    dcae.dcae.to(device)
    dcae.dcae.eval()
    return dcae


def get_mel_from_z(dcae, z, device='cuda'):
    with torch.no_grad():
        z_denorm = z / 0.1786 + (-1.9091)
        mel = dcae.dcae.decode(z_denorm).sample
        return mel


def extract_features(mel):
    """Extract comprehensive features from mel."""
    B, C, F, T = mel.shape
    mel_mono = mel.mean(dim=1)  # [B, 128, T]
    mel_power = torch.exp(mel_mono)

    features = {}

    # Spectral features
    freq_bins = torch.arange(F, device=mel.device).float()
    total_power = mel_power.sum(dim=1, keepdim=True).clamp(min=1e-8)
    features['centroid'] = (mel_power * freq_bins.view(1, -1, 1) / total_power).sum(dim=1).mean()
    features['energy'] = mel_power.sum(dim=1).mean()
    features['peak_bin'] = mel_mono.argmax(dim=1).float().mean()

    # Band energies
    features['low_energy'] = mel_power[:, :32, :].sum(dim=1).mean()
    features['mid_energy'] = mel_power[:, 32:96, :].sum(dim=1).mean()
    features['high_energy'] = mel_power[:, 96:, :].sum(dim=1).mean()

    # TEMPORAL features (what dims 64-127 might control)
    energy_over_time = mel_power.sum(dim=1)  # [B, T]
    features['temporal_mean'] = energy_over_time.mean()
    features['temporal_std'] = energy_over_time.std()

    # Attack/decay shape
    if T > 4:
        features['attack'] = energy_over_time[:, :T//4].mean()
        features['sustain'] = energy_over_time[:, T//4:3*T//4].mean()
        features['release'] = energy_over_time[:, 3*T//4:].mean()

    # Stereo difference (if dims 64-127 control stereo)
    if C == 2:
        stereo_diff = (mel[:, 0] - mel[:, 1]).abs().mean()
        features['stereo_diff'] = stereo_diff

    # Spectral flux (change over time)
    if T > 1:
        flux = (mel_mono[:, :, 1:] - mel_mono[:, :, :-1]).abs().mean()
        features['spectral_flux'] = flux

    return features


def question1_dims_64_127(dcae, base_z, device='cuda'):
    """
    Q1: Are dims 64-127 really inactive?
    Test if they control amplitude, temporal shape, stereo, or spectral flux.
    """
    print("\n" + "=" * 70)
    print("Q1: What do dims 64-127 control?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    base_mel = get_mel_from_z(dcae, base_z, device)
    base_features = extract_features(base_mel)

    # Test dims 64-127
    influences = defaultdict(list)

    for dim in range(64, 128):
        z_perturbed = z_flat.clone()
        z_perturbed[:, dim, :] += 0.5
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d, device)
        perturbed_features = extract_features(perturbed_mel)

        for feat_name in base_features:
            diff = (perturbed_features[feat_name] - base_features[feat_name]).item()
            influences[feat_name].append(diff)

    print("\nDims 64-127 influence on features:")
    print("-" * 50)

    for feat_name in influences:
        vals = np.array(influences[feat_name])
        max_influence = np.max(np.abs(vals))
        mean_influence = np.mean(np.abs(vals))
        most_influential_dim = 64 + np.argmax(np.abs(vals))

        print(f"  {feat_name:15s}: max={max_influence:.4f} (dim {most_influential_dim}), mean={mean_influence:.4f}")

    # Compare to dims 48-63
    print("\nComparison to dims 48-63:")
    influences_48_63 = defaultdict(list)

    for dim in range(48, 64):
        z_perturbed = z_flat.clone()
        z_perturbed[:, dim, :] += 0.5
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d, device)
        perturbed_features = extract_features(perturbed_mel)

        for feat_name in base_features:
            diff = (perturbed_features[feat_name] - base_features[feat_name]).item()
            influences_48_63[feat_name].append(diff)

    print("\n  Feature          | dims 48-63 | dims 64-127 | Ratio")
    print("  " + "-" * 55)
    for feat_name in influences:
        v1 = np.mean(np.abs(influences_48_63[feat_name]))
        v2 = np.mean(np.abs(influences[feat_name]))
        ratio = v1 / (v2 + 1e-8)
        print(f"  {feat_name:15s} | {v1:10.4f} | {v2:11.4f} | {ratio:5.1f}x")


def question2_quadratic_fit(dcae, base_z, device='cuda'):
    """
    Q2: Can we fit freq = a*z² + b*z + c directly?
    Test if quadratic polynomial accurately predicts output.
    """
    print("\n" + "=" * 70)
    print("Q2: Can we fit quadratic z → output directly?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    # For top influential dims, collect (z_value, output_feature) pairs
    top_dims = [62, 60, 61, 59, 63]  # From previous discovery

    for dim in top_dims:
        print(f"\nDim {dim}:")

        z_values = []
        centroid_values = []
        energy_values = []

        # Sample many z values
        for delta in np.linspace(-2, 2, 21):
            z_perturbed = z_flat.clone()
            z_perturbed[:, dim, :] += delta
            z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

            mel = get_mel_from_z(dcae, z_perturbed_4d, device)
            features = extract_features(mel)

            z_values.append(delta)
            centroid_values.append(features['centroid'].item())
            energy_values.append(features['energy'].item())

        z_values = np.array(z_values)
        centroid_values = np.array(centroid_values)
        energy_values = np.array(energy_values)

        # Fit quadratic
        quad_coeffs = np.polyfit(z_values, centroid_values, 2)
        quad_pred = np.polyval(quad_coeffs, z_values)
        quad_r2 = 1 - np.sum((centroid_values - quad_pred)**2) / np.sum((centroid_values - centroid_values.mean())**2)

        # Fit cubic (for comparison)
        cubic_coeffs = np.polyfit(z_values, centroid_values, 3)
        cubic_pred = np.polyval(cubic_coeffs, z_values)
        cubic_r2 = 1 - np.sum((centroid_values - cubic_pred)**2) / np.sum((centroid_values - centroid_values.mean())**2)

        # Fit linear
        lin_coeffs = np.polyfit(z_values, centroid_values, 1)
        lin_pred = np.polyval(lin_coeffs, z_values)
        lin_r2 = 1 - np.sum((centroid_values - lin_pred)**2) / np.sum((centroid_values - centroid_values.mean())**2)

        print(f"  Centroid prediction:")
        print(f"    Linear R²:    {lin_r2:.4f}")
        print(f"    Quadratic R²: {quad_r2:.4f}")
        print(f"    Cubic R²:     {cubic_r2:.4f}")
        print(f"    Coefficients: {quad_coeffs[0]:.4f}z² + {quad_coeffs[1]:.4f}z + {quad_coeffs[2]:.4f}")


def question3_weak_dims(dcae, base_z, device='cuda'):
    """
    Q3: What do weak dims 0-47 control?
    They clustered together - maybe fine detail or residual?
    """
    print("\n" + "=" * 70)
    print("Q3: What do weak dims 0-47 control?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    base_mel = get_mel_from_z(dcae, base_z, device)
    base_features = extract_features(base_mel)

    # Test cumulative effect of many weak dims
    print("\nCumulative effect test:")
    print("(Perturbing multiple dims together)")

    for n_dims in [5, 10, 20, 47]:
        z_perturbed = z_flat.clone()
        z_perturbed[:, :n_dims, :] += 0.5  # Perturb first n dims together
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d, device)
        perturbed_features = extract_features(perturbed_mel)

        print(f"\n  Perturbing dims 0-{n_dims-1}:")
        for feat_name in ['centroid', 'energy', 'spectral_flux', 'temporal_std']:
            if feat_name in base_features and feat_name in perturbed_features:
                diff = (perturbed_features[feat_name] - base_features[feat_name]).item()
                print(f"    {feat_name}: {diff:+.4f}")

    # Compare to same number of strong dims
    print("\n\nComparison: 16 weak dims vs 16 strong dims (48-63):")

    # Weak dims
    z_weak = z_flat.clone()
    z_weak[:, :16, :] += 0.5
    mel_weak = get_mel_from_z(dcae, z_weak.reshape(B, C, H, T), device)
    feat_weak = extract_features(mel_weak)

    # Strong dims
    z_strong = z_flat.clone()
    z_strong[:, 48:64, :] += 0.5
    mel_strong = get_mel_from_z(dcae, z_strong.reshape(B, C, H, T), device)
    feat_strong = extract_features(mel_strong)

    print(f"\n  Feature          | Weak (0-15) | Strong (48-63)")
    print("  " + "-" * 45)
    for feat_name in ['centroid', 'energy', 'peak_bin', 'spectral_flux']:
        diff_weak = (feat_weak[feat_name] - base_features[feat_name]).item()
        diff_strong = (feat_strong[feat_name] - base_features[feat_name]).item()
        print(f"  {feat_name:15s} | {diff_weak:+11.4f} | {diff_strong:+11.4f}")


def question4_temporal_structure(dcae, base_z, device='cuda'):
    """
    Q4: Is temporal spread convolution-like or attention-like?

    Convolution: local, symmetric, depends on distance only
    Attention: can attend to specific distant frames, asymmetric
    """
    print("\n" + "=" * 70)
    print("Q4: Temporal structure - convolution or attention?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    if T < 10:
        print("  Need more time steps for this analysis")
        return

    # Test: perturb at different positions, measure influence spread
    print("\nTesting influence spread from different perturbation positions:")

    dim = 62  # Most influential dim
    positions = [2, T//4, T//2, 3*T//4, T-3]

    base_mel = get_mel_from_z(dcae, base_z, device)
    base_energy = torch.exp(base_mel.mean(dim=1)).sum(dim=1)  # [B, T_mel]
    T_mel = base_energy.shape[-1]

    for t_perturb in positions:
        print(f"\n  Perturb at t={t_perturb} (of {T}):")

        z_perturbed = z_flat.clone()
        z_perturbed[:, dim, t_perturb] += 2.0
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d, device)
        perturbed_energy = torch.exp(perturbed_mel.mean(dim=1)).sum(dim=1)

        diff = (perturbed_energy - base_energy).squeeze().cpu().numpy()

        # Find where influence is strongest
        t_perturb_mel = t_perturb * 8  # Mel has 8x time resolution

        # Analyze spread
        peak_idx = np.argmax(np.abs(diff))
        peak_offset = peak_idx - t_perturb_mel

        # Check symmetry
        if t_perturb_mel > 16 and t_perturb_mel < T_mel - 16:
            left_influence = np.sum(np.abs(diff[t_perturb_mel-16:t_perturb_mel]))
            right_influence = np.sum(np.abs(diff[t_perturb_mel:t_perturb_mel+16]))
            symmetry = min(left_influence, right_influence) / (max(left_influence, right_influence) + 1e-8)
            print(f"    Peak offset: {peak_offset} frames from perturbation")
            print(f"    Left/right symmetry: {symmetry:.2%}")

    # Test: does perturbation at t affect specific distant frames?
    print("\n\nTesting for attention-like patterns (distant specific influence):")

    z_perturbed = z_flat.clone()
    z_perturbed[:, dim, T//2] += 2.0  # Perturb middle
    perturbed_mel = get_mel_from_z(dcae, z_perturbed.reshape(B, C, H, T), device)
    perturbed_energy = torch.exp(perturbed_mel.mean(dim=1)).sum(dim=1)

    diff = (perturbed_energy - base_energy).squeeze().cpu().numpy()
    t_perturb_mel = (T//2) * 8

    # Check if there are peaks far from perturbation
    distant_threshold = 32  # >32 frames away
    near_influence = np.sum(np.abs(diff[max(0, t_perturb_mel-distant_threshold):min(T_mel, t_perturb_mel+distant_threshold)]))
    far_influence = np.sum(np.abs(diff[:max(0, t_perturb_mel-distant_threshold)])) + \
                   np.sum(np.abs(diff[min(T_mel, t_perturb_mel+distant_threshold):]))

    print(f"  Near influence (±{distant_threshold} frames): {near_influence:.4f}")
    print(f"  Far influence (>{distant_threshold} frames): {far_influence:.4f}")
    print(f"  Ratio near/far: {near_influence/(far_influence+1e-8):.1f}x")

    if far_influence > near_influence * 0.1:
        print("\n  → ATTENTION-LIKE: Significant influence at distant frames")
    else:
        print("\n  → CONVOLUTION-LIKE: Influence is primarily local")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nLoading DCAE...")
    dcae = load_dcae(device)

    # Create base z with more time steps for temporal analysis
    T = 20
    base_z = torch.randn(1, 8, 16, T, device=device) * 0.1

    question1_dims_64_127(dcae, base_z, device)
    question2_quadratic_fit(dcae, base_z, device)
    question3_weak_dims(dcae, base_z, device)
    question4_temporal_structure(dcae, base_z, device)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
