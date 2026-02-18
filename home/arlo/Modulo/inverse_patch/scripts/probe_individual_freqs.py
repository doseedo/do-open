#!/usr/bin/env python3
"""
Probe: Does z encode INDIVIDUAL frequencies, or only AGGREGATES?

Previous discovery: z → centroid (R²=1.0, quadratic)
But centroid is an average. It doesn't tell us individual sines.

This script probes for individual frequency control:
1. Perturb z dims, measure change in mel PEAKS (not just centroid)
2. Check if peaks move independently or all together
3. See if z can control multiple simultaneous frequencies

If peaks move together → z only encodes aggregate (centroid)
If peaks can move independently → z encodes individual frequencies
"""

import torch
import numpy as np
import sys
from scipy.signal import find_peaks
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


def mel_bin_to_hz(bin_idx, n_mels=128, f_min=40, f_max=16000):
    """Convert mel bin index to Hz (approximate)."""
    # Mel scale conversion
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_val = mel_min + (mel_max - mel_min) * bin_idx / n_mels
    hz = 700 * (10 ** (mel_val / 2595) - 1)
    return hz


def find_mel_peaks(mel_frame, height_threshold=0.3, distance=3):
    """Find peaks in a mel spectrogram frame."""
    mel_np = mel_frame.cpu().numpy()

    # Normalize
    mel_norm = (mel_np - mel_np.min()) / (mel_np.max() - mel_np.min() + 1e-8)

    # Find peaks
    peaks, properties = find_peaks(mel_norm, height=height_threshold, distance=distance)

    # Convert to Hz
    peak_hz = [mel_bin_to_hz(p) for p in peaks]
    peak_heights = mel_norm[peaks] if len(peaks) > 0 else []

    return peaks, peak_hz, peak_heights


def probe_peak_independence(dcae, base_z, device='cuda'):
    """
    Test: Do mel peaks move independently or together?

    If dims control individual frequencies, perturbing one dim
    should move one peak while others stay fixed.
    """
    print("\n" + "=" * 70)
    print("PROBE 1: Peak Independence")
    print("Do peaks move independently when z dims are perturbed?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    # Get base mel and peaks
    base_mel = get_mel_from_z(dcae, base_z, device)
    base_mel_mono = base_mel.mean(dim=1)[0]  # [128, T_mel]

    # Look at middle frame
    t_frame = base_mel_mono.shape[1] // 2
    base_frame = base_mel_mono[:, t_frame]

    base_peaks, base_hz, base_heights = find_mel_peaks(base_frame)
    print(f"\nBase peaks at: {[f'{h:.0f} Hz' for h in base_hz[:5]]}")

    # Test dims 48-63 (frequency control dims)
    peak_movements = defaultdict(list)

    for dim in range(48, 64):
        z_perturbed = z_flat.clone()
        z_perturbed[:, dim, :] += 1.0
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d, device)
        perturbed_frame = perturbed_mel.mean(dim=1)[0, :, t_frame]

        perturbed_peaks, perturbed_hz, _ = find_mel_peaks(perturbed_frame)

        # Track how each base peak moved
        for i, (base_p, base_h) in enumerate(zip(base_peaks[:5], base_hz[:5])):
            # Find closest peak in perturbed
            if len(perturbed_hz) > 0:
                closest_idx = np.argmin([abs(ph - base_h) for ph in perturbed_hz])
                movement = perturbed_hz[closest_idx] - base_h
            else:
                movement = 0
            peak_movements[i].append(movement)

    # Analyze: do peaks move together or independently?
    print("\nPeak movements when perturbing dims 48-63:")
    print("-" * 50)

    correlations = []
    for i in range(min(4, len(peak_movements))):
        for j in range(i + 1, min(5, len(peak_movements))):
            if len(peak_movements[i]) > 0 and len(peak_movements[j]) > 0:
                corr = np.corrcoef(peak_movements[i], peak_movements[j])[0, 1]
                correlations.append(corr)
                print(f"  Peak {i} vs Peak {j} correlation: {corr:.3f}")

    avg_corr = np.mean(correlations) if correlations else 0
    print(f"\n  Average inter-peak correlation: {avg_corr:.3f}")

    if avg_corr > 0.8:
        print("  → AGGREGATE CONTROL: Peaks move together (centroid shift)")
    elif avg_corr < 0.3:
        print("  → INDEPENDENT CONTROL: Peaks can move separately!")
    else:
        print("  → MIXED: Some independence, some coupling")

    return peak_movements, avg_corr


def probe_harmonic_structure(dcae, base_z, device='cuda'):
    """
    Test: Does z encode harmonic relationships?

    Real instruments have f0 + harmonics (2*f0, 3*f0, etc.)
    If z encodes harmonics, perturbing f0 should shift all harmonics proportionally.
    """
    print("\n" + "=" * 70)
    print("PROBE 2: Harmonic Structure")
    print("Do peaks maintain harmonic ratios when perturbed?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    # Get base peaks
    base_mel = get_mel_from_z(dcae, base_z, device)
    base_frame = base_mel.mean(dim=1)[0, :, base_mel.shape[-1] // 2]
    base_peaks, base_hz, _ = find_mel_peaks(base_frame, height_threshold=0.2)

    if len(base_hz) < 3:
        print("  Not enough peaks to analyze harmonic structure")
        return

    # Check if base peaks are harmonically related
    print(f"\nBase peaks: {[f'{h:.0f}' for h in base_hz[:6]]} Hz")

    # Find potential f0 (lowest significant peak)
    f0_candidate = base_hz[0] if base_hz[0] > 50 else base_hz[1] if len(base_hz) > 1 else 100

    print(f"Potential f0: {f0_candidate:.0f} Hz")
    print(f"Expected harmonics: {[f'{f0_candidate * n:.0f}' for n in range(1, 6)]} Hz")

    # Check ratios
    ratios = [h / f0_candidate for h in base_hz[:6]]
    print(f"Actual ratios to f0: {[f'{r:.2f}' for r in ratios]}")

    # Perturb and check if ratios are preserved
    print("\nPerturbing dims 60-63 (high frequency control):")

    for dim in [60, 61, 62, 63]:
        z_perturbed = z_flat.clone()
        z_perturbed[:, dim, :] += 1.0
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d, device)
        perturbed_frame = perturbed_mel.mean(dim=1)[0, :, perturbed_mel.shape[-1] // 2]
        _, perturbed_hz, _ = find_mel_peaks(perturbed_frame, height_threshold=0.2)

        if len(perturbed_hz) >= 2:
            new_f0 = perturbed_hz[0] if perturbed_hz[0] > 50 else perturbed_hz[1] if len(perturbed_hz) > 1 else f0_candidate
            new_ratios = [h / new_f0 for h in perturbed_hz[:6]]

            f0_shift = new_f0 - f0_candidate
            ratio_change = np.mean([abs(new_ratios[i] - ratios[i]) for i in range(min(len(new_ratios), len(ratios)))])

            print(f"  Dim {dim}: f0 shift = {f0_shift:+.0f} Hz, ratio change = {ratio_change:.3f}")


def probe_multimodal_control(dcae, base_z, device='cuda'):
    """
    Test: Can z create multiple independent frequency components?

    E.g., can we have energy at both 200 Hz AND 1000 Hz independently?
    """
    print("\n" + "=" * 70)
    print("PROBE 3: Multimodal Control")
    print("Can different z regions control different frequency bands?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    base_mel = get_mel_from_z(dcae, base_z, device)
    base_mel_mono = base_mel.mean(dim=1)[0]  # [128, T_mel]
    t_frame = base_mel_mono.shape[1] // 2

    # Define frequency bands
    bands = {
        'low': (0, 32),      # ~40-300 Hz
        'mid': (32, 64),     # ~300-1000 Hz
        'high': (64, 96),    # ~1000-4000 Hz
        'very_high': (96, 128),  # ~4000-16000 Hz
    }

    def get_band_energy(mel_frame, band):
        return mel_frame[band[0]:band[1]].mean().item()

    base_energies = {name: get_band_energy(base_mel_mono[:, t_frame], band)
                     for name, band in bands.items()}

    print(f"\nBase band energies: {base_energies}")

    # Test: which z dims control which bands?
    print("\nDim influence on frequency bands:")
    print("-" * 60)

    # Group dims by their discovered roles
    dim_groups = {
        'z[48-51]': list(range(48, 52)),   # Low band control?
        'z[52-55]': list(range(52, 56)),   # Mid band control?
        'z[56-59]': list(range(56, 60)),   # High band control?
        'z[60-63]': list(range(60, 64)),   # Very high control?
    }

    for group_name, dims in dim_groups.items():
        # Perturb all dims in group together
        z_perturbed = z_flat.clone()
        for dim in dims:
            z_perturbed[:, dim, :] += 0.5
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d, device)
        perturbed_frame = perturbed_mel.mean(dim=1)[0, :, t_frame]

        print(f"\n  {group_name}:")
        for band_name, band in bands.items():
            base_e = base_energies[band_name]
            perturbed_e = get_band_energy(perturbed_frame, band)
            change = perturbed_e - base_e
            pct = (change / (abs(base_e) + 1e-6)) * 100
            bar = "+" * int(min(abs(pct), 50) / 5) if pct > 0 else "-" * int(min(abs(pct), 50) / 5)
            print(f"    {band_name:10s}: {change:+.3f} ({pct:+.1f}%) {bar}")


def probe_phase_relationship(dcae, base_z, device='cuda'):
    """
    Test: Is the peak structure consistent across time?

    If z encodes persistent frequencies, peaks should be similar across frames.
    If peaks vary wildly, the frequency info might be in the vocoder, not z.
    """
    print("\n" + "=" * 70)
    print("PROBE 4: Temporal Peak Consistency")
    print("Are peaks consistent across time frames?")
    print("=" * 70)

    base_mel = get_mel_from_z(dcae, base_z, device)
    base_mel_mono = base_mel.mean(dim=1)[0]  # [128, T_mel]

    T_mel = base_mel_mono.shape[1]

    # Get peaks at multiple time points
    all_peaks = []
    for t in range(0, T_mel, max(1, T_mel // 10)):
        frame = base_mel_mono[:, t]
        _, peak_hz, _ = find_mel_peaks(frame, height_threshold=0.2)
        all_peaks.append(set(int(h / 50) * 50 for h in peak_hz[:5]))  # Quantize to 50 Hz

    # Check overlap
    if len(all_peaks) >= 2:
        overlaps = []
        for i in range(len(all_peaks) - 1):
            if len(all_peaks[i]) > 0 and len(all_peaks[i+1]) > 0:
                overlap = len(all_peaks[i] & all_peaks[i+1]) / len(all_peaks[i] | all_peaks[i+1])
                overlaps.append(overlap)

        avg_overlap = np.mean(overlaps) if overlaps else 0
        print(f"\n  Average peak overlap between adjacent frames: {avg_overlap:.1%}")

        if avg_overlap > 0.7:
            print("  → CONSISTENT: Same frequencies persist across time")
            print("     z encodes stable frequency content")
        else:
            print("  → VARIABLE: Peaks change significantly over time")
            print("     Frequency detail may be temporal, not spectral")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nLoading DCAE...")
    dcae = load_dcae(device)

    # Create base z
    T = 16
    base_z = torch.randn(1, 8, 16, T, device=device) * 0.2  # Slightly larger for more peaks

    peak_movements, avg_corr = probe_peak_independence(dcae, base_z, device)
    probe_harmonic_structure(dcae, base_z, device)
    probe_multimodal_control(dcae, base_z, device)
    probe_phase_relationship(dcae, base_z, device)

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
  If peaks move TOGETHER (high correlation):
    → z encodes aggregate spectral shape only
    → Individual sines emerge from vocoder
    → Can't extract precise sine control from z

  If peaks move INDEPENDENTLY (low correlation):
    → z encodes individual frequencies
    → We haven't found the right mapping yet
    → More probing needed to find the structure

  If peaks have HARMONIC structure:
    → z encodes f0 + harmonic template
    → Our harmonic mapper approach was right
    → Need to find the f0 control more precisely
    """)


if __name__ == "__main__":
    main()
