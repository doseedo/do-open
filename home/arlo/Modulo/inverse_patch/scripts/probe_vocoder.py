#!/usr/bin/env python3
"""
Probe the VOCODER: Discover mel → sines mapping.

We discovered: z → band energies (quadratic)
Now discover: mel bins → output sines

Questions:
1. Which mel bins control which output frequencies?
2. Is the mapping linear, quadratic, or something else?
3. Do bins interact or are they independent?
4. Is there a deterministic mel_bin → Hz relationship?

This completes the chain:
z → [quadratic] → mel → [to discover] → sines
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from collections import defaultdict

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE


def load_dcae(device='cuda'):
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH)
    dcae.dcae.to(device)
    dcae.dcae.eval()
    # Also load vocoder
    dcae.vocoder.to(device)
    dcae.vocoder.eval()
    return dcae


def mel_bin_to_hz(bin_idx, n_mels=128, f_min=40, f_max=16000):
    """Convert mel bin index to center frequency in Hz."""
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_val = mel_min + (mel_max - mel_min) * bin_idx / n_mels
    hz = 700 * (10 ** (mel_val / 2595) - 1)
    return hz


def hz_to_mel_bin(hz, n_mels=128, f_min=40, f_max=16000):
    """Convert Hz to mel bin index."""
    mel_val = 2595 * np.log10(1 + hz / 700)
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    bin_idx = (mel_val - mel_min) / (mel_max - mel_min) * n_mels
    return bin_idx


def extract_spectrum_peaks(audio, sr=44100, n_peaks=10):
    """Extract dominant frequencies from audio using FFT."""
    audio_np = audio.cpu().numpy().flatten()

    # FFT
    n_fft = min(4096, len(audio_np))
    spectrum = np.abs(rfft(audio_np, n=n_fft))
    freqs = rfftfreq(n_fft, 1/sr)

    # Find peaks
    peaks, properties = find_peaks(spectrum, height=spectrum.max() * 0.1, distance=5)

    if len(peaks) == 0:
        return [], []

    # Sort by amplitude
    peak_amps = spectrum[peaks]
    sorted_idx = np.argsort(peak_amps)[::-1][:n_peaks]
    top_peaks = peaks[sorted_idx]

    peak_freqs = freqs[top_peaks]
    peak_amps = spectrum[top_peaks]

    return peak_freqs.tolist(), peak_amps.tolist()


def vocoder_decode(dcae, mel, device='cuda'):
    """Run vocoder on mel spectrogram."""
    with torch.no_grad():
        # Vocoder expects [B, n_mels, T]
        # Our mel is [B, 2, 128, T] (stereo)
        # Use mono for simplicity
        mel_mono = mel.mean(dim=1)  # [B, 128, T]

        # Denormalize mel (from [-1, 1] to actual range)
        mel_denorm = mel_mono * 0.5 + 0.5  # [0, 1]
        mel_denorm = mel_denorm * (3.0 - (-11.0)) + (-11.0)  # [-11, 3]

        audio = dcae.vocoder.decode(mel_denorm)
        return audio


def probe_single_mel_bins(dcae, device='cuda'):
    """
    Probe: Which mel bins control which output frequencies?

    Perturb each mel bin, measure which frequencies appear in output.
    """
    print("\n" + "=" * 70)
    print("PROBE 1: Single Mel Bin → Output Frequencies")
    print("=" * 70)

    # Create a quiet baseline mel
    T_mel = 64  # ~0.75 seconds
    base_mel = torch.ones(1, 2, 128, T_mel, device=device) * (-10.0)  # Very quiet

    # For each mel bin, add energy and see what frequencies appear
    bin_to_freqs = {}

    print("\nProbing mel bins (every 4th bin)...")
    for mel_bin in range(0, 128, 4):
        # Add energy to this bin
        test_mel = base_mel.clone()
        test_mel[:, :, mel_bin, :] = 0.0  # Add energy (0 is moderate in log scale)

        # Decode with vocoder
        try:
            audio = vocoder_decode(dcae, test_mel, device)
            audio_np = audio[0].cpu()

            # Extract dominant frequencies
            peak_freqs, peak_amps = extract_spectrum_peaks(audio_np, sr=44100, n_peaks=5)

            # Expected frequency for this mel bin
            expected_hz = mel_bin_to_hz(mel_bin)

            if peak_freqs:
                closest_peak = min(peak_freqs, key=lambda f: abs(f - expected_hz))
                error_hz = abs(closest_peak - expected_hz)
                error_pct = (error_hz / expected_hz) * 100 if expected_hz > 0 else 0

                bin_to_freqs[mel_bin] = {
                    'expected_hz': expected_hz,
                    'actual_peaks': peak_freqs[:3],
                    'closest_peak': closest_peak,
                    'error_hz': error_hz,
                    'error_pct': error_pct,
                }

                if mel_bin % 16 == 0:
                    print(f"  Bin {mel_bin:3d}: expected {expected_hz:6.0f} Hz, "
                          f"got {closest_peak:6.0f} Hz (error: {error_pct:.1f}%)")

        except Exception as e:
            print(f"  Bin {mel_bin}: Error - {e}")

    # Analyze
    print("\n" + "-" * 50)
    print("ANALYSIS: Mel Bin → Output Frequency Mapping")

    errors = [v['error_pct'] for v in bin_to_freqs.values() if v['error_pct'] < 100]
    if errors:
        print(f"  Average error: {np.mean(errors):.1f}%")
        print(f"  Median error:  {np.median(errors):.1f}%")
        print(f"  Max error:     {np.max(errors):.1f}%")

        if np.median(errors) < 20:
            print("\n  → DETERMINISTIC: Mel bin maps predictably to output Hz")
        else:
            print("\n  → NON-DETERMINISTIC: Mel bin → Hz mapping is noisy")

    return bin_to_freqs


def probe_mel_linearity(dcae, device='cuda'):
    """
    Probe: Is the mel energy → output amplitude mapping linear?
    """
    print("\n" + "=" * 70)
    print("PROBE 2: Mel Energy → Output Amplitude (Linearity)")
    print("=" * 70)

    T_mel = 64
    test_bin = 64  # Mid-frequency bin (~1000 Hz)
    expected_hz = mel_bin_to_hz(test_bin)

    print(f"\nTesting bin {test_bin} (expected ~{expected_hz:.0f} Hz)")

    energies = []
    amplitudes = []

    # Test range of mel energies
    for mel_energy in np.linspace(-10, 2, 13):
        base_mel = torch.ones(1, 2, 128, T_mel, device=device) * (-10.0)
        base_mel[:, :, test_bin, :] = mel_energy

        try:
            audio = vocoder_decode(dcae, base_mel, device)
            audio_np = audio[0].cpu().numpy().flatten()

            # Measure amplitude at expected frequency
            n_fft = 4096
            spectrum = np.abs(rfft(audio_np, n=n_fft))
            freqs = rfftfreq(n_fft, 1/44100)

            # Find bin closest to expected frequency
            freq_bin = np.argmin(np.abs(freqs - expected_hz))
            amplitude = spectrum[freq_bin]

            energies.append(mel_energy)
            amplitudes.append(amplitude)

        except Exception as e:
            print(f"  Energy {mel_energy}: Error - {e}")

    # Analyze linearity
    if len(energies) > 3:
        energies = np.array(energies)
        amplitudes = np.array(amplitudes)

        # Linear fit
        coeffs = np.polyfit(energies, amplitudes, 1)
        linear_pred = np.polyval(coeffs, energies)
        linear_r2 = 1 - np.sum((amplitudes - linear_pred)**2) / np.sum((amplitudes - amplitudes.mean())**2)

        # Exponential fit (since mel is log-energy)
        log_amps = np.log(amplitudes + 1e-10)
        exp_coeffs = np.polyfit(energies, log_amps, 1)
        exp_pred = np.exp(np.polyval(exp_coeffs, energies))
        exp_r2 = 1 - np.sum((amplitudes - exp_pred)**2) / np.sum((amplitudes - amplitudes.mean())**2)

        print(f"\n  Linear fit R²:      {linear_r2:.4f}")
        print(f"  Exponential fit R²: {exp_r2:.4f}")

        if exp_r2 > linear_r2 and exp_r2 > 0.9:
            print(f"\n  → EXPONENTIAL: amplitude = exp({exp_coeffs[0]:.3f} * mel_energy)")
        elif linear_r2 > 0.9:
            print(f"\n  → LINEAR: amplitude = {coeffs[0]:.3f} * mel_energy + {coeffs[1]:.3f}")
        else:
            print("\n  → COMPLEX: Neither linear nor exponential fits well")

    return energies, amplitudes


def probe_bin_independence(dcae, device='cuda'):
    """
    Probe: Do mel bins interact or are they independent?

    If independent: energy(bin_A + bin_B) = energy(bin_A) + energy(bin_B)
    """
    print("\n" + "=" * 70)
    print("PROBE 3: Mel Bin Independence")
    print("=" * 70)

    T_mel = 64
    base_mel = torch.ones(1, 2, 128, T_mel, device=device) * (-10.0)

    # Test two bins
    bin_a = 48   # ~500 Hz
    bin_b = 80   # ~2500 Hz
    hz_a = mel_bin_to_hz(bin_a)
    hz_b = mel_bin_to_hz(bin_b)

    print(f"\nTesting bins {bin_a} ({hz_a:.0f} Hz) and {bin_b} ({hz_b:.0f} Hz)")

    def get_energy_at_freq(audio, target_hz):
        audio_np = audio[0].cpu().numpy().flatten()
        n_fft = 4096
        spectrum = np.abs(rfft(audio_np, n=n_fft))
        freqs = rfftfreq(n_fft, 1/44100)
        freq_bin = np.argmin(np.abs(freqs - target_hz))
        return spectrum[freq_bin]

    # Measure energy with only bin A
    mel_a = base_mel.clone()
    mel_a[:, :, bin_a, :] = 0.0
    audio_a = vocoder_decode(dcae, mel_a, device)
    energy_a_at_a = get_energy_at_freq(audio_a, hz_a)
    energy_a_at_b = get_energy_at_freq(audio_a, hz_b)

    # Measure energy with only bin B
    mel_b = base_mel.clone()
    mel_b[:, :, bin_b, :] = 0.0
    audio_b = vocoder_decode(dcae, mel_b, device)
    energy_b_at_a = get_energy_at_freq(audio_b, hz_a)
    energy_b_at_b = get_energy_at_freq(audio_b, hz_b)

    # Measure energy with both bins
    mel_ab = base_mel.clone()
    mel_ab[:, :, bin_a, :] = 0.0
    mel_ab[:, :, bin_b, :] = 0.0
    audio_ab = vocoder_decode(dcae, mel_ab, device)
    energy_ab_at_a = get_energy_at_freq(audio_ab, hz_a)
    energy_ab_at_b = get_energy_at_freq(audio_ab, hz_b)

    print(f"\n  Energy at {hz_a:.0f} Hz:")
    print(f"    Only bin A:    {energy_a_at_a:.4f}")
    print(f"    Only bin B:    {energy_b_at_a:.4f}")
    print(f"    Both A+B:      {energy_ab_at_a:.4f}")
    print(f"    Expected sum:  {energy_a_at_a + energy_b_at_a:.4f}")

    print(f"\n  Energy at {hz_b:.0f} Hz:")
    print(f"    Only bin A:    {energy_a_at_b:.4f}")
    print(f"    Only bin B:    {energy_b_at_b:.4f}")
    print(f"    Both A+B:      {energy_ab_at_b:.4f}")
    print(f"    Expected sum:  {energy_a_at_b + energy_b_at_b:.4f}")

    # Check additivity
    diff_a = abs(energy_ab_at_a - (energy_a_at_a + energy_b_at_a))
    diff_b = abs(energy_ab_at_b - (energy_a_at_b + energy_b_at_b))

    relative_diff_a = diff_a / (energy_ab_at_a + 1e-10)
    relative_diff_b = diff_b / (energy_ab_at_b + 1e-10)

    print(f"\n  Additivity error:")
    print(f"    At {hz_a:.0f} Hz: {relative_diff_a:.1%}")
    print(f"    At {hz_b:.0f} Hz: {relative_diff_b:.1%}")

    if relative_diff_a < 0.2 and relative_diff_b < 0.2:
        print("\n  → INDEPENDENT: Bins don't interact significantly")
    else:
        print("\n  → INTERACTING: Bins affect each other's output")


def probe_harmonic_generation(dcae, device='cuda'):
    """
    Probe: Does exciting one mel bin create harmonics?

    The vocoder might generate harmonics from a single mel bin input.
    """
    print("\n" + "=" * 70)
    print("PROBE 4: Harmonic Generation")
    print("Does single mel bin create harmonics?")
    print("=" * 70)

    T_mel = 64
    test_bin = 32  # ~250 Hz (good for seeing harmonics)
    expected_hz = mel_bin_to_hz(test_bin)

    print(f"\nExciting only bin {test_bin} (expected ~{expected_hz:.0f} Hz)")

    base_mel = torch.ones(1, 2, 128, T_mel, device=device) * (-10.0)
    base_mel[:, :, test_bin, :] = 0.0  # Add energy to just this bin

    audio = vocoder_decode(dcae, base_mel, device)
    peak_freqs, peak_amps = extract_spectrum_peaks(audio[0].cpu(), sr=44100, n_peaks=10)

    print(f"\n  Input: Single mel bin at ~{expected_hz:.0f} Hz")
    print(f"  Output frequencies: {[f'{f:.0f}' for f in peak_freqs]} Hz")

    if len(peak_freqs) >= 2:
        # Check if peaks are harmonically related
        f0 = min(peak_freqs)
        ratios = [f / f0 for f in peak_freqs]
        harmonic_ratios = [round(r) for r in ratios]

        print(f"\n  Potential f0: {f0:.0f} Hz")
        print(f"  Ratios to f0: {[f'{r:.2f}' for r in ratios]}")
        print(f"  Nearest integers: {harmonic_ratios}")

        # Check if ratios are close to integers
        harmonic_errors = [abs(r - round(r)) for r in ratios]
        avg_error = np.mean(harmonic_errors)

        if avg_error < 0.1:
            print(f"\n  → HARMONIC: Vocoder generates harmonic series from single bin")
        else:
            print(f"\n  → NON-HARMONIC: Multiple frequencies but not harmonically related")
    else:
        print("\n  → SINGLE FREQUENCY: No harmonics generated")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nLoading DCAE + Vocoder...")
    dcae = load_dcae(device)

    bin_to_freqs = probe_single_mel_bins(dcae, device)
    energies, amplitudes = probe_mel_linearity(dcae, device)
    probe_bin_independence(dcae, device)
    probe_harmonic_generation(dcae, device)

    print("\n" + "=" * 70)
    print("CONCLUSIONS: Mel → Sines Mapping")
    print("=" * 70)
    print("""
  If mel bin → output Hz is DETERMINISTIC:
    → We can predict exact sine frequencies from mel
    → Chain: z → bands → mel bins → sine Hz

  If amplitude mapping is EXPONENTIAL:
    → sine_amp = exp(k * mel_energy)
    → Matches log-mel representation

  If bins are INDEPENDENT:
    → Each mel bin can be processed separately
    → No complex interactions to model

  If vocoder creates HARMONICS:
    → Single mel bin → multiple sines
    → Need to account for this in mapping
    """)


if __name__ == "__main__":
    main()
