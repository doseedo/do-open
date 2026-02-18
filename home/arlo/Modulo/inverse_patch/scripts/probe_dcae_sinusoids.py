#!/usr/bin/env python3
"""
Probe DCAE's Latent Space for Sinusoidal Structure

Key questions:
1. Does z change smoothly with frequency?
2. Does z scale linearly with amplitude?
3. Is DCAE linear: z(A + B) ≈ z(A) + z(B)?
4. Is there a "frequency axis" in z-space?
5. Can we decode pure sinusoids from z manipulation?

If DCAE learned a linear frequency representation, we can build
sinusoid-based controls directly on top of it.
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False

SAMPLE_RATE = 44100
DURATION = 2.0

DCAE_PATH = "/home/arlo/Data/ACE-Step"
if DCAE_PATH not in sys.path:
    sys.path.insert(0, DCAE_PATH)

DEFAULT_DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
DEFAULT_VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_dcae(device: str):
    codec = MusicDCAE(
        source_sample_rate=SAMPLE_RATE,
        dcae_checkpoint_path=DEFAULT_DCAE_PATH,
        vocoder_checkpoint_path=DEFAULT_VOCODER_PATH,
    ).to(device)
    codec.eval()
    return codec


def encode_audio(codec, audio: np.ndarray, device='cuda'):
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    audio_stereo = audio_tensor.expand(-1, 2, -1).to(device)
    audio_lengths = torch.tensor([audio_stereo.shape[-1]], device=device)
    with torch.no_grad():
        latent, _ = codec.encode(audio_stereo, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return latent


def decode_latent(codec, z, device='cuda'):
    if z.dim() == 3:
        z = z.unsqueeze(0)
    audio_lengths = torch.tensor([int(DURATION * SAMPLE_RATE)], device=device)
    with torch.no_grad():
        sr, pred_wavs = codec.decode(z, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return pred_wavs[0].mean(dim=0).cpu().numpy()


def generate_sinusoid(freq: float, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Generate pure sinusoid."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), dtype=np.float32)
    audio = amp * np.sin(2 * np.pi * freq * t + phase)
    return audio.astype(np.float32)


def generate_two_sinusoids(f1: float, f2: float, a1: float = 0.5, a2: float = 0.5) -> np.ndarray:
    """Generate sum of two sinusoids."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), dtype=np.float32)
    audio = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    return audio.astype(np.float32)


# ============================================================
# Test 1: Frequency Continuity
# ============================================================

def test_frequency_continuity(codec, device: str, output_dir: Path):
    """Does z change smoothly with frequency?"""

    print("\n" + "="*60)
    print("Test 1: Frequency Continuity")
    print("="*60)

    frequencies = np.logspace(np.log10(110), np.log10(4400), 30)  # 110 Hz to 4.4 kHz, log-spaced
    z_samples = []

    for freq in frequencies:
        audio = generate_sinusoid(freq, amp=0.8)
        z = encode_audio(codec, audio, device)
        z_samples.append(z.cpu().numpy().flatten())

    Z = np.array(z_samples)  # [30, latent_dim]

    # Compute pairwise distances
    distances = []
    for i in range(len(frequencies) - 1):
        dist = np.linalg.norm(Z[i+1] - Z[i])
        distances.append(dist)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Distance vs frequency step
    axes[0, 0].plot(frequencies[1:], distances, 'b.-')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('z distance to previous')
    axes[0, 0].set_title('Z-space Distance Between Adjacent Frequencies')
    axes[0, 0].set_xscale('log')

    # PCA of z
    pca = PCA(n_components=3)
    Z_pca = pca.fit_transform(Z)

    axes[0, 1].scatter(Z_pca[:, 0], Z_pca[:, 1], c=np.log10(frequencies), cmap='viridis')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('PCA of Z (color = log frequency)')
    axes[0, 1].colorbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])

    # Correlation: can we predict frequency from z?
    reg = Ridge(alpha=1.0)
    reg.fit(Z, np.log10(frequencies))
    freq_pred = reg.predict(Z)
    r2 = 1 - ((np.log10(frequencies) - freq_pred)**2).sum() / ((np.log10(frequencies) - np.log10(frequencies).mean())**2).sum()

    axes[1, 0].scatter(np.log10(frequencies), freq_pred, alpha=0.7)
    axes[1, 0].plot([np.log10(frequencies).min(), np.log10(frequencies).max()],
                    [np.log10(frequencies).min(), np.log10(frequencies).max()], 'r--')
    axes[1, 0].set_xlabel('True log(frequency)')
    axes[1, 0].set_ylabel('Predicted log(frequency)')
    axes[1, 0].set_title(f'Frequency Extraction R² = {r2:.4f}')

    # 3D trajectory
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.plot(Z_pca[:, 0], Z_pca[:, 1], Z_pca[:, 2], 'b.-')
    ax3d.scatter(Z_pca[0, 0], Z_pca[0, 1], Z_pca[0, 2], c='green', s=100, label='Low freq')
    ax3d.scatter(Z_pca[-1, 0], Z_pca[-1, 1], Z_pca[-1, 2], c='red', s=100, label='High freq')
    ax3d.set_xlabel('PC1')
    ax3d.set_ylabel('PC2')
    ax3d.set_zlabel('PC3')
    ax3d.set_title('Z Trajectory Through Frequencies')
    ax3d.legend()

    plt.tight_layout()
    plt.savefig(str(output_dir / 'frequency_continuity.png'), dpi=150)
    plt.close()

    print(f"  Frequency extraction R² = {r2:.4f}")
    if r2 > 0.9:
        print("  → Frequency is LINEARLY EXTRACTABLE from z!")
    elif r2 > 0.7:
        print("  → Frequency is partially extractable")
    else:
        print("  → Frequency encoding is highly nonlinear")

    return r2


# ============================================================
# Test 2: Amplitude Linearity
# ============================================================

def test_amplitude_linearity(codec, device: str, output_dir: Path):
    """Does z scale linearly with amplitude?"""

    print("\n" + "="*60)
    print("Test 2: Amplitude Linearity")
    print("="*60)

    freq = 440  # Fixed frequency
    amplitudes = np.linspace(0.1, 1.0, 10)
    z_samples = []

    for amp in amplitudes:
        audio = generate_sinusoid(freq, amp=amp)
        z = encode_audio(codec, audio, device)
        z_samples.append(z.cpu().numpy().flatten())

    Z = np.array(z_samples)

    # If linear: z(2*a) ≈ 2*z(a) → z should scale proportionally
    # Check: ||z||/amp should be constant

    norms = np.linalg.norm(Z, axis=1)
    ratios = norms / amplitudes

    # Variance of ratios (should be low if linear)
    ratio_var = np.var(ratios) / np.mean(ratios)**2

    print(f"  ||z||/amplitude ratio: mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}")
    print(f"  Normalized variance: {ratio_var:.4f}")

    # Linear regression: amp → z
    reg = Ridge(alpha=1.0)
    reg.fit(amplitudes.reshape(-1, 1), Z)
    Z_pred = reg.predict(amplitudes.reshape(-1, 1))

    ss_res = ((Z - Z_pred)**2).sum()
    ss_tot = ((Z - Z.mean(axis=0))**2).sum()
    r2 = 1 - ss_res / ss_tot

    print(f"  Amplitude → z R² = {r2:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(amplitudes, norms, 'b.-')
    axes[0].set_xlabel('Amplitude')
    axes[0].set_ylabel('||z||')
    axes[0].set_title('Z Norm vs Amplitude')

    axes[1].plot(amplitudes, ratios, 'b.-')
    axes[1].axhline(np.mean(ratios), color='r', linestyle='--', label='Mean')
    axes[1].set_xlabel('Amplitude')
    axes[1].set_ylabel('||z|| / Amplitude')
    axes[1].set_title(f'Ratio (should be constant if linear)')
    axes[1].legend()

    # PCA trajectory
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)
    axes[2].scatter(Z_pca[:, 0], Z_pca[:, 1], c=amplitudes, cmap='viridis', s=100)
    for i, amp in enumerate(amplitudes):
        axes[2].annotate(f'{amp:.1f}', (Z_pca[i, 0], Z_pca[i, 1]))
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].set_title('Z Trajectory with Amplitude')

    plt.tight_layout()
    plt.savefig(str(output_dir / 'amplitude_linearity.png'), dpi=150)
    plt.close()

    if ratio_var < 0.01:
        print("  → Amplitude scales LINEARLY!")
    elif ratio_var < 0.1:
        print("  → Amplitude scaling is approximately linear")
    else:
        print("  → Amplitude scaling is nonlinear")

    return r2


# ============================================================
# Test 3: Superposition Linearity
# ============================================================

def test_superposition_linearity(codec, device: str, output_dir: Path):
    """Is z(A + B) ≈ z(A) + z(B)?"""

    print("\n" + "="*60)
    print("Test 3: Superposition Linearity")
    print("="*60)

    # Test pairs of frequencies
    freq_pairs = [
        (220, 440),   # Octave
        (220, 330),   # Fifth
        (440, 880),   # Octave
        (440, 660),   # Fifth
        (110, 440),   # Two octaves
    ]

    errors = []

    for f1, f2 in freq_pairs:
        # Individual sinusoids
        audio_1 = generate_sinusoid(f1, amp=0.5)
        audio_2 = generate_sinusoid(f2, amp=0.5)
        audio_sum = generate_two_sinusoids(f1, f2, a1=0.5, a2=0.5)

        z_1 = encode_audio(codec, audio_1, device)
        z_2 = encode_audio(codec, audio_2, device)
        z_sum = encode_audio(codec, audio_sum, device)

        # Predicted sum
        z_sum_pred = z_1 + z_2

        # Error
        error = (z_sum - z_sum_pred).norm().item() / z_sum.norm().item()
        errors.append(error)

        print(f"  {f1}Hz + {f2}Hz: linearity error = {error:.4f}")

    mean_error = np.mean(errors)
    print(f"\n  Mean linearity error: {mean_error:.4f}")

    if mean_error < 0.1:
        print("  → DCAE is approximately LINEAR in superposition!")
        print("  → z ≈ linear transform of frequency content")
    elif mean_error < 0.3:
        print("  → DCAE has some linear structure")
    else:
        print("  → DCAE is NONLINEAR - z(A+B) ≠ z(A) + z(B)")

    return mean_error


# ============================================================
# Test 4: Frequency Axis Discovery
# ============================================================

def test_frequency_axis(codec, device: str, output_dir: Path):
    """Can we find a linear axis that corresponds to frequency?"""

    print("\n" + "="*60)
    print("Test 4: Frequency Axis Discovery")
    print("="*60)

    # Generate sinusoids at many frequencies
    frequencies = np.logspace(np.log10(110), np.log10(2200), 20)
    z_samples = []

    for freq in frequencies:
        audio = generate_sinusoid(freq, amp=0.8)
        z = encode_audio(codec, audio, device)
        z_samples.append(z.squeeze(0).cpu())  # Remove batch dim

    Z = torch.stack(z_samples)  # [20, C, H, T]
    Z_flat = Z.reshape(len(frequencies), -1).numpy()

    # Find direction that maximizes correlation with log-frequency
    log_freqs = np.log10(frequencies)

    # Linear regression to find direction
    reg = Ridge(alpha=1.0)
    reg.fit(log_freqs.reshape(-1, 1), Z_flat)

    direction = reg.coef_.flatten()
    direction = direction / np.linalg.norm(direction)

    # Project all z onto this direction
    projections = Z_flat @ direction

    # Correlation
    corr = np.corrcoef(projections, log_freqs)[0, 1]
    print(f"  Frequency direction correlation: {corr:.4f}")

    # Test: can we SYNTHESIZE frequency changes by moving along this direction?
    print("\n  Testing frequency synthesis along discovered axis...")

    # Start from middle frequency
    mid_idx = len(frequencies) // 2
    z_base = Z[mid_idx].clone()

    # Move along direction
    direction_tensor = torch.from_numpy(direction).float().reshape(Z[0].shape)

    test_deltas = [-2, -1, 0, 1, 2]
    scale = np.std(projections)

    for delta in test_deltas:
        z_moved = z_base + delta * scale * direction_tensor.to(z_base.device)
        audio = decode_latent(codec, z_moved.to(device), device)

        filename = f"freq_axis_delta{delta:+d}.wav"
        torchaudio.save(str(output_dir / filename),
                       torch.from_numpy(audio).unsqueeze(0).float(), SAMPLE_RATE)

    print(f"  Saved frequency axis test files")
    print(f"  Listen: does delta+2 sound higher pitched than delta-2?")

    return corr


# ============================================================
# Test 5: Harmonic Structure
# ============================================================

def test_harmonic_structure(codec, device: str, output_dir: Path):
    """Does DCAE preserve harmonic relationships?"""

    print("\n" + "="*60)
    print("Test 5: Harmonic Structure")
    print("="*60)

    # Generate fundamental and harmonics
    f0 = 220  # Fundamental
    harmonics = [f0 * n for n in range(1, 9)]  # 8 harmonics

    z_harmonics = []
    for h in harmonics:
        if h < SAMPLE_RATE / 2:
            audio = generate_sinusoid(h, amp=0.8)
            z = encode_audio(codec, audio, device)
            z_harmonics.append(z.cpu().numpy().flatten())

    Z_harm = np.array(z_harmonics)

    # Check: are harmonic latents equidistant in some direction?
    pca = PCA(n_components=3)
    Z_pca = pca.fit_transform(Z_harm)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PCA trajectory
    axes[0].plot(Z_pca[:, 0], Z_pca[:, 1], 'b.-', markersize=10)
    for i, h in enumerate(harmonics[:len(Z_harm)]):
        axes[0].annotate(f'H{i+1}\n{h}Hz', (Z_pca[i, 0], Z_pca[i, 1]))
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Harmonic Series in Z-Space')

    # Distance between consecutive harmonics
    dists = [np.linalg.norm(Z_harm[i+1] - Z_harm[i]) for i in range(len(Z_harm)-1)]
    axes[1].bar(range(1, len(dists)+1), dists)
    axes[1].set_xlabel('Harmonic Transition (H_n → H_{n+1})')
    axes[1].set_ylabel('Z-Space Distance')
    axes[1].set_title('Distance Between Adjacent Harmonics')

    plt.tight_layout()
    plt.savefig(str(output_dir / 'harmonic_structure.png'), dpi=150)
    plt.close()

    # Check linearity of harmonic progression
    harmonic_numbers = np.array(range(1, len(Z_harm)+1))
    reg = Ridge(alpha=1.0)
    reg.fit(harmonic_numbers.reshape(-1, 1), Z_harm)
    Z_pred = reg.predict(harmonic_numbers.reshape(-1, 1))

    ss_res = ((Z_harm - Z_pred)**2).sum()
    ss_tot = ((Z_harm - Z_harm.mean(axis=0))**2).sum()
    r2 = 1 - ss_res / ss_tot

    print(f"  Harmonic number → z R² = {r2:.4f}")

    if r2 > 0.9:
        print("  → Harmonics are LINEARLY ORGANIZED in z-space!")
    elif r2 > 0.7:
        print("  → Partial linear harmonic structure")
    else:
        print("  → Harmonic structure is nonlinear")

    return r2


# ============================================================
# Test 6: Decode Pure Sinusoids
# ============================================================

def test_decode_interpolation(codec, device: str, output_dir: Path):
    """Can we decode interpolated z values to get interpolated frequencies?"""

    print("\n" + "="*60)
    print("Test 6: Decode Interpolated Z")
    print("="*60)

    # Two reference frequencies
    f_low = 220
    f_high = 880

    audio_low = generate_sinusoid(f_low, amp=0.8)
    audio_high = generate_sinusoid(f_high, amp=0.8)

    z_low = encode_audio(codec, audio_low, device)
    z_high = encode_audio(codec, audio_high, device)

    # Interpolate
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"  Interpolating between {f_low}Hz and {f_high}Hz:")

    for alpha in alphas:
        z_interp = (1 - alpha) * z_low + alpha * z_high
        audio_interp = decode_latent(codec, z_interp, device)

        expected_freq = f_low * (f_high/f_low)**alpha  # Log interpolation
        filename = f"interp_z_{int(alpha*100):03d}_expect{int(expected_freq)}hz.wav"
        torchaudio.save(str(output_dir / filename),
                       torch.from_numpy(audio_interp).unsqueeze(0).float(), SAMPLE_RATE)

        print(f"    alpha={alpha:.2f} → expected ~{expected_freq:.0f}Hz")

    print("  Listen: do the interpolated files sound like intermediate frequencies?")
    print("  Or do they have artifacts (the 'blob' problem)?")


# ============================================================
# Main
# ============================================================

def main():
    print("="*60)
    print("Probing DCAE for Sinusoidal Structure")
    print("="*60)
    print("\nKey question: Does DCAE encode sinusoids linearly?")
    print("If yes, we can build sinusoid controls on top of DCAE.")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.cuda.empty_cache()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    output_dir = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/sinusoid_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading DCAE...")
    codec = load_dcae(device)
    print("DCAE loaded!")

    # Run all tests
    results = {}

    results['freq_r2'] = test_frequency_continuity(codec, device, output_dir)
    clear_memory()

    results['amp_r2'] = test_amplitude_linearity(codec, device, output_dir)
    clear_memory()

    results['superposition_error'] = test_superposition_linearity(codec, device, output_dir)
    clear_memory()

    results['freq_axis_corr'] = test_frequency_axis(codec, device, output_dir)
    clear_memory()

    results['harmonic_r2'] = test_harmonic_structure(codec, device, output_dir)
    clear_memory()

    test_decode_interpolation(codec, device, output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Is DCAE Sinusoidally Structured?")
    print("="*60)

    print(f"\n  Frequency extractable (z → freq): R² = {results['freq_r2']:.4f}")
    print(f"  Amplitude linear: R² = {results['amp_r2']:.4f}")
    print(f"  Superposition linear: error = {results['superposition_error']:.4f}")
    print(f"  Frequency axis exists: corr = {results['freq_axis_corr']:.4f}")
    print(f"  Harmonics linear: R² = {results['harmonic_r2']:.4f}")

    # Overall verdict
    linear_score = (
        (results['freq_r2'] > 0.8) +
        (results['amp_r2'] > 0.8) +
        (results['superposition_error'] < 0.2) +
        (abs(results['freq_axis_corr']) > 0.9) +
        (results['harmonic_r2'] > 0.8)
    )

    print(f"\n  Linear tests passed: {linear_score}/5")

    if linear_score >= 4:
        print("\n  VERDICT: DCAE has STRONG sinusoidal structure!")
        print("  → Can build sinusoid controls directly on z-space")
        print("  → No need for DDSP - use DCAE + linear mappings")
    elif linear_score >= 2:
        print("\n  VERDICT: DCAE has PARTIAL sinusoidal structure")
        print("  → Some linear operations work")
        print("  → Need nonlinear mappings for full control")
    else:
        print("\n  VERDICT: DCAE is highly NONLINEAR")
        print("  → Sinusoidal structure is buried deep")
        print("  → Need learned transforms or new codec")

    print(f"\n  Outputs saved to: {output_dir}")
    print("  Check the .png files for visualizations")
    print("  Listen to the .wav files for audio tests")


if __name__ == "__main__":
    main()
