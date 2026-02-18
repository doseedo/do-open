#!/usr/bin/env python3
"""
Proper SMS synthesis with all components:
- 128 sines with phases
- 8 noise bands

Test if full SMS → audio preserves quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
import orjson


def synthesize_sms_full(freqs, amps, phases, noise_amps, sr=44100, hop_length=512):
    """
    Full SMS synthesis with sines + phases + noise.

    Args:
        freqs: [T, n_sines] Hz
        amps: [T, n_sines]
        phases: [T, n_sines]
        noise_amps: [T, n_noise_bands]
        sr: sample rate
        hop_length: samples per frame
    """
    T, n_sines = freqs.shape
    _, n_noise_bands = noise_amps.shape
    n_samples = T * hop_length

    # ===== SINES =====
    # Upsample parameters
    freqs_up = F.interpolate(
        freqs.T.unsqueeze(0), size=n_samples, mode='linear', align_corners=True
    ).squeeze(0).T  # [n_samples, n_sines]

    amps_up = F.interpolate(
        amps.T.unsqueeze(0), size=n_samples, mode='linear', align_corners=True
    ).squeeze(0).T

    phases_up = F.interpolate(
        phases.T.unsqueeze(0), size=n_samples, mode='linear', align_corners=True
    ).squeeze(0).T

    # Phase accumulation from initial phases
    dt = 1.0 / sr
    phase_inc = 2 * np.pi * freqs_up * dt
    # Start from provided phases at each frame boundary, interpolate between
    # Simpler: just use cumsum + initial offset
    phase_accum = torch.cumsum(phase_inc, dim=0)
    # Add interpolated initial phases
    phase_total = phase_accum + phases_up

    # Synthesize sines
    sines = amps_up * torch.sin(phase_total)
    audio_sines = sines.sum(dim=1)

    # ===== NOISE =====
    # Generate noise in frequency bands
    # 8 bands spanning roughly 0-22kHz (ERB-like spacing)
    band_edges = [0, 100, 300, 600, 1200, 2400, 4800, 9600, 22050]

    # Upsample noise amps
    noise_up = F.interpolate(
        noise_amps.T.unsqueeze(0), size=n_samples, mode='linear', align_corners=True
    ).squeeze(0).T  # [n_samples, n_noise_bands]

    # Generate filtered noise for each band
    audio_noise = torch.zeros(n_samples)

    from scipy import signal

    for band_idx in range(n_noise_bands):
        low_freq = band_edges[band_idx]
        high_freq = band_edges[band_idx + 1]

        # Generate white noise
        np.random.seed(band_idx)
        noise = np.random.randn(n_samples)

        # Normalize frequencies
        nyq = sr / 2
        low = max(low_freq / nyq, 0.001)
        high = min(high_freq / nyq, 0.999)

        if low >= high:
            continue

        try:
            if low < 0.01:
                b, a = signal.butter(4, high, btype='low')
            else:
                b, a = signal.butter(4, [low, high], btype='band')

            noise_filtered = signal.filtfilt(b, a, noise)

            # Normalize filtered noise to unit RMS, then apply envelope
            rms = np.sqrt(np.mean(noise_filtered**2))
            if rms > 0:
                noise_filtered = noise_filtered / rms

            noise_filtered = torch.from_numpy(noise_filtered.astype(np.float32))

            # Apply time-varying amplitude envelope
            audio_noise += noise_filtered * noise_up[:, band_idx]

        except Exception as e:
            print(f"Band {band_idx} failed: {e}")

    # Scale noise relative to sines (noise_amps are in same scale as sines)
    # Normalize total noise energy to match the ratio in the data
    sines_energy = amps.sum().item()
    noise_energy = noise_amps.sum().item()
    if noise_energy > 0:
        target_ratio = noise_energy / (sines_energy + noise_energy)
        current_sines_rms = np.sqrt(np.mean(audio_sines.numpy()**2))
        current_noise_rms = np.sqrt(np.mean(audio_noise.numpy()**2))
        if current_noise_rms > 0:
            desired_noise_rms = current_sines_rms * (target_ratio / (1 - target_ratio + 1e-8))
            audio_noise = audio_noise * (desired_noise_rms / current_noise_rms)

    # ===== COMBINE =====
    audio = audio_sines + audio_noise

    # Handle NaN/inf
    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio.numpy()


def synthesize_sms_sines_only(freqs, amps, phases, sr=44100, hop_length=512):
    """Synthesize only sines (no noise) for comparison."""
    T, n_sines = freqs.shape
    n_samples = T * hop_length

    freqs_up = F.interpolate(
        freqs.T.unsqueeze(0), size=n_samples, mode='linear', align_corners=True
    ).squeeze(0).T

    amps_up = F.interpolate(
        amps.T.unsqueeze(0), size=n_samples, mode='linear', align_corners=True
    ).squeeze(0).T

    phases_up = F.interpolate(
        phases.T.unsqueeze(0), size=n_samples, mode='linear', align_corners=True
    ).squeeze(0).T

    dt = 1.0 / sr
    phase_inc = 2 * np.pi * freqs_up * dt
    phase_accum = torch.cumsum(phase_inc, dim=0)
    phase_total = phase_accum + phases_up

    sines = amps_up * torch.sin(phase_total)
    audio = sines.sum(dim=1)
    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio.numpy()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load DCAE for original audio
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    )
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    # Load SMS data
    sms_path = 'data/sms_v4/sms_000011.pt'
    data = torch.load(sms_path, weights_only=True, map_location='cpu')

    freqs = data['freqs']  # [T, 128]
    amps = data['amps']
    phases = data['phases']
    noise_amps = data['noise_amps']  # [T, 8]
    lat_path = data['latent_path']

    print(f"SMS data:")
    print(f"  Freqs: {freqs.shape}")
    print(f"  Amps: {amps.shape}")
    print(f"  Phases: {phases.shape}")
    print(f"  Noise: {noise_amps.shape}")

    # Get original DCAE audio
    print("\nGenerating original DCAE audio...")
    lat_data = torch.load(lat_path, weights_only=True, map_location='cpu')
    z = lat_data.get('latents', lat_data)
    if z.dim() == 3:
        z = z.unsqueeze(0)
    z = z.to(device)

    with torch.no_grad():
        z_denorm = z / dcae.scale_factor + dcae.shift_factor
        mel_dcae = dcae.dcae.decoder(z_denorm).mean(dim=1)
        mel_scaled = mel_dcae * 0.5 + 0.5
        mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value
        audio_original = dcae.vocoder.decode(mel_scaled).squeeze()
        audio_original = audio_original / (audio_original.abs().max() + 1e-8) * 0.9
    audio_original_np = audio_original.cpu().numpy()

    # Synthesize from SMS
    print("\nSynthesizing from SMS (sines only)...")
    audio_sines = synthesize_sms_sines_only(freqs, amps, phases)

    print("Synthesizing from SMS (sines + noise)...")
    audio_full = synthesize_sms_full(freqs, amps, phases, noise_amps)

    # Match lengths
    min_len = min(len(audio_original_np), len(audio_sines), len(audio_full))
    audio_original_np = audio_original_np[:min_len]
    audio_sines = audio_sines[:min_len]
    audio_full = audio_full[:min_len]

    # Save
    out_dir = 'outputs/proper_sms'
    os.makedirs(out_dir, exist_ok=True)

    sf.write(f'{out_dir}/1_original_dcae.wav', audio_original_np, 44100)
    sf.write(f'{out_dir}/2_sms_sines_only.wav', audio_sines, 44100)
    sf.write(f'{out_dir}/3_sms_sines_plus_noise.wav', audio_full, 44100)

    # Metrics
    print("\nSpectral comparison:")

    def get_centroid(audio):
        return np.mean(librosa.feature.spectral_centroid(y=audio, sr=44100))

    c_orig = get_centroid(audio_original_np)
    c_sines = get_centroid(audio_sines)
    c_full = get_centroid(audio_full)

    print(f"  Original DCAE: {c_orig:.0f} Hz")
    print(f"  SMS sines only: {c_sines:.0f} Hz ({c_sines/c_orig*100:.0f}%)")
    print(f"  SMS sines+noise: {c_full:.0f} Hz ({c_full/c_orig*100:.0f}%)")

    # Energy comparison
    print("\nEnergy comparison:")
    print(f"  Sines energy: {amps.sum().item():.3f}")
    print(f"  Noise energy: {noise_amps.sum().item():.3f}")
    print(f"  Noise / Total: {noise_amps.sum().item() / (amps.sum().item() + noise_amps.sum().item()) * 100:.1f}%")

    print(f"\nSaved to {out_dir}/")
    print("  1_original_dcae.wav - Ground truth")
    print("  2_sms_sines_only.wav - 128 sines with phases (no noise)")
    print("  3_sms_sines_plus_noise.wav - 128 sines + 8 noise bands")


if __name__ == "__main__":
    main()
