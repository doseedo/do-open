#!/usr/bin/env python3
"""
Compare operation-based synthesis with DCAE across varied z values.
"""

import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

from operation_synthesizer import OperationSynthesizer


def compute_spectral_features(audio, sr=44100):
    """Compute spectral features for comparison."""
    from scipy import signal

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=2048, noverlap=1536)

    # Spectral centroid
    power = Sxx + 1e-10
    centroid = np.sum(f[:, None] * power, axis=0) / np.sum(power, axis=0)
    avg_centroid = np.mean(centroid)

    # Band energies
    low_idx = f < 300
    mid_idx = (f >= 300) & (f < 3000)
    high_idx = f >= 3000

    low_energy = np.sum(Sxx[low_idx])
    mid_energy = np.sum(Sxx[mid_idx])
    high_energy = np.sum(Sxx[high_idx])
    total = low_energy + mid_energy + high_energy + 1e-10

    return {
        'centroid': avg_centroid,
        'low_ratio': low_energy / total,
        'mid_ratio': mid_energy / total,
        'high_ratio': high_energy / total,
        'total_energy': np.sum(Sxx)
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load DCAE
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    )
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    synth = OperationSynthesizer(sr=44100)

    # Test with varied z patterns
    tests = [
        ('high_harm_presence', 48, 2.0),   # High harmonic presence
        ('low_harm_presence', 48, -2.0),   # Low harmonic presence
        ('high_f0', 115, 2.0),             # High f0
        ('low_f0', 115, -2.0),             # Low f0
        ('high_centroid', 86, 2.0),        # Bright
        ('low_centroid', 70, 2.0),         # Dark (negative centroid dim)
        ('fast_decay', 22, 2.0),           # Fast decay
        ('slow_decay', 22, -2.0),          # Slow decay
    ]

    results = []
    out_dir = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/outputs/comparison'
    os.makedirs(out_dir, exist_ok=True)

    for name, dim, value in tests:
        # Create z with specific pattern
        z = torch.zeros(1, 8, 16, 32, device=device)
        z_flat = z.reshape(1, 128, 32)
        z_flat[:, dim, :] = value
        z = z_flat.reshape(1, 8, 16, 32)

        # Operation synthesis
        audio_ops, program = synth.synthesize(z)
        audio_ops_np = audio_ops[0].cpu().numpy()

        # DCAE synthesis
        with torch.no_grad():
            z_denorm = z / dcae.scale_factor + dcae.shift_factor
            mel = dcae.dcae.decoder(z_denorm).mean(dim=1)
            mel_scaled = mel * 0.5 + 0.5
            mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value
            audio_dcae = dcae.vocoder.decode(mel_scaled).squeeze()
            audio_dcae = audio_dcae / (audio_dcae.abs().max() + 1e-8) * 0.9
        audio_dcae_np = audio_dcae.cpu().numpy()

        # Compute spectral features
        ops_features = compute_spectral_features(audio_ops_np)
        dcae_features = compute_spectral_features(audio_dcae_np)

        # Save audio
        sf.write(f'{out_dir}/{name}_ops.wav', audio_ops_np, 44100)
        sf.write(f'{out_dir}/{name}_dcae.wav', audio_dcae_np, 44100)

        print(f"\n{name} (dim {dim} = {value}):")
        print(f"  Program: f0={program['f0'][0]:.1f}Hz, n_harm={program['n_harmonics'][0]}, "
              f"centroid={program['harmonic_centroid'][0]:.2f}, decay={program['decay_rate'][0]:.3f}")
        print(f"  Ops spectral: centroid={ops_features['centroid']:.0f}Hz, "
              f"low={ops_features['low_ratio']:.2f}, mid={ops_features['mid_ratio']:.2f}")
        print(f"  DCAE spectral: centroid={dcae_features['centroid']:.0f}Hz, "
              f"low={dcae_features['low_ratio']:.2f}, mid={dcae_features['mid_ratio']:.2f}")

        results.append({
            'name': name,
            'dim': dim,
            'value': value,
            'program': program,
            'ops_features': ops_features,
            'dcae_features': dcae_features
        })

    # Plot comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, r in enumerate(results):
        ax = axes[i // 4, i % 4]
        ax.bar(['Ops', 'DCAE'], [r['ops_features']['centroid'], r['dcae_features']['centroid']])
        ax.set_title(r['name'])
        ax.set_ylabel('Spectral Centroid (Hz)')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/spectral_comparison.png', dpi=150)
    print(f"\nSaved comparison plot to {out_dir}/spectral_comparison.png")


if __name__ == "__main__":
    main()
