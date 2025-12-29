#!/usr/bin/env python3
"""
Check DCAE encode/decode accuracy by comparing:
1. Original source audio
2. Decoded 'natural' latent from training pair

This tells us how much information is lost in the latent representation.
"""

import os
import sys
import json
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/pitchshift/v9')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')


def load_dcae(device: str = 'cuda'):
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        device_id=int(device.split(':')[-1]) if ':' in device else 0
    )
    components.load_dcae()
    return components.music_dcae


@torch.no_grad()
def decode_latent(dcae, latent, device: str = 'cuda'):
    latent = latent.to(device)
    result = dcae.decode(latent)
    if isinstance(result, tuple):
        sr, wavs = result
        audio = wavs[0]
    else:
        audio = result
    return audio.cpu(), 44100


def load_and_resample(path, target_sr=44100):
    """Load audio and resample to target SR."""
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio


def compute_metrics(original, decoded):
    """Compute audio similarity metrics."""
    # Align lengths
    min_len = min(original.shape[-1], decoded.shape[-1])
    original = original[..., :min_len]
    decoded = decoded[..., :min_len]

    # L1 distance
    l1 = (original - decoded).abs().mean().item()

    # MSE
    mse = ((original - decoded) ** 2).mean().item()

    # Correlation
    orig_flat = original.flatten()
    dec_flat = decoded.flatten()
    corr = torch.corrcoef(torch.stack([orig_flat, dec_flat]))[0, 1].item()

    # SNR (signal to noise ratio)
    signal_power = (original ** 2).mean().item()
    noise_power = ((original - decoded) ** 2).mean().item()
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    return {
        'l1': l1,
        'mse': mse,
        'correlation': corr,
        'snr_db': snr_db,
    }


def main():
    device = 'cuda'
    output_dir = Path('/tmp/dcae_accuracy_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = '/mnt/msdd2/pitchshift_v9_formant_pairs/manifest.json'

    print("Loading DCAE...")
    dcae = load_dcae(device)

    # Load manifest
    with open(manifest_path) as f:
        data = json.load(f)

    pairs = data['pairs']

    # Filter to pairs where source audio exists
    valid_pairs = []
    for p in pairs:
        pair_data = torch.load(p['pair_path'], map_location='cpu')
        source_path = pair_data.get('source_audio', '')
        if source_path and os.path.exists(source_path):
            valid_pairs.append((p, source_path))

    print(f"Found {len(valid_pairs)} pairs with existing source audio")

    # Sample 10 random pairs
    random.seed(123)
    selected = random.sample(valid_pairs, min(10, len(valid_pairs)))

    all_metrics = []

    print(f"\nTesting {len(selected)} samples...")
    print("=" * 70)

    for i, (pair, source_path) in enumerate(selected):
        pair_path = pair['pair_path']
        pair_data = torch.load(pair_path, map_location='cpu')

        natural_latent = pair_data['natural'].unsqueeze(0).to(device)

        print(f"\n[{i+1}/{len(selected)}] {Path(source_path).name}")
        print(f"  Latent shape: {natural_latent.shape}")

        # Decode latent
        decoded_audio, sr = decode_latent(dcae, natural_latent, device)
        decoded_audio = decoded_audio.squeeze(0)  # [2, T]

        # Load original
        try:
            original_audio = load_and_resample(source_path, sr)
        except Exception as e:
            print(f"  ERROR loading original: {e}")
            continue

        print(f"  Original: {original_audio.shape}, Decoded: {decoded_audio.shape}")

        # Compute metrics
        metrics = compute_metrics(original_audio, decoded_audio)
        all_metrics.append(metrics)

        print(f"  L1: {metrics['l1']:.4f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  SNR: {metrics['snr_db']:.1f} dB")

        # Save first 3 for listening
        if i < 3:
            sample_dir = output_dir / f"sample_{i+1}"
            sample_dir.mkdir(exist_ok=True)

            # Trim to same length
            min_len = min(original_audio.shape[-1], decoded_audio.shape[-1])

            torchaudio.save(str(sample_dir / "1_original.wav"), original_audio[..., :min_len], sr)
            torchaudio.save(str(sample_dir / "2_decoded.wav"), decoded_audio[..., :min_len], sr)

            with open(sample_dir / "info.txt", 'w') as f:
                f.write(f"Source: {source_path}\n")
                f.write(f"L1: {metrics['l1']:.4f}\n")
                f.write(f"Correlation: {metrics['correlation']:.4f}\n")
                f.write(f"SNR: {metrics['snr_db']:.1f} dB\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - DCAE Encode/Decode Accuracy")
    print("=" * 70)

    avg_l1 = np.mean([m['l1'] for m in all_metrics])
    avg_corr = np.mean([m['correlation'] for m in all_metrics])
    avg_snr = np.mean([m['snr_db'] for m in all_metrics])

    print(f"Average L1:          {avg_l1:.4f}")
    print(f"Average Correlation: {avg_corr:.4f}")
    print(f"Average SNR:         {avg_snr:.1f} dB")
    print(f"\nSamples saved to: {output_dir}")


if __name__ == '__main__':
    main()
