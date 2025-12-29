#!/usr/bin/env python3
"""
Listening Test for Formant Correction V2

Decode latents to audio for perceptual evaluation:
1. shifted.wav - Input with formant artifacts
2. corrected.wav - After model correction
3. natural.wav - Target (original audio)
"""

import os
import sys
import json
import random
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/pitchshift/v9')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')

from train_paired_distmatch import FormantCorrectorV2


def load_dcae(device: str = 'cuda'):
    """Load the DCAE encoder/decoder."""
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        device_id=int(device.split(':')[-1]) if ':' in device else 0
    )
    components.load_dcae()
    return components.music_dcae


@torch.no_grad()
def decode_latent(dcae, latent, device: str = 'cuda'):
    """Decode latent back to audio."""
    latent = latent.to(device)
    result = dcae.decode(latent)
    if isinstance(result, tuple):
        sr, wavs = result
        audio = wavs[0]
    else:
        audio = result
    return audio.cpu(), 44100


def main():
    device = 'cuda'
    output_dir = Path('/tmp/formant_listening_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = '/mnt/msdd2/pitchshift_v9_formant_pairs/manifest.json'
    checkpoint_path = '/mnt/msdd2/pitchshift_checkpoints/formant_paired_distmatch_v2/best.pt'

    print("Loading DCAE...")
    dcae = load_dcae(device)

    print("Loading FormantCorrectorV2...")
    model = FormantCorrectorV2(
        hidden_channels=256,
        num_blocks=8,
        use_attention=True,
        direct_output=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, loss={checkpoint['loss']:.4f}")

    # Load manifest
    with open(manifest_path) as f:
        data = json.load(f)

    pairs = data['pairs']

    # Select samples: 2 UP, 2 DOWN
    up_pairs = [p for p in pairs if p['direction'] == 1]
    down_pairs = [p for p in pairs if p['direction'] == 0]

    random.seed(42)
    selected = random.sample(up_pairs, 2) + random.sample(down_pairs, 2)

    print(f"\nProcessing {len(selected)} samples...")

    for i, pair in enumerate(selected):
        pair_path = pair['pair_path']
        direction = pair['direction']
        direction_name = "UP" if direction == 1 else "DOWN"

        print(f"\n[{i+1}/{len(selected)}] {direction_name}: {Path(pair_path).name}")

        # Load pair data
        pair_data = torch.load(pair_path, map_location='cpu')
        shifted = pair_data['shifted'].unsqueeze(0).to(device)  # [1, C, H, T]
        natural = pair_data['natural'].unsqueeze(0).to(device)

        print(f"  Latent shape: {shifted.shape}")

        # Run through model
        direction_t = torch.tensor([direction], device=device)
        corrected = model(shifted, direction_t)

        # Decode all three
        print("  Decoding shifted...")
        shifted_audio, sr = decode_latent(dcae, shifted, device)

        print("  Decoding corrected...")
        corrected_audio, sr = decode_latent(dcae, corrected, device)

        print("  Decoding natural...")
        natural_audio, sr = decode_latent(dcae, natural, device)

        # Save
        sample_dir = output_dir / f"sample_{i+1}_{direction_name}"
        sample_dir.mkdir(exist_ok=True)

        torchaudio.save(str(sample_dir / "1_shifted.wav"), shifted_audio.squeeze(0), sr)
        torchaudio.save(str(sample_dir / "2_corrected.wav"), corrected_audio.squeeze(0), sr)
        torchaudio.save(str(sample_dir / "3_natural.wav"), natural_audio.squeeze(0), sr)

        # Compute latent-space metrics
        shifted_cpu = shifted.cpu()
        corrected_cpu = corrected.cpu()
        natural_cpu = natural.cpu()

        baseline_dist = (shifted_cpu - natural_cpu).abs().mean().item()
        corrected_dist = (corrected_cpu - natural_cpu).abs().mean().item()
        improvement = (baseline_dist - corrected_dist) / baseline_dist * 100

        print(f"  Baseline L1: {baseline_dist:.4f}")
        print(f"  Corrected L1: {corrected_dist:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")

        # Write info file
        with open(sample_dir / "info.txt", 'w') as f:
            f.write(f"Direction: {direction_name}\n")
            f.write(f"Pair: {pair_path}\n")
            f.write(f"Baseline L1: {baseline_dist:.4f}\n")
            f.write(f"Corrected L1: {corrected_dist:.4f}\n")
            f.write(f"Improvement: {improvement:+.1f}%\n")

    print(f"\n{'='*60}")
    print(f"Listening test samples saved to: {output_dir}")
    print(f"{'='*60}")
    print("\nFor each sample, compare:")
    print("  1_shifted.wav   - Input with formant artifacts")
    print("  2_corrected.wav - After model correction")
    print("  3_natural.wav   - Target (original audio)")


if __name__ == '__main__':
    main()
