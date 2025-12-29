#!/usr/bin/env python3
"""
Inference with Cycle-Consistent Register Translator

Uses G_h2l (HIGH → LOW) for formant transfer, then sox for pitch shift.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, '/home/arlo/Data/pitchshift/v6')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')

from models import RegisterTranslatorDirect


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def load_cycle_model(checkpoint_path: str, device: str = 'cuda'):
    """Load G_h2l from cycle-trained checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = RegisterTranslatorDirect()
    model.load_state_dict(checkpoint['G_h2l_state_dict'])
    model = model.to(device).eval()

    return model


def sox_pitch_shift(input_path: str, output_path: str, semitones: float):
    cents = int(semitones * 100)
    cmd = ['sox', input_path, output_path, 'pitch', str(cents)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sox failed: {result.stderr}")


def generate_listening_test(
    model,
    dcae,
    manifest_path: str,
    output_dir: str,
    num_samples: int = 10,
    shift_semitones: float = -12,
    device: str = 'cuda',
):
    import json
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get high register trumpet entries
    entries = []
    for e in manifest:
        if e.get('sub_group') != 'trumpet':
            continue

        latent_path = fix_path(e.get('latent_path', ''))
        cond = e.get('conditioning_paths') or {}
        f0_path = fix_path(cond.get('f0', ''))

        if not latent_path or not f0_path:
            continue
        if not os.path.exists(latent_path) or not os.path.exists(f0_path):
            continue

        try:
            f0 = np.load(f0_path)
            f0_valid = f0[f0 > 20]
            if len(f0_valid) < 10:
                continue
            median_midi = 12 * np.log2(np.median(f0_valid) / 440) + 69
        except:
            continue

        # Only high register (> 70)
        if median_midi > 70:
            entries.append({
                'latent_path': latent_path,
                'f0_path': f0_path,
                'median_midi': median_midi,
            })

    step = max(1, len(entries) // num_samples)
    selected = [entries[i * step] for i in range(min(num_samples, len(entries)))]

    print(f"Generating {len(selected)} listening test samples...")

    for i, entry in enumerate(tqdm(selected)):
        latent_data = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
        if isinstance(latent_data, dict):
            latent = latent_data.get('latents', latent_data.get('latent'))
        else:
            latent = latent_data
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)

        T = min(latent.shape[-1], 256)
        latent = latent[:, :, :, :T].to(device)

        source_midi = entry['median_midi']
        target_midi = source_midi + shift_semitones

        # 1. Original
        with torch.no_grad():
            sr, wavs = dcae.decode(latent)
        original_audio = wavs[0]
        original_path = os.path.join(output_dir, f"{i:02d}_original_pitch{source_midi:.0f}.wav")
        torchaudio.save(original_path, original_audio.cpu(), sr)

        # 2. Naive sox
        naive_path = os.path.join(output_dir, f"{i:02d}_naive_sox_to{target_midi:.0f}.wav")
        sox_pitch_shift(original_path, naive_path, shift_semitones)

        # 3. Translated formants (cycle model)
        with torch.no_grad():
            translated = model(latent)
            sr, wavs = dcae.decode(translated)
        translated_audio = wavs[0]
        translated_path = os.path.join(output_dir, f"{i:02d}_translated_formants.wav")
        torchaudio.save(translated_path, translated_audio.cpu(), sr)

        # 4. Final (translated + sox)
        final_path = os.path.join(output_dir, f"{i:02d}_final_to{target_midi:.0f}.wav")
        sox_pitch_shift(translated_path, final_path, shift_semitones)

        print(f"  Sample {i}: {source_midi:.0f} → {target_midi:.0f}")

    print(f"\nListening test saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--shift', type=float, default=-12)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print(f"Loading cycle model from: {args.checkpoint}")
    model = load_cycle_model(args.checkpoint, args.device)

    print("Loading DCAE...")
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        device_id=0
    )
    components.load_dcae()
    dcae = components.music_dcae

    generate_listening_test(
        model=model,
        dcae=dcae,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        shift_semitones=args.shift,
        device=args.device,
    )


if __name__ == "__main__":
    main()
