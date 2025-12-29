#!/usr/bin/env python3
"""
Inference with Formant Corrector (v7)

Pipeline:
1. HIGH audio → sox shift DOWN → corrupted LOW
2. Encode corrupted → model fixes formants → corrected latent
3. Decode → natural-sounding LOW audio
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data/pitchshift/v7')
sys.path.insert(0, '/home/arlo/Data/pitchshift/v6')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')

from models import RegisterTranslatorDirect


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def load_model(checkpoint_path: str, device: str = 'cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RegisterTranslatorDirect()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    return model


def sox_pitch_shift(input_path: str, output_path: str, semitones: float):
    cents = int(semitones * 100)
    subprocess.run(['sox', input_path, output_path, 'pitch', str(cents)],
                   capture_output=True, check=True)


def sox_pitch_shift_tensor(waveform: torch.Tensor, sr: int, semitones: float) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_in:
        in_path = f_in.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
        out_path = f_out.name

    try:
        torchaudio.save(in_path, waveform, sr)
        sox_pitch_shift(in_path, out_path, semitones)
        shifted, _ = torchaudio.load(out_path)
        return shifted
    finally:
        os.unlink(in_path)
        os.unlink(out_path)


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

    os.makedirs(output_dir, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get HIGH register entries
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

        if median_midi > 70:
            entries.append({
                'latent_path': latent_path,
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

        # 1. Original HIGH audio
        with torch.no_grad():
            sr, wavs = dcae.decode(latent)
        original_audio = wavs[0].cpu()
        original_path = os.path.join(output_dir, f"{i:02d}_original_pitch{source_midi:.0f}.wav")
        torchaudio.save(original_path, original_audio, sr)

        # 2. Naive sox shift (wrong formants)
        naive_path = os.path.join(output_dir, f"{i:02d}_naive_sox_to{target_midi:.0f}.wav")
        sox_pitch_shift(original_path, naive_path, shift_semitones)

        # 3. v7 pipeline: sox shift → encode → model → decode
        # a) Sox shift audio
        shifted_audio = sox_pitch_shift_tensor(original_audio, sr, shift_semitones)

        # b) Encode shifted audio
        with torch.no_grad():
            shifted_latent = dcae.encode(shifted_audio.unsqueeze(0).to(device))

        # c) Model corrects formants
        with torch.no_grad():
            corrected_latent = model(shifted_latent)

        # d) Decode corrected
        with torch.no_grad():
            sr, wavs = dcae.decode(corrected_latent)
        corrected_audio = wavs[0].cpu()

        corrected_path = os.path.join(output_dir, f"{i:02d}_corrected_to{target_midi:.0f}.wav")
        torchaudio.save(corrected_path, corrected_audio, sr)

        # 4. Also save the sox-shifted-only (before correction) for comparison
        shifted_only_path = os.path.join(output_dir, f"{i:02d}_shifted_uncorrected.wav")
        torchaudio.save(shifted_only_path, shifted_audio, sr)

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

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

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
