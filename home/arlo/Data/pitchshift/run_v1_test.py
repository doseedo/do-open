#!/usr/bin/env python3
"""
Test V1 pitch shift model with multiple shift amounts.
Generates: original, librosa-shifted, model-corrected for ±3, ±6, ±12 semitones.
"""

import os
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/dø')

from models_pitch_shift import RegisterAwareTranslator


def load_dcae():
    """Load DCAE model."""
    from do.pipeline_do import DoTrainComponents
    comps = DoTrainComponents(checkpoint_dir='/home/arlo/Data/ACE-Step/checkpoints', dtype="float32")
    dcae = comps.load_dcae()
    return dcae.cuda().eval()


def load_translator(checkpoint_path):
    """Load translator model."""
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model = RegisterAwareTranslator()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.cuda().eval()


def pitch_shift_audio(audio_tensor, sr, semitones):
    """Pitch shift using torchaudio (phase vocoder)."""
    if semitones == 0:
        return audio_tensor

    # torchaudio pitch_shift expects [channels, samples]
    shifted = F_audio.pitch_shift(audio_tensor, sr, semitones)
    return shifted


def estimate_pitch(audio_tensor, sr):
    """Estimate dominant MIDI pitch - just return default for trumpet."""
    # For simplicity, return a typical trumpet pitch
    return 70  # Bb4


@torch.no_grad()
def run_test(input_wav, output_dir, checkpoint_path):
    """Run full test with multiple shift amounts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading DCAE...")
    dcae = load_dcae()
    print("Loading translator...")
    translator = load_translator(checkpoint_path)

    # Load audio
    print(f"Loading audio: {input_wav}")
    audio, sr = torchaudio.load(input_wav)

    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    # Estimate pitch
    source_pitch = estimate_pitch(audio, sr)
    print(f"Estimated source pitch: {source_pitch} MIDI")

    # Save original
    stem = Path(input_wav).stem
    orig_path = output_dir / f"{stem}_00_original.wav"
    torchaudio.save(str(orig_path), audio, sr)
    print(f"Saved: {orig_path}")

    # Test shifts
    shifts = [-12, -6, -3, 3, 6, 12]

    results = {'original': str(orig_path)}

    for shift in shifts:
        print(f"\n=== Processing shift {shift:+d} semitones ===")

        # 1. Pitch shift using torchaudio
        shifted = pitch_shift_audio(audio, sr, shift)
        shifted = shifted / (shifted.abs().max() + 1e-8) * 0.9

        naive_path = output_dir / f"{stem}_{shift:+03d}_pitchshift.wav"
        torchaudio.save(str(naive_path), shifted, sr)
        print(f"Saved pitch-shifted: {naive_path}")

        # 2. Model corrected
        shifted_tensor = shifted.unsqueeze(0).cuda()

        # Encode to latent
        latent_out = dcae.encode(shifted_tensor, sr=sr)
        # DCAE returns tuple - extract the latent
        latent = latent_out[0] if isinstance(latent_out, tuple) else latent_out

        # Apply translator
        target_pitch = max(48, min(96, source_pitch + shift))
        target_tensor = torch.tensor([target_pitch], device='cuda')
        shift_tensor = torch.tensor([float(shift)], device='cuda')

        corrected_latent = translator(latent, target_tensor, shift_tensor)

        # Decode - returns (sr, [audio_tensor])
        decode_out = dcae.decode(corrected_latent)
        corrected_audio = decode_out[1][0]  # Get audio from list
        corrected_audio = corrected_audio.squeeze(0).cpu()
        corrected_audio = corrected_audio / (corrected_audio.abs().max() + 1e-8) * 0.9

        corrected_path = output_dir / f"{stem}_{shift:+03d}_corrected.wav"
        torchaudio.save(str(corrected_path), corrected_audio, 44100)
        print(f"Saved corrected: {corrected_path}")

        results[f'shift_{shift:+d}_naive'] = str(naive_path)
        results[f'shift_{shift:+d}_corrected'] = str(corrected_path)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\nFiles generated:")
    for name, path in sorted(results.items()):
        print(f"  {name}: {Path(path).name}")

    return results


if __name__ == "__main__":
    # Find a good trumpet test file
    import glob
    trumpet_files = glob.glob("/mnt/msdd2/mutedtptaudiofiles/*.wav")[:5]

    if not trumpet_files:
        print("No trumpet files found!")
        sys.exit(1)

    # Use first one
    test_file = trumpet_files[0]
    print(f"Using test file: {test_file}")

    run_test(
        input_wav=test_file,
        output_dir="/home/arlo/Data/pitchshift/v1_test_output",
        checkpoint_path="/mnt/msdd/pitchshift_checkpoints/best.pt",
    )
