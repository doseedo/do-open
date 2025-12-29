#!/usr/bin/env python3
"""
Test V8 pitch shift model.
Generates: original, pitch-shifted, V8-corrected for various shift amounts.

V8 improvements over V2:
- Better noise reduction in silence (higher silence_weight)
- Better pitch preservation (content_loss for residual model)
- Stronger transformation (residual_scale=0.3)
"""

import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/pitchshift/v8')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')

from models_v8 import RegisterTranslator


def load_dcae():
    """Load DCAE model."""
    from do.pipeline_do import DoTrainComponents
    comps = DoTrainComponents(checkpoint_dir='/home/arlo/Data/ACE-Step/checkpoints', dtype="float32")
    dcae = comps.load_dcae()
    return dcae.cuda().eval()


def load_v8_model(checkpoint_path):
    """Load V8 translator model."""
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    model = RegisterTranslator()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.cuda().eval()


def estimate_pitch_from_audio(audio_tensor, sr):
    """Estimate dominant MIDI pitch from audio."""
    # Simple approach: use a typical trumpet range
    # In production, use a pitch detector
    return 70  # Bb4 - typical trumpet pitch


@torch.no_grad()
def run_test(input_wav, output_dir, checkpoint_path):
    """Run full test with multiple shift amounts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading DCAE...")
    dcae = load_dcae()
    print("Loading V8 model...")
    model = load_v8_model(checkpoint_path)

    # Load audio
    print(f"Loading audio: {input_wav}")
    audio, sr = torchaudio.load(input_wav)

    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    # Estimate source pitch
    source_pitch = estimate_pitch_from_audio(audio, sr)
    print(f"Source pitch: {source_pitch} MIDI")

    # Save original
    stem = Path(input_wav).stem
    orig_path = output_dir / f"{stem}_00_original.wav"
    torchaudio.save(str(orig_path), audio, sr)
    print(f"Saved: {orig_path}")

    # Test shifts
    shifts = [-12, -6, -3, 3, 6, 12]

    for shift in shifts:
        print(f"\n=== Processing shift {shift:+d} semitones ===")

        # 1. Pitch shift using torchaudio
        shifted = F_audio.pitch_shift(audio, sr, shift)
        shifted = shifted / (shifted.abs().max() + 1e-8) * 0.9

        naive_path = output_dir / f"{stem}_{shift:+03d}_pitchshift.wav"
        torchaudio.save(str(naive_path), shifted, sr)
        print(f"Saved pitch-shifted: {naive_path}")

        # 2. V8 model corrected
        shifted_tensor = shifted.unsqueeze(0).cuda()

        # Encode to latent
        latent_out = dcae.encode(shifted_tensor, sr=sr)
        latent = latent_out[0] if isinstance(latent_out, tuple) else latent_out

        # Apply V8 translator
        # Source pitch = where the audio IS now (after pitch shift)
        # Target pitch = where we want the timbre to match
        shifted_pitch = source_pitch + shift
        shifted_pitch = max(48, min(96, shifted_pitch))

        src_tensor = torch.tensor([shifted_pitch], device='cuda')
        tgt_tensor = torch.tensor([shifted_pitch], device='cuda')  # Same - we want natural timbre at this pitch

        corrected_latent = model(latent, src_tensor, tgt_tensor)

        # Decode
        decode_out = dcae.decode(corrected_latent)
        corrected_audio = decode_out[1][0]
        corrected_audio = corrected_audio.squeeze(0).cpu()
        corrected_audio = corrected_audio / (corrected_audio.abs().max() + 1e-8) * 0.9

        corrected_path = output_dir / f"{stem}_{shift:+03d}_v8corrected.wav"
        torchaudio.save(str(corrected_path), corrected_audio, 44100)
        print(f"Saved V8 corrected: {corrected_path}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("\nCompare files:")
    print("  *_pitchshift.wav  = naive pitch shift (may have artifacts)")
    print("  *_v8corrected.wav = V8 timbre-corrected (improved noise handling)")


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Input wav file')
    parser.add_argument('--output_dir', type=str, default='/home/arlo/Data/pitchshift/v8_test_output')
    parser.add_argument('--checkpoint', type=str, default='/mnt/msdd2/pitchshift_checkpoints/v8/best.pt')
    args = parser.parse_args()

    if args.input:
        test_file = args.input
    else:
        # Find a trumpet test file
        trumpet_files = glob.glob("/mnt/msdd2/mutedtptaudiofiles/*.wav")[:5]
        if not trumpet_files:
            print("No trumpet files found!")
            sys.exit(1)
        test_file = trumpet_files[0]

    print(f"Using test file: {test_file}")

    run_test(
        input_wav=test_file,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
    )
