#!/usr/bin/env python3
"""
Register Translator Inference

Pipeline for pitch-corrected register transfer:
1. HIGH register audio → DCAE encode → HIGH latent
2. HIGH latent → Register Translator → LOW-formant latent (still high pitch in latent)
3. LOW-formant latent → DCAE decode → audio with low formants but high pitch
4. Audio → sox pitch shift DOWN → correct pitch with correct formants

Key insight: sox shift preserves formants, model applies formant transfer.
Combining them: high pitch + sox down = low pitch, high formants + model = low formants
Result: low pitch + low formants = realistic low register sound
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

sys.path.insert(0, '/home/arlo/Data/pitchshift/v6')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')

from models import RegisterTranslator, RegisterTranslatorDirect, RegisterTranslatorAdaptive, RegisterTranslator2D


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained register translator."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', 'residual')

    if model_type == '2d':
        model = RegisterTranslator2D()
    elif model_type == 'direct':
        model = RegisterTranslatorDirect()
    elif model_type == 'adaptive':
        model = RegisterTranslatorAdaptive()
    else:
        model = RegisterTranslator()

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    config = {
        'model_type': model_type,
        'split_midi': checkpoint.get('split_midi', 65.0),
    }

    return model, config


def sox_pitch_shift(input_path: str, output_path: str, semitones: float):
    """
    Apply pitch shift using sox.

    Sox pitch shift preserves formants (unlike speed change).
    Negative semitones = shift down, positive = shift up.
    """
    # sox pitch shift uses cents: 100 cents = 1 semitone
    cents = int(semitones * 100)

    cmd = [
        'sox', input_path, output_path,
        'pitch', str(cents)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sox failed: {result.stderr}")


def process_single(
    model,
    dcae,
    audio_path: str,
    output_dir: str,
    shift_semitones: float = -12,  # Default: shift down one octave
    device: str = 'cuda',
):
    """
    Process single audio file through register translator.

    Args:
        model: Register translator model
        dcae: DCAE model for encode/decode
        audio_path: Path to input audio
        output_dir: Directory for outputs
        shift_semitones: How many semitones to shift (negative = down)
        device: Device to use
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = Path(audio_path).stem

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)
        sr = 44100

    # Encode to latent
    with torch.no_grad():
        latent = dcae.encode(waveform.unsqueeze(0).to(device))
        # latent: [B, C, H, T]

    # Apply register translator
    with torch.no_grad():
        translated = model(latent)

    # Decode to audio
    with torch.no_grad():
        _, wavs = dcae.decode(translated)
        translated_audio = wavs[0]  # [C, samples]

    # Save intermediate (translated but not pitch-shifted)
    intermediate_path = os.path.join(output_dir, f"{basename}_translated.wav")
    torchaudio.save(intermediate_path, translated_audio.cpu(), sr)

    # Apply sox pitch shift
    final_path = os.path.join(output_dir, f"{basename}_shifted_{shift_semitones:+.0f}st.wav")
    sox_pitch_shift(intermediate_path, final_path, shift_semitones)

    # Also save original for comparison
    original_path = os.path.join(output_dir, f"{basename}_original.wav")
    torchaudio.save(original_path, waveform, sr)

    # And a naive sox-only shift for comparison
    naive_path = os.path.join(output_dir, f"{basename}_naive_sox_{shift_semitones:+.0f}st.wav")
    sox_pitch_shift(original_path, naive_path, shift_semitones)

    print(f"Processed: {basename}")
    print(f"  Original: {original_path}")
    print(f"  Translated (no shift): {intermediate_path}")
    print(f"  Final (translated + sox): {final_path}")
    print(f"  Naive (sox only): {naive_path}")

    return {
        'original': original_path,
        'translated': intermediate_path,
        'final': final_path,
        'naive': naive_path,
    }


def generate_listening_test(
    model,
    dcae,
    manifest_path: str,
    output_dir: str,
    num_samples: int = 10,
    shift_semitones: float = -12,
    device: str = 'cuda',
):
    """
    Generate listening test from manifest entries.

    Selects high register samples, applies register transfer + pitch shift.
    """
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
        f0_path = fix_path(e.get('conditioning_paths', {}).get('f0', ''))

        if not latent_path or not f0_path:
            continue
        if not os.path.exists(latent_path) or not os.path.exists(f0_path):
            continue

        # Check pitch
        try:
            f0 = np.load(f0_path)
            f0_valid = f0[f0 > 20]
            if len(f0_valid) < 10:
                continue
            median_midi = 12 * np.log2(np.median(f0_valid) / 440) + 69
        except:
            continue

        # Only use high register samples
        if median_midi > 65:  # Above F4
            entries.append({
                'latent_path': latent_path,
                'f0_path': f0_path,
                'median_midi': median_midi,
            })

    # Sample diverse entries
    step = max(1, len(entries) // num_samples)
    selected = [entries[i * step] for i in range(min(num_samples, len(entries)))]

    print(f"Generating {len(selected)} listening test samples...")

    for i, entry in enumerate(tqdm(selected)):
        # Load latent
        latent_data = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
        if isinstance(latent_data, dict):
            latent = latent_data.get('latents', latent_data.get('latent'))
        else:
            latent = latent_data
        if latent.dim() == 4:
            pass  # Already [B, C, H, T]
        else:
            latent = latent.unsqueeze(0)  # [C, H, T] -> [1, C, H, T]

        # Cap length for speed
        T = min(latent.shape[-1], 256)
        latent = latent[:, :, :, :T].to(device)

        source_midi = entry['median_midi']
        target_midi = source_midi + shift_semitones

        # 1. Original decode (no changes)
        with torch.no_grad():
            sr, wavs = dcae.decode(latent)
        original_audio = wavs[0]
        original_path = os.path.join(output_dir, f"{i:02d}_original_pitch{source_midi:.0f}.wav")
        torchaudio.save(original_path, original_audio.cpu(), sr)

        # 2. Naive sox pitch shift (formants preserved = wrong)
        naive_path = os.path.join(output_dir, f"{i:02d}_naive_sox_to{target_midi:.0f}.wav")
        sox_pitch_shift(original_path, naive_path, shift_semitones)

        # 3. Register translated (formants changed, pitch unchanged)
        with torch.no_grad():
            translated = model(latent)
            sr, wavs = dcae.decode(translated)
        translated_audio = wavs[0]
        translated_path = os.path.join(output_dir, f"{i:02d}_translated_formants.wav")
        torchaudio.save(translated_path, translated_audio.cpu(), sr)

        # 4. Translated + sox shift (formants changed + pitch shifted = correct)
        final_path = os.path.join(output_dir, f"{i:02d}_final_to{target_midi:.0f}.wav")
        sox_pitch_shift(translated_path, final_path, shift_semitones)

        print(f"  Sample {i}: {source_midi:.0f} → {target_midi:.0f}")

    print(f"\nListening test saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Register Translator Inference")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json',
                        help='Manifest for listening test')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--audio', type=str, default=None,
                        help='Single audio file to process')
    parser.add_argument('--shift', type=float, default=-12,
                        help='Semitones to shift (negative = down)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of listening test samples')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model type: {config['model_type']}")

    # Load DCAE
    print("Loading DCAE...")
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
        device_id=0
    )
    components.load_dcae()
    dcae = components.music_dcae

    if args.audio:
        # Process single file
        process_single(
            model=model,
            dcae=dcae,
            audio_path=args.audio,
            output_dir=args.output_dir,
            shift_semitones=args.shift,
            device=args.device,
        )
    else:
        # Generate listening test
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
