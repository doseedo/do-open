#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVC Voice Conversion Wrapper

Converts input audio to match a reference speaker's voice using RVC.

Usage:
    python rvc_voice_converter.py \
        --input vocals.wav \
        --speaker-ref reference_voice.wav \
        --model-path /path/to/rvc_model.pth \
        --output output.wav
"""

import argparse
import sys
import os
import time
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Add RVC-WebUI to path
sys.path.insert(0, '/home/arlo/Data/RVC-WebUI')


def load_rvc_model(model_path: str, device: str = 'cuda:0'):
    """
    Load RVC model from checkpoint.

    Args:
        model_path: Path to RVC model (.pth file)
        device: Device to load model on

    Returns:
        Loaded RVC model
    """
    print(f"[Loading RVC model from {model_path}...]")

    try:
        # Import RVC modules
        from infer.modules.vc.modules import VC
        from configs.config import Config

        # Initialize config
        config = Config()

        # Initialize VC module
        vc = VC(config)

        # Load model
        vc.get_vc(model_path)

        print(f"✅ Model loaded successfully")
        return vc

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


def convert_voice(
    input_audio_path: str,
    speaker_ref_path: str,
    model_path: str,
    output_path: str,
    pitch_shift: int = 0,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
    device: str = 'cuda:0'
):
    """
    Convert input audio to match reference speaker voice.

    Args:
        input_audio_path: Path to input audio to convert
        speaker_ref_path: Path to reference speaker audio (for training/model selection)
        model_path: Path to trained RVC model
        output_path: Path to save converted audio
        pitch_shift: Pitch shift in semitones (default: 0)
        index_rate: Feature retrieval strength (0-1, default: 0.5)
        filter_radius: Median filter radius for pitch smoothing (default: 3)
        resample_sr: Resample output to this sample rate (0=no resampling)
        rms_mix_rate: Volume envelope mixing (0-1, default: 0.25)
        protect: Protect voiceless consonants (0-0.5, default: 0.33)
        device: Torch device
    """
    start_time = time.time()

    print("=" * 80)
    print("RVC Voice Conversion")
    print("=" * 80)
    print(f"Input Audio:      {input_audio_path}")
    print(f"Speaker Ref:      {speaker_ref_path}")
    print(f"Model:            {model_path}")
    print(f"Output:           {output_path}")
    print(f"Pitch Shift:      {pitch_shift:+d} semitones")
    print(f"Index Rate:       {index_rate}")
    print(f"Filter Radius:    {filter_radius}")
    print(f"RMS Mix:          {rms_mix_rate}")
    print(f"Protect:          {protect}")
    print("=" * 80)

    # Validate inputs
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    if model_path != "none" and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Load RVC model
    print("\n[1/3] Loading RVC model...")
    load_start = time.time()
    vc = load_rvc_model(model_path, device=device)
    load_time = time.time() - load_start
    print(f"   ✓ Model loaded in {load_time:.2f}s")

    # Load input audio
    print("\n[2/3] Loading input audio...")
    audio_start = time.time()
    audio, sr = librosa.load(input_audio_path, sr=None, mono=True)
    audio_time = time.time() - audio_start
    print(f"   ✓ Audio loaded: {len(audio)/sr:.2f}s @ {sr}Hz in {audio_time:.2f}s")

    # Convert voice
    print("\n[3/3] Converting voice...")
    convert_start = time.time()

    try:
        # Perform voice conversion
        converted_audio = vc.vc_single(
            sid=0,  # Speaker ID (0 for single-speaker model)
            input_audio_path=input_audio_path,
            f0_up_key=pitch_shift,
            f0_file=None,
            f0_method="rmvpe",  # Pitch extraction method (rmvpe, crepe, harvest, dio)
            file_index="",  # Feature index path (optional)
            index_rate=index_rate,
            filter_radius=filter_radius,
            resample_sr=resample_sr,
            rms_mix_rate=rms_mix_rate,
            protect=protect
        )

        convert_time = time.time() - convert_start
        print(f"   ✓ Voice converted in {convert_time:.2f}s")

        # Save output
        print(f"\n[Saving output to {output_path}...]")
        sf.write(output_path, converted_audio[1][0], converted_audio[0])

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"✅ Voice conversion complete!")
        print(f"   Output: {output_path}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   - Model load: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
        print(f"   - Audio load: {audio_time:.2f}s ({audio_time/total_time*100:.1f}%)")
        print(f"   - Conversion: {convert_time:.2f}s ({convert_time/total_time*100:.1f}%)")
        print("=" * 80)

        return converted_audio

    except Exception as e:
        print(f"❌ Voice conversion failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='RVC Voice Conversion - Convert audio to match reference speaker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python rvc_voice_converter.py \
      --input vocals.wav \
      --speaker-ref my_voice.wav \
      --model /path/to/model.pth \
      --output converted.wav

  # With pitch shift (+2 semitones)
  python rvc_voice_converter.py \
      --input vocals.wav \
      --speaker-ref my_voice.wav \
      --model /path/to/model.pth \
      --pitch-shift 2 \
      --output converted.wav

  # High quality conversion
  python rvc_voice_converter.py \
      --input vocals.wav \
      --speaker-ref my_voice.wav \
      --model /path/to/model.pth \
      --index-rate 0.75 \
      --protect 0.5 \
      --output converted.wav
        """
    )

    # Required
    parser.add_argument('--input', type=str, required=True, help='Input audio file to convert')
    parser.add_argument('--speaker-ref', type=str, required=True,
                       help='Reference speaker audio (for model selection/training)')
    parser.add_argument('--model', type=str, required=True, help='Path to RVC model (.pth)')
    parser.add_argument('--output', type=str, required=True, help='Output audio file path')

    # Optional conversion parameters
    parser.add_argument('--pitch-shift', type=int, default=0,
                       help='Pitch shift in semitones (default: 0)')
    parser.add_argument('--index-rate', type=float, default=0.5,
                       help='Feature retrieval strength 0-1 (default: 0.5, higher=more similar to reference)')
    parser.add_argument('--filter-radius', type=int, default=3,
                       help='Median filter radius for pitch smoothing (default: 3)')
    parser.add_argument('--resample-sr', type=int, default=0,
                       help='Resample output to this sample rate (0=no resampling)')
    parser.add_argument('--rms-mix', type=float, default=0.25,
                       help='Volume envelope mixing 0-1 (default: 0.25)')
    parser.add_argument('--protect', type=float, default=0.33,
                       help='Protect voiceless consonants 0-0.5 (default: 0.33, higher=preserve more)')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')

    args = parser.parse_args()

    # Convert voice
    convert_voice(
        input_audio_path=args.input,
        speaker_ref_path=args.speaker_ref,
        model_path=args.model,
        output_path=args.output,
        pitch_shift=args.pitch_shift,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        resample_sr=args.resample_sr,
        rms_mix_rate=args.rms_mix,
        protect=args.protect,
        device=args.device
    )


if __name__ == '__main__':
    main()
