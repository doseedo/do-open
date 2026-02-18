#!/usr/bin/env python3
"""
Extract MIDI and lyrics from audio, then generate with 0.3 noise using dummy conditionings.

This script:
1. Extracts MIDI (piano roll) from audio using basic_pitch
2. Extracts lyrics using Whisper
3. Calls infer_vox_model.py with noise_level=0.3 and dummy conditionings

Usage:
    python extract_and_generate_midi_vocals.py \
        --audio /path/to/vocals.wav \
        --output output.wav
"""

import argparse
import sys
import os
import subprocess
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
import pretty_midi

# Add project path
sys.path.insert(0, '/home/arlo/Data')


def extract_midi_from_audio(audio_path: str, output_midi_path: str) -> str:
    """
    Extract MIDI from audio using basic_pitch ONNX model.

    Returns:
        Path to extracted MIDI file
    """
    print(f"\n{'='*60}")
    print("Extracting MIDI from audio using basic_pitch")
    print(f"{'='*60}")

    try:
        import basic_pitch
        from basic_pitch.inference import predict as basicpitch_predict

        # Use ONNX model directly (same as test_extract_local.py)
        onnx_model = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"

        if not onnx_model.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_model}")

        print(f"Using ONNX model: {onnx_model}")

        # Predict with smoothed parameters (same as working extraction)
        _, midi_pm, _ = basicpitch_predict(
            str(audio_path),
            model_or_model_path=str(onnx_model),
            onset_threshold=0.55,      # Reduces false onsets
            frame_threshold=0.35,      # Smooths weak pitch variations
            minimum_note_length=188    # Filters rapid fluctuations
        )

        # Write MIDI file
        os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
        midi_pm.write(str(output_midi_path))

        print(f"✅ MIDI extracted: {output_midi_path}")
        return output_midi_path

    except Exception as e:
        print(f"⚠️  basic_pitch extraction failed: {e}")
        print("Falling back to librosa pitch detection...")
        return extract_midi_simple(audio_path, output_midi_path)


def extract_midi_simple(audio_path: str, output_midi_path: str) -> str:
    """
    Fallback MIDI extraction using librosa pitch detection.
    """
    import librosa

    print("Using librosa for pitch detection...")

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Extract pitch using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )

    # Create MIDI
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Convert f0 to MIDI notes
    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # Group consecutive voiced frames into notes
    note_start = None
    note_pitch = None

    for i, (freq, is_voiced) in enumerate(zip(f0, voiced_flag)):
        if is_voiced and not np.isnan(freq):
            midi_pitch = int(np.round(librosa.hz_to_midi(freq)))

            if note_start is None:
                # Start new note
                note_start = times[i]
                note_pitch = midi_pitch
            elif abs(midi_pitch - note_pitch) > 1:
                # Pitch changed significantly, end previous note and start new one
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note_pitch,
                    start=note_start,
                    end=times[i]
                )
                instrument.notes.append(note)
                note_start = times[i]
                note_pitch = midi_pitch
        else:
            # Unvoiced, end current note if any
            if note_start is not None:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note_pitch,
                    start=note_start,
                    end=times[i]
                )
                instrument.notes.append(note)
                note_start = None
                note_pitch = None

    # Add final note if still active
    if note_start is not None:
        note = pretty_midi.Note(
            velocity=100,
            pitch=note_pitch,
            start=note_start,
            end=times[-1]
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_midi_path)

    print(f"✅ MIDI extracted (librosa): {output_midi_path}")
    print(f"   Notes: {len(instrument.notes)}")

    return output_midi_path


def extract_lyrics_from_audio(audio_path: str, output_txt_path: str) -> str:
    """
    Extract lyrics from audio using Whisper.

    Returns:
        Path to lyrics text file
    """
    print(f"\n{'='*60}")
    print("Extracting lyrics using Whisper")
    print(f"{'='*60}")

    try:
        import whisper

        model = whisper.load_model('base')
        result = model.transcribe(audio_path, language='en')

        lyrics = result['text'].strip()

        # Save to file
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, 'w') as f:
            f.write(lyrics)

        print(f"✅ Lyrics extracted: {output_txt_path}")
        print(f"   Text: {lyrics[:80]}..." if len(lyrics) > 80 else f"   Text: {lyrics}")

        return output_txt_path

    except Exception as e:
        print(f"⚠️  Whisper extraction failed: {e}")
        print("Creating empty lyrics file...")

        with open(output_txt_path, 'w') as f:
            f.write("")

        return output_txt_path


def generate_with_midi_and_lyrics(
    midi_path: str,
    lyrics_path: str,
    audio_path: str,
    output_path: str,
    noise_level: float = 0.3,
    group: str = "Vocals",
    subgroup: str = "Lead Vocals",
    steps: int = 60,
    seed: int = 0,
    cfg_weight: float = 2.0
):
    """
    Call infer_vox_model_midi_only.py with MIDI and lyrics, using dummy conditionings.
    """
    print(f"\n{'='*60}")
    print(f"Generating vocals with noise_level={noise_level}")
    print(f"{'='*60}")

    # Read lyrics
    with open(lyrics_path, 'r') as f:
        lyrics = f.read().strip()

    # Get audio duration for generation
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[-1] / sr

    print(f"Audio duration: {duration:.2f}s")
    print(f"MIDI file: {midi_path}")
    print(f"Lyrics: {lyrics[:60]}..." if len(lyrics) > 60 else f"Lyrics: {lyrics}")

    # Build command
    cmd = [
        "conda", "run", "-n", "ace_step",
        "python", "/home/arlo/Data/infer_vox_model_midi_only.py",
        "--midi", midi_path,
        "--lyrics", lyrics,
        "--group", group,
        "--subgroup", subgroup,
        "--duration", str(duration),
        "--steps", str(steps),
        "--seed", str(seed),
        "--cfg_weight", "1.0",  # Use 1.0 for 2x faster generation (skip CFG)
        "--output", output_path
    ]

    print(f"\nRunning inference...")
    print(f"Command: {' '.join(cmd)}")

    # Run inference (increased timeout for longer audio files)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 minutes

    if result.returncode != 0:
        print(f"❌ Generation failed!")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Generation failed with exit code {result.returncode}")

    print(result.stdout)
    print(f"\n✅ Generation complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract MIDI and lyrics from audio, generate with 0.3 noise and dummy conditionings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python extract_and_generate_midi_vocals.py \\
      --audio /home/arlo/Data/test/vocals.wav \\
      --output /home/arlo/Data/test/output_extracted_midi_vocals.wav
        """
    )

    # Required
    parser.add_argument('--audio', type=str, required=True, help='Input audio file (WAV)')
    parser.add_argument('--output', type=str, required=True, help='Output WAV file path')

    # Optional
    parser.add_argument('--noise-level', type=float, default=0.3,
                        help='Noise level (0.0-1.0, default: 0.3 for 70%% GT + 30%% noise)')
    parser.add_argument('--group', type=str, default='vocal', help='Instrument group')
    parser.add_argument('--subgroup', type=str, default='lead_vocal', help='Instrument subgroup')
    parser.add_argument('--steps', type=int, default=60, help='Diffusion steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cfg-weight', type=float, default=2.0, help='CFG weight')

    # Intermediate file paths (optional)
    parser.add_argument('--midi-output', type=str, default=None,
                        help='Where to save extracted MIDI (default: auto)')
    parser.add_argument('--lyrics-output', type=str, default=None,
                        help='Where to save extracted lyrics (default: auto)')
    parser.add_argument('--keep-intermediates', action='store_true',
                        help='Keep intermediate MIDI and lyrics files')

    args = parser.parse_args()

    # Setup intermediate paths
    audio_stem = Path(args.audio).stem
    work_dir = Path(args.output).parent

    midi_path = args.midi_output or str(work_dir / f"{audio_stem}_extracted.mid")
    lyrics_path = args.lyrics_output or str(work_dir / f"{audio_stem}_extracted_lyrics.txt")

    print("="*60)
    print("MIDI + Lyrics Extraction and Generation Pipeline")
    print("="*60)
    print(f"Input audio: {args.audio}")
    print(f"Output: {args.output}")
    print(f"Noise level: {args.noise_level} ({(1-args.noise_level)*100:.0f}% GT + {args.noise_level*100:.0f}% noise)")
    print("="*60)

    try:
        # Step 1: Extract MIDI
        extract_midi_from_audio(args.audio, midi_path)

        # Step 2: Extract lyrics
        extract_lyrics_from_audio(args.audio, lyrics_path)

        # Step 3: Generate with model
        generate_with_midi_and_lyrics(
            midi_path=midi_path,
            lyrics_path=lyrics_path,
            audio_path=args.audio,
            output_path=args.output,
            noise_level=args.noise_level,
            group=args.group,
            subgroup=args.subgroup,
            steps=args.steps,
            seed=args.seed,
            cfg_weight=args.cfg_weight
        )

        # Cleanup intermediate files if requested
        if not args.keep_intermediates:
            print(f"\nCleaning up intermediate files...")
            if os.path.exists(midi_path):
                os.remove(midi_path)
                print(f"  Removed: {midi_path}")
            if os.path.exists(lyrics_path):
                os.remove(lyrics_path)
                print(f"  Removed: {lyrics_path}")
        else:
            print(f"\nIntermediate files saved:")
            print(f"  MIDI: {midi_path}")
            print(f"  Lyrics: {lyrics_path}")

        print("\n" + "="*60)
        print("✅ Pipeline complete!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
