#!/usr/bin/env python3
"""
eSpeak MIDI Vocoder - Better syllable-to-MIDI alignment

Approach:
1. Generate entire lyrics with espeak at flat pitch (monotone)
2. Align syllables to MIDI notes based on timing
3. Apply pitch modulation using rubberband or librosa to match MIDI pitches
4. This creates syllable sounds that follow the melody

Usage:
    python espeak_midi_vocoder.py \\
        --midi input.mid \\
        --lyrics "do re mi fa sol" \\
        --output output.wav
"""

import sys
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pretty_midi
import librosa
import soundfile as sf
from typing import List, Tuple


def generate_flat_speech(lyrics: str, duration: float, output_path: str, sample_rate: int = 22050):
    """
    Generate speech at flat pitch using espeak.

    Args:
        lyrics: Text to synthesize
        duration: Target duration in seconds
        output_path: Output WAV path
        sample_rate: Sample rate
    """
    # Calculate speech rate to match duration
    # espeak default is ~175 wpm, adjust based on word count and duration
    words = lyrics.split()
    words_per_second = len(words) / duration if duration > 0 else 3
    wpm = int(words_per_second * 60)
    wpm = max(80, min(450, wpm))  # Clamp to espeak range

    # Generate with espeak at fixed pitch
    cmd = [
        'espeak-ng',
        '-v', 'en-us',
        '-s', str(wpm),
        '-p', '50',  # Fixed pitch (0-99, 50 is middle)
        '-w', output_path,
        lyrics
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    # Load and resample if needed
    audio, sr = librosa.load(output_path, sr=sample_rate)

    # Time-stretch to match exact duration
    current_duration = len(audio) / sample_rate
    if current_duration > 0:
        stretch_rate = current_duration / duration
        audio = librosa.effects.time_stretch(audio, rate=stretch_rate)

    # Save stretched version
    sf.write(output_path, audio, sample_rate)

    return output_path


def apply_midi_pitch_modulation(
    audio: np.ndarray,
    midi_path: str,
    sample_rate: int = 22050,
    base_pitch: int = 60
) -> np.ndarray:
    """
    Apply pitch modulation to audio based on MIDI notes.

    Uses librosa to pitch-shift different regions to match MIDI melody.

    Args:
        audio: Input audio array
        midi_path: MIDI file with melody
        sample_rate: Audio sample rate
        base_pitch: Base MIDI pitch of the flat speech (default C4 = 60)

    Returns:
        Pitch-modulated audio
    """
    # Load MIDI
    midi = pretty_midi.PrettyMIDI(midi_path)

    if len(midi.instruments) == 0 or len(midi.instruments[0].notes) == 0:
        return audio

    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)
    total_duration = midi.get_end_time()

    # Create output buffer
    output = np.zeros_like(audio)

    # Process each MIDI note region
    for note in notes:
        # Calculate pitch shift in semitones
        pitch_shift = note.pitch - base_pitch

        # Get audio segment for this note's time range
        start_sample = int(note.start * sample_rate)
        end_sample = int(note.end * sample_rate)

        # Clamp to audio bounds
        start_sample = max(0, min(start_sample, len(audio)))
        end_sample = max(0, min(end_sample, len(audio)))

        if start_sample >= end_sample:
            continue

        # Extract segment
        segment = audio[start_sample:end_sample]

        # Pitch shift this segment
        if pitch_shift != 0:
            segment = librosa.effects.pitch_shift(
                segment,
                sr=sample_rate,
                n_steps=pitch_shift
            )

        # Place in output (additive mixing in case of overlaps)
        output[start_sample:start_sample + len(segment)] += segment

    return output


def synthesize_midi_vocoded_speech(
    midi_path: str,
    lyrics: str,
    output_path: str,
    sample_rate: int = 22050,
    base_pitch: int = 60,
    verbose: bool = True
) -> str:
    """
    Main synthesis function: espeak + MIDI pitch modulation.

    Args:
        midi_path: Path to MIDI file
        lyrics: Lyrics text
        output_path: Output WAV path
        sample_rate: Sample rate
        base_pitch: Base MIDI pitch for espeak
        verbose: Print progress

    Returns:
        Path to output file
    """
    if verbose:
        print(f"\n{'='*70}")
        print("eSpeak MIDI Vocoder")
        print(f"{'='*70}")
        print(f"MIDI: {midi_path}")
        print(f"Lyrics: {lyrics}")
        print(f"Output: {output_path}")

    # Get MIDI duration
    midi = pretty_midi.PrettyMIDI(midi_path)
    duration = midi.get_end_time()

    if verbose:
        print(f"\nMIDI duration: {duration:.2f}s")
        print(f"Notes: {len(midi.instruments[0].notes) if midi.instruments else 0}")

    # Step 1: Generate flat speech
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    if verbose:
        print(f"\n[1/3] Generating flat speech with espeak...")

    generate_flat_speech(lyrics, duration, tmp_path, sample_rate)

    # Step 2: Load flat speech
    if verbose:
        print(f"[2/3] Loading audio...")

    audio, sr = librosa.load(tmp_path, sr=sample_rate)

    # Step 3: Apply MIDI pitch modulation
    if verbose:
        print(f"[3/3] Applying MIDI pitch modulation...")

    modulated_audio = apply_midi_pitch_modulation(
        audio,
        midi_path,
        sample_rate=sample_rate,
        base_pitch=base_pitch
    )

    # Normalize
    max_val = np.max(np.abs(modulated_audio))
    if max_val > 0:
        modulated_audio = modulated_audio / (max_val * 1.1)

    # Save
    sf.write(output_path, modulated_audio, sample_rate)

    # Cleanup
    Path(tmp_path).unlink()

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Synthesis complete: {output_path}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"{'='*70}\n")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='eSpeak MIDI Vocoder - Syllable-aware singing synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python espeak_midi_vocoder.py \\
      --midi melody.mid \\
      --lyrics "do re mi fa sol" \\
      --output output.wav

  # With custom base pitch
  python espeak_midi_vocoder.py \\
      --midi melody.mid \\
      --lyrics "hello world" \\
      --output output.wav \\
      --base-pitch 65
        """
    )

    parser.add_argument('--midi', required=True, help='Input MIDI file')
    parser.add_argument('--lyrics', required=True, help='Lyrics text')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--base-pitch', type=int, default=60,
                       help='Base MIDI pitch for espeak (default: 60 = C4)')
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help='Sample rate (default: 22050)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')

    args = parser.parse_args()

    synthesize_midi_vocoded_speech(
        midi_path=args.midi,
        lyrics=args.lyrics,
        output_path=args.output,
        sample_rate=args.sample_rate,
        base_pitch=args.base_pitch,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
