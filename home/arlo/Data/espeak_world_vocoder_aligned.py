#!/usr/bin/env python3
"""
eSpeak + WORLD Vocoder - Syllable-aligned version

Properly aligns each syllable/word to its corresponding MIDI note.

Usage:
    python espeak_world_vocoder_aligned.py \\
        --midi input.mid \\
        --lyrics "do re mi fa sol" \\
        --output output.wav
"""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pretty_midi
import pyworld as pw
import soundfile as sf
import librosa


def generate_syllable_audio(syllable: str, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate audio for a single syllable with espeak.

    Args:
        syllable: Single syllable/word
        duration: Target duration in seconds
        sample_rate: Sample rate

    Returns:
        Audio array
    """
    # Generate with espeak
    cmd = [
        'espeak-ng',
        '-v', 'en-us',
        '-s', '150',  # Normal speed
        '-p', '50',   # Middle pitch
        '--stdout',
        syllable
    ]

    result = subprocess.run(cmd, capture_output=True, check=True)

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(result.stdout)
        tmp_path = tmp.name

    # Load audio
    audio, sr = librosa.load(tmp_path, sr=sample_rate)

    # Time-stretch to exact duration using resampling (more stable than librosa.effects.time_stretch)
    current_duration = len(audio) / sample_rate
    if current_duration > 0 and duration > 0:
        # Calculate target length
        target_samples = int(duration * sample_rate)
        # Use librosa's resample for time-stretching (more stable)
        if len(audio) != target_samples:
            # Resample to achieve the desired duration
            audio = librosa.resample(audio, orig_sr=len(audio)/current_duration, target_sr=len(audio)/duration)

    # Pad or trim to exact length
    target_samples = int(duration * sample_rate)
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    # Cleanup
    Path(tmp_path).unlink()

    return audio


def align_syllables_to_midi(lyrics: str, midi_path: str):
    """
    Align syllables/words to MIDI notes.

    Returns list of (syllable, note) tuples.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    if not midi.instruments or not midi.instruments[0].notes:
        raise ValueError("MIDI has no notes")

    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)
    syllables = lyrics.split()

    # Simple 1:1 alignment (one syllable per note)
    # If more notes than syllables, repeat last syllable
    # If more syllables than notes, combine extra syllables

    alignments = []

    for i, note in enumerate(notes):
        if i < len(syllables):
            syl = syllables[i]
        elif syllables:
            syl = syllables[-1]  # Repeat last
        else:
            syl = ""

        alignments.append((syl, note))

    return alignments


def vocode_audio_segment(audio: np.ndarray, target_freq: float, sample_rate: int = 16000,
                          frame_period: float = 5.0) -> np.ndarray:
    """
    Apply WORLD vocoding to set pitch to target frequency.

    Args:
        audio: Input audio segment
        target_freq: Target F0 in Hz
        sample_rate: Sample rate
        frame_period: Frame period in ms

    Returns:
        Vocoded audio
    """
    audio = audio.astype(np.float64)

    # Decompose with WORLD
    f0, time_axis = pw.harvest(audio, sample_rate, frame_period=frame_period)
    sp = pw.cheaptrick(audio, f0, time_axis, sample_rate)
    ap = pw.d4c(audio, f0, time_axis, sample_rate)

    # Set all voiced frames to target frequency
    modified_f0 = np.full_like(f0, target_freq)

    # Synthesize
    synthesized = pw.synthesize(modified_f0, sp, ap, sample_rate, frame_period=frame_period)

    return synthesized.astype(np.float32)


def synthesize_syllable_aligned(
    midi_path: str,
    lyrics: str,
    output_path: str,
    sample_rate: int = 16000,
    frame_period: float = 5.0,
    verbose: bool = True
) -> str:
    """
    Synthesize with proper syllable-to-note alignment.

    Args:
        midi_path: MIDI file path
        lyrics: Lyrics (space-separated syllables/words)
        output_path: Output WAV path
        sample_rate: Sample rate
        frame_period: WORLD frame period in ms
        verbose: Print progress

    Returns:
        Output file path
    """
    if verbose:
        print(f"\n{'='*70}")
        print("eSpeak + WORLD Vocoder (Syllable-Aligned)")
        print(f"{'='*70}")
        print(f"MIDI: {midi_path}")
        print(f"Lyrics: {lyrics}")

    # Get MIDI info
    midi = pretty_midi.PrettyMIDI(midi_path)
    total_duration = midi.get_end_time()

    if verbose:
        print(f"\nTotal duration: {total_duration:.2f}s")

    # Align syllables to notes
    alignments = align_syllables_to_midi(lyrics, midi_path)

    if verbose:
        print(f"\nAlignments:")
        for i, (syl, note) in enumerate(alignments):
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            print(f"  {i+1}. '{syl}' → MIDI {note.pitch} ({freq:.1f} Hz) @ {note.start:.2f}-{note.end:.2f}s")

    # Create output buffer
    total_samples = int(total_duration * sample_rate)
    output_audio = np.zeros(total_samples, dtype=np.float32)

    # Process each aligned syllable
    if verbose:
        print(f"\nSynthesizing syllables:")

    for i, (syllable, note) in enumerate(alignments):
        if not syllable.strip():
            continue

        # Calculate note duration
        note_duration = note.end - note.start

        # Generate syllable audio
        if verbose:
            print(f"  [{i+1}/{len(alignments)}] Generating '{syllable}' ({note_duration:.2f}s)...")

        syl_audio = generate_syllable_audio(syllable, note_duration, sample_rate)

        # Apply vocoding to set pitch
        target_freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))

        if verbose:
            print(f"       Vocoding to {target_freq:.1f} Hz...")

        vocoded = vocode_audio_segment(syl_audio, target_freq, sample_rate, frame_period)

        # Place in output buffer
        start_sample = int(note.start * sample_rate)
        end_sample = start_sample + len(vocoded)

        # Ensure we don't exceed buffer
        if end_sample > len(output_audio):
            vocoded = vocoded[:len(output_audio) - start_sample]
            end_sample = len(output_audio)

        # Mix into output
        output_audio[start_sample:end_sample] += vocoded

        if verbose:
            print(f"       Placed at {note.start:.2f}s ✓")

    # Normalize
    max_val = np.max(np.abs(output_audio))
    if max_val > 0:
        output_audio = output_audio / (max_val * 1.1)

    # Upsample to 44.1kHz for compatibility
    if sample_rate != 44100:
        output_audio = librosa.resample(output_audio, orig_sr=sample_rate, target_sr=44100)
        save_sr = 44100
    else:
        save_sr = sample_rate

    # Save
    sf.write(output_path, output_audio, save_sr)

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Complete: {output_path}")
        print(f"   Duration: {len(output_audio)/save_sr:.2f}s")
        print(f"   Sample rate: {save_sr} Hz")
        print(f"{'='*70}\n")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='eSpeak + WORLD Vocoder - Syllable-aligned synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One syllable per note (space-separated)
  python espeak_world_vocoder_aligned.py \\
      --midi melody.mid \\
      --lyrics "do re mi fa sol" \\
      --output output.wav

  # Using JSON mapping file
  python espeak_world_vocoder_aligned.py \\
      --midi melody.mid \\
      --lyrics-map mapping.json \\
      --output output.wav
        """
    )

    parser.add_argument('--midi', required=True, help='Input MIDI file')
    parser.add_argument('--lyrics', help='Lyrics (space-separated syllables)')
    parser.add_argument('--lyrics-map', help='JSON file with note index -> syllable mapping ({"0": "do", "1": "re", ...})')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate (default: 16000)')
    parser.add_argument('--frame-period', type=float, default=5.0,
                       help='Frame period in ms (default: 5.0)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Determine lyrics source
    if args.lyrics_map:
        # Load from JSON mapping file
        import json
        with open(args.lyrics_map, 'r') as f:
            note_syllable_map = json.load(f)

        # Convert to space-separated string in order
        # Sort by note index
        sorted_indices = sorted([int(k) for k in note_syllable_map.keys()])
        lyrics_parts = [note_syllable_map[str(i)] for i in sorted_indices if str(i) in note_syllable_map]
        lyrics = ' '.join(lyrics_parts)

        if not args.quiet:
            print(f"Loaded lyrics from mapping: {lyrics}")

    elif args.lyrics:
        lyrics = args.lyrics
    else:
        raise ValueError("Must provide either --lyrics or --lyrics-map")

    synthesize_syllable_aligned(
        midi_path=args.midi,
        lyrics=lyrics,
        output_path=args.output,
        sample_rate=args.sample_rate,
        frame_period=args.frame_period,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
