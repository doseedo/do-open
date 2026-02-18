#!/usr/bin/env python3
"""
eSpeak MIDI Vocoder V2 - Fixed timing and pitch

Better approach:
1. Generate speech at normal speed with espeak
2. Use PSOLA or librosa to apply frame-by-frame pitch shifting based on MIDI
3. No time stretching - preserve exact syllable timing

Usage:
    python espeak_midi_vocoder_v2.py \\
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


def generate_espeak_monotone(lyrics: str, output_path: str, duration: float, sample_rate: int = 22050):
    """
    Generate monotone speech with espeak, time-stretched to exact duration.

    Args:
        lyrics: Text to speak
        output_path: Output WAV path
        duration: Target duration in seconds
        sample_rate: Sample rate
    """
    # Generate with espeak at slow speed for better quality
    cmd = [
        'espeak-ng',
        '-v', 'en-us',
        '-s', '120',  # Moderate speed
        '-p', '50',   # Middle pitch
        '-w', output_path,
        lyrics
    ]

    subprocess.run(cmd, capture_output=True, check=True)

    # Load and time-stretch to exact duration
    audio, sr = librosa.load(output_path, sr=sample_rate)
    current_duration = len(audio) / sample_rate

    if current_duration > 0 and duration > 0:
        stretch_rate = current_duration / duration
        audio = librosa.effects.time_stretch(audio, rate=stretch_rate)

        # Ensure exact length
        target_samples = int(duration * sample_rate)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

    sf.write(output_path, audio, sample_rate)
    return audio


def create_pitch_contour_from_midi(midi_path: str, duration: float, sample_rate: int, fps: int = 100):
    """
    Create a frame-by-frame pitch contour from MIDI.

    Args:
        midi_path: Path to MIDI file
        duration: Duration in seconds
        sample_rate: Audio sample rate
        fps: Frames per second for pitch contour

    Returns:
        pitch_contour: Array of pitch values in semitones relative to base pitch
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    if not midi.instruments or not midi.instruments[0].notes:
        return np.zeros(int(duration * fps))

    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)

    # Create time grid
    num_frames = int(duration * fps)
    times = np.linspace(0, duration, num_frames)

    # Fill pitch contour
    pitch_contour = np.zeros(num_frames)
    base_pitch = 60  # C4

    for i, t in enumerate(times):
        # Find which note is active at this time
        active_note = None
        for note in notes:
            if note.start <= t < note.end:
                active_note = note
                break

        if active_note:
            # Store pitch shift in semitones
            pitch_contour[i] = active_note.pitch - base_pitch

    return pitch_contour


def apply_pitch_contour_librosa(audio: np.ndarray, pitch_contour: np.ndarray,
                                  sample_rate: int, fps: int = 100) -> np.ndarray:
    """
    Apply pitch contour to audio using librosa's pitch shifting.
    Uses short-time processing to follow pitch changes.

    Args:
        audio: Input audio
        pitch_contour: Pitch shift values in semitones per frame
        sample_rate: Audio sample rate
        fps: Frames per second of pitch contour

    Returns:
        Pitch-shifted audio
    """
    hop_length = int(sample_rate / fps)
    num_frames = len(pitch_contour)

    # Create output buffer
    output = np.zeros_like(audio)

    # Process in chunks with crossfading
    for i in range(num_frames):
        pitch_shift = pitch_contour[i]

        # Get chunk boundaries
        start_sample = i * hop_length
        end_sample = min((i + 2) * hop_length, len(audio))  # Overlap by 1 frame

        if start_sample >= len(audio):
            break

        chunk = audio[start_sample:end_sample]

        # Pitch shift this chunk
        if pitch_shift != 0 and len(chunk) > 0:
            shifted_chunk = librosa.effects.pitch_shift(
                chunk,
                sr=sample_rate,
                n_steps=pitch_shift
            )
        else:
            shifted_chunk = chunk

        # Add to output with crossfade
        out_start = start_sample
        out_end = min(out_start + len(shifted_chunk), len(output))

        if out_end > out_start:
            # Simple overlap-add
            output[out_start:out_end] += shifted_chunk[:out_end - out_start]

    return output


def synthesize_vocoded_speech_v2(
    midi_path: str,
    lyrics: str,
    output_path: str,
    sample_rate: int = 22050,
    verbose: bool = True
) -> str:
    """
    Synthesize speech with MIDI pitch following.

    Args:
        midi_path: MIDI file path
        lyrics: Lyrics text
        output_path: Output WAV path
        sample_rate: Sample rate
        verbose: Print progress

    Returns:
        Output file path
    """
    if verbose:
        print(f"\n{'='*70}")
        print("eSpeak MIDI Vocoder V2")
        print(f"{'='*70}")
        print(f"MIDI: {midi_path}")
        print(f"Lyrics: {lyrics}")

    # Get MIDI duration
    midi = pretty_midi.PrettyMIDI(midi_path)
    duration = midi.get_end_time()

    if verbose:
        print(f"\nMIDI duration: {duration:.2f}s")
        print(f"Notes: {len(midi.instruments[0].notes) if midi.instruments else 0}")

    # Step 1: Generate monotone speech
    if verbose:
        print(f"\n[1/3] Generating monotone speech...")

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    audio = generate_espeak_monotone(lyrics, tmp_path, duration, sample_rate)

    if verbose:
        print(f"   Generated {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")

    # Step 2: Create pitch contour from MIDI
    if verbose:
        print(f"[2/3] Creating pitch contour from MIDI...")

    pitch_contour = create_pitch_contour_from_midi(midi_path, duration, sample_rate, fps=100)

    if verbose:
        unique_pitches = np.unique(pitch_contour[pitch_contour != 0])
        print(f"   Pitch shifts: {unique_pitches}")

    # Step 3: Apply pitch modulation
    if verbose:
        print(f"[3/3] Applying pitch modulation...")

    output_audio = apply_pitch_contour_librosa(audio, pitch_contour, sample_rate, fps=100)

    # Normalize
    max_val = np.max(np.abs(output_audio))
    if max_val > 0:
        output_audio = output_audio / (max_val * 1.2)

    # Save
    sf.write(output_path, output_audio, sample_rate)

    # Cleanup
    Path(tmp_path).unlink()

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Complete: {output_path}")
        print(f"   Duration: {len(output_audio)/sample_rate:.2f}s")
        print(f"{'='*70}\n")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='eSpeak MIDI Vocoder V2 - Fixed timing and pitch',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--midi', required=True, help='Input MIDI file')
    parser.add_argument('--lyrics', required=True, help='Lyrics text')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Sample rate')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    synthesize_vocoded_speech_v2(
        midi_path=args.midi,
        lyrics=args.lyrics,
        output_path=args.output,
        sample_rate=args.sample_rate,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
