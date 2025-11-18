#!/usr/bin/env python3
"""
eSpeak + WORLD Vocoder - High-quality pitch manipulation

WORLD vocoder is the industry standard for pitch/formant manipulation.
It decomposes speech into:
- F0 (fundamental frequency/pitch)
- Spectral envelope (formants/timbre)
- Aperiodicity (breathiness)

This allows clean pitch modification while preserving voice quality.

Usage:
    python espeak_world_vocoder.py \\
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


def generate_espeak_speech(lyrics: str, output_path: str, duration: float, sample_rate: int = 16000):
    """
    Generate speech with espeak.

    Note: WORLD works best with 16kHz audio.

    Args:
        lyrics: Text to synthesize
        output_path: Output file path
        duration: Target duration in seconds
        sample_rate: Sample rate (WORLD recommends 16kHz)
    """
    # Calculate WPM to match target duration roughly
    words = lyrics.split()
    words_per_second = len(words) / duration if duration > 0 else 3
    wpm = int(words_per_second * 60)
    wpm = max(80, min(450, wpm))

    # Generate with espeak
    cmd = [
        'espeak-ng',
        '-v', 'en-us',
        '-s', str(wpm),
        '-p', '50',  # Middle pitch
        '--stdout'
    ]
    cmd.append(lyrics)

    result = subprocess.run(cmd, capture_output=True, check=True)

    # Write raw audio
    with open(output_path, 'wb') as f:
        f.write(result.stdout)

    # Load and resample
    import librosa
    audio, sr = librosa.load(output_path, sr=sample_rate)

    # Time-stretch to exact duration
    current_duration = len(audio) / sample_rate
    if current_duration > 0 and duration > 0:
        stretch_rate = current_duration / duration
        audio = librosa.effects.time_stretch(audio, rate=stretch_rate)

        # Pad or trim to exact length
        target_samples = int(duration * sample_rate)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

    # Save
    sf.write(output_path, audio, sample_rate)

    return audio


def create_midi_f0_contour(midi_path: str, duration: float, sample_rate: int, frame_period: float = 5.0):
    """
    Create F0 (frequency) contour from MIDI notes.

    Args:
        midi_path: MIDI file path
        duration: Duration in seconds
        sample_rate: Audio sample rate
        frame_period: Frame period in milliseconds (default 5.0ms for WORLD)

    Returns:
        f0: Fundamental frequency contour in Hz
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    if not midi.instruments or not midi.instruments[0].notes:
        num_frames = int(duration * 1000.0 / frame_period) + 1
        return np.zeros(num_frames)

    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)

    # Calculate number of frames
    num_frames = int(duration * 1000.0 / frame_period) + 1

    # Create time array for frames
    times = np.arange(num_frames) * frame_period / 1000.0

    # Create F0 contour
    f0 = np.zeros(num_frames)

    for i, t in enumerate(times):
        # Find active note at this time
        for note in notes:
            if note.start <= t < note.end:
                # Convert MIDI pitch to frequency
                f0[i] = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
                break

    return f0


def vocode_with_world(
    audio: np.ndarray,
    target_f0: np.ndarray,
    sample_rate: int = 16000,
    frame_period: float = 5.0
) -> np.ndarray:
    """
    Apply pitch modification using WORLD vocoder.

    Args:
        audio: Input audio (mono)
        target_f0: Target F0 contour in Hz
        sample_rate: Sample rate
        frame_period: Frame period in milliseconds

    Returns:
        Modified audio with new pitch contour
    """
    # Convert to float64 for WORLD
    audio = audio.astype(np.float64)

    # 1. Decompose speech using WORLD
    f0, time_axis = pw.harvest(audio, sample_rate, frame_period=frame_period)
    sp = pw.cheaptrick(audio, f0, time_axis, sample_rate)
    ap = pw.d4c(audio, f0, time_axis, sample_rate)

    # 2. Modify F0 with target contour
    # Where target F0 is 0 (unvoiced), keep original
    # Where target F0 > 0, use target
    modified_f0 = f0.copy()

    # Ensure same length
    min_len = min(len(modified_f0), len(target_f0))

    for i in range(min_len):
        if target_f0[i] > 0:
            modified_f0[i] = target_f0[i]

    # 3. Synthesize with modified F0
    synthesized = pw.synthesize(modified_f0, sp, ap, sample_rate, frame_period=frame_period)

    return synthesized.astype(np.float32)


def synthesize_world_vocoded_speech(
    midi_path: str,
    lyrics: str,
    output_path: str,
    sample_rate: int = 16000,
    frame_period: float = 5.0,
    verbose: bool = True
) -> str:
    """
    Main synthesis function using WORLD vocoder.

    Args:
        midi_path: MIDI file path
        lyrics: Lyrics text
        output_path: Output WAV path
        sample_rate: Sample rate (16kHz recommended for WORLD)
        frame_period: WORLD frame period in ms
        verbose: Print progress

    Returns:
        Output file path
    """
    if verbose:
        print(f"\n{'='*70}")
        print("eSpeak + WORLD Vocoder")
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
        for i, note in enumerate(midi.instruments[0].notes[:10]):  # Show first 10
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            print(f"  Note {i+1}: MIDI {note.pitch} = {freq:.1f} Hz, {note.start:.2f}-{note.end:.2f}s")

    # Step 1: Generate speech with espeak
    if verbose:
        print(f"\n[1/3] Generating speech with espeak...")

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    audio = generate_espeak_speech(lyrics, tmp_path, duration, sample_rate)

    if verbose:
        print(f"   Generated {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")

    # Step 2: Create F0 contour from MIDI
    if verbose:
        print(f"[2/3] Creating F0 contour from MIDI...")

    target_f0 = create_midi_f0_contour(midi_path, duration, sample_rate, frame_period)

    if verbose:
        voiced_frames = np.sum(target_f0 > 0)
        print(f"   F0 frames: {len(target_f0)} ({voiced_frames} voiced)")
        unique_f0 = np.unique(target_f0[target_f0 > 0])
        if len(unique_f0) > 0:
            print(f"   F0 range: {unique_f0.min():.1f} - {unique_f0.max():.1f} Hz")

    # Step 3: Apply WORLD vocoding
    if verbose:
        print(f"[3/3] Applying WORLD vocoder pitch modification...")

    vocoded_audio = vocode_with_world(audio, target_f0, sample_rate, frame_period)

    # Normalize
    max_val = np.max(np.abs(vocoded_audio))
    if max_val > 0:
        vocoded_audio = vocoded_audio / (max_val * 1.1)

    # Save (upsample to 44.1kHz if needed for compatibility)
    if sample_rate != 44100:
        import librosa
        vocoded_audio = librosa.resample(vocoded_audio, orig_sr=sample_rate, target_sr=44100)
        save_sr = 44100
    else:
        save_sr = sample_rate

    sf.write(output_path, vocoded_audio, save_sr)

    # Cleanup
    Path(tmp_path).unlink()

    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Complete: {output_path}")
        print(f"   Duration: {len(vocoded_audio)/save_sr:.2f}s")
        print(f"   Sample rate: {save_sr} Hz")
        print(f"{'='*70}\n")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='eSpeak + WORLD Vocoder - High-quality pitch manipulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python espeak_world_vocoder.py \\
      --midi melody.mid \\
      --lyrics "do re mi fa sol" \\
      --output output.wav

  # With custom frame period (lower = more accurate, slower)
  python espeak_world_vocoder.py \\
      --midi melody.mid \\
      --lyrics "hello world" \\
      --output output.wav \\
      --frame-period 2.5
        """
    )

    parser.add_argument('--midi', required=True, help='Input MIDI file')
    parser.add_argument('--lyrics', required=True, help='Lyrics text')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate (default: 16000, recommended for WORLD)')
    parser.add_argument('--frame-period', type=float, default=5.0,
                       help='Frame period in ms (default: 5.0)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    synthesize_world_vocoded_speech(
        midi_path=args.midi,
        lyrics=args.lyrics,
        output_path=args.output,
        sample_rate=args.sample_rate,
        frame_period=args.frame_period,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
