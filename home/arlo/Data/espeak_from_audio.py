#!/usr/bin/env python3
"""
eSpeak Singing Synthesis from Audio

Extracts MIDI and lyrics from audio, then generates singing voice with eSpeak.

Usage:
    python espeak_from_audio.py \\
        --audio /path/to/vocals.wav \\
        --output singing.wav
"""

import argparse
import subprocess
import tempfile
import os
from pathlib import Path


def extract_midi_and_lyrics(audio_path: str, tmpdir: str) -> tuple:
    """
    Extract MIDI and lyrics from audio file using extract script.

    Returns:
        (midi_path, lyrics_text) tuple
    """
    print(f"\n{'='*70}")
    print("Extracting MIDI and Lyrics from Audio")
    print(f"{'='*70}")
    print(f"Input audio: {audio_path}")

    # Output paths
    midi_path = os.path.join(tmpdir, 'extracted.mid')
    lyrics_path = os.path.join(tmpdir, 'extracted_lyrics.txt')

    # Use the existing extraction script that works in ace_step environment
    print("\nCalling extraction functions from extract_and_generate_midi_vocals.py...")

    import sys
    sys.path.insert(0, '/home/arlo/Data')

    # Import extraction functions
    from extract_and_generate_midi_vocals import extract_midi_from_audio, extract_lyrics_from_audio

    # Extract MIDI
    extract_midi_from_audio(audio_path, midi_path)

    # Extract lyrics
    extract_lyrics_from_audio(audio_path, lyrics_path)

    # Read lyrics text
    with open(lyrics_path, 'r') as f:
        lyrics = f.read().strip()

    return midi_path, lyrics


def synthesize_with_espeak(midi_path: str, lyrics: str, output_path: str, **kwargs):
    """
    Call espeak_singing_synth.py to generate singing voice.
    """
    print(f"\n{'='*70}")
    print("Synthesizing Singing Voice with eSpeak")
    print(f"{'='*70}")

    cmd = [
        'python', '/home/arlo/Data/espeak_singing_synth.py',
        '--midi', midi_path,
        '--lyrics', lyrics,
        '--output', output_path
    ]

    # Add optional parameters
    if 'voice' in kwargs:
        cmd.extend(['--voice', kwargs['voice']])
    if 'rate' in kwargs:
        cmd.extend(['--rate', str(kwargs['rate'])])
    if 'base_pitch' in kwargs:
        cmd.extend(['--base-pitch', str(kwargs['base_pitch'])])

    print(f"\nRunning eSpeak singing synthesis...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ eSpeak synthesis failed!")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"eSpeak synthesis failed: {result.stderr}")

    print(result.stdout)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate eSpeak singing voice from audio file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python espeak_from_audio.py \\
      --audio /home/arlo/Data/test/vocals.wav \\
      --output espeak_vocals.wav
        """
    )

    parser.add_argument('--audio', type=str, required=True,
                       help='Input audio file (WAV)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output WAV file')

    # Optional eSpeak parameters
    parser.add_argument('--voice', type=str, default='en-us',
                       help='eSpeak voice (default: en-us)')
    parser.add_argument('--rate', type=int, default=150,
                       help='Speech rate in WPM (default: 150)')
    parser.add_argument('--base-pitch', type=int, default=60,
                       help='Base MIDI pitch (default: 60 = C4)')
    parser.add_argument('--keep-intermediates', action='store_true',
                       help='Keep extracted MIDI and lyrics files')

    args = parser.parse_args()

    print("="*70)
    print("eSpeak Singing Synthesis from Audio")
    print("="*70)
    print(f"Input: {args.audio}")
    print(f"Output: {args.output}")
    print("="*70)

    # Create temp directory for intermediate files
    tmpdir = tempfile.mkdtemp(prefix='espeak_extract_')

    try:
        # Step 1: Extract MIDI and lyrics
        midi_path, lyrics = extract_midi_and_lyrics(args.audio, tmpdir)

        # Step 2: Synthesize with eSpeak
        synthesize_with_espeak(
            midi_path=midi_path,
            lyrics=lyrics,
            output_path=args.output,
            voice=args.voice,
            rate=args.rate,
            base_pitch=args.base_pitch
        )

        # Optionally save intermediates
        if args.keep_intermediates:
            import shutil
            output_dir = Path(args.output).parent
            saved_midi = output_dir / (Path(args.output).stem + '_extracted.mid')
            saved_lyrics = output_dir / (Path(args.output).stem + '_extracted_lyrics.txt')

            shutil.copy(midi_path, saved_midi)
            with open(saved_lyrics, 'w') as f:
                f.write(lyrics)

            print(f"\n📁 Intermediate files saved:")
            print(f"   MIDI: {saved_midi}")
            print(f"   Lyrics: {saved_lyrics}")

        print(f"\n{'='*70}")
        print("✅ Complete! eSpeak singing voice generated successfully")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
