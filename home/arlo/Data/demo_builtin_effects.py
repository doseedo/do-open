#!/usr/bin/env python3
"""
Demo script showing pedalboard's built-in effects (as alternative to VST)
"""

from pedalboard import Pedalboard, Reverb, Delay, Chorus, Distortion, Compressor
from pedalboard.io import AudioFile
import sys

def process_with_builtin_reverb(input_file, output_file):
    """Process audio with built-in reverb effect (similar to VintageVerb)"""

    print(f"Processing {input_file} with built-in reverb...")

    # Create a reverb effect (similar to what VintageVerb would do)
    board = Pedalboard([
        Reverb(
            room_size=0.75,      # Large room
            damping=0.5,         # Medium damping
            wet_level=0.33,      # Mix wet/dry
            dry_level=0.67,      # Dry signal level
            width=1.0,           # Stereo width
        )
    ])

    # Read, process, and write audio
    with AudioFile(input_file) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
        num_channels = f.num_channels

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {num_channels}")

    # Process
    processed = board(audio, sample_rate)

    # Write output
    with AudioFile(output_file, 'w', sample_rate, num_channels) as f:
        f.write(processed)

    print(f"✓ Saved to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python demo_builtin_effects.py <input.wav> <output.wav>")
        sys.exit(1)

    process_with_builtin_reverb(sys.argv[1], sys.argv[2])
