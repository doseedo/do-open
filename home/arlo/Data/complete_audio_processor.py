#!/usr/bin/env python3
"""
Complete Audio Processor
Supports both VST plugins and built-in effects
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from pedalboard import (
    Pedalboard, load_plugin,
    Reverb, Delay, Chorus, Distortion, Compressor, Gain, Limiter,
    Phaser, Bitcrush, LadderFilter, HighpassFilter, LowpassFilter
)
from pedalboard.io import AudioFile


# Preset effect chains
PRESETS = {
    'vintage_reverb': [
        Reverb(room_size=0.8, damping=0.7, wet_level=0.3, dry_level=0.7, width=1.0),
        Chorus(rate_hz=0.5, depth=0.15, mix=0.2),
        Gain(gain_db=-2)
    ],
    'hall_reverb': [
        Reverb(room_size=0.95, damping=0.4, wet_level=0.4, dry_level=0.6, width=1.0),
        Limiter(threshold_db=-1.0)
    ],
    'tape_echo': [
        Delay(delay_seconds=0.375, feedback=0.45, mix=0.3),
        Bitcrush(bit_depth=12),
        Gain(gain_db=-3),
        HighpassFilter(cutoff_frequency_hz=200)
    ],
    'dream_space': [
        Reverb(room_size=0.9, damping=0.6, wet_level=0.5, dry_level=0.5),
        Chorus(rate_hz=0.3, depth=0.25, mix=0.3),
        Delay(delay_seconds=0.5, feedback=0.3, mix=0.2),
        Limiter(threshold_db=-1.0)
    ],
    'lo_fi': [
        Bitcrush(bit_depth=8),
        LowpassFilter(cutoff_frequency_hz=4000),
        Gain(gain_db=-2)
    ],
    'warm_compress': [
        Compressor(threshold_db=-12, ratio=4.0, attack_ms=10, release_ms=100),
        Gain(gain_db=3),
        Limiter(threshold_db=-0.5)
    ]
}


def process_with_vst(input_file, output_file, vst_path, **vst_params):
    """Process audio with a VST plugin"""
    print(f"\n{'='*60}")
    print("VST PLUGIN PROCESSING")
    print(f"{'='*60}")
    print(f"Plugin: {vst_path}")

    try:
        plugin = load_plugin(vst_path)
        print(f"✓ Loaded: {plugin}")

        # Set parameters
        if vst_params:
            print("\nParameters:")
            for key, value in vst_params.items():
                try:
                    setattr(plugin, key, value)
                    print(f"  {key} = {value}")
                except Exception as e:
                    print(f"  Warning: {key} = {value} (error: {e})")

        board = Pedalboard([plugin])

    except Exception as e:
        print(f"✗ Error loading VST: {e}")
        print("\nNote: Only VST3 plugins are supported on Linux")
        return False

    # Process
    with AudioFile(input_file) as f:
        audio = f.read(f.frames)
        sr = f.samplerate
        channels = f.num_channels

    print(f"\nInput: {input_file}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Channels: {channels}")
    print(f"  Duration: {len(audio[0])/sr:.2f}s")

    processed = board(audio, sr)

    with AudioFile(output_file, 'w', sr, channels) as f:
        f.write(processed)

    print(f"\n✓ Output: {output_file}")
    print(f"{'='*60}\n")
    return True


def process_with_preset(input_file, output_file, preset_name):
    """Process audio with a built-in preset"""
    if preset_name not in PRESETS:
        print(f"Error: Unknown preset '{preset_name}'")
        print(f"Available presets: {', '.join(PRESETS.keys())}")
        return False

    print(f"\n{'='*60}")
    print(f"PRESET: {preset_name.upper()}")
    print(f"{'='*60}")

    effects = PRESETS[preset_name]
    print("Effects chain:")
    for i, effect in enumerate(effects, 1):
        print(f"  {i}. {effect}")

    board = Pedalboard(effects)

    # Process
    with AudioFile(input_file) as f:
        audio = f.read(f.frames)
        sr = f.samplerate
        channels = f.num_channels

    print(f"\nInput: {input_file}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Channels: {channels}")
    print(f"  Duration: {len(audio[0])/sr:.2f}s")

    processed = board(audio, sr)

    with AudioFile(output_file, 'w', sr, channels) as f:
        f.write(processed)

    print(f"\n✓ Output: {output_file}")
    print(f"{'='*60}\n")
    return True


def create_test_audio(output_path, freq=440, duration=3):
    """Create a test audio file"""
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))

    # Create a more interesting test signal
    fundamental = np.sin(2 * np.pi * freq * t)
    harmonic1 = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    harmonic2 = 0.15 * np.sin(2 * np.pi * freq * 3 * t)

    audio = (fundamental + harmonic1 + harmonic2) * 0.5
    audio_stereo = np.array([audio, audio])

    with AudioFile(output_path, 'w', sr, 2) as f:
        f.write(audio_stereo)

    print(f"✓ Created test audio: {output_path}")
    print(f"  Frequency: {freq} Hz, Duration: {duration}s")


def main():
    parser = argparse.ArgumentParser(
        description='Process audio with VST plugins or built-in effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
PRESETS AVAILABLE:
{chr(10).join(f'  - {name}' for name in PRESETS.keys())}

EXAMPLES:

1. Process with built-in preset:
   python complete_audio_processor.py -i input.wav -o output.wav --preset vintage_reverb

2. Process with VST plugin:
   python complete_audio_processor.py -i input.wav -o output.wav --vst /path/to/plugin.vst3

3. Create test audio:
   python complete_audio_processor.py --create-test test.wav

4. List presets:
   python complete_audio_processor.py --list-presets
        """
    )

    parser.add_argument('-i', '--input', help='Input audio file')
    parser.add_argument('-o', '--output', help='Output audio file')
    parser.add_argument('--preset', choices=list(PRESETS.keys()),
                        help='Built-in effect preset')
    parser.add_argument('--vst', help='Path to VST3 plugin')
    parser.add_argument('--vst-param', action='append', nargs=1,
                        metavar='KEY=VALUE', help='VST parameter (repeatable)')
    parser.add_argument('--create-test', metavar='OUTPUT',
                        help='Create test audio file')
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets')

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("\nAvailable Presets:")
        print("=" * 60)
        for name, effects in PRESETS.items():
            print(f"\n{name}:")
            for effect in effects:
                print(f"  - {effect}")
        print("\n" + "=" * 60)
        return

    # Create test audio
    if args.create_test:
        create_test_audio(args.create_test)
        return

    # Normal processing
    if not args.input or not args.output:
        parser.error("Both --input and --output are required for processing")

    # Process with VST
    if args.vst:
        vst_params = {}
        if args.vst_param:
            for param in args.vst_param:
                key, value = param[0].split('=', 1)
                try:
                    vst_params[key] = float(value) if '.' in value else int(value)
                except ValueError:
                    vst_params[key] = value

        success = process_with_vst(args.input, args.output, args.vst, **vst_params)
        sys.exit(0 if success else 1)

    # Process with preset
    elif args.preset:
        success = process_with_preset(args.input, args.output, args.preset)
        sys.exit(0 if success else 1)

    else:
        parser.error("Either --preset or --vst must be specified")


if __name__ == '__main__':
    main()
