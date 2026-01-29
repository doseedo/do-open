#!/usr/bin/env python3
"""Command line interface for arrangement generation."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Generate MIDI arrangements from v53 checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a 32-bar arrangement with default instruments
  python -m generation.cli --checkpoint checkpoint_v53.npz --output out.mid

  # Specify instruments
  python -m generation.cli --checkpoint checkpoint_v53.npz --output out.mid \\
      --instruments "Trumpet,Trombone,Alto Sax,Piano"

  # Add variation for more randomness
  python -m generation.cli --checkpoint checkpoint_v53.npz --output out.mid \\
      --variation 0.3

  # Set specific seed pattern
  python -m generation.cli --checkpoint checkpoint_v53.npz --output out.mid \\
      --seed-pattern 130
"""
    )

    parser.add_argument(
        '--checkpoint', '-c',
        required=True,
        help='Path to checkpoint .npz file'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output MIDI file path'
    )

    parser.add_argument(
        '--instruments', '-i',
        default='Trumpet,Trombone,Alto Sax,Tenor Sax,Piano',
        help='Comma-separated list of instruments'
    )

    parser.add_argument(
        '--patterns', '-n',
        type=int,
        default=16,
        help='Number of patterns to chain (default: 16)'
    )

    parser.add_argument(
        '--seed-pattern', '-s',
        type=int,
        default=None,
        help='Seed pattern ID (random if not specified)'
    )

    parser.add_argument(
        '--base-pitch', '-p',
        type=int,
        default=60,
        help='Base MIDI pitch (default: 60 = middle C)'
    )

    parser.add_argument(
        '--variation', '-v',
        type=float,
        default=0.0,
        help='Variation level 0.0-1.0 (default: 0.0)'
    )

    parser.add_argument(
        '--no-meta',
        action='store_true',
        help='Disable meta-pattern chaining (use random transforms)'
    )

    parser.add_argument(
        '--tempo',
        type=int,
        default=120,
        help='Tempo in BPM (default: 120)'
    )

    args = parser.parse_args()

    try:
        from .generator import ArrangementGenerator
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from generation.generator import ArrangementGenerator

    print(f"Loading checkpoint: {args.checkpoint}")
    generator = ArrangementGenerator(args.checkpoint)
    generator.midi_writer.tempo = args.tempo

    instruments = [i.strip() for i in args.instruments.split(',')]

    print(f"Generating arrangement:")
    print(f"  Instruments: {instruments}")
    print(f"  Patterns: {args.patterns}")
    print(f"  Base pitch: {args.base_pitch}")
    print(f"  Variation: {args.variation}")
    print(f"  Use meta-patterns: {not args.no_meta}")

    output_path = generator.generate_and_save(
        output_path=args.output,
        seed_pattern_id=args.seed_pattern,
        n_patterns=args.patterns,
        instruments=instruments,
        base_pitch=args.base_pitch,
        use_meta_patterns=not args.no_meta,
        variation=args.variation,
    )

    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
