#!/usr/bin/env python3
"""
Big Band Generator - Command Line Interface
============================================

Simple CLI for generating professional big band arrangements.

Usage:
    # Simple generation
    python generate_big_band.py --style basie --tempo 140

    # Full options
    python generate_big_band.py \\
        --style ellington \\
        --tempo 120 \\
        --key Eb \\
        --form aaba \\
        --progression jazz_blues \\
        --output my_arrangement.mid

Examples:
    # Count Basie swing at 180 BPM
    python generate_big_band.py --style basie --tempo 180 --key C

    # Duke Ellington ballad in Eb
    python generate_big_band.py --style ellington --tempo 80 --key Eb

    # Modern big band with Coltrane changes
    python generate_big_band.py --style thad_jones --progression coltrane_changes

Author: Agent 18 - Integration Architecture Designer
"""

import sys
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from api.big_band_api import BigBandGenerator, list_available_styles
except ImportError as e:
    print(f"ERROR: Could not import BigBandGenerator: {e}")
    print("\nMake sure you're running from the correct directory:")
    print("  cd /path/to/Do")
    print("  python midi_generator/tools/big_band/generate_big_band.py")
    sys.exit(1)


def main():
    """Main CLI entry point"""

    parser = argparse.ArgumentParser(
        description="Professional Big Band Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Count Basie swing
  %(prog)s --style basie --tempo 140 --key C --output basie_swing.mid

  # Duke Ellington ballad
  %(prog)s --style ellington --tempo 80 --key Eb --output ellington_ballad.mid

  # Modern big band
  %(prog)s --style thad_jones --tempo 160 --form aaba --output modern_jazz.mid

Available Styles:
  """ + ", ".join(list_available_styles()) + """

Progression Types:
  jazz_blues, rhythm_changes, ii_V_I, minor_ii_V_i, coltrane_changes,
  autumn_leaves, take_five, blue_bossa, dorian_vamp, mixolydian_rock,
  plr_film, modal_interchange, reharmonized_blues, quartal_harmony
        """
    )

    # Required arguments
    parser.add_argument(
        '--style',
        type=str,
        default='basie',
        choices=list_available_styles(),
        help='Arranging style (default: basie)'
    )

    # Optional arguments
    parser.add_argument(
        '--tempo',
        type=int,
        default=140,
        help='Tempo in BPM (default: 140)'
    )

    parser.add_argument(
        '--key',
        type=str,
        default='C',
        help='Key signature (C, Eb, F#, etc.) (default: C)'
    )

    parser.add_argument(
        '--form',
        type=str,
        default='aaba',
        choices=['aaba', 'blues'],
        help='Musical form (default: aaba)'
    )

    parser.add_argument(
        '--progression',
        type=str,
        default='jazz_blues',
        help='Chord progression type (default: jazz_blues)'
    )

    parser.add_argument(
        '--swing',
        type=float,
        default=None,
        help='Swing ratio override (0.5-0.67, default: style-specific)'
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output filename (default: <style>_<tempo>bpm.mid)'
    )

    parser.add_argument(
        '--list-styles',
        action='store_true',
        help='List available styles and exit'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Big Band Generator 1.0.0 (Agent 18)'
    )

    args = parser.parse_args()

    # List styles if requested
    if args.list_styles:
        print("\nAvailable Big Band Styles:")
        print("=" * 50)
        for style in list_available_styles():
            print(f"  - {style}")
        print()
        return 0

    # Determine output filename
    if args.output is None:
        args.output = f"{args.style}_{args.tempo}bpm.mid"

    # Print banner
    print("\n" + "=" * 80)
    print("BIG BAND GENERATOR - Command Line Interface")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Style:       {args.style}")
    print(f"  Tempo:       {args.tempo} BPM")
    print(f"  Key:         {args.key}")
    print(f"  Form:        {args.form}")
    print(f"  Progression: {args.progression}")
    if args.swing:
        print(f"  Swing:       {args.swing:.2f}")
    print(f"  Output:      {args.output}")
    print()

    try:
        # Create generator
        generator = BigBandGenerator(
            style=args.style,
            tempo=args.tempo,
            key=args.key,
            form=args.form,
            progression_type=args.progression,
            swing_ratio=args.swing
        )

        # Generate
        midi = generator.generate()

        # Save
        midi.save(args.output)

        print()
        print(f"✅ Success! Saved: {args.output}")
        print()
        print("Professional features applied:")
        print(f"  ✅ {generator.style_profile.style_name} arranging style")
        print(f"  ✅ Proper {args.form.upper()} form structure")
        print(f"  ✅ {generator.style_profile.harmony_complexity*100:.0f}% harmonic complexity")
        print(f"  ✅ Swing ratio: {generator.style_profile.swing_ratio:.2f}")
        print(f"  ✅ {generator.form.total_bars} bars total")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
