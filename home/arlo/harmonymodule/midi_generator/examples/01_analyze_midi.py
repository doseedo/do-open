#!/usr/bin/env python3
"""
Example 01: Analyze MIDI File

Demonstrates comprehensive MIDI analysis capabilities:
- Key detection (Krumhansl-Schmuckler algorithm)
- Chord recognition
- Tempo and time signature detection
- Statistical analysis
- Melodic and rhythmic analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.midi_analyzer import MidiAnalyzer


def main():
    """Analyze a MIDI file and print detailed report."""

    # Example: Analyze any MIDI file
    # For demonstration, we'll show how to use it

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                        MIDI ANALYZER EXAMPLE                                ║
║                                                                             ║
║  This example demonstrates comprehensive MIDI file analysis.               ║
║                                                                             ║
║  Usage:                                                                     ║
║    python 01_analyze_midi.py <midi_file>                                   ║
║                                                                             ║
║  Features:                                                                  ║
║    • Key detection (Krumhansl-Schmuckler algorithm)                        ║
║    • Chord recognition with confidence scores                              ║
║    • Tempo and time signature detection                                    ║
║    • Pitch class and interval histograms                                   ║
║    • Melodic contour and range analysis                                    ║
║    • Groove deviation (timing humanization)                                ║
║    • Harmonic complexity metrics                                           ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Check command line arguments
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a MIDI file path")
        print("\nExample:")
        print("  python 01_analyze_midi.py /path/to/your/file.mid")
        return

    midi_path = sys.argv[1]

    # Verify file exists
    if not Path(midi_path).exists():
        print(f"❌ Error: File not found: {midi_path}")
        return

    # Analyze the MIDI file
    print(f"\n🔍 Analyzing: {Path(midi_path).name}\n")

    analyzer = MidiAnalyzer(midi_path)

    # Perform complete analysis
    result = analyzer.analyze(
        detect_key=True,
        detect_chords=True,
        analyze_rhythm=True,
        analyze_melody=True
    )

    # Print comprehensive report
    analyzer.print_analysis()

    # Access specific analysis results programmatically
    print("\n" + "="*80)
    print("PROGRAMMATIC ACCESS EXAMPLES")
    print("="*80 + "\n")

    if result.key:
        print(f"Detected key: {result.key}")
        print(f"Key confidence: {result.key.confidence:.1%}")

    if result.chords:
        print(f"\nTotal chords: {len(result.chords)}")
        print("First 5 chords:")
        for i, chord in enumerate(result.chords[:5], 1):
            print(f"  {i}. {chord} (confidence: {chord.confidence:.1%})")

    if result.melodic_range != (0, 0):
        low, high = result.melodic_range
        print(f"\nMelodic range: {high - low} semitones")

    if result.pitch_class_histogram:
        print("\nMost common pitch classes:")
        sorted_pcs = sorted(result.pitch_class_histogram.items(),
                          key=lambda x: x[1], reverse=True)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for pc, count in sorted_pcs[:5]:
            print(f"  {note_names[pc]}: {count} occurrences")

    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
