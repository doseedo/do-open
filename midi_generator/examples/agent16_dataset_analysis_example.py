#!/usr/bin/env python3
"""
AGENT 16: Dataset Analysis Example
===================================

This example demonstrates how to use the Dataset Analyzer to:
1. Analyze multiple MIDI files from a dataset
2. Extract statistical patterns
3. Measure swing ratios, comping rhythms, and melodic intervals
4. Compare generated music to real recordings
5. Extract bebop licks and walking bass patterns

Author: Agent 16 - MIDI Dataset Analysis Engine
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.dataset_analyzer import DatasetAnalyzer
from analysis.midi_analyzer import MidiAnalyzer


def example_1_analyze_single_file():
    """Example 1: Analyze a single MIDI file."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Analyze Single MIDI File")
    print("="*80 + "\n")

    # Use one of the existing MIDI files
    midi_path = "/home/user/Do/midi_generator/swing_fixed.mid"

    if not Path(midi_path).exists():
        print(f"File not found: {midi_path}")
        print("Please provide a valid MIDI file path")
        return

    # Analyze
    analyzer = MidiAnalyzer(midi_path)
    result = analyzer.analyze()

    # Print results
    analyzer.print_analysis()


def example_2_analyze_dataset():
    """Example 2: Analyze multiple MIDI files and extract statistics."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Analyze Dataset of MIDI Files")
    print("="*80 + "\n")

    # Collect all MIDI files in midi_generator directory
    midi_dir = Path("/home/user/Do/midi_generator")
    midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
    midi_files = [str(f) for f in midi_files]

    if not midi_files:
        print("No MIDI files found in directory")
        return

    print(f"Found {len(midi_files)} MIDI files to analyze\n")

    # Create analyzer
    dataset_analyzer = DatasetAnalyzer()

    # Analyze all files
    stats = dataset_analyzer.analyze_dataset(
        midi_files,
        analyze_chords=True,
        analyze_swing=True,
        analyze_comping=True,
        analyze_intervals=True,
        verbose=True
    )

    # Results are printed automatically by analyze_dataset

    # Save statistics
    output_path = "/home/user/Do/midi_generator/dataset_statistics.json"
    dataset_analyzer.save_statistics(output_path)


def example_3_measure_swing_ratio():
    """Example 3: Measure swing ratio from real recordings."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Measure Swing Ratio")
    print("="*80 + "\n")

    midi_path = "/home/user/Do/midi_generator/swing_fixed.mid"

    if not Path(midi_path).exists():
        print(f"File not found: {midi_path}")
        return

    # Analyze file
    analyzer = MidiAnalyzer(midi_path)
    result = analyzer.analyze()

    # Measure swing
    dataset_analyzer = DatasetAnalyzer()
    swing_measurement = dataset_analyzer._measure_swing_ratio(result)

    if swing_measurement:
        print("Swing Measurement:")
        print(swing_measurement)
        print(f"\nInterpretation:")
        if swing_measurement.swing_ratio < 0.55:
            print("  - Light swing (close to straight eighths)")
        elif swing_measurement.swing_ratio < 0.62:
            print("  - Medium swing")
        elif swing_measurement.swing_ratio < 0.68:
            print("  - Heavy swing (close to triplet feel)")
        else:
            print("  - Very heavy swing / shuffle")
    else:
        print("Could not measure swing ratio (insufficient data)")


def example_4_extract_patterns():
    """Example 4: Extract melodic patterns and chord progressions."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Extract Musical Patterns")
    print("="*80 + "\n")

    # Find MIDI files
    midi_dir = Path("/home/user/Do/midi_generator")
    midi_files = list(midi_dir.glob("*.mid"))
    midi_files = [str(f) for f in midi_files]

    if not midi_files:
        print("No MIDI files found")
        return

    # Create analyzer and analyze dataset
    dataset_analyzer = DatasetAnalyzer()
    stats = dataset_analyzer.analyze_dataset(
        midi_files,
        verbose=False
    )

    # Extract bebop licks
    print("Extracting melodic patterns (bebop licks)...")
    licks = dataset_analyzer.extract_bebop_licks(min_length=4, max_length=8)

    print(f"\nExtracted {len(licks)} common melodic patterns")
    print("\nTop 10 patterns:")
    for i, lick in enumerate(licks[:10], 1):
        lick_str = ' '.join(f'{interval:+d}' for interval in lick)
        print(f"  {i}. [{lick_str}]")

    # Extract walking bass patterns
    print("\n\nExtracting walking bass patterns...")
    bass_patterns = dataset_analyzer.extract_walking_bass_patterns()

    print(f"\nExtracted {len(bass_patterns)} bass patterns")
    if bass_patterns:
        print("\nTop 5 patterns (MIDI note numbers):")
        for i, pattern in enumerate(bass_patterns[:5], 1):
            print(f"  {i}. {pattern}")


def example_5_validate_generated_music():
    """Example 5: Compare generated music to real dataset."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Validate Generated Music Against Dataset")
    print("="*80 + "\n")

    # Build reference dataset
    midi_dir = Path("/home/user/Do/midi_generator")
    reference_files = list(midi_dir.glob("*.mid"))
    reference_files = [str(f) for f in reference_files if "swing" in str(f).lower()]

    if not reference_files:
        print("No reference MIDI files found")
        return

    print(f"Building reference dataset from {len(reference_files)} files...\n")

    # Analyze reference dataset
    dataset_analyzer = DatasetAnalyzer()
    dataset_analyzer.analyze_dataset(reference_files, verbose=False)

    # Compare to a generated file (in this example, we'll use one of the existing files)
    generated_file = "/home/user/Do/midi_generator/swing_v3.mid"

    if not Path(generated_file).exists():
        print(f"Generated file not found: {generated_file}")
        return

    print(f"Comparing: {Path(generated_file).name}")
    print("-" * 80 + "\n")

    # Run comparison
    comparison = dataset_analyzer.compare_generated_to_dataset(generated_file)

    # Print results
    print("Validation Metrics:")
    print("-" * 80)

    for metric, score in sorted(comparison.items()):
        # Color code based on score
        if score >= 0.85:
            status = "✓ EXCELLENT"
        elif score >= 0.70:
            status = "✓ GOOD"
        elif score >= 0.50:
            status = "○ FAIR"
        else:
            status = "✗ NEEDS WORK"

        print(f"  {metric:25s}: {score:6.2%}  {status}")

    print("-" * 80)

    # Overall assessment
    overall = comparison.get('overall_authenticity', 0)
    print(f"\nOverall Authenticity: {overall:.2%}")

    if overall >= 0.85:
        print("Assessment: Generated music is highly authentic!")
    elif overall >= 0.70:
        print("Assessment: Generated music is good, with room for improvement.")
    else:
        print("Assessment: Generated music needs significant improvement.")


def example_6_swing_tempo_correlation():
    """Example 6: Analyze relationship between tempo and swing ratio."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Swing Ratio vs. Tempo Correlation")
    print("="*80 + "\n")

    midi_dir = Path("/home/user/Do/midi_generator")
    midi_files = list(midi_dir.glob("*.mid"))
    midi_files = [str(f) for f in midi_files]

    if not midi_files:
        print("No MIDI files found")
        return

    # Analyze dataset
    dataset_analyzer = DatasetAnalyzer()
    stats = dataset_analyzer.analyze_dataset(
        midi_files,
        analyze_swing=True,
        verbose=False
    )

    if not stats.swing_measurements:
        print("No swing measurements available")
        return

    # Print correlation
    print(f"Swing-Tempo Correlation: {stats.swing_tempo_correlation:.3f}")
    print()

    if stats.swing_tempo_correlation < -0.3:
        print("Interpretation: Negative correlation detected!")
        print("  → Swing ratio decreases as tempo increases")
        print("  → Fast tempos use lighter swing")
        print("  → Slow tempos use heavier swing")
    elif stats.swing_tempo_correlation > 0.3:
        print("Interpretation: Positive correlation detected!")
        print("  → Swing ratio increases as tempo increases")
    else:
        print("Interpretation: No significant correlation")
        print("  → Swing ratio is independent of tempo")

    # Show data points
    print(f"\nSwing Measurements (n={len(stats.swing_measurements)}):")
    print("-" * 60)

    for measurement in sorted(stats.swing_measurements, key=lambda s: s.tempo):
        print(f"  {measurement}")


def main():
    """Run all examples."""
    examples = [
        ("Analyze Single MIDI File", example_1_analyze_single_file),
        ("Analyze Dataset", example_2_analyze_dataset),
        ("Measure Swing Ratio", example_3_measure_swing_ratio),
        ("Extract Musical Patterns", example_4_extract_patterns),
        ("Validate Generated Music", example_5_validate_generated_music),
        ("Swing-Tempo Correlation", example_6_swing_tempo_correlation),
    ]

    print("\n" + "="*80)
    print("AGENT 16: DATASET ANALYSIS EXAMPLES")
    print("="*80)
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nEnter example number (1-6), 'all' to run all, or 'q' to quit: ", end='')

    try:
        choice = input().strip().lower()

        if choice == 'q':
            print("Exiting...")
            return

        if choice == 'all':
            for name, example_func in examples:
                print(f"\n\nRunning: {name}")
                print("="*80)
                example_func()
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                name, example_func = examples[idx]
                print(f"\n\nRunning: {name}")
                print("="*80)
                example_func()
            else:
                print("Invalid choice")

    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")


if __name__ == "__main__":
    # Run example 2 by default (full dataset analysis)
    example_2_analyze_dataset()

    # Uncomment to run interactive menu
    # main()
