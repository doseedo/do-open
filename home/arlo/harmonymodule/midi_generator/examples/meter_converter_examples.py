#!/usr/bin/env python3
"""
Meter Converter - Comprehensive Usage Examples

This file demonstrates all major features of the Meter Converter module
with practical, real-world examples.

Examples:
1. Basic conversions (4/4 → 3/4, 5/4, 7/8)
2. Jazz to odd meters (Take Five style)
3. Progressive rock polymetric
4. Compound meter conversions
5. Metric modulation demonstrations
6. Phrase-aware conversions
7. Batch processing
8. Custom configurations

Author: Agent 7 - Modular Fusion Enhancement Project
Date: 2025-11-19
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation.meter_converter import (
    MeterConverter,
    MetricModulator,
    PhrasePreserver,
    MeterUtilities,
    TimeSignatureInfo,
    ConversionStrategy,
    MeterConversionParams,
    MeterFamily,
    PhraseBoundary,
    convert_midi_meter
)

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from fractions import Fraction


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_demo_midi(time_sig=(4, 4), tempo=120, num_measures=8, filename="demo.mid"):
    """
    Create a demo MIDI file for testing.

    Args:
        time_sig: Time signature tuple (numerator, denominator)
        tempo: Tempo in BPM
        num_measures: Number of measures
        filename: Output filename

    Returns:
        Path to created file
    """
    midi = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    midi.tracks.append(track)

    # Add metadata
    track.append(MetaMessage('track_name', name='Demo Track', time=0))
    track.append(MetaMessage('time_signature',
                            numerator=time_sig[0],
                            denominator=time_sig[1],
                            time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

    # Add notes (simple melody)
    ppqn = 480
    beats_per_measure = (time_sig[0] * 4) / time_sig[1]
    ticks_per_measure = int(ppqn * beats_per_measure)

    # Create a simple melodic pattern
    scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale

    for measure in range(num_measures):
        for beat in range(int(beats_per_measure)):
            pitch = scale[(measure * int(beats_per_measure) + beat) % len(scale)]

            # Calculate delta time
            if measure == 0 and beat == 0:
                delta = 0
            else:
                delta = ppqn

            track.append(Message('note_on', note=pitch, velocity=80,
                               channel=0, time=delta))
            track.append(Message('note_off', note=pitch, velocity=0,
                               channel=0, time=ppqn // 2))

    # End of track
    track.append(MetaMessage('end_of_track', time=0))

    # Save file
    filepath = Path(__file__).parent / filename
    midi.save(str(filepath))
    print(f"Created demo file: {filepath}")

    return str(filepath)


def print_result_summary(result, description):
    """Print a formatted summary of conversion result."""
    print("\n" + "="*70)
    print(f"EXAMPLE: {description}")
    print("="*70)

    if result.success:
        print("✓ Conversion successful!")
        print(f"  New time signature: {result.new_time_signature}")
        print(f"  Tempo change factor: {result.tempo_change_factor:.3f}x")

        if result.warnings:
            print(f"  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"    - {warning}")

        if result.stats:
            print(f"  Stats:")
            for key, value in result.stats.items():
                print(f"    - {key}: {value}")
    else:
        print("✗ Conversion failed!")
        if result.warnings:
            for warning in result.warnings:
                print(f"  - {warning}")

    print("="*70 + "\n")


# ==============================================================================
# EXAMPLE 1: Basic Conversions
# ==============================================================================

def example_1_basic_conversions():
    """
    Example 1: Basic time signature conversions

    Demonstrates:
    - 4/4 to 3/4 (waltz)
    - 4/4 to 2/4 (march)
    - 4/4 to 6/8 (compound)
    """
    print("\n" + "#"*70)
    print("# EXAMPLE 1: Basic Conversions")
    print("#"*70)

    # Create demo file in 4/4
    demo_file = create_demo_midi((4, 4), 120, 8, "demo_4_4.mid")

    # Convert to 3/4 (waltz)
    print("\n1A. Converting 4/4 to 3/4 (Waltz)")
    converter = MeterConverter(demo_file)
    result = converter.convert_meter(3, 4)

    output_file = Path(__file__).parent / "output_3_4_waltz.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "4/4 to 3/4 Conversion")

    # Convert to 2/4 (march)
    print("\n1B. Converting 4/4 to 2/4 (March)")
    result = converter.convert_meter(2, 4)

    output_file = Path(__file__).parent / "output_2_4_march.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "4/4 to 2/4 Conversion")

    # Convert to 6/8 (compound)
    print("\n1C. Converting 4/4 to 6/8 (Compound)")
    params = MeterConversionParams(strategy=ConversionStrategy.STRETCH)
    result = converter.convert_meter(6, 8, params=params)

    output_file = Path(__file__).parent / "output_6_8_compound.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "4/4 to 6/8 Conversion")


# ==============================================================================
# EXAMPLE 2: Jazz to Odd Meters (Take Five Style)
# ==============================================================================

def example_2_jazz_odd_meters():
    """
    Example 2: Jazz compositions in odd meters

    Demonstrates:
    - 4/4 to 5/4 (Take Five - Dave Brubeck)
    - Custom groupings [3+2] vs [2+3]
    - Phrase-aware conversion for jazz phrasing
    """
    print("\n" + "#"*70)
    print("# EXAMPLE 2: Jazz to Odd Meters (Take Five Style)")
    print("#"*70)

    # Create jazzy demo file
    demo_file = create_demo_midi((4, 4), 180, 16, "jazz_standard.mid")

    converter = MeterConverter(demo_file)

    # Convert to 5/4 with [3+2] grouping (Take Five)
    print("\n2A. Converting to 5/4 with [3+2] grouping (Take Five)")
    params = MeterConversionParams(
        strategy=ConversionStrategy.PHRASE_AWARE,
        preserve_phrase_structure=True
    )

    result = converter.convert_meter(
        new_numerator=5,
        new_denominator=4,
        new_grouping=[3, 2],
        params=params
    )

    output_file = Path(__file__).parent / "jazz_5_4_take_five.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "Jazz 4/4 to 5/4 [3+2]")

    # Convert to 5/4 with [2+3] grouping (alternative)
    print("\n2B. Converting to 5/4 with [2+3] grouping")
    result = converter.convert_meter(
        new_numerator=5,
        new_denominator=4,
        new_grouping=[2, 3],
        params=params
    )

    output_file = Path(__file__).parent / "jazz_5_4_alt_grouping.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "Jazz 4/4 to 5/4 [2+3]")


# ==============================================================================
# EXAMPLE 3: Progressive Rock Polymetric
# ==============================================================================

def example_3_prog_rock():
    """
    Example 3: Progressive rock polymetric patterns

    Demonstrates:
    - 4/4 to 7/8 (Pink Floyd "Money" style)
    - 4/4 to 13/8 (Tool/Meshuggah style)
    - Complex groupings
    """
    print("\n" + "#"*70)
    print("# EXAMPLE 3: Progressive Rock Polymetric")
    print("#"*70)

    demo_file = create_demo_midi((4, 4), 140, 8, "rock_riff.mid")
    converter = MeterConverter(demo_file)

    # 7/8 with [2+2+3] grouping (Money style)
    print("\n3A. Converting to 7/8 [2+2+3] (Pink Floyd style)")
    params = MeterConversionParams(strategy=ConversionStrategy.REDISTRIBUTE)

    result = converter.convert_meter(
        new_numerator=7,
        new_denominator=8,
        new_grouping=[2, 2, 3],
        params=params
    )

    output_file = Path(__file__).parent / "rock_7_8_money.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "Rock 4/4 to 7/8 [2+2+3]")

    # 13/8 (Tool style)
    print("\n3B. Converting to 13/8 [3+3+3+2+2] (Tool style)")
    result = converter.convert_meter(
        new_numerator=13,
        new_denominator=8,
        new_grouping=[3, 3, 3, 2, 2],
        params=params
    )

    output_file = Path(__file__).parent / "rock_13_8_tool.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "Rock 4/4 to 13/8 [3+3+3+2+2]")


# ==============================================================================
# EXAMPLE 4: Metric Modulation
# ==============================================================================

def example_4_metric_modulation():
    """
    Example 4: Elliott Carter-style metric modulation

    Demonstrates:
    - Finding pivot rhythms
    - Calculating tempo relationships
    - Smooth metric transitions
    """
    print("\n" + "#"*70)
    print("# EXAMPLE 4: Metric Modulation (Elliott Carter)")
    print("#"*70)

    demo_file = create_demo_midi((4, 4), 120, 8, "classical.mid")
    converter = MeterConverter(demo_file)

    # Use metric modulation strategy
    print("\n4A. Metric modulation 4/4 to 3/4")
    params = MeterConversionParams(strategy=ConversionStrategy.METRIC_MODULATION)

    result = converter.convert_meter(3, 4, params=params)

    if 'pivot_rhythm' in result.stats:
        pivot = result.stats['pivot_rhythm']
        print(f"\nPivot rhythm details:")
        print(f"  Name: {pivot.get('name')}")
        print(f"  Old tempo: {pivot.get('old_tempo')} BPM")
        print(f"  New tempo: {pivot.get('new_tempo')} BPM")

    output_file = Path(__file__).parent / "metric_mod_3_4.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "Metric Modulation 4/4 to 3/4")

    # Demonstrate pivot calculation
    print("\n4B. Calculating pivot rhythms manually")
    old_ts = TimeSignatureInfo(4, 4)
    new_ts = TimeSignatureInfo(5, 4)

    pivot_value, tempo_ratio = MetricModulator.find_best_pivot(old_ts, new_ts)

    print(f"\nBest pivot for 4/4 → 5/4:")
    print(f"  Pivot value: {pivot_value}")
    print(f"  Tempo ratio: {tempo_ratio:.3f}")
    print(f"  If old tempo = 120 BPM, new tempo = {120 * tempo_ratio:.1f} BPM")

    # Try different pivots
    for pivot_name, pivot_frac in [
        ('Quarter note', Fraction(1, 1)),
        ('Eighth note', Fraction(1, 2)),
        ('Dotted quarter', Fraction(3, 2))
    ]:
        rel = MetricModulator.calculate_tempo_relationship(old_ts, new_ts, pivot_frac)
        print(f"\n  With {pivot_name} pivot:")
        print(f"    Tempo ratio: {rel['tempo_ratio']:.3f}")
        print(f"    New tempo: {120 * rel['tempo_ratio']:.1f} BPM")


# ==============================================================================
# EXAMPLE 5: Phrase-Aware Conversion
# ==============================================================================

def example_5_phrase_aware():
    """
    Example 5: Phrase-aware conversion

    Demonstrates:
    - Automatic phrase detection
    - Preserving phrase structure
    - Custom phrase boundaries
    """
    print("\n" + "#"*70)
    print("# EXAMPLE 5: Phrase-Aware Conversion")
    print("#"*70)

    # Create file with clear phrases
    demo_file = create_demo_midi((4, 4), 120, 16, "song_with_phrases.mid")

    converter = MeterConverter(demo_file)

    # Detect phrases
    print("\n5A. Detecting phrase boundaries")
    boundaries = converter._detect_phrase_boundaries()

    print(f"\nFound {len(boundaries)} phrase boundaries:")
    for i, boundary in enumerate(boundaries):
        print(f"  Boundary {i+1}:")
        print(f"    Measure: {boundary.measure_number}")
        print(f"    Type: {boundary.boundary_type}")
        print(f"    Confidence: {boundary.confidence:.2f}")

    # Convert with phrase awareness
    print("\n5B. Converting with phrase awareness")
    params = MeterConversionParams(
        strategy=ConversionStrategy.PHRASE_AWARE,
        preserve_phrase_structure=True
    )

    result = converter.convert_meter(3, 4, params=params)

    output_file = Path(__file__).parent / "phrase_aware_3_4.mid"
    if result.success:
        result.new_midi.save(str(output_file))
        print(f"Saved: {output_file}")

    print_result_summary(result, "Phrase-Aware 4/4 to 3/4")

    # Custom phrase boundaries
    print("\n5C. Using custom phrase boundaries")
    custom_boundaries = [
        PhraseBoundary(
            measure_number=4,
            tick_position=converter.current_time_sig.ticks_per_measure * 4,
            boundary_type='cadence',
            confidence=1.0
        ),
        PhraseBoundary(
            measure_number=8,
            tick_position=converter.current_time_sig.ticks_per_measure * 8,
            boundary_type='end',
            confidence=1.0
        )
    ]

    converter.phrase_boundaries = custom_boundaries
    result = converter.convert_meter(5, 4, params=params)

    print_result_summary(result, "Custom Phrase Boundaries 4/4 to 5/4")


# ==============================================================================
# EXAMPLE 6: Quantization Control
# ==============================================================================

def example_6_quantization():
    """
    Example 6: Quantization control

    Demonstrates:
    - Different quantization strengths
    - Groove preservation
    - Accent patterns
    """
    print("\n" + "#"*70)
    print("# EXAMPLE 6: Quantization Control")
    print("#"*70)

    demo_file = create_demo_midi((4, 4), 120, 8, "groove.mid")
    converter = MeterConverter(demo_file)

    # Different quantization strengths
    for strength in [1.0, 0.5, 0.0]:
        print(f"\n6.{['A', 'B', 'C'][int((1.0 - strength) / 0.5)]}. "
              f"Quantization strength: {strength}")

        params = MeterConversionParams(
            strategy=ConversionStrategy.STRETCH,
            quantize_output=True,
            quantize_strength=strength
        )

        result = converter.convert_meter(7, 8, params=params)

        output_file = Path(__file__).parent / f"quantize_{int(strength*10)}.mid"
        if result.success:
            result.new_midi.save(str(output_file))
            print(f"Saved: {output_file}")

    # Demonstrate accent patterns
    print("\n6D. Meter accent patterns")
    for num, denom, grouping in [
        (4, 4, None),
        (3, 4, None),
        (7, 8, [2, 2, 3]),
        (5, 4, [3, 2])
    ]:
        ts = TimeSignatureInfo(num, denom, grouping=grouping)
        accents = MeterUtilities.get_meter_accent_pattern(ts)
        print(f"\n{ts}:")
        print(f"  Accent pattern: {accents}")


# ==============================================================================
# EXAMPLE 7: Batch Processing
# ==============================================================================

def example_7_batch_processing():
    """
    Example 7: Batch processing multiple files

    Demonstrates:
    - Processing multiple files
    - Different target meters
    - Error handling
    """
    print("\n" + "#"*70)
    print("# EXAMPLE 7: Batch Processing")
    print("#"*70)

    # Create several demo files
    demo_files = []
    for i, (ts, tempo) in enumerate([
        ((4, 4), 120),
        ((3, 4), 140),
        ((6, 8), 90)
    ]):
        filename = f"batch_demo_{i+1}.mid"
        filepath = create_demo_midi(ts, tempo, 8, filename)
        demo_files.append(filepath)

    # Batch convert all to 5/4
    print("\n7A. Converting all files to 5/4")
    results = []

    for filepath in demo_files:
        converter = MeterConverter(filepath)
        result = converter.convert_meter(5, 4)
        results.append((filepath, result))

        if result.success:
            output_name = Path(filepath).stem + "_5_4.mid"
            output_path = Path(__file__).parent / output_name
            result.new_midi.save(str(output_path))
            print(f"✓ Converted {Path(filepath).name} → {output_name}")
        else:
            print(f"✗ Failed {Path(filepath).name}")

    # Summary
    successful = sum(1 for _, r in results if r.success)
    print(f"\nBatch processing complete: {successful}/{len(demo_files)} successful")


# ==============================================================================
# EXAMPLE 8: All Strategies Comparison
# ==============================================================================

def example_8_strategy_comparison():
    """
    Example 8: Compare all conversion strategies

    Demonstrates:
    - All four strategies on same file
    - Performance comparison
    - Quality differences
    """
    print("\n" + "#"*70)
    print("# EXAMPLE 8: Strategy Comparison")
    print("#"*70)

    demo_file = create_demo_midi((4, 4), 120, 8, "strategy_test.mid")

    strategies = [
        (ConversionStrategy.STRETCH, "Stretch"),
        (ConversionStrategy.REDISTRIBUTE, "Redistribute"),
        (ConversionStrategy.METRIC_MODULATION, "Metric Modulation"),
        (ConversionStrategy.PHRASE_AWARE, "Phrase Aware")
    ]

    converter = MeterConverter(demo_file)

    print("\nConverting 4/4 to 7/8 using all strategies:\n")

    for strategy, name in strategies:
        print(f"8{chr(65 + strategies.index((strategy, name)))}. {name} Strategy")

        params = MeterConversionParams(strategy=strategy)
        result = converter.convert_meter(7, 8, new_grouping=[2, 2, 3], params=params)

        if result.success:
            output_file = Path(__file__).parent / f"strategy_{name.lower().replace(' ', '_')}.mid"
            result.new_midi.save(str(output_file))
            print(f"  ✓ Success - saved to {output_file.name}")
            print(f"  Tempo factor: {result.tempo_change_factor:.3f}x")

            if result.stats:
                print(f"  Stats: {list(result.stats.keys())}")
        else:
            print(f"  ✗ Failed")

        print()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("METER CONVERTER - COMPREHENSIVE EXAMPLES")
    print("Agent 7 - Modular Fusion Enhancement Project")
    print("="*70)

    examples = [
        ("Basic Conversions", example_1_basic_conversions),
        ("Jazz Odd Meters", example_2_jazz_odd_meters),
        ("Progressive Rock", example_3_prog_rock),
        ("Metric Modulation", example_4_metric_modulation),
        ("Phrase-Aware", example_5_phrase_aware),
        ("Quantization", example_6_quantization),
        ("Batch Processing", example_7_batch_processing),
        ("Strategy Comparison", example_8_strategy_comparison)
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all examples")

    try:
        choice = input("\nSelect example (0-8): ").strip()

        if choice == "0":
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\n✗ Error in {name}: {e}")
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            name, func = examples[int(choice) - 1]
            func()
        else:
            print("Invalid choice. Running all examples...")
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\n✗ Error in {name}: {e}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")

    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
