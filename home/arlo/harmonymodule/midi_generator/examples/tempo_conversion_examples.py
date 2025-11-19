#!/usr/bin/env python3
"""
Tempo Conversion System - Comprehensive Examples

This file demonstrates all features of the tempo conversion system with
practical, real-world examples.

Examples included:
1. Basic tempo conversion
2. Genre-aware conversion (jazz, EDM, funk)
3. Double-time and half-time feels
4. Compatibility checking
5. Multiple conversion strategies
6. Custom genre definitions
7. Analysis and reporting
8. Integration with other modules

Author: Agent 6 - Tempo Conversion & Style Adaptation
Date: 2025-11-19
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from transformation.tempo_converter import (
        TempoConverter,
        TempoConversionParams,
        ConversionStrategy,
        TempoFeelType,
        convert_midi_tempo,
        analyze_tempo_compatibility,
        GENRE_TEMPO_CHARACTERISTICS
    )
    imports_successful = True
except ImportError as e:
    print(f"Warning: Could not import tempo_converter: {e}")
    imports_successful = False

# Try to import mido for MIDI creation
try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    mido_available = True
except ImportError:
    print("Warning: mido not available - some examples will be skipped")
    mido_available = False


def create_example_midi(tempo: float = 120, filename: str = "example.mid"):
    """
    Create a simple example MIDI file for demonstration.

    Args:
        tempo: Tempo in BPM
        filename: Output filename

    Returns:
        MidiFile object
    """
    if not mido_available:
        print("Cannot create MIDI: mido not available")
        return None

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

    # Set time signature
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))

    # Add track name
    track.append(MetaMessage('track_name', name='Example Track', time=0))

    # Add a simple melody (C major scale)
    ticks_per_beat = mid.ticks_per_beat
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
    durations = [1, 1, 1, 1, 1, 1, 1, 2]  # In beats

    current_time = 0
    for i, (pitch, duration) in enumerate(zip(pitches, durations)):
        # Note on
        track.append(Message('note_on', note=pitch, velocity=80,
                           time=0 if i == 0 else ticks_per_beat))

        # Note off
        track.append(Message('note_off', note=pitch, velocity=0,
                           time=int(ticks_per_beat * duration * 0.8)))  # 80% of duration

    # End of track
    track.append(MetaMessage('end_of_track', time=ticks_per_beat))

    # Save
    mid.save(filename)
    print(f"✓ Created example MIDI: {filename} at {tempo} BPM")

    return mid


# ==============================================================================
# EXAMPLE 1: BASIC TEMPO CONVERSION
# ==============================================================================

def example_01_basic_conversion():
    """
    Example 1: Basic tempo conversion

    Demonstrates the simplest way to convert tempo.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BASIC TEMPO CONVERSION")
    print("=" * 70)

    if not mido_available or not imports_successful:
        print("Skipping - required modules not available")
        return

    # Create example MIDI at 120 BPM
    midi_obj = create_example_midi(120, "basic_120.mid")

    # Load and convert
    print("\n1. Loading MIDI file...")
    converter = TempoConverter(midi_object=midi_obj)
    print(f"   Current tempo: {converter.current_tempo} BPM")

    # Convert to 160 BPM
    print("\n2. Converting to 160 BPM...")
    converter.convert_tempo(160)
    print(f"   New tempo: {converter.current_tempo} BPM")

    # Save
    print("\n3. Saving result...")
    converter.save("basic_160.mid")

    # Show analysis
    print("\n4. Analysis:")
    print(converter.get_analysis_report())


# ==============================================================================
# EXAMPLE 2: GENRE-AWARE CONVERSION - JAZZ
# ==============================================================================

def example_02_jazz_ballad_to_uptempo():
    """
    Example 2: Convert jazz ballad to up-tempo jazz

    Demonstrates genre-aware conversion with swing preservation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: JAZZ BALLAD TO UP-TEMPO")
    print("=" * 70)

    if not mido_available or not imports_successful:
        print("Skipping - required modules not available")
        return

    # Create jazz ballad at 70 BPM
    print("\n1. Creating jazz ballad at 70 BPM...")
    midi_obj = create_example_midi(70, "jazz_ballad_70.mid")

    converter = TempoConverter(midi_object=midi_obj)
    print(f"   Current tempo: {converter.current_tempo} BPM (ballad)")

    # Convert to up-tempo jazz (200 BPM)
    print("\n2. Converting to up-tempo jazz (200 BPM)...")
    params = TempoConversionParams(
        target_tempo=200,
        genre='jazz_uptempo',
        strategy=ConversionStrategy.GENRE_AWARE,
        preserve_swing=True,
        adjust_articulation=True
    )

    converter.convert_tempo_with_params(params)
    print(f"   New tempo: {converter.current_tempo} BPM (up-tempo)")

    # Save
    converter.save("jazz_uptempo_200.mid")

    # Show what happened
    print("\n3. Conversion details:")
    print(f"   - Tempo ratio: {converter.analysis.tempo_ratio:.2f}x")
    print(f"   - Feel change: {converter.analysis.feel_change}")
    print(f"   - Notes adjusted: {converter.analysis.notes_adjusted}")

    if converter.analysis.articulation_changes:
        print("\n4. Articulation changes:")
        for style, factor in converter.analysis.articulation_changes.items():
            print(f"   - {style}: {factor:.2f}x")


# ==============================================================================
# EXAMPLE 3: EDM HALF-TIME FEEL
# ==============================================================================

def example_03_edm_half_time():
    """
    Example 3: Create EDM half-time feel

    Demonstrates half-time conversion for dubstep/EDM.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: EDM HALF-TIME FEEL")
    print("=" * 70)

    if not mido_available or not imports_successful:
        print("Skipping - required modules not available")
        return

    # Create EDM track at 140 BPM
    print("\n1. Creating EDM track at 140 BPM...")
    midi_obj = create_example_midi(140, "edm_140.mid")

    converter = TempoConverter(midi_object=midi_obj)
    print(f"   Current tempo: {converter.current_tempo} BPM (normal feel)")

    # Convert to half-time feel
    print("\n2. Converting to half-time feel...")
    converter.convert_to_half_time(genre='dubstep')
    print(f"   New tempo: {converter.current_tempo} BPM (half-time feel)")
    print("   → Feels like 70 BPM with more subdivision")

    # Save
    converter.save("edm_halftime_70.mid")

    print("\n3. Result:")
    print("   The track now has a half-time feel while maintaining")
    print("   the energy of the original 140 BPM.")


# ==============================================================================
# EXAMPLE 4: COMPATIBILITY CHECKING
# ==============================================================================

def example_04_check_compatibility():
    """
    Example 4: Check tempo compatibility before converting

    Demonstrates using the compatibility analyzer.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: COMPATIBILITY CHECKING")
    print("=" * 70)

    if not imports_successful:
        print("Skipping - tempo_converter not available")
        return

    print("\n1. Checking various tempo conversions:\n")

    # Test 1: Normal change (good)
    analysis = analyze_tempo_compatibility(120, 140, 'jazz_medium')
    print(f"   120 → 140 BPM (Jazz):")
    print(f"     Ratio: {analysis['ratio']:.2f}x")
    print(f"     Category: {analysis['ratio_category']}")
    print(f"     Recommended: {'✓ Yes' if analysis['recommended'] else '✗ No'}")

    # Test 2: Double-time (interesting)
    analysis = analyze_tempo_compatibility(90, 180, 'jazz_medium')
    print(f"\n   90 → 180 BPM (Jazz):")
    print(f"     Ratio: {analysis['ratio']:.2f}x")
    print(f"     Feel change: {analysis['feel_change']}")
    print(f"     Recommended: {'✓ Yes' if analysis['recommended'] else '✗ No'}")

    # Test 3: Out of range (warning)
    analysis = analyze_tempo_compatibility(100, 200, 'funk')
    print(f"\n   100 → 200 BPM (Funk):")
    print(f"     Ratio: {analysis['ratio']:.2f}x")
    print(f"     Recommended: {'✓ Yes' if analysis['recommended'] else '✗ No'}")
    if analysis['warnings']:
        print(f"     Warnings:")
        for warning in analysis['warnings']:
            print(f"       ⚠ {warning}")

    # Test 4: Extreme change (not recommended)
    analysis = analyze_tempo_compatibility(60, 240, None)
    print(f"\n   60 → 240 BPM:")
    print(f"     Ratio: {analysis['ratio']:.2f}x")
    print(f"     Category: {analysis['ratio_category']}")
    print(f"     Recommended: {'✓ Yes' if analysis['recommended'] else '✗ No'}")
    if analysis['suggestions']:
        print(f"     Suggestions:")
        for suggestion in analysis['suggestions']:
            print(f"       💡 {suggestion}")


# ==============================================================================
# EXAMPLE 5: CONVERSION STRATEGIES COMPARISON
# ==============================================================================

def example_05_strategy_comparison():
    """
    Example 5: Compare different conversion strategies

    Shows the difference between SIMPLE, SMART, and GENRE_AWARE strategies.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: CONVERSION STRATEGIES COMPARISON")
    print("=" * 70)

    if not mido_available or not imports_successful:
        print("Skipping - required modules not available")
        return

    # Create test MIDI
    midi_obj = create_example_midi(100, "strategy_test_100.mid")

    print("\n1. Converting 100 → 180 BPM with different strategies:\n")

    # Strategy 1: SIMPLE
    print("   SIMPLE Strategy:")
    converter1 = TempoConverter(midi_object=midi_obj)
    converter1.convert_tempo(180, strategy=ConversionStrategy.SIMPLE)
    converter1.save("strategy_simple_180.mid")
    print(f"     - Notes adjusted: {converter1.analysis.notes_adjusted}")
    print(f"     - Articulation changes: {len(converter1.analysis.articulation_changes)}")

    # Strategy 2: SMART
    print("\n   SMART Strategy:")
    converter2 = TempoConverter(midi_object=midi_obj)
    converter2.convert_tempo(180, strategy=ConversionStrategy.SMART)
    converter2.save("strategy_smart_180.mid")
    print(f"     - Notes adjusted: {converter2.analysis.notes_adjusted}")
    print(f"     - Articulation changes: {len(converter2.analysis.articulation_changes)}")

    # Strategy 3: GENRE_AWARE
    print("\n   GENRE_AWARE Strategy (Jazz):")
    converter3 = TempoConverter(midi_object=midi_obj)
    params = TempoConversionParams(
        target_tempo=180,
        genre='jazz_medium',
        strategy=ConversionStrategy.GENRE_AWARE
    )
    converter3.convert_tempo_with_params(params)
    converter3.save("strategy_genre_180.mid")
    print(f"     - Notes adjusted: {converter3.analysis.notes_adjusted}")
    print(f"     - Articulation changes: {len(converter3.analysis.articulation_changes)}")
    print(f"     - Genre: {converter3.analysis.genre}")

    print("\n2. Comparison:")
    print("   SIMPLE:       Fastest, minimal adjustments")
    print("   SMART:        Intelligent articulation adjustments")
    print("   GENRE_AWARE:  Full genre-specific optimization")


# ==============================================================================
# EXAMPLE 6: DOUBLE-TIME CONVERSION
# ==============================================================================

def example_06_double_time_feel():
    """
    Example 6: Create double-time feel

    Demonstrates double-time conversion.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: DOUBLE-TIME FEEL")
    print("=" * 70)

    if not mido_available or not imports_successful:
        print("Skipping - required modules not available")
        return

    # Create medium tempo track
    print("\n1. Creating track at 90 BPM...")
    midi_obj = create_example_midi(90, "medium_90.mid")

    converter = TempoConverter(midi_object=midi_obj)
    print(f"   Current tempo: {converter.current_tempo} BPM")

    # Convert to double-time
    print("\n2. Converting to double-time feel...")
    converter.convert_to_double_time(genre='jazz_medium')
    print(f"   New tempo: {converter.current_tempo} BPM")

    # Save
    converter.save("double_time_180.mid")

    print("\n3. What changed:")
    print("   - Tempo doubled (90 → 180 BPM)")
    print("   - Articulations shortened for clarity")
    print("   - Energy increased")
    print("   - Maintains musical character")


# ==============================================================================
# EXAMPLE 7: GENRE TEMPO RANGES
# ==============================================================================

def example_07_genre_ranges():
    """
    Example 7: Explore genre tempo ranges

    Shows tempo characteristics for different genres.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: GENRE TEMPO RANGES")
    print("=" * 70)

    if not imports_successful:
        print("Skipping - tempo_converter not available")
        return

    print("\nGenre Tempo Characteristics:\n")

    genres_to_show = [
        'jazz_ballad', 'jazz_medium', 'jazz_uptempo',
        'funk', 'house', 'techno', 'dubstep',
        'bossa_nova', 'salsa', 'hip_hop'
    ]

    for genre in genres_to_show:
        if genre in GENRE_TEMPO_CHARACTERISTICS:
            chars = GENRE_TEMPO_CHARACTERISTICS[genre]
            opt_min, opt_max = chars['optimal_range']
            acc_min, acc_max = chars['acceptable_range']

            print(f"   {genre.upper().replace('_', ' ')}:")
            print(f"     Optimal:    {opt_min}-{opt_max} BPM")
            print(f"     Acceptable: {acc_min}-{acc_max} BPM")
            print(f"     Swing:      {chars['swing_factor']:.2f}")
            print(f"     Feel:       {[f.value for f in chars['feel_types']]}")
            print()


# ==============================================================================
# EXAMPLE 8: MULTIPLE CONVERSIONS
# ==============================================================================

def example_08_multiple_conversions():
    """
    Example 8: Multiple conversions and history tracking

    Shows conversion history and chaining.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: MULTIPLE CONVERSIONS & HISTORY")
    print("=" * 70)

    if not mido_available or not imports_successful:
        print("Skipping - required modules not available")
        return

    # Create starting MIDI
    print("\n1. Creating MIDI at 100 BPM...")
    midi_obj = create_example_midi(100, "multi_start_100.mid")

    converter = TempoConverter(midi_object=midi_obj)

    # Multiple conversions
    print("\n2. Performing multiple conversions:")
    tempos = [120, 140, 160, 140, 120]

    for i, tempo in enumerate(tempos, 1):
        converter.convert_tempo(tempo)
        print(f"   Step {i}: → {tempo} BPM")
        converter.save(f"multi_step{i}_{tempo}.mid")

    # Show history
    print("\n3. Conversion history:")
    for i, analysis in enumerate(converter.conversion_history, 1):
        print(f"   {i}. {analysis.source_tempo:.0f} → {analysis.target_tempo:.0f} BPM "
              f"(ratio: {analysis.tempo_ratio:.2f}x)")


# ==============================================================================
# EXAMPLE 9: CUSTOM PARAMETERS
# ==============================================================================

def example_09_custom_parameters():
    """
    Example 9: Using custom conversion parameters

    Demonstrates detailed parameter control.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 9: CUSTOM CONVERSION PARAMETERS")
    print("=" * 70)

    if not mido_available or not imports_successful:
        print("Skipping - required modules not available")
        return

    # Create MIDI
    midi_obj = create_example_midi(120, "custom_120.mid")
    converter = TempoConverter(midi_object=midi_obj)

    # Custom parameters
    print("\n1. Converting with custom parameters...")
    params = TempoConversionParams(
        target_tempo=180,
        source_tempo=120,  # Explicit source tempo
        genre='jazz_medium',
        strategy=ConversionStrategy.GENRE_AWARE,
        preserve_swing=True,
        adjust_articulation=True,
        adjust_subdivisions=True,
        force_conversion=False,
        maintain_phrase_structure=True,
        transition_smoothness=0.7  # Smooth transition
    )

    converter.convert_tempo_with_params(params)
    converter.save("custom_180.mid")

    print("\n2. Parameters used:")
    print(f"   Target tempo: {params.target_tempo} BPM")
    print(f"   Genre: {params.genre}")
    print(f"   Strategy: {params.strategy.value}")
    print(f"   Preserve swing: {params.preserve_swing}")
    print(f"   Adjust articulation: {params.adjust_articulation}")
    print(f"   Maintain phrases: {params.maintain_phrase_structure}")


# ==============================================================================
# EXAMPLE 10: GRADUAL TEMPO CHANGES
# ==============================================================================

def example_10_gradual_changes():
    """
    Example 10: Create gradual tempo changes

    Shows how to create smooth tempo transitions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 10: GRADUAL TEMPO CHANGES")
    print("=" * 70)

    if not mido_available or not imports_successful:
        print("Skipping - required modules not available")
        return

    print("\n1. Creating series with gradual tempo increase:")
    print("   60 → 80 → 100 → 120 → 140 → 160 BPM")

    midi_obj = create_example_midi(60, "gradual_start_60.mid")
    converter = TempoConverter(midi_object=midi_obj)

    # Gradual increase
    tempos = [60, 80, 100, 120, 140, 160]
    print("\n2. Conversions:")

    for i, tempo in enumerate(tempos):
        if tempo != 60:  # Skip first (already at 60)
            converter.convert_tempo(tempo)
        converter.save(f"gradual_{tempo}.mid")
        print(f"   ✓ Created gradual_{tempo}.mid")

    print("\n3. Result:")
    print("   Created 6 files showing smooth tempo progression")


# ==============================================================================
# MAIN DEMONSTRATION
# ==============================================================================

def run_all_examples():
    """Run all examples in sequence."""
    print("=" * 70)
    print("TEMPO CONVERSION SYSTEM - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    print("\nThis demonstration will create multiple MIDI files showing")
    print("different tempo conversion techniques.\n")

    if not imports_successful:
        print("ERROR: Could not import tempo_converter module")
        print("Please ensure the module is in the correct location.")
        return

    if not mido_available:
        print("WARNING: mido not available - some examples will be skipped")
        print("Install with: pip install mido\n")

    # Run examples
    examples = [
        ("Basic Conversion", example_01_basic_conversion),
        ("Jazz Ballad to Up-tempo", example_02_jazz_ballad_to_uptempo),
        ("EDM Half-Time Feel", example_03_edm_half_time),
        ("Compatibility Checking", example_04_check_compatibility),
        ("Strategy Comparison", example_05_strategy_comparison),
        ("Double-Time Feel", example_06_double_time_feel),
        ("Genre Tempo Ranges", example_07_genre_ranges),
        ("Multiple Conversions", example_08_multiple_conversions),
        ("Custom Parameters", example_09_custom_parameters),
        ("Gradual Changes", example_10_gradual_changes),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nCheck the current directory for generated MIDI files.")
    print("Each example created files demonstrating specific features.")


if __name__ == "__main__":
    run_all_examples()
