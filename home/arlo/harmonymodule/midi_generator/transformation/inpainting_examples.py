#!/usr/bin/env python3
"""
MIDI Inpainting Engine - Comprehensive Examples

This module provides practical examples demonstrating all features
of the Inpainting Engine:

1. Basic Reharmonization
2. Jazz Reharmonization
3. Genre Morphing
4. Melody Preservation
5. Section Variation
6. Chromatic Reharmonization
7. Style Transitions
8. Advanced Voice Leading
9. Multi-Track Inpainting
10. Creative Applications

Author: Agent 4 - Inpainting Engine
Date: 2025
"""

import sys
from pathlib import Path
import tempfile
import os
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from transformation.inpainting_engine import (
    InpaintingEngine,
    ChordSubstitutionEngine,
    StyleTransitionBlender,
    MelodyPreserver
)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_sample_midi(filename: str, chord_progression: list = None,
                      measures: int = 16, tempo: int = 120):
    """
    Create a sample MIDI file for demonstration

    Args:
        filename: Output MIDI file path
        chord_progression: List of chord symbols (optional)
        measures: Number of measures
        tempo: Tempo in BPM
    """
    if chord_progression is None:
        # Default: ii-V-I in C major
        chord_progression = ['Dm7', 'G7', 'Cmaj7', 'Cmaj7'] * (measures // 4)

    mid = MidiFile(ticks_per_beat=480)

    # Track 0: Meta track
    meta_track = MidiTrack()
    meta_track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    meta_track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    meta_track.append(MetaMessage('end_of_track', time=0))
    mid.tracks.append(meta_track)

    # Track 1: Melody
    melody_track = MidiTrack()
    melody_notes = [60, 62, 64, 65, 67, 69, 71, 72, 72, 71, 69, 67, 65, 64, 62, 60]

    for i in range(measures * 4):  # 4 quarter notes per measure
        note = melody_notes[i % len(melody_notes)]
        melody_track.append(Message('note_on', note=note, velocity=80, time=0))
        melody_track.append(Message('note_off', note=note, velocity=0, time=480))

    melody_track.append(MetaMessage('end_of_track', time=0))
    mid.tracks.append(melody_track)

    # Track 2: Chords (quarter note chords)
    chord_track = MidiTrack()

    # Simple chord voicings
    chord_voicings = {
        'Cmaj7': [60, 64, 67, 71],
        'Dm7': [62, 65, 69, 72],
        'Em7': [64, 67, 71, 74],
        'Fmaj7': [65, 69, 72, 76],
        'G7': [67, 71, 74, 77],
        'Am7': [69, 72, 76, 79],
        'Dm9': [62, 65, 69, 72, 76],
        'G7#9': [67, 71, 74, 77, 81],
        'Cmaj9': [60, 64, 67, 71, 74],
    }

    for i in range(measures):
        chord_symbol = chord_progression[i % len(chord_progression)]
        voicing = chord_voicings.get(chord_symbol, [60, 64, 67])

        # Play chord for whole measure (whole note)
        for note in voicing:
            chord_track.append(Message('note_on', note=note, velocity=60, time=0))

        # Note offs at end of measure
        for j, note in enumerate(voicing):
            time = 1920 if j == len(voicing) - 1 else 0
            chord_track.append(Message('note_off', note=note, velocity=0, time=time))

    chord_track.append(MetaMessage('end_of_track', time=0))
    mid.tracks.append(chord_track)

    mid.save(filename)
    print(f"Created sample MIDI: {filename}")
    return filename


# ==============================================================================
# EXAMPLE 1: Basic Reharmonization
# ==============================================================================

def example_01_basic_reharmonization():
    """
    Example 1: Basic Reharmonization

    Reharmonize measures 5-8 of a simple ii-V-I progression
    with more colorful jazz chords.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Reharmonization")
    print("=" * 70)

    # Create sample MIDI
    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, "example_01_input.mid")
    output_file = os.path.join(temp_dir, "example_01_output.mid")

    create_sample_midi(input_file, measures=16)

    # Load with inpainting engine
    engine = InpaintingEngine(input_file)
    analysis = engine.analyze()

    print(f"\nOriginal file:")
    print(f"  - Measures: {analysis['num_measures']}")
    print(f"  - Tempo: {analysis['tempo']} BPM")
    print(f"  - Key: {analysis['key']}")

    # Define new chord progression for measures 5-8
    original_chords = ['Dm7', 'G7', 'Cmaj7', 'Cmaj7']
    new_chords = ['Dm9', 'Db7#11', 'Cmaj9#11', 'A7#9']  # More colorful

    print(f"\nReharmonizing measures 5-8:")
    print(f"  Original: {original_chords}")
    print(f"  New:      {new_chords}")

    # Reharmonize
    regenerated = engine.inpaint_measures(
        track_numbers=[1, 2],  # Melody and chords
        start_measure=5,
        end_measure=8,
        new_chords=new_chords
    )

    print(f"\nRegenerated {len(regenerated)} tracks")
    for track_num, notes in regenerated.items():
        print(f"  Track {track_num}: {len(notes)} notes")

    # Export
    engine.export(output_file)
    print(f"\nExported to: {output_file}")

    # Cleanup
    print("\n✓ Example completed successfully")


# ==============================================================================
# EXAMPLE 2: Jazz Reharmonization with Substitutions
# ==============================================================================

def example_02_jazz_reharmonization():
    """
    Example 2: Jazz Reharmonization

    Use ChordSubstitutionEngine to create jazz reharmonization
    with tritone subs, secondary dominants, and extended harmony.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Jazz Reharmonization with Substitutions")
    print("=" * 70)

    # Original progression
    original = ['Dm7', 'G7', 'Cmaj7', 'Cmaj7',
                'Am7', 'D7', 'Dm7', 'G7']

    print(f"\nOriginal progression:")
    for i, chord in enumerate(original, 1):
        print(f"  {i}. {chord}")

    # Apply jazz reharmonization
    print(f"\nApplying jazz reharmonization techniques:")

    # Technique 1: Add extensions
    print("\n1. Adding extensions:")
    extended = [
        ChordSubstitutionEngine.extended_harmony(chord, [9])
        if '7' in chord else chord
        for chord in original
    ]
    for i, (orig, ext) in enumerate(zip(original, extended), 1):
        if orig != ext:
            print(f"  {orig} → {ext}")

    # Technique 2: Tritone substitution
    print("\n2. Tritone substitution on dominant chords:")
    tritone_sub = ChordSubstitutionEngine.tritone_substitute(original[1])
    print(f"  {original[1]} → {tritone_sub}")

    # Technique 3: Secondary dominants
    print("\n3. Adding secondary dominants:")
    sec_dom, target = ChordSubstitutionEngine.secondary_dominant(original[0], 'C')
    print(f"  Before {target}: insert {sec_dom}")

    # Technique 4: Full reharmonization
    print("\n4. Complete jazz reharmonization:")
    jazzed = ChordSubstitutionEngine.reharmonize(original, style='jazz', key='C')

    for i, (orig, new) in enumerate(zip(original, jazzed), 1):
        marker = "←" if orig != new else ""
        print(f"  {i}. {orig:10s} → {new:15s} {marker}")

    # Create MIDI and apply
    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, "example_02_input.mid")
    output_file = os.path.join(temp_dir, "example_02_output.mid")

    create_sample_midi(input_file, chord_progression=original, measures=8)

    engine = InpaintingEngine(input_file)
    engine.analyze()

    engine.inpaint_measures(
        track_numbers=[1, 2],
        start_measure=1,
        end_measure=8,
        new_chords=jazzed
    )

    engine.export(output_file)

    print(f"\n✓ Exported jazzed version to: {output_file}")


# ==============================================================================
# EXAMPLE 3: Genre Morphing
# ==============================================================================

def example_03_genre_morphing():
    """
    Example 3: Genre Morphing

    Smoothly transition from jazz to funk over 4 measures
    using StyleTransitionBlender.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Genre Morphing (Jazz → Funk)")
    print("=" * 70)

    # Create blended transition
    print("\nCreating 4-measure transition from jazz to funk:")

    transition = StyleTransitionBlender.blend_styles(
        style_a='jazz',
        style_b='funk',
        blend_measures=4,
        blend_type='linear'
    )

    print("\nBlend progression:")
    for i, params in enumerate(transition, 1):
        jazz_pct = params['weight_a'] * 100
        funk_pct = params['weight_b'] * 100
        print(f"  Measure {i}: {jazz_pct:.0f}% jazz, {funk_pct:.0f}% funk")
        print(f"    - Tempo: {params['tempo']} BPM")
        print(f"    - Swing: {params['swing_factor']:.2f}")
        print(f"    - Syncopation: {params['syncopation']:.2f}")

    # Apply to MIDI
    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, "example_03_input.mid")
    output_file = os.path.join(temp_dir, "example_03_output.mid")

    create_sample_midi(input_file, measures=12)

    engine = InpaintingEngine(input_file)
    engine.analyze()

    # Measures 1-4: Jazz
    print("\nMeasures 1-4: Jazz style")

    # Measures 5-8: Transition (use funk genre as approximation)
    print("Measures 5-8: Transitioning to funk")
    engine.inpaint_measures(
        track_numbers=[1, 2],
        start_measure=5,
        end_measure=8,
        new_genre='funk'
    )

    # Measures 9-12: Funk
    print("Measures 9-12: Funk style")

    engine.export(output_file)
    print(f"\n✓ Exported morphed version to: {output_file}")


# ==============================================================================
# EXAMPLE 4: Melody Preservation
# ==============================================================================

def example_04_melody_preservation():
    """
    Example 4: Melody Preservation

    Reharmonize a progression while preserving the original melody.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Melody Preservation")
    print("=" * 70)

    from analysis.midi_analyzer import NoteEvent

    # Create original melody
    original_melody = [
        NoteEvent(0.0, 0.5, 0, 480, 60, 80, 0, 1),      # C
        NoteEvent(0.5, 0.5, 480, 480, 64, 80, 0, 1),    # E
        NoteEvent(1.0, 0.5, 960, 480, 67, 80, 0, 1),    # G
        NoteEvent(1.5, 0.5, 1440, 480, 71, 80, 0, 1),   # B
    ]

    original_chords = ['Cmaj7', 'Cmaj7']
    new_chords = ['Am7', 'Dm7']

    print("\nOriginal melody notes:")
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for note in original_melody:
        name = note_names[note.pitch % 12]
        octave = note.pitch // 12 - 1
        print(f"  {name}{octave} (MIDI {note.pitch})")

    print(f"\nOriginal chords: {original_chords}")
    print(f"New chords:      {new_chords}")

    # Preserve melody
    preserver = MelodyPreserver(original_melody, original_chords)

    print("\nAdjustment strategies:")

    # Strategy 1: Minimal adjustment
    print("\n1. Minimal adjustment (only fix clashes):")
    adjusted_minimal = preserver.reharmonize(new_chords, adjustment_strategy='minimal')
    for orig, adj in zip(original_melody, adjusted_minimal):
        change = "→ changed" if orig.pitch != adj.pitch else "✓ kept"
        print(f"  MIDI {orig.pitch} {change}")

    # Strategy 2: Prefer chord tones
    print("\n2. Prefer chord tones:")
    adjusted_chord_tones = preserver.reharmonize(new_chords, adjustment_strategy='chord_tones')
    for orig, adj in zip(original_melody, adjusted_chord_tones):
        change = f"→ {adj.pitch}" if orig.pitch != adj.pitch else "✓ same"
        print(f"  MIDI {orig.pitch} {change}")

    # Strategy 3: Chromatic (allow passing tones)
    print("\n3. Chromatic (allow passing tones):")
    adjusted_chromatic = preserver.reharmonize(new_chords, adjustment_strategy='chromatic')
    for orig, adj in zip(original_melody, adjusted_chromatic):
        change = f"→ {adj.pitch}" if orig.pitch != adj.pitch else "✓ same"
        print(f"  MIDI {orig.pitch} {change}")

    print("\n✓ Example completed")


# ==============================================================================
# EXAMPLE 5: Section Variation
# ==============================================================================

def example_05_section_variation():
    """
    Example 5: Section Variation

    Create variations of repeated sections (e.g., different verse harmonizations).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Section Variation")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, "example_05_input.mid")
    output_file = os.path.join(temp_dir, "example_05_output.mid")

    # Original progression (repeated for verse 1 and verse 2)
    original_progression = ['Dm7', 'G7', 'Cmaj7', 'Am7'] * 2

    create_sample_midi(input_file, chord_progression=original_progression, measures=8)

    engine = InpaintingEngine(input_file)
    engine.analyze()

    print("\nOriginal progression (verses 1 & 2):")
    print(f"  {original_progression[:4]}")

    # Variation 1: Add chromatic passing chords (verse 2)
    print("\nVerse 2 - Variation 1: Chromatic passing chords")
    chromatic = ChordSubstitutionEngine.reharmonize(
        original_progression[:4],
        style='chromatic'
    )
    print(f"  {chromatic}")

    engine.inpaint_measures(
        track_numbers=[2],  # Just the chord track
        start_measure=5,
        end_measure=8,
        new_chords=chromatic
    )

    # Variation 2: Jazz extensions (if there was a verse 3)
    print("\nHypothetical Verse 3 - Variation 2: Jazz extensions")
    jazzed = ChordSubstitutionEngine.reharmonize(
        original_progression[:4],
        style='jazz'
    )
    print(f"  {jazzed}")

    engine.export(output_file)
    print(f"\n✓ Exported with variation to: {output_file}")


# ==============================================================================
# EXAMPLE 6: Chromatic Reharmonization
# ==============================================================================

def example_06_chromatic_reharmonization():
    """
    Example 6: Chromatic Reharmonization

    Add chromatic passing chords and approach chords.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Chromatic Reharmonization")
    print("=" * 70)

    original = ['Cmaj7', 'Am7', 'Dm7', 'G7']

    print(f"\nOriginal (4 chords):")
    print(f"  {' - '.join(original)}")

    # Apply chromatic reharmonization
    chromatic = ChordSubstitutionEngine.reharmonize(original, style='chromatic')

    print(f"\nChromatic (with passing chords):")
    print(f"  {' - '.join(chromatic)}")
    print(f"\nAdded {len(chromatic) - len(original)} passing chords")

    # Show what was added
    print("\nChromatic approach technique:")
    print("  - Diminished 7th chords as passing chords")
    print("  - Half-step approach to target chords")
    print("  - Creates smooth chromatic voice leading")

    print("\n✓ Example completed")


# ==============================================================================
# EXAMPLE 7: Progressive Style Transition
# ==============================================================================

def example_07_progressive_transition():
    """
    Example 7: Progressive Style Transition

    Create smooth transitions using different blend curves.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Progressive Style Transition (Blend Curves)")
    print("=" * 70)

    print("\nComparing blend curve types:")

    measures = 6

    # Linear
    print(f"\n1. LINEAR blend ({measures} measures):")
    linear = StyleTransitionBlender.blend_styles(
        'jazz', 'edm', measures, 'linear'
    )
    for i, params in enumerate(linear, 1):
        print(f"  M{i}: Jazz {params['weight_a']*100:5.1f}%  EDM {params['weight_b']*100:5.1f}%")

    # Exponential
    print(f"\n2. EXPONENTIAL blend ({measures} measures):")
    exponential = StyleTransitionBlender.blend_styles(
        'jazz', 'edm', measures, 'exponential'
    )
    for i, params in enumerate(exponential, 1):
        print(f"  M{i}: Jazz {params['weight_a']*100:5.1f}%  EDM {params['weight_b']*100:5.1f}%")

    # S-curve
    print(f"\n3. S-CURVE blend ({measures} measures):")
    scurve = StyleTransitionBlender.blend_styles(
        'jazz', 'edm', measures, 's-curve'
    )
    for i, params in enumerate(scurve, 1):
        print(f"  M{i}: Jazz {params['weight_a']*100:5.1f}%  EDM {params['weight_b']*100:5.1f}%")

    print("\nCharacteristics:")
    print("  - Linear: Constant rate of change")
    print("  - Exponential: Slow start, fast finish")
    print("  - S-curve: Slow start, fast middle, slow finish (most musical)")

    print("\n✓ Example completed")


# ==============================================================================
# EXAMPLE 8: Multi-Track Inpainting
# ==============================================================================

def example_08_multi_track_inpainting():
    """
    Example 8: Multi-Track Inpainting

    Regenerate multiple tracks with coordinated changes.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Multi-Track Inpainting")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, "example_08_input.mid")
    output_file = os.path.join(temp_dir, "example_08_output.mid")

    create_sample_midi(input_file, measures=8)

    engine = InpaintingEngine(input_file)
    engine.analyze()

    print("\nRegenerating multiple tracks:")
    print("  - Track 1: Melody (preserve)")
    print("  - Track 2: Chords (regenerate)")

    new_chords = ['Fmaj9', 'Bm7b5', 'E7alt', 'Am9',
                  'Dm9', 'G7#9', 'Cmaj9#11', 'Cmaj9#11']

    # Preserve melody
    print("\nStep 1: Preserve melody")
    engine.inpaint_measures(
        track_numbers=[1],
        start_measure=1,
        end_measure=8,
        new_chords=new_chords,
        preserve_melody=True
    )

    # Regenerate harmony
    print("Step 2: Regenerate harmony")
    engine.inpaint_measures(
        track_numbers=[2],
        start_measure=1,
        end_measure=8,
        new_chords=new_chords,
        preserve_rhythm=False
    )

    engine.export(output_file)
    print(f"\n✓ Exported multi-track inpainting to: {output_file}")


# ==============================================================================
# EXAMPLE 9: Boundary Smoothing Demonstration
# ==============================================================================

def example_09_boundary_smoothing():
    """
    Example 9: Boundary Smoothing

    Demonstrate how the engine creates smooth transitions at boundaries.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Boundary Smoothing")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp()
    input_file = os.path.join(temp_dir, "example_09_input.mid")

    create_sample_midi(input_file, measures=12)

    engine = InpaintingEngine(input_file)
    engine.analyze()

    print("\nAnalyzing boundaries for inpaint region (measures 5-8):")

    # Extract entry context
    entry = engine._extract_entry_context([1], measure=4)
    print(f"\nEntry context (end of measure 4):")
    print(f"  - Last pitches: {entry[1].last_pitches}")
    print(f"  - Rhythm density: {entry[1].rhythm_density:.2f} notes/beat")
    print(f"  - Average velocity: {entry[1].average_velocity}")

    # Extract exit context
    exit_ctx = engine._extract_exit_context([1], measure=9)
    print(f"\nExit context (start of measure 9):")
    print(f"  - First pitches: {exit_ctx[1].first_pitches}")
    print(f"  - Rhythm density: {exit_ctx[1].rhythm_density:.2f} notes/beat")
    print(f"  - Average velocity: {exit_ctx[1].average_velocity}")

    print("\nBoundary smoothing techniques:")
    print("  1. Voice leading: First note approaches entry pitch by step")
    print("  2. Voice leading: Last note approaches exit pitch by step")
    print("  3. Rhythm: Match rhythm density at boundaries")
    print("  4. Dynamics: Smooth velocity transitions")

    print("\n✓ Example completed")


# ==============================================================================
# EXAMPLE 10: Creative Application - Modal Interchange
# ==============================================================================

def example_10_modal_interchange():
    """
    Example 10: Creative Application

    Use modal interchange to borrow chords from parallel modes.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 10: Modal Interchange")
    print("=" * 70)

    print("\nC Major borrowing from C Minor (parallel minor):")
    print("\nCommon borrowed chords:")
    print("  - bIII (Eb major) - from C minor")
    print("  - bVI (Ab major) - from C minor")
    print("  - bVII (Bb major) - from C minor/mixolydian")
    print("  - iv (Fm) - from C minor")

    original = ['Cmaj7', 'Fmaj7', 'G7', 'Cmaj7']
    print(f"\nOriginal (C major): {original}")

    # Manually create modal interchange
    modal_interchange = ['Cmaj7', 'Fm7', 'Bb7', 'Cmaj7']  # Borrowed iv and bVII
    print(f"With borrowed chords: {modal_interchange}")

    print("\nEffect:")
    print("  - Darker, more colorful harmony")
    print("  - Common in Beatles, jazz, film music")
    print("  - Creates emotional depth")

    # Show romantic style reharmonization (uses modal interchange)
    romantic = ChordSubstitutionEngine.reharmonize(original, style='romantic')
    print(f"\nRomantic style: {romantic}")

    print("\n✓ Example completed")


# ==============================================================================
# MAIN - Run All Examples
# ==============================================================================

def run_all_examples():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("MIDI INPAINTING ENGINE - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates all features of the Inpainting Engine.")
    print("Each example is self-contained and creates temporary MIDI files.")

    examples = [
        ("Basic Reharmonization", example_01_basic_reharmonization),
        ("Jazz Reharmonization", example_02_jazz_reharmonization),
        ("Genre Morphing", example_03_genre_morphing),
        ("Melody Preservation", example_04_melody_preservation),
        ("Section Variation", example_05_section_variation),
        ("Chromatic Reharmonization", example_06_chromatic_reharmonization),
        ("Progressive Transition", example_07_progressive_transition),
        ("Multi-Track Inpainting", example_08_multi_track_inpainting),
        ("Boundary Smoothing", example_09_boundary_smoothing),
        ("Modal Interchange", example_10_modal_interchange),
    ]

    print(f"\nTotal examples: {len(examples)}\n")

    for i, (name, func) in enumerate(examples, 1):
        try:
            print(f"\n[{i}/{len(examples)}] Running: {name}")
            func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


def run_single_example(example_num: int):
    """Run a single example by number"""
    examples = {
        1: example_01_basic_reharmonization,
        2: example_02_jazz_reharmonization,
        3: example_03_genre_morphing,
        4: example_04_melody_preservation,
        5: example_05_section_variation,
        6: example_06_chromatic_reharmonization,
        7: example_07_progressive_transition,
        8: example_08_multi_track_inpainting,
        9: example_09_boundary_smoothing,
        10: example_10_modal_interchange,
    }

    if example_num in examples:
        examples[example_num]()
    else:
        print(f"Example {example_num} not found. Choose 1-10.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run specific example
        try:
            example_num = int(sys.argv[1])
            run_single_example(example_num)
        except ValueError:
            print("Usage: python inpainting_examples.py [example_number]")
            print("       python inpainting_examples.py  (runs all examples)")
    else:
        # Run all examples
        run_all_examples()
