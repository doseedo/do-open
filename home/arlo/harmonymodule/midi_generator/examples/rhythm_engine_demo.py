"""
Comprehensive Demo of Advanced Rhythm Engine

This example demonstrates all major features of the rhythm engine and groove library:
1. Basic rhythm pattern generation
2. Polyrhythm generation
3. Euclidean rhythms and African timelines
4. Famous drum grooves
5. Humanization and groove application
6. Rhythm transformations
7. Genre-specific timing profiles
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.rhythm_engine import (
    RhythmEngine,
    RhythmNote,
    PolyrhythmSpec,
    TimingStyle,
    GrooveTemplate
)
from algorithms.groove_library import GrooveLibrary
from midi.midi_constants import GM_DRUM_MAP, PPQN_HIGH_RES


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_rhythm_info(notes, name="Pattern"):
    """Print rhythm pattern information"""
    if not notes:
        print(f"{name}: No notes")
        return

    print(f"{name}:")
    print(f"  Notes: {len(notes)}")
    print(f"  Duration: {max(n.tick for n in notes)} ticks")
    print(f"  Velocity range: {min(n.velocity for n in notes)}-{max(n.velocity for n in notes)}")
    print(f"  First 5 notes:")
    for note in notes[:5]:
        pitch_name = [k for k, v in GM_DRUM_MAP.items() if v == note.pitch]
        pitch_str = pitch_name[0] if pitch_name else f"Pitch {note.pitch}"
        print(f"    Tick {note.tick:5d}: {pitch_str:20s} vel={note.velocity:3d}")


def demo_basic_patterns():
    """Demo 1: Basic rhythm pattern generation"""
    print_section("1. Basic Rhythm Patterns")

    engine = RhythmEngine(ppqn=PPQN_HIGH_RES)

    # Create a simple 4-on-the-floor kick pattern
    print("\nCreating 4-on-the-floor kick pattern...")
    kick_pattern = []
    for beat in range(4):
        tick = beat * PPQN_HIGH_RES * 4  # Each beat is 4x ppqn for quarter notes
        kick_pattern.append(RhythmNote(
            tick=tick,
            duration=PPQN_HIGH_RES * 2,
            velocity=90,
            pitch=GM_DRUM_MAP['BASS_DRUM_1']
        ))

    print_rhythm_info(kick_pattern, "4-on-the-floor")

    # Create hi-hat on 8th notes
    print("\nCreating hi-hat 8th note pattern...")
    hihat_pattern = []
    for eighth in range(8):
        tick = eighth * PPQN_HIGH_RES * 2  # 8th notes
        hihat_pattern.append(RhythmNote(
            tick=tick,
            duration=PPQN_HIGH_RES,
            velocity=70,
            pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
        ))

    print_rhythm_info(hihat_pattern, "Hi-hat 8ths")


def demo_polyrhythms():
    """Demo 2: Polyrhythm generation"""
    print_section("2. Polyrhythm Generation")

    engine = RhythmEngine(ppqn=PPQN_HIGH_RES)

    # 3 against 4
    print("\nGenerating 3 against 4 polyrhythm...")
    spec_3v4 = PolyrhythmSpec(
        ratio_a=3,
        ratio_b=4,
        beats=4,
        velocity_a=95,
        velocity_b=75,
        pitch_a=GM_DRUM_MAP['COWBELL'],
        pitch_b=GM_DRUM_MAP['CLAVES']
    )

    rhythm_a, rhythm_b = engine.polyrhythm_generator.generate_polyrhythm(spec_3v4)
    print_rhythm_info(rhythm_a, "3-rhythm (cowbell)")
    print_rhythm_info(rhythm_b, "4-rhythm (claves)")

    # 5 against 7 (complex)
    print("\nGenerating 5 against 7 polyrhythm (complex)...")
    spec_5v7 = PolyrhythmSpec(
        ratio_a=5,
        ratio_b=7,
        beats=4,
        velocity_a=90,
        velocity_b=70,
        pitch_a=GM_DRUM_MAP['HIGH_TOM'],
        pitch_b=GM_DRUM_MAP['LOW_TOM']
    )

    rhythm_a, rhythm_b = engine.polyrhythm_generator.generate_polyrhythm(spec_5v7)
    print_rhythm_info(rhythm_a, "5-rhythm")
    print_rhythm_info(rhythm_b, "7-rhythm")


def demo_euclidean_rhythms():
    """Demo 3: Euclidean rhythms and African timelines"""
    print_section("3. Euclidean Rhythms and African Timelines")

    engine = RhythmEngine(ppqn=PPQN_HIGH_RES)

    # Euclidean rhythm: 5 hits in 8 steps
    print("\nGenerating Euclidean rhythm (5, 8)...")
    euclidean = engine.polyrhythm_generator.generate_euclidean_rhythm(
        hits=5,
        steps=8,
        velocity=85,
        pitch=GM_DRUM_MAP['BASS_DRUM_1'],
        duration_beats=4
    )
    print_rhythm_info(euclidean, "Euclidean (5,8)")

    # Euclidean rhythm: 7 hits in 12 steps (complex)
    print("\nGenerating Euclidean rhythm (7, 12)...")
    euclidean_complex = engine.polyrhythm_generator.generate_euclidean_rhythm(
        hits=7,
        steps=12,
        velocity=80,
        pitch=GM_DRUM_MAP['ACOUSTIC_SNARE'],
        duration_beats=4
    )
    print_rhythm_info(euclidean_complex, "Euclidean (7,12)")

    # African timeline patterns
    print("\nGenerating African timeline patterns...")

    timelines = ['son_clave', 'rumba_clave', 'gankogui', 'soukous']
    for timeline_name in timelines:
        timeline = engine.polyrhythm_generator.generate_african_timeline(
            pattern_name=timeline_name,
            duration_beats=4,
            velocity=90,
            pitch=GM_DRUM_MAP['COWBELL']
        )
        print_rhythm_info(timeline, f"Timeline: {timeline_name}")


def demo_famous_grooves():
    """Demo 4: Famous drum grooves"""
    print_section("4. Famous Drum Grooves")

    library = GrooveLibrary(ppqn=PPQN_HIGH_RES)

    print("\nAvailable grooves:")
    for groove_name in library.list_grooves():
        print(f"  - {groove_name}")

    # Load and display some famous grooves
    grooves_to_demo = ['purdie_shuffle', 'funky_drummer', 'afrobeat_pattern']

    for groove_name in grooves_to_demo:
        print(f"\nLoading {groove_name}...")
        groove = library.get_groove(groove_name)
        print_rhythm_info(groove, groove_name)

        # Show unique drums used
        unique_pitches = set(n.pitch for n in groove if n.pitch)
        drum_names = []
        for pitch in unique_pitches:
            names = [k for k, v in GM_DRUM_MAP.items() if v == pitch]
            if names:
                drum_names.append(names[0])
        print(f"  Drums used: {', '.join(drum_names)}")


def demo_humanization():
    """Demo 5: Humanization and feel"""
    print_section("5. Humanization and Feel")

    engine = RhythmEngine(ppqn=PPQN_HIGH_RES)

    # Create a mechanical pattern
    mechanical = []
    for i in range(16):
        mechanical.append(RhythmNote(
            tick=i * PPQN_HIGH_RES,
            duration=PPQN_HIGH_RES // 2,
            velocity=80,
            pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
        ))

    print("\nOriginal mechanical pattern:")
    print_rhythm_info(mechanical, "Mechanical")

    # Apply different timing styles
    styles = [
        TimingStyle.TIGHT,
        TimingStyle.LAID_BACK,
        TimingStyle.RUSHING,
        TimingStyle.HUMAN
    ]

    for style in styles:
        humanized = engine.humanizer.humanize_timing(mechanical, style=style)
        humanized = engine.humanizer.humanize_velocity(humanized, variation=0.2)
        print_rhythm_info(humanized, f"Humanized ({style.value})")

    # Drummer feel (ghost notes, flams)
    print("\nAdding drummer feel (ghost notes, flams)...")
    drummer_feel = engine.humanizer.add_drummer_feel(
        mechanical[:8],  # Just first 8 notes
        ghost_note_probability=0.3,
        flam_probability=0.2
    )
    print_rhythm_info(drummer_feel, "With drummer feel")


def demo_groove_templates():
    """Demo 6: Groove template extraction and application"""
    print_section("6. Groove Templates")

    engine = RhythmEngine(ppqn=PPQN_HIGH_RES)
    library = GrooveLibrary(ppqn=PPQN_HIGH_RES)

    # Extract groove from famous pattern
    print("\nExtracting groove from Funky Drummer...")
    funky_drummer = library.get_groove('funky_drummer')

    template = engine.groove_engine.extract_groove_from_notes(
        funky_drummer,
        grid_division=16,
        name='funky_drummer_groove',
        description='Extracted from Funky Drummer break'
    )

    print(f"Extracted template: {template.name}")
    print(f"  Grid division: {template.grid_division}")
    print(f"  Swing ratio: {template.swing_ratio:.3f}")
    print(f"  Timing offsets: {len(template.timing_offsets)} points")
    print(f"  Velocity curve: {len(template.velocity_curve)} points")

    # Apply to a different pattern
    print("\nApplying Funky Drummer groove to straight 16ths...")
    straight_pattern = []
    for i in range(16):
        straight_pattern.append(RhythmNote(
            tick=i * PPQN_HIGH_RES // 4,
            duration=PPQN_HIGH_RES // 8,
            velocity=70,
            pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
        ))

    grooved = engine.groove_engine.apply_groove(
        straight_pattern,
        template,
        intensity=0.75
    )

    print_rhythm_info(straight_pattern, "Original straight")
    print_rhythm_info(grooved, "With Funky Drummer groove")


def demo_transformations():
    """Demo 7: Rhythm transformations"""
    print_section("7. Rhythm Transformations")

    engine = RhythmEngine(ppqn=PPQN_HIGH_RES)

    # Create base pattern
    base_pattern = []
    for i in [0, 2, 3, 6]:  # Irregular pattern
        base_pattern.append(RhythmNote(
            tick=i * PPQN_HIGH_RES,
            duration=PPQN_HIGH_RES // 2,
            velocity=80,
            pitch=GM_DRUM_MAP['BASS_DRUM_1']
        ))

    print_rhythm_info(base_pattern, "Original pattern")

    # Augmentation (slower)
    print("\nAugmentation (2x slower)...")
    augmented = engine.transformer.augment(base_pattern, factor=2.0)
    print_rhythm_info(augmented, "Augmented")

    # Diminution (faster)
    print("\nDiminution (2x faster)...")
    diminished = engine.transformer.diminute(base_pattern, factor=2.0)
    print_rhythm_info(diminished, "Diminished")

    # Retrograde (reverse)
    print("\nRetrograde (reversed)...")
    retrograde = engine.transformer.retrograde(base_pattern)
    print_rhythm_info(retrograde, "Retrograde")

    # Swing conversion
    print("\nConverting straight to swing...")
    straight = []
    for i in range(8):
        straight.append(RhythmNote(
            tick=i * PPQN_HIGH_RES,
            duration=PPQN_HIGH_RES // 2,
            velocity=75,
            pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
        ))

    swing = engine.transformer.convert_swing(
        straight,
        from_ratio=0.5,   # Straight
        to_ratio=0.67,    # Triplet swing
        grid_division=8
    )

    print_rhythm_info(straight, "Straight 8ths")
    print_rhythm_info(swing, "Triplet swing")


def demo_genre_profiles():
    """Demo 8: Genre-specific timing profiles"""
    print_section("8. Genre Timing Profiles")

    library = GrooveLibrary(ppqn=PPQN_HIGH_RES)

    print("\nAvailable genre profiles:")
    for genre in library.list_genre_profiles():
        print(f"  - {genre}")

    # Compare different genres
    print("\nComparing genre characteristics:")

    genres_to_compare = ['jazz_bebop', 'funk', 'reggae', 'electronic_edm', 'metal']

    for genre_name in genres_to_compare:
        profile = library.get_genre_profile(genre_name)
        if profile:
            print(f"\n{profile.name}:")
            print(f"  Description: {profile.description}")
            print(f"  Timing deviation: {profile.avg_deviation_ms:.1f}±{profile.deviation_std_ms:.1f}ms")
            print(f"  Early/Late bias: {profile.early_late_bias:+.1f}ms")
            print(f"  Swing ratio: {profile.swing_ratio:.3f}")
            print(f"  Velocity variation: {profile.velocity_variation:.2f}")
            print(f"  Accent strength: {profile.accent_strength:.1f}x")


def demo_instrument_timing():
    """Demo 9: Instrument-specific timing characteristics"""
    print_section("9. Instrument Timing Characteristics")

    library = GrooveLibrary(ppqn=PPQN_HIGH_RES)

    print("\nInstrument timing profiles:")

    for instrument_name in library.list_instrument_timings():
        chars = library.get_instrument_timing(instrument_name)
        if chars:
            print(f"\n{chars.name}:")
            print(f"  Average offset: {chars.avg_offset_ms:+.1f}ms")
            print(f"  Attack time: {chars.attack_time_ms:.1f}ms")
            print(f"  Natural jitter: {chars.natural_jitter_ms:.1f}ms")
            print(f"  Velocity sensitivity: {chars.velocity_sensitivity:.2f}")


def demo_complete_workflow():
    """Demo 10: Complete workflow combining multiple features"""
    print_section("10. Complete Workflow Example")

    engine = RhythmEngine(ppqn=PPQN_HIGH_RES)
    library = GrooveLibrary(ppqn=PPQN_HIGH_RES)

    print("\nBuilding a complete drum pattern with multiple techniques:")

    # Step 1: Start with a famous groove
    print("\n1. Starting with Purdie Shuffle...")
    base_groove = library.get_groove('purdie_shuffle')

    # Step 2: Add a polyrhythmic cowbell
    print("2. Adding 3:2 polyrhythmic cowbell...")
    poly_spec = PolyrhythmSpec(
        ratio_a=3,
        ratio_b=2,
        beats=4,
        velocity_a=75,
        velocity_b=0,  # Don't use rhythm B
        pitch_a=GM_DRUM_MAP['COWBELL']
    )
    cowbell, _ = engine.polyrhythm_generator.generate_polyrhythm(poly_spec)

    # Step 3: Get genre timing profile
    print("3. Applying funk timing profile...")
    funk_profile = library.get_genre_profile('funk')
    print(f"   Using profile: {funk_profile.name}")

    # Step 4: Humanize everything
    print("4. Humanizing with laid-back feel...")
    humanized_groove = engine.humanizer.humanize_timing(
        base_groove,
        style=TimingStyle.LAID_BACK
    )
    humanized_groove = engine.humanizer.humanize_velocity(
        humanized_groove,
        variation=funk_profile.velocity_variation
    )

    humanized_cowbell = engine.humanizer.humanize_timing(
        cowbell,
        style=TimingStyle.LAID_BACK
    )

    # Combine
    complete_pattern = humanized_groove + humanized_cowbell
    complete_pattern = sorted(complete_pattern, key=lambda n: n.tick)

    print_rhythm_info(complete_pattern, "Complete pattern")

    print("\nPattern breakdown:")
    print(f"  Total notes: {len(complete_pattern)}")
    print(f"  From base groove: {len(humanized_groove)}")
    print(f"  From cowbell: {len(humanized_cowbell)}")

    # Show unique instruments
    unique_pitches = set(n.pitch for n in complete_pattern if n.pitch)
    drum_names = []
    for pitch in unique_pitches:
        names = [k for k, v in GM_DRUM_MAP.items() if v == pitch]
        if names:
            drum_names.append(names[0])
    print(f"  Instruments: {', '.join(drum_names)}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print(" " * 15 + "RHYTHM ENGINE - COMPREHENSIVE DEMO")
    print("=" * 70)

    try:
        demo_basic_patterns()
        demo_polyrhythms()
        demo_euclidean_rhythms()
        demo_famous_grooves()
        demo_humanization()
        demo_groove_templates()
        demo_transformations()
        demo_genre_profiles()
        demo_instrument_timing()
        demo_complete_workflow()

        print("\n" + "=" * 70)
        print(" " * 20 + "ALL DEMOS COMPLETED!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
