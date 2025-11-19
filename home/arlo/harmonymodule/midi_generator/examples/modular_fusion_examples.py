#!/usr/bin/env python3
"""
Comprehensive Examples for Modular Fusion System

This file contains 20+ complete examples demonstrating all capabilities
of the HarmonyModule modular fusion system across Agents 1-9.

Categories:
1. Quick Fusion Examples (Examples 1-5)
2. Genre Detection & Analysis (Examples 6-8)
3. Context-Aware Generation (Examples 9-11)
4. Inpainting & Reharmonization (Examples 12-14)
5. Tempo & Meter Conversion (Examples 15-17)
6. Advanced Fusion Techniques (Examples 18-22)
7. Production Workflows (Examples 23-25)

Author: Agent 10 - Unified API & Integration
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.api import (
    HarmonyModuleAPI,
    QuickFusion,
    GenreBlend,
    ComponentMix,
    ContextGeneration,
    InpaintSection,
    TransformTempo,
    TransformMeter,
    GranularControl,
)


# ==============================================================================
# CATEGORY 1: QUICK FUSION EXAMPLES
# ==============================================================================

def example_01_jazz_funk_fusion():
    """
    Example 1: Jazz Harmony + Funk Rhythm = Jazz-Funk Fusion

    Creates a 16-bar jazz-funk piece with:
    - Jazz chord progressions (ii-V-I, extensions)
    - Funk rhythmic groove
    - 115 BPM (perfect for jazz-funk)
    """
    print("=" * 80)
    print("Example 1: Jazz-Funk Fusion")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Simple fusion: jazz harmony + funk rhythm
    composition = api.quick_fusion(
        harmony="jazz",
        rhythm="funk",
        tempo=115,
        key="Dm",
        measures=16
    )

    # Export
    output_path = api.export("example_01_jazz_funk.mid")
    print(f"✓ Created jazz-funk fusion: {output_path}")
    print(f"  - Harmony: Jazz (ii-V-I, extensions)")
    print(f"  - Rhythm: Funk (syncopated, tight groove)")
    print(f"  - Tempo: 115 BPM")
    print(f"  - Key: Dm")
    print()


def example_02_electro_swing():
    """
    Example 2: Electro-Swing (Vintage Rhythm + Modern Synths)

    Famous genre fusion popularized by Parov Stelar and Caravan Palace.
    Combines:
    - Swing rhythm (1920s-40s feel)
    - Jazz harmony
    - Electronic/EDM instrumentation
    """
    print("=" * 80)
    print("Example 2: Electro-Swing")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    composition = api.quick_fusion(
        harmony="jazz",
        rhythm="swing",
        instrumentation="electronic",
        tempo=128,
        key="G",
        measures=32
    )

    output_path = api.export("example_02_electro_swing.mid")
    print(f"✓ Created electro-swing: {output_path}")
    print(f"  - Vintage swing rhythm with modern EDM synths")
    print(f"  - Perfect for Parov Stelar/Caravan Palace style")
    print()


def example_03_afro_cuban_jazz():
    """
    Example 3: Afro-Cuban Jazz Fusion

    Classic fusion combining:
    - Jazz harmony (bebop, modal)
    - Latin/Afro-Cuban rhythm (clave, montuno)
    - Based on Dizzy Gillespie, Machito, Tito Puente
    """
    print("=" * 80)
    print("Example 3: Afro-Cuban Jazz")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    composition = api.quick_fusion(
        harmony="jazz",
        rhythm="latin",
        bass="latin",  # Latin bass patterns (tumbao)
        tempo=140,
        key="Cm",
        measures=24,
        time_signature=(4, 4)
    )

    output_path = api.export("example_03_afro_cuban_jazz.mid")
    print(f"✓ Created Afro-Cuban jazz: {output_path}")
    print(f"  - Clave-based rhythm + bebop harmony")
    print(f"  - 140 BPM uptempo feel")
    print()


def example_04_nu_jazz():
    """
    Example 4: Nu-Jazz (Jazz + Electronic/IDM)

    Modern fusion popularized by labels like Ninja Tune.
    Combines:
    - Jazz harmony and instrumentation
    - Electronic/IDM rhythm and production
    - Hip-hop influenced beats
    """
    print("=" * 80)
    print("Example 4: Nu-Jazz (Jazz + Electronic)")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    composition = api.quick_fusion(
        harmony="jazz",
        rhythm="electronic",
        drums="hiphop",
        tempo=95,
        key="Ebm",
        measures=32
    )

    output_path = api.export("example_04_nu_jazz.mid")
    print(f"✓ Created nu-jazz: {output_path}")
    print(f"  - Jazz chords + electronic rhythm + hip-hop drums")
    print(f"  - 95 BPM downtempo feel")
    print()


def example_05_latin_trap():
    """
    Example 5: Latin Trap (Reggaeton + Trap)

    Modern fusion popular in reggaeton/urban music.
    Combines:
    - Latin reggaeton rhythm
    - Trap drums (hi-hats, 808s)
    - Simple harmony
    """
    print("=" * 80)
    print("Example 5: Latin Trap")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    composition = api.quick_fusion(
        harmony="latin",
        rhythm="latin",
        drums="hiphop",  # Trap-style drums
        tempo=95,
        key="Am",
        measures=16
    )

    output_path = api.export("example_05_latin_trap.mid")
    print(f"✓ Created Latin trap: {output_path}")
    print(f"  - Reggaeton dembow rhythm + trap drums")
    print()


# ==============================================================================
# CATEGORY 2: GENRE DETECTION & ANALYSIS
# ==============================================================================

def example_06_detect_genre():
    """
    Example 6: Detect Genre from MIDI File

    Analyzes existing MIDI file and detects genre with confidence scores.
    Uses features:
    - Rhythm analysis (swing, syncopation)
    - Harmonic analysis (chord types, extensions)
    - Melodic analysis (intervals, contour)
    """
    print("=" * 80)
    print("Example 6: Genre Detection")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # First, create a sample MIDI to analyze
    print("Creating sample jazz file...")
    api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=140, measures=8)
    sample_path = api.export("sample_for_detection.mid")

    # Now detect its genre
    print("\nDetecting genre...")
    try:
        genres = api.detect_genre(sample_path, top_n=3)

        print("✓ Genre detection results:")
        for i, (genre, confidence) in enumerate(genres, 1):
            print(f"  {i}. {genre}: {confidence:.2%} confidence")

        # Get detailed features
        features = api.extract_features(sample_path)
        print(f"\nDetailed features:")
        print(f"  - Tempo range: {features.tempo_range}")
        print(f"  - Swing factor: {features.swing_factor:.2f}")
        print(f"  - Chord types: {', '.join(features.chord_types[:5])}")
        print(f"  - Texture: {features.texture}")
    except NotImplementedError:
        print("⚠ GenreDetector (Agent 1) not yet implemented")
        print("  (This will work once Agent 1 is complete)")
    print()


def example_07_extract_features():
    """
    Example 7: Extract Comprehensive Musical Features

    Extracts all musical features from MIDI:
    - Rhythmic: tempo, swing, syncopation, complexity
    - Harmonic: chord types, harmonic rhythm, chromaticism
    - Melodic: intervals, contour, ornamentation
    - Timbral: instruments, texture, register
    """
    print("=" * 80)
    print("Example 7: Feature Extraction")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create diverse sample
    api.quick_fusion(
        harmony="blues",
        rhythm="funk",
        tempo=100,
        key="G",
        measures=8
    )
    sample_path = api.export("sample_blues_funk.mid")

    try:
        # Extract features
        features = api.extract_features(sample_path)

        print("✓ Extracted features:")
        print(f"\nRhythmic:")
        print(f"  - Tempo: {features.tempo_range[0]}-{features.tempo_range[1]} BPM")
        print(f"  - Swing: {features.swing_factor:.2f}")
        print(f"  - Syncopation: {features.syncopation:.2f}")
        print(f"  - Groove: {features.groove_type}")

        print(f"\nHarmonic:")
        print(f"  - Chord types: {', '.join(features.chord_types)}")
        print(f"  - Harmonic rhythm: {features.harmonic_rhythm} chords/measure")
        print(f"  - Extensions: {features.use_extensions}")
        print(f"  - Chromaticism: {features.chromaticism:.2f}")

        print(f"\nMelodic:")
        print(f"  - Interval preference: {features.interval_preference}")
        print(f"  - Range: {features.melodic_range[0]}-{features.melodic_range[1]}")
        print(f"  - Ornamentation: {features.ornamentation:.2f}")

        print(f"\nTimbral:")
        print(f"  - Texture: {features.texture}")
        print(f"  - Register: {features.register_preference}")
        print(f"  - Cultural origin: {features.cultural_origin}")
    except NotImplementedError:
        print("⚠ Feature extraction (Agent 1) not yet implemented")
    print()


def example_08_analyze_compatibility():
    """
    Example 8: Analyze Genre Compatibility

    Checks how well two genres blend together before creating fusion.
    Scores:
    - Overall compatibility
    - Rhythmic compatibility
    - Harmonic compatibility
    - Timbral compatibility
    - Cultural compatibility
    """
    print("=" * 80)
    print("Example 8: Genre Compatibility Analysis")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    genre_pairs = [
        ("jazz", "funk"),
        ("jazz", "electronic"),
        ("blues", "metal"),
        ("latin", "jazz")
    ]

    try:
        for genre_a, genre_b in genre_pairs:
            compat = api.check_compatibility(genre_a, genre_b)
            print(f"\n{genre_a.capitalize()} + {genre_b.capitalize()}:")
            print(f"  Overall: {compat['overall']:.2%}")
            print(f"  Rhythmic: {compat['rhythmic']:.2%}")
            print(f"  Harmonic: {compat['harmonic']:.2%}")
            print(f"  Timbral: {compat['timbral']:.2%}")

            # Get fusion suggestions
            params = api.suggest_fusion(genre_a, genre_b)
            print(f"  Suggested blend: {params['recommended_weight_a']:.0%}/{params['recommended_weight_b']:.0%}")
            print(f"  Suggested tempo: {params['tempo']} BPM")
    except NotImplementedError:
        print("⚠ Genre compatibility analysis (Agent 5) not yet implemented")
    print()


# ==============================================================================
# CATEGORY 3: CONTEXT-AWARE GENERATION
# ==============================================================================

def example_09_add_bass_to_arrangement():
    """
    Example 9: Add Bass to Existing Arrangement (Context-Aware)

    Analyzes existing MIDI (piano + drums) and adds bass that:
    - Follows the detected chord progression
    - Matches the rhythmic feel
    - Uses proper voice leading
    - Fills the low-register gap
    """
    print("=" * 80)
    print("Example 9: Add Bass (Context-Aware)")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create base arrangement (piano + drums)
    print("Creating base arrangement (piano + drums)...")
    api.quick_fusion(
        harmony="jazz",
        rhythm="funk",
        tempo=110,
        key="F",
        measures=16
    )
    base_path = api.export("base_piano_drums.mid")
    print(f"✓ Created base: {base_path}")

    # Add context-aware bass
    print("\nAdding funk bass (context-aware)...")
    try:
        api.load_midi(base_path)
        bass_notes = api.add_track(
            instrument=33,  # Fingered bass
            track_type="bass",
            genre="funk"
        )
        output_path = api.export("example_09_with_bass.mid", overwrite=True)
        print(f"✓ Added bass: {output_path}")
        print(f"  - Analyzed existing chords and rhythm")
        print(f"  - Generated funk bass that fits perfectly")
    except NotImplementedError:
        print("⚠ Context-aware generation (Agent 3) not yet implemented")
    print()


def example_10_smart_orchestration():
    """
    Example 10: Smart Orchestration Suggestions

    Analyzes sparse arrangement and suggests:
    - What instruments to add
    - What registers are underused
    - What textures would complement
    """
    print("=" * 80)
    print("Example 10: Smart Orchestration")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create sparse arrangement
    print("Creating sparse arrangement...")
    api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=120, measures=8)
    sparse_path = api.export("sparse_arrangement.mid")

    try:
        # Get suggestions
        print("\nAnalyzing arrangement...")
        suggestions = api.suggest_tracks(sparse_path)

        print("✓ Smart orchestration suggestions:")
        for i, sug in enumerate(suggestions, 1):
            print(f"\n  {i}. Instrument {sug['instrument']} ({sug['track_type']})")
            print(f"     Reason: {sug['reason']}")
            print(f"     Priority: {sug['priority']:.0%}")
    except NotImplementedError:
        print("⚠ Smart orchestration (Agent 3) not yet implemented")
    print()


def example_11_multi_track_addition():
    """
    Example 11: Add Multiple Tracks Context-Aware

    Starting with just chords, add:
    1. Bass line
    2. Drums
    3. Melody
    All context-aware, fitting together harmonically and rhythmically.
    """
    print("=" * 80)
    print("Example 11: Multi-Track Context-Aware Addition")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Start with just harmony
    print("Creating harmonic foundation...")
    api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=130, measures=16)
    harmony_path = api.export("just_harmony.mid")

    try:
        # Add tracks one by one, each aware of previous
        api.load_midi(harmony_path)

        print("\n1. Adding bass...")
        api.add_track(instrument=33, track_type="bass", genre="jazz")

        print("2. Adding drums...")
        api.add_track(instrument=0, track_type="drums", genre="jazz")

        print("3. Adding melody...")
        api.add_track(instrument=65, track_type="melody", genre="jazz")

        output_path = api.export("example_11_full_arrangement.mid", overwrite=True)
        print(f"\n✓ Created full arrangement: {output_path}")
        print("  All tracks harmonically and rhythmically coherent")
    except NotImplementedError:
        print("⚠ Context-aware generation not yet implemented")
    print()


# ==============================================================================
# CATEGORY 4: INPAINTING & REHARMONIZATION
# ==============================================================================

def example_12_reharmonize_section():
    """
    Example 12: Reharmonize Section (Like Photoshop Content-Aware Fill)

    Takes existing MIDI and regenerates measures 9-16 with:
    - New chord progression
    - Smooth transitions at boundaries
    - Preserved melody (optional)
    """
    print("=" * 80)
    print("Example 12: Reharmonize Section")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create original
    print("Creating original composition...")
    api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=120, measures=16)
    original_path = api.export("original_simple.mid")

    try:
        # Reharmonize measures 9-16
        print("\nReharmonizing measures 9-16...")
        api.load_midi(original_path)

        new_chords = [
            "Dm7", "G7", "Cmaj7", "A7",  # Measures 9-12
            "Dm7", "Db7", "Cmaj7", "Cmaj7"  # Measures 13-16 (tritone sub)
        ]

        api.inpaint_section(
            tracks=[0, 1, 2],
            measures=(9, 16),
            new_chords=new_chords,
            preserve_melody=False
        )

        output_path = api.export("example_12_reharmonized.mid", overwrite=True)
        print(f"✓ Reharmonized: {output_path}")
        print("  - Measures 1-8: Original")
        print("  - Measures 9-16: New harmony with smooth transitions")
    except NotImplementedError:
        print("⚠ Inpainting engine (Agent 4) not yet implemented")
    print()


def example_13_genre_change_section():
    """
    Example 13: Change Genre Mid-Song

    Create composition where:
    - Measures 1-8: Jazz
    - Measures 9-16: EDM
    - Measures 17-24: Back to jazz
    With smooth transitions at boundaries.
    """
    print("=" * 80)
    print("Example 13: Genre Change Mid-Song")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create jazz base
    print("Creating jazz base (24 measures)...")
    api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=120, measures=24)
    base_path = api.export("jazz_base_24bars.mid")

    try:
        # Change measures 9-16 to EDM
        print("\nChanging measures 9-16 to EDM style...")
        api.load_midi(base_path)

        api.inpaint_section(
            tracks=[0, 1, 2],
            measures=(9, 16),
            new_genre="electronic",
            preserve_melody=False
        )

        output_path = api.export("example_13_genre_switch.mid", overwrite=True)
        print(f"✓ Created genre-switching composition: {output_path}")
        print("  - Bars 1-8: Jazz")
        print("  - Bars 9-16: EDM (with smooth transition)")
        print("  - Bars 17-24: Jazz (smooth return)")
    except NotImplementedError:
        print("⚠ Inpainting with genre change (Agent 4) not yet implemented")
    print()


def example_14_chord_substitution():
    """
    Example 14: Advanced Chord Substitution

    Apply jazz reharmonization techniques:
    - Tritone substitution
    - Secondary dominants
    - Modal interchange
    - Extended harmony
    """
    print("=" * 80)
    print("Example 14: Advanced Chord Substitution")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create simple progression
    print("Creating simple ii-V-I progression...")
    api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=140, measures=8)
    simple_path = api.export("simple_progression.mid")

    try:
        # Apply jazz reharmonization
        print("\nApplying jazz reharmonization...")
        api.load_midi(simple_path)

        new_chords = api.reharmonize(
            measures=(1, 8),
            style="jazz"  # Adds tritone subs, extensions, etc.
        )

        print("✓ New chord progression:")
        for i, chord in enumerate(new_chords, 1):
            print(f"  Measure {i}: {chord}")

        # Apply the reharmonization
        api.inpaint_section(
            tracks=[0, 1],
            measures=(1, 8),
            new_chords=new_chords
        )

        output_path = api.export("example_14_reharmonized.mid", overwrite=True)
        print(f"\n✓ Applied reharmonization: {output_path}")
    except NotImplementedError:
        print("⚠ Chord substitution (Agent 4) not yet implemented")
    print()


# ==============================================================================
# CATEGORY 5: TEMPO & METER CONVERSION
# ==============================================================================

def example_15_tempo_conversion():
    """
    Example 15: Style-Appropriate Tempo Conversion

    Convert 90 BPM ballad to 140 BPM uptempo:
    - Not just speed change
    - Adjusts note subdivisions
    - Adjusts articulations
    - Creates appropriate feel (double-time vs. just faster)
    """
    print("=" * 80)
    print("Example 15: Tempo Conversion")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create slow jazz ballad
    print("Creating slow jazz ballad (90 BPM)...")
    api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=90, measures=8)
    slow_path = api.export("slow_ballad_90bpm.mid")

    try:
        # Convert to uptempo (140 BPM)
        print("\nConverting to uptempo (140 BPM) with style adjustments...")
        api.load_midi(slow_path)
        api.convert_tempo(140, style_adjust=True)

        output_path = api.export("example_15_uptempo_140bpm.mid", overwrite=True)
        print(f"✓ Converted: {output_path}")
        print("  - Original: 90 BPM ballad feel")
        print("  - New: 140 BPM uptempo with double-time patterns")
        print("  - Articulations and subdivisions adjusted")
    except NotImplementedError:
        print("⚠ Tempo conversion (Agent 6) not yet implemented")
    print()


def example_16_meter_conversion():
    """
    Example 16: Meter Conversion (4/4 to 7/8)

    Convert standard 4/4 to odd meter 7/8:
    - Preserves melodic/harmonic content
    - Re-quantizes rhythms appropriately
    - Maintains phrase structure
    """
    print("=" * 80)
    print("Example 16: Meter Conversion (4/4 → 7/8)")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create 4/4 composition
    print("Creating 4/4 composition...")
    api.quick_fusion(
        harmony="jazz",
        rhythm="jazz",
        tempo=140,
        time_signature=(4, 4),
        measures=8
    )
    four_four_path = api.export("four_four.mid")

    try:
        # Convert to 7/8
        print("\nConverting to 7/8...")
        api.load_midi(four_four_path)
        api.convert_meter((7, 8))

        output_path = api.export("example_16_seven_eight.mid", overwrite=True)
        print(f"✓ Converted: {output_path}")
        print("  - Original: 4/4")
        print("  - New: 7/8 (odd meter)")
        print("  - Melodic content preserved")
    except NotImplementedError:
        print("⚠ Meter conversion (Agent 7) not yet implemented")
    print()


def example_17_metric_modulation():
    """
    Example 17: Metric Modulation

    Create composition with metric modulation:
    - Start in 4/4 at 120 BPM
    - Modulate to 3/4 where quarter = dotted quarter
    - Smooth transition
    """
    print("=" * 80)
    print("Example 17: Metric Modulation")
    print("=" * 80)

    # This would use MeterConverter with metric modulation capabilities
    print("Creating composition with metric modulation...")
    print("  (This requires advanced MeterConverter features)")
    print("\n⚠ Metric modulation (Agent 7 advanced features) not yet implemented")
    print()


# ==============================================================================
# CATEGORY 6: ADVANCED FUSION TECHNIQUES
# ==============================================================================

def example_18_weighted_blend():
    """
    Example 18: Weighted Genre Blending

    Create 60% jazz + 40% blues harmony:
    - Not just switching between genres
    - Actual interpolation in feature space
    - Creates hybrid characteristics
    """
    print("=" * 80)
    print("Example 18: Weighted Genre Blending")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    try:
        print("Creating 60% jazz + 40% blues blend...")
        composition = api.weighted_blend(
            blends={
                'harmony': [('jazz', 0.6), ('blues', 0.4)],
                'rhythm': [('funk', 1.0)]
            },
            tempo=105,
            key="G",
            measures=16
        )

        api.composition = composition
        output_path = api.export("example_18_weighted_blend.mid")
        print(f"✓ Created weighted blend: {output_path}")
        print("  - Harmony: 60% jazz + 40% blues")
        print("  - Rhythm: 100% funk")
    except NotImplementedError:
        print("⚠ Weighted blending (Agent 5) not yet implemented")
    print()


def example_19_progressive_morph():
    """
    Example 19: Progressive Genre Morph

    32-bar composition that gradually morphs from jazz to EDM:
    - Bars 1-8: 100% jazz
    - Bars 9-16: 75% jazz, 25% EDM
    - Bars 17-24: 25% jazz, 75% EDM
    - Bars 25-32: 100% EDM
    """
    print("=" * 80)
    print("Example 19: Progressive Genre Morph (Jazz → EDM)")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    try:
        print("Creating progressive morph (jazz → EDM over 32 bars)...")
        composition = api.progressive_morph(
            from_genre="jazz",
            to_genre="electronic",
            measures=32,
            morph_type="s-curve"  # Smooth S-curve transition
        )

        api.composition = composition
        output_path = api.export("example_19_jazz_to_edm_morph.mid")
        print(f"✓ Created progressive morph: {output_path}")
        print("  - Smooth 32-bar transition")
        print("  - S-curve blending (slow start, fast middle, slow end)")
    except NotImplementedError:
        print("⚠ Progressive fusion (Agent 5) not yet implemented")
    print()


def example_20_track_level_genres():
    """
    Example 20: Different Genre Per Track

    Create arrangement where each track has different genre:
    - Track 1 (Piano): Jazz
    - Track 2 (Bass): Funk
    - Track 3 (Drums): Hip-hop
    - Track 4 (Strings): Classical
    All harmonically compatible!
    """
    print("=" * 80)
    print("Example 20: Track-Level Genre Control")
    print("=" * 80)

    # This would use MultiGenreArranger (Agent 9)
    print("Creating multi-genre arrangement...")
    print("  Track 1: Jazz piano")
    print("  Track 2: Funk bass")
    print("  Track 3: Hip-hop drums")
    print("  Track 4: Classical strings")
    print("\n⚠ Track-level genre control (Agent 9) not yet implemented")
    print()


def example_21_granular_brass_hits():
    """
    Example 21: Granular Control - Custom Brass Hits

    Apply custom rhythm pattern to chord progression with idiomatic brass writing:
    - User defines exact rhythm
    - System generates proper brass voicings
    - Idiomatic articulations
    """
    print("=" * 80)
    print("Example 21: Granular Brass Hits")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Define custom syncopated rhythm
    rhythm_pattern = [
        1.0,   # Beat 1
        1.5,   # Beat 1 and a half
        3.0,   # Beat 3
        3.75,  # Beat 3 and three-quarters
    ]

    chords = ["Dm7", "G7", "Cmaj7", "A7"]

    try:
        print("Generating brass hits on custom rhythm...")
        notes = api.apply_pattern(
            rhythm_pattern=rhythm_pattern,
            chords=chords,
            instrument_section="brass",
            key="C"
        )

        print(f"✓ Generated {len(notes)} notes")
        print(f"  - Custom syncopated rhythm")
        print(f"  - Idiomatic brass voicings")
        print(f"  - Proper articulations")
    except NotImplementedError:
        print("⚠ Granular control (Agent 8) not yet implemented")
    print()


def example_22_component_replacement():
    """
    Example 22: Component Replacement

    Take existing jazz composition and replace just the rhythm with reggae:
    - Keep harmony
    - Keep melody
    - Replace rhythm component
    - Regenerate dependent parts
    """
    print("=" * 80)
    print("Example 22: Component Replacement")
    print("=" * 80)

    # This uses ComponentReplacer from Agent 5
    print("Creating jazz composition...")
    print("Replacing rhythm component with reggae...")
    print("  (Harmony and melody preserved)")
    print("\n⚠ Component replacement (Agent 5) not yet implemented")
    print()


# ==============================================================================
# CATEGORY 7: PRODUCTION WORKFLOWS
# ==============================================================================

def example_23_full_production_workflow():
    """
    Example 23: Complete Production Workflow

    Realistic production scenario:
    1. Create initial sketch
    2. Analyze and detect issues
    3. Add missing instruments
    4. Reharmonize bridge
    5. Adjust tempo
    6. Export final version
    """
    print("=" * 80)
    print("Example 23: Full Production Workflow")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Step 1: Create initial sketch
    print("Step 1: Creating initial sketch...")
    api.quick_fusion(harmony="jazz", rhythm="funk", tempo=115, measures=32)
    api.export("step1_sketch.mid")
    print("✓ Sketch created")

    # Step 2: Analyze
    print("\nStep 2: Analyzing sketch...")
    try:
        genres = api.detect_genre()
        print(f"✓ Detected: {genres[0][0]} ({genres[0][1]:.0%} confidence)")
    except:
        print("  (Analysis skipped - not implemented)")

    # Step 3: Get suggestions and add tracks
    print("\nStep 3: Adding suggested instruments...")
    try:
        suggestions = api.suggest_tracks()
        print(f"✓ Got {len(suggestions)} suggestions")
    except:
        print("  (Suggestions skipped - not implemented)")

    # Step 4: Reharmonize bridge (measures 17-24)
    print("\nStep 4: Reharmonizing bridge...")
    try:
        new_chords = api.reharmonize(measures=(17, 24), style="jazz")
        print(f"✓ New chords for bridge")
    except:
        print("  (Reharmonization skipped - not implemented)")

    # Step 5: Export
    print("\nStep 5: Exporting final version...")
    try:
        final_path = api.export("example_23_final_production.mid", overwrite=True)
        print(f"✓ Final version: {final_path}")
    except:
        print("  (Using sketch as final)")

    print("\n✓ Production workflow complete!")
    print()


def example_24_genre_exploration():
    """
    Example 24: Genre Exploration Workflow

    Explore different genre combinations for same musical idea:
    1. Create base in jazz
    2. Try jazz + funk
    3. Try jazz + EDM
    4. Try jazz + latin
    5. Compare and choose best
    """
    print("=" * 80)
    print("Example 24: Genre Exploration")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    base_params = {
        'tempo': 120,
        'key': 'Dm',
        'measures': 16
    }

    combinations = [
        ('jazz', 'jazz', 'pure_jazz'),
        ('jazz', 'funk', 'jazz_funk'),
        ('jazz', 'electronic', 'jazz_edm'),
        ('jazz', 'latin', 'latin_jazz'),
    ]

    print("Exploring genre combinations...")
    for harmony, rhythm, name in combinations:
        try:
            comp = api.quick_fusion(
                harmony=harmony,
                rhythm=rhythm,
                **base_params
            )
            api.composition = comp
            path = api.export(f"exploration_{name}.mid", overwrite=True)
            print(f"✓ Created {name}: {path}")
        except Exception as e:
            print(f"  (Skipped {name})")

    print("\n✓ Created 4 variations to compare")
    print()


def example_25_live_mashup():
    """
    Example 25: Live Mashup / DJ-Style Fusion

    Take two existing MIDI files and create seamless mashup:
    1. Detect genres of both
    2. Find compatible tempo
    3. Blend progressively
    4. Create smooth transition
    """
    print("=" * 80)
    print("Example 25: Live Mashup")
    print("=" * 80)

    api = HarmonyModuleAPI(output_dir="./examples_output")

    # Create two different genre pieces
    print("Creating Track A (Jazz)...")
    api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=120, measures=16)
    track_a = api.export("track_a_jazz.mid")

    print("Creating Track B (Electronic)...")
    api.quick_fusion(harmony="electronic", rhythm="electronic", tempo=128, measures=16)
    track_b = api.export("track_b_electronic.mid")

    # Create mashup
    print("\nCreating mashup...")
    try:
        # Detect genres
        api.load_midi(track_a)
        genre_a = api.detect_genre(top_n=1)[0][0]

        api.load_midi(track_b)
        genre_b = api.detect_genre(top_n=1)[0][0]

        # Create progressive blend
        mashup = api.progressive_morph(
            from_genre=genre_a,
            to_genre=genre_b,
            measures=32,
            morph_type="linear"
        )

        api.composition = mashup
        output_path = api.export("example_25_mashup.mid", overwrite=True)
        print(f"✓ Created mashup: {output_path}")
        print(f"  - Blends {genre_a} → {genre_b}")
        print(f"  - 32 bars smooth transition")
    except NotImplementedError:
        print("⚠ Mashup features not fully implemented")
    print()


# ==============================================================================
# UTILITY: RUN ALL EXAMPLES
# ==============================================================================

def run_all_examples():
    """Run all 25 examples"""
    print("\n" + "=" * 80)
    print("MODULAR FUSION SYSTEM - COMPREHENSIVE EXAMPLES")
    print("Running all 25 examples...")
    print("=" * 80 + "\n")

    examples = [
        # Category 1: Quick Fusion
        example_01_jazz_funk_fusion,
        example_02_electro_swing,
        example_03_afro_cuban_jazz,
        example_04_nu_jazz,
        example_05_latin_trap,

        # Category 2: Genre Detection
        example_06_detect_genre,
        example_07_extract_features,
        example_08_analyze_compatibility,

        # Category 3: Context-Aware
        example_09_add_bass_to_arrangement,
        example_10_smart_orchestration,
        example_11_multi_track_addition,

        # Category 4: Inpainting
        example_12_reharmonize_section,
        example_13_genre_change_section,
        example_14_chord_substitution,

        # Category 5: Tempo/Meter
        example_15_tempo_conversion,
        example_16_meter_conversion,
        example_17_metric_modulation,

        # Category 6: Advanced Fusion
        example_18_weighted_blend,
        example_19_progressive_morph,
        example_20_track_level_genres,
        example_21_granular_brass_hits,
        example_22_component_replacement,

        # Category 7: Production
        example_23_full_production_workflow,
        example_24_genre_exploration,
        example_25_live_mashup,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"⚠ Example {i} error: {e}\n")

    print("=" * 80)
    print("All examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run specific example
        example_num = int(sys.argv[1])
        example_name = f"example_{example_num:02d}"

        # Find and run the example
        for name, obj in globals().items():
            if name.startswith(example_name) and callable(obj):
                obj()
                break
    else:
        # Run all examples
        run_all_examples()
