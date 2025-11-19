#!/usr/bin/env python3
"""
Modular Fusion Examples - Agent 5 Enhancement

Demonstrates all features of the enhanced Style Fusion system:
1. N-way component mixing (ModularFusion)
2. Component replacement (ComponentReplacer)
3. Detailed compatibility analysis (GenreCompatibilityAnalyzer)
4. Track-level fusion (TrackLevelFusion)
5. Progressive genre morphing (ProgressiveFusion)

Each example is self-contained and demonstrates a specific use case.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from midi_generator.generators.style_fusion import (
    ModularFusion,
    ComponentReplacer,
    GenreCompatibilityAnalyzer,
    TrackLevelFusion,
    ProgressiveFusion,
    ComponentType,
    GENRE_PROFILES,
    StyleFusion
)


# ==============================================================================
# EXAMPLE 1: Basic Component Fusion
# ==============================================================================

def example_01_basic_component_fusion():
    """
    Example 1: Mix Jazz harmony with Funk rhythm

    Use case: Create a jazz-funk fusion composition
    Result: Complex jazz chords over funky syncopated groove
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Component Fusion - Jazz Harmony + Funk Rhythm")
    print("="*70)

    modular = ModularFusion()

    # Create fusion with jazz harmony and funk rhythm
    result = modular.fuse_components(
        rhythm_genre="funk",
        harmony_genre="jazz",
        tempo=108
    )

    print(f"\nCreated: {result.name}")
    print(f"Tempo: {result.metadata['tempo']} BPM")
    print(f"\nFeatures:")
    print(f"  - Rhythm: {result.features.groove_type} (from Funk)")
    print(f"  - Syncopation: {result.features.syncopation:.2f}")
    print(f"  - Harmony: {result.features.harmonic_rhythm:.1f} chords/measure (from Jazz)")
    print(f"  - Chord types: {', '.join(result.features.chord_types[:5])}...")
    print(f"  - Extensions: {result.features.use_extensions}")

    print("\nPerfect for: Neo-soul, jazz-funk, fusion")


# ==============================================================================
# EXAMPLE 2: Complex Multi-Genre Fusion
# ==============================================================================

def example_02_complex_multi_genre():
    """
    Example 2: Four-way component mix

    Components:
    - Rhythm: Latin (clave, high syncopation)
    - Harmony: Jazz (extended chords, chromaticism)
    - Melody: Blues (stepwise, ornamentation)
    - Instrumentation: Electronic (synths)

    Result: Electro-Latin-Jazz-Blues fusion
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Complex Multi-Genre Fusion (4 genres)")
    print("="*70)

    modular = ModularFusion()

    result = modular.fuse_components(
        rhythm_genre="latin",
        harmony_genre="jazz",
        melody_genre="blues",
        instrumentation_genre="electronic",
        tempo=118
    )

    print(f"\nCreated: {result.name}")
    print(f"Tempo: {result.metadata['tempo']} BPM")
    print(f"\nComponent Breakdown:")
    print(f"  1. Rhythm (Latin):")
    print(f"     - Basis: {result.features.rhythmic_basis}")
    print(f"     - Groove: {result.features.groove_type}")
    print(f"     - Syncopation: {result.features.syncopation:.2f}")
    print(f"\n  2. Harmony (Jazz):")
    print(f"     - Chord types: {len(result.features.chord_types)} types")
    print(f"     - Harmonic rhythm: {result.features.harmonic_rhythm:.1f} chords/measure")
    print(f"     - Chromaticism: {result.features.chromaticism:.2f}")
    print(f"\n  3. Melody (Blues):")
    print(f"     - Interval preference: {result.features.interval_preference}")
    print(f"     - Ornamentation: {result.features.ornamentation:.2f}")
    print(f"\n  4. Instrumentation (Electronic):")
    print(f"     - Instruments: {result.features.instruments}")
    print(f"     - Texture: {result.features.texture}")

    print("\nPerfect for: Experimental fusion, world music electronica")


# ==============================================================================
# EXAMPLE 3: Weighted N-Way Fusion
# ==============================================================================

def example_03_weighted_nway_fusion():
    """
    Example 3: Weighted blending using barycentric coordinates

    Blend harmony from 3 genres:
    - 50% Jazz (extended chords, complex)
    - 30% Blues (simpler, bluesy)
    - 20% Funk (rhythmic harmony)

    Result: Smooth interpolation in feature space
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Weighted N-Way Fusion (Barycentric Blending)")
    print("="*70)

    modular = ModularFusion()

    # Define weighted components
    result = modular.weighted_fusion([
        (ComponentType.HARMONY, "jazz", 0.5),
        (ComponentType.HARMONY, "blues", 0.3),
        (ComponentType.HARMONY, "funk", 0.2),
        (ComponentType.RHYTHM, "latin", 1.0)
    ])

    print(f"\nCreated: {result.name}")
    print(f"Fusion type: {result.metadata['fusion_type']}")
    print(f"\nWeighted Harmony Blend:")
    print(f"  - Jazz: 50%")
    print(f"  - Blues: 30%")
    print(f"  - Funk: 20%")
    print(f"\nResulting Features:")
    print(f"  - Harmonic rhythm: {result.features.harmonic_rhythm:.2f} chords/measure")
    print(f"  - Chromaticism: {result.features.chromaticism:.2f}")
    print(f"  - Total chord types: {len(result.features.chord_types)}")
    print(f"  - Chord vocabulary: {', '.join(result.features.chord_types[:6])}...")

    print("\nPerfect for: Smooth genre transitions, progressive compositions")


# ==============================================================================
# EXAMPLE 4: Component Replacement
# ==============================================================================

def example_04_component_replacement():
    """
    Example 4: Replace individual components

    Start with: Jazz composition
    Replace: Rhythm component with Funk
    Keep: Jazz harmony, melody, instrumentation

    Result: Jazz-Funk hybrid with precise control
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Component Replacement - Surgical Genre Mixing")
    print("="*70)

    # Start with jazz
    original = GENRE_PROFILES['jazz']
    print(f"\nOriginal: {original.name}")
    print(f"  - Swing: {original.swing_factor:.2f}")
    print(f"  - Groove: {original.groove_type}")
    print(f"  - Chords: {', '.join(original.chord_types[:4])}...")

    # Replace rhythm with funk
    replacer = ComponentReplacer(original)
    modified = replacer.replace_component(ComponentType.RHYTHM, "funk")

    print(f"\nAfter replacing rhythm with Funk:")
    print(f"  - New name: {modified.name}")
    print(f"  - Swing: {modified.swing_factor:.2f} (changed)")
    print(f"  - Groove: {modified.groove_type} (changed)")
    print(f"  - Chords: {', '.join(modified.chord_types[:4])}... (preserved)")

    # Multiple replacements
    multi_modified = replacer.replace_multiple({
        ComponentType.RHYTHM: "latin",
        ComponentType.INSTRUMENTATION: "electronic"
    })

    print(f"\nAfter multiple replacements (Latin rhythm + Electronic instruments):")
    print(f"  - New name: {multi_modified.name}")
    print(f"  - Rhythmic basis: {multi_modified.rhythmic_basis}")
    print(f"  - Instruments: {multi_modified.instruments[:3]}...")
    print(f"  - Chords: {', '.join(multi_modified.chord_types[:4])}... (still preserved)")

    print("\nPerfect for: Remixing, arrangement variations")


# ==============================================================================
# EXAMPLE 5: Compatibility Analysis
# ==============================================================================

def example_05_compatibility_analysis():
    """
    Example 5: Analyze genre compatibility before fusion

    Compare multiple genre combinations to find best matches
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Detailed Compatibility Analysis")
    print("="*70)

    # Analyze several genre pairs
    pairs = [
        ("jazz", "funk"),
        ("jazz", "electronic"),
        ("funk", "hiphop"),
        ("latin", "jazz"),
        ("blues", "electronic")
    ]

    print("\nCompatibility Scores (0.0-1.0):")
    print("-" * 70)
    print(f"{'Genre Pair':<25} {'Overall':<10} {'Rhythmic':<10} {'Harmonic':<10} {'Cultural':<10}")
    print("-" * 70)

    for genre_a, genre_b in pairs:
        compat = GenreCompatibilityAnalyzer.analyze_compatibility(genre_a, genre_b)
        print(f"{f'{genre_a} + {genre_b}':<25} "
              f"{compat['overall']:<10.2f} "
              f"{compat['rhythmic']:<10.2f} "
              f"{compat['harmonic']:<10.2f} "
              f"{compat['cultural']:<10.2f}")

    # Get fusion suggestions
    print("\n\nFusion Parameter Suggestions:")
    print("-" * 70)

    params = GenreCompatibilityAnalyzer.suggest_fusion_parameters("jazz", "electronic")
    print(f"\nJazz + Electronic Fusion:")
    print(f"  Recommended weights: {params['recommended_weight_a']:.1f} (jazz) / "
          f"{params['recommended_weight_b']:.1f} (electronic)")
    print(f"  Compromise tempo: {params['tempo']} BPM")
    print(f"  Focus component: {params['focus_component'].value}")
    print(f"  Overall compatibility: {params['compatibility_scores']['overall']:.2f}")

    print("\nPerfect for: Pre-fusion analysis, optimal blend calculation")


# ==============================================================================
# EXAMPLE 6: Track-Level Fusion
# ==============================================================================

def example_06_track_level_fusion():
    """
    Example 6: Different genre per track

    Create multi-genre arrangement:
    - Track 0: Funk bass
    - Track 1: Jazz piano/harmony
    - Track 2: Hip-hop drums
    - Track 3: Electronic pads

    All tracks follow global harmony and tempo
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Track-Level Fusion - Multi-Genre Arrangement")
    print("="*70)

    track_fusion = TrackLevelFusion(tempo=95, key="Dm", time_signature=(4, 4))

    # Assign genres to tracks
    track_fusion.set_track_genre(0, ComponentType.BASS, "funk")
    track_fusion.set_track_genre(1, ComponentType.HARMONY, "jazz")
    track_fusion.set_track_genre(2, ComponentType.DRUMS, "hiphop")
    track_fusion.set_track_genre(3, ComponentType.INSTRUMENTATION, "electronic")

    # Set global harmony
    track_fusion.set_global_harmony(["Dm7", "G7", "Cmaj7", "A7alt"])

    # Generate arrangement plan
    arrangement = track_fusion.generate_arrangement_plan()

    print(f"\nArrangement Details:")
    print(f"  Tempo: {arrangement['tempo']} BPM")
    print(f"  Key: {arrangement['key']}")
    print(f"  Time signature: {arrangement['time_signature']}")
    print(f"  Global harmony: {arrangement['global_harmony']}")

    print(f"\n  Tracks:")
    for track_num, track_info in arrangement['tracks'].items():
        print(f"    Track {track_num}: {track_info['genre'].capitalize()} "
              f"({track_info['component']})")
        features = track_info['features']
        print(f"      - Swing: {features.swing_factor:.2f}")
        print(f"      - Instruments: {features.instruments[:2]}...")

    print(f"\n  Inter-track Compatibility:")
    for (ta, tb), score in arrangement['compatibility_matrix'].items():
        genre_a = arrangement['tracks'][ta]['genre']
        genre_b = arrangement['tracks'][tb]['genre']
        status = "✓ Good" if score > 0.7 else "⚠ Check"
        print(f"    {genre_a} ↔ {genre_b}: {score:.2f} {status}")

    print("\nPerfect for: Multi-genre productions, complex arrangements")


# ==============================================================================
# EXAMPLE 7: Progressive Fusion - Linear
# ==============================================================================

def example_07_progressive_linear():
    """
    Example 7: Gradual linear transition

    Morph from Jazz to Electronic over 16 measures
    Smooth, constant-rate transition
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Progressive Fusion - Linear Transition")
    print("="*70)

    progressive = ProgressiveFusion("jazz", "electronic", 16)
    measures = progressive.generate_progressive_fusion(morph_type="linear", tempo=120)

    print(f"\nTransition: Jazz → Electronic")
    print(f"Total measures: 16")
    print(f"Type: Linear (constant rate)")
    print(f"\nMeasure-by-Measure Breakdown:")
    print("-" * 70)

    checkpoints = [0, 4, 8, 12, 15]
    for i in checkpoints:
        weights = progressive._calculate_morph_weights("linear")
        jazz_pct = int(weights[i] * 100)
        elec_pct = 100 - jazz_pct

        measure = measures[i]
        print(f"\n  Measure {i+1}: {jazz_pct}% Jazz, {elec_pct}% Electronic")
        print(f"    - Swing: {measure.swing_factor:.2f}")
        print(f"    - Chromaticism: {measure.chromaticism:.2f}")
        print(f"    - Extensions: {measure.use_extensions}")

    print("\nPerfect for: Smooth transitions, build-ups, breakdowns")


# ==============================================================================
# EXAMPLE 8: Progressive Fusion - S-Curve
# ==============================================================================

def example_08_progressive_scurve():
    """
    Example 8: S-curve transition (slow-fast-slow)

    Morph from Funk to Blues with sigmoid curve
    Slow start, fast middle, slow end
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Progressive Fusion - S-Curve Transition")
    print("="*70)

    progressive = ProgressiveFusion("funk", "blues", 12)
    measures = progressive.generate_progressive_fusion(morph_type="s-curve", tempo=100)

    print(f"\nTransition: Funk → Blues")
    print(f"Total measures: 12")
    print(f"Type: S-Curve (slow-fast-slow, natural feel)")

    # Show transition curve
    weights = progressive._calculate_morph_weights("s-curve")

    print(f"\nTransition Curve:")
    print("-" * 70)
    for i in range(0, 12, 2):
        funk_pct = int(weights[i] * 100)
        bar_length = int(funk_pct / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  Measure {i+1:2d}: {bar} {funk_pct}% Funk")

    print(f"\nRate of change:")
    for i in [0, 3, 6, 9]:
        if i < len(weights) - 1:
            rate = abs(weights[i] - weights[i+1])
            print(f"  Measures {i+1}-{i+2}: {rate:.3f} (delta)")

    print("\nPerfect for: Natural-sounding transitions, musical phrasing")


# ==============================================================================
# EXAMPLE 9: Progressive Fusion - Exponential
# ==============================================================================

def example_09_progressive_exponential():
    """
    Example 9: Exponential decay transition

    Morph from Latin to Hip-Hop with exponential curve
    Fast initial change, slow later
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Progressive Fusion - Exponential Transition")
    print("="*70)

    progressive = ProgressiveFusion("latin", "hiphop", 10)
    measures = progressive.generate_progressive_fusion(morph_type="exponential", tempo=105)

    print(f"\nTransition: Latin → Hip-Hop")
    print(f"Total measures: 10")
    print(f"Type: Exponential Decay (fast start, slow end)")

    weights = progressive._calculate_morph_weights("exponential")

    print(f"\nTransition Progress:")
    print("-" * 70)
    for i in range(10):
        latin_pct = int(weights[i] * 100)
        hiphop_pct = 100 - latin_pct
        print(f"  Measure {i+1:2d}: {latin_pct:3d}% Latin, {hiphop_pct:3d}% Hip-Hop")

    print(f"\nKey transition points:")
    print(f"  - Measure 1: {int(weights[0]*100)}% Latin (start)")
    print(f"  - Measure 3: {int(weights[2]*100)}% Latin (major change)")
    print(f"  - Measure 7: {int(weights[6]*100)}% Latin (stabilizing)")
    print(f"  - Measure 10: {int(weights[9]*100)}% Latin (end)")

    print("\nPerfect for: Dramatic shifts, drop transitions")


# ==============================================================================
# EXAMPLE 10: Complete Production Workflow
# ==============================================================================

def example_10_complete_workflow():
    """
    Example 10: End-to-end production workflow

    Steps:
    1. Analyze genre compatibility
    2. Create initial fusion
    3. Replace component for variation
    4. Set up multi-track arrangement
    5. Create progressive transition
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: Complete Production Workflow")
    print("="*70)

    print("\n--- STEP 1: Analyze Compatibility ---")
    compat = GenreCompatibilityAnalyzer.analyze_compatibility("jazz", "funk")
    print(f"Jazz + Funk compatibility: {compat['overall']:.2f} (✓ Good match)")

    print("\n--- STEP 2: Create Initial Fusion ---")
    modular = ModularFusion()
    initial = modular.fuse_components("funk", "jazz", tempo=110)
    print(f"Created: {initial.name}")
    print(f"Tempo: {initial.metadata['tempo']} BPM")

    print("\n--- STEP 3: Create Section Variation ---")
    replacer = ComponentReplacer(initial.features)
    variation = replacer.replace_component(ComponentType.RHYTHM, "latin")
    print(f"Variation (for bridge): {variation.name}")
    print(f"Changed rhythmic basis to: {variation.rhythmic_basis}")

    print("\n--- STEP 4: Set Up Multi-Track Arrangement ---")
    track_fusion = TrackLevelFusion(tempo=110, key="Dm")
    track_fusion.set_track_genre(0, ComponentType.BASS, "funk")
    track_fusion.set_track_genre(1, ComponentType.HARMONY, "jazz")
    track_fusion.set_track_genre(2, ComponentType.DRUMS, "hiphop")
    track_fusion.set_global_harmony(["Dm7", "G7", "Cmaj7", "Fmaj7"])
    arrangement = track_fusion.generate_arrangement_plan()
    print(f"Tracks configured: {len(arrangement['tracks'])}")
    print(f"Global harmony: {arrangement['global_harmony']}")

    print("\n--- STEP 5: Create Outro Transition ---")
    progressive = ProgressiveFusion("jazz", "electronic", 8)
    outro = progressive.generate_progressive_fusion(morph_type="s-curve", tempo=110)
    print(f"Outro transition: {len(outro)} measures (jazz → electronic)")

    print("\n--- FINAL STRUCTURE ---")
    print("""
    Intro (8 bars):      Jazz-Funk fusion
    Verse (16 bars):     Multi-track arrangement (funk bass, jazz harmony, hip-hop drums)
    Bridge (8 bars):     Latin rhythm variation
    Chorus (16 bars):    Back to main fusion
    Outro (8 bars):      Progressive morph to electronic

    Total: 56 bars of seamless multi-genre composition
    """)

    print("Perfect for: Full productions, professional arrangements")


# ==============================================================================
# MAIN - Run All Examples
# ==============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" MODULAR FUSION EXAMPLES - Agent 5 Enhancement")
    print(" Comprehensive demonstration of all features")
    print("="*70)

    examples = [
        ("Basic Component Fusion", example_01_basic_component_fusion),
        ("Complex Multi-Genre Fusion", example_02_complex_multi_genre),
        ("Weighted N-Way Fusion", example_03_weighted_nway_fusion),
        ("Component Replacement", example_04_component_replacement),
        ("Compatibility Analysis", example_05_compatibility_analysis),
        ("Track-Level Fusion", example_06_track_level_fusion),
        ("Progressive Linear", example_07_progressive_linear),
        ("Progressive S-Curve", example_08_progressive_scurve),
        ("Progressive Exponential", example_09_progressive_exponential),
        ("Complete Workflow", example_10_complete_workflow),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i:2d}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(" All examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
