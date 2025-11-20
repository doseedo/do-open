#!/usr/bin/env python3
"""
Agent 18: Harmony Specialist - Demo & Examples
===============================================

Demonstrates the advanced harmony analysis and generation capabilities
of the Harmony Specialist agent.

Features demonstrated:
1. Jazz voicing analysis and generation
2. Modal chord progression generation
3. Functional harmony analysis
4. Voice leading optimization
5. Reharmonization techniques
6. Comprehensive feature extraction

Author: Agent 18 - Harmony Specialist
Date: 2025
License: MIT
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experts.harmony_specialist import (
    HarmonySpecialist,
    Chord,
    ChordProgression,
    Note,
    VoicingType,
    ChordQuality,
    Mode,
    HarmonicFunction,
    analyze_harmony,
    generate_jazz_voicing,
)


def demo_jazz_voicings():
    """Demonstrate jazz voicing analysis and generation."""
    print("=" * 70)
    print("DEMO 1: Jazz Voicing Analysis and Generation")
    print("=" * 70)
    print()

    specialist = HarmonySpecialist()

    # Generate different voicings of Cmaj7
    root = 0  # C
    quality = ChordQuality.MAJOR_7

    print("Generating different voicings of Cmaj7:")
    print()

    voicing_types = [
        VoicingType.CLOSE_POSITION,
        VoicingType.DROP_2,
        VoicingType.DROP_3,
        VoicingType.ROOTLESS_A,
        VoicingType.SHELL,
    ]

    for voicing_type in voicing_types:
        chord = generate_jazz_voicing(root, quality, voicing_type)
        analysis = specialist.analyze_jazz_voicings(chord)

        print(f"{voicing_type.value.upper()}:")
        print(f"  Pitches: {chord.pitches}")
        print(f"  Spacing: {analysis.spacing}")
        print(f"  Range: {analysis.range} semitones")
        print(f"  Density: {analysis.density:.2f} notes/octave")
        print(f"  Avg spacing: {analysis.average_spacing:.2f} semitones")
        print()

    print()


def demo_modal_progressions():
    """Demonstrate modal chord progression generation."""
    print("=" * 70)
    print("DEMO 2: Modal Chord Progressions")
    print("=" * 70)
    print()

    specialist = HarmonySpecialist()

    modes = [
        Mode.DORIAN,
        Mode.LYDIAN,
        Mode.MIXOLYDIAN,
        Mode.PHRYGIAN,
    ]

    for mode in modes:
        print(f"Generating {mode.value.upper()} progression in D:")
        progression = specialist.generate_modal_progression(
            mode=mode,
            length=4,
            key=2  # D
        )

        for i, chord in enumerate(progression.chords):
            print(f"  Chord {i+1}: Root={chord.root}, Quality={chord.quality}, Pitches={chord.pitches}")

        print()

    print()


def demo_functional_harmony():
    """Demonstrate functional harmony analysis."""
    print("=" * 70)
    print("DEMO 3: Functional Harmony Analysis")
    print("=" * 70)
    print()

    specialist = HarmonySpecialist()

    # Create a classic ii-V-I progression in C
    print("Analyzing ii-V-I progression in C major:")
    print()

    # Dm7 (ii)
    dm7 = specialist._analyze_chord([62, 65, 69, 72])  # D-F-A-C
    # G7 (V)
    g7 = specialist._analyze_chord([67, 71, 74, 77])   # G-B-D-F
    # Cmaj7 (I)
    cmaj7 = specialist._analyze_chord([60, 64, 67, 71])  # C-E-G-B

    progression = ChordProgression(
        chords=[dm7, g7, cmaj7],
        key=0,  # C
        mode=Mode.IONIAN
    )

    functions = specialist.analyze_functional_harmony(progression)

    for i, (chord, function) in enumerate(zip(progression.chords, functions)):
        print(f"Chord {i+1}:")
        print(f"  Pitches: {chord.pitches}")
        print(f"  Root: {chord.root}")
        print(f"  Quality: {chord.quality}")
        print(f"  Function: {function.value}")
        print()

    print()


def demo_voice_leading():
    """Demonstrate voice leading analysis and optimization."""
    print("=" * 70)
    print("DEMO 4: Voice Leading Analysis")
    print("=" * 70)
    print()

    specialist = HarmonySpecialist()

    # Create two chords for voice leading analysis
    print("Analyzing voice leading: Cmaj7 -> Dm7")
    print()

    cmaj7 = specialist._analyze_chord([60, 64, 67, 71])  # C-E-G-B
    dm7 = specialist._analyze_chord([62, 65, 69, 72])    # D-F-A-C

    analysis = specialist.analyze_voice_leading(cmaj7, dm7)

    print(f"Voice motions (semitones): {analysis.voice_motions}")
    print(f"Motion types: {analysis.motion_types}")
    print(f"Total motion: {analysis.total_motion} semitones")
    print(f"Common tones: {analysis.common_tones}")
    print(f"Smoothness: {analysis.smoothness:.2f}")
    print(f"Quality score: {analysis.quality_score:.2f}")
    print(f"Violations: {analysis.violations if analysis.violations else 'None'}")
    print()

    # Demonstrate voice leading optimization
    print("Optimizing voice leading in a progression:")
    print()

    progression = ChordProgression(
        chords=[cmaj7, dm7, g7 := specialist._analyze_chord([67, 71, 74, 77]), cmaj7],
        key=0
    )

    print("Before optimization:")
    for i, chord in enumerate(progression.chords):
        print(f"  Chord {i+1}: {chord.pitches}")

    optimized = specialist.optimize_voice_leading(progression)

    print("\nAfter optimization:")
    for i, chord in enumerate(optimized.chords):
        print(f"  Chord {i+1}: {chord.pitches}")

    print()


def demo_reharmonization():
    """Demonstrate reharmonization techniques."""
    print("=" * 70)
    print("DEMO 5: Reharmonization")
    print("=" * 70)
    print()

    specialist = HarmonySpecialist()

    # Create a simple melody
    melody = [
        Note(pitch=72, start_time=0.0, duration=1.0),   # C
        Note(pitch=74, start_time=1.0, duration=1.0),   # D
        Note(pitch=76, start_time=2.0, duration=1.0),   # E
        Note(pitch=77, start_time=3.0, duration=1.0),   # F
        Note(pitch=79, start_time=4.0, duration=1.0),   # G
        Note(pitch=81, start_time=5.0, duration=1.0),   # A
        Note(pitch=83, start_time=6.0, duration=1.0),   # B
        Note(pitch=84, start_time=7.0, duration=1.0),   # C
    ]

    print("Melody: C-D-E-F-G-A-B-C")
    print()

    styles = ['jazz', 'classical', 'contemporary']

    for style in styles:
        print(f"Reharmonization style: {style.upper()}")
        progression = specialist.reharmonize(melody, style=style, key=0)

        for i, chord in enumerate(progression.chords):
            melody_note = melody[i].pitch % 12 if i < len(melody) else 0
            note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
            print(f"  Beat {i+1} (melody: {note_names[melody_note]}): "
                  f"Root={chord.root}, Pitches={chord.pitches}")

        print()

    print()


def demo_feature_extraction():
    """Demonstrate comprehensive feature extraction."""
    print("=" * 70)
    print("DEMO 6: Comprehensive Harmony Feature Extraction")
    print("=" * 70)
    print()

    specialist = HarmonySpecialist()

    # Create a chord progression
    chords = [
        specialist._analyze_chord([60, 64, 67, 71]),   # Cmaj7
        specialist._analyze_chord([62, 65, 69, 72]),   # Dm7
        specialist._analyze_chord([67, 71, 74, 77]),   # G7
        specialist._analyze_chord([60, 64, 67, 71]),   # Cmaj7
    ]

    specialist.chords = chords
    specialist.key = 0
    specialist.mode = Mode.IONIAN

    # Extract features
    features = specialist.extract_features()

    print("Extracted Harmony Features (50+ parameters):")
    print()

    # Voicing features
    print("VOICING FEATURES:")
    print(f"  Average voicing density: {features.avg_voicing_density:.2f}")
    print(f"  Average voicing range: {features.avg_voicing_range:.2f}")
    print(f"  Close position ratio: {features.close_position_ratio:.2f}")
    print(f"  Open position ratio: {features.open_position_ratio:.2f}")
    print(f"  Drop 2 usage: {features.drop2_usage:.2f}")
    print(f"  Rootless voicing usage: {features.rootless_usage:.2f}")
    print()

    # Modal features
    print("MODAL HARMONY FEATURES:")
    print(f"  Primary mode: {features.primary_mode}")
    print(f"  Dorian characteristic: {features.dorian_characteristic:.2f}")
    print(f"  Lydian characteristic: {features.lydian_characteristic:.2f}")
    print(f"  Mixolydian characteristic: {features.mixolydian_characteristic:.2f}")
    print()

    # Functional features
    print("FUNCTIONAL HARMONY FEATURES:")
    print(f"  Tonic ratio: {features.tonic_ratio:.2f}")
    print(f"  Subdominant ratio: {features.subdominant_ratio:.2f}")
    print(f"  Dominant ratio: {features.dominant_ratio:.2f}")
    print(f"  Functional clarity: {features.functional_clarity:.2f}")
    print(f"  Cadence strength: {features.cadence_strength:.2f}")
    print()

    # Voice leading features
    print("VOICE LEADING FEATURES:")
    print(f"  Average voice motion: {features.avg_voice_motion:.2f}")
    print(f"  Stepwise motion ratio: {features.stepwise_motion_ratio:.2f}")
    print(f"  Leap ratio: {features.leap_ratio:.2f}")
    print(f"  Contrary motion ratio: {features.contrary_motion_ratio:.2f}")
    print(f"  Voice leading smoothness: {features.voice_leading_smoothness:.2f}")
    print(f"  Common tone retention: {features.common_tone_retention:.2f}")
    print(f"  Voice independence: {features.voice_independence:.2f}")
    print()

    # Convert to dictionary for ML
    features_dict = specialist.to_dict()
    print(f"Total features for ML: {len(features_dict)}")
    print()


def demo_comprehensive():
    """Run a comprehensive demonstration of all features."""
    print("=" * 70)
    print("AGENT 18: HARMONY SPECIALIST - COMPREHENSIVE DEMO")
    print("=" * 70)
    print()
    print("This demo showcases all major capabilities of the Harmony Specialist:")
    print("  1. Jazz voicing analysis and generation")
    print("  2. Modal chord progression generation")
    print("  3. Functional harmony analysis")
    print("  4. Voice leading analysis and optimization")
    print("  5. Reharmonization techniques")
    print("  6. Comprehensive feature extraction (50+ parameters)")
    print()

    # Run all demos
    demo_jazz_voicings()
    demo_modal_progressions()
    demo_functional_harmony()
    demo_voice_leading()
    demo_reharmonization()
    demo_feature_extraction()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("The Harmony Specialist provides 50+ specialized harmony parameters")
    print("for advanced harmonic analysis and generation in the Musical Program")
    print("Synthesis system.")
    print()
    print("These features integrate with:")
    print("  - Agent 8: Deep Feature Extractor (1000+ features)")
    print("  - Agent 9: Feature-Parameter Mapper (XGBoost models)")
    print("  - Agent 14: Synthetic Training Data Generator")
    print("  - Agent 15: Model Training Specialist")
    print()


def test_integration():
    """Test integration with the broader system."""
    print("=" * 70)
    print("INTEGRATION TEST")
    print("=" * 70)
    print()

    specialist = HarmonySpecialist()

    # Create test progression
    print("Creating test chord progression...")
    progression = specialist.generate_modal_progression(Mode.DORIAN, length=8, key=2)
    print(f"Generated {len(progression.chords)} chords")
    print()

    # Extract features
    print("Extracting harmony features...")
    specialist.chords = progression.chords
    specialist.key = progression.key
    specialist.mode = progression.mode
    features = specialist.extract_features()
    print(f"Extracted features from {len(specialist.chords)} chords")
    print()

    # Convert to ML format
    print("Converting to ML format...")
    features_dict = specialist.to_dict()
    print(f"Generated {len(features_dict)} features for XGBoost training")
    print()

    # Display sample features
    print("Sample features:")
    sample_keys = list(features_dict.keys())[:10]
    for key in sample_keys:
        print(f"  {key}: {features_dict[key]}")
    print()

    print("Integration test PASSED ✓")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent 18 Harmony Specialist Demo")
    parser.add_argument(
        "--demo",
        choices=["all", "voicings", "modal", "functional", "voice_leading", "reharmonization", "features", "integration"],
        default="all",
        help="Which demo to run"
    )

    args = parser.parse_args()

    if args.demo == "all":
        demo_comprehensive()
    elif args.demo == "voicings":
        demo_jazz_voicings()
    elif args.demo == "modal":
        demo_modal_progressions()
    elif args.demo == "functional":
        demo_functional_harmony()
    elif args.demo == "voice_leading":
        demo_voice_leading()
    elif args.demo == "reharmonization":
        demo_reharmonization()
    elif args.demo == "features":
        demo_feature_extraction()
    elif args.demo == "integration":
        test_integration()

    print("For more information, see:")
    print("  - midi_generator/experts/harmony_specialist.py (source code)")
    print("  - midi_generator/AGENT_18_HARMONY_SPECIALIST.md (documentation)")
