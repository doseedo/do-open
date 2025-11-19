#!/usr/bin/env python3
"""
Film Scoring Engine - Usage Examples

Demonstrates all features of the advanced film scoring engine:
1. Video analysis
2. Leitmotif systems
3. Progression morphing
4. Tension-based generation
5. Advanced compositional techniques
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from film_scoring_engine import (
    FilmScoringEngine,
    FilmScoringTechniques,
    LeitmotifEngine,
    Leitmotif,
    VideoAnalyzer,
    TensionArc,
    MoodCategory,
    TensionLevel,
    ScoringSyncType,
    SMPTETimecode,
    score_video_to_midi,
)


# ============================================================================
# EXAMPLE 1: Simple Video-to-MIDI
# ============================================================================

def example_1_basic_video_scoring():
    """
    Simplest usage: Analyze video and generate adaptive score
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Video Scoring")
    print("="*70)

    # Note: Replace with actual video path
    video_path = "/path/to/your/video.mp4"

    # One-line convenience function
    try:
        midi_path = score_video_to_midi(
            video_path=video_path,
            output_midi="example1_output.mid",
            bpm=120
        )
        print(f"\n✅ Generated score: {midi_path}")
    except FileNotFoundError:
        print(f"\n⚠️  Video not found: {video_path}")
        print("   Replace with actual video path to run this example")


# ============================================================================
# EXAMPLE 2: Progression Morphing Based on Video
# ============================================================================

def example_2_progression_morphing():
    """
    Start with a base progression and morph it based on video mood/tension
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Progression Morphing")
    print("="*70)

    # Base progression (I-vi-IV-V in C major)
    base_progression = {
        0: "Cmaj7",
        4: "Am7",
        8: "Fmaj7",
        12: "G7"
    }

    print(f"\nBase progression: {base_progression}")

    # Simulate video features (without actual video)
    # In real usage, this comes from VideoAnalyzer
    techniques = FilmScoringTechniques()

    # Scenario 1: Bright, happy scene (low tension)
    print("\n--- Scenario 1: Bright, Happy Scene ---")
    morphed_happy = techniques.morph_progression(
        base_progression,
        target_mood=MoodCategory.WARM_BRIGHT,
        tension=0.2  # Low tension
    )
    print(f"Morphed (happy): {morphed_happy}")

    # Scenario 2: Dark, tense scene (high tension)
    print("\n--- Scenario 2: Dark, Tense Scene ---")
    morphed_tense = techniques.morph_progression(
        base_progression,
        target_mood=MoodCategory.COOL_DARK,
        tension=0.8  # High tension
    )
    print(f"Morphed (tense): {morphed_tense}")

    # Scenario 3: Mysterious scene
    print("\n--- Scenario 3: Mysterious Scene ---")
    morphed_mystery = techniques.morph_progression(
        base_progression,
        target_mood=MoodCategory.DESATURATED,
        tension=0.6
    )
    print(f"Morphed (mystery): {morphed_mystery}")


# ============================================================================
# EXAMPLE 3: Leitmotif System (Star Wars Style)
# ============================================================================

def example_3_leitmotif_system():
    """
    Create character/location themes with variations
    Like John Williams' Star Wars approach
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Leitmotif System (Character Themes)")
    print("="*70)

    engine = LeitmotifEngine()

    # Define hero theme (major, heroic)
    hero_theme = Leitmotif(
        name="Hero Theme",
        chord_progression={
            0: "C",
            2: "F",
            4: "G",
            6: "C"
        },
        melody_contour=[1, 3, 5, 8, 5, 3, 1],  # Ascending then descending
        harmonic_character="major",
        tempo_range=(120, 140),
        heroic_variation="C",
        tragic_variation="Cm"
    )

    # Define villain theme (minor, ominous)
    villain_theme = Leitmotif(
        name="Villain Theme",
        chord_progression={
            0: "Am",
            2: "Dm",
            4: "E7",
            6: "Am"
        },
        melody_contour=[1, 2, 3, 5, 4, 3, 1],
        harmonic_character="minor",
        tempo_range=(80, 100),
        mysterious_variation="Adim"
    )

    # Register themes
    engine.register_motif(hero_theme)
    engine.register_motif(villain_theme)

    # Get variations based on dramatic context
    print("\n--- Hero's Triumphant Moment (Low Tension) ---")
    hero_triumph = engine.get_variation(
        "Hero Theme",
        tension=0.2,  # Low tension, triumphant
        tempo_factor=1.0
    )
    print(f"Hero (triumph): {hero_triumph}")

    print("\n--- Hero in Danger (High Tension) ---")
    hero_danger = engine.get_variation(
        "Hero Theme",
        tension=0.9,  # High tension, slower
        tempo_factor=0.7  # Slower, more dramatic
    )
    print(f"Hero (danger): {hero_danger}")

    print("\n--- Villain Appears (Medium Tension) ---")
    villain_appears = engine.get_variation(
        "Villain Theme",
        tension=0.6,
        transpose_semitones=-2  # Transpose down for darker sound
    )
    print(f"Villain: {villain_appears}")


# ============================================================================
# EXAMPLE 4: Chromatic Voice Leading (Zimmer Style)
# ============================================================================

def example_4_chromatic_techniques():
    """
    Advanced harmonic techniques from Hans Zimmer / John Williams
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Chromatic Voice Leading (Zimmer/Williams Style)")
    print("="*70)

    techniques = FilmScoringTechniques()

    # Chromatic voice leading (smooth half-step motion)
    print("\n--- Chromatic Voice Leading: Cm → Eb ---")
    chromatic_seq = techniques.chromatic_voice_leading(
        start_chord="Cm",
        end_chord="Eb",
        steps=4
    )
    print(f"Chromatic sequence: {chromatic_seq}")
    print("Usage: Tense build-up, suspenseful transition")

    # Ostinato patterns (repeating for suspense)
    print("\n--- Ostinato Patterns ---")

    suspense_ostinato = techniques.ostinato_pattern(
        root_note="C",
        pattern_type="suspense"
    )
    print(f"Suspense ostinato: {suspense_ostinato}")
    print("Usage: Thriller scenes, ticking clock")

    action_ostinato = techniques.ostinato_pattern(
        root_note="E",
        pattern_type="action"
    )
    print(f"Action ostinato: {action_ostinato}")
    print("Usage: Chase scenes, battle sequences")

    mystery_ostinato = techniques.ostinato_pattern(
        root_note="F",
        pattern_type="mystery"
    )
    print(f"Mystery ostinato: {mystery_ostinato}")
    print("Usage: Detective work, investigation")


# ============================================================================
# EXAMPLE 5: Tension Arc Mapping
# ============================================================================

def example_5_tension_arc():
    """
    Map emotional tension over time and generate music accordingly
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Tension Arc Mapping")
    print("="*70)

    # Create manual tension arc (simulates video analysis result)
    tension_arc = TensionArc(
        timestamps=[0.0, 15.0, 30.0, 45.0, 60.0],
        tension_values=[0.2, 0.4, 0.7, 0.9, 0.3]  # Build then release
    )

    techniques = FilmScoringTechniques()

    print("\nTension arc over 60 seconds:")
    for t, tension in zip(tension_arc.timestamps, tension_arc.tension_values):
        print(f"  {t:5.1f}s: Tension = {tension:.2f}")

    # Sample tension at different points
    print("\nChord complexity at different times:")
    for time_point in [0, 10, 20, 30, 40, 50, 60]:
        tension = tension_arc.get_tension_at(time_point)
        chord_quality = techniques.tension_to_chord_complexity(tension)
        print(f"  {time_point:3d}s: tension={tension:.2f} → use '{chord_quality}' chords")


# ============================================================================
# EXAMPLE 6: Mood-Based Scale Selection
# ============================================================================

def example_6_mood_to_scale():
    """
    Map visual mood (from color analysis) to musical scales
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Visual Mood → Musical Scale Mapping")
    print("="*70)

    techniques = FilmScoringTechniques()

    moods = [
        MoodCategory.WARM_BRIGHT,
        MoodCategory.WARM_DARK,
        MoodCategory.COOL_BRIGHT,
        MoodCategory.COOL_DARK,
        MoodCategory.SATURATED,
        MoodCategory.DESATURATED,
    ]

    print("\nMood-to-Scale Mappings:")
    for mood in moods:
        scale = techniques.mood_to_scale_context(mood)
        print(f"  {mood.value:20s} → {scale}")


# ============================================================================
# EXAMPLE 7: SMPTE Timecode
# ============================================================================

def example_7_smpte_timecode():
    """
    Work with SMPTE timecode for frame-accurate synchronization
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: SMPTE Timecode (Frame-Accurate Sync)")
    print("="*70)

    # Create timecode
    tc1 = SMPTETimecode(hours=0, minutes=1, seconds=30, frames=12, framerate=24.0)
    print(f"\nTimecode 1: {tc1} @ {tc1.framerate}fps")
    print(f"  = {tc1.to_seconds():.3f} seconds")

    # Create from seconds
    tc2 = SMPTETimecode.from_seconds(95.5, framerate=24.0)
    print(f"\nTimecode 2 (from 95.5s): {tc2}")

    # Hit points (musical sync points)
    print("\nExample Hit Points (Music Sync to Picture):")
    hit_points = [
        (SMPTETimecode(0, 0, 5, 0), "Door slams", "accent"),
        (SMPTETimecode(0, 0, 15, 12), "Character enters", "chord_change"),
        (SMPTETimecode(0, 0, 30, 0), "Reveal moment", "modulation"),
        (SMPTETimecode(0, 1, 0, 0), "Climax", "accent"),
    ]

    for tc, description, musical_event in hit_points:
        print(f"  {tc} ({tc.to_seconds():6.2f}s) - {description:20s} → {musical_event}")


# ============================================================================
# EXAMPLE 8: Full Integration Example
# ============================================================================

def example_8_full_integration():
    """
    Complete example: Video analysis → Leitmotifs → Adaptive scoring
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Full Integration (Advanced)")
    print("="*70)

    # Note: This requires actual video file
    video_path = "/path/to/video.mp4"

    # Setup
    engine = FilmScoringEngine(video_path=video_path, bpm=120)

    # Register leitmotifs
    hero_theme = Leitmotif(
        name="Hero",
        chord_progression={0: "C", 4: "F", 8: "G", 12: "C"},
        harmonic_character="major"
    )
    engine.leitmotif_engine.register_motif(hero_theme)

    # Analyze video (if file exists)
    try:
        features = engine.analyze_video()

        # Generate score using tension arc
        midi_path = engine.generate_score(
            base_progression=hero_theme.chord_progression,
            scoring_approach=ScoringSyncType.TENSION_ARC,
            output_path="example8_full_score.mid"
        )

        print(f"\n✅ Full score generated: {midi_path}")

    except FileNotFoundError:
        print(f"\n⚠️  Video not found: {video_path}")
        print("   This example requires actual video file")
        print("\n   Workflow:")
        print("   1. Video analysis extracts scenes, colors, mood")
        print("   2. Leitmotifs applied based on dramatic context")
        print("   3. Progressions morphed to match tension arc")
        print("   4. MIDI generated with frame-accurate sync")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FILM SCORING ENGINE - COMPREHENSIVE EXAMPLES")
    print("="*70)

    # Run examples (skip video-dependent ones if no video)
    # example_1_basic_video_scoring()  # Requires video
    example_2_progression_morphing()
    example_3_leitmotif_system()
    example_4_chromatic_techniques()
    example_5_tension_arc()
    example_6_mood_to_scale()
    example_7_smpte_timecode()
    # example_8_full_integration()  # Requires video

    print("\n" + "="*70)
    print("✅ Examples complete!")
    print("="*70)
    print("\nTo run with actual video:")
    print("  python film_scoring_examples.py --video /path/to/video.mp4")
    print()
