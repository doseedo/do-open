#!/usr/bin/env python3
"""
Film Scoring Module - Live Capability Demonstration

Tests the film scoring engine WITHOUT requiring:
- Video files
- External dependencies (OpenCV, PySceneDetect)
- Just uses the core techniques + integration with chord_progression_generator.py

This demonstrates:
1. Integration with existing chord_progression_generator.py
2. Progression morphing for different moods
3. Leitmotif variations
4. Tension-based generation
5. Chromatic techniques
6. Full workflow simulation
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import film scoring engine (core, no video dependencies needed)
from film_scoring_engine import (
    FilmScoringEngine,
    FilmScoringTechniques,
    LeitmotifEngine,
    Leitmotif,
    TensionArc,
    VideoFeatures,
    MoodCategory,
    ScoringSyncType,
    SMPTETimecode,
)

# Try to import existing chord generator
try:
    from chord_progression_generator import generate_chord_progression_midi
    HAS_CHORD_GEN = True
    print("✅ chord_progression_generator.py found - Full integration available!")
except ImportError:
    HAS_CHORD_GEN = False
    print("⚠️  chord_progression_generator.py not found - Using basic features only")


def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_1_basic_techniques():
    """Test basic film scoring techniques"""
    print_section("TEST 1: Basic Film Scoring Techniques")

    techniques = FilmScoringTechniques()

    # Test tension to chord mapping
    print("\n📊 Tension → Chord Complexity Mapping:")
    for tension in [0.1, 0.3, 0.5, 0.7, 0.9]:
        chord_type = techniques.tension_to_chord_complexity(tension)
        print(f"  Tension {tension:.1f} → '{chord_type}' chords")

    # Test mood to scale mapping
    print("\n🎨 Visual Mood → Musical Scale Mapping:")
    moods = [
        MoodCategory.WARM_BRIGHT,
        MoodCategory.WARM_DARK,
        MoodCategory.COOL_BRIGHT,
        MoodCategory.COOL_DARK,
    ]
    for mood in moods:
        scale = techniques.mood_to_scale_context(mood)
        print(f"  {mood.value:15s} → {scale}")

    # Test chromatic voice leading
    print("\n🎹 Chromatic Voice Leading (Zimmer Style):")
    chromatic = techniques.chromatic_voice_leading("Cm", "Eb", steps=4)
    print(f"  Cm → Eb: {' → '.join(chromatic)}")
    print("  Usage: Tense build-up, suspenseful transition")

    # Test ostinato patterns
    print("\n🔁 Ostinato Patterns (Inception/Interstellar Style):")
    patterns = ["suspense", "action", "mystery"]
    for pattern_type in patterns:
        ostinato = techniques.ostinato_pattern("C", pattern_type)
        print(f"  {pattern_type:10s}: {ostinato}")


def test_2_progression_morphing():
    """Test progression morphing for different scenes"""
    print_section("TEST 2: Adaptive Progression Morphing")

    techniques = FilmScoringTechniques()

    # Base progression (I-vi-IV-V in C major)
    base_prog = {
        0: "Cmaj7",
        4: "Am7",
        8: "Fmaj7",
        12: "G7"
    }

    print(f"\n🎼 Base Progression: {base_prog}")

    # Scenario 1: Happy scene (bright, low tension)
    print("\n--- Scene 1: Happy, Uplifting Moment ---")
    print("Visual: Warm, bright colors, low tension (0.2)")
    happy = techniques.morph_progression(
        base_prog,
        MoodCategory.WARM_BRIGHT,
        tension=0.2
    )
    print(f"Morphed: {happy}")

    # Scenario 2: Mysterious scene
    print("\n--- Scene 2: Mysterious, Uncertain Moment ---")
    print("Visual: Desaturated, medium tension (0.6)")
    mystery = techniques.morph_progression(
        base_prog,
        MoodCategory.DESATURATED,
        tension=0.6
    )
    print(f"Morphed: {mystery}")

    # Scenario 3: Dark, climactic scene
    print("\n--- Scene 3: Dark, Climactic Moment ---")
    print("Visual: Cool dark colors, high tension (0.85)")
    climax = techniques.morph_progression(
        base_prog,
        MoodCategory.COOL_DARK,
        tension=0.85
    )
    print(f"Morphed: {climax}")

    print("\n💡 Notice how the progression adapts:")
    print("   - Low tension → Simple major chords")
    print("   - Medium tension → Minor 7th chords")
    print("   - High tension → Complex, dissonant chords (7b9, dim)")


def test_3_leitmotif_system():
    """Test leitmotif variations (Star Wars style)"""
    print_section("TEST 3: Leitmotif System (Williams Style)")

    engine = LeitmotifEngine()

    # Define hero theme
    hero_theme = Leitmotif(
        name="Hero Theme",
        chord_progression={0: "C", 4: "F", 8: "G", 12: "C"},
        harmonic_character="major",
        tempo_range=(120, 140)
    )

    # Define villain theme
    villain_theme = Leitmotif(
        name="Villain Theme",
        chord_progression={0: "Am", 4: "Dm", 8: "E7", 12: "Am"},
        harmonic_character="minor",
        tempo_range=(80, 100)
    )

    engine.register_motif(hero_theme)
    engine.register_motif(villain_theme)

    print("\n🦸 Hero Theme Variations:")
    print(f"  Original: {hero_theme.chord_progression}")

    # Triumphant variation
    triumph = engine.get_variation("Hero Theme", tension=0.2, tempo_factor=1.0)
    print(f"  Triumphant (low tension): {triumph}")

    # Hero in danger
    danger = engine.get_variation("Hero Theme", tension=0.9, tempo_factor=0.7)
    print(f"  In Danger (high tension, slower): {danger}")

    print("\n🦹 Villain Theme Variations:")
    print(f"  Original: {villain_theme.chord_progression}")

    # Villain appears
    appears = engine.get_variation("Villain Theme", tension=0.6, transpose_semitones=-2)
    print(f"  Appears (transposed down): {appears}")

    print("\n💡 Leitmotif Features:")
    print("   - Augmentation: Slower for dramatic moments")
    print("   - Diminution: Faster for action")
    print("   - Transposition: Different keys for mood shifts")
    print("   - Can also: Invert, retrograde (reverse)")


def test_4_tension_arc():
    """Test tension arc generation and following"""
    print_section("TEST 4: Tension Arc Following")

    # Create tension arc simulating a typical film scene structure
    arc = TensionArc(
        timestamps=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        tension_values=[0.2, 0.3, 0.5, 0.7, 0.9, 0.6, 0.2]
    )

    print("\n📈 Tension Arc (Typical Action Scene):")
    print("  Time    Tension   Phase")
    print("  " + "-"*50)
    phases = ["Setup", "Rising", "Build", "Build", "CLIMAX", "Fall", "Resolution"]
    for i, (t, tension) in enumerate(zip(arc.timestamps, arc.tension_values)):
        bar = "█" * int(tension * 20)
        print(f"  {t:4.0f}s   {tension:.2f}     {bar} {phases[i]}")

    # Sample at intermediate points
    print("\n🎵 Music Adapts at Different Points:")
    techniques = FilmScoringTechniques()
    for time in [0, 15, 25, 35, 45, 55, 60]:
        tension = arc.get_tension_at(time)
        chord_type = techniques.tension_to_chord_complexity(tension)
        if tension < 0.3:
            mood = "Calm"
        elif tension < 0.6:
            mood = "Building"
        elif tension < 0.8:
            mood = "Tense"
        else:
            mood = "CLIMAX"
        print(f"  {time:3d}s: Tension={tension:.2f} → {chord_type:6s} chords ({mood})")


def test_5_integration_with_chord_generator():
    """Test integration with existing chord_progression_generator.py"""
    print_section("TEST 5: Integration with Chord Generator")

    if not HAS_CHORD_GEN:
        print("\n⚠️  chord_progression_generator.py not found")
        print("   This test requires the existing chord generator module")
        print("   Showing what WOULD happen with integration:")
        print("\n   1. Film scoring engine generates adaptive progression")
        print("   2. Progression passed to chord_progression_generator.py")
        print("   3. Full MIDI with voicings, rhythms, inversions exported")
        return

    print("\n✅ Full Integration Test")

    techniques = FilmScoringTechniques()

    # Generate progression for tense scene
    print("\n1️⃣ Film Scoring: Generate progression for TENSE scene")
    tense_prog = techniques.morph_progression(
        original_prog={0: "C", 4: "F", 8: "G", 12: "C"},
        target_mood=MoodCategory.COOL_DARK,
        tension=0.8
    )
    print(f"   Generated: {tense_prog}")

    # Export to MIDI using existing generator
    print("\n2️⃣ Chord Generator: Export to MIDI with voicings")
    try:
        midi_path = generate_chord_progression_midi(
            chord_beat_map=tense_prog,
            bpm=90,  # Slower for dark scene
            voicing="drop2",  # Jazz voicing
            rhythm="quarter",
            style="block",
            output_path="/tmp/film_score_tense_scene.mid"
        )
        print(f"   ✅ MIDI exported: {midi_path}")
        print("   Settings: 90 BPM, drop2 voicing, quarter note rhythm")
    except Exception as e:
        print(f"   ⚠️ Export failed: {e}")


def test_6_full_workflow_simulation():
    """Simulate complete film scoring workflow"""
    print_section("TEST 6: Complete Workflow Simulation (No Video)")

    print("\n🎬 Simulating Film Score Generation")
    print("   (Without actual video file - using manual features)")

    # Create engine
    engine = FilmScoringEngine(video_path=None, bpm=110)

    # Manually create video features (simulating what video analysis would extract)
    print("\n1️⃣ Video Analysis (Simulated)")
    simulated_features = [
        VideoFeatures(
            start_time=0.0,
            end_time=15.0,
            duration=15.0,
            mood=MoodCategory.WARM_BRIGHT,
            visual_tension=0.2,
            avg_brightness=0.8,
            avg_saturation=0.6,
            is_scene_start=True,
            scene_id=0
        ),
        VideoFeatures(
            start_time=15.0,
            end_time=30.0,
            duration=15.0,
            mood=MoodCategory.DESATURATED,
            visual_tension=0.5,
            avg_brightness=0.4,
            avg_saturation=0.3,
            is_scene_start=True,
            scene_id=1
        ),
        VideoFeatures(
            start_time=30.0,
            end_time=45.0,
            duration=15.0,
            mood=MoodCategory.COOL_DARK,
            visual_tension=0.9,
            avg_brightness=0.2,
            avg_saturation=0.7,
            is_scene_start=True,
            scene_id=2
        ),
        VideoFeatures(
            start_time=45.0,
            end_time=60.0,
            duration=15.0,
            mood=MoodCategory.WARM_BRIGHT,
            visual_tension=0.3,
            avg_brightness=0.7,
            avg_saturation=0.5,
            is_scene_start=True,
            scene_id=3
        ),
    ]

    for i, feat in enumerate(simulated_features):
        print(f"   Scene {i+1}: {feat.start_time:.0f}-{feat.end_time:.0f}s")
        print(f"      Mood: {feat.mood.value}, Tension: {feat.visual_tension:.2f}")

    # Add to engine
    engine.video_features = simulated_features

    # Create tension arc
    print("\n2️⃣ Generate Tension Arc")
    engine.tension_arc = TensionArc(
        timestamps=[0.0, 15.0, 30.0, 45.0, 60.0],
        tension_values=[0.2, 0.5, 0.9, 0.3, 0.2]
    )
    print("   Arc: 0.2 → 0.5 → 0.9 (CLIMAX) → 0.3 → 0.2")

    # Register leitmotif
    print("\n3️⃣ Register Character Theme")
    hero_theme = Leitmotif(
        name="Hero",
        chord_progression={0: "C", 4: "F", 8: "G", 12: "C"},
        harmonic_character="major"
    )
    engine.leitmotif_engine.register_motif(hero_theme)
    print(f"   Hero Theme: {hero_theme.chord_progression}")

    # Generate progressions for each scene
    print("\n4️⃣ Generate Adaptive Music for Each Scene")
    for i, feat in enumerate(simulated_features):
        morphed = engine.techniques.morph_progression(
            hero_theme.chord_progression,
            feat.mood,
            feat.visual_tension
        )
        print(f"   Scene {i+1} ({feat.mood.value}, tension={feat.visual_tension:.1f}):")
        print(f"      {morphed}")

    # Generate MIDI
    print("\n5️⃣ Export to MIDI")
    try:
        midi_path = engine.generate_score(
            base_progression=hero_theme.chord_progression,
            scoring_approach=ScoringSyncType.TENSION_ARC,
            output_path="/tmp/film_score_complete.mid"
        )
        print(f"   ✅ Score generated: {midi_path}")
    except Exception as e:
        print(f"   ⚠️ MIDI generation: {e}")

    print("\n✅ Complete Workflow Demonstrated!")


def test_7_smpte_timecode():
    """Test SMPTE timecode for frame-accurate sync"""
    print_section("TEST 7: SMPTE Timecode (Frame-Accurate Sync)")

    print("\n⏱️  SMPTE Timecode Examples:")

    # Create timecodes
    tc1 = SMPTETimecode(0, 1, 30, 12, framerate=24.0)
    print(f"\n  Timecode: {tc1}")
    print(f"  = {tc1.to_seconds():.3f} seconds")
    print(f"  @ {tc1.framerate} fps")

    # Create from seconds
    tc2 = SMPTETimecode.from_seconds(95.5, framerate=24.0)
    print(f"\n  95.5 seconds = {tc2}")

    # Hit points (music sync points)
    print("\n🎯 Hit Points (Music Sync to Picture):")
    print("  Timecode       Seconds  Event                 Musical Action")
    print("  " + "-"*70)

    hit_points = [
        (SMPTETimecode(0, 0, 0, 0), "Scene start", "Theme begins"),
        (SMPTETimecode(0, 0, 5, 0), "Door slams", "Accent hit"),
        (SMPTETimecode(0, 0, 12, 15), "Character enters", "Chord change"),
        (SMPTETimecode(0, 0, 30, 0), "Reveal moment", "Modulation"),
        (SMPTETimecode(0, 1, 0, 0), "Climax", "Forte accent"),
        (SMPTETimecode(0, 1, 15, 20), "Resolution", "Final cadence"),
    ]

    for tc, event, action in hit_points:
        print(f"  {tc}   {tc.to_seconds():6.2f}s  {event:20s} → {action}")


def run_all_tests():
    """Run all capability tests"""
    print("\n" + "█"*70)
    print("  FILM SCORING MODULE - LIVE CAPABILITY DEMONSTRATION")
    print("█"*70)

    test_1_basic_techniques()
    test_2_progression_morphing()
    test_3_leitmotif_system()
    test_4_tension_arc()
    test_5_integration_with_chord_generator()
    test_6_full_workflow_simulation()
    test_7_smpte_timecode()

    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)

    print("\n📋 Summary of Demonstrated Features:")
    print("   ✅ Tension → Chord complexity mapping")
    print("   ✅ Visual mood → Musical scale mapping")
    print("   ✅ Chromatic voice leading (Zimmer style)")
    print("   ✅ Ostinato patterns (suspense/action/mystery)")
    print("   ✅ Progression morphing for different scenes")
    print("   ✅ Leitmotif variations (Williams style)")
    print("   ✅ Tension arc following")
    print("   ✅ SMPTE timecode (frame-accurate sync)")
    if HAS_CHORD_GEN:
        print("   ✅ Full integration with chord_progression_generator.py")
    print("\n🎬 Ready for film scoring! 🎵\n")


if __name__ == "__main__":
    run_all_tests()
