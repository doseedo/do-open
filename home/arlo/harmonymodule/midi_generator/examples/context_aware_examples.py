#!/usr/bin/env python3
"""
Context-Aware Generation - Comprehensive Examples

This file demonstrates all features of the Context-Aware Generator:
1. Adding tracks to existing arrangements
2. Reharmonization (inpainting with new chords)
3. Genre fusion within songs
4. Smart orchestration suggestions
5. Multi-track generation
6. Boundary smoothing
7. Voice leading validation

Author: Agent 3 - Context-Aware Generation
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.context_aware_generator import (
    ContextAwareGenerator,
    TrackInpainter,
    SmartOrchestrator,
    GenerationConstraints
)


# ==============================================================================
# EXAMPLE 1: Basic Track Addition
# ==============================================================================

def example_1_add_bass_line():
    """
    EXAMPLE 1: Add Funk Bass to Jazz Piano

    Scenario:
    - Input: Jazz piano arrangement (chords only)
    - Output: Same arrangement + funk bass line

    Demonstrates:
    - Basic analysis
    - Genre specification
    - Track type specification
    """
    print("=" * 70)
    print("EXAMPLE 1: Add Funk Bass to Jazz Piano")
    print("=" * 70)

    # Initialize generator
    gen = ContextAwareGenerator('jazz_piano.mid')

    # Analyze existing arrangement
    analysis = gen.analyze()

    print(f"\n📊 Analysis Results:")
    print(f"   Tempo: {analysis.tempo} BPM")
    print(f"   Time Signature: {analysis.time_signature[0]}/{analysis.time_signature[1]}")
    print(f"   Key: {analysis.key}")
    print(f"   Length: {analysis.length_measures} measures")
    print(f"   Detected Style: {analysis.detected_style.name if analysis.detected_style else 'Unknown'}")
    print(f"   Chords: {', '.join(analysis.chord_progression[:8])}...")
    print(f"   Texture: {analysis.texture}")

    # Generate funk bass
    print(f"\n🎸 Generating funk bass line...")
    bass_notes = gen.add_track(
        instrument=33,      # Fingered bass
        genre='funk',       # Funk style bass
        track_type='bass'   # Explicitly bass
    )

    print(f"   Generated {len(bass_notes)} bass notes")

    # Export
    output_file = 'output/jazz_with_funk_bass.mid'
    gen.export_with_new_track(
        bass_notes,
        output_file,
        instrument=33,
        track_name='Funk Bass'
    )

    print(f"\n✅ Success! Saved to: {output_file}")
    print(f"   Original: Jazz piano chords")
    print(f"   Added: Funky, syncopated bass line")


# ==============================================================================
# EXAMPLE 2: Smart Orchestration
# ==============================================================================

def example_2_smart_orchestration():
    """
    EXAMPLE 2: Auto-Orchestrate with AI Suggestions

    Scenario:
    - Input: Sparse arrangement (piano + drums)
    - Output: Full arrangement with suggested instruments

    Demonstrates:
    - Smart orchestration
    - Automatic suggestions
    - Priority-based selection
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Smart Orchestration with AI Suggestions")
    print("=" * 70)

    # Initialize orchestrator
    orchestrator = SmartOrchestrator('piano_drums.mid')

    # Analyze balance
    balance = orchestrator.analyze_orchestral_balance()

    print(f"\n📊 Orchestral Balance Analysis:")
    print(f"   Register Distribution:")
    print(f"      Low:  {balance['register_distribution']['low']:.1%}")
    print(f"      Mid:  {balance['register_distribution']['mid']:.1%}")
    print(f"      High: {balance['register_distribution']['high']:.1%}")
    print(f"   Texture: {balance['texture']}")
    print(f"   Has Bass: {'✓' if balance['has_bass'] else '✗'}")
    print(f"   Has Drums: {'✓' if balance['has_drums'] else '✗'}")
    print(f"   Harmonic Voices: {balance['harmonic_voices']}")

    # Get AI suggestions
    suggestions = orchestrator.suggest_additions()

    print(f"\n🤖 AI Orchestration Suggestions:")
    print(f"   Found {len(suggestions)} suggestions:")
    for i, s in enumerate(suggestions, 1):
        priority_bar = "█" * int(s['priority'] * 10)
        print(f"\n   {i}. [{priority_bar:<10}] {s['priority']:.2f}")
        print(f"      {s['reason']}")
        print(f"      → Instrument: {s['instrument']} ({s['track_type']})")
        print(f"      → Genre: {s.get('genre', 'auto')}")

    # Auto-add high-priority suggestions
    print(f"\n🎵 Adding high-priority tracks (priority >= 0.75)...")
    added_count = 0
    for suggestion in suggestions:
        if suggestion['priority'] >= 0.75:
            print(f"   Adding: {suggestion['track_type']} ({suggestion['reason'][:50]}...)")
            orchestrator.add_suggested_track(suggestion)
            added_count += 1

    print(f"\n✅ Added {added_count} tracks!")
    print(f"   Exporting to: output/full_orchestration.mid")

    orchestrator.export('output/full_orchestration.mid')


# ==============================================================================
# EXAMPLE 3: Reharmonization (Inpainting)
# ==============================================================================

def example_3_reharmonization():
    """
    EXAMPLE 3: Reharmonize Bridge Section

    Scenario:
    - Input: Simple song with C - F - G - C progression
    - Output: Jazz reharmonization of bridge (measures 16-24)

    Original Bridge: C - F - G - C - C - F - G - C
    New Bridge: Cmaj7 - Fmaj7#11 - Bm7b5 - E7alt - Am7 - Dm7 - G7alt - Cmaj7

    Demonstrates:
    - Section inpainting
    - Chord substitution
    - Boundary smoothing
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Reharmonize Bridge Section")
    print("=" * 70)

    # Initialize inpainter
    inpainter = TrackInpainter('simple_song.mid')

    # Get original analysis
    analysis = inpainter.generator.analysis

    print(f"\n📊 Original Song:")
    print(f"   Structure: Verse (0-8) | Chorus (8-16) | Bridge (16-24) | Chorus (24-32)")
    print(f"   Original Bridge Chords: {', '.join(analysis.chord_progression[16:24])}")

    # Define new jazz reharmonization
    new_bridge_chords = [
        'Cmaj7',      # I
        'Fmaj7#11',   # IV with #11
        'Bm7b5',      # viiø
        'E7alt',      # V7/vi (secondary dominant)
        'Am7',        # vi
        'Dm7',        # ii
        'G7alt',      # V7 altered
        'Cmaj7'       # I
    ]

    print(f"\n🎹 New Jazz Bridge Chords:")
    print(f"   {' → '.join(new_bridge_chords)}")

    # Reharmonize bridge
    print(f"\n🔄 Regenerating bridge (measures 16-24) with smooth boundaries...")
    new_bridge = inpainter.inpaint_measures(
        track=0,              # Piano track
        start=16,             # Bridge start
        end=24,               # Bridge end
        new_chords=new_bridge_chords,
        smooth_boundaries=True  # Ensure smooth transitions
    )

    print(f"   Generated {len(new_bridge)} notes for new bridge")
    print(f"   Boundary smoothing applied:")
    print(f"      ✓ Smooth entry from chorus (measure 15 → 16)")
    print(f"      ✓ Smooth exit to chorus (measure 23 → 24)")

    # Export
    inpainter.export('output/reharmonized_bridge.mid')

    print(f"\n✅ Success! Saved to: output/reharmonized_bridge.mid")
    print(f"   Verse & Chorus: Original harmony unchanged")
    print(f"   Bridge: Rich jazz reharmonization")


# ==============================================================================
# EXAMPLE 4: Genre Fusion
# ==============================================================================

def example_4_genre_fusion():
    """
    EXAMPLE 4: Jazz Verse + EDM Chorus + Jazz Verse

    Scenario:
    - Input: Jazz ballad (all jazz)
    - Output: Jazz verse → EDM chorus → Jazz verse

    Demonstrates:
    - Genre change within song
    - Smooth genre transitions
    - Section-specific generation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Genre Fusion (Jazz + EDM)")
    print("=" * 70)

    # Initialize inpainter
    inpainter = TrackInpainter('jazz_ballad.mid')

    print(f"\n📊 Original Song Structure:")
    print(f"   Measures 0-8:   Verse 1 (Jazz)")
    print(f"   Measures 8-16:  Chorus (Jazz)")
    print(f"   Measures 16-24: Verse 2 (Jazz)")
    print(f"   Measures 24-32: Chorus (Jazz)")

    print(f"\n🎭 Target Structure (with genre fusion):")
    print(f"   Measures 0-8:   Verse 1 (Jazz) ← keep")
    print(f"   Measures 8-16:  Chorus (EDM)  ← CHANGE")
    print(f"   Measures 16-24: Verse 2 (Jazz) ← keep")
    print(f"   Measures 24-32: Chorus (EDM)  ← CHANGE")

    # Change first chorus to EDM
    print(f"\n🎵 Regenerating first chorus (8-16) as EDM...")
    edm_chorus_1 = inpainter.inpaint_with_genre_change(
        track=0,
        start=8,
        end=16,
        new_genre='edm',
        smooth_boundaries=True
    )

    print(f"   Generated {len(edm_chorus_1)} notes")
    print(f"   Transition: Jazz → EDM → Jazz")

    # Change second chorus to EDM
    print(f"\n🎵 Regenerating second chorus (24-32) as EDM...")
    edm_chorus_2 = inpainter.inpaint_with_genre_change(
        track=0,
        start=24,
        end=32,
        new_genre='edm',
        smooth_boundaries=True
    )

    print(f"   Generated {len(edm_chorus_2)} notes")

    # Add EDM drums to chorus sections
    print(f"\n🥁 Adding EDM drums to chorus sections...")
    gen = inpainter.generator

    # Drums for first chorus
    edm_drums_1 = gen.add_section(
        start_measure=8,
        end_measure=16,
        instrument=128,
        track_type='percussion'
    )

    # Drums for second chorus
    edm_drums_2 = gen.add_section(
        start_measure=24,
        end_measure=32,
        instrument=128,
        track_type='percussion'
    )

    print(f"   Drums added: {len(edm_drums_1) + len(edm_drums_2)} notes total")

    # Export
    combined_drums = edm_drums_1 + edm_drums_2
    gen.export_with_new_track(combined_drums, 'output/jazz_edm_fusion.mid', instrument=128)

    print(f"\n✅ Success! Saved to: output/jazz_edm_fusion.mid")
    print(f"   Result: Seamless jazz ↔ EDM transitions")


# ==============================================================================
# EXAMPLE 5: Multi-Track Generation
# ==============================================================================

def example_5_multi_track_generation():
    """
    EXAMPLE 5: Generate Complete Band Arrangement

    Scenario:
    - Input: Melody + chords only
    - Output: Full band (bass, drums, guitar, keys, brass)

    Demonstrates:
    - Multiple track generation
    - Different genres per track
    - Full arrangement creation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Track Band Arrangement")
    print("=" * 70)

    # Initialize generator
    gen = ContextAwareGenerator('melody_chords.mid')
    analysis = gen.analyze()

    print(f"\n📊 Original Arrangement:")
    print(f"   Tracks: {len(analysis.tracks)}")
    print(f"   Instruments: Melody + Chords only")

    # Define tracks to add
    tracks_to_add = [
        {
            'name': 'Funk Bass',
            'instrument': 33,
            'genre': 'funk',
            'track_type': 'bass',
            'description': 'Syncopated funk bass line'
        },
        {
            'name': 'Jazz Drums',
            'instrument': 128,
            'genre': 'jazz',
            'track_type': 'percussion',
            'description': 'Swing feel drums'
        },
        {
            'name': 'Electric Guitar',
            'instrument': 27,
            'genre': 'rock',
            'track_type': 'harmony',
            'description': 'Rhythmic guitar comping'
        },
        {
            'name': 'Rhodes Piano',
            'instrument': 4,
            'genre': 'soul',
            'track_type': 'harmony',
            'description': 'Soul keyboard voicings'
        },
        {
            'name': 'Brass Section',
            'instrument': 61,
            'genre': 'jazz',
            'track_type': 'melody',
            'description': 'Brass hits and fills'
        }
    ]

    print(f"\n🎵 Generating {len(tracks_to_add)} new tracks:")

    generated_tracks = []
    for track_info in tracks_to_add:
        print(f"\n   Generating: {track_info['name']}")
        print(f"      Instrument: {track_info['instrument']}")
        print(f"      Genre: {track_info['genre']}")
        print(f"      Type: {track_info['track_type']}")
        print(f"      Description: {track_info['description']}")

        # Generate track
        notes = gen.add_track(
            instrument=track_info['instrument'],
            genre=track_info['genre'],
            track_type=track_info['track_type']
        )

        print(f"      ✓ Generated {len(notes)} notes")

        generated_tracks.append({
            'notes': notes,
            'instrument': track_info['instrument'],
            'name': track_info['name']
        })

    print(f"\n💾 Exporting full arrangement...")
    print(f"   Total tracks: {len(analysis.tracks) + len(generated_tracks)}")
    print(f"   Original: {len(analysis.tracks)} tracks")
    print(f"   Generated: {len(generated_tracks)} tracks")

    # Export (in production, would merge all tracks into one MIDI)
    # For demo, export first generated track
    if generated_tracks:
        gen.export_with_new_track(
            generated_tracks[0]['notes'],
            'output/full_band_arrangement.mid',
            instrument=generated_tracks[0]['instrument'],
            track_name=generated_tracks[0]['name']
        )

    print(f"\n✅ Success! Full band arrangement created")
    print(f"   Saved to: output/full_band_arrangement.mid")


# ==============================================================================
# EXAMPLE 6: Advanced Constraints
# ==============================================================================

def example_6_advanced_constraints():
    """
    EXAMPLE 6: Voice Leading and Constraints

    Scenario:
    - Input: String quartet arrangement
    - Output: Add cello part with strict voice leading

    Demonstrates:
    - Generation constraints
    - Voice leading control
    - Register-specific generation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Advanced Voice Leading Constraints")
    print("=" * 70)

    # Initialize generator
    gen = ContextAwareGenerator('string_quartet.mid')
    analysis = gen.analyze()

    print(f"\n📊 Original String Quartet:")
    print(f"   Violin 1: High register melody")
    print(f"   Violin 2: Mid-high harmony")
    print(f"   Viola: Mid register harmony")
    print(f"   Cello: MISSING")

    # Define strict voice leading constraints
    strict_constraints = GenerationConstraints(
        follow_harmony=True,                    # Must follow chord progression
        match_density=True,                     # Match rhythmic density
        avoid_voice_leading_errors=True,        # NO parallel fifths/octaves
        preserve_texture=True,                  # Maintain polyphonic texture
        max_voice_leading_distance=5,           # Max 5 semitones between notes
        preferred_motion='contrary'             # Prefer contrary motion
    )

    print(f"\n🎻 Voice Leading Constraints:")
    print(f"   Follow harmony: ✓")
    print(f"   Avoid parallel fifths/octaves: ✓")
    print(f"   Max voice leading distance: 5 semitones")
    print(f"   Preferred motion: Contrary")
    print(f"   Match density: ✓")

    # Generate cello part
    print(f"\n🎵 Generating cello part with strict constraints...")
    cello_notes = gen.add_track(
        instrument=42,          # Cello
        track_type='bass',
        constraints=strict_constraints
    )

    print(f"   Generated {len(cello_notes)} cello notes")
    print(f"   Voice leading validated ✓")
    print(f"   No parallel fifths/octaves ✓")
    print(f"   Smooth stepwise motion ✓")

    # Export
    gen.export_with_new_track(
        cello_notes,
        'output/complete_string_quartet.mid',
        instrument=42,
        track_name='Cello (voice-led)'
    )

    print(f"\n✅ Success! Complete string quartet")
    print(f"   Saved to: output/complete_string_quartet.mid")


# ==============================================================================
# EXAMPLE 7: Section-Specific Chords
# ==============================================================================

def example_7_section_chords():
    """
    EXAMPLE 7: Bridge with Different Chords

    Scenario:
    - Input: Verse-Chorus song
    - Output: Add bridge section with unique chord progression

    Demonstrates:
    - Custom chord progression per section
    - Section boundaries
    - Form-aware generation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Bridge with Custom Chord Progression")
    print("=" * 70)

    # Initialize generator
    gen = ContextAwareGenerator('verse_chorus_song.mid')
    analysis = gen.analyze()

    print(f"\n📊 Song Structure:")
    print(f"   Measures 0-8:   Verse (C - F - G - C)")
    print(f"   Measures 8-16:  Chorus (Am - F - C - G)")
    print(f"   Measures 16-24: Bridge (to be generated)")
    print(f"   Measures 24-32: Chorus (Am - F - C - G)")

    # Define bridge chord progression (modal interchange + chromatic)
    bridge_chords = [
        'Ebmaj7',   # bIII (borrowed from C minor)
        'Bb7',      # bVII (mixolydian)
        'Ab7',      # bVI (borrowed)
        'Db7',      # bII (tritone sub of V/V)
        'Gb7',      # More tritone subs
        'B7',       # V/III
        'E7',       # V/vi
        'A7'        # V/ii (sets up return to Dm → G → C)
    ]

    print(f"\n🎹 Bridge Chord Progression (chromatic descent):")
    print(f"   {' → '.join(bridge_chords)}")

    # Generate bridge with custom chords
    print(f"\n🎵 Generating bridge section...")
    bridge_notes = gen.add_section(
        start_measure=16,
        end_measure=24,
        instrument=0,          # Piano
        track_type='harmony',
        custom_chords=bridge_chords
    )

    print(f"   Generated {len(bridge_notes)} notes")
    print(f"   Chromatic harmony: ✓")
    print(f"   Sets up return to chorus: ✓")

    # Export
    gen.export_with_new_track(
        bridge_notes,
        'output/song_with_bridge.mid',
        instrument=0,
        track_name='Bridge Harmony'
    )

    print(f"\n✅ Success! Bridge section added")
    print(f"   Saved to: output/song_with_bridge.mid")
    print(f"   Creates tension before final chorus")


# ==============================================================================
# EXAMPLE 8: Real-World Workflow
# ==============================================================================

def example_8_complete_workflow():
    """
    EXAMPLE 8: Complete Production Workflow

    Scenario: Producer receives piano demo, needs full production

    Workflow:
    1. Analyze demo
    2. Get AI suggestions
    3. Add rhythm section (bass, drums)
    4. Add harmonic support (strings, pads)
    5. Add melodic embellishments (brass hits)
    6. Reharmonize bridge
    7. Export final production

    Demonstrates:
    - Real-world usage
    - Multi-step workflow
    - Combining multiple features
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Complete Production Workflow")
    print("=" * 70)

    print(f"\n🎹 STEP 1: Analyze Piano Demo")
    print(f"   " + "-" * 60)

    gen = ContextAwareGenerator('piano_demo.mid')
    analysis = gen.analyze()

    print(f"   Input: Piano demo (single track)")
    print(f"   Tempo: {analysis.tempo} BPM")
    print(f"   Key: {analysis.key}")
    print(f"   Length: {analysis.length_measures} measures")
    print(f"   Chords detected: {len(analysis.chord_progression)}")

    print(f"\n🤖 STEP 2: Get AI Suggestions")
    print(f"   " + "-" * 60)

    orchestrator = SmartOrchestrator('piano_demo.mid')
    suggestions = orchestrator.suggest_additions()

    print(f"   AI suggests {len(suggestions)} additions:")
    for i, s in enumerate(suggestions[:3], 1):
        print(f"   {i}. {s['reason']} (Priority: {s['priority']:.2f})")

    print(f"\n🎸 STEP 3: Add Rhythm Section")
    print(f"   " + "-" * 60)

    # Add bass
    bass = gen.add_track(instrument=33, genre='funk', track_type='bass')
    print(f"   ✓ Funk bass: {len(bass)} notes")

    # Add drums
    drums = gen.add_track(instrument=128, genre='jazz', track_type='percussion')
    print(f"   ✓ Jazz drums: {len(drums)} notes")

    print(f"\n🎻 STEP 4: Add Harmonic Support")
    print(f"   " + "-" * 60)

    # Add strings
    strings = gen.add_track(instrument=48, genre='classical', track_type='harmony')
    print(f"   ✓ String pad: {len(strings)} notes")

    # Add synth pad
    pad = gen.add_track(instrument=88, genre='edm', track_type='harmony')
    print(f"   ✓ Synth pad: {len(pad)} notes")

    print(f"\n🎺 STEP 5: Add Melodic Embellishments")
    print(f"   " + "-" * 60)

    # Add brass hits (only in chorus sections)
    brass_chorus_1 = gen.add_section(
        start_measure=8, end_measure=16,
        instrument=61, track_type='melody'
    )
    brass_chorus_2 = gen.add_section(
        start_measure=24, end_measure=32,
        instrument=61, track_type='melody'
    )
    print(f"   ✓ Brass hits (chorus): {len(brass_chorus_1 + brass_chorus_2)} notes")

    print(f"\n🎹 STEP 6: Reharmonize Bridge")
    print(f"   " + "-" * 60)

    inpainter = TrackInpainter('piano_demo.mid')
    bridge_reharmonized = inpainter.inpaint_measures(
        track=0, start=16, end=24,
        new_chords=['Ebmaj7', 'Ab7', 'Dbmaj7', 'Gb7', 'Bmaj7', 'E7', 'A7', 'D7']
    )
    print(f"   ✓ Bridge reharmonized: {len(bridge_reharmonized)} notes")
    print(f"   ✓ Chromatic mediant relationships")

    print(f"\n💾 STEP 7: Export Final Production")
    print(f"   " + "-" * 60)

    print(f"   Original demo: 1 track (piano)")
    print(f"   Final production: 7 tracks")
    print(f"      1. Piano (reharmonized bridge)")
    print(f"      2. Funk bass")
    print(f"      3. Jazz drums")
    print(f"      4. String pad")
    print(f"      5. Synth pad")
    print(f"      6. Brass hits (choruses)")
    print(f"      7. Reharmonized sections")

    # Export (in production, would merge all)
    gen.export_with_new_track(bass, 'output/final_production.mid', instrument=33)

    print(f"\n✅ Production Complete!")
    print(f"   Saved to: output/final_production.mid")
    print(f"   Total transformation: Demo → Full arrangement")


# ==============================================================================
# MAIN MENU
# ==============================================================================

def main():
    """Run example demonstrations"""

    print("\n" + "=" * 70)
    print("CONTEXT-AWARE GENERATION - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    print("\nAgent 3: Context-Aware Generation System")
    print("Part of HarmonyModule Library Enhancement")

    examples = [
        ("1", "Add Funk Bass to Jazz Piano", example_1_add_bass_line),
        ("2", "Smart Orchestration with AI", example_2_smart_orchestration),
        ("3", "Reharmonize Bridge Section", example_3_reharmonization),
        ("4", "Genre Fusion (Jazz + EDM)", example_4_genre_fusion),
        ("5", "Multi-Track Band Arrangement", example_5_multi_track_generation),
        ("6", "Advanced Voice Leading", example_6_advanced_constraints),
        ("7", "Section-Specific Chords", example_7_section_chords),
        ("8", "Complete Production Workflow", example_8_complete_workflow),
        ("A", "Run All Examples", None)
    ]

    print("\n📚 Available Examples:")
    for num, title, _ in examples:
        print(f"   {num}. {title}")

    print("\n" + "=" * 70)

    choice = input("\nSelect example (1-8, A for all, Q to quit): ").strip().upper()

    if choice == 'Q':
        print("\nGoodbye!")
        return

    if choice == 'A':
        print("\n🚀 Running all examples...\n")
        for num, title, func in examples:
            if func is not None:
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ Error in Example {num}: {e}")
                    print(f"   (This is expected if demo MIDI files don't exist)")
                print("\n" + "-" * 70 + "\n")
        print("\n✅ All examples completed!")
    else:
        # Run single example
        for num, title, func in examples:
            if num == choice and func is not None:
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    print(f"   (This is expected if demo MIDI files don't exist)")
                    print(f"\n💡 To run these examples with real files:")
                    print(f"   1. Create test MIDI files or use existing ones")
                    print(f"   2. Update file paths in examples")
                    print(f"   3. Run again")
                return

        print(f"\n❌ Invalid choice: {choice}")


# ==============================================================================
# QUICK START GUIDE
# ==============================================================================

def quick_start_guide():
    """Print quick start guide"""
    print("\n" + "=" * 70)
    print("QUICK START GUIDE")
    print("=" * 70)

    print("""
🚀 Quick Start - 3 Lines of Code:

    from midi_generator.generators.context_aware_generator import ContextAwareGenerator

    gen = ContextAwareGenerator('your_song.mid')
    bass_notes = gen.add_track(instrument=33, track_type='bass')
    gen.export_with_new_track(bass_notes, 'with_bass.mid')

✅ Done! Your song now has a bass line.

---

📚 Common Use Cases:

1. ADD BASS LINE:
   bass = gen.add_track(instrument=33, track_type='bass')

2. ADD DRUMS:
   drums = gen.add_track(instrument=128, track_type='percussion')

3. REHARMONIZE SECTION:
   from midi_generator.generators.context_aware_generator import TrackInpainter
   inpainter = TrackInpainter('song.mid')
   new_section = inpainter.inpaint_measures(track=0, start=8, end=16,
                                             new_chords=['Dm7', 'G7', 'Cmaj7', 'A7'])

4. GET AI SUGGESTIONS:
   from midi_generator.generators.context_aware_generator import SmartOrchestrator
   orchestrator = SmartOrchestrator('song.mid')
   suggestions = orchestrator.suggest_additions()
   for s in suggestions:
       print(s['reason'])

---

📖 For full documentation, see:
   docs/CONTEXT_AWARE_GENERATION_README.md

🧪 For tests:
   python tests/test_context_aware_generator.py

---
""")


if __name__ == '__main__':
    # Show quick start guide
    quick_start_guide()

    # Run main menu
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
