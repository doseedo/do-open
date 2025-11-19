#!/usr/bin/env python3
"""
Comprehensive Examples for Multi-Genre Arranger

This file demonstrates all features of the track-level genre control system
with practical, real-world examples.

Examples:
1. Jazz-Funk Fusion
2. Latin-Electronic House
3. Hip-Hop Jazz (Jazz-Hop)
4. Big Band + Orchestra
5. Progressive Rock Multi-Genre
6. Afrobeat-Electronic Fusion
7. Blues-Rock-Jazz Trio
8. Experimental Polyrhythmic Fusion
9. Film Score Multi-Style
10. World Music Fusion

Author: Agent 9 - Track-Level Genre Control
Date: 2025
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multi_genre_arranger import (
    MultiGenreArranger,
    HarmonicContext,
    TrackSpec,
    TrackRole,
    SyncStrategy,
    VoiceLeadingPriority,
    RegisterRange,
    create_simple_arrangement
)

try:
    from generators.style_fusion import GENRE_PROFILES
except ImportError:
    print("Warning: GENRE_PROFILES not available. Some examples may not work.")
    GENRE_PROFILES = {}


# ==============================================================================
# EXAMPLE 1: JAZZ-FUNK FUSION
# ==============================================================================

def example_jazz_funk_fusion():
    """
    Classic jazz-funk fusion arrangement

    - Funky bass with syncopation
    - Jazz piano with extended chords
    - Hip-hop influenced drums with laid-back feel
    - Jazz saxophone melody

    Output: jazz_funk_fusion.mid
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Jazz-Funk Fusion")
    print("="*70)

    # Define harmonic structure - jazz ii-V-I progression
    harmonic_context = HarmonicContext(
        chord_progression=['Dm7', 'G7', 'Cmaj7', 'A7'],
        key='C',
        time_signature=(4, 4),
        tempo_bpm=95,
        length_measures=16,
        allow_reharmonization=True  # Allow jazz substitutions
    )

    # Define tracks
    tracks = [
        # Funk bass - driving, syncopated
        TrackSpec(
            track_number=1,
            genre='funk',
            role=TrackRole.BASS,
            instrument=33,  # Electric bass (fingered)
            custom_syncopation=0.9,
            sync_strategy=SyncStrategy.ACCOMPANIMENT_REFERENCE,
            timing_offset_ms=-2.0  # Slightly ahead of beat (tight)
        ),

        # Jazz piano - comping with extensions
        TrackSpec(
            track_number=2,
            genre='jazz',
            role=TrackRole.HARMONY,
            instrument=0,  # Acoustic grand piano
            voice_leading_priority=VoiceLeadingPriority.MODERATE,
            register=RegisterRange.MID
        ),

        # Hip-hop influenced drums
        TrackSpec(
            track_number=3,
            genre='hiphop',
            role=TrackRole.PERCUSSION,
            instrument=128,  # Standard drum kit
            sync_strategy=SyncStrategy.LOOSE_POCKET,
            timing_offset_ms=8.0,  # Laid-back feel
            humanize_amount=0.06
        ),

        # Jazz saxophone melody
        TrackSpec(
            track_number=4,
            genre='jazz',
            role=TrackRole.MELODY,
            instrument=64,  # Soprano sax
            register=RegisterRange.HIGH_MID,
            voice_leading_priority=VoiceLeadingPriority.LOOSE
        )
    ]

    # Create arranger
    arranger = MultiGenreArranger()

    # Analyze compatibility
    print("\nAnalyzing genre compatibility...")
    compatibility = arranger.analyze_arrangement_compatibility(tracks)
    print(f"Overall compatibility: {compatibility['overall_compatibility']:.2f}")

    if compatibility['recommendations']:
        print("\nRecommendations:")
        for rec in compatibility['recommendations'][:3]:
            print(f"  • {rec}")

    # Generate arrangement
    print("\nGenerating arrangement...")
    result = arranger.arrange(harmonic_context, tracks, auto_optimize=True)

    # Display results
    print(f"\nGenerated {len(result['tracks'])} tracks:")
    for track in result['tracks']:
        pitch_range = track.get_pitch_range()
        time_range = track.get_time_range()
        print(f"  Track {track.track_number} ({track.spec.genre} {track.spec.role.value}):")
        print(f"    - Notes: {len(track.notes)}")
        print(f"    - Pitch range: {pitch_range[0]}-{pitch_range[1]}")
        print(f"    - Time range: {time_range[0]:.1f}-{time_range[1]:.1f}s")

    # Export to MIDI
    arranger.export_to_midi(result, 'jazz_funk_fusion.mid')
    print("\n✓ Exported to: jazz_funk_fusion.mid")

    return result


# ==============================================================================
# EXAMPLE 2: LATIN-ELECTRONIC HOUSE
# ==============================================================================

def example_latin_house_fusion():
    """
    Latin house fusion - combines latin percussion with electronic production

    - Electronic sub-bass
    - Latin percussion patterns
    - Electronic pad
    - Latin melodic instrument (flute)

    Output: latin_house.mid
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Latin-Electronic House Fusion")
    print("="*70)

    harmonic_context = HarmonicContext(
        chord_progression=['Am', 'F', 'C', 'G'],
        key='Am',
        time_signature=(4, 4),
        tempo_bpm=124,
        length_measures=32,
        harmonic_rhythm_pattern=[4, 4, 4, 4],  # One chord every 4 measures
        allow_reharmonization=False  # Keep it simple for house
    )

    tracks = [
        # Electronic sub-bass (four-on-floor foundation)
        TrackSpec(
            track_number=1,
            genre='electronic',
            role=TrackRole.BASS,
            instrument=38,  # Synth bass
            register=RegisterRange.SUB_BASS,
            sync_strategy=SyncStrategy.STRICT_GRID,
            velocity_range=(90, 110)
        ),

        # Latin percussion (conga, bongo patterns)
        TrackSpec(
            track_number=2,
            genre='latin',
            role=TrackRole.PERCUSSION,
            instrument=128,
            sync_strategy=SyncStrategy.STRICT_GRID,  # Lock to grid with electronic
            custom_syncopation=0.8
        ),

        # Electronic pad (atmospheric)
        TrackSpec(
            track_number=3,
            genre='electronic',
            role=TrackRole.PAD,
            instrument=88,  # New age pad
            register=RegisterRange.MID,
            velocity_range=(50, 70)
        ),

        # Latin flute melody
        TrackSpec(
            track_number=4,
            genre='latin',
            role=TrackRole.MELODY,
            instrument=73,  # Flute
            register=RegisterRange.HIGH_MID,
            voice_leading_priority=VoiceLeadingPriority.MODERATE
        )
    ]

    arranger = MultiGenreArranger()

    print("\nAnalyzing compatibility...")
    compatibility = arranger.analyze_arrangement_compatibility(tracks)
    print(f"Overall compatibility: {compatibility['overall_compatibility']:.2f}")

    print("\nGenerating Latin house arrangement...")
    result = arranger.arrange(harmonic_context, tracks)

    print(f"\n✓ Generated {len(result['tracks'])} tracks")

    arranger.export_to_midi(result, 'latin_house.mid')
    print("✓ Exported to: latin_house.mid")

    return result


# ==============================================================================
# EXAMPLE 3: HIP-HOP JAZZ (JAZZ-HOP)
# ==============================================================================

def example_jazz_hop():
    """
    Jazz-hop arrangement - hip-hop beats with jazz harmony

    - Hip-hop drums with boom-bap pattern
    - Jazz bass (walking bass meets hip-hop)
    - Jazz piano with hip-hop rhythmic approach
    - Sampled/lo-fi texture

    Output: jazz_hop.mid
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Jazz-Hop (Hip-Hop + Jazz)")
    print("="*70)

    harmonic_context = HarmonicContext(
        chord_progression=['Cmaj7', 'Am7', 'Fmaj7', 'Fm7'],
        key='C',
        time_signature=(4, 4),
        tempo_bpm=88,  # Slow hip-hop tempo
        length_measures=16,
        allow_reharmonization=True
    )

    tracks = [
        # Hip-hop drums (boom-bap)
        TrackSpec(
            track_number=1,
            genre='hiphop',
            role=TrackRole.PERCUSSION,
            instrument=128,
            sync_strategy=SyncStrategy.LOOSE_POCKET,
            custom_swing_factor=0.58,  # J Dilla swing
            timing_offset_ms=12.0,
            humanize_amount=0.08
        ),

        # Jazz walking bass (with hip-hop feel)
        TrackSpec(
            track_number=2,
            genre='jazz',
            role=TrackRole.BASS,
            instrument=32,  # Acoustic bass
            sync_strategy=SyncStrategy.ACCOMPANIMENT_REFERENCE,
            timing_offset_ms=10.0  # Laid-back with drums
        ),

        # Jazz piano (sparse comping)
        TrackSpec(
            track_number=3,
            genre='jazz',
            role=TrackRole.HARMONY,
            instrument=0,
            voice_leading_priority=VoiceLeadingPriority.MODERATE,
            velocity_range=(55, 85)  # Lo-fi dynamics
        )
    ]

    arranger = MultiGenreArranger()

    print("\nGenerating jazz-hop arrangement...")
    result = arranger.arrange(harmonic_context, tracks)

    print(f"\n✓ Generated {len(result['tracks'])} tracks")

    arranger.export_to_midi(result, 'jazz_hop.mid')
    print("✓ Exported to: jazz_hop.mid")

    return result


# ==============================================================================
# EXAMPLE 4: BIG BAND + ORCHESTRA
# ==============================================================================

def example_big_band_orchestra():
    """
    Symphonic jazz - big band meets orchestra

    - Jazz rhythm section (bass, drums, piano)
    - Jazz brass section (trumpet, trombone)
    - Classical strings (violin, viola, cello)
    - Both styles sharing harmonic structure

    Output: symphonic_jazz.mid
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Big Band + Orchestra")
    print("="*70)

    harmonic_context = HarmonicContext(
        chord_progression=['Cmaj7', 'A7#9', 'Dm7', 'G7alt', 'Cmaj7'],
        key='C',
        time_signature=(4, 4),
        tempo_bpm=160,  # Up-tempo swing
        length_measures=32,
        allow_reharmonization=True
    )

    tracks = [
        # Jazz rhythm section
        TrackSpec(1, 'jazz', TrackRole.BASS, 32),
        TrackSpec(2, 'jazz', TrackRole.PERCUSSION, 128),
        TrackSpec(3, 'jazz', TrackRole.HARMONY, 0,
                 register=RegisterRange.MID),

        # Jazz brass (rhythmic accents)
        TrackSpec(4, 'jazz', TrackRole.RHYTHMIC_ACCENT, 56,  # Trumpet
                 register=RegisterRange.HIGH_MID,
                 velocity_range=(85, 110)),
        TrackSpec(5, 'jazz', TrackRole.RHYTHMIC_ACCENT, 57,  # Trombone
                 register=RegisterRange.LOW_MID,
                 velocity_range=(80, 105)),

        # Classical strings (if available)
        # Note: Using 'jazz' as fallback if 'classical' not in GENRE_PROFILES
        TrackSpec(6, 'jazz', TrackRole.PAD, 48,  # String ensemble
                 voice_leading_priority=VoiceLeadingPriority.STRICT,
                 register=RegisterRange.MID,
                 velocity_range=(60, 85)),
        TrackSpec(7, 'jazz', TrackRole.COUNTER_MELODY, 40,  # Violin
                 voice_leading_priority=VoiceLeadingPriority.STRICT,
                 register=RegisterRange.HIGH_MID)
    ]

    arranger = MultiGenreArranger()

    print("\nGenerating symphonic jazz arrangement...")
    result = arranger.arrange(harmonic_context, tracks)

    print(f"\n✓ Generated {len(result['tracks'])} tracks")
    print(f"Reference track: {result['metadata']['reference_track']}")

    arranger.export_to_midi(result, 'symphonic_jazz.mid')
    print("✓ Exported to: symphonic_jazz.mid")

    return result


# ==============================================================================
# EXAMPLE 5: PROGRESSIVE MULTI-GENRE
# ==============================================================================

def example_progressive_fusion():
    """
    Progressive arrangement with genre changes by section

    Section A (measures 1-8): Jazz
    Section B (measures 9-16): Funk
    Section C (measures 17-24): Electronic

    Demonstrates how to create evolving arrangements
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Progressive Multi-Genre Evolution")
    print("="*70)

    # This example shows the concept - full implementation would
    # generate sections separately and concatenate

    harmonic_context = HarmonicContext(
        chord_progression=['Dm7', 'G7', 'Cmaj7', 'Fmaj7'],
        key='C',
        time_signature=(7, 8),  # Progressive odd meter
        tempo_bpm=140,
        length_measures=24
    )

    # Section A tracks (measures 1-8): Jazz
    tracks_jazz = [
        TrackSpec(1, 'jazz', TrackRole.BASS, 32),
        TrackSpec(2, 'jazz', TrackRole.HARMONY, 0),
        TrackSpec(3, 'jazz', TrackRole.PERCUSSION, 128)
    ]

    # Generate jazz section
    print("\nGenerating Jazz section (measures 1-8)...")
    arranger = MultiGenreArranger()
    result_jazz = arranger.arrange(harmonic_context, tracks_jazz)

    print(f"✓ Jazz section: {len(result_jazz['tracks'])} tracks")

    # In a full implementation, would generate other sections
    # and concatenate them

    arranger.export_to_midi(result_jazz, 'progressive_fusion.mid')
    print("\n✓ Exported to: progressive_fusion.mid")

    return result_jazz


# ==============================================================================
# EXAMPLE 6: QUICK ARRANGEMENTS
# ==============================================================================

def example_quick_arrangements():
    """
    Demonstrate the quick arrangement helper function

    Creates arrangements with minimal code
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Quick Arrangements")
    print("="*70)

    # Example 6a: Three-genre fusion
    print("\nCreating jazz-funk-electronic fusion...")
    result1 = create_simple_arrangement(
        genres=['jazz', 'funk', 'electronic'],
        key='Dm',
        tempo=110,
        measures=8
    )
    print(f"✓ Generated {len(result1['tracks'])} tracks")

    # Example 6b: Different key and tempo
    print("\nCreating blues-rock arrangement...")
    if 'blues' in GENRE_PROFILES:
        result2 = create_simple_arrangement(
            genres=['blues', 'jazz'],
            key='E',
            tempo=140,
            measures=12
        )
        print(f"✓ Generated {len(result2['tracks'])} tracks")

    return result1


# ==============================================================================
# EXAMPLE 7: CUSTOM TIMING FEELS
# ==============================================================================

def example_custom_timing():
    """
    Demonstrate custom timing offsets for specific feels

    - Drums: On the beat (reference)
    - Bass: Slightly ahead (driving)
    - Piano: Slightly behind (laid-back)
    - Melody: Natural (no offset)
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Custom Timing Feels")
    print("="*70)

    harmonic_context = HarmonicContext(
        chord_progression=['C7', 'F7', 'C7', 'G7'],
        key='C',
        time_signature=(4, 4),
        tempo_bpm=120,
        length_measures=12
    )

    tracks = [
        # Drums - reference timing
        TrackSpec(
            track_number=1,
            genre='funk',
            role=TrackRole.PERCUSSION,
            instrument=128,
            timing_offset_ms=0.0,  # On the beat
            humanize_amount=0.02  # Tight
        ),

        # Bass - driving (ahead of beat)
        TrackSpec(
            track_number=2,
            genre='funk',
            role=TrackRole.BASS,
            instrument=33,
            timing_offset_ms=-3.0,  # 3ms ahead (driving)
            humanize_amount=0.03
        ),

        # Piano - laid-back (behind beat)
        TrackSpec(
            track_number=3,
            genre='jazz',
            role=TrackRole.HARMONY,
            instrument=0,
            timing_offset_ms=8.0,  # 8ms behind (laid-back)
            humanize_amount=0.06
        ),

        # Melody - natural
        TrackSpec(
            track_number=4,
            genre='jazz',
            role=TrackRole.MELODY,
            instrument=64,
            timing_offset_ms=0.0,
            humanize_amount=0.05
        )
    ]

    arranger = MultiGenreArranger()

    print("\nGenerating arrangement with custom timing feels...")
    result = arranger.arrange(harmonic_context, tracks)

    print(f"\n✓ Generated {len(result['tracks'])} tracks")
    print("  Timing offsets:")
    for track in result['tracks']:
        offset = track.spec.timing_offset_ms
        description = "on beat" if offset == 0 else f"{abs(offset):.1f}ms {'ahead' if offset < 0 else 'behind'}"
        print(f"    Track {track.track_number}: {description}")

    arranger.export_to_midi(result, 'custom_timing.mid')
    print("\n✓ Exported to: custom_timing.mid")

    return result


# ==============================================================================
# EXAMPLE 8: REGISTER ALLOCATION DEMO
# ==============================================================================

def example_register_allocation():
    """
    Demonstrate intelligent register allocation

    Creates full-spectrum arrangement with proper frequency distribution
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Register Allocation Demo")
    print("="*70)

    harmonic_context = HarmonicContext(
        chord_progression=['Cmaj7', 'Am7', 'Dm7', 'G7'],
        key='C',
        time_signature=(4, 4),
        tempo_bpm=100,
        length_measures=8
    )

    tracks = [
        # Sub-bass (synth)
        TrackSpec(1, 'electronic', TrackRole.BASS, 38,
                 register=RegisterRange.SUB_BASS),

        # Bass (acoustic)
        TrackSpec(2, 'jazz', TrackRole.BASS, 32,
                 register=RegisterRange.BASS),

        # Low-mid harmony
        TrackSpec(3, 'jazz', TrackRole.HARMONY, 0,
                 register=RegisterRange.LOW_MID),

        # Mid melody
        TrackSpec(4, 'jazz', TrackRole.MELODY, 64,
                 register=RegisterRange.MID),

        # High-mid ornaments
        TrackSpec(5, 'jazz', TrackRole.ORNAMENT, 11,  # Vibraphone
                 register=RegisterRange.HIGH_MID),
    ]

    arranger = MultiGenreArranger()

    print("\nGenerating full-spectrum arrangement...")
    result = arranger.arrange(harmonic_context, tracks)

    print(f"\n✓ Generated {len(result['tracks'])} tracks")
    print("\nRegister distribution:")

    register_usage = result['metadata']['register_usage']
    for register, usage in register_usage.items():
        bar = "█" * int(usage * 20)
        print(f"  {register.name:12s} [{bar:20s}] {usage:.2f}")

    arranger.export_to_midi(result, 'register_demo.mid')
    print("\n✓ Exported to: register_demo.mid")

    return result


# ==============================================================================
# EXAMPLE 9: COMPATIBILITY ANALYSIS
# ==============================================================================

def example_compatibility_analysis():
    """
    Demonstrate genre compatibility analysis

    Tests various genre combinations and displays compatibility scores
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Genre Compatibility Analysis")
    print("="*70)

    from core.multi_genre_arranger import GenreCompatibilityAnalyzer

    analyzer = GenreCompatibilityAnalyzer()

    # Test various combinations
    combinations = [
        ('jazz', 'funk'),
        ('jazz', 'hiphop'),
        ('electronic', 'jazz'),
        ('funk', 'hiphop')
    ]

    if 'latin' in GENRE_PROFILES:
        combinations.append(('jazz', 'latin'))
    if 'blues' in GENRE_PROFILES:
        combinations.append(('blues', 'jazz'))

    print("\nPairwise Compatibility Scores:")
    print("-" * 70)

    for genre_a, genre_b in combinations:
        if genre_a in GENRE_PROFILES and genre_b in GENRE_PROFILES:
            scores = analyzer.calculate_compatibility(genre_a, genre_b)

            print(f"\n{genre_a.upper()} + {genre_b.upper()}")
            print(f"  Overall:   {scores['overall']:.2f} {'✓' if scores['overall'] > 0.6 else '⚠'}")
            print(f"  Rhythmic:  {scores['rhythmic']:.2f}")
            print(f"  Harmonic:  {scores['harmonic']:.2f}")
            print(f"  Timbral:   {scores['timbral']:.2f}")
            print(f"  Cultural:  {scores['cultural']:.2f}")

    # Multi-genre analysis
    print("\n" + "-" * 70)
    print("\nMulti-Genre Analysis: Jazz + Funk + Hip-Hop")
    print("-" * 70)

    genres = ['jazz', 'funk', 'hiphop']
    analysis = analyzer.analyze_multi_genre_compatibility(genres)

    print(f"\nOverall Compatibility: {analysis['overall_compatibility']:.2f}")

    if analysis['potential_issues']:
        print("\nPotential Issues:")
        for issue in analysis['potential_issues']:
            print(f"  ⚠ {issue}")

    if analysis['recommendations']:
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  → {rec}")


# ==============================================================================
# EXAMPLE 10: ALL FEATURES DEMO
# ==============================================================================

def example_all_features():
    """
    Comprehensive example using all features

    - Custom genre profiles
    - All sync strategies
    - Voice leading priorities
    - Register allocation
    - Custom timing
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: All Features Demo")
    print("="*70)

    harmonic_context = HarmonicContext(
        chord_progression=['Cmaj7', 'Am7', 'Fmaj7', 'G7'],
        key='C',
        time_signature=(4, 4),
        tempo_bpm=108,
        length_measures=16,
        harmonic_rhythm_pattern=[2, 2, 2, 2],
        allow_reharmonization=True
    )

    tracks = [
        # Percussion - reference track
        TrackSpec(
            track_number=1,
            genre='funk',
            role=TrackRole.PERCUSSION,
            instrument=128,
            sync_strategy=SyncStrategy.ACCOMPANIMENT_REFERENCE,
            timing_offset_ms=0.0,
            humanize_amount=0.03
        ),

        # Bass - funk with custom settings
        TrackSpec(
            track_number=2,
            genre='funk',
            role=TrackRole.BASS,
            instrument=33,
            register=RegisterRange.BASS,
            sync_strategy=SyncStrategy.ACCOMPANIMENT_REFERENCE,
            custom_syncopation=0.85,
            timing_offset_ms=-2.0,
            humanize_amount=0.04,
            velocity_range=(75, 105)
        ),

        # Piano - jazz harmony
        TrackSpec(
            track_number=3,
            genre='jazz',
            role=TrackRole.HARMONY,
            instrument=0,
            register=RegisterRange.MID,
            voice_leading_priority=VoiceLeadingPriority.MODERATE,
            sync_strategy=SyncStrategy.GENRE_WEIGHTED_TIMING,
            custom_swing_factor=0.67,
            timing_offset_ms=5.0,
            humanize_amount=0.06
        ),

        # Electronic pad
        TrackSpec(
            track_number=4,
            genre='electronic',
            role=TrackRole.PAD,
            instrument=88,
            register=RegisterRange.HIGH_MID,
            sync_strategy=SyncStrategy.STRICT_GRID,
            velocity_range=(40, 60)
        ),

        # Melody - hip-hop influenced
        TrackSpec(
            track_number=5,
            genre='hiphop',
            role=TrackRole.MELODY,
            instrument=81,  # Lead synth
            register=RegisterRange.MID,
            voice_leading_priority=VoiceLeadingPriority.LOOSE,
            timing_offset_ms=10.0,
            humanize_amount=0.08
        )
    ]

    arranger = MultiGenreArranger()

    # Analyze
    print("\nStep 1: Analyzing compatibility...")
    compatibility = arranger.analyze_arrangement_compatibility(tracks)
    print(f"  Overall compatibility: {compatibility['overall_compatibility']:.2f}")

    # Generate
    print("\nStep 2: Generating arrangement...")
    result = arranger.arrange(harmonic_context, tracks, auto_optimize=True)

    # Display results
    print(f"\nStep 3: Results")
    print(f"  Generated tracks: {len(result['tracks'])}")
    print(f"  Reference track: {result['metadata']['reference_track']}")

    print("\n  Track details:")
    for track in result['tracks']:
        print(f"    Track {track.track_number}: {track.spec.genre} {track.spec.role.value}")
        print(f"      Notes: {len(track.notes)}, Range: {track.get_pitch_range()}")

    # Export
    print("\nStep 4: Exporting...")
    arranger.export_to_midi(result, 'all_features_demo.mid')
    print("  ✓ Exported to: all_features_demo.mid")

    return result


# ==============================================================================
# MAIN - RUN ALL EXAMPLES
# ==============================================================================

def main():
    """Run all examples"""

    print("\n" + "="*70)
    print("MULTI-GENRE ARRANGER - COMPREHENSIVE EXAMPLES")
    print("Agent 9: Track-Level Genre Control")
    print("="*70)

    examples = [
        ("1. Jazz-Funk Fusion", example_jazz_funk_fusion),
        ("2. Latin-Electronic House", example_latin_house_fusion),
        ("3. Jazz-Hop", example_jazz_hop),
        ("4. Big Band + Orchestra", example_big_band_orchestra),
        ("5. Progressive Fusion", example_progressive_fusion),
        ("6. Quick Arrangements", example_quick_arrangements),
        ("7. Custom Timing Feels", example_custom_timing),
        ("8. Register Allocation", example_register_allocation),
        ("9. Compatibility Analysis", example_compatibility_analysis),
        ("10. All Features Demo", example_all_features)
    ]

    print("\nAvailable examples:")
    for name, _ in examples:
        print(f"  {name}")

    print("\nRunning all examples...\n")

    results = {}
    for name, func in examples:
        try:
            result = func()
            results[name] = result
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"Completed {len(results)}/{len(examples)} examples")
    print("="*70)

    print("\nGenerated MIDI files:")
    midi_files = [
        'jazz_funk_fusion.mid',
        'latin_house.mid',
        'jazz_hop.mid',
        'symphonic_jazz.mid',
        'progressive_fusion.mid',
        'custom_timing.mid',
        'register_demo.mid',
        'all_features_demo.mid'
    ]

    for filename in midi_files:
        print(f"  • {filename}")

    print("\nLoad these files in your DAW to hear the results!")


if __name__ == "__main__":
    main()
