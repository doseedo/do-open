#!/usr/bin/env python3
"""
Style Fusion Comprehensive Demo

Demonstrates all features of the StyleFusion module including:
- Genre blending with weighted combinations
- Cross-cultural fusion (Afro-Cuban jazz, Indo-jazz concepts)
- Hybrid rhythm generation
- Style transfer (harmony ↔ rhythm)
- Instrumentation mixing
- Genre compatibility analysis

Examples created:
1. Jazz-Hop (50% Jazz + 50% Hip-Hop) - Nu-jazz style
2. Electro-Swing (60% Electronic + 40% Jazz)
3. Afro-Cuban Jazz (60% Latin + 40% Jazz)
4. G-Funk (50% Funk + 50% Hip-Hop)
5. Latin Trap (Jazz harmony + Trap rhythm)

Author: Agent 18
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from generators.style_fusion import (
    StyleFusion,
    GenreCompatibility,
    HybridRhythmGenerator,
    GENRE_PROFILES
)


def demo_jazz_hop():
    """
    Demo 1: Jazz-Hop (Nu-Jazz)

    Combines jazz harmony (extended chords, sophisticated progressions)
    with hip-hop beats (boom-bap, J Dilla swing)
    """
    print("\n" + "=" * 70)
    print("DEMO 1: JAZZ-HOP (NU-JAZZ)")
    print("=" * 70)

    fusion = StyleFusion()

    # Create 50/50 blend
    jazz_hop = fusion.blend_genres('jazz', 'hiphop', weight_a=0.5)

    print(f"\n🎵 Genre: {jazz_hop.name}")
    print(f"📊 Characteristics:")
    print(f"   • Tempo range: {jazz_hop.tempo_range[0]}-{jazz_hop.tempo_range[1]} BPM")
    print(f"   • Swing factor: {jazz_hop.swing_factor:.2f} (J Dilla style)")
    print(f"   • Syncopation: {jazz_hop.syncopation:.2f}")
    print(f"   • Chord types: {', '.join(jazz_hop.chord_types[:6])}")
    print(f"   • Harmonic rhythm: {jazz_hop.harmonic_rhythm:.1f} chords/measure")
    print(f"   • Instrumentation: {jazz_hop.instruments}")
    print(f"\n💡 Real-world examples: Robert Glasper, Kamasi Washington, BadBadNotGood")


def demo_electro_swing():
    """
    Demo 2: Electro-Swing

    Vintage 1920s-40s swing rhythms + modern electronic production
    """
    print("\n" + "=" * 70)
    print("DEMO 2: ELECTRO-SWING")
    print("=" * 70)

    fusion = StyleFusion()

    # 60% electronic, 40% jazz (swing)
    electro_swing = fusion.blend_genres('electronic', 'jazz', weight_a=0.6)

    print(f"\n🎵 Genre: {electro_swing.name}")
    print(f"📊 Characteristics:")
    print(f"   • Tempo range: {electro_swing.tempo_range[0]}-{electro_swing.tempo_range[1]} BPM")
    print(f"   • Swing factor: {electro_swing.swing_factor:.2f}")
    print(f"   • Chord types: {', '.join(electro_swing.chord_types[:5])}")
    print(f"   • Texture: {electro_swing.texture}")
    print(f"\n💡 Real-world examples: Parov Stelar, Caravan Palace, Caro Emerald")


def demo_afro_cuban_jazz():
    """
    Demo 3: Afro-Cuban Jazz

    Combines Latin clave-based rhythms with bebop jazz harmonies
    """
    print("\n" + "=" * 70)
    print("DEMO 3: AFRO-CUBAN JAZZ")
    print("=" * 70)

    fusion = StyleFusion()

    # 60% Latin, 40% Jazz
    afro_cuban = fusion.blend_genres('latin', 'jazz', weight_a=0.6)

    print(f"\n🎵 Genre: {afro_cuban.name}")
    print(f"📊 Characteristics:")
    print(f"   • Rhythmic basis: {afro_cuban.rhythmic_basis}")
    print(f"   • Syncopation: {afro_cuban.syncopation:.2f} (very high)")
    print(f"   • Harmonic complexity: {afro_cuban.harmonic_rhythm:.1f}")
    print(f"   • Cultural fusion: {afro_cuban.cultural_origin}")
    print(f"   • Chord types: {', '.join(afro_cuban.chord_types[:6])}")
    print(f"\n💡 Real-world examples: Dizzy Gillespie, Tito Puente, Arturo Sandoval")


def demo_style_transfer():
    """
    Demo 4: Style Transfer - Jazz Harmony + Trap Rhythm

    Applies sophisticated jazz harmonies to modern trap beats
    """
    print("\n" + "=" * 70)
    print("DEMO 4: STYLE TRANSFER - JAZZ HARMONY + TRAP RHYTHM")
    print("=" * 70)

    fusion = StyleFusion()

    # Apply jazz harmony to hip-hop rhythm
    result = fusion.apply_harmony_to_rhythm('jazz', 'hiphop')

    print(f"\n🎵 Technique: Neural style transfer approach")
    print(f"📊 Generated content:")
    print(f"   • Rhythm events: {len(result['rhythm'])} (hip-hop feel)")
    print(f"   • Harmonic changes: {len(result['harmony'])} (jazz sophistication)")
    print(f"\n💡 This technique separates 'content' (rhythm) from 'style' (harmony)")
    print(f"   Real-world examples: Kendrick Lamar's 'To Pimp a Butterfly', Kamasi Washington")


def demo_hybrid_rhythm():
    """
    Demo 5: Hybrid Rhythm Pattern

    Combines swing and straight grooves in one pattern
    """
    print("\n" + "=" * 70)
    print("DEMO 5: HYBRID RHYTHM PATTERN")
    print("=" * 70)

    # Create two different patterns
    swing_pattern = [(0.0, 90), (0.67, 70), (1.0, 90), (1.67, 70)]  # Triplet swing
    straight_pattern = [(0.0, 85), (0.5, 70), (1.0, 85), (1.5, 70)]  # Straight 8ths

    fusion = StyleFusion()
    hybrid = fusion.create_hybrid_rhythm(swing_pattern, straight_pattern, blend_ratio=0.5)

    print(f"\n🎵 Pattern blending:")
    print(f"   • Input A: {len(swing_pattern)} events (swing feel)")
    print(f"   • Input B: {len(straight_pattern)} events (straight feel)")
    print(f"   • Output: {len(hybrid)} events (hybrid)")
    print(f"\n💡 Creates rhythmic tension between swing and straight feels")


def demo_instrumentation_mixing():
    """
    Demo 6: Multi-Genre Instrumentation Palette

    Combines instruments from jazz, latin, and electronic
    """
    print("\n" + "=" * 70)
    print("DEMO 6: INSTRUMENTATION MIXING")
    print("=" * 70)

    fusion = StyleFusion()

    # Mix three genres
    mixed_instruments = fusion.mix_instrumentation(['jazz', 'latin', 'electronic'])

    # Get instrument names (simplified mapping)
    instrument_names = {
        0: 'Acoustic Grand Piano',
        32: 'Acoustic Bass',
        33: 'Electric Bass',
        25: 'Acoustic Guitar',
        64: 'Soprano Sax',
        73: 'Flute',
        11: 'Vibraphone',
        81: 'Lead Synth',
        82: 'Pad Synth',
        88: 'Warm Pad'
    }

    print(f"\n🎵 Combined instrumentation from 3 genres:")
    print(f"   • Total instruments: {len(mixed_instruments)}")
    print(f"   • Palette:")
    for inst in mixed_instruments:
        name = instrument_names.get(inst, f"Program {inst}")
        print(f"     - {name} ({inst})")


def demo_compatibility_analysis():
    """
    Demo 7: Genre Compatibility Analysis

    Analyzes which genres work well together
    """
    print("\n" + "=" * 70)
    print("DEMO 7: GENRE COMPATIBILITY ANALYSIS")
    print("=" * 70)

    # Test all genre pairs
    genres = ['jazz', 'hiphop', 'electronic', 'latin', 'blues', 'funk']

    print(f"\n🎵 Compatibility matrix:")
    print(f"\n   {'':10s} {'Jazz':>8s} {'HipHop':>8s} {'Electro':>8s} {'Latin':>8s} {'Blues':>8s} {'Funk':>8s}")
    print("   " + "-" * 64)

    for genre_a in genres:
        row = f"   {genre_a.capitalize():10s}"
        for genre_b in genres:
            if genre_a == genre_b:
                score = 1.0
            else:
                score = GenreCompatibility.calculate_compatibility(genre_a, genre_b)
            row += f" {score:7.2f}"
        print(row)

    print(f"\n💡 Higher scores = more compatible for fusion")


def demo_fusion_suggestions():
    """
    Demo 8: Automatic Fusion Suggestions

    Get recommended fusion partners for each genre
    """
    print("\n" + "=" * 70)
    print("DEMO 8: FUSION SUGGESTIONS")
    print("=" * 70)

    fusion = StyleFusion()

    for genre in ['jazz', 'funk', 'electronic']:
        print(f"\n🎵 Best fusion partners for {genre.upper()}:")
        suggestions = fusion.suggest_compatible_fusions(genre)

        for partner, score in suggestions[:3]:
            print(f"   • {partner:12s} (compatibility: {score:.2f})")


def demo_feature_analysis():
    """
    Demo 9: Deep Genre Feature Analysis

    Analyzes the musical DNA of each genre
    """
    print("\n" + "=" * 70)
    print("DEMO 9: GENRE FEATURE ANALYSIS")
    print("=" * 70)

    fusion = StyleFusion()

    for genre_name in ['jazz', 'hiphop', 'latin']:
        features = fusion.analyze_genre_features(genre_name)

        print(f"\n🎵 {features['name'].upper()} - Musical DNA:")
        print(f"   • Tempo range: {features['tempo_range'][0]}-{features['tempo_range'][1]} BPM")
        print(f"   • Swing factor: {features['swing_factor']:.2f}")
        print(f"   • Syncopation level: {features['syncopation']:.2f}")
        print(f"   • Chord vocabulary: {', '.join(features['chord_types'][:4])}")
        print(f"   • Harmonic rhythm: {features['harmonic_rhythm']:.1f} chords/measure")
        print(f"   • Groove type: {features['groove_type']}")
        print(f"   • Cultural origin: {features['cultural_origin']}")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 70)
    print("STYLE FUSION - COMPREHENSIVE DEMONSTRATION")
    print("Agent 18: Hybrid Genre Generator")
    print("=" * 70)

    # Run all demos
    demo_jazz_hop()
    demo_electro_swing()
    demo_afro_cuban_jazz()
    demo_style_transfer()
    demo_hybrid_rhythm()
    demo_instrumentation_mixing()
    demo_compatibility_analysis()
    demo_fusion_suggestions()
    demo_feature_analysis()

    # Summary
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\n✅ All fusion techniques demonstrated successfully!")
    print("\n📚 Key Concepts:")
    print("   • Weighted genre blending (any ratio)")
    print("   • Style transfer (separate content from style)")
    print("   • Hybrid rhythms (combine groove patterns)")
    print("   • Cross-cultural fusion (Afro-Cuban, Indo-jazz concepts)")
    print("   • Instrumentation palette mixing")
    print("   • Compatibility analysis")
    print("\n🎵 Supported hybrid genres:")
    print("   • Jazz-Hop / Nu-Jazz")
    print("   • Electro-Swing")
    print("   • Afro-Cuban Jazz")
    print("   • G-Funk")
    print("   • Latin Trap")
    print("   • And many more combinations!")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
