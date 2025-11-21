#!/usr/bin/env python3
"""
Hip-Hop Generator Demo

Demonstrates the capabilities of the hip-hop music generator across
all sub-genres with various production techniques.

Examples:
- Boom Bap (90s Golden Age)
- Trap (Modern)
- Lo-Fi Hip-Hop
- Drill (Chicago/UK)
- Conscious Rap
- G-Funk (West Coast)

Author: Agent 41 - Hip-Hop Module
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genres.hiphop import (
    HipHopGenerator,
    HipHopStyle,
    BoomBapDrums,
    TrapDrums,
    DrillDrums,
    LoFiDrums,
    Bass808Engine,
    SampleChopper,
    HipHopHarmony,
    SampleSliceMode,
    apply_mpc_swing
)


def demo_boom_bap():
    """
    Demonstrate 90s Boom Bap style.

    Characteristics:
    - 85-95 BPM
    - Hard-hitting MPC-sampled drums
    - Prominent kick and snare
    - MPC swing (54%)
    - Minimal melodic elements
    """
    print("\n" + "=" * 70)
    print("BOOM BAP (90s Golden Age)")
    print("=" * 70)
    print("Artists: Wu-Tang Clan, Nas, A Tribe Called Quest, Gang Starr")
    print("Production: SP-1200, MPC60, E-mu SP-12")

    generator = HipHopGenerator(
        style=HipHopStyle.BOOM_BAP,
        tempo=92,
        key_root=57,  # A
        swing_amount=0.54  # Classic MPC swing
    )

    # Generate 16-bar loop (verse section)
    beat = generator.generate_beat(bars=16, include_808=True, include_samples=True)

    print(f"\nGenerated 16-bar boom bap beat:")
    print(f"  Tempo: 92 BPM")
    print(f"  Key: A minor")
    print(f"  Swing: 54% (MPC)")
    print(f"\nTrack breakdown:")
    for track_name, notes in beat.items():
        print(f"  - {track_name.upper()}: {len(notes)} notes")

    # Show chord progression
    progression = HipHopHarmony.get_progression(HipHopStyle.BOOM_BAP, root=57)
    print(f"\nChord progression:")
    for i, (root, quality, duration) in enumerate(progression):
        print(f"  Chord {i+1}: {quality} (root: {root}, {duration} beats)")

    return beat


def demo_trap():
    """
    Demonstrate modern Trap style.

    Characteristics:
    - 130-170 BPM (double-time feel)
    - Rapid hi-hat rolls (32nd/64th notes)
    - Sliding 808s with pitch bends
    - Sparse arrangement
    - Half-time snare (on beat 3)
    """
    print("\n" + "=" * 70)
    print("TRAP (Modern)")
    print("=" * 70)
    print("Artists: Future, Migos, Travis Scott, 21 Savage")
    print("Production: Metro Boomin, Zaytoven, Southside, TM88")

    generator = HipHopGenerator(
        style=HipHopStyle.TRAP,
        tempo=140,
        key_root=54,  # F#
        swing_amount=0.5  # Straight (no swing in trap)
    )

    # Generate 8-bar loop
    beat = generator.generate_beat(bars=8, include_808=True, include_samples=False)

    print(f"\nGenerated 8-bar trap beat:")
    print(f"  Tempo: 140 BPM (feels like 70 BPM half-time)")
    print(f"  Key: F# minor")
    print(f"  Swing: None (straight quantization)")
    print(f"\nTrack breakdown:")
    for track_name, notes in beat.items():
        print(f"  - {track_name.upper()}: {len(notes)} notes")

    # Show trap-specific drum pattern
    trap_drums = TrapDrums.generate_pattern(bars=4, hi_hat_density='roll')
    print(f"\nTrap drum analysis:")
    print(f"  Hi-hat rolls: {len([n for n in trap_drums.hihat if n.start_time % 0.125 == 0])} rapid hits")
    print(f"  Kick pattern: {len(trap_drums.kick)} hits (sparse)")
    print(f"  Snare: {len(trap_drums.snare)} hits (half-time on beat 3)")

    # Show 808 pattern
    bass_pattern = Bass808Engine.generate_808_pattern(HipHopStyle.TRAP, bars=4, root_note=30)
    print(f"\n808 Bass:")
    print(f"  Notes: {len(bass_pattern)}")
    print(f"  Pitch range: {min(n.pitch for n in bass_pattern)} - {max(n.pitch for n in bass_pattern)}")
    print(f"  Includes pitch slides/bends")

    return beat


def demo_lofi():
    """
    Demonstrate Lo-Fi Hip-Hop style.

    Characteristics:
    - 70-90 BPM
    - Off-grid quantization (humanized)
    - Soft, muted drums
    - J Dilla swing (56%)
    - Jazz-influenced harmony
    - "Dusty" aesthetic
    """
    print("\n" + "=" * 70)
    print("LO-FI HIP-HOP")
    print("=" * 70)
    print("Artists: Nujabes, J Dilla, ChilledCow, Jinsang")
    print("Production: MPC3000, SP-404, Tape saturation")

    generator = HipHopGenerator(
        style=HipHopStyle.LOFI,
        tempo=78,
        key_root=60,  # C
        swing_amount=0.56  # J Dilla swing
    )

    # Generate 8-bar loop
    beat = generator.generate_beat(bars=8, include_808=False, include_samples=True)

    print(f"\nGenerated 8-bar lo-fi beat:")
    print(f"  Tempo: 78 BPM")
    print(f"  Key: C minor pentatonic")
    print(f"  Swing: 56% (J Dilla feel)")
    print(f"\nTrack breakdown:")
    for track_name, notes in beat.items():
        print(f"  - {track_name.upper()}: {len(notes)} notes")

    # Show lo-fi specific features
    lofi_drums = LoFiDrums.generate_pattern(bars=4, swing_amount=0.56)
    print(f"\nLo-Fi characteristics:")
    print(f"  Soft dynamics: velocities 40-75 (vs. 90-127 in trap)")
    print(f"  Off-grid timing: ±0.05 beat randomization")
    print(f"  Total notes: {len(lofi_drums.get_all_notes())}")

    # Show jazz-influenced harmony
    progression = HipHopHarmony.get_progression(HipHopStyle.LOFI, root=60)
    print(f"\nJazz-influenced chord progression:")
    for i, (root, quality, duration) in enumerate(progression):
        print(f"  Chord {i+1}: {quality} (root: {root})")

    return beat


def demo_drill():
    """
    Demonstrate Drill style (Chicago/UK).

    Characteristics:
    - 60-80 BPM (half-time feel)
    - Very sparse arrangement
    - Sliding 808s
    - Dark, menacing atmosphere
    - Minimal hi-hats
    - Phrygian/dark minor modes
    """
    print("\n" + "=" * 70)
    print("DRILL (Chicago/UK)")
    print("=" * 70)
    print("Artists: Chief Keef, Pop Smoke, Headie One, Digga D")
    print("Production: Young Chop, AXL Beats, 808Melo")

    generator = HipHopGenerator(
        style=HipHopStyle.DRILL,
        tempo=70,
        key_root=56,  # G#
        swing_amount=0.5  # Straight
    )

    # Generate 8-bar loop
    beat = generator.generate_beat(bars=8, include_808=True, include_samples=False)

    print(f"\nGenerated 8-bar drill beat:")
    print(f"  Tempo: 70 BPM (feels like 140 double-time)")
    print(f"  Key: G# Phrygian (dark mode)")
    print(f"  Swing: None")
    print(f"\nTrack breakdown:")
    for track_name, notes in beat.items():
        print(f"  - {track_name.upper()}: {len(notes)} notes")

    # Show drill-specific features
    drill_drums = DrillDrums.generate_pattern(bars=4)
    print(f"\nDrill characteristics:")
    print(f"  Sparse arrangement: {len(drill_drums.get_all_notes())} total notes (vs. ~100 in trap)")
    print(f"  Minimal hi-hats: {len(drill_drums.hihat)} hits")
    print(f"  Heavy kick emphasis: velocities 110-127")

    # Show dark harmony
    scale = HipHopHarmony.get_scale(HipHopStyle.DRILL, root=56, octaves=1)
    print(f"\nPhrygian scale (dark mode):")
    print(f"  Notes: {scale}")

    return beat


def demo_conscious():
    """
    Demonstrate Conscious Rap style.

    Characteristics:
    - 85-100 BPM
    - Live instrumentation feel
    - Complex harmony (Dorian mode)
    - Jazz/soul influences
    - Musical storytelling
    """
    print("\n" + "=" * 70)
    print("CONSCIOUS RAP")
    print("=" * 70)
    print("Artists: Kendrick Lamar, J. Cole, Common, Mos Def, Talib Kweli")
    print("Production: 9th Wonder, Pete Rock, DJ Premier")

    generator = HipHopGenerator(
        style=HipHopStyle.CONSCIOUS,
        tempo=94,
        key_root=62,  # D
        swing_amount=0.54
    )

    # Generate 16-bar loop
    beat = generator.generate_beat(bars=16, include_808=False, include_samples=True)

    print(f"\nGenerated 16-bar conscious rap beat:")
    print(f"  Tempo: 94 BPM")
    print(f"  Key: D Dorian")
    print(f"  Swing: 54% (MPC)")
    print(f"\nTrack breakdown:")
    for track_name, notes in beat.items():
        print(f"  - {track_name.upper()}: {len(notes)} notes")

    # Show Dorian harmony
    scale = HipHopHarmony.get_scale(HipHopStyle.CONSCIOUS, root=62, octaves=1)
    progression = HipHopHarmony.get_progression(HipHopStyle.CONSCIOUS, root=62)
    print(f"\nDorian mode characteristics:")
    print(f"  Scale: {len(scale)} notes (full 7-note scale)")
    print(f"  Progression: {len(progression)} chords (complex)")

    return beat


def demo_g_funk():
    """
    Demonstrate G-Funk (West Coast) style.

    Characteristics:
    - 90-105 BPM
    - Synthesizer leads (talk box, portamento)
    - Funk basslines
    - Major key progressions
    - Smooth, laid-back feel
    """
    print("\n" + "=" * 70)
    print("G-FUNK (West Coast)")
    print("=" * 70)
    print("Artists: Dr. Dre, Snoop Dogg, Warren G, Nate Dogg")
    print("Production: Dr. Dre, DJ Quik, Warren G")

    generator = HipHopGenerator(
        style=HipHopStyle.G_FUNK,
        tempo=98,
        key_root=65,  # F
        swing_amount=0.52  # Light swing
    )

    # Generate 8-bar loop
    beat = generator.generate_beat(bars=8, include_808=True, include_samples=False)

    print(f"\nGenerated 8-bar G-funk beat:")
    print(f"  Tempo: 98 BPM")
    print(f"  Key: F major")
    print(f"  Swing: 52% (light funk swing)")
    print(f"\nTrack breakdown:")
    for track_name, notes in beat.items():
        print(f"  - {track_name.upper()}: {len(notes)} notes")

    # Show G-funk harmony (major progressions)
    progression = HipHopHarmony.get_progression(HipHopStyle.G_FUNK, root=65)
    print(f"\nG-Funk chord progression (major key):")
    for i, (root, quality, duration) in enumerate(progression):
        print(f"  Chord {i+1}: {quality} (root: {root})")

    return beat


def demo_sample_chopping():
    """
    Demonstrate sample chopping techniques.

    Shows:
    - 4-slice chopping
    - 8-slice chopping
    - 16-slice chopping
    - Rearrangement patterns
    """
    print("\n" + "=" * 70)
    print("SAMPLE CHOPPING TECHNIQUES")
    print("=" * 70)
    print("Technique: Slicing and rearranging audio samples")
    print("Pioneers: J Dilla, 9th Wonder, Pete Rock, RZA")

    # 4-slice pattern
    print("\n4-Slice Pattern:")
    chops_4 = SampleChopper.generate_chop_pattern(
        bars=2,
        slice_mode=SampleSliceMode.FOUR_SLICE,
        root_note=60
    )
    print(f"  Slices: 4")
    print(f"  Duration: 2 bars")
    print(f"  Notes generated: {len(chops_4)}")

    # 8-slice pattern
    print("\n8-Slice Pattern:")
    chops_8 = SampleChopper.generate_chop_pattern(
        bars=2,
        slice_mode=SampleSliceMode.EIGHT_SLICE,
        root_note=60
    )
    print(f"  Slices: 8")
    print(f"  Duration: 2 bars")
    print(f"  Notes generated: {len(chops_8)}")

    # 16-slice pattern (MPC-style)
    print("\n16-Slice Pattern (MPC):")
    chops_16 = SampleChopper.generate_chop_pattern(
        bars=2,
        slice_mode=SampleSliceMode.SIXTEEN_SLICE,
        root_note=60
    )
    print(f"  Slices: 16")
    print(f"  Duration: 2 bars")
    print(f"  Notes generated: {len(chops_16)}")
    print(f"  Rearrangement: Random pattern applied")

    print("\nChopping techniques:")
    print("  - Reverse slices")
    print("  - Pitch shift (varispeed)")
    print("  - Stutter/repeat")
    print("  - Slice rearrangement")


def demo_mpc_swing():
    """
    Demonstrate MPC swing timing.

    Shows:
    - Straight quantization (50%)
    - Light swing (52%)
    - MPC swing (54%)
    - J Dilla swing (56%)
    - Heavy swing (62% - triplet feel)
    """
    print("\n" + "=" * 70)
    print("MPC SWING TIMING")
    print("=" * 70)
    print("Algorithm: Roger Linn (Akai MPC)")
    print("Effect: Delays offbeat notes for groove")

    # Create test pattern (straight 16th notes)
    from genres.hiphop import HipHopNote
    test_pattern = [
        HipHopNote(
            pitch=42,  # Hi-hat
            velocity=80,
            start_time=i * 0.25,  # 16th notes
            duration=0.1
        )
        for i in range(16)
    ]

    swing_amounts = {
        'Straight (no swing)': 0.50,
        'Light swing': 0.52,
        'MPC swing': 0.54,
        'J Dilla swing': 0.56,
        'Triplet swing': 0.62
    }

    print("\nSwing amount comparison:")
    for name, amount in swing_amounts.items():
        swung = apply_mpc_swing(test_pattern, swing_amount=amount)
        print(f"\n  {name} ({int(amount*100)}%):")
        print(f"    Offbeat delay: {(amount - 0.5) * 0.5:.3f} beats")
        print(f"    Feel: {'straight' if amount == 0.5 else 'swung'}")


def demo_complete_production():
    """
    Demonstrate complete beat production workflow.

    Shows all elements combined:
    - Drums
    - 808 Bass
    - Sample chops
    - Swing timing
    - Arrangement
    """
    print("\n" + "=" * 70)
    print("COMPLETE BEAT PRODUCTION")
    print("=" * 70)
    print("Full production workflow example")

    # Create a boom bap beat with all elements
    print("\nProduction: 90s Boom Bap Instrumental")
    print("-" * 70)

    generator = HipHopGenerator(
        style=HipHopStyle.BOOM_BAP,
        tempo=92,
        key_root=57,  # A minor
        swing_amount=0.54
    )

    # Intro (4 bars - drums only)
    print("\nIntro (4 bars):")
    intro = generator.generate_beat(bars=4, include_808=False, include_samples=False)
    print(f"  Drums only: {len(intro['drums'])} notes")

    # Verse (16 bars - full arrangement)
    print("\nVerse (16 bars):")
    verse = generator.generate_beat(bars=16, include_808=True, include_samples=True)
    print(f"  Drums: {len(verse['drums'])} notes")
    print(f"  808 Bass: {len(verse['808'])} notes")
    print(f"  Sample chops: {len(verse['samples'])} notes")
    total_verse = sum(len(notes) for notes in verse.values())
    print(f"  Total: {total_verse} notes")

    # Hook (8 bars - variation)
    print("\nHook (8 bars):")
    hook = generator.generate_beat(bars=8, include_808=True, include_samples=True)
    total_hook = sum(len(notes) for notes in hook.values())
    print(f"  Total: {total_hook} notes")
    print(f"  Variation: Different sample arrangement")

    # Complete arrangement
    print("\nComplete arrangement structure:")
    print("  Intro:  4 bars  (drums only)")
    print("  Verse:  16 bars (full)")
    print("  Hook:   8 bars  (full + variation)")
    print("  Verse:  16 bars (full)")
    print("  Hook:   8 bars  (full + variation)")
    print("  Outro:  4 bars  (drums fadeout)")
    print(f"\n  Total: 56 bars @ 92 BPM = ~2:30 duration")


# ============================================================================
# Main Demo Runner
# ============================================================================

def run_all_demos():
    """Run all demonstration functions"""
    print("\n" + "=" * 70)
    print("HIP-HOP MUSIC GENERATOR - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nDemonstrating 6 sub-genres + production techniques")
    print("Agent 41 - Hip-Hop/Rap Module")

    # Sub-genre demos
    demo_boom_bap()
    demo_trap()
    demo_lofi()
    demo_drill()
    demo_conscious()
    demo_g_funk()

    # Technique demos
    demo_sample_chopping()
    demo_mpc_swing()
    demo_complete_production()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nAll hip-hop styles and techniques demonstrated successfully!")
    print("\nFeatures shown:")
    print("  ✓ 6 sub-genres (Boom Bap, Trap, Lo-Fi, Drill, Conscious, G-Funk)")
    print("  ✓ Drum pattern generation (style-specific)")
    print("  ✓ 808 bass with pitch slides")
    print("  ✓ Sample chopping and rearrangement")
    print("  ✓ MPC/J Dilla swing timing")
    print("  ✓ Harmonic structures (minor vamps, Dorian, Phrygian)")
    print("  ✓ Complete beat arrangement")


if __name__ == "__main__":
    run_all_demos()
