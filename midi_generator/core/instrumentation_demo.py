#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Demo - Agent 21 Instrumentation Specialist
======================================================

Interactive demonstration of instrumentation specialist capabilities.

Showcases:
- Ensemble selection across genres
- Piano voicing generation
- Bass pattern generation
- Drum pattern generation
- Brass section voicing
- Orchestration recommendations
- Blend analysis

Author: Agent 21
Date: 2025
License: MIT
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.instrumentation_specialist import (
    InstrumentationSpecialist,
    PianoVoicingGenerator,
    BassPatternGenerator,
    DrumPatternGenerator,
    BrassSectionVoicing,
    InstrumentRole,
    EnsembleType,
    VoicingType,
    BassPattern,
    DrumPattern,
    TextureDensity,
    OrchestrationKnowledge,
)
from core.instrument_library import (
    get_instrument,
    InstrumentFamily,
)


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def print_subheader(title: str):
    """Print formatted subsection header"""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def demo_ensemble_selection():
    """Demo 1: Ensemble Selection"""
    print_header("DEMO 1: ENSEMBLE SELECTION")

    specialist = InstrumentationSpecialist()

    # Jazz Trio
    print_subheader("Jazz Trio - Sparse Texture")
    trio = specialist.select_ensemble("jazz", texture_density=0.3)
    print(f"Ensemble Type: {trio.ensemble_type.value}")
    print(f"Texture Density: {trio.texture_density.value}")
    print(f"\nInstruments:")
    for inst in trio.instruments:
        role = trio.roles.get(inst.name, InstrumentRole.HARMONY)
        print(f"  • {inst.name:20} - Role: {role.value:15} - Family: {inst.family.value}")
    print(f"\nBlend Score: {trio.blend_score:.2f}/1.00")
    print(f"Voicing Preference: {trio.voicing_type.value}")

    # Big Band
    print_subheader("Big Band - Dense Texture")
    big_band = specialist.select_ensemble("jazz", texture_density=0.9)
    print(f"Ensemble Type: {big_band.ensemble_type.value}")
    print(f"Texture Density: {big_band.texture_density.value}")
    print(f"\nInstruments ({len(big_band.instruments)} total):")

    # Group by family
    by_family = {}
    for inst in big_band.instruments:
        family = inst.family.value
        if family not in by_family:
            by_family[family] = []
        by_family[family].append(inst.name)

    for family, instruments in sorted(by_family.items()):
        print(f"  {family.upper()}:")
        for inst in instruments:
            print(f"    • {inst}")

    print(f"\nBlend Score: {big_band.blend_score:.2f}/1.00")

    # String Quartet
    print_subheader("String Quartet - Classical Chamber")
    quartet = specialist.select_ensemble(
        "classical",
        texture_density=0.5,
        custom_instruments=["Violin", "Violin", "Viola", "Cello"]
    )
    print(f"Ensemble Type: {quartet.ensemble_type.value}")
    print(f"\nInstruments:")
    for inst in quartet.instruments:
        role = quartet.roles.get(inst.name, InstrumentRole.HARMONY)
        range_info = f"{inst.range.lowest_note}-{inst.range.highest_note}"
        print(f"  • {inst.name:20} - Role: {role.value:15} - Range: {range_info}")
    print(f"\nBlend Score: {quartet.blend_score:.2f}/1.00")


def demo_piano_voicings():
    """Demo 2: Piano Voicing Generation"""
    print_header("DEMO 2: PIANO VOICING GENERATION")

    generator = PianoVoicingGenerator()

    # Define chord
    root = 60  # Middle C
    chord_tones = [0, 4, 7, 10]  # C E G Bb (dominant 7th)
    extensions = [14, 17]  # 9th and 11th

    print(f"Chord: C7(9,11)")
    print(f"Root Note: {root} (MIDI)")
    print(f"Chord Tones: {chord_tones} (C, E, G, Bb)")
    print(f"Extensions: {extensions} (D, F)")

    # Drop-2
    print_subheader("Drop-2 Voicing")
    drop2 = generator.generate_drop_2(root, chord_tones, extensions)
    print(f"Voicing Type: {drop2.voicing_type.value}")
    print(f"Notes (MIDI): {drop2.notes}")
    intervals = [n - root for n in drop2.notes]
    print(f"Intervals from Root: {intervals}")
    print(f"Spread: {drop2.spread} octave(s)")
    print("\nIdeal for: Jazz piano comping, guitar")

    # Drop-3
    print_subheader("Drop-3 Voicing")
    drop3 = generator.generate_drop_3(root, chord_tones, extensions)
    print(f"Voicing Type: {drop3.voicing_type.value}")
    print(f"Notes (MIDI): {drop3.notes}")
    intervals = [n - root for n in drop3.notes]
    print(f"Intervals from Root: {intervals}")
    print("\nIdeal for: Guitar, wider piano voicings")

    # Rootless (Bill Evans style)
    print_subheader("Rootless Voicing (Type A) - Bill Evans Style")
    rootless = generator.generate_rootless(root, chord_tones, extensions, type_a=True)
    print(f"Voicing Type: {rootless.voicing_type.value}")
    print(f"Notes (MIDI): {rootless.notes}")
    print(f"Omissions: {rootless.omissions} (root omitted - bass plays it)")
    intervals = [n - root for n in rootless.notes]
    print(f"Intervals from Root: {intervals}")
    print("\nIdeal for: Jazz piano trio, modern jazz")
    print("Note: Bass player covers the root")

    # Quartal (McCoy Tyner style)
    print_subheader("Quartal Voicing - McCoy Tyner Style")
    quartal = generator.generate_quartal(root, num_voices=5)
    print(f"Voicing Type: {quartal.voicing_type.value}")
    print(f"Notes (MIDI): {quartal.notes}")
    intervals_between = [quartal.notes[i+1] - quartal.notes[i] for i in range(len(quartal.notes)-1)]
    print(f"Intervals: {intervals_between} (all perfect 4ths)")
    print("\nIdeal for: Modal jazz, contemporary classical")
    print("Sound: Open, modern, ambiguous")

    # Cluster (Herbie Hancock style)
    print_subheader("Cluster Voicing - Herbie Hancock Style")
    cluster = generator.generate_cluster(root, num_voices=4, whole_tone=False)
    print(f"Voicing Type: {cluster.voicing_type.value}")
    print(f"Notes (MIDI): {cluster.notes}")
    intervals_between = [cluster.notes[i+1] - cluster.notes[i] for i in range(len(cluster.notes)-1)]
    print(f"Intervals: {intervals_between} (all half steps)")
    print("\nIdeal for: Modern jazz, impressionistic passages")
    print("Sound: Dense, colorful, dissonant")

    # Shell voicing
    print_subheader("Shell Voicing - Essential Tones Only")
    shell = generator.generate_shell(root, chord_quality="dominant")
    print(f"Voicing Type: {shell.voicing_type.value}")
    print(f"Notes (MIDI): {shell.notes}")
    print(f"Chord Tones: {shell.chord_tones} (root, 3rd, 7th only)")
    print("\nIdeal for: Sparse comping, giving soloist space")
    print("Sound: Clear, minimal, supportive")


def demo_bass_patterns():
    """Demo 3: Bass Pattern Generation"""
    print_header("DEMO 3: BASS PATTERN GENERATION")

    generator = BassPatternGenerator()
    root = 48  # Low C

    # Walking Bass
    print_subheader("Walking Bass - Jazz Swing")
    walking = generator.generate_walking_bass(
        root,
        [(root, 4.0), (root + 7, 4.0)],  # C for 4 beats, G for 4 beats
        style="swing"
    )
    print(f"Pattern: C major to G major (4 beats each)")
    print(f"Total notes: {len(walking)}")
    print(f"\nFirst 8 notes:")
    for i, (note, duration) in enumerate(walking[:8]):
        interval = note - root
        print(f"  Beat {i+1}: Note {note} (root + {interval:2d}) - Duration: {duration:.2f} beats")
    print("\nCharacteristics:")
    print("  • Chord tones on strong beats")
    print("  • Chromatic approach tones on weak beats")
    print("  • Stepwise motion creates walking feel")

    # Pedal Tone
    print_subheader("Pedal Tone - Sustained Bass")
    pedal = generator.generate_pedal(root, duration=16.0)
    print(f"Pattern: Sustained {root} for {pedal[0][1]} beats")
    print(f"Note: {pedal[0][0]} (MIDI)")
    print("\nIdeal for:")
    print("  • Modal vamps")
    print("  • Building tension")
    print("  • Anchoring harmony")

    # Two-Feel
    print_subheader("Two-Feel - Medium Swing")
    two_feel = generator.generate_two_feel(root, num_bars=4)
    print(f"Pattern: 4 bars of two-feel")
    print(f"Total notes: {len(two_feel)}")
    print(f"\nPattern:")
    for i, (note, duration) in enumerate(two_feel):
        beat = i * 2 + 1
        interval = note - root
        tone = "Root" if interval == 0 else "Fifth" if interval == 7 else f"+{interval}"
        print(f"  Bar {i//2 + 1}, Beat {beat}: {tone:10} - Duration: {duration:.1f} beats")
    print("\nIdeal for:")
    print("  • Medium tempo swing")
    print("  • Ballad verses")
    print("  • Creating space")

    # Funk Bass
    print_subheader("Funk Bass - Syncopated Groove")
    funk = generator.generate_funk_bass(root, pattern_type="basic")
    print(f"Pattern: Basic funk groove")
    print(f"Total notes: {len(funk)}")
    print(f"\nPattern (one bar):")
    for i, (note, duration) in enumerate(funk):
        interval = note - root
        octave_str = "(octave below)" if interval < 0 else "(octave above)" if interval > 12 else ""
        print(f"  Note {i+1}: MIDI {note:2d} (root {interval:+3d}) {octave_str:15} - {duration:.2f} beats")
    print("\nCharacteristics:")
    print("  • Heavy syncopation")
    print("  • Octave jumps")
    print("  • Sixteenth-note rhythms")
    print("  • Percussive, rhythmic feel")


def demo_drum_patterns():
    """Demo 4: Drum Pattern Generation"""
    print_header("DEMO 4: DRUM PATTERN GENERATION")

    generator = DrumPatternGenerator()

    # Swing
    print_subheader("Swing Pattern - Medium Jazz")
    swing = generator.generate_swing_pattern(feel="medium")
    print(f"Feel: Medium Swing")
    print(f"Voices: {', '.join(swing.keys())}")
    print(f"\nRide Cymbal Pattern (triplet swing feel):")
    for beat, velocity in swing['ride'][:6]:
        print(f"  Beat {beat:.3f}: Velocity {velocity:3d}")
    print(f"\nHi-Hat (2 and 4):")
    for beat, velocity in swing['hi_hat']:
        print(f"  Beat {beat:.1f}: Velocity {velocity:3d}")
    print("\nCharacteristics:")
    print("  • Ride cymbal: Ding-ding-a-ding (triplet feel)")
    print("  • Hi-hat: 2 and 4 (backbeat)")
    print("  • Snare: Light backbeat")
    print("  • Kick: Feathering, interactive")

    # Rock
    print_subheader("Rock Pattern - Basic 8th Notes")
    rock = generator.generate_rock_pattern(style="basic")
    print(f"Style: Basic Rock")
    print(f"Voices: {', '.join(rock.keys())}")
    print(f"\nHi-Hat Pattern (8th notes):")
    for i, (beat, velocity) in enumerate(rock['hi_hat'][:4]):
        print(f"  Beat {beat:.1f}: Velocity {velocity:3d}")
    print(f"\nSnare (backbeat on 2 and 4):")
    for beat, velocity in rock['snare']:
        print(f"  Beat {beat:.1f}: Velocity {velocity:3d}")
    print(f"\nKick (bass drum):")
    for beat, velocity in rock['kick']:
        print(f"  Beat {beat:.1f}: Velocity {velocity:3d}")

    # Funk
    print_subheader("Funk Pattern - Syncopated Groove")
    funk = generator.generate_funk_pattern()
    print(f"Style: Funk (Clyde Stubblefield inspired)")
    print(f"Total hi-hat hits: {len(funk['hi_hat'])}")
    print(f"\nSnare Pattern (with ghost notes):")
    for beat, velocity in funk['snare']:
        note_type = "Ghost note" if velocity < 80 else "Backbeat"
        print(f"  Beat {beat:.2f}: Velocity {velocity:3d} - {note_type}")
    print("\nCharacteristics:")
    print("  • 16th-note hi-hat")
    print("  • Ghost notes on snare")
    print("  • Syncopated kick pattern")
    print("  • Pocket groove, danceable")

    # Bossa Nova
    print_subheader("Bossa Nova Pattern - Brazilian Jazz")
    bossa = generator.generate_latin_pattern(style="bossa")
    print(f"Style: Bossa Nova")
    print(f"Voices: {', '.join(bossa.keys())}")
    print(f"\nRide/Cross-stick Pattern:")
    for beat, velocity in bossa['ride']:
        print(f"  Beat {beat:.1f}: Velocity {velocity:3d}")
    print("\nCharacteristics:")
    print("  • Cross-stick or rim clicks")
    print("  • Syncopated kick pattern")
    print("  • Light, crisp snare")
    print("  • Characteristic 3-2 clave feel")


def demo_brass_voicings():
    """Demo 5: Brass Section Voicing"""
    print_header("DEMO 5: BRASS SECTION VOICING")

    voicer = BrassSectionVoicing()
    root = 60  # Middle C
    chord_tones = [0, 4, 7, 10]  # Dominant 7th

    print(f"Chord: C7 (C E G Bb)")
    print(f"Root: {root} (MIDI)")

    # Four-Way Close
    print_subheader("Four-Way Close Voicing")
    four_close = voicer.four_way_close(root, chord_tones)
    print(f"Instrumentation:")
    for inst, note in sorted(four_close.items(), key=lambda x: -x[1]):
        interval = note - root
        print(f"  {inst:20} - MIDI {note:3d} (root + {interval:2d})")

    notes = sorted(four_close.values(), reverse=True)
    spread = max(notes) - min(notes)
    print(f"\nSpread: {spread} semitones (within an octave)")
    print("\nCharacteristics:")
    print("  • All voices within an octave of lead")
    print("  • Clean, crisp section sound")
    print("  • Classic big band voicing")
    print("  • Good for soli passages")

    # Four-Way Open
    print_subheader("Four-Way Open (Spread) Voicing")
    four_open = voicer.four_way_open(root, chord_tones)
    print(f"Instrumentation:")
    for inst, note in sorted(four_open.items(), key=lambda x: -x[1]):
        interval = note - root
        print(f"  {inst:20} - MIDI {note:3d} (root + {interval:2d})")

    notes = sorted(four_open.values(), reverse=True)
    spread = max(notes) - min(notes)
    print(f"\nSpread: {spread} semitones")
    print("\nCharacteristics:")
    print("  • Wider spacing between voices")
    print("  • Fuller, richer sound")
    print("  • More orchestral")
    print("  • Good for backgrounds")

    # Five-Way Close
    print_subheader("Five-Way Close Voicing")
    five_close = voicer.five_way_close(root, chord_tones, extensions=[14])  # Add 9th
    print(f"Instrumentation (with extensions):")
    for inst, note in sorted(five_close.items(), key=lambda x: -x[1]):
        interval = note - root
        print(f"  {inst:20} - MIDI {note:3d} (root + {interval:2d})")

    print("\nCharacteristics:")
    print("  • Full big band brass section")
    print("  • Can accommodate extensions")
    print("  • Dense, powerful sound")
    print("  • Classic shout chorus voicing")

    # Double Lead
    print_subheader("Double Lead Voicing")
    lead_note = root + 10  # Bb (the 7th)
    double = voicer.double_lead(root, chord_tones, lead_note)
    print(f"Lead Note: {lead_note} (Bb)")
    print(f"\nInstrumentation:")
    for inst, note in sorted(double.items(), key=lambda x: -x[1]):
        interval = note - root
        doubled = " *** DOUBLED ***" if "double" in inst.lower() else ""
        print(f"  {inst:25} - MIDI {note:3d} (root + {interval:2d}) {doubled}")

    print("\nCharacteristics:")
    print("  • Lead melody doubled for emphasis")
    print("  • Bright, powerful lead sound")
    print("  • Basie and Ellington favorite")
    print("  • Great for featured melodies")


def demo_orchestration_knowledge():
    """Demo 6: Orchestration Knowledge Base"""
    print_header("DEMO 6: ORCHESTRATION KNOWLEDGE")

    knowledge = OrchestrationKnowledge()

    # Blend compatibility
    print_subheader("Instrument Blend Compatibility Database")
    print(f"Total blend relationships: {len(knowledge.BLEND_COMPATIBILITY_DB)}")
    print(f"\nExcellent Blends (score >= 0.9):")
    excellent = [b for b in knowledge.BLEND_COMPATIBILITY_DB if b.compatibility_score >= 0.9]
    for blend in excellent[:5]:
        print(f"  • {blend.instrument_a:15} + {blend.instrument_b:15} = {blend.compatibility_score:.2f}")
        print(f"    Context: {blend.context}")
        print(f"    Notes: {blend.notes}\n")

    # Orchestration rules
    print_subheader("Orchestration Rules from Master Treatises")
    print(f"Total rules: {len(knowledge.ORCHESTRATION_RULES)}")
    print(f"\nHigh-priority rules:")
    high_priority = [r for r in knowledge.ORCHESTRATION_RULES if r.priority >= 8]
    for rule in high_priority:
        print(f"  • {rule.name}")
        print(f"    Condition: {rule.condition}")
        print(f"    Recommendation: {rule.recommendation}")
        print(f"    Priority: {rule.priority}/10")
        if rule.genre_specific:
            print(f"    Genre-specific: {', '.join(rule.genre_specific)}")
        print()

    # Doubling guidelines
    print_subheader("Doubling Guidelines")
    for doubling_type, info in knowledge.DOUBLING_RULES.items():
        print(f"\n{doubling_type.upper().replace('_', ' ')}:")
        print(f"  Description: {info['description']}")
        print(f"  Good for: {', '.join(info['good_for'])}")
        if 'avoid' in info:
            print(f"  Avoid: {', '.join(info['avoid'])}")
        if 'classic' in info:
            print(f"  Classic combinations: {', '.join(info['classic'])}")


def demo_full_workflow():
    """Demo 7: Complete Workflow"""
    print_header("DEMO 7: COMPLETE WORKFLOW - JAZZ BALLAD")

    specialist = InstrumentationSpecialist()

    print("Scenario: Creating instrumentation for a jazz ballad")
    print("Requirements:")
    print("  • Genre: Jazz")
    print("  • Tempo: Slow (ballad)")
    print("  • Texture: Medium (intimate but full)")
    print("  • Style: Modern jazz quartet")

    # Step 1: Select ensemble
    print_subheader("Step 1: Select Ensemble")
    profile = specialist.select_ensemble("jazz", texture_density=0.5)
    print(f"Selected: {profile.ensemble_type.value}")
    print(f"Instruments:")
    for inst in profile.instruments:
        role = profile.roles.get(inst.name, InstrumentRole.HARMONY)
        print(f"  • {inst.name:20} - {role.value}")

    # Step 2: Choose voicing style
    print_subheader("Step 2: Assign Piano Voicing Style")
    profile.voicing_type = VoicingType.ROOTLESS
    print(f"Voicing: {profile.voicing_type.value}")
    print("Rationale: Rootless voicings work well for ballads")
    print("          Bass plays root, piano plays 3rd, 7th, and colors")

    # Step 3: Generate specific voicing
    print_subheader("Step 3: Generate Specific Voicing for Cmaj7")
    root = 60
    chord_tones = [0, 4, 7, 11]  # C E G B (major 7th)
    extensions = [14]  # 9th
    voicing = specialist.assign_voicing(profile, root, chord_tones, extensions)
    print(f"Chord: Cmaj7(9)")
    if voicing:
        for inst, notes in voicing.items():
            print(f"  {inst}: {notes}")

    # Step 4: Bass line
    print_subheader("Step 4: Generate Bass Line")
    profile.bass_pattern = BassPattern.PEDAL
    bass_line = specialist.generate_bass_line(BassPattern.PEDAL, 48, duration=8.0)
    print(f"Pattern: {profile.bass_pattern.value}")
    print(f"Note: {bass_line[0][0]} (MIDI), Duration: {bass_line[0][1]} beats")
    print("Rationale: Pedal tone creates stability for ballad")

    # Step 5: Drum pattern
    print_subheader("Step 5: Generate Drum Pattern")
    profile.drum_pattern = DrumPattern.BRUSHES
    drums = specialist.generate_drum_pattern(DrumPattern.SWING, feel="slow")
    print(f"Pattern: Slow swing (brushes)")
    print(f"Voices: {', '.join(drums.keys())}")
    print("Rationale: Brushes create intimate ballad feel")

    # Step 6: Articulations
    print_subheader("Step 6: Assign Articulations")
    articulations = specialist.assign_articulations(profile, musical_context="legato")
    print("Articulations:")
    for inst, art in articulations.items():
        print(f"  • {inst:20} - {art.value}")
    print("Rationale: Legato articulations for smooth ballad")

    # Step 7: Blend analysis
    print_subheader("Step 7: Analyze Ensemble Blend")
    blend_score = specialist.calculate_blend_score(profile.instruments)
    print(f"Blend Score: {blend_score:.2f}/1.00")
    if blend_score >= 0.8:
        print("Assessment: Excellent blend - instruments work well together")
    elif blend_score >= 0.6:
        print("Assessment: Good blend - suitable for this context")
    else:
        print("Assessment: Consider instrument substitutions for better blend")

    # Final summary
    print_subheader("Final Configuration Summary")
    print(f"Ensemble: {profile.ensemble_type.value}")
    print(f"Texture: {profile.texture_density.value}")
    print(f"Voicing: {profile.voicing_type.value}")
    print(f"Bass Pattern: {profile.bass_pattern.value if profile.bass_pattern else 'Not set'}")
    print(f"Drum Pattern: {profile.drum_pattern.value if profile.drum_pattern else 'Not set'}")
    print(f"Blend Score: {blend_score:.2f}")
    print("\n✓ Ready for MIDI generation!")


def main():
    """
    Run complete interactive demo
    """
    print("=" * 80)
    print("AGENT 21: INSTRUMENTATION SPECIALIST".center(80))
    print("Interactive Demonstration".center(80))
    print("=" * 80)

    print("\nThis demo showcases the comprehensive instrumentation capabilities")
    print("of the Agent 21 Instrumentation Specialist engine.\n")

    demos = [
        ("Ensemble Selection", demo_ensemble_selection),
        ("Piano Voicing Generation", demo_piano_voicings),
        ("Bass Pattern Generation", demo_bass_patterns),
        ("Drum Pattern Generation", demo_drum_patterns),
        ("Brass Section Voicing", demo_brass_voicings),
        ("Orchestration Knowledge", demo_orchestration_knowledge),
        ("Complete Workflow", demo_full_workflow),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
            input(f"\nPress ENTER to continue to next demo ({i}/{len(demos)})...")
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error in demo: {str(e)}")
            import traceback
            traceback.print_exc()

    print_header("DEMO COMPLETE")
    print("Thank you for exploring Agent 21: Instrumentation Specialist!")
    print("\nFor more information, see:")
    print("  • instrumentation_specialist.py - Main engine")
    print("  • instrumentation_test.py - Validation suite")
    print("  • INSTRUMENTATION_README.md - Full documentation")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
