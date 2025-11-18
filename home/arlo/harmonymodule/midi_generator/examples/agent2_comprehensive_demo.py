#!/usr/bin/env python3
"""
Comprehensive Demonstration of Agent 2 Algorithms

This script demonstrates all three advanced melody algorithms implemented
by Agent 2:
1. L-Systems (Lindenmayer Systems)
2. Cellular Automata
3. Constraint Satisfaction Problems

Each algorithm generates melodies and exports them to MIDI files for comparison.

Author: MIDI Generator Library - Agent 2
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.lsystem import LSystem, MusicalGrammar
from algorithms.cellular_automata import ElementaryCA, GameOfLife, MusicalCA
from algorithms.constraint_solver import MelodicCSP
from core.music_theory import Scale, ScaleType, MIDDLE_C, note_number_to_name

try:
    import mido
    from mido import Message, MidiFile, MidiTrack
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido library not available. MIDI export disabled.")


def notes_to_midi(notes, filename: str, tempo: int = 120):
    """
    Convert notes to MIDI file.

    Args:
        notes: List of Note or MusicEvent objects
        filename: Output filename
        tempo: Tempo in BPM
    """
    if not MIDO_AVAILABLE:
        print(f"Cannot create {filename} - mido not available")
        return

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))

    # Convert notes to MIDI messages
    events = []

    for note in notes:
        # Get attributes (works for both Note and MusicEvent)
        pitch = note.pitch
        start = note.start
        duration = note.duration
        velocity = note.velocity if hasattr(note, 'velocity') else 80

        # Convert beats to ticks (480 ticks per beat)
        ticks_per_beat = 480
        start_ticks = int(start * ticks_per_beat)
        duration_ticks = int(duration * ticks_per_beat)

        events.append((start_ticks, 'note_on', pitch, velocity))
        events.append((start_ticks + duration_ticks, 'note_off', pitch, 0))

    # Sort by time
    events.sort(key=lambda x: x[0])

    # Convert to delta times
    current_time = 0
    for abs_time, msg_type, pitch, velocity in events:
        delta = abs_time - current_time
        current_time = abs_time

        if msg_type == 'note_on':
            track.append(Message('note_on', note=pitch, velocity=velocity, time=delta))
        else:
            track.append(Message('note_off', note=pitch, velocity=velocity, time=delta))

    # Save
    mid.save(filename)
    print(f"✓ Saved MIDI: {filename}")


def demonstrate_lsystems():
    """Demonstrate L-System melody generation"""
    print("\n" + "=" * 80)
    print("1. L-SYSTEM MELODY GENERATION")
    print("=" * 80)

    c_major = Scale(MIDDLE_C, ScaleType.MAJOR)
    d_minor = Scale(MIDDLE_C + 2, ScaleType.NATURAL_MINOR)

    # Bach chorale style
    print("\n[1.1] Bach Chorale Style")
    bach = MusicalGrammar.bach_chorale()
    bach_string = bach.derive()
    bach_notes = bach.interpret_musical(bach_string, c_major, base_duration=0.5)
    print(f"  Generated {len(bach_notes)} notes")
    print(f"  Sample: {[note_number_to_name(n.pitch) for n in bach_notes[:8]]}")
    notes_to_midi(bach_notes, "/tmp/lsystem_bach.mid")

    # Minimalist
    print("\n[1.2] Minimalist Pattern (Glass/Reich)")
    minimalist = MusicalGrammar.minimalist()
    min_string = minimalist.derive()
    min_notes = minimalist.interpret_musical(min_string, c_major, base_duration=0.25)
    print(f"  Generated {len(min_notes)} notes")
    notes_to_midi(min_notes, "/tmp/lsystem_minimalist.mid")

    # Jazz bebop
    print("\n[1.3] Jazz Bebop")
    bebop = MusicalGrammar.jazz_bebop()
    bebop_string = bebop.derive()
    bebop_notes = bebop.interpret_musical(bebop_string, d_minor, base_duration=0.2)
    print(f"  Generated {len(bebop_notes)} notes")
    notes_to_midi(bebop_notes, "/tmp/lsystem_bebop.mid")

    # Fractal
    print("\n[1.4] Fractal Self-Similar Melody")
    fractal = MusicalGrammar.fractal_melody()
    fractal_string = fractal.derive(iterations=3)
    fractal_notes = fractal.interpret_musical(fractal_string, c_major, base_duration=0.5)
    print(f"  Generated {len(fractal_notes)} notes with hierarchical structure")
    notes_to_midi(fractal_notes, "/tmp/lsystem_fractal.mid")

    # Romantic
    print("\n[1.5] Romantic Expressive Melody")
    romantic = MusicalGrammar.romantic_expression()
    romantic_string = romantic.derive()
    romantic_notes = romantic.interpret_musical(romantic_string, c_major, base_duration=0.75)
    print(f"  Generated {len(romantic_notes)} notes")
    notes_to_midi(romantic_notes, "/tmp/lsystem_romantic.mid")


def demonstrate_cellular_automata():
    """Demonstrate Cellular Automata melody generation"""
    print("\n" + "=" * 80)
    print("2. CELLULAR AUTOMATA MELODY GENERATION")
    print("=" * 80)

    c_major = Scale(MIDDLE_C, ScaleType.MAJOR)
    pentatonic = Scale(MIDDLE_C, ScaleType.MINOR_PENTATONIC)

    # Rule 30 (chaotic)
    print("\n[2.1] Elementary CA - Rule 30 (Chaotic)")
    ca30 = ElementaryCA(width=32, rule=30)
    ca30.set_initial_state(mode='center')
    ca30.evolve(24)
    print("  CA Pattern (first 6 generations):")
    for gen in ca30.history[:6]:
        print("  " + ''.join(['█' if cell else '·' for cell in gen]))

    ca30_notes = ca30.to_melody(c_major, mode='horizontal', base_duration=0.25)
    print(f"  Generated {len(ca30_notes)} notes")
    notes_to_midi(ca30_notes, "/tmp/ca_rule30.mid")

    # Rule 110 (complex)
    print("\n[2.2] Elementary CA - Rule 110 (Turing Complete)")
    ca110 = ElementaryCA(width=32, rule=110)
    ca110.set_initial_state(mode='random')
    ca110.evolve(24)
    ca110_notes = ca110.to_melody(pentatonic, mode='horizontal', base_duration=0.25)
    print(f"  Generated {len(ca110_notes)} notes")
    notes_to_midi(ca110_notes, "/tmp/ca_rule110.mid")

    # Rule 90 (Sierpinski triangle)
    print("\n[2.3] Elementary CA - Rule 90 (Sierpinski Fractal)")
    ca90 = ElementaryCA(width=32, rule=90)
    ca90.set_initial_state(mode='center')
    ca90.evolve(24)
    ca90_notes = ca90.to_melody(c_major, mode='vertical', base_duration=0.25)
    print(f"  Generated {len(ca90_notes)} notes")
    notes_to_midi(ca90_notes, "/tmp/ca_rule90.mid")

    # Game of Life - Glider
    print("\n[2.4] Game of Life - Glider Pattern")
    gol = GameOfLife(width=24, height=24)
    gol.set_initial_state(mode='glider')
    gol.evolve(15)
    gol_notes = gol.to_melody(c_major, base_duration=0.3)
    print(f"  Generated {len(gol_notes)} notes from evolving glider")
    notes_to_midi(gol_notes, "/tmp/ca_game_of_life.mid")

    # Musical CA
    print("\n[2.5] Musical CA - Custom Melodic Rules")
    mca = MusicalCA(width=32)
    mca.set_initial_state(mode='seed')
    mca.evolve(16, rule_type='melodic')
    print("  Musical CA Evolution (first 4 generations):")
    for gen in mca.history[:4]:
        print("  " + ' '.join([str(cell) for cell in gen]))


def demonstrate_constraint_solver():
    """Demonstrate Constraint Satisfaction Problem solver"""
    print("\n" + "=" * 80)
    print("3. CONSTRAINT SATISFACTION PROBLEM SOLVER")
    print("=" * 80)

    c_major = Scale(MIDDLE_C, ScaleType.MAJOR)
    a_minor = Scale(MIDDLE_C + 9, ScaleType.NATURAL_MINOR)

    # Balanced melody
    print("\n[3.1] Balanced Melody (Stepwise, Arch Contour)")
    csp_balanced = MelodicCSP(c_major, num_notes=12, min_pitch=60, max_pitch=79)
    balanced_melody = csp_balanced.generate(style='balanced')
    print(f"  Generated: {[note_number_to_name(n) for n in balanced_melody]}")

    # Convert to Note objects for MIDI
    balanced_notes = []
    for i, pitch in enumerate(balanced_melody):
        from algorithms.lsystem import Note
        balanced_notes.append(Note(pitch=pitch, start=i*0.5, duration=0.5, velocity=80))
    notes_to_midi(balanced_notes, "/tmp/csp_balanced.mid")

    # Ascending melody
    print("\n[3.2] Ascending Melody")
    ascending = csp_balanced.generate(style='ascending')
    print(f"  Generated: {[note_number_to_name(n) for n in ascending]}")

    ascending_notes = []
    for i, pitch in enumerate(ascending):
        from algorithms.lsystem import Note
        ascending_notes.append(Note(pitch=pitch, start=i*0.5, duration=0.5, velocity=80))
    notes_to_midi(ascending_notes, "/tmp/csp_ascending.mid")

    # Arch-shaped melody
    print("\n[3.3] Arch-Shaped Melody (Ascending then Descending)")
    arch = csp_balanced.generate(style='arch')
    print(f"  Generated: {[note_number_to_name(n) for n in arch]}")

    arch_notes = []
    for i, pitch in enumerate(arch):
        from algorithms.lsystem import Note
        arch_notes.append(Note(pitch=pitch, start=i*0.5, duration=0.5, velocity=80))
    notes_to_midi(arch_notes, "/tmp/csp_arch.mid")

    # Minor key melody
    print("\n[3.4] Minor Key Melody (Balanced)")
    csp_minor = MelodicCSP(a_minor, num_notes=12, min_pitch=57, max_pitch=76)
    minor_melody = csp_minor.generate(style='balanced')
    print(f"  Generated: {[note_number_to_name(n) for n in minor_melody]}")

    minor_notes = []
    for i, pitch in enumerate(minor_melody):
        from algorithms.lsystem import Note
        minor_notes.append(Note(pitch=pitch, start=i*0.5, duration=0.5, velocity=80))
    notes_to_midi(minor_notes, "/tmp/csp_minor.mid")


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("AGENT 2 COMPREHENSIVE ALGORITHM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo showcases three cutting-edge melody generation algorithms:")
    print("  1. L-Systems (Lindenmayer Systems) - Grammar-based generation")
    print("  2. Cellular Automata - Emergent pattern generation")
    print("  3. Constraint Satisfaction - Rule-based composition")
    print("\nAll generated melodies will be exported to /tmp/ as MIDI files.")
    print("=" * 80)

    # Run demonstrations
    demonstrate_lsystems()
    demonstrate_cellular_automata()
    demonstrate_constraint_solver()

    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nGenerated MIDI files (in /tmp/):")
    print("  L-Systems:")
    print("    - lsystem_bach.mid (Bach chorale style)")
    print("    - lsystem_minimalist.mid (Glass/Reich style)")
    print("    - lsystem_bebop.mid (Jazz bebop)")
    print("    - lsystem_fractal.mid (Fractal self-similar)")
    print("    - lsystem_romantic.mid (Romantic expression)")
    print("\n  Cellular Automata:")
    print("    - ca_rule30.mid (Chaotic)")
    print("    - ca_rule110.mid (Turing complete)")
    print("    - ca_rule90.mid (Sierpinski fractal)")
    print("    - ca_game_of_life.mid (Glider pattern)")
    print("\n  Constraint Solver:")
    print("    - csp_balanced.mid (Balanced melody)")
    print("    - csp_ascending.mid (Ascending)")
    print("    - csp_arch.mid (Arch-shaped)")
    print("    - csp_minor.mid (Minor key)")
    print("\nTotal: ~2,000 lines of advanced algorithmic composition code!")
    print("=" * 80)


if __name__ == "__main__":
    main()
