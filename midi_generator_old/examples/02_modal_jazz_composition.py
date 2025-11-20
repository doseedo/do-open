#!/usr/bin/env python3
"""
Example 2: Modal Jazz Composition
==================================

Demonstrates modal harmony techniques for jazz composition in the
style of Miles Davis, John Coltrane, and Bill Evans.

Techniques used:
- Dorian and Lydian modal progressions
- Modal vamps (two-chord oscillations)
- Modal interchange (borrowed chords)
- Pedal point harmony

Musical context: Modal jazz emerged in the late 1950s (Kind of Blue)
as an alternative to bebop's complex chord changes, focusing on
exploration of single modes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.modal_harmony import Mode, HarmonicMinorMode, MelodicMinorMode
from generators.advanced_harmony_generator import AdvancedHarmonyGenerator


def main():
    print("=" * 70)
    print("MODAL JAZZ COMPOSITION")
    print("=" * 70)

    # Example 1: D Dorian vamp (So What style)
    print("\n1. D DORIAN VAMP ('So What' style)")
    print("-" * 70)
    gen_d = AdvancedHarmonyGenerator(root=2, octave=4)  # D root
    dorian_vamp = gen_d.generate_modal_progression(Mode.DORIAN, "vamp", length=8)
    print(f"Mode: {dorian_vamp['mode']}")
    print(f"Brightness: {dorian_vamp['brightness']}/7 (darker minor)")
    print(f"Characteristic degrees: {dorian_vamp['characteristic_degrees']}")
    print(f"Number of chords: {len(dorian_vamp['midi_notes'])}")
    print("\nThe major 6th degree gives Dorian its characteristic sound")

    # Example 2: F Lydian progression (bright, ethereal)
    print("\n2. F LYDIAN PROGRESSION (Bright, ethereal)")
    print("-" * 70)
    gen_f = AdvancedHarmonyGenerator(root=5, octave=4)  # F root
    lydian_prog = gen_f.generate_modal_progression(Mode.LYDIAN, "plagal", length=4)
    print(f"Mode: {lydian_prog['mode']}")
    print(f"Brightness: {lydian_prog['brightness']}/7 (brightest mode)")
    print(f"Characteristic: Raised 4th (♯11) creates dreamy quality")
    print(f"Chords: {len(lydian_prog['midi_notes'])}")

    # Example 3: G Mixolydian (bluesy, rock-influenced)
    print("\n3. G MIXOLYDIAN PROGRESSION (Bluesy)")
    print("-" * 70)
    gen_g = AdvancedHarmonyGenerator(root=7, octave=4)  # G root
    mixo_prog = gen_g.generate_modal_progression(Mode.MIXOLYDIAN, "characteristic", length=4)
    print(f"Mode: {mixo_prog['mode']}")
    print(f"Brightness: {mixo_prog['brightness']}/7")
    print(f"Characteristic: ♭7 gives major scale a bluesy edge")
    print(f"Common in: Rock, blues, folk")

    # Example 4: Modal interchange (borrowing)
    print("\n4. MODAL INTERCHANGE (C Major borrowing from C Minor)")
    print("-" * 70)
    gen_c = AdvancedHarmonyGenerator(root=0, octave=4)  # C root
    interchange = gen_c.generate_modal_interchange(
        Mode.IONIAN,
        Mode.AEOLIAN,
        primary_degrees=[1, 6, 2, 5, 1],
        insert_positions=[1, 3]  # Borrow chords at positions 1 and 3
    )
    print(f"Primary mode: {interchange['primary_mode']}")
    print(f"Borrowed from: {interchange['borrowed_mode']}")
    print(f"Borrowed positions: {interchange['borrowed_positions']}")
    print(f"Total chords: {len(interchange['midi_notes'])}")
    print("\nBorrowed chords add color while maintaining tonal center")

    # Example 5: Phrygian dominant (Spanish/flamenco jazz)
    print("\n5. E PHRYGIAN DOMINANT (Spanish/Flamenco jazz)")
    print("-" * 70)
    gen_e = AdvancedHarmonyGenerator(root=4, octave=4)  # E root
    # Phrygian dominant is mode 5 of harmonic minor
    # We'll use modal progression for demonstration
    phrygian = gen_e.generate_modal_progression(Mode.PHRYGIAN, "characteristic", length=4)
    print(f"Mode: {phrygian['mode']}")
    print(f"Brightness: {phrygian['brightness']}/7 (very dark)")
    print(f"Characteristic: ♭2 creates exotic, Spanish flavor")
    print(f"Used by: Chick Corea, Miles Davis (Sketches of Spain)")

    # Example 6: Lydian Dominant (jazz dominant substitute)
    print("\n6. C LYDIAN DOMINANT (Jazz dominant chord)")
    print("-" * 70)
    # Lydian Dominant = mode 4 of melodic minor
    # Simulated here with modal generation
    print("Lydian Dominant: Major with ♯4 and ♭7")
    print("Intervals: 1 2 3 ♯4 5 6 ♭7")
    print("Chord: Dominant 7th with ♯11")
    print("Use: Over any dominant 7 chord for jazzy color")
    print("Example: C7♯11 in progression to F major")

    # Example 7: Brightness comparison
    print("\n7. MODE BRIGHTNESS COMPARISON")
    print("-" * 70)
    print("Darkest to Brightest:")
    print("  1. Locrian     (rarely used, diminished quality)")
    print("  2. Phrygian    (dark, Spanish/flamenco)")
    print("  3. Aeolian     (natural minor)")
    print("  4. Dorian      (minor with major 6th)")
    print("  5. Mixolydian  (major with ♭7, bluesy)")
    print("  6. Ionian      (major scale)")
    print("  7. Lydian      (brightest, raised 4th)")

    # Example 8: Modal cadences
    print("\n8. MODAL CADENCES (Non-functional endings)")
    print("-" * 70)
    print("Unlike functional harmony (V-I), modal music uses:")

    plagal = gen_c.generate_modal_cadence(Mode.IONIAN, "plagal")
    print(f"\nPlagal cadence (IV-I): {len(plagal['midi_notes'])} chords")
    print("  Creates gentle, peaceful resolution")

    phrygian_cad = gen_e.generate_modal_cadence(Mode.PHRYGIAN, "phrygian")
    print(f"\nPhrygian cadence (♭II-I): {len(phrygian_cad['midi_notes'])} chords")
    print("  Creates exotic, Spanish resolution")

    print("\n" + "=" * 70)
    print("MODAL JAZZ COMPOSITION TIPS:")
    print("- Use one mode per section (8-16 bars)")
    print("- Avoid V-I cadences (breaks modal character)")
    print("- Emphasize characteristic tones in melody")
    print("- Use pedal points and drones")
    print("- Modal interchange adds color without modulation")
    print("- Dorian and Mixolydian are most common in jazz")
    print("=" * 70)


if __name__ == "__main__":
    main()
