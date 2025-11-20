#!/usr/bin/env python3
"""
Example 1: Neo-Riemannian Film Score Progression
=================================================

Demonstrates Neo-Riemannian transformations for creating smooth,
chromatic progressions common in contemporary film scoring.

Techniques used:
- PLR transformations for minimal voice leading
- Hexatonic cycles for dramatic shifts
- Chromatic mediant relationships
- Voice leading analysis

Musical context: This style is common in John Williams, Hans Zimmer,
and other modern film composers who use chromatic harmony with
smooth voice leading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.neo_riemannian import Triad, TriadQuality, TransformationChain, HexatonicSystem
from generators.advanced_harmony_generator import AdvancedHarmonyGenerator


def main():
    print("=" * 70)
    print("NEO-RIEMANNIAN FILM SCORE PROGRESSION")
    print("=" * 70)

    gen = AdvancedHarmonyGenerator(root=0, octave=4)  # C root

    # Example 1: Classic PLR progression (Williams-style)
    print("\n1. CLASSIC PLR PROGRESSION")
    print("-" * 70)
    prog1 = gen.generate_neo_riemannian("P L R P", voice_lead=True)
    print(f"Progression: {' → '.join(prog1['triads'])}")
    print(f"Total voice motion: {prog1['voice_leading_analysis']['total_motion']} semitones")
    print(f"Average per step: {prog1['voice_leading_analysis']['avg_motion']:.2f} semitones")
    print(f"\nMIDI notes:")
    for i, (triad, notes) in enumerate(zip(prog1['triads'], prog1['midi_notes'])):
        print(f"  {triad:6} {notes}")

    # Example 2: Dramatic hexatonic shift
    print("\n2. HEXATONIC CYCLE (Dramatic chromatic motion)")
    print("-" * 70)
    hex_prog = gen.generate_hexatonic_cycle(pole=0)  # Northern hexatonic
    print(f"System: {hex_prog['pole']} Hexatonic")
    print(f"Cycle: {' → '.join(hex_prog['triads'])}")
    print(f"Pitch classes: {hex_prog['pitch_classes']}")
    print("\nThis creates smooth voice leading through chromatic space")

    # Example 3: Chromatic mediant progression (epic trailer style)
    print("\n3. CHROMATIC MEDIANT PROGRESSION (Epic/Trailer style)")
    print("-" * 70)
    mediant_prog = gen.generate_chromatic_mediant_prog("UCM UFM LCM")
    print(f"Pattern: {mediant_prog['pattern']}")
    print(f"Progression: {' → '.join(mediant_prog['triads'])}")
    print("\nThis creates large, dramatic jumps common in trailer music")

    # Example 4: Complex transformation chain
    print("\n4. COMPLEX TRANSFORMATION CHAIN")
    print("-" * 70)
    complex_prog = gen.generate_neo_riemannian("L P L P R L", voice_lead=True)
    print(f"Transformations: {complex_prog['transformations']}")
    print(f"Progression: {' → '.join(complex_prog['triads'])}")
    print(f"Voice leading efficiency: {complex_prog['voice_leading_analysis']['avg_motion']:.2f} semitones")
    print("\nMotion per step:", complex_prog['voice_leading_analysis']['motion_per_step'])

    # Example 5: Film cue structure (A-B-A form with transformations)
    print("\n5. FILM CUE STRUCTURE (A-B-A with Neo-Riemannian)")
    print("-" * 70)
    print("Section A (Heroic):")
    section_a = gen.generate_neo_riemannian("P L P", voice_lead=True)
    print(f"  {' → '.join(section_a['triads'])}")

    print("\nSection B (Dark/Tension):")
    gen_b = AdvancedHarmonyGenerator(root=section_a['midi_notes'][-1][0] % 12, octave=4)
    section_b = gen_b.generate_neo_riemannian("R L R L", voice_lead=True)
    print(f"  {' → '.join(section_b['triads'])}")

    print("\nSection A' (Return):")
    print(f"  {' → '.join(section_a['triads'])}")

    print("\n" + "=" * 70)
    print("USAGE NOTES:")
    print("- P transformation: Minimal motion (mode change)")
    print("- L transformation: Chromatic shift, preserves third")
    print("- R transformation: Relative relationship")
    print("- Hexatonic cycles: Perfect for dramatic scene shifts")
    print("- Chromatic mediants: Epic, larger-than-life quality")
    print("=" * 70)


if __name__ == "__main__":
    main()
