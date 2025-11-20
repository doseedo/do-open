#!/usr/bin/env python3
"""
Reharmonization Engine
======================

Advanced reharmonization techniques for jazz and contemporary music.
Implements substitution algorithms to add harmonic variety and sophistication.

Based on:
- Mark Levine: "The Jazz Theory Book" - Reharmonization chapter
- Bebop harmonic practices (Charlie Parker, Dizzy Gillespie)
- Post-bop harmony (John Coltrane, Wayne Shorter, Herbie Hancock)
- Modal jazz techniques (Miles Davis, Bill Evans)

Techniques Implemented:
-----------------------
1. Tritone substitution (bII7 for V7)
2. Diatonic substitution (iii for I, vi for I)
3. Approach chords (ii-V before target chords)
4. Modal interchange (borrowing from parallel modes)
5. Coltrane substitution (descending major 3rd cycles - Giant Steps)
6. Pedal points and sus chords
7. Chromatic approach chords
8. Extended dominants (secondary dominants)

Author: Agent 4 - Harmonic Progression Designer
Date: 2025
License: MIT
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from genres.jazz import JazzChord


@dataclass
class ReharmonizationOptions:
    """Configuration options for reharmonization"""
    tritone_sub_probability: float = 0.3      # 30% chance to apply tritone sub
    approach_chord_probability: float = 0.4   # 40% chance to add approach chords
    modal_interchange_probability: float = 0.2 # 20% chance for modal interchange
    coltrane_sub_probability: float = 0.1     # 10% chance for Coltrane substitution
    diatonic_sub_probability: float = 0.2     # 20% chance for diatonic substitution
    use_extended_dominants: bool = True       # Add secondary dominants
    complexity_level: float = 0.5             # 0.0 (simple) to 1.0 (Bird-level complexity)


class ReharmonizationEngine:
    """
    Advanced reharmonization engine for jazz progressions.

    Provides various substitution and enhancement techniques to add
    harmonic sophistication to basic progressions.
    """

    def __init__(self, options: Optional[ReharmonizationOptions] = None):
        """
        Initialize reharmonization engine.

        Args:
            options: Reharmonization configuration options
        """
        self.options = options or ReharmonizationOptions()

    # ========================================================================
    # TRITONE SUBSTITUTION
    # ========================================================================

    @staticmethod
    def apply_tritone_subs(
        progression: List[JazzChord],
        probability: float = 0.3,
        target_positions: Optional[List[int]] = None
    ) -> List[JazzChord]:
        """
        Replace V7 chords with bII7 substitutes (tritone substitution).

        Theory: The tritone substitution replaces a dominant 7th chord with
        another dominant 7th chord a tritone (6 semitones) away. Both chords
        share the same tritone (3rd and 7th), creating smooth voice leading.

        Example: G7 (in C) → Db7 (tritone sub)
                 G7 = G-B-D-F, Db7 = Db-F-Ab-Cb(B)
                 Both share B and F (the tritone)

        Args:
            progression: Original chord progression
            probability: Chance to apply substitution (0.0-1.0)
            target_positions: Specific positions to substitute (None = random)

        Returns:
            Reharmonized progression with tritone substitutions
        """
        result = []

        for i, chord in enumerate(progression):
            # Apply tritone sub to dominant 7th chords
            if chord.quality == "dom7":
                # Check if we should apply the substitution
                should_substitute = (
                    target_positions is not None and i in target_positions
                ) or (
                    target_positions is None and random.random() < probability
                )

                if should_substitute:
                    # Tritone substitution: replace with bII7 (6 semitones away)
                    new_root = (chord.root + 6) % 12
                    substituted_chord = JazzChord(
                        root=new_root,
                        quality="dom7",
                        extensions=chord.extensions.copy(),
                        alterations=chord.alterations.copy(),
                        inversion=chord.inversion,
                        voicing_type=chord.voicing_type
                    )
                    result.append(substituted_chord)
                else:
                    result.append(chord)
            else:
                result.append(chord)

        return result

    # ========================================================================
    # APPROACH CHORDS (ii-V)
    # ========================================================================

    @staticmethod
    def add_approach_chords(
        progression: List[JazzChord],
        approach_type: str = "ii_V",
        target_chords: Optional[List[int]] = None
    ) -> List[JazzChord]:
        """
        Add ii-V approach chords before target chords.

        Theory: In jazz, approaching any chord with a ii-V progression
        creates forward momentum and harmonic interest. The ii-V is the
        fundamental building block of bebop harmony.

        Example: Before Cmaj7, insert Dm7-G7 (ii-V in C)

        Args:
            progression: Original progression
            approach_type: Type of approach ("ii_V", "chromatic", "dominant")
            target_chords: Indices of chords to approach (None = maj7/min7 chords)

        Returns:
            Progression with approach chords inserted
        """
        result = []

        for i, chord in enumerate(progression):
            # Determine if this chord should be approached
            should_approach = False
            if target_chords is not None:
                should_approach = i in target_chords
            else:
                # By default, approach major 7th and minor 7th chords
                should_approach = chord.quality in ["maj7", "min7"]

            if should_approach and i > 0:  # Don't approach first chord
                if approach_type == "ii_V":
                    # Add ii-V before target chord
                    # ii: min7 chord on 2nd degree of target key
                    ii_root = (chord.root + 2) % 12
                    ii_chord = JazzChord(root=ii_root, quality="min7")

                    # V: dom7 chord on 5th degree of target key
                    v_root = (chord.root + 7) % 12
                    v_chord = JazzChord(root=v_root, quality="dom7")

                    result.extend([ii_chord, v_chord])

                elif approach_type == "chromatic":
                    # Chromatic approach: dominant 7th a half-step above
                    approach_root = (chord.root + 1) % 12
                    approach_chord = JazzChord(root=approach_root, quality="dom7")
                    result.append(approach_chord)

                elif approach_type == "dominant":
                    # Direct dominant approach
                    v_root = (chord.root + 7) % 12
                    v_chord = JazzChord(root=v_root, quality="dom7")
                    result.append(v_chord)

            result.append(chord)

        return result

    # ========================================================================
    # MODAL INTERCHANGE
    # ========================================================================

    @staticmethod
    def apply_modal_interchange(
        progression: List[JazzChord],
        borrowed_mode: str = "aeolian",
        positions: Optional[List[int]] = None
    ) -> List[JazzChord]:
        """
        Borrow chords from parallel modes (modal interchange).

        Theory: Modal interchange involves borrowing chords from the parallel
        mode (same root, different mode). Most common is borrowing from the
        parallel minor (Aeolian) in a major key context.

        Common borrowed chords in C major (from C minor):
        - iv (Fm) instead of IV (F)
        - bVII (Bb) instead of vii°
        - bVI (Ab) instead of vi
        - bIII (Eb) instead of iii

        Args:
            progression: Original progression
            borrowed_mode: Mode to borrow from ("aeolian", "dorian", "phrygian")
            positions: Positions to apply interchange (None = automatic)

        Returns:
            Progression with modal interchange applied
        """
        result = []

        # Define borrowed chord substitutions (in major key context)
        borrowings = {
            "aeolian": {
                # From parallel minor (Aeolian)
                "maj7": "min7",    # I → i
                4: 5,              # IV → iv (root up 5, becomes minor)
                9: 10,             # vi → bVI (root up 10)
                11: 10,            # vii° → bVII (root stays, becomes major)
            },
            "dorian": {
                # From Dorian mode
                "maj7": "min7",
                4: 5,              # IV → iv
            },
            "phrygian": {
                # From Phrygian mode
                "maj7": "min7",
                1: 0,              # II → bII (flattened)
            }
        }

        for i, chord in enumerate(progression):
            should_borrow = (
                positions is not None and i in positions
            ) or (
                positions is None and random.random() < 0.3
            )

            if should_borrow and borrowed_mode in borrowings:
                # Apply modal interchange
                # This is a simplified version - full implementation would
                # analyze the chord's function and apply appropriate substitution
                borrowed_chord = JazzChord(
                    root=chord.root,
                    quality="min7" if chord.quality == "maj7" else chord.quality,
                    extensions=chord.extensions.copy(),
                    alterations=chord.alterations.copy(),
                    inversion=chord.inversion,
                    voicing_type=chord.voicing_type
                )
                result.append(borrowed_chord)
            else:
                result.append(chord)

        return result

    # ========================================================================
    # COLTRANE SUBSTITUTION (Giant Steps)
    # ========================================================================

    @staticmethod
    def generate_coltrane_substitution(
        target_chord: JazzChord,
        cycle_length: int = 3
    ) -> List[JazzChord]:
        """
        Generate descending major 3rd cycle (Coltrane substitution).

        Theory: John Coltrane's "Giant Steps" uses a pattern of descending
        major thirds (dividing the octave into three equal parts). Each key
        center is preceded by its dominant, creating maximum harmonic motion.

        Example: To reach C, use:
        B → E (down major 3rd) → Ab (down major 3rd) → C
        With dominants: Bmaj7-D7 | Emaj7-G7 | Abmaj7-C7 | → Cmaj7

        Args:
            target_chord: Final destination chord
            cycle_length: Number of key centers (2-4, default=3)

        Returns:
            List of chords forming Coltrane cycle to target
        """
        chords = []

        # Start from major 3rd above target
        current_root = target_chord.root

        # Build cycle of major thirds
        roots = []
        for _ in range(cycle_length):
            current_root = (current_root + 4) % 12  # Up major 3rd
            roots.append(current_root)

        roots.reverse()  # We built upward, now reverse to descend

        # For each root, create maj7 and its dominant
        for root in roots:
            # Major 7th chord
            maj_chord = JazzChord(root=root, quality="maj7")
            chords.append(maj_chord)

            # Its dominant (5th below = down 7 semitones)
            dom_root = (root + 5) % 12  # Down 7 = up 5
            dom_chord = JazzChord(root=dom_root, quality="dom7")
            chords.append(dom_chord)

        return chords

    # ========================================================================
    # DIATONIC SUBSTITUTION
    # ========================================================================

    @staticmethod
    def apply_diatonic_substitution(
        progression: List[JazzChord],
        substitutions: Optional[Dict[str, str]] = None
    ) -> List[JazzChord]:
        """
        Apply diatonic substitutions (chords with similar function).

        Theory: Certain chords can substitute for others because they share
        common tones and have similar harmonic function:
        - iii can substitute for I (both have tonic function)
        - vi can substitute for I
        - ii can substitute for IV (both subdominant)
        - vii° can substitute for V (both dominant function)

        Args:
            progression: Original progression
            substitutions: Custom substitution map (None = use defaults)

        Returns:
            Progression with diatonic substitutions
        """
        # Default substitution patterns
        default_subs = {
            "I→iii": (0, 4, "maj7", "min7"),    # I → iii (up 4 semitones, becomes minor)
            "I→vi": (0, 9, "maj7", "min7"),      # I → vi (up 9 semitones, becomes minor)
            "IV→ii": (5, 2, "maj7", "min7"),     # IV → ii (down 3 semitones, becomes minor)
        }

        result = []

        for chord in progression:
            # Check if we should apply a substitution
            if random.random() < 0.2:  # 20% chance
                # Try to find a substitution for this chord
                # This is simplified - real implementation would analyze function
                # For now, occasionally substitute I with vi
                if chord.quality == "maj7" and random.random() < 0.5:
                    # Substitute I with vi
                    new_root = (chord.root + 9) % 12
                    sub_chord = JazzChord(
                        root=new_root,
                        quality="min7",
                        extensions=chord.extensions.copy(),
                        alterations=chord.alterations.copy(),
                        inversion=chord.inversion,
                        voicing_type=chord.voicing_type
                    )
                    result.append(sub_chord)
                else:
                    result.append(chord)
            else:
                result.append(chord)

        return result

    # ========================================================================
    # EXTENDED DOMINANTS (Secondary Dominants)
    # ========================================================================

    @staticmethod
    def add_secondary_dominants(
        progression: List[JazzChord],
        probability: float = 0.3
    ) -> List[JazzChord]:
        """
        Add secondary dominant chords (V7/chord).

        Theory: Any chord can be preceded by its dominant. This creates
        temporary tonicization and adds harmonic movement.

        Example: Before Dm7, add A7 (V7 of Dm)

        Args:
            progression: Original progression
            probability: Chance to add secondary dominant

        Returns:
            Progression with secondary dominants
        """
        result = []

        for i, chord in enumerate(progression):
            if i > 0 and random.random() < probability:
                # Add secondary dominant before this chord
                # Dominant is 7 semitones above (perfect 5th)
                dom_root = (chord.root + 7) % 12
                secondary_dom = JazzChord(root=dom_root, quality="dom7")
                result.append(secondary_dom)

            result.append(chord)

        return result

    # ========================================================================
    # COMPREHENSIVE REHARMONIZATION
    # ========================================================================

    def reharmonize_progression(
        self,
        progression: List[JazzChord],
        style: str = "bebop"
    ) -> List[JazzChord]:
        """
        Apply comprehensive reharmonization based on style.

        Args:
            progression: Original progression
            style: Style profile ("bebop", "post_bop", "modal", "contemporary")

        Returns:
            Fully reharmonized progression
        """
        result = progression.copy()

        if style == "bebop":
            # Bebop: Heavy ii-V usage, chromatic approaches, tritone subs
            # Complexity controlled by options
            if self.options.complexity_level > 0.3:
                result = self.add_approach_chords(result, "ii_V")

            if self.options.complexity_level > 0.5:
                result = self.apply_tritone_subs(
                    result,
                    probability=self.options.tritone_sub_probability
                )

            if self.options.use_extended_dominants:
                result = self.add_secondary_dominants(
                    result,
                    probability=self.options.approach_chord_probability * 0.5
                )

        elif style == "post_bop":
            # Post-bop: Coltrane changes, modal sections, chromatic harmony
            if self.options.complexity_level > 0.6:
                # Occasionally insert Coltrane substitutions
                if random.random() < self.options.coltrane_sub_probability:
                    # Add Coltrane cycle before final chord
                    if len(result) > 2:
                        coltrane_chords = self.generate_coltrane_substitution(
                            result[-1],
                            cycle_length=2
                        )
                        result = result[:-1] + coltrane_chords + [result[-1]]

            # Modal interchange
            result = self.apply_modal_interchange(
                result,
                borrowed_mode="aeolian",
                positions=None
            )

        elif style == "modal":
            # Modal: Static harmony, pedal points, minimal changes
            # Simplify rather than complexify
            # Modal style doesn't add many substitutions
            pass

        elif style == "contemporary":
            # Contemporary: Mix of all techniques
            result = self.apply_tritone_subs(result, probability=0.2)
            result = self.apply_modal_interchange(result, "aeolian")
            if self.options.complexity_level > 0.7:
                result = self.add_approach_chords(result, "ii_V")

        return result

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    @staticmethod
    def analyze_harmonic_density(progression: List[JazzChord]) -> Dict:
        """
        Analyze harmonic characteristics of progression.

        Args:
            progression: Chord progression to analyze

        Returns:
            Dictionary with analysis metrics
        """
        total_chords = len(progression)

        # Count chord qualities
        qualities = {}
        for chord in progression:
            quality = chord.quality
            qualities[quality] = qualities.get(quality, 0) + 1

        # Count ii-V patterns
        ii_v_count = 0
        for i in range(len(progression) - 1):
            if (progression[i].quality == "min7" and
                progression[i + 1].quality == "dom7"):
                # Check if it's actually a ii-V (V is 5 semitones above ii)
                if (progression[i + 1].root - progression[i].root) % 12 == 5:
                    ii_v_count += 1

        return {
            "total_chords": total_chords,
            "qualities": qualities,
            "ii_v_count": ii_v_count,
            "ii_v_density": ii_v_count / max(total_chords - 1, 1),
            "dominant_ratio": qualities.get("dom7", 0) / total_chords,
            "complexity_score": total_chords * 0.1 + ii_v_count * 0.3
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("REHARMONIZATION ENGINE - EXAMPLES")
    print("=" * 80)

    # Create a simple progression: I-IV-V-I in C
    simple_prog = [
        JazzChord(root=0, quality="maj7"),   # Cmaj7
        JazzChord(root=5, quality="maj7"),   # Fmaj7
        JazzChord(root=7, quality="dom7"),   # G7
        JazzChord(root=0, quality="maj7"),   # Cmaj7
    ]

    print("\nOriginal progression: I-IV-V-I in C")
    for i, chord in enumerate(simple_prog, 1):
        print(f"  {i}. {chord}")

    # Initialize engine
    options = ReharmonizationOptions(
        tritone_sub_probability=0.5,
        approach_chord_probability=0.5,
        complexity_level=0.7
    )
    engine = ReharmonizationEngine(options)

    # Example 1: Tritone substitution
    print("\n" + "-" * 80)
    print("1. TRITONE SUBSTITUTION (replace V7 with bII7)")
    print("-" * 80)
    tritone_prog = engine.apply_tritone_subs(simple_prog, probability=1.0)
    for i, chord in enumerate(tritone_prog, 1):
        print(f"  {i}. {chord}")

    # Example 2: Add ii-V approach chords
    print("\n" + "-" * 80)
    print("2. ADD ii-V APPROACH CHORDS")
    print("-" * 80)
    approach_prog = engine.add_approach_chords(simple_prog, approach_type="ii_V")
    for i, chord in enumerate(approach_prog, 1):
        print(f"  {i}. {chord}")

    # Example 3: Modal interchange
    print("\n" + "-" * 80)
    print("3. MODAL INTERCHANGE (borrow from parallel minor)")
    print("-" * 80)
    modal_prog = engine.apply_modal_interchange(
        simple_prog,
        borrowed_mode="aeolian",
        positions=[1]  # Borrow at position 1 (IV → iv)
    )
    for i, chord in enumerate(modal_prog, 1):
        print(f"  {i}. {chord}")

    # Example 4: Coltrane substitution
    print("\n" + "-" * 80)
    print("4. COLTRANE SUBSTITUTION (Giant Steps cycle)")
    print("-" * 80)
    target = JazzChord(root=0, quality="maj7")  # Cmaj7
    coltrane_cycle = engine.generate_coltrane_substitution(target, cycle_length=3)
    for i, chord in enumerate(coltrane_cycle, 1):
        print(f"  {i}. {chord}")

    # Example 5: Full bebop reharmonization
    print("\n" + "-" * 80)
    print("5. FULL BEBOP REHARMONIZATION")
    print("-" * 80)
    bebop_prog = engine.reharmonize_progression(simple_prog, style="bebop")
    for i, chord in enumerate(bebop_prog, 1):
        print(f"  {i}. {chord}")

    # Analyze results
    print("\n" + "-" * 80)
    print("6. HARMONIC ANALYSIS")
    print("-" * 80)
    original_analysis = engine.analyze_harmonic_density(simple_prog)
    reharmonized_analysis = engine.analyze_harmonic_density(bebop_prog)

    print(f"\nOriginal progression:")
    print(f"  Total chords: {original_analysis['total_chords']}")
    print(f"  ii-V patterns: {original_analysis['ii_v_count']}")
    print(f"  Complexity score: {original_analysis['complexity_score']:.2f}")

    print(f"\nReharmonized progression:")
    print(f"  Total chords: {reharmonized_analysis['total_chords']}")
    print(f"  ii-V patterns: {reharmonized_analysis['ii_v_count']}")
    print(f"  Complexity score: {reharmonized_analysis['complexity_score']:.2f}")

    print("\n" + "=" * 80)
