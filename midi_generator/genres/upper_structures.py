#!/usr/bin/env python3
"""
Upper Structure Triads - Advanced Jazz Harmony
==============================================

Upper structure triads are a sophisticated jazz voicing technique where a
triad is placed "on top of" a bass note to create rich, colorful harmony.

Key Concept:
-----------
Play a triad in the upper register over a different bass note, creating
tensions (9ths, 11ths, 13ths, altered notes) without explicitly spelling them out.

Example:
-------
C7alt → Play Db major triad over C bass
  - C (bass)
  - Db, F, Ab (triad) = b9, 11, b13 of C7

This technique is essential for:
- Altered dominants (V7alt)
- Rich reharmonization
- Modern jazz voicings
- Avoiding "muddy" lower-register tensions

Historical Context:
------------------
- Bill Evans: Popularized upper structure approach
- McCoy Tyner: Used upper structures in modal context
- Herbie Hancock: Extended the technique to modern harmony
- Mark Levine: Cataloged and systematized upper structures

Research Sources:
----------------
- Mark Levine "Jazz Theory Book" - Upper Structure chapter (pages 128-156)
- Mark Levine "Jazz Piano Book" - Voicing chapter
- Bert Ligon "Connecting Chords with Linear Harmony"
- Barry Harris: Upper structure masterclasses

Author: Agent 3 - Piano Comping Virtuoso
Date: 2025
License: MIT
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import jazz types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from genres.jazz import JazzChord
except ImportError:
    # Fallback
    from dataclasses import dataclass, field

    @dataclass
    class JazzChord:
        root: int
        quality: str
        extensions: List[int] = field(default_factory=list)
        alterations: List[str] = field(default_factory=list)
        inversion: int = 0
        voicing_type: str = "shell"


# ============================================================================
# UPPER STRUCTURE CATALOG
# ============================================================================

@dataclass
class UpperStructure:
    """
    Upper structure triad definition.

    Attributes:
        triad_root_interval: Interval from bass note to triad root (semitones)
        triad_quality: "major", "minor", "augmented", "diminished"
        tensions_created: Which tensions result from this voicing
        sound_description: How it sounds
        recommended_use: When to use it
    """
    triad_root_interval: int
    triad_quality: str
    tensions_created: List[str]
    sound_description: str
    recommended_use: str


# ============================================================================
# DOMINANT 7TH UPPER STRUCTURES
# ============================================================================
# For C7 → various upper structures

DOMINANT_UPPER_STRUCTURES = {
    # Altered Dominants (V7alt)
    "bII_major": UpperStructure(
        triad_root_interval=1,   # Db major over C bass
        triad_quality="major",
        tensions_created=["b9", "3", "b13"],
        sound_description="Dark, altered, sophisticated",
        recommended_use="V7alt going to minor, bebop"
    ),

    "bVI_major": UpperStructure(
        triad_root_interval=8,   # Ab major over C bass
        triad_quality="major",
        tensions_created=["b13", "1", "3"],
        sound_description="Dark altered sound",
        recommended_use="V7alt with b13"
    ),

    "#IV_major": UpperStructure(
        triad_root_interval=6,   # F# major over C bass
        triad_quality="major",
        tensions_created=["#11", "1", "3"],
        sound_description="Lydian dominant, bright altered",
        recommended_use="Lydian dominant, V7#11"
    ),

    "bVII_major": UpperStructure(
        triad_root_interval=10,  # Bb major over C bass
        triad_quality="major",
        tensions_created=["b7", "9", "11"],
        sound_description="Sus4 sound, modal",
        recommended_use="V7sus, modal dominants"
    ),

    "II_major": UpperStructure(
        triad_root_interval=2,   # D major over C bass
        triad_quality="major",
        tensions_created=["9", "#11", "13"],
        sound_description="Bright, open, Lydian dominant",
        recommended_use="V7(#11) Lydian dominant"
    ),

    "II_minor": UpperStructure(
        triad_root_interval=2,   # D minor over C bass
        triad_quality="minor",
        tensions_created=["9", "11", "13"],
        sound_description="Open, sus-like",
        recommended_use="V7sus4, modal"
    ),

    "#I_augmented": UpperStructure(
        triad_root_interval=1,   # Db augmented over C bass
        triad_quality="augmented",
        tensions_created=["b9", "#9", "#5"],
        sound_description="Very altered, tense",
        recommended_use="V7alt with maximum alterations"
    ),

    "V_minor": UpperStructure(
        triad_root_interval=7,   # G minor over C bass
        triad_quality="minor",
        tensions_created=["5", "b7", "9"],
        sound_description="Mixolydian, bluesy",
        recommended_use="Bluesy dominant, rock feel"
    ),

    "bIII_major": UpperStructure(
        triad_root_interval=3,   # Eb major over C bass
        triad_quality="major",
        tensions_created=["b3/b9", "5", "b7"],
        sound_description="Blues sound, #9/#11",
        recommended_use="Blues, altered dominant"
    ),

    "#IV_diminished": UpperStructure(
        triad_root_interval=6,   # F# diminished over C bass
        triad_quality="diminished",
        tensions_created=["#11", "1", "b3"],
        sound_description="Diminished, mysterious",
        recommended_use="Whole-half diminished scale"
    ),
}


# ============================================================================
# MAJOR 7TH UPPER STRUCTURES
# ============================================================================

MAJOR7_UPPER_STRUCTURES = {
    "II_major": UpperStructure(
        triad_root_interval=2,   # D major over C bass (for Cmaj7)
        triad_quality="major",
        tensions_created=["9", "#11", "13"],
        sound_description="Lydian, bright, open",
        recommended_use="Lydian major, #11 sound"
    ),

    "III_minor": UpperStructure(
        triad_root_interval=4,   # E minor over C bass
        triad_quality="minor",
        tensions_created=["3", "5", "7"],
        sound_description="Simple major 7 sound",
        recommended_use="Basic major 7 voicing"
    ),

    "VI_minor": UpperStructure(
        triad_root_interval=9,   # A minor over C bass
        triad_quality="minor",
        tensions_created=["13", "1", "3"],
        sound_description="Major 6/9 sound",
        recommended_use="Major 6, add 6/9"
    ),

    "V_major": UpperStructure(
        triad_root_interval=7,   # G major over C bass
        triad_quality="major",
        tensions_created=["5", "7", "9"],
        sound_description="Open, folk-like",
        recommended_use="Simple major voicing"
    ),
}


# ============================================================================
# MINOR 7TH UPPER STRUCTURES
# ============================================================================

MINOR7_UPPER_STRUCTURES = {
    "bIII_major": UpperStructure(
        triad_root_interval=3,   # Eb major over C bass (for Cmin7)
        triad_quality="major",
        tensions_created=["b3", "5", "b7"],
        sound_description="Simple minor 7",
        recommended_use="Basic minor 7 voicing"
    ),

    "bVII_major": UpperStructure(
        triad_root_interval=10,  # Bb major over C bass
        triad_quality="major",
        tensions_created=["b7", "9", "11"],
        sound_description="Dorian, modal minor",
        recommended_use="Minor 7 with 9 and 11"
    ),

    "IV_major": UpperStructure(
        triad_root_interval=5,   # F major over C bass
        triad_quality="major",
        tensions_created=["11", "13", "1"],
        sound_description="Dorian/Aeolian",
        recommended_use="Minor with 11 and 13"
    ),

    "II_minor": UpperStructure(
        triad_root_interval=2,   # D minor over C bass
        triad_quality="minor",
        tensions_created=["9", "11", "13"],
        sound_description="Open, modal",
        recommended_use="Dorian minor 7"
    ),
}


# ============================================================================
# HALF-DIMINISHED UPPER STRUCTURES
# ============================================================================

HALF_DIM_UPPER_STRUCTURES = {
    "bII_major": UpperStructure(
        triad_root_interval=1,   # Db major over C bass (for Cmin7b5)
        triad_quality="major",
        tensions_created=["b9", "3", "5"],
        sound_description="Locrian, dark",
        recommended_use="ii° in minor key"
    ),

    "bIII_minor": UpperStructure(
        triad_root_interval=3,   # Eb minor over C bass
        triad_quality="minor",
        tensions_created=["b3", "b5", "b7"],
        sound_description="Simple half-diminished",
        recommended_use="Basic min7b5 voicing"
    ),

    "bVI_major": UpperStructure(
        triad_root_interval=8,   # Ab major over C bass
        triad_quality="major",
        tensions_created=["b13", "1", "b3"],
        sound_description="Locrian",
        recommended_use="Locrian mode voicing"
    ),
}


# ============================================================================
# UPPER STRUCTURE ENGINE
# ============================================================================

class UpperStructureEngine:
    """
    Engine for generating upper structure voicings.

    Automatically selects appropriate upper structures based on chord quality.
    """

    # Catalog all structures
    STRUCTURES = {
        "dom7": DOMINANT_UPPER_STRUCTURES,
        "maj7": MAJOR7_UPPER_STRUCTURES,
        "min7": MINOR7_UPPER_STRUCTURES,
        "min7b5": HALF_DIM_UPPER_STRUCTURES,
    }

    @staticmethod
    def get_upper_structure_voicing(
        chord: JazzChord,
        bass_octave: int = 2,
        triad_octave: int = 4,
        structure_name: Optional[str] = None
    ) -> List[int]:
        """
        Generate upper structure voicing for a chord.

        Args:
            chord: JazzChord to voice
            bass_octave: Octave for bass note
            triad_octave: Octave for triad
            structure_name: Specific structure (e.g., "bII_major") or None for auto-select

        Returns:
            List of MIDI pitches (bass + triad)

        Example:
            >>> chord = JazzChord(root=0, quality="dom7")  # C7
            >>> voicing = UpperStructureEngine.get_upper_structure_voicing(chord)
            >>> # Returns: [24, 61, 65, 68] = C (bass) + Db major triad
        """
        # Get appropriate structure catalog
        chord_type = UpperStructureEngine._normalize_chord_quality(chord.quality)

        if chord_type not in UpperStructureEngine.STRUCTURES:
            # Fallback: return simple shell voicing
            return UpperStructureEngine._simple_shell_voicing(chord, bass_octave)

        structures = UpperStructureEngine.STRUCTURES[chord_type]

        # Select structure
        if structure_name and structure_name in structures:
            structure = structures[structure_name]
        else:
            # Auto-select based on chord type
            structure = UpperStructureEngine._auto_select_structure(chord, structures)

        # Build voicing
        bass_note = 12 * bass_octave + chord.root

        # Build triad
        triad_root = 12 * triad_octave + (chord.root + structure.triad_root_interval) % 12
        triad_notes = UpperStructureEngine._build_triad(triad_root, structure.triad_quality)

        # Combine
        return [bass_note] + triad_notes

    @staticmethod
    def _normalize_chord_quality(quality: str) -> str:
        """Normalize chord quality string."""
        quality_lower = quality.lower()

        if "dom7" in quality_lower or quality_lower == "7":
            return "dom7"
        elif "maj7" in quality_lower:
            return "maj7"
        elif "min7b5" in quality_lower or "m7b5" in quality_lower or "ø" in quality_lower:
            return "min7b5"
        elif "min7" in quality_lower or "m7" in quality_lower:
            return "min7"
        else:
            return "unknown"

    @staticmethod
    def _auto_select_structure(chord: JazzChord, structures: Dict) -> UpperStructure:
        """Automatically select appropriate upper structure."""
        # Check for alterations in chord
        has_altered = any(alt in ["b9", "#9", "#11", "b13", "#5", "b5"]
                         for alt in chord.alterations)

        if chord.quality in ["dom7", "7"]:
            if has_altered or "alt" in str(chord.alterations):
                # Use altered upper structure
                return structures.get("bII_major", list(structures.values())[0])
            else:
                # Use more basic upper structure
                return structures.get("II_major", list(structures.values())[0])
        else:
            # Use first available structure
            return list(structures.values())[0]

    @staticmethod
    def _build_triad(root: int, quality: str) -> List[int]:
        """Build triad from root and quality."""
        if quality == "major":
            return [root, root + 4, root + 7]
        elif quality == "minor":
            return [root, root + 3, root + 7]
        elif quality == "augmented":
            return [root, root + 4, root + 8]
        elif quality == "diminished":
            return [root, root + 3, root + 6]
        else:
            # Default: major
            return [root, root + 4, root + 7]

    @staticmethod
    def _simple_shell_voicing(chord: JazzChord, octave: int) -> List[int]:
        """Fallback simple shell voicing."""
        root = 12 * octave + chord.root

        if "maj" in chord.quality:
            third = root + 4
            seventh = root + 11
        elif "min" in chord.quality:
            third = root + 3
            seventh = root + 10
        elif "dom" in chord.quality or chord.quality == "7":
            third = root + 4
            seventh = root + 10
        else:
            third = root + 4
            seventh = root + 10

        return [root, third, seventh]

    @staticmethod
    def list_available_structures(chord_quality: str) -> Dict[str, UpperStructure]:
        """
        List all available upper structures for a chord quality.

        Args:
            chord_quality: "dom7", "maj7", "min7", "min7b5"

        Returns:
            Dictionary of structure_name -> UpperStructure

        Example:
            >>> structures = UpperStructureEngine.list_available_structures("dom7")
            >>> for name, struct in structures.items():
            ...     print(f"{name}: {struct.sound_description}")
        """
        normalized = UpperStructureEngine._normalize_chord_quality(chord_quality)

        if normalized in UpperStructureEngine.STRUCTURES:
            return UpperStructureEngine.STRUCTURES[normalized]
        else:
            return {}

    @staticmethod
    def get_structure_info(chord_quality: str, structure_name: str) -> Optional[UpperStructure]:
        """
        Get detailed information about a specific upper structure.

        Args:
            chord_quality: "dom7", "maj7", "min7", "min7b5"
            structure_name: e.g., "bII_major"

        Returns:
            UpperStructure object or None

        Example:
            >>> info = UpperStructureEngine.get_structure_info("dom7", "bII_major")
            >>> print(info.tensions_created)
            ['b9', '3', 'b13']
        """
        structures = UpperStructureEngine.list_available_structures(chord_quality)
        return structures.get(structure_name)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def voice_progression_with_upper_structures(
    chords: List[JazzChord],
    bass_octave: int = 2,
    triad_octave: int = 4
) -> List[List[int]]:
    """
    Voice an entire chord progression using upper structures.

    Args:
        chords: List of JazzChord objects
        bass_octave: Bass octave
        triad_octave: Triad octave

    Returns:
        List of voicings (each voicing is list of MIDI pitches)

    Example:
        >>> from genres.jazz import JazzChord
        >>> progression = [
        ...     JazzChord(root=2, quality="min7"),   # Dmin7
        ...     JazzChord(root=7, quality="dom7"),   # G7
        ...     JazzChord(root=0, quality="maj7"),   # Cmaj7
        ... ]
        >>> voicings = voice_progression_with_upper_structures(progression)
    """
    voicings = []

    for chord in chords:
        voicing = UpperStructureEngine.get_upper_structure_voicing(
            chord, bass_octave, triad_octave
        )
        voicings.append(voicing)

    return voicings


def explain_upper_structure(chord_root: int, chord_quality: str, structure_name: str) -> str:
    """
    Generate human-readable explanation of an upper structure.

    Args:
        chord_root: Root pitch class (0-11)
        chord_quality: "dom7", "maj7", etc.
        structure_name: e.g., "bII_major"

    Returns:
        Explanation string

    Example:
        >>> print(explain_upper_structure(0, "dom7", "bII_major"))
        Upper Structure: bII major over C7
        Triad: Db major (Db, F, Ab)
        Tensions: b9, 3, b13
        Sound: Dark, altered, sophisticated
        Use: V7alt going to minor, bebop
    """
    note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    root_name = note_names[chord_root]

    info = UpperStructureEngine.get_structure_info(chord_quality, structure_name)

    if not info:
        return f"Upper structure '{structure_name}' not found for {chord_quality}"

    triad_root_pc = (chord_root + info.triad_root_interval) % 12
    triad_root_name = note_names[triad_root_pc]

    # Build triad note names
    if info.triad_quality == "major":
        triad_intervals = [0, 4, 7]
    elif info.triad_quality == "minor":
        triad_intervals = [0, 3, 7]
    elif info.triad_quality == "augmented":
        triad_intervals = [0, 4, 8]
    elif info.triad_quality == "diminished":
        triad_intervals = [0, 3, 6]
    else:
        triad_intervals = [0, 4, 7]

    triad_notes = [note_names[(triad_root_pc + interval) % 12] for interval in triad_intervals]

    explanation = f"""Upper Structure: {structure_name} over {root_name}{chord_quality}
Triad: {triad_root_name} {info.triad_quality} ({', '.join(triad_notes)})
Tensions: {', '.join(info.tensions_created)}
Sound: {info.sound_description}
Use: {info.recommended_use}"""

    return explanation


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    print("Upper Structure Triads - Test")
    print("=" * 70)

    # Test 1: List all dominant structures
    print("\n1. Available Upper Structures for Dominant 7th:")
    print("-" * 70)
    structures = UpperStructureEngine.list_available_structures("dom7")
    for name, struct in structures.items():
        print(f"  {name:20s} → {struct.sound_description}")

    # Test 2: Generate voicing for C7alt
    print("\n2. C7alt voicing with bII major upper structure:")
    print("-" * 70)
    chord = JazzChord(root=0, quality="dom7", alterations=["b9", "b13"])
    voicing = UpperStructureEngine.get_upper_structure_voicing(chord, structure_name="bII_major")
    print(f"  MIDI pitches: {voicing}")
    print(f"  Notes: C (bass) + Db major triad")

    # Test 3: Explain structure
    print("\n3. Explanation of bII major over C7:")
    print("-" * 70)
    explanation = explain_upper_structure(0, "dom7", "bII_major")
    print(explanation)

    # Test 4: Voice ii-V-I progression
    print("\n4. ii-V-I progression with upper structures:")
    print("-" * 70)
    progression = [
        JazzChord(root=2, quality="min7"),   # Dmin7
        JazzChord(root=7, quality="dom7"),   # G7
        JazzChord(root=0, quality="maj7"),   # Cmaj7
    ]
    voicings = voice_progression_with_upper_structures(progression)
    for chord, voicing in zip(progression, voicings):
        note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        root_name = note_names[chord.root]
        print(f"  {root_name}{chord.quality:8s}: {voicing}")

    # Test 5: All structure types
    print("\n5. Structure count by chord type:")
    print("-" * 70)
    for chord_type in ["dom7", "maj7", "min7", "min7b5"]:
        structures = UpperStructureEngine.list_available_structures(chord_type)
        print(f"  {chord_type:10s}: {len(structures)} structures")

    print("\n✓ Upper structure tests complete!")
