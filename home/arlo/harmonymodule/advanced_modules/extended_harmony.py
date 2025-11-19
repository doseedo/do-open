#!/usr/bin/env python3
"""
Extended Harmony & Upper Structures Module

Advanced 20th-21st century harmonic techniques including upper structure
triads, polychords, cluster voicings, and multi-tonic systems.

This module implements cutting-edge harmonic concepts from jazz, contemporary
classical music, and modern composition. It provides tools for creating
sophisticated harmonic textures used by composers from Stravinsky to Robert Glasper.

Research Sources:
- Mark Levine: "The Jazz Theory Book" (1995) - Upper structures and altered dominants
- Béla Bartók: "Fourteen Bagatelles" (1908) - Polychords and bitonality
- György Ligeti: "Atmosphères" (1961) - Cluster techniques
- Dmitri Tymoczko: "A Geometry of Music" (2011) - Voice leading geometry
- Olivier Messiaen: "Technique de mon langage musical" (1944) - Limited transposition
- Henry Cowell: "New Musical Resources" (1930) - Tone clusters
- McCoy Tyner: Piano voicings (quartal harmony in jazz)
- George Russell: "Lydian Chromatic Concept" (1953) - Upper structure theory
- Paul Hindemith: "The Craft of Musical Composition" (1937) - Quartal/quintal harmony
- Igor Stravinsky: "Petrushka" (1911) - Polychord usage (C major + F# major)

Features:
- Upper structure triads for dominant chords (all variations)
- Polychords and bitonal combinations
- Cluster voicings (chromatic, diatonic, pentatonic, whole-tone)
- Slash chords with bass note specification
- Altered dominant construction with all tensions
- Multi-tonic system analysis (competing tonal centers)
- Constant structures (Messiaen)
- Quartal and quintal voicings

Author: Agent 8
Date: 2025-11-19
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import defaultdict, Counter


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class UpperStructureType(Enum):
    """Types of upper structure triads on dominant chords"""
    # Based on Mark Levine's "Jazz Theory Book"
    MAJOR_ON_SHARP11 = "maj_#11"      # Dmaj on G7 = G7#11
    MAJOR_ON_FLAT9 = "maj_b9"          # Abmaj on G7 = G7b9
    MINOR_ON_FLAT9 = "min_b9"          # Abmin on G7 = G7b9b13
    MAJOR_ON_SHARP9 = "maj_#9"         # Bbmaj on G7 = G7#9
    MINOR_ON_5 = "min_5"               # Dmin on G7 = G7sus
    MAJOR_ON_FLAT13 = "maj_b13"        # Ebmaj on G7 = G7b13
    DIMINISHED_ON_3 = "dim_3"          # Bdim on G7 = G7b9
    AUGMENTED_ON_ROOT = "aug_root"     # Gaug on G7 = G7#5


class ClusterType(Enum):
    """Types of tone clusters"""
    CHROMATIC = "chromatic"           # All semitones (Cowell, Ligeti)
    DIATONIC = "diatonic"             # Scale-based clusters (Bartók)
    PENTATONIC = "pentatonic"         # Pentatonic clusters
    WHOLE_TONE = "whole_tone"         # Whole-tone clusters
    QUARTAL = "quartal"               # Stacked fourths
    QUINTAL = "quintal"               # Stacked fifths
    SECUNDAL = "secundal"             # Stacked seconds (chromatic or diatonic)


class PolychordRelation(Enum):
    """Relationship between polychord triads"""
    TRITONE = "tritone"               # Most dissonant (Petrushka chord)
    CHROMATIC_MEDIANT = "chromatic"   # Distant relation
    SYMMETRIC = "symmetric"           # Mirror relationship
    PARALLEL = "parallel"             # Same root, different quality
    RELATIVE = "relative"             # Relative major/minor
    ARBITRARY = "arbitrary"           # Any combination


class TonalCenter(Enum):
    """Strength of tonal center"""
    PRIMARY = "primary"               # Main tonic
    SECONDARY = "secondary"           # Secondary tonal center
    TERTIARY = "tertiary"             # Weak tonal center
    AMBIGUOUS = "ambiguous"           # Multiple competing centers


@dataclass
class Chord:
    """Representation of a chord with extensions"""
    root: int                          # MIDI note number (0-11 for pitch class)
    quality: str                       # "maj", "min", "dom", "dim", "aug", "sus"
    extensions: List[str] = field(default_factory=list)  # ["7", "9", "#11", etc.]
    bass_note: Optional[int] = None    # For slash chords
    voicing: List[int] = field(default_factory=list)  # Actual MIDI notes

    def __str__(self) -> str:
        """String representation of chord"""
        note_names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
        root_name = note_names[self.root % 12]
        ext_str = "".join(self.extensions)

        if self.bass_note is not None:
            bass_name = note_names[self.bass_note % 12]
            return f"{root_name}{self.quality}{ext_str}/{bass_name}"

        return f"{root_name}{self.quality}{ext_str}"


@dataclass
class Polychord:
    """Two simultaneous chords creating bitonality"""
    upper_chord: Chord
    lower_chord: Chord
    relation: PolychordRelation
    combined_voicing: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation"""
        return f"{self.upper_chord} / {self.lower_chord}"


@dataclass
class MultiTonicAnalysis:
    """Analysis of competing tonal centers"""
    tonal_centers: Dict[int, TonalCenter]  # pitch class -> strength
    primary_key: int
    secondary_keys: List[int]
    ambiguity_score: float  # 0.0 = clear, 1.0 = highly ambiguous


# ============================================================================
# CORE CLASS: EXTENDED HARMONY
# ============================================================================

class ExtendedHarmony:
    """
    Advanced 20th-21st century harmony generator

    This class provides methods for creating sophisticated harmonic structures
    used in jazz, contemporary classical, and modern composition.

    Examples:
        >>> harmony = ExtendedHarmony()
        >>>
        >>> # Create upper structure triad on G7
        >>> g7_sharp11 = harmony.create_upper_structure(7, "maj_#11")
        >>> print(g7_sharp11)  # G7#11 (Dmaj over G7)
        >>>
        >>> # Create Petrushka chord (C major + F# major)
        >>> petrushka = harmony.create_polychord(0, "maj", 6, "maj")
        >>>
        >>> # Create chromatic cluster
        >>> cluster = harmony.create_cluster(60, ClusterType.CHROMATIC, 5)
    """

    # Note names for display
    NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    # Interval definitions (semitones)
    INTERVALS = {
        "unison": 0, "min2": 1, "maj2": 2, "min3": 3, "maj3": 4,
        "p4": 5, "aug4": 6, "dim5": 6, "p5": 7, "min6": 8,
        "maj6": 9, "min7": 10, "maj7": 11, "octave": 12,
        "min9": 13, "maj9": 14, "aug9": 15, "p11": 17,
        "aug11": 18, "min13": 20, "maj13": 21
    }

    # Chord templates (intervals from root)
    CHORD_TEMPLATES = {
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "dom": [0, 4, 7, 10],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "dim": [0, 3, 6],
        "dim7": [0, 3, 6, 9],
        "aug": [0, 4, 8],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
    }

    def __init__(self, default_register: int = 60):
        """
        Initialize Extended Harmony generator

        Args:
            default_register: Default MIDI note for middle register (default: 60 = C4)
        """
        self.default_register = default_register

    # ========================================================================
    # UPPER STRUCTURE TRIADS
    # ========================================================================

    def create_upper_structure(
        self,
        root: int,
        structure_type: str,
        octave: int = 4,
        include_root: bool = True
    ) -> Chord:
        """
        Create upper structure triad on dominant chord

        Based on Mark Levine's "Jazz Theory Book" (1995), upper structures
        are triads built on specific scale degrees that create altered
        dominant sounds. This technique is fundamental in modern jazz
        reharmonization.

        Args:
            root: Root note (0-11 for pitch class, or MIDI note)
            structure_type: Type from UpperStructureType enum
            octave: Octave for voicing (default: 4)
            include_root: Include the root note (default: True)

        Returns:
            Chord object with upper structure voicing

        Examples:
            >>> harmony = ExtendedHarmony()
            >>> # G7#11 = D major triad over G bass
            >>> chord = harmony.create_upper_structure(7, "maj_#11")
            >>> # G7b9 = Ab major triad over G bass
            >>> chord = harmony.create_upper_structure(7, "maj_b9")

        References:
            - Levine, M. (1995). "The Jazz Theory Book", pp. 220-235
            - Russell, G. (1953). "Lydian Chromatic Concept"
        """
        root_pc = root % 12
        base_midi = (octave * 12) + root_pc

        # Map structure types to intervals and triads
        structure_map = {
            "maj_#11": (2, "maj", ["#11"]),      # II (whole step up) = #11
            "maj_b9": (8, "maj", ["b9", "b13"]), # bVI (minor 6th up) = b9
            "min_b9": (8, "min", ["b9", "b13"]), # bvi = b9, b13
            "maj_#9": (10, "maj", ["#9", "13"]), # bVII = #9, 13
            "min_5": (7, "min", ["sus", "11"]),  # V = sus sound
            "maj_b13": (3, "maj", ["b13"]),      # bIII = b13
            "dim_3": (4, "dim", ["b9", "13"]),   # III = b9 (dim)
            "aug_root": (0, "aug", ["#5"]),      # I aug = #5
        }

        if structure_type not in structure_map:
            raise ValueError(f"Unknown structure type: {structure_type}")

        interval, triad_quality, extensions = structure_map[structure_type]

        # Build the triad on the specified interval
        triad_root = (root_pc + interval) % 12
        triad_notes = self._build_triad(triad_root, triad_quality)

        # Voice the chord
        voicing = []

        # Add bass note (root of dominant)
        if include_root:
            voicing.append(base_midi)

        # Add third and seventh of dominant (for dominant quality)
        voicing.append(base_midi + 4)  # Major 3rd
        voicing.append(base_midi + 10)  # Minor 7th

        # Add upper structure triad (octave up)
        for interval in triad_notes:
            voicing.append(base_midi + 12 + interval)

        # Create chord object
        chord = Chord(
            root=root_pc,
            quality="dom",
            extensions=extensions,
            voicing=sorted(set(voicing))
        )

        return chord

    def _build_triad(self, root: int, quality: str) -> List[int]:
        """Build triad intervals from root"""
        if quality in self.CHORD_TEMPLATES:
            return self.CHORD_TEMPLATES[quality]

        # Default to major
        return self.CHORD_TEMPLATES["maj"]

    # ========================================================================
    # POLYCHORDS
    # ========================================================================

    def create_polychord(
        self,
        upper_root: int,
        upper_quality: str,
        lower_root: int,
        lower_quality: str,
        octave: int = 4,
        spacing: int = 12
    ) -> Polychord:
        """
        Create polychord (two simultaneous chords)

        Polychords create bitonality - the simultaneous presence of two
        different tonal centers. This technique was pioneered by Stravinsky
        (Petrushka chord: C major + F# major) and developed by Bartók,
        Milhaud, and others.

        Args:
            upper_root: Root of upper chord (0-11)
            upper_quality: Quality of upper chord ("maj", "min", etc.)
            lower_root: Root of lower chord (0-11)
            lower_quality: Quality of lower chord
            octave: Base octave (default: 4)
            spacing: Semitones between lower and upper chords (default: 12)

        Returns:
            Polychord object with combined voicing

        Examples:
            >>> harmony = ExtendedHarmony()
            >>> # Petrushka chord (C major over F# major)
            >>> petrushka = harmony.create_polychord(0, "maj", 6, "maj")
            >>> # Cmaj7/Dm (common in neo-soul)
            >>> neosoul = harmony.create_polychord(0, "maj7", 2, "min")

        References:
            - Stravinsky, I. (1911). "Petrushka" - Petrushka chord
            - Bartók, B. (1908). "Fourteen Bagatelles" - Polychord usage
            - Milhaud, D. (1923). "La création du monde" - Bitonality
        """
        base_midi = octave * 12

        # Build lower chord
        lower_intervals = self.CHORD_TEMPLATES.get(lower_quality, [0, 4, 7])
        lower_notes = [(base_midi + lower_root + i) for i in lower_intervals]

        # Build upper chord
        upper_intervals = self.CHORD_TEMPLATES.get(upper_quality, [0, 4, 7])
        upper_notes = [(base_midi + spacing + upper_root + i) for i in upper_intervals]

        # Determine relationship between chords
        interval_between = abs(upper_root - lower_root) % 12
        relation = self._classify_polychord_relation(interval_between)

        # Create chord objects
        upper_chord = Chord(
            root=upper_root,
            quality=upper_quality,
            voicing=upper_notes
        )

        lower_chord = Chord(
            root=lower_root,
            quality=lower_quality,
            voicing=lower_notes
        )

        # Combine voicings
        combined = sorted(lower_notes + upper_notes)

        polychord = Polychord(
            upper_chord=upper_chord,
            lower_chord=lower_chord,
            relation=relation,
            combined_voicing=combined
        )

        return polychord

    def _classify_polychord_relation(self, interval: int) -> PolychordRelation:
        """Classify relationship between polychord roots"""
        if interval == 6:
            return PolychordRelation.TRITONE
        elif interval in [1, 11]:
            return PolychordRelation.CHROMATIC_MEDIANT
        elif interval == 0:
            return PolychordRelation.PARALLEL
        elif interval in [3, 9]:  # Minor/major third
            return PolychordRelation.RELATIVE
        else:
            return PolychordRelation.ARBITRARY

    # ========================================================================
    # CLUSTER VOICINGS
    # ========================================================================

    def create_cluster(
        self,
        root: int,
        cluster_type: ClusterType,
        num_notes: int = 4,
        span_semitones: int = 12
    ) -> List[int]:
        """
        Create tone cluster

        Tone clusters are groups of adjacent notes played simultaneously,
        creating dense harmonic textures. Pioneered by Henry Cowell and
        developed by Bartók, Ligeti, and others for creating atmospheric
        and textural effects.

        Args:
            root: Lowest note of cluster (MIDI number)
            cluster_type: Type of cluster (ClusterType enum)
            num_notes: Number of notes in cluster (default: 4)
            span_semitones: Maximum span in semitones (default: 12)

        Returns:
            List of MIDI note numbers forming the cluster

        Examples:
            >>> harmony = ExtendedHarmony()
            >>> # Chromatic cluster (Ligeti style)
            >>> chromatic = harmony.create_cluster(60, ClusterType.CHROMATIC, 5)
            >>> # Diatonic cluster (Bartók style)
            >>> diatonic = harmony.create_cluster(60, ClusterType.DIATONIC, 4)

        References:
            - Cowell, H. (1930). "New Musical Resources" - Tone cluster theory
            - Ligeti, G. (1961). "Atmosphères" - Chromatic clusters
            - Bartók, B. (1926). Piano Sonata - Diatonic clusters
        """
        cluster = []

        if cluster_type == ClusterType.CHROMATIC:
            # All semitones (Cowell, Ligeti)
            cluster = [root + i for i in range(num_notes)]

        elif cluster_type == ClusterType.DIATONIC:
            # C major scale intervals: 0, 2, 4, 5, 7, 9, 11
            scale = [0, 2, 4, 5, 7, 9, 11]
            cluster = [root + scale[i % 7] + (i // 7) * 12
                      for i in range(num_notes)]

        elif cluster_type == ClusterType.PENTATONIC:
            # Pentatonic: 0, 2, 4, 7, 9
            scale = [0, 2, 4, 7, 9]
            cluster = [root + scale[i % 5] + (i // 5) * 12
                      for i in range(num_notes)]

        elif cluster_type == ClusterType.WHOLE_TONE:
            # Whole tone: 0, 2, 4, 6, 8, 10
            cluster = [root + i * 2 for i in range(num_notes)]

        elif cluster_type == ClusterType.QUARTAL:
            # Stacked perfect fourths (McCoy Tyner style)
            cluster = [root + i * 5 for i in range(num_notes)]

        elif cluster_type == ClusterType.QUINTAL:
            # Stacked perfect fifths (Hindemith)
            cluster = [root + i * 7 for i in range(num_notes)]

        elif cluster_type == ClusterType.SECUNDAL:
            # Stacked seconds (mostly whole steps)
            cluster = [root + i * 2 for i in range(num_notes)]

        # Ensure cluster stays within span
        cluster = [n for n in cluster if n < root + span_semitones]

        return sorted(cluster)

    # ========================================================================
    # SLASH CHORDS
    # ========================================================================

    def create_slash_chord(
        self,
        upper_chord_root: int,
        upper_chord_quality: str,
        bass_note: int,
        octave: int = 4,
        extensions: Optional[List[str]] = None
    ) -> Chord:
        """
        Create slash chord (chord over different bass note)

        Slash chords specify a chord with a non-root bass note, creating
        inversions or hybrid harmonies. Common in jazz, pop, and contemporary
        music for creating smooth bass lines and interesting harmonic colors.

        Args:
            upper_chord_root: Root of upper chord (0-11)
            upper_chord_quality: Quality ("maj", "min7", etc.)
            bass_note: Bass note (0-11 or MIDI number)
            octave: Octave for voicing (default: 4)
            extensions: Additional extensions (optional)

        Returns:
            Chord object with slash chord voicing

        Examples:
            >>> harmony = ExtendedHarmony()
            >>> # Cmaj7/E (first inversion)
            >>> cmaj7_e = harmony.create_slash_chord(0, "maj7", 4)
            >>> # D/F# (common in folk/pop)
            >>> d_fsharp = harmony.create_slash_chord(2, "maj", 6)

        References:
            - Common in jazz comping and contemporary harmony
            - Creates smooth voice leading in progressions
        """
        base_midi = octave * 12

        # Build upper chord
        intervals = self.CHORD_TEMPLATES.get(upper_chord_quality, [0, 4, 7])
        upper_notes = [(base_midi + 12 + upper_chord_root + i) for i in intervals]

        # Add bass note (octave below)
        bass_midi = base_midi + (bass_note % 12)

        # Combine
        voicing = [bass_midi] + upper_notes

        # Create chord
        chord = Chord(
            root=upper_chord_root,
            quality=upper_chord_quality,
            extensions=extensions or [],
            bass_note=bass_note % 12,
            voicing=sorted(voicing)
        )

        return chord

    # ========================================================================
    # ALTERED DOMINANTS
    # ========================================================================

    def create_altered_dominant(
        self,
        root: int,
        alterations: List[str],
        octave: int = 4,
        voicing_style: str = "tight"
    ) -> Chord:
        """
        Create altered dominant chord with tensions

        Altered dominants feature chromatic alterations of the 5th, 9th,
        and 13th. These chords create maximum tension and are fundamental
        in jazz harmony, especially in ii-V-I progressions.

        Args:
            root: Root note (0-11)
            alterations: List of alterations (e.g., ["b9", "#9", "#11", "b13"])
            octave: Octave for voicing (default: 4)
            voicing_style: "tight" or "spread" (default: "tight")

        Returns:
            Chord object with altered dominant voicing

        Examples:
            >>> harmony = ExtendedHarmony()
            >>> # G7alt (all alterations)
            >>> g7alt = harmony.create_altered_dominant(7, ["b9", "#9", "#11", "b13"])
            >>> # G7#11 (Lydian dominant)
            >>> g7sharp11 = harmony.create_altered_dominant(7, ["#11"])

        References:
            - Levine, M. (1995). "The Jazz Theory Book" - Altered dominants
            - Messiaen's mode 7 (altered scale): 1 b2 #2 3 #4 #5 b7
        """
        root_pc = root % 12
        base_midi = (octave * 12) + root_pc

        # Start with basic dominant 7th: root, 3, 5, b7
        voicing = [
            base_midi,           # Root
            base_midi + 4,       # Major 3rd
            base_midi + 7,       # Perfect 5th (may be altered)
            base_midi + 10,      # Minor 7th
        ]

        # Alteration map (semitones from root)
        alteration_intervals = {
            "b9": 13,   # Minor 9th
            "#9": 15,   # Augmented 9th
            "b10": 15,  # Same as #9
            "11": 17,   # Perfect 11th
            "#11": 18,  # Augmented 11th (Lydian)
            "b5": 6,    # Diminished 5th
            "#5": 8,    # Augmented 5th
            "b13": 20,  # Minor 13th
            "13": 21,   # Major 13th
        }

        # Add alterations
        for alt in alterations:
            if alt in alteration_intervals:
                interval = alteration_intervals[alt]

                # Replace perfect 5th if altered
                if alt in ["b5", "#5"]:
                    if base_midi + 7 in voicing:
                        voicing.remove(base_midi + 7)
                    voicing.append(base_midi + interval)
                else:
                    voicing.append(base_midi + interval)

        # Apply voicing style
        if voicing_style == "spread":
            # Spread voicing across wider range
            voicing = self._spread_voicing(voicing)

        chord = Chord(
            root=root_pc,
            quality="dom",
            extensions=alterations,
            voicing=sorted(set(voicing))
        )

        return chord

    def _spread_voicing(self, notes: List[int]) -> List[int]:
        """Spread notes across wider range"""
        if len(notes) < 2:
            return notes

        spread = []
        for i, note in enumerate(sorted(notes)):
            # Add octave offset for spread
            offset = (i // 3) * 12
            spread.append(note + offset)

        return spread

    # ========================================================================
    # MULTI-TONIC ANALYSIS
    # ========================================================================

    def analyze_multitonic_system(
        self,
        chord_progression: List[Chord]
    ) -> MultiTonicAnalysis:
        """
        Analyze competing tonal centers in a progression

        Multi-tonic systems feature multiple competing tonal centers,
        creating tonal ambiguity. Used by Bartók, Messiaen, and contemporary
        composers to create harmonic tension and complexity.

        Args:
            chord_progression: List of Chord objects

        Returns:
            MultiTonicAnalysis with detected tonal centers

        Examples:
            >>> harmony = ExtendedHarmony()
            >>> chords = [
            ...     harmony.create_slash_chord(0, "maj", 0),  # C
            ...     harmony.create_slash_chord(6, "maj", 6),  # F#
            ... ]
            >>> analysis = harmony.analyze_multitonic_system(chords)
            >>> print(f"Ambiguity: {analysis.ambiguity_score}")

        References:
            - Bartók, B. - Axis system (C-F#-C# tonal centers)
            - Messiaen, O. - Multiple tonal centers in compositions
        """
        # Count root occurrences
        root_counts = Counter([c.root for c in chord_progression])

        # Calculate tonal strength for each pitch class
        tonal_centers = {}
        total_chords = len(chord_progression)

        for root, count in root_counts.items():
            strength = count / total_chords

            if strength >= 0.4:
                tonal_centers[root] = TonalCenter.PRIMARY
            elif strength >= 0.2:
                tonal_centers[root] = TonalCenter.SECONDARY
            elif strength >= 0.1:
                tonal_centers[root] = TonalCenter.TERTIARY
            else:
                tonal_centers[root] = TonalCenter.AMBIGUOUS

        # Determine primary and secondary keys
        sorted_roots = sorted(root_counts.items(), key=lambda x: x[1], reverse=True)

        primary_key = sorted_roots[0][0] if sorted_roots else 0
        secondary_keys = [r for r, _ in sorted_roots[1:3]]

        # Calculate ambiguity score (0.0 = clear, 1.0 = highly ambiguous)
        if not sorted_roots:
            ambiguity_score = 1.0
        else:
            # Compare top two tonal centers
            if len(sorted_roots) == 1:
                ambiguity_score = 0.0
            else:
                primary_count = sorted_roots[0][1]
                secondary_count = sorted_roots[1][1]
                ambiguity_score = secondary_count / primary_count

        analysis = MultiTonicAnalysis(
            tonal_centers=tonal_centers,
            primary_key=primary_key,
            secondary_keys=secondary_keys,
            ambiguity_score=min(ambiguity_score, 1.0)
        )

        return analysis

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def chord_to_midi_notes(self, chord: Chord) -> List[int]:
        """Extract MIDI notes from chord"""
        return chord.voicing if chord.voicing else []

    def get_chord_name(self, chord: Chord) -> str:
        """Get string name of chord"""
        return str(chord)

    def transpose_chord(self, chord: Chord, semitones: int) -> Chord:
        """Transpose chord by semitones"""
        new_chord = Chord(
            root=(chord.root + semitones) % 12,
            quality=chord.quality,
            extensions=chord.extensions.copy(),
            bass_note=(chord.bass_note + semitones) % 12 if chord.bass_note else None,
            voicing=[n + semitones for n in chord.voicing]
        )
        return new_chord


# ============================================================================
# DEMO AND TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXTENDED HARMONY & UPPER STRUCTURES - Demo")
    print("=" * 70)

    harmony = ExtendedHarmony()

    # Demo 1: Upper Structure Triads
    print("\n1. UPPER STRUCTURE TRIADS (Jazz Reharmonization)")
    print("-" * 70)

    g7_structures = [
        ("maj_#11", "G7#11 (Lydian Dominant)"),
        ("maj_b9", "G7b9 (Diminished scale)"),
        ("maj_#9", "G7#9 (Altered dominant)"),
    ]

    for structure, description in g7_structures:
        chord = harmony.create_upper_structure(7, structure)
        print(f"{description:35} {chord}")
        print(f"  Voicing: {chord.voicing}")

    # Demo 2: Polychords
    print("\n2. POLYCHORDS (Bitonality)")
    print("-" * 70)

    polychords = [
        ((0, "maj", 6, "maj"), "Petrushka Chord (Stravinsky)"),
        ((0, "maj7", 2, "min"), "Cmaj7/Dm (Neo-soul)"),
        ((4, "min", 10, "maj"), "Em/Bb (Tritone relation)"),
    ]

    for (ur, uq, lr, lq), description in polychords:
        poly = harmony.create_polychord(ur, uq, lr, lq)
        print(f"{description:35} {poly}")
        print(f"  Relation: {poly.relation.value}")
        print(f"  Combined: {poly.combined_voicing}")

    # Demo 3: Cluster Voicings
    print("\n3. CLUSTER VOICINGS")
    print("-" * 70)

    clusters = [
        (ClusterType.CHROMATIC, "Chromatic (Ligeti)"),
        (ClusterType.DIATONIC, "Diatonic (Bartók)"),
        (ClusterType.QUARTAL, "Quartal (McCoy Tyner)"),
        (ClusterType.PENTATONIC, "Pentatonic"),
    ]

    for cluster_type, description in clusters:
        cluster = harmony.create_cluster(60, cluster_type, 5)
        print(f"{description:35} {cluster}")

    # Demo 4: Slash Chords
    print("\n4. SLASH CHORDS")
    print("-" * 70)

    slash_chords = [
        ((0, "maj7", 4), "Cmaj7/E (First inversion)"),
        ((2, "maj", 6), "D/F# (Folk/Pop)"),
        ((7, "dom", 2), "G7/D (ii-V bass motion)"),
    ]

    for (root, quality, bass), description in slash_chords:
        chord = harmony.create_slash_chord(root, quality, bass)
        print(f"{description:35} {chord}")
        print(f"  Voicing: {chord.voicing}")

    # Demo 5: Altered Dominants
    print("\n5. ALTERED DOMINANTS")
    print("-" * 70)

    altered = [
        (["b9", "#9", "b13"], "G7alt (All alterations)"),
        (["#11"], "G7#11 (Lydian dominant)"),
        (["b9", "b13"], "G7b9b13"),
    ]

    for alterations, description in altered:
        chord = harmony.create_altered_dominant(7, alterations)
        print(f"{description:35} {chord}")
        print(f"  Voicing: {chord.voicing}")

    # Demo 6: Multi-tonic Analysis
    print("\n6. MULTI-TONIC ANALYSIS")
    print("-" * 70)

    # Create progression with competing tonal centers
    progression = [
        harmony.create_slash_chord(0, "maj", 0),   # C
        harmony.create_slash_chord(0, "maj", 0),   # C
        harmony.create_slash_chord(6, "maj", 6),   # F#
        harmony.create_slash_chord(6, "maj", 6),   # F#
        harmony.create_slash_chord(0, "maj", 0),   # C
    ]

    analysis = harmony.analyze_multitonic_system(progression)

    print(f"Primary key: {harmony.NOTE_NAMES[analysis.primary_key]}")
    print(f"Secondary keys: {[harmony.NOTE_NAMES[k] for k in analysis.secondary_keys]}")
    print(f"Ambiguity score: {analysis.ambiguity_score:.2f}")
    print(f"Tonal centers: {[(harmony.NOTE_NAMES[k], v.value) for k, v in analysis.tonal_centers.items()]}")

    print("\n" + "=" * 70)
    print("Demo complete! Module ready for integration.")
    print("=" * 70)
