#!/usr/bin/env python3
"""
Neo-Riemannian Theory Implementation
=====================================

Comprehensive implementation of Neo-Riemannian transformations for modern harmonic
analysis and composition. This module provides tools for navigating tonal space
using transformational operations, enabling smooth voice leading and chromaticism
commonly found in film music, late-Romantic repertoire, and contemporary composition.

Theory Background:
-----------------
Neo-Riemannian theory models harmonic relationships through operations on triads
rather than functional progressions. The three primary transformations (PLR) relate
triads by minimal voice leading:

- P (Parallel): C major → C minor (change mode, common root)
- L (Leading-tone): C major → E minor (change mode, common third)
- R (Relative): C major → A minor (change mode, common fifth)

These operations generate networks of triads (Tonnetz) that reveal chromatic
relationships and enable complex progressions with smooth voice leading.

Applications:
------------
- Film scoring (Williams, Zimmer, Desplat)
- Late-Romantic harmony (Wagner, Liszt, Brahms)
- Maximally smooth voice leading
- Hexatonic and octatonic cycles
- Chromatic mediant progressions

References:
----------
- Richard Cohn: "Neo-Riemannian Operations, Parsimonious Trichords" (1997)
- David Lewin: "Generalized Musical Intervals and Transformations" (1987)
- Dmitri Tymoczko: "A Geometry of Music" (2011)

Author: Agent 3 - Advanced Harmony & Modal Systems
License: MIT
"""

from typing import List, Tuple, Dict, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
import math


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class TriadQuality(Enum):
    """Triad quality enumeration"""
    MAJOR = "major"
    MINOR = "minor"
    AUGMENTED = "augmented"
    DIMINISHED = "diminished"


@dataclass(frozen=True)
class Triad:
    """
    Immutable triad representation.

    Attributes:
        root: Root note as MIDI pitch class (0-11, C=0)
        quality: Triad quality (major/minor/augmented/diminished)
    """
    root: int
    quality: TriadQuality

    def __post_init__(self):
        """Validate triad data"""
        # Ensure root is pitch class (0-11)
        object.__setattr__(self, 'root', self.root % 12)

    def get_pitches(self) -> Tuple[int, int, int]:
        """
        Get pitch classes of triad in root position.

        Returns:
            Tuple of three pitch classes (root, third, fifth)
        """
        root = self.root
        if self.quality == TriadQuality.MAJOR:
            return (root, (root + 4) % 12, (root + 7) % 12)
        elif self.quality == TriadQuality.MINOR:
            return (root, (root + 3) % 12, (root + 7) % 12)
        elif self.quality == TriadQuality.AUGMENTED:
            return (root, (root + 4) % 12, (root + 8) % 12)
        else:  # DIMINISHED
            return (root, (root + 3) % 12, (root + 6) % 12)

    def to_midi_notes(self, octave: int = 4) -> List[int]:
        """
        Convert to MIDI note numbers in close position.

        Args:
            octave: Base octave for root note

        Returns:
            List of MIDI note numbers
        """
        pitches = self.get_pitches()
        base = 12 * octave
        return [base + pc for pc in pitches]

    def __str__(self) -> str:
        """String representation"""
        note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        root_name = note_names[self.root]

        if self.quality == TriadQuality.MAJOR:
            return root_name
        elif self.quality == TriadQuality.MINOR:
            return f"{root_name}m"
        elif self.quality == TriadQuality.AUGMENTED:
            return f"{root_name}+"
        else:
            return f"{root_name}°"

    def __repr__(self) -> str:
        return f"Triad({self})"


# ============================================================================
# NEO-RIEMANNIAN TRANSFORMATIONS
# ============================================================================

class NeoRiemannianTransformations:
    """
    Core Neo-Riemannian transformations (PLR and extended operations).

    This class implements the fundamental transformations that relate triads
    through minimal voice leading. All transformations preserve two common tones.
    """

    @staticmethod
    def parallel(triad: Triad) -> Triad:
        """
        P (Parallel) transformation: Change mode, preserve root.

        C major → C minor (or vice versa)
        Voice leading: Single semitone motion in the third

        Args:
            triad: Input triad

        Returns:
            Transformed triad
        """
        if triad.quality == TriadQuality.MAJOR:
            return Triad(triad.root, TriadQuality.MINOR)
        elif triad.quality == TriadQuality.MINOR:
            return Triad(triad.root, TriadQuality.MAJOR)
        else:
            # Augmented and diminished triads don't have standard P transform
            return triad

    @staticmethod
    def leading_tone(triad: Triad) -> Triad:
        """
        L (Leading-tone) transformation: Change mode, preserve third.

        C major → E minor (or E minor → C major)
        Voice leading: Semitone motion in root and fifth

        Args:
            triad: Input triad

        Returns:
            Transformed triad
        """
        if triad.quality == TriadQuality.MAJOR:
            # Major → minor a major third above
            new_root = (triad.root + 4) % 12
            return Triad(new_root, TriadQuality.MINOR)
        elif triad.quality == TriadQuality.MINOR:
            # Minor → major a minor third below
            new_root = (triad.root - 4) % 12
            return Triad(new_root, TriadQuality.MAJOR)
        else:
            return triad

    @staticmethod
    def relative(triad: Triad) -> Triad:
        """
        R (Relative) transformation: Change mode, preserve fifth.

        C major → A minor (or A minor → C major)
        Voice leading: Whole tone motion in root and third

        Args:
            triad: Input triad

        Returns:
            Transformed triad
        """
        if triad.quality == TriadQuality.MAJOR:
            # Major → minor a major third below (relative minor)
            new_root = (triad.root - 3) % 12
            return Triad(new_root, TriadQuality.MINOR)
        elif triad.quality == TriadQuality.MINOR:
            # Minor → major a minor third above (relative major)
            new_root = (triad.root + 3) % 12
            return Triad(new_root, TriadQuality.MAJOR)
        else:
            return triad

    @staticmethod
    def nebenverwandt(triad: Triad) -> Triad:
        """
        N (Nebenverwandt) transformation: Combination of R, L, and P.

        Moves between triads a fifth apart with mode change.
        C major → F minor

        Args:
            triad: Input triad

        Returns:
            Transformed triad
        """
        if triad.quality == TriadQuality.MAJOR:
            new_root = (triad.root + 7) % 12
            return Triad(new_root, TriadQuality.MINOR)
        elif triad.quality == TriadQuality.MINOR:
            new_root = (triad.root - 7) % 12
            return Triad(new_root, TriadQuality.MAJOR)
        else:
            return triad

    @staticmethod
    def slide(triad: Triad) -> Triad:
        """
        S (Slide) transformation: Chromatic slide with mode change.

        C major → Db minor (or vice versa)
        Voice leading: Two voices move by semitone in contrary motion

        Args:
            triad: Input triad

        Returns:
            Transformed triad
        """
        if triad.quality == TriadQuality.MAJOR:
            new_root = (triad.root + 1) % 12
            return Triad(new_root, TriadQuality.MINOR)
        elif triad.quality == TriadQuality.MINOR:
            new_root = (triad.root - 1) % 12
            return Triad(new_root, TriadQuality.MAJOR)
        else:
            return triad

    @staticmethod
    def hexatonic_pole(triad: Triad) -> Triad:
        """
        H (Hexatonic pole) transformation: L then P, or P then L.

        Relates triads sharing two common tones, opposite poles of hexatonic system.
        C major → Ab minor

        Args:
            triad: Input triad

        Returns:
            Transformed triad
        """
        # H = LP = PL
        temp = NeoRiemannianTransformations.leading_tone(triad)
        return NeoRiemannianTransformations.parallel(temp)

    @staticmethod
    def dominant(triad: Triad) -> Triad:
        """
        D (Dominant) transformation: Move to dominant.

        C major → G major
        Not a Neo-Riemannian transformation but useful for composition.

        Args:
            triad: Input triad

        Returns:
            Transformed triad (preserves quality)
        """
        new_root = (triad.root + 7) % 12
        return Triad(new_root, triad.quality)

    @staticmethod
    def subdominant(triad: Triad) -> Triad:
        """
        SD (Subdominant) transformation: Move to subdominant.

        C major → F major
        Not a Neo-Riemannian transformation but useful for composition.

        Args:
            triad: Input triad

        Returns:
            Transformed triad (preserves quality)
        """
        new_root = (triad.root + 5) % 12
        return Triad(new_root, triad.quality)


# ============================================================================
# TONNETZ (TONAL NETWORK)
# ============================================================================

class Tonnetz:
    """
    Tonnetz (tonal network) representation for navigating harmonic space.

    The Tonnetz is a lattice where:
    - Horizontal axis: Perfect fifths
    - Diagonal axis 1: Major thirds
    - Diagonal axis 2: Minor thirds

    Each triangle represents a major or minor triad, with shared edges
    representing common tones.
    """

    def __init__(self):
        """Initialize Tonnetz with all major and minor triads"""
        self.major_triads = [Triad(i, TriadQuality.MAJOR) for i in range(12)]
        self.minor_triads = [Triad(i, TriadQuality.MINOR) for i in range(12)]
        self.all_triads = self.major_triads + self.minor_triads

        # Build adjacency graph
        self._build_graph()

    def _build_graph(self):
        """Build adjacency graph based on Neo-Riemannian transformations"""
        self.adjacency: Dict[Triad, Dict[str, Triad]] = {}

        for triad in self.all_triads:
            transforms = NeoRiemannianTransformations()
            self.adjacency[triad] = {
                'P': transforms.parallel(triad),
                'L': transforms.leading_tone(triad),
                'R': transforms.relative(triad),
                'N': transforms.nebenverwandt(triad),
                'S': transforms.slide(triad),
                'H': transforms.hexatonic_pole(triad),
                'D': transforms.dominant(triad),
                'SD': transforms.subdominant(triad),
            }

    def get_neighbors(self, triad: Triad, transformation: str = None) -> Union[Triad, Dict[str, Triad]]:
        """
        Get neighboring triads via transformations.

        Args:
            triad: Starting triad
            transformation: Specific transformation ('P', 'L', 'R', etc.) or None for all

        Returns:
            Single triad if transformation specified, otherwise dict of all neighbors
        """
        if transformation:
            return self.adjacency[triad][transformation]
        return self.adjacency[triad]

    def distance(self, triad1: Triad, triad2: Triad) -> int:
        """
        Calculate minimal path distance between two triads.

        Uses BFS to find shortest path in transformation space.

        Args:
            triad1: Starting triad
            triad2: Target triad

        Returns:
            Number of transformations in shortest path
        """
        if triad1 == triad2:
            return 0

        # BFS
        from collections import deque
        queue = deque([(triad1, 0)])
        visited = {triad1}

        while queue:
            current, dist = queue.popleft()

            for neighbor in self.adjacency[current].values():
                if neighbor == triad2:
                    return dist + 1

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return -1  # Should never happen for triads

    def find_path(self, triad1: Triad, triad2: Triad) -> Optional[List[Tuple[str, Triad]]]:
        """
        Find a path of transformations from triad1 to triad2.

        Args:
            triad1: Starting triad
            triad2: Target triad

        Returns:
            List of (transformation_name, resulting_triad) tuples, or None
        """
        if triad1 == triad2:
            return []

        # BFS with path tracking
        from collections import deque
        queue = deque([(triad1, [])])
        visited = {triad1}

        while queue:
            current, path = queue.popleft()

            for trans_name, neighbor in self.adjacency[current].items():
                if neighbor == triad2:
                    return path + [(trans_name, neighbor)]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(trans_name, neighbor)]))

        return None


# ============================================================================
# TRANSFORMATION CHAINS
# ============================================================================

class TransformationChain:
    """
    Generate and manipulate chains of Neo-Riemannian transformations.

    Useful for creating harmonic progressions with specific voice-leading properties.
    """

    def __init__(self, starting_triad: Triad):
        """
        Initialize transformation chain.

        Args:
            starting_triad: Initial triad
        """
        self.chain: List[Triad] = [starting_triad]
        self.transformations: List[str] = []
        self.tonnetz = Tonnetz()

    def apply(self, transformation: str) -> 'TransformationChain':
        """
        Apply a transformation and add result to chain.

        Args:
            transformation: Transformation name ('P', 'L', 'R', etc.)

        Returns:
            Self for method chaining
        """
        current = self.chain[-1]
        next_triad = self.tonnetz.get_neighbors(current, transformation)
        self.chain.append(next_triad)
        self.transformations.append(transformation)
        return self

    def apply_sequence(self, sequence: str) -> 'TransformationChain':
        """
        Apply a sequence of transformations.

        Args:
            sequence: String of transformations (e.g., "PLR" or "P L R")

        Returns:
            Self for method chaining
        """
        # Parse sequence (allow spaces or concatenated)
        trans_list = sequence.upper().replace(' ', '')

        # Handle multi-character transformations
        i = 0
        while i < len(trans_list):
            if i + 1 < len(trans_list) and trans_list[i:i+2] == 'SD':
                self.apply('SD')
                i += 2
            else:
                self.apply(trans_list[i])
                i += 1

        return self

    def get_progression(self) -> List[Triad]:
        """Get the full progression of triads"""
        return self.chain.copy()

    def get_midi_progression(self, octave: int = 4, voice_lead: bool = True) -> List[List[int]]:
        """
        Get MIDI note progression with optional voice leading.

        Args:
            octave: Base octave
            voice_lead: Apply smooth voice leading if True

        Returns:
            List of chord voicings as MIDI note lists
        """
        if not voice_lead:
            return [triad.to_midi_notes(octave) for triad in self.chain]

        # Apply voice leading
        progression = [self.chain[0].to_midi_notes(octave)]

        for i in range(1, len(self.chain)):
            prev_voicing = progression[-1]
            next_pitches = self.chain[i].get_pitches()

            # Find minimal motion voicing
            next_voicing = self._find_closest_voicing(prev_voicing, next_pitches, octave)
            progression.append(next_voicing)

        return progression

    def _find_closest_voicing(self, prev_voicing: List[int],
                               target_pitches: Tuple[int, int, int],
                               base_octave: int) -> List[int]:
        """
        Find closest voicing of target pitches to previous voicing.

        Uses minimal total voice motion heuristic.
        """
        min_distance = float('inf')
        best_voicing = None

        # Try different octave combinations
        for oct_offset in range(-1, 2):
            octave = base_octave + oct_offset
            base = 12 * octave

            # Try all inversions
            for root_octave in range(-1, 2):
                for third_octave in range(-1, 2):
                    for fifth_octave in range(-1, 2):
                        voicing = [
                            base + target_pitches[0] + 12 * root_octave,
                            base + target_pitches[1] + 12 * third_octave,
                            base + target_pitches[2] + 12 * fifth_octave
                        ]

                        # Calculate total motion
                        distance = sum(abs(voicing[i] - prev_voicing[i]) for i in range(3))

                        if distance < min_distance:
                            min_distance = distance
                            best_voicing = voicing

        return best_voicing

    def __repr__(self) -> str:
        """String representation"""
        triad_str = " → ".join(str(t) for t in self.chain)
        trans_str = ", ".join(self.transformations) if self.transformations else "none"
        return f"TransformationChain({triad_str}) via [{trans_str}]"


# ============================================================================
# HEXATONIC AND OCTATONIC SYSTEMS
# ============================================================================

class HexatonicSystem:
    """
    Hexatonic (6-note) systems generated by L and P transformations.

    Each hexatonic system contains 4 triads related by L and P, sharing
    6 distinct pitch classes. Common in film music and late-Romantic repertoire.
    """

    # Four hexatonic systems (poles)
    NORTHERN = 0  # C, E, Ab triads (major and minor)
    SOUTHERN = 1  # Db, F, A triads
    EASTERN = 2   # D, Gb, Bb triads
    WESTERN = 3   # Eb, G, B triads

    def __init__(self, pole: int = NORTHERN):
        """
        Initialize hexatonic system.

        Args:
            pole: System identifier (NORTHERN, SOUTHERN, EASTERN, WESTERN)
        """
        self.pole = pole
        self.triads = self._generate_triads()
        self.pitch_classes = self._get_pitch_classes()

    def _generate_triads(self) -> List[Triad]:
        """Generate the 4 triads in this hexatonic system"""
        # Starting roots for each pole (major triads)
        roots = [
            [0, 4, 8],   # NORTHERN: C, E, Ab
            [1, 5, 9],   # SOUTHERN: Db, F, A
            [2, 6, 10],  # EASTERN: D, Gb, Bb
            [3, 7, 11],  # WESTERN: Eb, G, B
        ]

        pole_roots = roots[self.pole]
        triads = []

        for root in pole_roots:
            triads.append(Triad(root, TriadQuality.MAJOR))
            triads.append(Triad(root, TriadQuality.MINOR))

        return triads

    def _get_pitch_classes(self) -> Set[int]:
        """Get all pitch classes in the system"""
        pcs = set()
        for triad in self.triads:
            pcs.update(triad.get_pitches())
        return pcs

    def get_cycle(self, start_triad: Triad = None) -> List[Triad]:
        """
        Get a cycle through all triads using L and P transformations.

        Args:
            start_triad: Starting triad (uses first in system if None)

        Returns:
            Ordered list of triads forming a cycle
        """
        if start_triad is None:
            start_triad = self.triads[0]

        # Cycle pattern: P, L, P, L, P, L (returns to start)
        chain = TransformationChain(start_triad)
        chain.apply_sequence("PLPLPL")

        return chain.get_progression()[:-1]  # Remove duplicate start

    def __repr__(self) -> str:
        pole_names = ["Northern", "Southern", "Eastern", "Western"]
        return f"HexatonicSystem({pole_names[self.pole]}): {self.triads}"


class ChromaticMediant:
    """
    Chromatic mediant relationships and progressions.

    Chromatic mediants are triads whose roots are a third apart (major or minor)
    and which share one common tone. Common in Romantic and film music.
    """

    @staticmethod
    def get_chromatic_mediants(triad: Triad) -> Dict[str, Triad]:
        """
        Get all chromatic mediant relationships for a triad.

        Args:
            triad: Reference triad

        Returns:
            Dictionary of mediant relationships
        """
        root = triad.root

        mediants = {}

        if triad.quality == TriadQuality.MAJOR:
            # Upper chromatic mediant (major third up, opposite quality)
            mediants['UCM'] = Triad((root + 4) % 12, TriadQuality.MINOR)
            # Lower chromatic mediant (major third down, opposite quality)
            mediants['LCM'] = Triad((root - 4) % 12, TriadQuality.MINOR)
            # Upper flat mediant (minor third up, same quality)
            mediants['UFM'] = Triad((root + 3) % 12, TriadQuality.MAJOR)
            # Lower flat mediant (minor third down, same quality)
            mediants['LFM'] = Triad((root - 3) % 12, TriadQuality.MAJOR)

        else:  # MINOR
            mediants['UCM'] = Triad((root + 4) % 12, TriadQuality.MAJOR)
            mediants['LCM'] = Triad((root - 4) % 12, TriadQuality.MAJOR)
            mediants['UFM'] = Triad((root + 3) % 12, TriadQuality.MINOR)
            mediants['LFM'] = Triad((root - 3) % 12, TriadQuality.MINOR)

        return mediants

    @staticmethod
    def create_mediant_progression(start: Triad, pattern: str = "UCM LCM") -> List[Triad]:
        """
        Create a progression using chromatic mediant relationships.

        Args:
            start: Starting triad
            pattern: Space-separated mediant codes (UCM, LCM, UFM, LFM)

        Returns:
            List of triads
        """
        progression = [start]
        current = start

        for mediant_type in pattern.split():
            mediants = ChromaticMediant.get_chromatic_mediants(current)
            if mediant_type in mediants:
                current = mediants[mediant_type]
                progression.append(current)

        return progression


# ============================================================================
# VOICE LEADING EFFICIENCY
# ============================================================================

class VoiceLeadingAnalyzer:
    """
    Analyze and optimize voice leading efficiency in progressions.

    Measures and optimizes the total motion between chords.
    """

    @staticmethod
    def calculate_voice_leading_distance(chord1: List[int], chord2: List[int]) -> float:
        """
        Calculate total semitone motion between two chords.

        Assumes chords are same length and pre-ordered.

        Args:
            chord1: First chord as MIDI notes
            chord2: Second chord as MIDI notes

        Returns:
            Total semitone distance
        """
        if len(chord1) != len(chord2):
            raise ValueError("Chords must have same number of voices")

        return sum(abs(chord2[i] - chord1[i]) for i in range(len(chord1)))

    @staticmethod
    def analyze_progression(progression: List[List[int]]) -> Dict[str, float]:
        """
        Analyze voice leading efficiency of a progression.

        Args:
            progression: List of chords as MIDI note lists

        Returns:
            Dictionary with analysis metrics
        """
        if len(progression) < 2:
            return {"total_motion": 0, "avg_motion": 0, "max_motion": 0}

        motions = []
        for i in range(len(progression) - 1):
            distance = VoiceLeadingAnalyzer.calculate_voice_leading_distance(
                progression[i], progression[i + 1]
            )
            motions.append(distance)

        return {
            "total_motion": sum(motions),
            "avg_motion": sum(motions) / len(motions),
            "max_motion": max(motions),
            "motion_per_step": motions
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEO-RIEMANNIAN THEORY - EXAMPLES")
    print("=" * 70)

    # Example 1: Basic transformations
    print("\n1. BASIC PLR TRANSFORMATIONS")
    print("-" * 70)
    c_major = Triad(0, TriadQuality.MAJOR)
    print(f"Starting triad: {c_major}")

    transforms = NeoRiemannianTransformations()
    print(f"P (Parallel):      {c_major} → {transforms.parallel(c_major)}")
    print(f"L (Leading-tone):  {c_major} → {transforms.leading_tone(c_major)}")
    print(f"R (Relative):      {c_major} → {transforms.relative(c_major)}")
    print(f"N (Nebenverwandt): {c_major} → {transforms.nebenverwandt(c_major)}")
    print(f"S (Slide):         {c_major} → {transforms.slide(c_major)}")
    print(f"H (Hexatonic):     {c_major} → {transforms.hexatonic_pole(c_major)}")

    # Example 2: Transformation chains
    print("\n2. TRANSFORMATION CHAINS")
    print("-" * 70)
    chain = TransformationChain(c_major)
    chain.apply_sequence("P L R")
    print(f"Chain: {chain}")
    print(f"Progression: {' → '.join(str(t) for t in chain.get_progression())}")

    # Example 3: Film music progression (common in John Williams)
    print("\n3. FILM MUSIC PROGRESSION (Chromatic Mediants)")
    print("-" * 70)
    film_chain = TransformationChain(c_major)
    film_chain.apply_sequence("L P L")  # Creates dramatic chromatic shift
    print(f"Progression: {' → '.join(str(t) for t in film_chain.get_progression())}")
    midi_prog = film_chain.get_midi_progression(voice_lead=True)
    print(f"\nMIDI notes (voice-led):")
    for i, chord in enumerate(midi_prog):
        print(f"  {film_chain.get_progression()[i]}: {chord}")

    # Example 4: Hexatonic system
    print("\n4. HEXATONIC SYSTEM")
    print("-" * 70)
    hex_sys = HexatonicSystem(HexatonicSystem.NORTHERN)
    print(f"Northern Hexatonic System:")
    print(f"  Triads: {[str(t) for t in hex_sys.triads]}")
    print(f"  Pitch classes: {sorted(hex_sys.pitch_classes)}")
    cycle = hex_sys.get_cycle()
    print(f"  Cycle: {' → '.join(str(t) for t in cycle)}")

    # Example 5: Chromatic mediants
    print("\n5. CHROMATIC MEDIANT RELATIONSHIPS")
    print("-" * 70)
    mediants = ChromaticMediant.get_chromatic_mediants(c_major)
    print(f"Chromatic mediants of {c_major}:")
    for name, triad in mediants.items():
        print(f"  {name}: {triad}")

    # Example 6: Voice leading analysis
    print("\n6. VOICE LEADING ANALYSIS")
    print("-" * 70)
    efficient_chain = TransformationChain(c_major)
    efficient_chain.apply_sequence("P L P")
    progression = efficient_chain.get_midi_progression(voice_lead=True)
    analysis = VoiceLeadingAnalyzer.analyze_progression(progression)
    print(f"Progression: {' → '.join(str(t) for t in efficient_chain.get_progression())}")
    print(f"Total motion: {analysis['total_motion']} semitones")
    print(f"Average motion per step: {analysis['avg_motion']:.2f} semitones")
    print(f"Motion per step: {analysis['motion_per_step']}")

    # Example 7: Tonnetz pathfinding
    print("\n7. TONNETZ PATHFINDING")
    print("-" * 70)
    tonnetz = Tonnetz()
    start = Triad(0, TriadQuality.MAJOR)  # C major
    end = Triad(8, TriadQuality.MINOR)    # Ab minor
    path = tonnetz.find_path(start, end)
    print(f"Finding path: {start} → {end}")
    if path:
        print(f"Path: {start}", end=" ")
        for trans, triad in path:
            print(f"→ [{trans}] → {triad}", end=" ")
        print(f"\nDistance: {len(path)} transformations")

    print("\n" + "=" * 70)
