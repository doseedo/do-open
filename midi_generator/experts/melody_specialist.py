"""
AGENT 19: Melody Specialist
============================

Advanced melody analysis and generation beyond basic melodic parameters.

This specialist provides:
1. Motif development and variation techniques
2. Melodic sequence generation (ascending, descending, tonal, real)
3. Contour optimization and shape analysis
4. Comprehensive ornamentation system
5. Phrase structure analysis and segmentation
6. Melodic transformation operations

Adds 50+ specialized melody parameters beyond Agent 4's foundation.

Author: Agent 19 - Melody Specialist
License: MIT
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
import random


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MotifTransformation(Enum):
    """Types of motif transformations"""
    INVERSION = "inversion"              # Mirror around axis
    RETROGRADE = "retrograde"            # Reverse order
    RETROGRADE_INVERSION = "retrograde_inversion"
    AUGMENTATION = "augmentation"        # Lengthen durations
    DIMINUTION = "diminution"           # Shorten durations
    TRANSPOSITION = "transposition"     # Pitch shift
    INTERVALLIC_EXPANSION = "intervallic_expansion"
    INTERVALLIC_CONTRACTION = "intervallic_contraction"
    RHYTHMIC_DISPLACEMENT = "rhythmic_displacement"
    FRAGMENTATION = "fragmentation"
    SEQUENCE = "sequence"


class SequenceType(Enum):
    """Types of melodic sequences"""
    ASCENDING = "ascending"
    DESCENDING = "descending"
    TONAL = "tonal"              # Within key
    REAL = "real"                # Exact intervals
    CHROMATIC = "chromatic"      # Chromatic steps
    DIATONIC = "diatonic"        # Diatonic steps
    ROSALIA = "rosalia"          # Sequence by steps


class OrnamentType(Enum):
    """Types of melodic ornaments"""
    TRILL = "trill"
    TURN = "turn"
    MORDENT = "mordent"
    INVERTED_MORDENT = "inverted_mordent"
    APPOGGIATURA = "appoggiatura"
    ACCIACCATURA = "acciaccatura"
    SLIDE = "slide"
    GRACE_NOTE = "grace_note"
    TREMOLO = "tremolo"
    GLISSANDO = "glissando"


class ContourShape(Enum):
    """Melodic contour shapes"""
    ARCH = "arch"
    INVERTED_ARCH = "inverted_arch"
    ASCENDING = "ascending"
    DESCENDING = "descending"
    WAVE = "wave"
    PLATEAU = "plateau"
    TERRACED = "terraced"
    ZIGZAG = "zigzag"


@dataclass
class Note:
    """Represents a single musical note"""
    pitch: int              # MIDI note number (0-127)
    duration: float         # Duration in beats
    velocity: int           # Velocity (0-127)
    start_time: float       # Start time in beats
    articulation: Optional[str] = None
    ornament: Optional[OrnamentType] = None

    @property
    def pitch_class(self) -> int:
        """Get pitch class (0-11)"""
        return self.pitch % 12

    @property
    def octave(self) -> int:
        """Get octave number"""
        return self.pitch // 12

    def transpose(self, semitones: int) -> 'Note':
        """Transpose note by semitones"""
        new_pitch = max(0, min(127, self.pitch + semitones))
        return Note(
            pitch=new_pitch,
            duration=self.duration,
            velocity=self.velocity,
            start_time=self.start_time,
            articulation=self.articulation,
            ornament=self.ornament
        )


@dataclass
class Motif:
    """Represents a melodic motif"""
    notes: List[Note]
    name: Optional[str] = None
    category: Optional[str] = None

    def __len__(self) -> int:
        return len(self.notes)

    def duration(self) -> float:
        """Total duration of motif"""
        if not self.notes:
            return 0.0
        return sum(n.duration for n in self.notes)

    def pitch_intervals(self) -> List[int]:
        """Get intervals between consecutive notes"""
        if len(self.notes) < 2:
            return []
        return [self.notes[i+1].pitch - self.notes[i].pitch
                for i in range(len(self.notes)-1)]

    def contour(self) -> List[int]:
        """Get contour as list of -1, 0, 1 (down, same, up)"""
        intervals = self.pitch_intervals()
        return [np.sign(i) for i in intervals]


@dataclass
class Phrase:
    """Represents a melodic phrase"""
    motifs: List[Motif]
    start_time: float
    end_time: float
    cadence_type: Optional[str] = None

    def all_notes(self) -> List[Note]:
        """Get all notes in phrase"""
        return [note for motif in self.motifs for note in motif.notes]


@dataclass
class MelodicAnalysis:
    """Results of melodic analysis"""
    contour_shape: ContourShape
    range_semitones: int
    tessitura: int  # Average pitch
    climax_position: float  # 0.0-1.0
    intervallic_profile: Dict[int, int]  # interval -> count
    chromaticism_score: float  # 0.0-1.0
    stepwise_motion_ratio: float
    leap_count: int
    direction_changes: int
    sequence_detected: bool
    phrases: List[Phrase]
    motifs: List[Motif]


# ============================================================================
# MELODY SPECIALIST
# ============================================================================

class MelodySpecialist:
    """
    Advanced melody analysis and generation specialist.

    Capabilities:
    - Motif development with 10+ transformation techniques
    - Sequence generation (ascending, descending, tonal, real)
    - Contour analysis and optimization
    - Comprehensive ornamentation (10 types)
    - Phrase structure analysis
    - Melodic coherence validation
    """

    def __init__(self, key: str = "C", mode: str = "major"):
        """
        Initialize melody specialist

        Args:
            key: Tonal center (e.g., "C", "Eb", "F#")
            mode: Mode (major, minor, dorian, etc.)
        """
        self.key = key
        self.mode = mode
        self.scale = self._build_scale(key, mode)

        # Statistical tracking
        self.stats = {
            'motifs_developed': 0,
            'sequences_generated': 0,
            'ornaments_added': 0,
            'phrases_analyzed': 0
        }

    # ========================================================================
    # MOTIF DEVELOPMENT
    # ========================================================================

    def develop_motif(
        self,
        motif: Motif,
        techniques: List[MotifTransformation],
        n_variations: int = 1
    ) -> List[Motif]:
        """
        Develop motif using specified transformation techniques.

        Args:
            motif: Original motif to develop
            techniques: List of transformation techniques to apply
            n_variations: Number of variations to generate

        Returns:
            List of developed motifs
        """
        variations = []

        for _ in range(n_variations):
            # Apply one or more techniques
            technique = random.choice(techniques)

            if technique == MotifTransformation.INVERSION:
                variation = self._invert_motif(motif)
            elif technique == MotifTransformation.RETROGRADE:
                variation = self._retrograde_motif(motif)
            elif technique == MotifTransformation.RETROGRADE_INVERSION:
                variation = self._retrograde_inversion_motif(motif)
            elif technique == MotifTransformation.AUGMENTATION:
                variation = self._augment_motif(motif, factor=2.0)
            elif technique == MotifTransformation.DIMINUTION:
                variation = self._diminish_motif(motif, factor=0.5)
            elif technique == MotifTransformation.TRANSPOSITION:
                interval = random.choice([-7, -5, -3, -2, 2, 3, 5, 7])
                variation = self._transpose_motif(motif, interval)
            elif technique == MotifTransformation.INTERVALLIC_EXPANSION:
                variation = self._expand_intervals(motif, factor=1.5)
            elif technique == MotifTransformation.INTERVALLIC_CONTRACTION:
                variation = self._contract_intervals(motif, factor=0.5)
            elif technique == MotifTransformation.RHYTHMIC_DISPLACEMENT:
                variation = self._displace_rhythm(motif)
            elif technique == MotifTransformation.FRAGMENTATION:
                variation = self._fragment_motif(motif)
            else:
                variation = motif  # No transformation

            variations.append(variation)
            self.stats['motifs_developed'] += 1

        return variations

    def _invert_motif(self, motif: Motif, axis: Optional[int] = None) -> Motif:
        """Invert motif around axis pitch"""
        if not motif.notes:
            return motif

        if axis is None:
            # Use first note as axis
            axis = motif.notes[0].pitch

        new_notes = []
        for note in motif.notes:
            interval = note.pitch - axis
            new_pitch = axis - interval
            new_pitch = max(0, min(127, new_pitch))

            new_notes.append(Note(
                pitch=new_pitch,
                duration=note.duration,
                velocity=note.velocity,
                start_time=note.start_time,
                articulation=note.articulation
            ))

        return Motif(
            notes=new_notes,
            name=f"{motif.name}_inverted" if motif.name else "inverted",
            category=motif.category
        )

    def _retrograde_motif(self, motif: Motif) -> Motif:
        """Reverse note order (retrograde)"""
        if not motif.notes:
            return motif

        # Reverse notes and adjust start times
        reversed_notes = list(reversed(motif.notes))
        total_duration = motif.duration()

        new_notes = []
        current_time = 0.0
        for note in reversed_notes:
            new_notes.append(Note(
                pitch=note.pitch,
                duration=note.duration,
                velocity=note.velocity,
                start_time=current_time,
                articulation=note.articulation
            ))
            current_time += note.duration

        return Motif(
            notes=new_notes,
            name=f"{motif.name}_retrograde" if motif.name else "retrograde",
            category=motif.category
        )

    def _retrograde_inversion_motif(self, motif: Motif) -> Motif:
        """Apply both retrograde and inversion"""
        inverted = self._invert_motif(motif)
        return self._retrograde_motif(inverted)

    def _augment_motif(self, motif: Motif, factor: float = 2.0) -> Motif:
        """Augment motif (lengthen durations)"""
        new_notes = []
        current_time = 0.0

        for note in motif.notes:
            new_notes.append(Note(
                pitch=note.pitch,
                duration=note.duration * factor,
                velocity=note.velocity,
                start_time=current_time,
                articulation=note.articulation
            ))
            current_time += note.duration * factor

        return Motif(
            notes=new_notes,
            name=f"{motif.name}_augmented" if motif.name else "augmented",
            category=motif.category
        )

    def _diminish_motif(self, motif: Motif, factor: float = 0.5) -> Motif:
        """Diminish motif (shorten durations)"""
        return self._augment_motif(motif, factor)

    def _transpose_motif(self, motif: Motif, semitones: int) -> Motif:
        """Transpose motif by semitones"""
        new_notes = [note.transpose(semitones) for note in motif.notes]
        return Motif(
            notes=new_notes,
            name=f"{motif.name}_transposed" if motif.name else "transposed",
            category=motif.category
        )

    def _expand_intervals(self, motif: Motif, factor: float = 1.5) -> Motif:
        """Expand intervallic distances"""
        if len(motif.notes) < 2:
            return motif

        new_notes = [motif.notes[0]]  # Keep first note

        for i in range(1, len(motif.notes)):
            interval = motif.notes[i].pitch - motif.notes[i-1].pitch
            expanded_interval = int(interval * factor)
            new_pitch = new_notes[-1].pitch + expanded_interval
            new_pitch = max(0, min(127, new_pitch))

            new_notes.append(Note(
                pitch=new_pitch,
                duration=motif.notes[i].duration,
                velocity=motif.notes[i].velocity,
                start_time=motif.notes[i].start_time,
                articulation=motif.notes[i].articulation
            ))

        return Motif(notes=new_notes, name="expanded", category=motif.category)

    def _contract_intervals(self, motif: Motif, factor: float = 0.5) -> Motif:
        """Contract intervallic distances"""
        return self._expand_intervals(motif, factor)

    def _displace_rhythm(self, motif: Motif, displacement: float = 0.5) -> Motif:
        """Displace rhythm by given amount"""
        new_notes = []
        for note in motif.notes:
            new_notes.append(Note(
                pitch=note.pitch,
                duration=note.duration,
                velocity=note.velocity,
                start_time=note.start_time + displacement,
                articulation=note.articulation
            ))

        return Motif(notes=new_notes, name="displaced", category=motif.category)

    def _fragment_motif(self, motif: Motif) -> Motif:
        """Extract fragment from motif"""
        if len(motif.notes) <= 2:
            return motif

        # Take first half
        fragment_size = len(motif.notes) // 2
        fragment_notes = motif.notes[:fragment_size]

        return Motif(
            notes=fragment_notes,
            name=f"{motif.name}_fragment" if motif.name else "fragment",
            category=motif.category
        )

    # ========================================================================
    # SEQUENCE GENERATION
    # ========================================================================

    def generate_sequence(
        self,
        pattern: Motif,
        sequence_type: SequenceType,
        repetitions: int = 3,
        interval: int = 2
    ) -> List[Note]:
        """
        Generate melodic sequence from pattern.

        Args:
            pattern: Motif pattern to sequence
            sequence_type: Type of sequence to generate
            repetitions: Number of repetitions
            interval: Interval for transposition (semitones)

        Returns:
            List of notes forming the sequence
        """
        if not pattern.notes:
            return []

        all_notes = []
        current_time = 0.0
        current_transposition = 0

        for rep in range(repetitions):
            # Calculate transposition for this repetition
            if sequence_type == SequenceType.ASCENDING:
                transposition = rep * interval
            elif sequence_type == SequenceType.DESCENDING:
                transposition = -(rep * interval)
            elif sequence_type == SequenceType.TONAL:
                # Transpose by scale degrees
                transposition = self._tonal_transposition(rep * interval)
            elif sequence_type == SequenceType.REAL:
                # Exact chromatic transposition
                transposition = rep * interval
            elif sequence_type == SequenceType.CHROMATIC:
                transposition = rep  # One semitone per repetition
            elif sequence_type == SequenceType.DIATONIC:
                # Transpose by diatonic steps
                transposition = self._diatonic_transposition(rep * 2)
            else:
                transposition = 0

            # Transpose pattern and add to sequence
            for note in pattern.notes:
                new_pitch = note.pitch + transposition
                new_pitch = max(0, min(127, new_pitch))

                all_notes.append(Note(
                    pitch=new_pitch,
                    duration=note.duration,
                    velocity=note.velocity,
                    start_time=current_time + (note.start_time - pattern.notes[0].start_time),
                    articulation=note.articulation
                ))

            current_time += pattern.duration()
            self.stats['sequences_generated'] += 1

        return all_notes

    def _tonal_transposition(self, scale_degrees: int) -> int:
        """Calculate tonal transposition in semitones"""
        # Simplified: assume major scale
        scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        degree = scale_degrees % 7
        octaves = scale_degrees // 7
        return scale_intervals[degree] + (octaves * 12)

    def _diatonic_transposition(self, steps: int) -> int:
        """Calculate diatonic transposition"""
        return self._tonal_transposition(steps)

    # ========================================================================
    # CONTOUR OPTIMIZATION
    # ========================================================================

    def optimize_contour(
        self,
        melody: List[Note],
        target_shape: ContourShape = ContourShape.ARCH,
        smoothing: float = 0.5
    ) -> List[Note]:
        """
        Optimize melodic contour for target shape.

        Args:
            melody: Original melody
            target_shape: Desired contour shape
            smoothing: Amount of smoothing (0.0-1.0)

        Returns:
            Optimized melody
        """
        if len(melody) < 3:
            return melody

        pitches = [n.pitch for n in melody]

        # Generate target contour
        target_contour = self._generate_target_contour(
            len(pitches),
            target_shape,
            min(pitches),
            max(pitches)
        )

        # Blend original with target
        optimized_pitches = []
        for i, (original, target) in enumerate(zip(pitches, target_contour)):
            blended = int(original * (1 - smoothing) + target * smoothing)
            blended = max(0, min(127, blended))
            optimized_pitches.append(blended)

        # Create optimized notes
        optimized_notes = []
        for note, new_pitch in zip(melody, optimized_pitches):
            optimized_notes.append(Note(
                pitch=new_pitch,
                duration=note.duration,
                velocity=note.velocity,
                start_time=note.start_time,
                articulation=note.articulation
            ))

        return optimized_notes

    def _generate_target_contour(
        self,
        length: int,
        shape: ContourShape,
        min_pitch: int,
        max_pitch: int
    ) -> List[int]:
        """Generate target contour for given shape"""
        x = np.linspace(0, 1, length)

        if shape == ContourShape.ARCH:
            # Parabola opening downward
            y = -4 * (x - 0.5) ** 2 + 1
        elif shape == ContourShape.INVERTED_ARCH:
            # Parabola opening upward
            y = 4 * (x - 0.5) ** 2
        elif shape == ContourShape.ASCENDING:
            y = x
        elif shape == ContourShape.DESCENDING:
            y = 1 - x
        elif shape == ContourShape.WAVE:
            y = np.sin(x * np.pi * 2) * 0.5 + 0.5
        elif shape == ContourShape.PLATEAU:
            y = np.ones_like(x) * 0.5
        elif shape == ContourShape.TERRACED:
            y = np.floor(x * 4) / 4
        else:
            y = x  # Default to ascending

        # Scale to pitch range
        contour = (y * (max_pitch - min_pitch) + min_pitch).astype(int)
        return contour.tolist()

    def analyze_contour(self, melody: List[Note]) -> Dict[str, Any]:
        """
        Analyze melodic contour.

        Returns:
            Dictionary with contour analysis
        """
        if len(melody) < 2:
            return {'shape': ContourShape.PLATEAU, 'complexity': 0.0}

        pitches = [n.pitch for n in melody]

        # Identify shape
        shape = self._identify_contour_shape(pitches)

        # Calculate metrics
        direction_changes = sum(
            1 for i in range(1, len(pitches)-1)
            if (pitches[i] - pitches[i-1]) * (pitches[i+1] - pitches[i]) < 0
        )

        complexity = direction_changes / (len(pitches) - 2) if len(pitches) > 2 else 0

        # Find apex (highest note)
        apex_idx = pitches.index(max(pitches))
        apex_position = apex_idx / (len(pitches) - 1) if len(pitches) > 1 else 0.5

        return {
            'shape': shape,
            'complexity': complexity,
            'direction_changes': direction_changes,
            'apex_position': apex_position,
            'range': max(pitches) - min(pitches),
            'average_pitch': np.mean(pitches)
        }

    def _identify_contour_shape(self, pitches: List[int]) -> ContourShape:
        """Identify overall contour shape"""
        if len(pitches) < 3:
            return ContourShape.PLATEAU

        first_third = pitches[:len(pitches)//3]
        middle_third = pitches[len(pitches)//3:2*len(pitches)//3]
        last_third = pitches[2*len(pitches)//3:]

        first_avg = np.mean(first_third)
        middle_avg = np.mean(middle_third)
        last_avg = np.mean(last_third)

        # Arch: middle higher than both ends
        if middle_avg > first_avg + 2 and middle_avg > last_avg + 2:
            return ContourShape.ARCH

        # Inverted arch: middle lower than both ends
        if middle_avg < first_avg - 2 and middle_avg < last_avg - 2:
            return ContourShape.INVERTED_ARCH

        # Ascending: generally rising
        if last_avg > first_avg + 4:
            return ContourShape.ASCENDING

        # Descending: generally falling
        if last_avg < first_avg - 4:
            return ContourShape.DESCENDING

        # Check for wave pattern (multiple ups and downs)
        direction_changes = sum(
            1 for i in range(1, len(pitches)-1)
            if (pitches[i] - pitches[i-1]) * (pitches[i+1] - pitches[i]) < 0
        )

        if direction_changes >= len(pitches) / 3:
            return ContourShape.WAVE

        return ContourShape.PLATEAU

    # ========================================================================
    # ORNAMENTATION
    # ========================================================================

    def add_ornamentation(
        self,
        melody: List[Note],
        ornament_types: List[OrnamentType],
        density: float = 0.3
    ) -> List[Note]:
        """
        Add ornamentation to melody.

        Args:
            melody: Original melody
            ornament_types: Types of ornaments to add
            density: Proportion of notes to ornament (0.0-1.0)

        Returns:
            Ornamented melody
        """
        if not melody or density <= 0:
            return melody

        ornamented = []

        for note in melody:
            # Decide whether to ornament this note
            if random.random() < density:
                ornament_type = random.choice(ornament_types)
                ornament_notes = self._apply_ornament(note, ornament_type)
                ornamented.extend(ornament_notes)
                self.stats['ornaments_added'] += 1
            else:
                ornamented.append(note)

        return ornamented

    def _apply_ornament(self, note: Note, ornament_type: OrnamentType) -> List[Note]:
        """Apply specific ornament to note"""
        if ornament_type == OrnamentType.TRILL:
            return self._create_trill(note)
        elif ornament_type == OrnamentType.TURN:
            return self._create_turn(note)
        elif ornament_type == OrnamentType.MORDENT:
            return self._create_mordent(note, upper=True)
        elif ornament_type == OrnamentType.INVERTED_MORDENT:
            return self._create_mordent(note, upper=False)
        elif ornament_type == OrnamentType.APPOGGIATURA:
            return self._create_appoggiatura(note)
        elif ornament_type == OrnamentType.ACCIACCATURA:
            return self._create_acciaccatura(note)
        elif ornament_type == OrnamentType.GRACE_NOTE:
            return self._create_grace_note(note)
        else:
            return [note]

    def _create_trill(self, note: Note, interval: int = 2) -> List[Note]:
        """Create trill ornament"""
        # Alternate between main note and upper neighbor
        note_duration = note.duration / 4
        trill_notes = []
        current_time = note.start_time

        for i in range(4):
            pitch = note.pitch if i % 2 == 0 else note.pitch + interval
            trill_notes.append(Note(
                pitch=pitch,
                duration=note_duration,
                velocity=note.velocity,
                start_time=current_time,
                articulation="staccato"
            ))
            current_time += note_duration

        return trill_notes

    def _create_turn(self, note: Note) -> List[Note]:
        """Create turn ornament (upper neighbor, main, lower neighbor, main)"""
        note_duration = note.duration / 4
        current_time = note.start_time

        pitches = [
            note.pitch + 2,  # Upper neighbor
            note.pitch,      # Main note
            note.pitch - 2,  # Lower neighbor
            note.pitch       # Main note
        ]

        turn_notes = []
        for pitch in pitches:
            turn_notes.append(Note(
                pitch=max(0, min(127, pitch)),
                duration=note_duration,
                velocity=note.velocity,
                start_time=current_time
            ))
            current_time += note_duration

        return turn_notes

    def _create_mordent(self, note: Note, upper: bool = True) -> List[Note]:
        """Create mordent (main note, neighbor, main note)"""
        note_duration = note.duration / 3
        neighbor = note.pitch + (2 if upper else -2)

        return [
            Note(note.pitch, note_duration, note.velocity, note.start_time),
            Note(neighbor, note_duration, note.velocity, note.start_time + note_duration),
            Note(note.pitch, note_duration, note.velocity, note.start_time + 2*note_duration)
        ]

    def _create_appoggiatura(self, note: Note) -> List[Note]:
        """Create appoggiatura (accented non-chord tone before main note)"""
        grace_duration = note.duration * 0.5
        main_duration = note.duration * 0.5

        # Approach from above or below
        approach_pitch = note.pitch + random.choice([-2, 2])

        return [
            Note(approach_pitch, grace_duration, note.velocity, note.start_time),
            Note(note.pitch, main_duration, note.velocity, note.start_time + grace_duration)
        ]

    def _create_acciaccatura(self, note: Note) -> List[Note]:
        """Create acciaccatura (quick grace note)"""
        grace_duration = note.duration * 0.1
        main_duration = note.duration * 0.9

        grace_pitch = note.pitch + random.choice([-1, 1])

        return [
            Note(grace_pitch, grace_duration, int(note.velocity * 0.7), note.start_time),
            Note(note.pitch, main_duration, note.velocity, note.start_time + grace_duration)
        ]

    def _create_grace_note(self, note: Note) -> List[Note]:
        """Create grace note"""
        grace_duration = note.duration * 0.15
        main_duration = note.duration * 0.85

        grace_pitch = note.pitch - 2

        return [
            Note(grace_pitch, grace_duration, int(note.velocity * 0.8), note.start_time),
            Note(note.pitch, main_duration, note.velocity, note.start_time + grace_duration)
        ]

    # ========================================================================
    # PHRASE STRUCTURE ANALYSIS
    # ========================================================================

    def analyze_phrase_structure(
        self,
        melody: List[Note],
        phrase_length: float = 4.0
    ) -> List[Phrase]:
        """
        Analyze and segment melody into phrases.

        Args:
            melody: Melody to analyze
            phrase_length: Expected phrase length in beats

        Returns:
            List of detected phrases
        """
        if not melody:
            return []

        phrases = []
        current_phrase_notes = []
        phrase_start = melody[0].start_time

        for i, note in enumerate(melody):
            current_phrase_notes.append(note)

            # Check for phrase boundary
            if self._is_phrase_boundary(melody, i, phrase_length):
                # Create phrase
                phrase_end = note.start_time + note.duration
                motifs = self._extract_motifs(current_phrase_notes)

                phrases.append(Phrase(
                    motifs=motifs,
                    start_time=phrase_start,
                    end_time=phrase_end,
                    cadence_type=self._detect_cadence_type(current_phrase_notes)
                ))

                # Start new phrase
                if i < len(melody) - 1:
                    current_phrase_notes = []
                    phrase_start = melody[i+1].start_time

        # Add final phrase if notes remain
        if current_phrase_notes:
            phrase_end = current_phrase_notes[-1].start_time + current_phrase_notes[-1].duration
            motifs = self._extract_motifs(current_phrase_notes)
            phrases.append(Phrase(
                motifs=motifs,
                start_time=phrase_start,
                end_time=phrase_end,
                cadence_type=self._detect_cadence_type(current_phrase_notes)
            ))

        self.stats['phrases_analyzed'] += len(phrases)
        return phrases

    def _is_phrase_boundary(
        self,
        melody: List[Note],
        index: int,
        expected_length: float
    ) -> bool:
        """Detect if position is a phrase boundary"""
        if index >= len(melody) - 1:
            return True

        note = melody[index]
        next_note = melody[index + 1]

        # Check for rest (gap between notes)
        gap = next_note.start_time - (note.start_time + note.duration)
        if gap > 0.5:  # Significant rest
            return True

        # Check for long note (phrase ending)
        if note.duration >= 2.0:
            return True

        # Check for interval leap (melodic boundary)
        interval = abs(next_note.pitch - note.pitch)
        if interval >= 7:  # Fifth or larger
            return True

        return False

    def _extract_motifs(self, notes: List[Note], motif_length: int = 4) -> List[Motif]:
        """Extract motifs from notes"""
        if len(notes) < motif_length:
            return [Motif(notes=notes, name="phrase")]

        motifs = []
        for i in range(0, len(notes), motif_length):
            motif_notes = notes[i:i+motif_length]
            if motif_notes:
                motifs.append(Motif(
                    notes=motif_notes,
                    name=f"motif_{i//motif_length}",
                    category="extracted"
                ))

        return motifs

    def _detect_cadence_type(self, notes: List[Note]) -> str:
        """Detect cadence type at end of phrase"""
        if len(notes) < 2:
            return "none"

        # Simplified cadence detection
        final_interval = notes[-1].pitch - notes[-2].pitch

        if final_interval == -2 or final_interval == -1:
            return "authentic"  # Descending to tonic
        elif final_interval == 2 or final_interval == 1:
            return "half"  # Ascending to dominant
        elif notes[-1].duration >= 2.0:
            return "plagal"  # Long final note
        else:
            return "imperfect"

    # ========================================================================
    # COMPREHENSIVE MELODIC ANALYSIS
    # ========================================================================

    def analyze_melody(self, melody: List[Note]) -> MelodicAnalysis:
        """
        Comprehensive melodic analysis.

        Args:
            melody: Melody to analyze

        Returns:
            Complete melodic analysis
        """
        if not melody:
            return self._empty_analysis()

        pitches = [n.pitch for n in melody]

        # Contour analysis
        contour = self.analyze_contour(melody)

        # Interval analysis
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        intervallic_profile = defaultdict(int)
        for interval in intervals:
            intervallic_profile[interval] += 1

        # Stepwise motion
        stepwise = sum(1 for i in intervals if abs(i) <= 2)
        stepwise_ratio = stepwise / len(intervals) if intervals else 0

        # Leaps
        leap_count = sum(1 for i in intervals if abs(i) > 2)

        # Chromaticism
        chromatic_count = sum(1 for i in intervals if abs(i) == 1)
        chromaticism_score = chromatic_count / len(intervals) if intervals else 0

        # Sequence detection
        sequence_detected = self._detect_sequence(melody)

        # Phrase structure
        phrases = self.analyze_phrase_structure(melody)

        # Extract motifs
        motifs = []
        for phrase in phrases:
            motifs.extend(phrase.motifs)

        return MelodicAnalysis(
            contour_shape=contour['shape'],
            range_semitones=max(pitches) - min(pitches),
            tessitura=int(np.mean(pitches)),
            climax_position=contour['apex_position'],
            intervallic_profile=dict(intervallic_profile),
            chromaticism_score=chromaticism_score,
            stepwise_motion_ratio=stepwise_ratio,
            leap_count=leap_count,
            direction_changes=contour['direction_changes'],
            sequence_detected=sequence_detected,
            phrases=phrases,
            motifs=motifs
        )

    def _detect_sequence(self, melody: List[Note], min_repetitions: int = 2) -> bool:
        """Detect if melody contains sequences"""
        if len(melody) < 6:
            return False

        # Look for repeated interval patterns
        intervals = [melody[i+1].pitch - melody[i].pitch
                    for i in range(len(melody)-1)]

        # Check for pattern repetition
        for pattern_length in range(2, len(intervals) // 2):
            for start in range(len(intervals) - pattern_length * min_repetitions):
                pattern = intervals[start:start+pattern_length]

                # Check if pattern repeats
                repeats = 1
                pos = start + pattern_length
                while pos + pattern_length <= len(intervals):
                    next_pattern = intervals[pos:pos+pattern_length]
                    # Allow transposition
                    if self._patterns_similar(pattern, next_pattern):
                        repeats += 1
                        pos += pattern_length
                    else:
                        break

                if repeats >= min_repetitions:
                    return True

        return False

    def _patterns_similar(self, pattern1: List[int], pattern2: List[int]) -> bool:
        """Check if two interval patterns are similar (allowing transposition)"""
        if len(pattern1) != len(pattern2):
            return False

        # Check if patterns have same shape (intervals match exactly)
        return pattern1 == pattern2

    def _empty_analysis(self) -> MelodicAnalysis:
        """Return empty analysis"""
        return MelodicAnalysis(
            contour_shape=ContourShape.PLATEAU,
            range_semitones=0,
            tessitura=60,
            climax_position=0.5,
            intervallic_profile={},
            chromaticism_score=0.0,
            stepwise_motion_ratio=0.0,
            leap_count=0,
            direction_changes=0,
            sequence_detected=False,
            phrases=[],
            motifs=[]
        )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _build_scale(self, key: str, mode: str) -> List[int]:
        """Build scale for key and mode"""
        # Simplified scale construction
        root_notes = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }

        root = root_notes.get(key, 0)

        # Major scale intervals
        if mode == "major":
            intervals = [0, 2, 4, 5, 7, 9, 11]
        elif mode == "minor":
            intervals = [0, 2, 3, 5, 7, 8, 10]
        elif mode == "dorian":
            intervals = [0, 2, 3, 5, 7, 9, 10]
        elif mode == "mixolydian":
            intervals = [0, 2, 4, 5, 7, 9, 10]
        else:
            intervals = [0, 2, 4, 5, 7, 9, 11]  # Default to major

        return [(root + i) % 12 for i in intervals]

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics"""
        return self.stats.copy()

    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'motifs_developed': 0,
            'sequences_generated': 0,
            'ornaments_added': 0,
            'phrases_analyzed': 0
        }


# ============================================================================
# MODULE INTERFACE
# ============================================================================

def create_example_melody() -> List[Note]:
    """Create example melody for testing"""
    pitches = [60, 62, 64, 65, 67, 65, 64, 62, 60]
    notes = []
    current_time = 0.0

    for pitch in pitches:
        notes.append(Note(
            pitch=pitch,
            duration=1.0,
            velocity=80,
            start_time=current_time
        ))
        current_time += 1.0

    return notes


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AGENT 19: MELODY SPECIALIST")
    print("=" * 80)
    print()

    # Initialize specialist
    specialist = MelodySpecialist(key="C", mode="major")

    # Create example melody
    melody = create_example_melody()
    print(f"Created example melody with {len(melody)} notes")
    print()

    # Test motif development
    print("Testing Motif Development...")
    motif = Motif(notes=melody[:4], name="example_motif")
    techniques = [
        MotifTransformation.INVERSION,
        MotifTransformation.RETROGRADE,
        MotifTransformation.AUGMENTATION
    ]
    variations = specialist.develop_motif(motif, techniques, n_variations=3)
    print(f"  Generated {len(variations)} variations")
    print()

    # Test sequence generation
    print("Testing Sequence Generation...")
    sequence = specialist.generate_sequence(
        motif,
        SequenceType.ASCENDING,
        repetitions=3,
        interval=2
    )
    print(f"  Generated sequence with {len(sequence)} notes")
    print()

    # Test contour optimization
    print("Testing Contour Optimization...")
    optimized = specialist.optimize_contour(
        melody,
        ContourShape.ARCH,
        smoothing=0.5
    )
    print(f"  Optimized melody to arch shape")
    print()

    # Test ornamentation
    print("Testing Ornamentation...")
    ornament_types = [OrnamentType.TRILL, OrnamentType.TURN, OrnamentType.MORDENT]
    ornamented = specialist.add_ornamentation(melody, ornament_types, density=0.3)
    print(f"  Added ornaments: {len(ornamented)} notes (from {len(melody)})")
    print()

    # Test phrase analysis
    print("Testing Phrase Structure Analysis...")
    phrases = specialist.analyze_phrase_structure(melody)
    print(f"  Detected {len(phrases)} phrases")
    for i, phrase in enumerate(phrases):
        print(f"    Phrase {i+1}: {len(phrase.motifs)} motifs, cadence: {phrase.cadence_type}")
    print()

    # Test comprehensive analysis
    print("Testing Comprehensive Melodic Analysis...")
    analysis = specialist.analyze_melody(melody)
    print(f"  Contour Shape: {analysis.contour_shape.value}")
    print(f"  Range: {analysis.range_semitones} semitones")
    print(f"  Tessitura: {analysis.tessitura}")
    print(f"  Climax Position: {analysis.climax_position:.2f}")
    print(f"  Stepwise Motion: {analysis.stepwise_motion_ratio:.1%}")
    print(f"  Chromaticism: {analysis.chromaticism_score:.1%}")
    print(f"  Direction Changes: {analysis.direction_changes}")
    print(f"  Sequence Detected: {analysis.sequence_detected}")
    print()

    # Show statistics
    stats = specialist.get_statistics()
    print("Processing Statistics:")
    print(f"  Motifs Developed: {stats['motifs_developed']}")
    print(f"  Sequences Generated: {stats['sequences_generated']}")
    print(f"  Ornaments Added: {stats['ornaments_added']}")
    print(f"  Phrases Analyzed: {stats['phrases_analyzed']}")
    print()

    print("=" * 80)
    print("✅ AGENT 19 MELODY SPECIALIST TEST COMPLETE")
    print("=" * 80)
