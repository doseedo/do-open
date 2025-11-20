#!/usr/bin/env python3
"""
Harmony Specialist - Agent 18
==============================

Expert module for advanced harmony analysis and generation, including:
- Jazz voicings (drop 2, drop 3, rootless, close position)
- Modal harmony (Dorian, Mixolydian, Lydian, etc.)
- Functional harmony (Tonic-Subdominant-Dominant relationships)
- Voice leading optimization (smooth voice motion, contrary motion)
- Reharmonization techniques (substitute chords, passing chords, approach chords)
- Chord extensions and alterations (9ths, 11ths, 13ths, altered dominants)
- Harmonic rhythm analysis
- Tension-resolution patterns

This module extends Agent 3's basic harmony parameters with 50+ specialized
harmony parameters for advanced harmonic generation and analysis.

Research Foundations
--------------------

**Jazz Voicings:**
- Levine (1995): "The Jazz Piano Book" - Comprehensive voicing techniques
- Dobbins (1984): "A Creative Approach to Jazz Piano Harmony"
- Haerle (1982): "Scales for Jazz Improvisation"

**Modal Harmony:**
- Russell (1953): "Lydian Chromatic Concept of Tonal Organization"
- Persichetti (1961): "Twentieth-Century Harmony"
- Vincent (1951): "The Diatonic Modes in Modern Music"

**Functional Harmony:**
- Riemann (1893): "Harmony Simplified"
- Schoenberg (1911): "Theory of Harmony"
- Rameau (1722): "Treatise on Harmony"

**Voice Leading:**
- Tymoczko (2011): "A Geometry of Music"
- Straus (2016): "Introduction to Post-Tonal Theory"
- Schuijer (2008): "Analyzing Atonal Music"

**Reharmonization:**
- Mehegan (1984): "Jazz Improvisation" (4 volumes)
- Haerle (1975): "The Jazz Language"
- Ligon (1996): "Jazz Theory Resources"

Author: Agent 18 - Harmony Specialist
Date: 2025
License: MIT
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum, auto
from pathlib import Path
import json
import math

try:
    import numpy as np
    from scipy import signal, stats
    from scipy.spatial.distance import euclidean
    NUMPY_AVAILABLE = True
    NDArray = np.ndarray
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy/SciPy not available. Install with: pip install numpy scipy")
    NDArray = Any

try:
    import mido
    from mido import MidiFile
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("Warning: mido not available. Install with: pip install mido")


# ==============================================================================
# ENUMERATIONS
# ==============================================================================

class VoicingType(Enum):
    """Types of jazz voicings."""
    CLOSE_POSITION = "close_position"          # All notes within an octave
    OPEN_POSITION = "open_position"            # Notes spread over more than an octave
    DROP_2 = "drop_2"                          # Second voice from top dropped an octave
    DROP_3 = "drop_3"                          # Third voice from top dropped an octave
    DROP_2_4 = "drop_2_4"                      # Second and fourth voices dropped
    ROOTLESS_A = "rootless_a"                  # 3-7-9-13 (without root)
    ROOTLESS_B = "rootless_b"                  # 7-9-3-13 (without root)
    FOURTHS = "fourths"                        # Quartal voicing
    CLUSTERS = "clusters"                      # Tone clusters
    SHELL = "shell"                            # Root-3rd-7th only
    SPREAD = "spread"                          # Wide spacing
    LOCKED_HANDS = "locked_hands"              # Melody doubled an octave with chords


class ChordQuality(Enum):
    """Chord quality types."""
    MAJOR = "major"                            # Major triad
    MINOR = "minor"                            # Minor triad
    DIMINISHED = "diminished"                  # Diminished triad
    AUGMENTED = "augmented"                    # Augmented triad
    MAJOR_7 = "major_7"                        # Major 7th
    MINOR_7 = "minor_7"                        # Minor 7th
    DOMINANT_7 = "dominant_7"                  # Dominant 7th
    HALF_DIMINISHED = "half_diminished"        # Half-diminished 7th
    FULLY_DIMINISHED = "fully_diminished"      # Fully diminished 7th
    MINOR_MAJOR_7 = "minor_major_7"            # Minor-major 7th
    AUGMENTED_7 = "augmented_7"                # Augmented 7th
    SUS_2 = "sus_2"                            # Suspended 2nd
    SUS_4 = "sus_4"                            # Suspended 4th


class Mode(Enum):
    """Musical modes."""
    IONIAN = "ionian"              # Major scale
    DORIAN = "dorian"              # Minor with raised 6th
    PHRYGIAN = "phrygian"          # Minor with lowered 2nd
    LYDIAN = "lydian"              # Major with raised 4th
    MIXOLYDIAN = "mixolydian"      # Major with lowered 7th
    AEOLIAN = "aeolian"            # Natural minor
    LOCRIAN = "locrian"            # Diminished
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"
    LYDIAN_DOMINANT = "lydian_dominant"  # Mixolydian #4
    ALTERED = "altered"            # Superlocrian
    DIMINISHED = "diminished"      # Octatonic
    WHOLE_TONE = "whole_tone"


class HarmonicFunction(Enum):
    """Functional harmony categories."""
    TONIC = "tonic"                # I, vi (stable)
    SUBDOMINANT = "subdominant"    # IV, ii (preparation)
    DOMINANT = "dominant"          # V, vii° (tension)
    PREDOMINANT = "predominant"    # IV, ii, vi (pre-dominant)
    SUBSTITUTE_DOMINANT = "substitute_dominant"  # SubV7
    CHROMATIC_MEDIANT = "chromatic_mediant"
    SECONDARY_DOMINANT = "secondary_dominant"
    BORROWED = "borrowed"          # Modal interchange
    PASSING = "passing"            # Passing chord
    APPROACH = "approach"          # Approach chord
    PEDAL = "pedal"                # Pedal point


class ReharmonizationTechnique(Enum):
    """Reharmonization techniques."""
    TRITONE_SUBSTITUTION = "tritone_substitution"
    DIATONIC_SUBSTITUTION = "diatonic_substitution"
    CHROMATIC_APPROACH = "chromatic_approach"
    DIMINISHED_PASSING = "diminished_passing"
    MODAL_INTERCHANGE = "modal_interchange"
    SECONDARY_DOMINANTS = "secondary_dominants"
    EXTENDED_DOMINANTS = "extended_dominants"
    COLTRANE_CHANGES = "coltrane_changes"
    PEDAL_POINT = "pedal_point"
    PARALLEL_MOTION = "parallel_motion"
    CONSTANT_STRUCTURE = "constant_structure"


class VoiceLeadingRule(Enum):
    """Voice leading rules and guidelines."""
    SMOOTH_MOTION = "smooth_motion"              # Prefer stepwise motion
    CONTRARY_MOTION = "contrary_motion"          # Voices move in opposite directions
    OBLIQUE_MOTION = "oblique_motion"            # One voice static
    PARALLEL_MOTION = "parallel_motion"          # Voices move in same direction
    NO_PARALLEL_FIFTHS = "no_parallel_fifths"    # Avoid parallel perfect 5ths
    NO_PARALLEL_OCTAVES = "no_parallel_octaves"  # Avoid parallel octaves
    RESOLVE_LEADING_TONE = "resolve_leading_tone"  # 7th scale degree up to tonic
    RESOLVE_7TH_DOWN = "resolve_7th_down"        # Chord 7th resolves down
    COMMON_TONE = "common_tone"                  # Keep common tones


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class Note:
    """Represents a single note."""
    pitch: int                    # MIDI pitch (0-127)
    velocity: int = 64            # Velocity (0-127)
    start_time: float = 0.0       # Start time in beats
    duration: float = 1.0         # Duration in beats
    channel: int = 0              # MIDI channel

    @property
    def pitch_class(self) -> int:
        """Pitch class (0-11)."""
        return self.pitch % 12

    @property
    def octave(self) -> int:
        """Octave number."""
        return self.pitch // 12

    def __repr__(self) -> str:
        return f"Note(pitch={self.pitch}, vel={self.velocity}, start={self.start_time:.2f}, dur={self.duration:.2f})"


@dataclass
class Chord:
    """Represents a chord with analysis."""
    pitches: List[int]                          # MIDI pitches
    root: Optional[int] = None                  # Root pitch class (0-11)
    quality: Optional[ChordQuality] = None      # Chord quality
    extensions: List[int] = field(default_factory=list)  # Extensions (9, 11, 13)
    alterations: List[str] = field(default_factory=list)  # e.g., "b9", "#11"
    inversion: int = 0                          # 0=root, 1=first, 2=second
    bass: Optional[int] = None                  # Bass note pitch class
    function: Optional[HarmonicFunction] = None  # Harmonic function
    voicing_type: Optional[VoicingType] = None  # Voicing type
    start_time: float = 0.0                     # Start time in beats
    duration: float = 1.0                       # Duration in beats

    @property
    def pitch_classes(self) -> Set[int]:
        """Set of pitch classes in chord."""
        return {p % 12 for p in self.pitches}

    @property
    def cardinality(self) -> int:
        """Number of unique pitch classes."""
        return len(self.pitch_classes)

    @property
    def span(self) -> int:
        """Span in semitones from lowest to highest note."""
        if not self.pitches:
            return 0
        return max(self.pitches) - min(self.pitches)

    def __repr__(self) -> str:
        return f"Chord(root={self.root}, quality={self.quality}, pitches={self.pitches})"


@dataclass
class VoicingAnalysis:
    """Analysis of a chord voicing."""
    voicing_type: VoicingType
    spacing: List[int]                          # Intervals between adjacent voices
    average_spacing: float                      # Average interval size
    range: int                                  # Total range in semitones
    density: float                              # Notes per octave
    doubling: List[int]                         # Doubled pitch classes
    voice_count: int                            # Number of voices
    open_strings: List[int] = field(default_factory=list)  # For guitar voicings
    fingering_difficulty: float = 0.0           # 0-1 scale

    def __repr__(self) -> str:
        return f"VoicingAnalysis(type={self.voicing_type}, voices={self.voice_count}, range={self.range})"


@dataclass
class ChordProgression:
    """Represents a chord progression."""
    chords: List[Chord]
    key: Optional[int] = None                   # Key center (0-11)
    mode: Optional[Mode] = None                 # Modal center
    functions: List[HarmonicFunction] = field(default_factory=list)
    harmonic_rhythm: List[float] = field(default_factory=list)  # Duration of each chord
    modulations: List[Tuple[int, int]] = field(default_factory=list)  # (index, new_key)

    def __len__(self) -> int:
        return len(self.chords)

    def __getitem__(self, index: int) -> Chord:
        return self.chords[index]


@dataclass
class VoiceLeadingAnalysis:
    """Analysis of voice leading between two chords."""
    chord1: Chord
    chord2: Chord
    voice_motions: List[int]                    # Semitones moved per voice
    motion_types: List[str]                     # Type of motion per voice pair
    total_motion: int                           # Sum of absolute motions
    smoothness: float                           # 0-1, higher is smoother
    common_tones: int                           # Number of common tones
    violations: List[str]                       # Voice leading violations
    quality_score: float                        # Overall quality 0-1

    def __repr__(self) -> str:
        return f"VoiceLeading(motion={self.total_motion}, smooth={self.smoothness:.2f}, common={self.common_tones})"


@dataclass
class HarmonyFeatures:
    """Comprehensive harmony features for a piece."""
    # Voicing features (20 parameters)
    voicing_types: Dict[VoicingType, float] = field(default_factory=dict)
    avg_voicing_density: float = 0.0
    avg_voicing_range: float = 0.0
    close_position_ratio: float = 0.0
    open_position_ratio: float = 0.0
    drop2_usage: float = 0.0
    drop3_usage: float = 0.0
    rootless_usage: float = 0.0
    shell_voicing_usage: float = 0.0
    fourths_usage: float = 0.0
    cluster_usage: float = 0.0

    # Modal harmony features (15 parameters)
    primary_mode: Optional[Mode] = None
    mode_distribution: Dict[Mode, float] = field(default_factory=dict)
    modal_mixture: float = 0.0
    modal_interchange_events: int = 0
    dorian_characteristic: float = 0.0
    lydian_characteristic: float = 0.0
    mixolydian_characteristic: float = 0.0
    phrygian_characteristic: float = 0.0
    locrian_characteristic: float = 0.0
    altered_scale_usage: float = 0.0
    whole_tone_usage: float = 0.0
    diminished_scale_usage: float = 0.0

    # Functional harmony features (10 parameters)
    tonic_ratio: float = 0.0
    subdominant_ratio: float = 0.0
    dominant_ratio: float = 0.0
    secondary_dominant_count: int = 0
    tritone_sub_usage: float = 0.0
    modal_interchange_ratio: float = 0.0
    chromatic_mediant_usage: float = 0.0
    functional_clarity: float = 0.0              # How clear the functional progression is
    cadence_strength: float = 0.0                # Strength of cadential points
    harmonic_tension_curve: List[float] = field(default_factory=list)

    # Voice leading features (15 parameters)
    avg_voice_motion: float = 0.0
    stepwise_motion_ratio: float = 0.0
    leap_ratio: float = 0.0
    contrary_motion_ratio: float = 0.0
    parallel_motion_ratio: float = 0.0
    oblique_motion_ratio: float = 0.0
    common_tone_retention: float = 0.0
    voice_leading_smoothness: float = 0.0
    voice_crossing_count: int = 0
    parallel_fifth_count: int = 0
    parallel_octave_count: int = 0
    voice_independence: float = 0.0              # How independent the voices are
    leading_tone_resolution: float = 0.0
    seventh_resolution: float = 0.0
    voice_range_per_voice: List[int] = field(default_factory=list)


# ==============================================================================
# HARMONY SPECIALIST
# ==============================================================================

class HarmonySpecialist:
    """
    Advanced harmony analysis and generation specialist.

    This agent provides:
    1. Jazz voicing analysis and generation
    2. Modal harmony detection and progression generation
    3. Functional harmony analysis
    4. Voice leading optimization
    5. Reharmonization techniques
    6. Chord extension and alteration analysis
    7. Harmonic rhythm analysis
    8. 50+ specialized harmony parameters
    """

    # Chord templates (intervals from root)
    CHORD_TEMPLATES = {
        ChordQuality.MAJOR: [0, 4, 7],
        ChordQuality.MINOR: [0, 3, 7],
        ChordQuality.DIMINISHED: [0, 3, 6],
        ChordQuality.AUGMENTED: [0, 4, 8],
        ChordQuality.MAJOR_7: [0, 4, 7, 11],
        ChordQuality.MINOR_7: [0, 3, 7, 10],
        ChordQuality.DOMINANT_7: [0, 4, 7, 10],
        ChordQuality.HALF_DIMINISHED: [0, 3, 6, 10],
        ChordQuality.FULLY_DIMINISHED: [0, 3, 6, 9],
        ChordQuality.MINOR_MAJOR_7: [0, 3, 7, 11],
        ChordQuality.AUGMENTED_7: [0, 4, 8, 10],
        ChordQuality.SUS_2: [0, 2, 7],
        ChordQuality.SUS_4: [0, 5, 7],
    }

    # Mode templates (intervals from tonic)
    MODE_TEMPLATES = {
        Mode.IONIAN: [0, 2, 4, 5, 7, 9, 11],         # Major
        Mode.DORIAN: [0, 2, 3, 5, 7, 9, 10],
        Mode.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
        Mode.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
        Mode.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
        Mode.AEOLIAN: [0, 2, 3, 5, 7, 8, 10],        # Natural minor
        Mode.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
        Mode.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
        Mode.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],
        Mode.LYDIAN_DOMINANT: [0, 2, 4, 6, 7, 9, 10],
        Mode.ALTERED: [0, 1, 3, 4, 6, 8, 10],
        Mode.DIMINISHED: [0, 2, 3, 5, 6, 8, 9, 11],  # Half-whole
        Mode.WHOLE_TONE: [0, 2, 4, 6, 8, 10],
    }

    # Functional harmony: scale degree to function mapping (in major)
    FUNCTIONAL_MAP = {
        0: HarmonicFunction.TONIC,           # I
        1: HarmonicFunction.PASSING,         # bII
        2: HarmonicFunction.SUBDOMINANT,     # ii
        3: HarmonicFunction.CHROMATIC_MEDIANT,  # bIII
        4: HarmonicFunction.SUBDOMINANT,     # IV
        5: HarmonicFunction.DOMINANT,        # V
        6: HarmonicFunction.TONIC,           # vi
        7: HarmonicFunction.DOMINANT,        # vii°
        8: HarmonicFunction.TONIC,           # I (octave)
        9: HarmonicFunction.SUBDOMINANT,     # ii
        10: HarmonicFunction.DOMINANT,       # V
        11: HarmonicFunction.DOMINANT,       # viio
    }

    def __init__(self):
        """Initialize the Harmony Specialist."""
        self.chords: List[Chord] = []
        self.key: Optional[int] = None
        self.mode: Optional[Mode] = None
        self.features: Optional[HarmonyFeatures] = None

    # ==========================================================================
    # MIDI PARSING
    # ==========================================================================

    def load_midi(self, midi_path: Union[str, Path]) -> None:
        """
        Load and parse MIDI file to extract chords.

        Args:
            midi_path: Path to MIDI file
        """
        if not MIDO_AVAILABLE:
            raise ImportError("mido is required for MIDI loading")

        midi_path = Path(midi_path)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        midi = MidiFile(midi_path)
        notes = self._extract_notes_from_midi(midi)
        self.chords = self._notes_to_chords(notes)
        self.key = self._detect_key(notes)
        self.mode = self._detect_mode(notes, self.key)

    def _extract_notes_from_midi(self, midi: 'MidiFile') -> List[Note]:
        """Extract notes from MIDI file."""
        notes = []
        for track in midi.tracks:
            time = 0.0
            active_notes = {}  # pitch -> start_time

            for msg in track:
                time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (time, msg.velocity, msg.channel)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity, channel = active_notes[msg.note]
                        duration = time - start_time
                        notes.append(Note(
                            pitch=msg.note,
                            velocity=velocity,
                            start_time=start_time,
                            duration=duration,
                            channel=channel
                        ))
                        del active_notes[msg.note]

        return sorted(notes, key=lambda n: n.start_time)

    def _notes_to_chords(self, notes: List[Note], time_window: float = 0.5) -> List[Chord]:
        """
        Convert notes to chords using temporal proximity.

        Args:
            notes: List of notes
            time_window: Time window in beats for chord grouping

        Returns:
            List of detected chords
        """
        if not notes:
            return []

        chords = []
        current_chord_notes = []
        current_start = notes[0].start_time

        for note in notes:
            if note.start_time - current_start <= time_window:
                current_chord_notes.append(note)
            else:
                if current_chord_notes:
                    chord = self._analyze_chord([n.pitch for n in current_chord_notes])
                    chord.start_time = current_start
                    chord.duration = note.start_time - current_start
                    chords.append(chord)

                current_chord_notes = [note]
                current_start = note.start_time

        # Add final chord
        if current_chord_notes:
            chord = self._analyze_chord([n.pitch for n in current_chord_notes])
            chord.start_time = current_start
            if notes:
                chord.duration = max(n.start_time + n.duration for n in notes) - current_start
            chords.append(chord)

        return chords

    # ==========================================================================
    # CHORD ANALYSIS
    # ==========================================================================

    def _analyze_chord(self, pitches: List[int]) -> Chord:
        """
        Analyze a chord to determine root, quality, extensions, etc.

        Args:
            pitches: List of MIDI pitches

        Returns:
            Analyzed Chord object
        """
        if not pitches:
            return Chord(pitches=[])

        pitch_classes = sorted(set(p % 12 for p in pitches))
        bass = min(pitches) % 12

        # Try to identify root and quality
        root, quality = self._identify_chord_quality(pitch_classes)

        # Detect extensions and alterations
        extensions, alterations = self._detect_extensions_alterations(pitch_classes, root, quality)

        # Detect inversion
        inversion = self._detect_inversion(pitch_classes, root)

        return Chord(
            pitches=sorted(pitches),
            root=root,
            quality=quality,
            extensions=extensions,
            alterations=alterations,
            inversion=inversion,
            bass=bass
        )

    def _identify_chord_quality(self, pitch_classes: List[int]) -> Tuple[Optional[int], Optional[ChordQuality]]:
        """
        Identify the root and quality of a chord.

        Args:
            pitch_classes: Sorted list of pitch classes

        Returns:
            Tuple of (root, quality)
        """
        if len(pitch_classes) < 2:
            return (pitch_classes[0] if pitch_classes else None, None)

        # Try each pitch class as potential root
        best_match = (None, None, 0)

        for root in pitch_classes:
            for quality, template in self.CHORD_TEMPLATES.items():
                # Normalize chord to root
                normalized = [(pc - root) % 12 for pc in pitch_classes]

                # Check how many template notes are present
                matches = sum(1 for interval in template if interval in normalized)
                match_ratio = matches / len(template)

                if match_ratio > best_match[2]:
                    best_match = (root, quality, match_ratio)

        # Return best match if good enough
        if best_match[2] >= 0.6:  # At least 60% match
            return (best_match[0], best_match[1])

        # Default to using lowest note as root
        return (pitch_classes[0], None)

    def _detect_extensions_alterations(self, pitch_classes: List[int], root: Optional[int],
                                      quality: Optional[ChordQuality]) -> Tuple[List[int], List[str]]:
        """
        Detect chord extensions (9, 11, 13) and alterations (b9, #11, etc.).

        Args:
            pitch_classes: List of pitch classes
            root: Root pitch class
            quality: Chord quality

        Returns:
            Tuple of (extensions list, alterations list)
        """
        if root is None or quality is None:
            return ([], [])

        extensions = []
        alterations = []

        # Normalize to root
        intervals = sorted(set((pc - root) % 12 for pc in pitch_classes))

        # Extension intervals
        extension_map = {
            2: (9, None),      # 9th
            1: (9, 'b9'),      # b9
            3: (9, '#9'),      # #9
            5: (11, None),     # 11th
            6: (11, '#11'),    # #11
            9: (13, None),     # 13th
            8: (13, 'b13'),    # b13
        }

        for interval, (ext, alt) in extension_map.items():
            if interval in intervals:
                if ext not in extensions:
                    extensions.append(ext)
                if alt:
                    alterations.append(alt)

        return (extensions, alterations)

    def _detect_inversion(self, pitch_classes: List[int], root: Optional[int]) -> int:
        """
        Detect chord inversion.

        Args:
            pitch_classes: List of pitch classes
            root: Root pitch class

        Returns:
            Inversion number (0=root, 1=first, 2=second, etc.)
        """
        if root is None or not pitch_classes:
            return 0

        bass = pitch_classes[0]
        if bass == root:
            return 0

        # Count how many chord tones are below the root
        root_idx = pitch_classes.index(root) if root in pitch_classes else 0
        return root_idx

    def analyze_jazz_voicings(self, chord: Chord) -> VoicingAnalysis:
        """
        Analyze jazz voicing characteristics of a chord.

        Args:
            chord: Chord to analyze

        Returns:
            VoicingAnalysis object
        """
        if not chord.pitches:
            return VoicingAnalysis(
                voicing_type=VoicingType.CLOSE_POSITION,
                spacing=[],
                average_spacing=0.0,
                range=0,
                density=0.0,
                doubling=[],
                voice_count=0
            )

        pitches = sorted(chord.pitches)
        voice_count = len(pitches)

        # Calculate spacing between adjacent voices
        spacing = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
        avg_spacing = sum(spacing) / len(spacing) if spacing else 0.0

        # Calculate range
        chord_range = pitches[-1] - pitches[0]

        # Calculate density (notes per octave)
        density = voice_count / (chord_range / 12 + 1) if chord_range > 0 else voice_count

        # Detect doubling
        pitch_classes = [p % 12 for p in pitches]
        doubling = [pc for pc, count in Counter(pitch_classes).items() if count > 1]

        # Identify voicing type
        voicing_type = self._identify_voicing_type(pitches, spacing, chord_range, chord)

        return VoicingAnalysis(
            voicing_type=voicing_type,
            spacing=spacing,
            average_spacing=avg_spacing,
            range=chord_range,
            density=density,
            doubling=doubling,
            voice_count=voice_count
        )

    def _identify_voicing_type(self, pitches: List[int], spacing: List[int],
                              chord_range: int, chord: Chord) -> VoicingType:
        """
        Identify the voicing type based on spacing and structure.

        Args:
            pitches: Sorted MIDI pitches
            spacing: Intervals between adjacent voices
            chord_range: Total range in semitones
            chord: Original chord object

        Returns:
            VoicingType
        """
        if chord_range <= 12:
            return VoicingType.CLOSE_POSITION

        # Check for rootless voicings (no root in chord)
        if chord.root is not None:
            pitch_classes = [p % 12 for p in pitches]
            if chord.root not in pitch_classes:
                # Check structure for rootless A vs B
                if len(pitches) >= 4:
                    intervals = [(pitches[i+1] - pitches[i]) % 12 for i in range(min(3, len(pitches)-1))]
                    # Rootless A: typically 3-7-9-13
                    # Rootless B: typically 7-9-3-13
                    if intervals[:2] == [4, 3] or intervals[:2] == [3, 4]:
                        return VoicingType.ROOTLESS_A
                    else:
                        return VoicingType.ROOTLESS_B

        # Check for drop 2 voicing
        if len(pitches) == 4:
            # In drop 2, second from top is dropped an octave
            # This creates a characteristic spacing pattern
            if len(spacing) >= 3 and spacing[1] > 7:
                return VoicingType.DROP_2

        # Check for drop 3 voicing
        if len(pitches) == 4:
            if len(spacing) >= 3 and spacing[0] > 7:
                return VoicingType.DROP_3

        # Check for fourths voicing (quartal harmony)
        if spacing and all(4 <= s <= 6 for s in spacing):
            return VoicingType.FOURTHS

        # Check for shell voicing (root-3rd-7th only)
        if len(pitches) == 3 and chord.root is not None:
            normalized = [(p - pitches[0]) % 12 for p in pitches]
            if normalized in [[0, 4, 11], [0, 3, 10]]:  # Maj7 or min7 shell
                return VoicingType.SHELL

        # Check for clusters (seconds)
        if spacing and all(s <= 3 for s in spacing):
            return VoicingType.CLUSTERS

        # Default to open position
        return VoicingType.OPEN_POSITION

    # ==========================================================================
    # MODAL HARMONY
    # ==========================================================================

    def _detect_key(self, notes: List[Note]) -> Optional[int]:
        """
        Detect the key of a piece using pitch class distribution.

        Args:
            notes: List of notes

        Returns:
            Key pitch class (0-11) or None
        """
        if not notes:
            return None

        # Count pitch class occurrences weighted by duration
        pc_weights = defaultdict(float)
        for note in notes:
            pc_weights[note.pitch_class] += note.duration

        # Use Krumhansl-Schmuckler key-finding algorithm
        # Key profiles for major and minor
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        best_key = 0
        best_correlation = -1.0

        for tonic in range(12):
            # Major key correlation
            observed = [pc_weights.get((tonic + i) % 12, 0.0) for i in range(12)]
            if sum(observed) > 0:
                correlation = self._correlation(observed, major_profile)
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_key = tonic

            # Minor key correlation
            correlation = self._correlation(observed, minor_profile)
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = tonic

        return best_key

    def _detect_mode(self, notes: List[Note], key: Optional[int]) -> Optional[Mode]:
        """
        Detect the modal center of a piece.

        Args:
            notes: List of notes
            key: Detected key

        Returns:
            Detected Mode or None
        """
        if not notes or key is None:
            return None

        # Count scale degrees relative to key
        scale_degrees = defaultdict(float)
        for note in notes:
            degree = (note.pitch_class - key) % 12
            scale_degrees[degree] += note.duration

        # Convert to list
        degree_list = [scale_degrees.get(i, 0.0) for i in range(12)]

        # Compare against mode templates
        best_mode = Mode.IONIAN
        best_match = 0.0

        for mode, template in self.MODE_TEMPLATES.items():
            # Calculate match score
            match = sum(degree_list[t] for t in template)
            if match > best_match:
                best_match = match
                best_mode = mode

        return best_mode

    def _correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if not x or not y or len(x) != len(y):
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)

        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        if denominator == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def generate_modal_progression(self, mode: Mode, length: int = 4,
                                   key: int = 0) -> ChordProgression:
        """
        Generate a modal chord progression.

        Args:
            mode: Mode to use
            length: Number of chords
            key: Tonic pitch class (0-11)

        Returns:
            ChordProgression object
        """
        if mode not in self.MODE_TEMPLATES:
            mode = Mode.IONIAN

        scale = [(key + interval) % 12 for interval in self.MODE_TEMPLATES[mode]]

        chords = []
        # Generate chords built on scale degrees
        for i in range(length):
            degree = i % len(scale)
            root = scale[degree]

            # Build triad from scale
            third = scale[(degree + 2) % len(scale)]
            fifth = scale[(degree + 4) % len(scale)]

            pitches = [root + 60, third + 60, fifth + 60]  # Middle octave
            chord = self._analyze_chord(pitches)
            chord.start_time = float(i * 4)  # 4 beats per chord
            chord.duration = 4.0
            chords.append(chord)

        return ChordProgression(chords=chords, key=key, mode=mode)

    # ==========================================================================
    # FUNCTIONAL HARMONY
    # ==========================================================================

    def analyze_functional_harmony(self, progression: ChordProgression) -> List[HarmonicFunction]:
        """
        Analyze the harmonic function of each chord in a progression.

        Args:
            progression: ChordProgression to analyze

        Returns:
            List of HarmonicFunction values
        """
        if progression.key is None:
            return [HarmonicFunction.TONIC] * len(progression)

        functions = []
        for chord in progression.chords:
            if chord.root is None:
                functions.append(HarmonicFunction.PASSING)
                continue

            # Calculate scale degree relative to key
            degree = (chord.root - progression.key) % 12

            # Map to function
            function = self.FUNCTIONAL_MAP.get(degree, HarmonicFunction.PASSING)

            # Refine based on chord quality
            if chord.quality == ChordQuality.DOMINANT_7:
                if degree == 5:
                    function = HarmonicFunction.DOMINANT
                elif degree == 1:  # bII7
                    function = HarmonicFunction.SUBSTITUTE_DOMINANT
                else:
                    function = HarmonicFunction.SECONDARY_DOMINANT

            functions.append(function)

        return functions

    # ==========================================================================
    # VOICE LEADING
    # ==========================================================================

    def analyze_voice_leading(self, chord1: Chord, chord2: Chord) -> VoiceLeadingAnalysis:
        """
        Analyze voice leading between two chords.

        Args:
            chord1: First chord
            chord2: Second chord

        Returns:
            VoiceLeadingAnalysis object
        """
        # Align voices (match cardinality)
        voices1 = sorted(chord1.pitches)
        voices2 = sorted(chord2.pitches)

        # Pad shorter chord if needed
        while len(voices1) < len(voices2):
            voices1.append(voices1[-1])
        while len(voices2) < len(voices1):
            voices2.append(voices2[-1])

        # Calculate voice motions
        voice_motions = [v2 - v1 for v1, v2 in zip(voices1, voices2)]
        total_motion = sum(abs(m) for m in voice_motions)

        # Analyze motion types
        motion_types = []
        for i in range(len(voices1) - 1):
            m1 = voice_motions[i]
            m2 = voice_motions[i + 1]

            if m1 == 0 and m2 == 0:
                motion_types.append("static")
            elif m1 == 0 or m2 == 0:
                motion_types.append("oblique")
            elif m1 * m2 < 0:  # Opposite signs
                motion_types.append("contrary")
            elif m1 * m2 > 0:  # Same sign
                motion_types.append("parallel")
            else:
                motion_types.append("similar")

        # Count common tones
        pc1 = set(p % 12 for p in voices1)
        pc2 = set(p % 12 for p in voices2)
        common_tones = len(pc1 & pc2)

        # Calculate smoothness (0-1, higher is smoother)
        max_motion = max(abs(m) for m in voice_motions) if voice_motions else 0
        smoothness = 1.0 - (max_motion / 24.0)  # Normalize to 2 octaves

        # Detect voice leading violations
        violations = []
        for i in range(len(voices1) - 1):
            interval1 = (voices1[i + 1] - voices1[i]) % 12
            interval2 = (voices2[i + 1] - voices2[i]) % 12

            # Parallel fifths
            if interval1 == 7 and interval2 == 7 and voice_motions[i] == voice_motions[i + 1] and voice_motions[i] != 0:
                violations.append(f"parallel_fifths_voices_{i}_{i+1}")

            # Parallel octaves
            if interval1 == 0 and interval2 == 0 and voice_motions[i] == voice_motions[i + 1] and voice_motions[i] != 0:
                violations.append(f"parallel_octaves_voices_{i}_{i+1}")

        # Calculate quality score
        quality_score = (
            smoothness * 0.4 +
            (common_tones / len(voices1)) * 0.3 +
            (1.0 - len(violations) * 0.1) * 0.3
        )
        quality_score = max(0.0, min(1.0, quality_score))

        return VoiceLeadingAnalysis(
            chord1=chord1,
            chord2=chord2,
            voice_motions=voice_motions,
            motion_types=motion_types,
            total_motion=total_motion,
            smoothness=smoothness,
            common_tones=common_tones,
            violations=violations,
            quality_score=quality_score
        )

    def optimize_voice_leading(self, progression: ChordProgression) -> ChordProgression:
        """
        Optimize voice leading in a chord progression by rearranging voices.

        Args:
            progression: ChordProgression to optimize

        Returns:
            Optimized ChordProgression
        """
        if len(progression.chords) < 2:
            return progression

        optimized_chords = [progression.chords[0]]

        for i in range(1, len(progression.chords)):
            prev_chord = optimized_chords[-1]
            current_chord = progression.chords[i]

            # Try different voicings of current chord
            best_voicing = current_chord
            best_score = 0.0

            # Generate voicing variants by permuting pitches
            from itertools import permutations
            for perm in permutations(current_chord.pitches):
                test_chord = Chord(
                    pitches=list(perm),
                    root=current_chord.root,
                    quality=current_chord.quality,
                    start_time=current_chord.start_time,
                    duration=current_chord.duration
                )

                analysis = self.analyze_voice_leading(prev_chord, test_chord)
                if analysis.quality_score > best_score:
                    best_score = analysis.quality_score
                    best_voicing = test_chord

            optimized_chords.append(best_voicing)

        return ChordProgression(
            chords=optimized_chords,
            key=progression.key,
            mode=progression.mode
        )

    # ==========================================================================
    # REHARMONIZATION
    # ==========================================================================

    def reharmonize(self, melody: List[Note], style: str = 'jazz',
                   key: Optional[int] = None) -> ChordProgression:
        """
        Reharmonize a melody using various techniques.

        Args:
            melody: List of melody notes
            style: Reharmonization style ('jazz', 'classical', 'contemporary')
            key: Key center (0-11), will be detected if None

        Returns:
            ChordProgression with reharmonized chords
        """
        if not melody:
            return ChordProgression(chords=[])

        if key is None:
            key = self._detect_key(melody)

        # Group melody notes into phrases
        phrases = self._segment_melody(melody)

        chords = []
        for phrase in phrases:
            # Analyze phrase for important notes
            strong_beats = self._identify_strong_beats(phrase)

            # Create chord for each strong beat
            for note in strong_beats:
                # Determine appropriate chord based on style
                if style == 'jazz':
                    chord = self._jazz_reharmonization(note, key)
                elif style == 'classical':
                    chord = self._classical_reharmonization(note, key)
                else:
                    chord = self._contemporary_reharmonization(note, key)

                chord.start_time = note.start_time
                chord.duration = note.duration
                chords.append(chord)

        return ChordProgression(chords=chords, key=key)

    def _segment_melody(self, melody: List[Note], phrase_length: float = 8.0) -> List[List[Note]]:
        """Segment melody into phrases."""
        if not melody:
            return []

        phrases = []
        current_phrase = []
        phrase_start = melody[0].start_time

        for note in melody:
            if note.start_time - phrase_start >= phrase_length:
                if current_phrase:
                    phrases.append(current_phrase)
                current_phrase = [note]
                phrase_start = note.start_time
            else:
                current_phrase.append(note)

        if current_phrase:
            phrases.append(current_phrase)

        return phrases

    def _identify_strong_beats(self, notes: List[Note]) -> List[Note]:
        """Identify notes on strong beats (1 and 3 in 4/4)."""
        strong_notes = []
        for note in notes:
            beat = note.start_time % 4
            if beat < 0.5 or 1.5 < beat < 2.5:  # Beats 1 and 3
                strong_notes.append(note)

        # If no strong beats found, use all notes
        return strong_notes if strong_notes else notes

    def _jazz_reharmonization(self, note: Note, key: int) -> Chord:
        """Create jazz reharmonization for a melody note."""
        # Use melody note as chord extension (9th, 11th, 13th)
        # Choose root a third below melody note for basic harmony
        root = (note.pitch_class - 4) % 12  # Major third below

        # Create dominant 7th chord
        pitches = [
            root + 48,           # Root (low)
            (root + 4) % 12 + 60,  # Third
            (root + 7) % 12 + 60,  # Fifth
            (root + 10) % 12 + 60, # b7
        ]

        return self._analyze_chord(pitches)

    def _classical_reharmonization(self, note: Note, key: int) -> Chord:
        """Create classical reharmonization for a melody note."""
        # Use diatonic harmony
        scale_degree = (note.pitch_class - key) % 12

        # Map to diatonic triad
        root = key + scale_degree
        pitches = [
            root + 48,
            (root + 4) % 12 + 60,
            (root + 7) % 12 + 60,
        ]

        return self._analyze_chord(pitches)

    def _contemporary_reharmonization(self, note: Note, key: int) -> Chord:
        """Create contemporary reharmonization for a melody note."""
        # Use upper structure triads and quartal harmony
        root = note.pitch_class

        # Quartal voicing
        pitches = [
            root + 48,
            (root + 5) % 12 + 60,   # Fourth above
            (root + 10) % 12 + 60,  # Fourth above that
        ]

        return self._analyze_chord(pitches)

    # ==========================================================================
    # FEATURE EXTRACTION
    # ==========================================================================

    def extract_features(self, midi_path: Optional[Union[str, Path]] = None) -> HarmonyFeatures:
        """
        Extract comprehensive harmony features from loaded chords or MIDI file.

        Args:
            midi_path: Optional path to MIDI file

        Returns:
            HarmonyFeatures object with 50+ harmony parameters
        """
        if midi_path is not None:
            self.load_midi(midi_path)

        if not self.chords:
            return HarmonyFeatures()

        features = HarmonyFeatures()

        # Extract voicing features
        self._extract_voicing_features(features)

        # Extract modal features
        self._extract_modal_features(features)

        # Extract functional features
        self._extract_functional_features(features)

        # Extract voice leading features
        self._extract_voice_leading_features(features)

        self.features = features
        return features

    def _extract_voicing_features(self, features: HarmonyFeatures) -> None:
        """Extract voicing-related features."""
        voicing_counts = Counter()
        total_range = 0
        total_density = 0
        total = len(self.chords)

        for chord in self.chords:
            analysis = self.analyze_jazz_voicings(chord)
            voicing_counts[analysis.voicing_type] += 1
            total_range += analysis.range
            total_density += analysis.density

        # Normalize counts to ratios
        for voicing_type in VoicingType:
            features.voicing_types[voicing_type] = voicing_counts[voicing_type] / total if total > 0 else 0.0

        features.avg_voicing_density = total_density / total if total > 0 else 0.0
        features.avg_voicing_range = total_range / total if total > 0 else 0.0

        # Specific voicing ratios
        features.close_position_ratio = features.voicing_types.get(VoicingType.CLOSE_POSITION, 0.0)
        features.open_position_ratio = features.voicing_types.get(VoicingType.OPEN_POSITION, 0.0)
        features.drop2_usage = features.voicing_types.get(VoicingType.DROP_2, 0.0)
        features.drop3_usage = features.voicing_types.get(VoicingType.DROP_3, 0.0)
        features.rootless_usage = (
            features.voicing_types.get(VoicingType.ROOTLESS_A, 0.0) +
            features.voicing_types.get(VoicingType.ROOTLESS_B, 0.0)
        )
        features.shell_voicing_usage = features.voicing_types.get(VoicingType.SHELL, 0.0)
        features.fourths_usage = features.voicing_types.get(VoicingType.FOURTHS, 0.0)
        features.cluster_usage = features.voicing_types.get(VoicingType.CLUSTERS, 0.0)

    def _extract_modal_features(self, features: HarmonyFeatures) -> None:
        """Extract modal harmony features."""
        if self.mode:
            features.primary_mode = self.mode

        # Count characteristic intervals for each mode
        mode_scores = {mode: 0.0 for mode in Mode}

        # Analyze scale degree usage
        if self.key is not None:
            all_pitches = []
            for chord in self.chords:
                all_pitches.extend(chord.pitches)

            pitch_classes = [p % 12 for p in all_pitches]
            scale_degrees = Counter((pc - self.key) % 12 for pc in pitch_classes)

            # Score each mode based on scale degree usage
            for mode, template in self.MODE_TEMPLATES.items():
                score = sum(scale_degrees.get(degree, 0) for degree in template)
                mode_scores[mode] = score

            # Normalize scores
            total_score = sum(mode_scores.values())
            if total_score > 0:
                features.mode_distribution = {
                    mode: score / total_score for mode, score in mode_scores.items()
                }

        # Mode-specific characteristics
        features.dorian_characteristic = features.mode_distribution.get(Mode.DORIAN, 0.0)
        features.lydian_characteristic = features.mode_distribution.get(Mode.LYDIAN, 0.0)
        features.mixolydian_characteristic = features.mode_distribution.get(Mode.MIXOLYDIAN, 0.0)
        features.phrygian_characteristic = features.mode_distribution.get(Mode.PHRYGIAN, 0.0)
        features.locrian_characteristic = features.mode_distribution.get(Mode.LOCRIAN, 0.0)
        features.altered_scale_usage = features.mode_distribution.get(Mode.ALTERED, 0.0)
        features.whole_tone_usage = features.mode_distribution.get(Mode.WHOLE_TONE, 0.0)
        features.diminished_scale_usage = features.mode_distribution.get(Mode.DIMINISHED, 0.0)

    def _extract_functional_features(self, features: HarmonyFeatures) -> None:
        """Extract functional harmony features."""
        progression = ChordProgression(chords=self.chords, key=self.key, mode=self.mode)
        functions = self.analyze_functional_harmony(progression)

        function_counts = Counter(functions)
        total = len(functions)

        features.tonic_ratio = function_counts[HarmonicFunction.TONIC] / total if total > 0 else 0.0
        features.subdominant_ratio = function_counts[HarmonicFunction.SUBDOMINANT] / total if total > 0 else 0.0
        features.dominant_ratio = function_counts[HarmonicFunction.DOMINANT] / total if total > 0 else 0.0
        features.secondary_dominant_count = function_counts[HarmonicFunction.SECONDARY_DOMINANT]
        features.tritone_sub_usage = function_counts[HarmonicFunction.SUBSTITUTE_DOMINANT] / total if total > 0 else 0.0
        features.modal_interchange_ratio = function_counts[HarmonicFunction.BORROWED] / total if total > 0 else 0.0
        features.chromatic_mediant_usage = function_counts[HarmonicFunction.CHROMATIC_MEDIANT] / total if total > 0 else 0.0

        # Functional clarity: how well-defined the harmonic functions are
        # Higher when strong T-S-D patterns are present
        tsd_pattern_count = 0
        for i in range(len(functions) - 2):
            if (functions[i] == HarmonicFunction.TONIC and
                functions[i+1] in [HarmonicFunction.SUBDOMINANT, HarmonicFunction.PREDOMINANT] and
                functions[i+2] == HarmonicFunction.DOMINANT):
                tsd_pattern_count += 1

        features.functional_clarity = tsd_pattern_count / max(1, len(functions) - 2)

        # Cadence strength: presence of V-I or IV-I patterns
        cadence_count = 0
        for i in range(len(functions) - 1):
            if functions[i] in [HarmonicFunction.DOMINANT, HarmonicFunction.SUBDOMINANT] and \
               functions[i+1] == HarmonicFunction.TONIC:
                cadence_count += 1

        features.cadence_strength = cadence_count / max(1, len(functions) - 1)

    def _extract_voice_leading_features(self, features: HarmonyFeatures) -> None:
        """Extract voice leading features."""
        if len(self.chords) < 2:
            return

        total_motion = 0
        stepwise_count = 0
        leap_count = 0
        contrary_count = 0
        parallel_count = 0
        oblique_count = 0
        common_tone_sum = 0
        smoothness_sum = 0.0
        voice_crossing = 0
        parallel_fifths = 0
        parallel_octaves = 0

        total_transitions = len(self.chords) - 1

        for i in range(total_transitions):
            analysis = self.analyze_voice_leading(self.chords[i], self.chords[i+1])

            total_motion += analysis.total_motion
            common_tone_sum += analysis.common_tones
            smoothness_sum += analysis.smoothness

            # Count motion types
            for motion in analysis.voice_motions:
                if abs(motion) <= 2:
                    stepwise_count += 1
                elif abs(motion) > 2:
                    leap_count += 1

            # Count motion direction types
            contrary_count += analysis.motion_types.count("contrary")
            parallel_count += analysis.motion_types.count("parallel")
            oblique_count += analysis.motion_types.count("oblique")

            # Count violations
            parallel_fifths += sum(1 for v in analysis.violations if 'fifth' in v)
            parallel_octaves += sum(1 for v in analysis.violations if 'octave' in v)

        # Calculate ratios
        total_motions = stepwise_count + leap_count
        features.avg_voice_motion = total_motion / total_transitions if total_transitions > 0 else 0.0
        features.stepwise_motion_ratio = stepwise_count / total_motions if total_motions > 0 else 0.0
        features.leap_ratio = leap_count / total_motions if total_motions > 0 else 0.0

        total_motion_types = contrary_count + parallel_count + oblique_count
        features.contrary_motion_ratio = contrary_count / total_motion_types if total_motion_types > 0 else 0.0
        features.parallel_motion_ratio = parallel_count / total_motion_types if total_motion_types > 0 else 0.0
        features.oblique_motion_ratio = oblique_count / total_motion_types if total_motion_types > 0 else 0.0

        features.common_tone_retention = common_tone_sum / total_transitions if total_transitions > 0 else 0.0
        features.voice_leading_smoothness = smoothness_sum / total_transitions if total_transitions > 0 else 0.0
        features.voice_crossing_count = voice_crossing
        features.parallel_fifth_count = parallel_fifths
        features.parallel_octave_count = parallel_octaves

        # Voice independence: higher when voices move independently
        features.voice_independence = features.contrary_motion_ratio * 0.6 + features.oblique_motion_ratio * 0.4

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert harmony features to dictionary for XGBoost training.

        Returns:
            Dictionary of feature names to values
        """
        if self.features is None:
            self.extract_features()

        features_dict = {}

        # Voicing features (20 params)
        for voicing_type in VoicingType:
            features_dict[f'voicing_{voicing_type.value}'] = self.features.voicing_types.get(voicing_type, 0.0)

        features_dict['avg_voicing_density'] = self.features.avg_voicing_density
        features_dict['avg_voicing_range'] = self.features.avg_voicing_range
        features_dict['close_position_ratio'] = self.features.close_position_ratio
        features_dict['open_position_ratio'] = self.features.open_position_ratio
        features_dict['drop2_usage'] = self.features.drop2_usage
        features_dict['drop3_usage'] = self.features.drop3_usage
        features_dict['rootless_usage'] = self.features.rootless_usage
        features_dict['shell_voicing_usage'] = self.features.shell_voicing_usage
        features_dict['fourths_usage'] = self.features.fourths_usage
        features_dict['cluster_usage'] = self.features.cluster_usage

        # Modal harmony features (15 params)
        features_dict['primary_mode'] = self.features.primary_mode.value if self.features.primary_mode else 'unknown'
        for mode in Mode:
            features_dict[f'mode_{mode.value}'] = self.features.mode_distribution.get(mode, 0.0)

        features_dict['modal_mixture'] = self.features.modal_mixture
        features_dict['modal_interchange_events'] = self.features.modal_interchange_events
        features_dict['dorian_characteristic'] = self.features.dorian_characteristic
        features_dict['lydian_characteristic'] = self.features.lydian_characteristic
        features_dict['mixolydian_characteristic'] = self.features.mixolydian_characteristic
        features_dict['phrygian_characteristic'] = self.features.phrygian_characteristic
        features_dict['locrian_characteristic'] = self.features.locrian_characteristic
        features_dict['altered_scale_usage'] = self.features.altered_scale_usage
        features_dict['whole_tone_usage'] = self.features.whole_tone_usage
        features_dict['diminished_scale_usage'] = self.features.diminished_scale_usage

        # Functional harmony features (10 params)
        features_dict['tonic_ratio'] = self.features.tonic_ratio
        features_dict['subdominant_ratio'] = self.features.subdominant_ratio
        features_dict['dominant_ratio'] = self.features.dominant_ratio
        features_dict['secondary_dominant_count'] = self.features.secondary_dominant_count
        features_dict['tritone_sub_usage'] = self.features.tritone_sub_usage
        features_dict['modal_interchange_ratio'] = self.features.modal_interchange_ratio
        features_dict['chromatic_mediant_usage'] = self.features.chromatic_mediant_usage
        features_dict['functional_clarity'] = self.features.functional_clarity
        features_dict['cadence_strength'] = self.features.cadence_strength

        # Voice leading features (15 params)
        features_dict['avg_voice_motion'] = self.features.avg_voice_motion
        features_dict['stepwise_motion_ratio'] = self.features.stepwise_motion_ratio
        features_dict['leap_ratio'] = self.features.leap_ratio
        features_dict['contrary_motion_ratio'] = self.features.contrary_motion_ratio
        features_dict['parallel_motion_ratio'] = self.features.parallel_motion_ratio
        features_dict['oblique_motion_ratio'] = self.features.oblique_motion_ratio
        features_dict['common_tone_retention'] = self.features.common_tone_retention
        features_dict['voice_leading_smoothness'] = self.features.voice_leading_smoothness
        features_dict['voice_crossing_count'] = self.features.voice_crossing_count
        features_dict['parallel_fifth_count'] = self.features.parallel_fifth_count
        features_dict['parallel_octave_count'] = self.features.parallel_octave_count
        features_dict['voice_independence'] = self.features.voice_independence
        features_dict['leading_tone_resolution'] = self.features.leading_tone_resolution
        features_dict['seventh_resolution'] = self.features.seventh_resolution

        return features_dict

    def __repr__(self) -> str:
        return f"HarmonySpecialist(chords={len(self.chords)}, key={self.key}, mode={self.mode})"


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def analyze_harmony(midi_path: Union[str, Path]) -> HarmonyFeatures:
    """
    Convenience function to analyze harmony in a MIDI file.

    Args:
        midi_path: Path to MIDI file

    Returns:
        HarmonyFeatures object
    """
    specialist = HarmonySpecialist()
    return specialist.extract_features(midi_path)


def generate_jazz_voicing(root: int, quality: ChordQuality,
                          voicing_type: VoicingType = VoicingType.DROP_2) -> Chord:
    """
    Generate a jazz voicing for a chord.

    Args:
        root: Root pitch class (0-11)
        quality: Chord quality
        voicing_type: Type of voicing

    Returns:
        Chord object with voicing
    """
    specialist = HarmonySpecialist()

    # Get chord template
    if quality not in specialist.CHORD_TEMPLATES:
        quality = ChordQuality.MAJOR_7

    intervals = specialist.CHORD_TEMPLATES[quality]

    # Start in middle range
    octave = 60  # Middle C
    pitches = [root + octave + interval for interval in intervals]

    # Apply voicing transformation
    if voicing_type == VoicingType.DROP_2 and len(pitches) >= 4:
        # Drop second from top down an octave
        pitches[-2] -= 12

    elif voicing_type == VoicingType.DROP_3 and len(pitches) >= 4:
        # Drop third from top down an octave
        pitches[-3] -= 12

    elif voicing_type in [VoicingType.ROOTLESS_A, VoicingType.ROOTLESS_B]:
        # Remove root
        pitches = pitches[1:]

    elif voicing_type == VoicingType.SHELL:
        # Keep only root, 3rd, 7th
        if len(pitches) >= 4:
            pitches = [pitches[0], pitches[1], pitches[3]]

    chord = specialist._analyze_chord(sorted(pitches))
    return chord


if __name__ == "__main__":
    print("Harmony Specialist - Agent 18")
    print("=" * 50)
    print()
    print("Advanced harmony analysis and generation system")
    print("Features:")
    print("  - Jazz voicings (drop 2, drop 3, rootless)")
    print("  - Modal harmony (Dorian, Mixolydian, etc.)")
    print("  - Functional harmony analysis")
    print("  - Voice leading optimization")
    print("  - Reharmonization techniques")
    print("  - 50+ harmony parameters")
    print()
    print("See examples/agent18_harmony_demo.py for usage examples")
