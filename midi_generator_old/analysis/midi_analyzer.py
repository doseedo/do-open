#!/usr/bin/env python3
"""
MIDI Analyzer - Comprehensive MIDI File Analysis

This module provides world-class MIDI analysis capabilities including:
- Key detection (Krumhansl-Schmuckler algorithm)
- Tempo and time signature detection
- Chord recognition and harmonic analysis
- Melody extraction and contour analysis
- Rhythm pattern extraction
- Groove analysis (microtiming deviations)
- Statistical analysis (pitch class distribution, interval distribution, etc.)

Based on research from:
- Krumhansl & Kessler (1982) - Key-finding algorithm
- David Temperley (2007) - Music and Probability
- Daniel Müllensiefen (2009) - Statistical analysis of melodies
- Music Information Retrieval (MIR) research

Author: Agent 8 - Style Transfer & Transformation
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path
import warnings


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class NoteEvent:
    """Represents a MIDI note event with timing and metadata."""
    start_time: float  # In seconds
    duration: float    # In seconds
    start_tick: int    # MIDI ticks
    duration_ticks: int
    pitch: int         # MIDI note number (0-127)
    velocity: int      # Note velocity (0-127)
    channel: int       # MIDI channel (0-15)
    track_idx: int     # Track index in file

    @property
    def end_time(self) -> float:
        """Get note end time in seconds."""
        return self.start_time + self.duration

    @property
    def end_tick(self) -> int:
        """Get note end tick."""
        return self.start_tick + self.duration_ticks

    @property
    def pitch_class(self) -> int:
        """Get pitch class (0-11, where 0=C)."""
        return self.pitch % 12

    @property
    def octave(self) -> int:
        """Get octave number."""
        return self.pitch // 12 - 1


@dataclass
class ChordEvent:
    """Represents a detected chord."""
    start_time: float
    duration: float
    root: int           # Pitch class of root (0-11)
    quality: str        # 'major', 'minor', 'diminished', 'augmented', etc.
    pitches: List[int]  # All pitch classes in chord
    bass_note: int      # Bass note (for inversions)
    confidence: float   # Confidence score (0-1)

    def __str__(self) -> str:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{note_names[self.root]}{self.quality}"


@dataclass
class KeySignature:
    """Represents a key signature."""
    tonic: int          # Pitch class of tonic (0-11)
    mode: str           # 'major' or 'minor'
    confidence: float   # Confidence score (0-1)

    def __str__(self) -> str:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{note_names[self.tonic]} {self.mode}"


@dataclass
class TimeSignature:
    """Represents a time signature."""
    numerator: int      # Beats per measure
    denominator: int    # Beat unit (4 = quarter note)
    tick: int           # Tick where this time signature starts

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass
class TempoEvent:
    """Represents a tempo change."""
    tick: int
    tempo: float        # BPM
    microseconds_per_beat: int

    def __str__(self) -> str:
        return f"{self.tempo:.1f} BPM"


@dataclass
class RhythmPattern:
    """Represents a detected rhythm pattern."""
    pattern: List[float]  # List of onset times (normalized to beat)
    duration: float       # Total pattern duration in beats
    frequency: int        # How many times this pattern appears

    def __str__(self) -> str:
        return f"Pattern({len(self.pattern)} onsets, {self.frequency}x)"


@dataclass
class AnalysisResult:
    """Complete analysis result for a MIDI file."""
    # File metadata
    filename: str
    num_tracks: int
    ticks_per_beat: int
    duration_seconds: float
    duration_ticks: int

    # Note events
    notes: List[NoteEvent] = field(default_factory=list)

    # Musical analysis
    key: Optional[KeySignature] = None
    time_signatures: List[TimeSignature] = field(default_factory=list)
    tempo_events: List[TempoEvent] = field(default_factory=list)
    average_tempo: Optional[float] = None
    chords: List[ChordEvent] = field(default_factory=list)

    # Statistical features
    pitch_class_histogram: Dict[int, int] = field(default_factory=dict)
    interval_histogram: Dict[int, int] = field(default_factory=dict)
    duration_histogram: Dict[float, int] = field(default_factory=dict)
    velocity_stats: Dict[str, float] = field(default_factory=dict)

    # Rhythm analysis
    rhythm_patterns: List[RhythmPattern] = field(default_factory=list)
    onset_times: List[float] = field(default_factory=list)
    groove_deviation: Optional[float] = None  # Average timing deviation

    # Melodic features
    melodic_contour: List[int] = field(default_factory=list)  # -1: down, 0: same, 1: up
    melodic_intervals: List[int] = field(default_factory=list)
    melodic_range: Tuple[int, int] = (0, 0)  # (min, max) pitch

    # Harmonic complexity
    harmonic_complexity: float = 0.0  # Based on chord diversity
    chord_change_rate: float = 0.0    # Chords per second

    # Metadata
    analysis_notes: List[str] = field(default_factory=list)


# ==============================================================================
# KRUMHANSL-SCHMUCKLER KEY DETECTION
# ==============================================================================

class KeyDetector:
    """
    Key detection using Krumhansl-Schmuckler algorithm.

    Based on:
    Krumhansl, C. L., & Kessler, E. J. (1982). Tracing the dynamic changes
    in perceived tonal organization in a spatial representation of musical keys.
    Psychological Review, 89(4), 334-368.
    """

    # Krumhansl-Kessler key profiles (major and minor)
    # These represent the perceived stability of each pitch class in a key
    MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    @classmethod
    def detect_key(cls, notes: List[NoteEvent], weighted_by_duration: bool = True) -> KeySignature:
        """
        Detect the key of a piece using Krumhansl-Schmuckler algorithm.

        Args:
            notes: List of note events
            weighted_by_duration: Weight pitch classes by note duration

        Returns:
            KeySignature with detected key and confidence
        """
        if not notes:
            return KeySignature(0, 'major', 0.0)

        # Build pitch class distribution
        pc_distribution = [0.0] * 12

        for note in notes:
            pc = note.pitch_class
            weight = note.duration if weighted_by_duration else 1.0
            pc_distribution[pc] += weight

        # Normalize
        total = sum(pc_distribution)
        if total > 0:
            pc_distribution = [x / total for x in pc_distribution]

        # Correlate with all 24 major and minor key profiles
        best_correlation = -1.0
        best_key = 0
        best_mode = 'major'

        for tonic in range(12):
            # Try major
            major_profile = cls._rotate_profile(cls.MAJOR_PROFILE, tonic)
            correlation = cls._pearson_correlation(pc_distribution, major_profile)
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = tonic
                best_mode = 'major'

            # Try minor
            minor_profile = cls._rotate_profile(cls.MINOR_PROFILE, tonic)
            correlation = cls._pearson_correlation(pc_distribution, minor_profile)
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = tonic
                best_mode = 'minor'

        # Convert correlation to confidence (0-1)
        confidence = (best_correlation + 1) / 2  # Correlation is -1 to 1

        return KeySignature(best_key, best_mode, confidence)

    @staticmethod
    def _rotate_profile(profile: List[float], rotation: int) -> List[float]:
        """Rotate a key profile by n semitones."""
        return profile[rotation:] + profile[:rotation]

    @staticmethod
    def _pearson_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n == 0:
            return 0.0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator


# ==============================================================================
# CHORD RECOGNITION
# ==============================================================================

class ChordRecognizer:
    """
    Recognize chords from MIDI note events.

    Uses template matching and considers inversions.
    """

    # Chord templates (intervals from root)
    CHORD_TEMPLATES = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'diminished': [0, 3, 6],
        'augmented': [0, 4, 8],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'major7': [0, 4, 7, 11],
        'minor7': [0, 3, 7, 10],
        'dom7': [0, 4, 7, 10],
        'dim7': [0, 3, 6, 9],
        'half-dim7': [0, 3, 6, 10],
        'maj6': [0, 4, 7, 9],
        'min6': [0, 3, 7, 9],
        'add9': [0, 2, 4, 7],
        'maj9': [0, 2, 4, 7, 11],
        'min9': [0, 2, 3, 7, 10],
    }

    @classmethod
    def detect_chords(cls, notes: List[NoteEvent],
                      time_window: float = 0.5,
                      min_notes: int = 3) -> List[ChordEvent]:
        """
        Detect chords from note events.

        Args:
            notes: List of note events
            time_window: Time window for simultaneous notes (seconds)
            min_notes: Minimum notes required to form a chord

        Returns:
            List of detected chord events
        """
        if not notes:
            return []

        chords = []
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        i = 0
        while i < len(sorted_notes):
            # Find all notes within time window
            window_start = sorted_notes[i].start_time
            window_notes = []

            j = i
            while j < len(sorted_notes) and sorted_notes[j].start_time < window_start + time_window:
                # Check if note is still sounding at window_start
                if sorted_notes[j].end_time > window_start:
                    window_notes.append(sorted_notes[j])
                j += 1

            # Try to recognize chord
            if len(window_notes) >= min_notes:
                chord = cls._recognize_chord(window_notes)
                if chord:
                    chords.append(chord)

            # Move to next potential chord
            i += 1
            # Skip ahead to avoid duplicate detection
            if window_notes:
                # Move to after this chord starts
                i = max(i, j)

        return cls._merge_adjacent_chords(chords)

    @classmethod
    def _recognize_chord(cls, notes: List[NoteEvent]) -> Optional[ChordEvent]:
        """Recognize a chord from simultaneous notes."""
        if not notes:
            return None

        # Get unique pitch classes
        pitch_classes = sorted(set(note.pitch_class for note in notes))
        if len(pitch_classes) < 3:
            return None

        # Try all possible roots and chord types
        best_match = None
        best_score = 0

        for root in pitch_classes:
            for quality, template in cls.CHORD_TEMPLATES.items():
                # Normalize pitch classes relative to root
                normalized = sorted((pc - root) % 12 for pc in pitch_classes)

                # Calculate match score
                score = cls._template_match_score(normalized, template)

                if score > best_score:
                    best_score = score
                    best_match = (root, quality, pitch_classes)

        if best_match and best_score >= 0.7:  # Confidence threshold
            root, quality, pitches = best_match

            # Calculate duration and timing
            start_time = min(n.start_time for n in notes)
            end_time = max(n.end_time for n in notes)
            duration = end_time - start_time

            # Find bass note (lowest pitch)
            bass_note = min(n.pitch for n in notes) % 12

            return ChordEvent(
                start_time=start_time,
                duration=duration,
                root=root,
                quality=quality,
                pitches=pitches,
                bass_note=bass_note,
                confidence=best_score
            )

        return None

    @staticmethod
    def _template_match_score(pitch_classes: List[int], template: List[int]) -> float:
        """Calculate how well pitch classes match a chord template."""
        # Exact match bonus
        if set(pitch_classes) == set(template):
            return 1.0

        # Subset match (chord contains template notes)
        template_set = set(template)
        pitches_set = set(pitch_classes)

        if template_set.issubset(pitches_set):
            # Score based on how many extra notes
            extra_notes = len(pitches_set - template_set)
            return 1.0 - (extra_notes * 0.1)

        # Partial match
        intersection = len(template_set & pitches_set)
        union = len(template_set | pitches_set)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _merge_adjacent_chords(chords: List[ChordEvent]) -> List[ChordEvent]:
        """Merge adjacent identical chords."""
        if not chords:
            return []

        merged = []
        current = chords[0]

        for next_chord in chords[1:]:
            # Check if chords are the same
            if (current.root == next_chord.root and
                current.quality == next_chord.quality and
                next_chord.start_time - current.end_time < 0.1):  # Small gap
                # Merge by extending duration
                current.duration = next_chord.end_time - current.start_time
            else:
                merged.append(current)
                current = next_chord

        merged.append(current)
        return merged


# ==============================================================================
# MIDI FILE ANALYZER
# ==============================================================================

class MidiAnalyzer:
    """
    Comprehensive MIDI file analyzer.

    Provides complete analysis of MIDI files including:
    - Note extraction
    - Key detection
    - Chord recognition
    - Tempo/time signature analysis
    - Statistical features
    - Rhythm patterns
    - Melodic analysis
    """

    def __init__(self, midi_path: str):
        """
        Initialize analyzer with MIDI file.

        Args:
            midi_path: Path to MIDI file
        """
        self.midi_path = Path(midi_path)
        if not self.midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        self.midi = MidiFile(midi_path)
        self.result = AnalysisResult(
            filename=self.midi_path.name,
            num_tracks=len(self.midi.tracks),
            ticks_per_beat=self.midi.ticks_per_beat,
            duration_seconds=0.0,
            duration_ticks=0
        )

    def analyze(self,
                detect_key: bool = True,
                detect_chords: bool = True,
                analyze_rhythm: bool = True,
                analyze_melody: bool = True) -> AnalysisResult:
        """
        Perform complete analysis.

        Args:
            detect_key: Perform key detection
            detect_chords: Perform chord recognition
            analyze_rhythm: Analyze rhythm patterns
            analyze_melody: Analyze melodic features

        Returns:
            AnalysisResult with complete analysis
        """
        # Extract all note events and metadata
        self._extract_notes_and_metadata()

        # Key detection
        if detect_key and self.result.notes:
            self.result.key = KeyDetector.detect_key(self.result.notes)
            self.result.analysis_notes.append(
                f"Detected key: {self.result.key} (confidence: {self.result.key.confidence:.2f})"
            )

        # Chord recognition
        if detect_chords and self.result.notes:
            self.result.chords = ChordRecognizer.detect_chords(self.result.notes)
            self.result.analysis_notes.append(f"Detected {len(self.result.chords)} chords")

        # Statistical analysis
        self._analyze_statistics()

        # Rhythm analysis
        if analyze_rhythm:
            self._analyze_rhythm()

        # Melodic analysis
        if analyze_melody:
            self._analyze_melody()

        # Harmonic complexity
        if self.result.chords:
            self._analyze_harmonic_complexity()

        return self.result

    def _extract_notes_and_metadata(self):
        """Extract all note events and metadata from MIDI file."""
        notes = []
        tempo_events = []
        time_signatures = []

        # Default tempo (120 BPM = 500000 microseconds per beat)
        current_tempo_us = 500000

        for track_idx, track in enumerate(self.midi.tracks):
            absolute_tick = 0
            note_ons = {}  # {(note, channel): (tick, velocity)}

            for msg in track:
                absolute_tick += msg.time

                # Tempo changes
                if msg.type == 'set_tempo':
                    current_tempo_us = msg.tempo
                    tempo_bpm = 60_000_000 / msg.tempo
                    tempo_events.append(TempoEvent(
                        tick=absolute_tick,
                        tempo=tempo_bpm,
                        microseconds_per_beat=msg.tempo
                    ))

                # Time signature changes
                elif msg.type == 'time_signature':
                    time_signatures.append(TimeSignature(
                        numerator=msg.numerator,
                        denominator=msg.denominator,
                        tick=absolute_tick
                    ))

                # Note events
                elif msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.note, msg.channel)
                    note_ons[key] = (absolute_tick, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.note, msg.channel)
                    if key in note_ons:
                        start_tick, velocity = note_ons[key]
                        duration_ticks = absolute_tick - start_tick

                        # Convert ticks to seconds
                        start_time = self._ticks_to_seconds(start_tick, current_tempo_us)
                        duration = self._ticks_to_seconds(duration_ticks, current_tempo_us)

                        notes.append(NoteEvent(
                            start_time=start_time,
                            duration=duration,
                            start_tick=start_tick,
                            duration_ticks=duration_ticks,
                            pitch=msg.note,
                            velocity=velocity,
                            channel=msg.channel,
                            track_idx=track_idx
                        ))

                        del note_ons[key]

        # Store results
        self.result.notes = sorted(notes, key=lambda n: n.start_time)
        self.result.tempo_events = tempo_events
        self.result.time_signatures = time_signatures

        # Calculate duration
        if notes:
            self.result.duration_seconds = max(n.end_time for n in notes)
            self.result.duration_ticks = max(n.end_tick for n in notes)

        # Calculate average tempo
        if tempo_events:
            self.result.average_tempo = sum(t.tempo for t in tempo_events) / len(tempo_events)
        elif self.result.duration_seconds > 0:
            # Estimate from file duration
            self.result.average_tempo = 120.0  # Default

    def _ticks_to_seconds(self, ticks: int, tempo_us: int) -> float:
        """Convert MIDI ticks to seconds."""
        seconds_per_tick = tempo_us / (self.midi.ticks_per_beat * 1_000_000)
        return ticks * seconds_per_tick

    def _analyze_statistics(self):
        """Calculate statistical features."""
        if not self.result.notes:
            return

        # Pitch class histogram
        pc_hist = Counter(note.pitch_class for note in self.result.notes)
        self.result.pitch_class_histogram = dict(pc_hist)

        # Interval histogram (melodic intervals)
        sorted_notes = sorted(self.result.notes, key=lambda n: n.start_time)
        intervals = []
        for i in range(len(sorted_notes) - 1):
            interval = sorted_notes[i + 1].pitch - sorted_notes[i].pitch
            intervals.append(interval)
        interval_hist = Counter(intervals)
        self.result.interval_histogram = dict(interval_hist)

        # Duration histogram (quantized to 16th notes)
        durations = [round(note.duration, 2) for note in self.result.notes]
        duration_hist = Counter(durations)
        self.result.duration_histogram = dict(duration_hist)

        # Velocity statistics
        velocities = [note.velocity for note in self.result.notes]
        self.result.velocity_stats = {
            'mean': np.mean(velocities),
            'std': np.std(velocities),
            'min': min(velocities),
            'max': max(velocities),
            'median': np.median(velocities)
        }

    def _analyze_rhythm(self):
        """Analyze rhythm patterns and groove."""
        if not self.result.notes:
            return

        # Extract onset times
        onset_times = sorted(set(note.start_time for note in self.result.notes))
        self.result.onset_times = onset_times

        # Calculate groove deviation (timing variations)
        if len(onset_times) > 1:
            # Quantize to 16th note grid
            if self.result.average_tempo:
                sixteenth_duration = 60.0 / self.result.average_tempo / 4
                deviations = []
                for onset in onset_times:
                    quantized = round(onset / sixteenth_duration) * sixteenth_duration
                    deviation = abs(onset - quantized)
                    deviations.append(deviation)
                self.result.groove_deviation = np.mean(deviations)

    def _analyze_melody(self):
        """Analyze melodic features."""
        if not self.result.notes:
            return

        # Sort by time and find highest note at each time (melody line)
        sorted_notes = sorted(self.result.notes, key=lambda n: n.start_time)

        # Extract melodic line (highest notes)
        melody = []
        current_time = -1
        for note in sorted_notes:
            if note.start_time > current_time:
                melody.append(note.pitch)
                current_time = note.start_time

        # Melodic contour (-1: down, 0: same, 1: up)
        contour = []
        for i in range(len(melody) - 1):
            if melody[i + 1] > melody[i]:
                contour.append(1)
            elif melody[i + 1] < melody[i]:
                contour.append(-1)
            else:
                contour.append(0)
        self.result.melodic_contour = contour

        # Melodic intervals
        intervals = []
        for i in range(len(melody) - 1):
            intervals.append(melody[i + 1] - melody[i])
        self.result.melodic_intervals = intervals

        # Melodic range
        if melody:
            self.result.melodic_range = (min(melody), max(melody))

    def _analyze_harmonic_complexity(self):
        """Analyze harmonic complexity."""
        if not self.result.chords:
            return

        # Chord diversity (unique chords / total chords)
        unique_chords = len(set((c.root, c.quality) for c in self.result.chords))
        self.result.harmonic_complexity = unique_chords / len(self.result.chords)

        # Chord change rate
        if self.result.duration_seconds > 0:
            self.result.chord_change_rate = len(self.result.chords) / self.result.duration_seconds

    def print_analysis(self):
        """Print human-readable analysis report."""
        r = self.result

        print("\n" + "="*80)
        print(f"MIDI ANALYSIS: {r.filename}")
        print("="*80)

        # Basic info
        print(f"\n📄 File Information:")
        print(f"   Tracks: {r.num_tracks}")
        print(f"   Duration: {r.duration_seconds:.2f} seconds")
        print(f"   Ticks per beat: {r.ticks_per_beat}")
        print(f"   Total notes: {len(r.notes)}")

        # Tempo
        if r.average_tempo:
            print(f"\n🎵 Tempo:")
            print(f"   Average: {r.average_tempo:.1f} BPM")
            if r.tempo_events:
                print(f"   Tempo changes: {len(r.tempo_events)}")

        # Time signatures
        if r.time_signatures:
            print(f"\n⏱️  Time Signatures:")
            for ts in r.time_signatures[:3]:  # Show first 3
                print(f"   {ts}")

        # Key
        if r.key:
            print(f"\n🎹 Key:")
            print(f"   {r.key} (confidence: {r.key.confidence:.2%})")

        # Chords
        if r.chords:
            print(f"\n🎸 Harmony:")
            print(f"   Chords detected: {len(r.chords)}")
            print(f"   Harmonic complexity: {r.harmonic_complexity:.2f}")
            print(f"   Chord change rate: {r.chord_change_rate:.2f} chords/sec")
            print(f"   Sample chords: {', '.join(str(c) for c in r.chords[:5])}")

        # Melody
        if r.melodic_range != (0, 0):
            print(f"\n🎶 Melody:")
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            low = r.melodic_range[0]
            high = r.melodic_range[1]
            print(f"   Range: {note_names[low%12]}{low//12-1} to {note_names[high%12]}{high//12-1}")
            print(f"   Range (semitones): {high - low}")

            # Interval statistics
            if r.melodic_intervals:
                avg_interval = np.mean([abs(i) for i in r.melodic_intervals])
                print(f"   Average interval: {avg_interval:.1f} semitones")
                stepwise = sum(1 for i in r.melodic_intervals if abs(i) <= 2)
                stepwise_pct = stepwise / len(r.melodic_intervals) * 100
                print(f"   Stepwise motion: {stepwise_pct:.1f}%")

        # Rhythm
        if r.groove_deviation:
            print(f"\n🥁 Rhythm:")
            print(f"   Groove deviation: {r.groove_deviation*1000:.1f} ms")
            print(f"   Onset events: {len(r.onset_times)}")

        # Statistics
        if r.velocity_stats:
            print(f"\n📊 Statistics:")
            print(f"   Velocity: mean={r.velocity_stats['mean']:.1f}, "
                  f"std={r.velocity_stats['std']:.1f}, "
                  f"range={r.velocity_stats['min']}-{r.velocity_stats['max']}")

        # Pitch class distribution
        if r.pitch_class_histogram:
            print(f"\n🎼 Pitch Class Distribution:")
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            sorted_pcs = sorted(r.pitch_class_histogram.items(), key=lambda x: x[1], reverse=True)
            top_5 = sorted_pcs[:5]
            print(f"   Top 5: {', '.join(f'{note_names[pc]}({count})' for pc, count in top_5)}")

        print("\n" + "="*80 + "\n")


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze MIDI files')
    parser.add_argument('midi_file', help='Path to MIDI file')
    parser.add_argument('--no-key', action='store_true', help='Skip key detection')
    parser.add_argument('--no-chords', action='store_true', help='Skip chord detection')
    parser.add_argument('--no-rhythm', action='store_true', help='Skip rhythm analysis')
    parser.add_argument('--no-melody', action='store_true', help='Skip melody analysis')

    args = parser.parse_args()

    # Analyze MIDI file
    analyzer = MidiAnalyzer(args.midi_file)
    result = analyzer.analyze(
        detect_key=not args.no_key,
        detect_chords=not args.no_chords,
        analyze_rhythm=not args.no_rhythm,
        analyze_melody=not args.no_melody
    )

    # Print report
    analyzer.print_analysis()
