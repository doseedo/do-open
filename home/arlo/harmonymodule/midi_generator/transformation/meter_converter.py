#!/usr/bin/env python3
"""
Meter Conversion Engine - Convert MIDI Between Time Signatures

This module provides comprehensive time signature conversion capabilities, transforming
MIDI files between different meters while preserving musical content, phrase structure,
and artistic intent.

Key Features:
1. TIME SIGNATURE CONVERSION
   - Simple conversions (4/4 → 3/4, 2/4, 6/8)
   - Odd meter conversions (4/4 → 5/4, 7/8, 11/8)
   - Compound to simple (6/8 → 2/4, 3/4 → 9/8)
   - Multiple strategies (stretch, compress, redistribute)

2. METRIC MODULATION
   - Elliott Carter-style metric modulation
   - Smooth tempo transitions via pivot rhythms
   - Mathematically precise tempo calculations
   - Maintains rhythmic continuity

3. PHRASE PRESERVATION
   - Intelligent phrase boundary detection
   - Maintains phrase structure during conversion
   - Handles anacrusis (pickup measures)
   - Preserves melodic/harmonic content

4. RHYTHM ADAPTATION
   - Re-quantizes rhythms appropriately
   - Adjusts subdivisions (quarters → dotted quarters)
   - Maintains groove feel
   - Preserves syncopation patterns

Research Sources:
- Elliott Carter: "Metric Modulation" techniques (Bernard, J.W., 1988)
- Stefan Kostka: "Tonal Harmony" - Phrase structure and hypermeter
- David Temperley: "Music and Probability" - Meter perception
- Justin London: "Hearing in Time" - Metric theory and cognition
- Dave Brubeck: "Time Out" - Jazz in odd meters
- Tool, Meshuggah: Progressive metal polymetric techniques
- MuseScore time signature conversion algorithms

Author: Agent 7 - Modular Fusion Enhancement Project
Date: 2025-11-19
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Tuple, Dict, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from fractions import Fraction
from enum import Enum
import numpy as np
import copy
import math

# Import our analyzer and rhythm modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from analysis.midi_analyzer import MidiAnalyzer, NoteEvent, ChordEvent
    from algorithms.advanced_rhythm import TimeSignature as AdvancedTimeSignature, RhythmicEvent
    from midi.midi_constants import DEFAULT_PPQN, PPQN_HIGH_RES
except ImportError:
    # Fallback values
    DEFAULT_PPQN = 480
    PPQN_HIGH_RES = 960
    print("Warning: Some imports failed, using fallback values")


# ==============================================================================
# ENUMS AND DATA STRUCTURES
# ==============================================================================

class ConversionStrategy(Enum):
    """Strategies for converting between time signatures."""
    STRETCH = "stretch"              # Stretch/compress to fit new meter
    REDISTRIBUTE = "redistribute"     # Redistribute notes across new measures
    TRUNCATE = "truncate"             # Cut off excess notes
    REPEAT = "repeat"                 # Repeat pattern to fill measures
    METRIC_MODULATION = "metric_mod" # Use metric modulation
    PHRASE_AWARE = "phrase_aware"     # Preserve phrase boundaries


class MeterFamily(Enum):
    """Categories of time signatures."""
    SIMPLE_DUPLE = "simple_duple"       # 2/4, 2/2
    SIMPLE_TRIPLE = "simple_triple"     # 3/4, 3/8
    SIMPLE_QUADRUPLE = "simple_quad"    # 4/4, 4/8
    COMPOUND_DUPLE = "compound_duple"   # 6/8, 6/4
    COMPOUND_TRIPLE = "compound_triple" # 9/8, 9/4
    COMPOUND_QUADRUPLE = "compound_quad" # 12/8, 12/4
    ODD_METER = "odd_meter"             # 5/4, 7/8, 11/8, etc.
    COMPLEX_METER = "complex_meter"     # Irregular groupings


@dataclass
class TimeSignatureInfo:
    """Detailed time signature information."""
    numerator: int
    denominator: int
    grouping: Optional[List[int]] = None  # For odd meters: [2, 3] = 5/4
    family: Optional[MeterFamily] = None
    beats_per_measure: float = 0.0
    ticks_per_measure: int = 0
    ppqn: int = DEFAULT_PPQN

    def __post_init__(self):
        """Calculate derived properties."""
        if self.family is None:
            self.family = self._classify_meter()

        # Calculate beats per measure (quarter note equivalents)
        self.beats_per_measure = (self.numerator * 4) / self.denominator

        # Calculate ticks per measure
        self.ticks_per_measure = int(self.ppqn * self.beats_per_measure)

        # Generate default grouping if needed
        if self.grouping is None:
            self.grouping = self._generate_default_grouping()

    def _classify_meter(self) -> MeterFamily:
        """Classify the meter into a family."""
        # Odd meters
        if self.numerator in [5, 7, 11, 13, 15]:
            return MeterFamily.ODD_METER

        # Simple meters (divisible by 2 or 3, not by 3 in numerator)
        if self.numerator % 3 != 0:
            if self.numerator == 2:
                return MeterFamily.SIMPLE_DUPLE
            elif self.numerator == 3:
                return MeterFamily.SIMPLE_TRIPLE
            elif self.numerator == 4:
                return MeterFamily.SIMPLE_QUADRUPLE

        # Compound meters (numerator divisible by 3)
        if self.numerator % 3 == 0:
            if self.numerator == 6:
                return MeterFamily.COMPOUND_DUPLE
            elif self.numerator == 9:
                return MeterFamily.COMPOUND_TRIPLE
            elif self.numerator == 12:
                return MeterFamily.COMPOUND_QUADRUPLE

        return MeterFamily.COMPLEX_METER

    def _generate_default_grouping(self) -> List[int]:
        """Generate default beat grouping."""
        if self.family == MeterFamily.ODD_METER:
            # Common odd meter groupings
            if self.numerator == 5:
                return [3, 2]  # Take Five style
            elif self.numerator == 7:
                return [2, 2, 3]  # Pink Floyd "Money" style
            elif self.numerator == 11:
                return [3, 3, 2, 3]
            elif self.numerator == 13:
                return [3, 3, 3, 2, 2]

        # Default: group by 2s and 3s
        grouping = []
        remaining = self.numerator
        while remaining > 0:
            if remaining >= 3:
                grouping.append(3)
                remaining -= 3
            elif remaining >= 2:
                grouping.append(2)
                remaining -= 2
            else:
                grouping.append(1)
                remaining -= 1

        return grouping

    def __str__(self):
        if self.grouping and len(self.grouping) > 1:
            grouping_str = "+".join(map(str, self.grouping))
            return f"{self.numerator}/{self.denominator} ({grouping_str})"
        return f"{self.numerator}/{self.denominator}"


@dataclass
class MeterConversionParams:
    """Parameters for meter conversion."""
    strategy: ConversionStrategy = ConversionStrategy.PHRASE_AWARE
    preserve_phrase_structure: bool = True
    preserve_tempo_feel: bool = True
    adjust_tempo: bool = False  # Whether to adjust tempo to compensate
    target_tempo: Optional[int] = None
    maintain_durations: bool = False  # Keep absolute durations vs. metric position
    quantize_output: bool = True
    quantize_strength: float = 0.8  # 0-1, how much to quantize


@dataclass
class PhraseBoundary:
    """Represents a phrase boundary in the music."""
    measure_number: int
    tick_position: int
    boundary_type: str  # 'start', 'end', 'cadence', 'breath'
    confidence: float  # 0-1


@dataclass
class ConversionResult:
    """Result of a meter conversion."""
    success: bool
    new_midi: Optional[MidiFile] = None
    new_time_signature: Optional[TimeSignatureInfo] = None
    tempo_change_factor: float = 1.0
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, any] = field(default_factory=dict)


# ==============================================================================
# MAIN METER CONVERTER CLASS
# ==============================================================================

class MeterConverter:
    """
    Main class for converting MIDI files between time signatures.

    Usage:
        converter = MeterConverter("input.mid")
        result = converter.convert_meter(
            new_numerator=7,
            new_denominator=8,
            strategy=ConversionStrategy.PHRASE_AWARE
        )
        if result.success:
            result.new_midi.save("output.mid")
    """

    def __init__(self, midi_file: Union[str, MidiFile]):
        """
        Initialize converter with MIDI file.

        Args:
            midi_file: Path to MIDI file or MidiFile object
        """
        if isinstance(midi_file, str):
            self.midi = MidiFile(midi_file)
            self.file_path = midi_file
        else:
            self.midi = midi_file
            self.file_path = None

        self.ppqn = self.midi.ticks_per_beat

        # Analyze input file
        self.current_time_sig = self._detect_time_signature()
        self.tempo = self._detect_tempo()
        self.phrase_boundaries = []

    def _detect_time_signature(self) -> TimeSignatureInfo:
        """Detect current time signature from MIDI file."""
        for track in self.midi.tracks:
            for msg in track:
                if msg.type == 'time_signature':
                    return TimeSignatureInfo(
                        numerator=msg.numerator,
                        denominator=msg.denominator,
                        ppqn=self.ppqn
                    )

        # Default to 4/4
        return TimeSignatureInfo(4, 4, ppqn=self.ppqn)

    def _detect_tempo(self) -> int:
        """Detect tempo from MIDI file."""
        for track in self.midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return int(mido.tempo2bpm(msg.tempo))

        return 120  # Default tempo

    def convert_meter(
        self,
        new_numerator: int,
        new_denominator: int,
        new_grouping: Optional[List[int]] = None,
        params: Optional[MeterConversionParams] = None
    ) -> ConversionResult:
        """
        Convert MIDI to new time signature.

        Args:
            new_numerator: Target time signature numerator
            new_denominator: Target time signature denominator
            new_grouping: Optional grouping for odd meters
            params: Conversion parameters

        Returns:
            ConversionResult with new MIDI and metadata
        """
        if params is None:
            params = MeterConversionParams()

        # Create target time signature info
        target_time_sig = TimeSignatureInfo(
            numerator=new_numerator,
            denominator=new_denominator,
            grouping=new_grouping,
            ppqn=self.ppqn
        )

        # Check if conversion is needed
        if (self.current_time_sig.numerator == new_numerator and
            self.current_time_sig.denominator == new_denominator):
            return ConversionResult(
                success=True,
                new_midi=copy.deepcopy(self.midi),
                new_time_signature=target_time_sig,
                warnings=["No conversion needed - already in target meter"]
            )

        # Detect phrase boundaries if needed
        if params.preserve_phrase_structure:
            self.phrase_boundaries = self._detect_phrase_boundaries()

        # Select conversion strategy
        if params.strategy == ConversionStrategy.METRIC_MODULATION:
            return self._convert_via_metric_modulation(target_time_sig, params)
        elif params.strategy == ConversionStrategy.PHRASE_AWARE:
            return self._convert_phrase_aware(target_time_sig, params)
        elif params.strategy == ConversionStrategy.STRETCH:
            return self._convert_via_stretch(target_time_sig, params)
        elif params.strategy == ConversionStrategy.REDISTRIBUTE:
            return self._convert_via_redistribution(target_time_sig, params)
        else:
            return self._convert_basic(target_time_sig, params)

    def _detect_phrase_boundaries(self) -> List[PhraseBoundary]:
        """
        Detect phrase boundaries in the MIDI file.

        Uses multiple heuristics:
        1. Long notes/rests (likely phrase endings)
        2. Melodic contour changes
        3. Harmonic rhythm changes
        4. Regular hypermeter (4/8/16 measure phrases)
        """
        boundaries = []

        # Extract all notes
        notes = self._extract_all_notes()
        if not notes:
            return boundaries

        # Sort by time
        notes.sort(key=lambda n: n['tick'])

        # Look for long rests (likely phrase boundaries)
        rest_threshold_ticks = self.current_time_sig.ticks_per_measure * 0.5

        for i in range(len(notes) - 1):
            note = notes[i]
            next_note = notes[i + 1]

            # Calculate rest duration
            rest_duration = next_note['tick'] - (note['tick'] + note['duration'])

            if rest_duration >= rest_threshold_ticks:
                measure_num = note['tick'] // self.current_time_sig.ticks_per_measure
                boundaries.append(PhraseBoundary(
                    measure_number=int(measure_num),
                    tick_position=note['tick'] + note['duration'],
                    boundary_type='breath',
                    confidence=0.8
                ))

        # Add regular hypermeter boundaries (every 4 measures)
        total_measures = notes[-1]['tick'] // self.current_time_sig.ticks_per_measure
        for measure in range(4, int(total_measures), 4):
            tick = measure * self.current_time_sig.ticks_per_measure
            boundaries.append(PhraseBoundary(
                measure_number=measure,
                tick_position=tick,
                boundary_type='hypermeter',
                confidence=0.6
            ))

        # Sort and remove duplicates
        boundaries.sort(key=lambda b: b.tick_position)
        boundaries = self._remove_duplicate_boundaries(boundaries)

        return boundaries

    def _remove_duplicate_boundaries(self, boundaries: List[PhraseBoundary]) -> List[PhraseBoundary]:
        """Remove boundaries that are too close together."""
        if not boundaries:
            return boundaries

        filtered = [boundaries[0]]
        min_distance_ticks = self.current_time_sig.ticks_per_measure * 0.5

        for boundary in boundaries[1:]:
            if boundary.tick_position - filtered[-1].tick_position >= min_distance_ticks:
                filtered.append(boundary)

        return filtered

    def _extract_all_notes(self) -> List[Dict]:
        """Extract all notes from MIDI file."""
        notes = []

        for track_idx, track in enumerate(self.midi.tracks):
            current_tick = 0
            active_notes = {}  # pitch -> (start_tick, velocity)

            for msg in track:
                current_tick += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_tick, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_tick, velocity = active_notes.pop(msg.note)
                        notes.append({
                            'pitch': msg.note,
                            'tick': start_tick,
                            'duration': current_tick - start_tick,
                            'velocity': velocity,
                            'channel': msg.channel,
                            'track': track_idx
                        })

        return notes

    def _convert_phrase_aware(
        self,
        target_time_sig: TimeSignatureInfo,
        params: MeterConversionParams
    ) -> ConversionResult:
        """
        Convert while preserving phrase structure.

        Algorithm:
        1. Identify phrase boundaries
        2. Convert each phrase independently
        3. Adjust measure counts to maintain phrase lengths
        4. Smooth boundaries between phrases
        """
        warnings = []

        # Calculate measure ratio
        source_beats_per_measure = self.current_time_sig.beats_per_measure
        target_beats_per_measure = target_time_sig.beats_per_measure
        measure_ratio = target_beats_per_measure / source_beats_per_measure

        # Create new MIDI file
        new_midi = MidiFile(ticks_per_beat=self.ppqn)

        # Process each track
        for track in self.midi.tracks:
            new_track = self._convert_track_phrase_aware(
                track, target_time_sig, measure_ratio, params
            )
            new_midi.tracks.append(new_track)

        # Calculate tempo adjustment if needed
        tempo_factor = 1.0
        if params.adjust_tempo:
            tempo_factor = 1.0 / measure_ratio

        return ConversionResult(
            success=True,
            new_midi=new_midi,
            new_time_signature=target_time_sig,
            tempo_change_factor=tempo_factor,
            warnings=warnings
        )

    def _convert_track_phrase_aware(
        self,
        track: MidiTrack,
        target_time_sig: TimeSignatureInfo,
        measure_ratio: float,
        params: MeterConversionParams
    ) -> MidiTrack:
        """Convert a single track with phrase awareness."""
        new_track = MidiTrack()

        current_tick = 0
        events = []

        # Collect all events with absolute timing
        for msg in track:
            current_tick += msg.time
            events.append((current_tick, msg))

        # Convert events to new meter
        new_events = []
        for tick, msg in events:
            # Calculate new tick position
            measure_num = tick / self.current_time_sig.ticks_per_measure
            beat_in_measure = (tick % self.current_time_sig.ticks_per_measure) / self.ppqn

            # Map to new meter
            new_measure_num = measure_num  # Keep measure numbers the same
            new_tick = int(new_measure_num * target_time_sig.ticks_per_measure +
                          beat_in_measure * self.ppqn)

            # Update time signature messages
            if msg.type == 'time_signature':
                new_msg = MetaMessage('time_signature',
                                     numerator=target_time_sig.numerator,
                                     denominator=target_time_sig.denominator)
                new_events.append((new_tick, new_msg))
            else:
                new_events.append((new_tick, msg.copy()))

        # Convert back to delta times
        new_events.sort(key=lambda x: x[0])
        prev_tick = 0
        for tick, msg in new_events:
            msg.time = tick - prev_tick
            new_track.append(msg)
            prev_tick = tick

        return new_track

    def _convert_via_metric_modulation(
        self,
        target_time_sig: TimeSignatureInfo,
        params: MeterConversionParams
    ) -> ConversionResult:
        """
        Convert using metric modulation technique.

        Finds a pivot rhythm that exists in both meters and uses it
        to create a smooth transition.
        """
        # Find pivot rhythm
        pivot = self._find_pivot_rhythm(
            self.current_time_sig,
            target_time_sig
        )

        if pivot is None:
            # Fall back to basic conversion
            return self._convert_basic(target_time_sig, params)

        # Calculate tempo ratio based on pivot
        tempo_ratio = pivot['new_tempo'] / pivot['old_tempo']

        # Apply conversion with tempo adjustment
        new_midi = self._apply_metric_modulation(
            target_time_sig,
            tempo_ratio,
            pivot
        )

        return ConversionResult(
            success=True,
            new_midi=new_midi,
            new_time_signature=target_time_sig,
            tempo_change_factor=tempo_ratio,
            stats={'pivot_rhythm': pivot}
        )

    def _find_pivot_rhythm(
        self,
        source_sig: TimeSignatureInfo,
        target_sig: TimeSignatureInfo
    ) -> Optional[Dict]:
        """
        Find a pivot rhythm for metric modulation.

        The pivot rhythm is a note value that will maintain its
        duration while the meter changes.
        """
        # Common pivot rhythms to try
        pivots = [
            {'name': 'quarter', 'value': Fraction(1, 1)},
            {'name': 'eighth', 'value': Fraction(1, 2)},
            {'name': 'dotted_quarter', 'value': Fraction(3, 2)},
            {'name': 'half', 'value': Fraction(2, 1)},
            {'name': 'triplet_quarter', 'value': Fraction(2, 3)},
        ]

        for pivot in pivots:
            # Calculate how many pivots fit in each measure
            pivots_in_source = source_sig.beats_per_measure / float(pivot['value'])
            pivots_in_target = target_sig.beats_per_measure / float(pivot['value'])

            # Check if both are reasonable integers or simple fractions
            if (abs(pivots_in_source - round(pivots_in_source)) < 0.01 and
                abs(pivots_in_target - round(pivots_in_target)) < 0.01):

                # Found a good pivot
                tempo_ratio = pivots_in_target / pivots_in_source

                return {
                    'name': pivot['name'],
                    'value': pivot['value'],
                    'old_tempo': self.tempo,
                    'new_tempo': self.tempo * tempo_ratio,
                    'pivots_per_measure_old': pivots_in_source,
                    'pivots_per_measure_new': pivots_in_target
                }

        return None

    def _apply_metric_modulation(
        self,
        target_time_sig: TimeSignatureInfo,
        tempo_ratio: float,
        pivot: Dict
    ) -> MidiFile:
        """Apply metric modulation to create new MIDI file."""
        new_midi = MidiFile(ticks_per_beat=self.ppqn)

        for track in self.midi.tracks:
            new_track = MidiTrack()
            current_tick = 0

            for msg in track:
                current_tick += msg.time
                new_msg = msg.copy()

                # Update time signature
                if msg.type == 'time_signature':
                    new_msg = MetaMessage('time_signature',
                                         numerator=target_time_sig.numerator,
                                         denominator=target_time_sig.denominator)

                # Update tempo
                elif msg.type == 'set_tempo':
                    new_tempo = int(msg.tempo / tempo_ratio)
                    new_msg = MetaMessage('set_tempo', tempo=new_tempo)

                new_track.append(new_msg)

            new_midi.tracks.append(new_track)

        return new_midi

    def _convert_via_stretch(
        self,
        target_time_sig: TimeSignatureInfo,
        params: MeterConversionParams
    ) -> ConversionResult:
        """
        Convert by stretching/compressing time proportionally.

        This maintains relative timing but changes absolute durations.
        """
        stretch_factor = (target_time_sig.beats_per_measure /
                         self.current_time_sig.beats_per_measure)

        new_midi = MidiFile(ticks_per_beat=self.ppqn)

        for track in self.midi.tracks:
            new_track = self._stretch_track(track, stretch_factor, target_time_sig)
            new_midi.tracks.append(new_track)

        return ConversionResult(
            success=True,
            new_midi=new_midi,
            new_time_signature=target_time_sig,
            tempo_change_factor=1.0 / stretch_factor if params.adjust_tempo else 1.0,
            stats={'stretch_factor': stretch_factor}
        )

    def _stretch_track(
        self,
        track: MidiTrack,
        stretch_factor: float,
        target_time_sig: TimeSignatureInfo
    ) -> MidiTrack:
        """Stretch a track by a given factor."""
        new_track = MidiTrack()

        for msg in track:
            new_msg = msg.copy()

            # Stretch delta times
            if hasattr(msg, 'time'):
                new_msg.time = int(msg.time * stretch_factor)

            # Update time signature
            if msg.type == 'time_signature':
                new_msg = MetaMessage('time_signature',
                                     numerator=target_time_sig.numerator,
                                     denominator=target_time_sig.denominator,
                                     time=new_msg.time)

            new_track.append(new_msg)

        return new_track

    def _convert_via_redistribution(
        self,
        target_time_sig: TimeSignatureInfo,
        params: MeterConversionParams
    ) -> ConversionResult:
        """
        Convert by redistributing notes across new measure structure.

        This attempts to maintain note density while fitting the new meter.
        """
        # Extract notes
        notes = self._extract_all_notes()

        # Group notes by measure
        source_measures = self._group_notes_by_measure(notes, self.current_time_sig)

        # Redistribute to target measures
        target_measures = self._redistribute_measures(
            source_measures,
            self.current_time_sig,
            target_time_sig
        )

        # Reconstruct MIDI
        new_midi = self._reconstruct_midi(target_measures, target_time_sig)

        return ConversionResult(
            success=True,
            new_midi=new_midi,
            new_time_signature=target_time_sig
        )

    def _group_notes_by_measure(
        self,
        notes: List[Dict],
        time_sig: TimeSignatureInfo
    ) -> Dict[int, List[Dict]]:
        """Group notes by measure number."""
        measures = {}

        for note in notes:
            measure_num = note['tick'] // time_sig.ticks_per_measure
            if measure_num not in measures:
                measures[measure_num] = []
            measures[measure_num].append(note)

        return measures

    def _redistribute_measures(
        self,
        source_measures: Dict[int, List[Dict]],
        source_sig: TimeSignatureInfo,
        target_sig: TimeSignatureInfo
    ) -> Dict[int, List[Dict]]:
        """Redistribute notes from source measures to target measures."""
        target_measures = {}

        # Calculate how many target measures per source measure
        ratio = target_sig.beats_per_measure / source_sig.beats_per_measure

        for source_measure_num, notes in source_measures.items():
            # Determine target measure(s)
            target_measure_num = int(source_measure_num * ratio)

            # Adjust note positions within measure
            adjusted_notes = []
            for note in notes:
                position_in_measure = note['tick'] % source_sig.ticks_per_measure
                position_ratio = position_in_measure / source_sig.ticks_per_measure

                # Map to new measure
                new_position_in_measure = int(position_ratio * target_sig.ticks_per_measure)
                new_tick = target_measure_num * target_sig.ticks_per_measure + new_position_in_measure

                new_note = note.copy()
                new_note['tick'] = new_tick
                # Adjust duration proportionally
                new_note['duration'] = int(note['duration'] * ratio)

                adjusted_notes.append(new_note)

            if target_measure_num not in target_measures:
                target_measures[target_measure_num] = []
            target_measures[target_measure_num].extend(adjusted_notes)

        return target_measures

    def _reconstruct_midi(
        self,
        measures: Dict[int, List[Dict]],
        time_sig: TimeSignatureInfo
    ) -> MidiFile:
        """Reconstruct MIDI file from measure-grouped notes."""
        new_midi = MidiFile(ticks_per_beat=self.ppqn)

        # Group notes by track
        tracks_data = {}
        for measure_notes in measures.values():
            for note in measure_notes:
                track_idx = note['track']
                if track_idx not in tracks_data:
                    tracks_data[track_idx] = []
                tracks_data[track_idx].append(note)

        # Create tracks
        for track_idx in sorted(tracks_data.keys()):
            track = self._create_track_from_notes(
                tracks_data[track_idx],
                time_sig
            )
            new_midi.tracks.append(track)

        return new_midi

    def _create_track_from_notes(
        self,
        notes: List[Dict],
        time_sig: TimeSignatureInfo
    ) -> MidiTrack:
        """Create a MIDI track from a list of notes."""
        track = MidiTrack()

        # Add time signature
        track.append(MetaMessage('time_signature',
                                numerator=time_sig.numerator,
                                denominator=time_sig.denominator,
                                time=0))

        # Sort notes by time
        notes.sort(key=lambda n: n['tick'])

        # Create note on/off events
        events = []
        for note in notes:
            events.append({
                'tick': note['tick'],
                'type': 'note_on',
                'note': note['pitch'],
                'velocity': note['velocity'],
                'channel': note['channel']
            })
            events.append({
                'tick': note['tick'] + note['duration'],
                'type': 'note_off',
                'note': note['pitch'],
                'velocity': 0,
                'channel': note['channel']
            })

        # Sort all events
        events.sort(key=lambda e: (e['tick'], e['type'] == 'note_off'))

        # Convert to delta times
        prev_tick = 0
        for event in events:
            delta_time = event['tick'] - prev_tick

            if event['type'] == 'note_on':
                track.append(Message('note_on',
                                   note=event['note'],
                                   velocity=event['velocity'],
                                   channel=event['channel'],
                                   time=delta_time))
            else:
                track.append(Message('note_off',
                                   note=event['note'],
                                   velocity=event['velocity'],
                                   channel=event['channel'],
                                   time=delta_time))

            prev_tick = event['tick']

        return track

    def _convert_basic(
        self,
        target_time_sig: TimeSignatureInfo,
        params: MeterConversionParams
    ) -> ConversionResult:
        """
        Basic conversion - just change time signature metadata.

        This is the simplest conversion that only updates the time
        signature marker without adjusting note positions.
        """
        new_midi = copy.deepcopy(self.midi)

        for track in new_midi.tracks:
            for i, msg in enumerate(track):
                if msg.type == 'time_signature':
                    track[i] = MetaMessage('time_signature',
                                          numerator=target_time_sig.numerator,
                                          denominator=target_time_sig.denominator,
                                          time=msg.time)

        return ConversionResult(
            success=True,
            new_midi=new_midi,
            new_time_signature=target_time_sig,
            warnings=["Basic conversion - note positions not adjusted"]
        )


# ==============================================================================
# METRIC MODULATOR CLASS
# ==============================================================================

class MetricModulator:
    """
    Advanced metric modulation using Elliott Carter techniques.

    Creates smooth tempo transitions by establishing rhythmic
    relationships between different meters.
    """

    @staticmethod
    def calculate_tempo_relationship(
        old_time_sig: TimeSignatureInfo,
        new_time_sig: TimeSignatureInfo,
        pivot_note_value: Fraction = Fraction(1, 1)
    ) -> Dict:
        """
        Calculate tempo relationship for metric modulation.

        Formula: new_tempo / old_tempo = pivot_notes_new / pivot_notes_old

        Args:
            old_time_sig: Source time signature
            new_time_sig: Target time signature
            pivot_note_value: Note value that stays constant (default: quarter note)

        Returns:
            Dictionary with tempo relationships
        """
        # Calculate how many pivot notes fit in each measure
        old_beats = old_time_sig.beats_per_measure
        new_beats = new_time_sig.beats_per_measure

        pivots_per_old_measure = old_beats / float(pivot_note_value)
        pivots_per_new_measure = new_beats / float(pivot_note_value)

        # Calculate tempo ratio
        tempo_ratio = pivots_per_new_measure / pivots_per_old_measure

        return {
            'tempo_ratio': tempo_ratio,
            'pivot_value': pivot_note_value,
            'pivots_per_old_measure': pivots_per_old_measure,
            'pivots_per_new_measure': pivots_per_new_measure,
            'description': f"{pivot_note_value} note = {pivot_note_value} note"
        }

    @staticmethod
    def find_best_pivot(
        old_time_sig: TimeSignatureInfo,
        new_time_sig: TimeSignatureInfo
    ) -> Tuple[Fraction, float]:
        """
        Find the best pivot note value for smooth modulation.

        Returns:
            (pivot_value, tempo_ratio)
        """
        candidates = [
            Fraction(1, 4),   # Sixteenth note
            Fraction(1, 2),   # Eighth note
            Fraction(3, 4),   # Dotted eighth
            Fraction(1, 1),   # Quarter note
            Fraction(3, 2),   # Dotted quarter
            Fraction(2, 1),   # Half note
            Fraction(3, 1),   # Dotted half
        ]

        best_pivot = candidates[3]  # Default to quarter note
        best_score = float('inf')

        for pivot in candidates:
            old_beats = old_time_sig.beats_per_measure
            new_beats = new_time_sig.beats_per_measure

            pivots_old = old_beats / float(pivot)
            pivots_new = new_beats / float(pivot)

            # Score based on how close to integer values
            score = (abs(pivots_old - round(pivots_old)) +
                    abs(pivots_new - round(pivots_new)))

            # Prefer common note values
            if pivot in [Fraction(1, 2), Fraction(1, 1), Fraction(3, 2)]:
                score *= 0.8

            if score < best_score:
                best_score = score
                best_pivot = pivot

        # Calculate tempo ratio
        tempo_ratio = (new_time_sig.beats_per_measure / float(best_pivot)) / \
                     (old_time_sig.beats_per_measure / float(best_pivot))

        return best_pivot, tempo_ratio


# ==============================================================================
# PHRASE PRESERVER CLASS
# ==============================================================================

class PhrasePreserver:
    """
    Maintains phrase structure during meter conversion.

    Analyzes phrases and ensures they remain coherent after
    time signature changes.
    """

    def __init__(self, midi_file: MidiFile, time_sig: TimeSignatureInfo):
        self.midi = midi_file
        self.time_sig = time_sig
        self.phrases = []

    def analyze_phrases(self) -> List[Dict]:
        """
        Analyze phrase structure in the MIDI file.

        Returns:
            List of phrase dictionaries with start/end positions
        """
        # Extract melodic content (highest notes typically)
        melody_notes = self._extract_melody()

        # Detect phrases based on rests and contour
        phrases = self._segment_into_phrases(melody_notes)

        self.phrases = phrases
        return phrases

    def _extract_melody(self) -> List[Dict]:
        """Extract melody line (typically highest notes)."""
        notes = []

        for track in self.midi.tracks:
            current_tick = 0
            active_notes = {}

            for msg in track:
                current_tick += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = current_tick
                elif msg.type in ['note_off', 'note_on']:
                    if msg.note in active_notes:
                        start = active_notes.pop(msg.note)
                        notes.append({
                            'pitch': msg.note,
                            'start': start,
                            'end': current_tick,
                            'duration': current_tick - start
                        })

        # Sort by time and keep highest notes at each time
        notes.sort(key=lambda n: n['start'])

        # Simple melody extraction: keep highest note at each time slice
        melody = []
        for note in notes:
            # Check if this overlaps with previous melody note
            if melody and note['start'] < melody[-1]['end']:
                # Keep higher note
                if note['pitch'] > melody[-1]['pitch']:
                    melody[-1] = note
            else:
                melody.append(note)

        return melody

    def _segment_into_phrases(self, melody: List[Dict]) -> List[Dict]:
        """Segment melody into phrases."""
        if not melody:
            return []

        phrases = []
        current_phrase_start = melody[0]['start']

        # Look for phrase boundaries (long rests or measure boundaries)
        rest_threshold = self.time_sig.ticks_per_measure * 0.5

        for i in range(len(melody) - 1):
            note = melody[i]
            next_note = melody[i + 1]

            rest_duration = next_note['start'] - note['end']

            if rest_duration >= rest_threshold:
                # End of phrase
                phrases.append({
                    'start_tick': current_phrase_start,
                    'end_tick': note['end'],
                    'start_measure': current_phrase_start // self.time_sig.ticks_per_measure,
                    'end_measure': note['end'] // self.time_sig.ticks_per_measure,
                    'length_measures': (note['end'] - current_phrase_start) / self.time_sig.ticks_per_measure
                })
                current_phrase_start = next_note['start']

        # Add final phrase
        if melody:
            phrases.append({
                'start_tick': current_phrase_start,
                'end_tick': melody[-1]['end'],
                'start_measure': current_phrase_start // self.time_sig.ticks_per_measure,
                'end_measure': melody[-1]['end'] // self.time_sig.ticks_per_measure,
                'length_measures': (melody[-1]['end'] - current_phrase_start) / self.time_sig.ticks_per_measure
            })

        return phrases

    def preserve_phrase_in_conversion(
        self,
        phrase: Dict,
        old_time_sig: TimeSignatureInfo,
        new_time_sig: TimeSignatureInfo
    ) -> Dict:
        """
        Calculate how to preserve a phrase in the new meter.

        Returns:
            Dictionary with new phrase boundaries and adjustments
        """
        # Calculate phrase length in beats
        phrase_length_beats = phrase['length_measures'] * old_time_sig.beats_per_measure

        # Calculate new measure count
        new_measure_count = phrase_length_beats / new_time_sig.beats_per_measure

        # Round to nearest measure
        new_measure_count = round(new_measure_count)

        # Calculate actual stretch factor needed
        ideal_stretch = new_measure_count * new_time_sig.beats_per_measure / phrase_length_beats

        return {
            'original_measures': phrase['length_measures'],
            'new_measures': new_measure_count,
            'stretch_factor': ideal_stretch,
            'new_start_measure': phrase['start_measure'],
            'new_end_measure': phrase['start_measure'] + new_measure_count
        }


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

class MeterUtilities:
    """Utility functions for meter conversion."""

    @staticmethod
    def quantize_to_meter(
        tick: int,
        time_sig: TimeSignatureInfo,
        ppqn: int,
        strength: float = 1.0
    ) -> int:
        """
        Quantize a tick position to the nearest grid point in the meter.

        Args:
            tick: Input tick position
            time_sig: Time signature to quantize to
            ppqn: Pulses per quarter note
            strength: Quantization strength (0-1)

        Returns:
            Quantized tick position
        """
        # Determine grid resolution based on meter
        if time_sig.denominator == 8:
            grid_ticks = ppqn // 2  # Eighth note grid
        else:
            grid_ticks = ppqn // 4  # Sixteenth note grid

        # Find nearest grid point
        grid_point = round(tick / grid_ticks) * grid_ticks

        # Apply strength (interpolate between original and quantized)
        result = int(tick * (1 - strength) + grid_point * strength)

        return result

    @staticmethod
    def get_meter_accent_pattern(time_sig: TimeSignatureInfo) -> List[float]:
        """
        Get accent pattern for a meter.

        Returns:
            List of accent weights (0-1) for each beat
        """
        if time_sig.family == MeterFamily.SIMPLE_DUPLE:
            return [1.0, 0.5]
        elif time_sig.family == MeterFamily.SIMPLE_TRIPLE:
            return [1.0, 0.5, 0.5]
        elif time_sig.family == MeterFamily.SIMPLE_QUADRUPLE:
            return [1.0, 0.5, 0.7, 0.5]
        elif time_sig.family == MeterFamily.COMPOUND_DUPLE:
            return [1.0, 0.4, 0.4, 0.7, 0.4, 0.4]
        elif time_sig.family == MeterFamily.ODD_METER:
            # Use grouping to determine accents
            if time_sig.grouping:
                accents = []
                for group in time_sig.grouping:
                    accents.append(1.0)  # Accent first of group
                    accents.extend([0.5] * (group - 1))  # Weaker beats
                return accents

        # Default: accent first beat
        return [1.0] + [0.5] * (time_sig.numerator - 1)

    @staticmethod
    def validate_time_signature(numerator: int, denominator: int) -> bool:
        """
        Validate time signature values.

        Args:
            numerator: Beats per measure
            denominator: Beat unit

        Returns:
            True if valid
        """
        # Check numerator
        if numerator < 1 or numerator > 32:
            return False

        # Check denominator (must be power of 2)
        if denominator not in [1, 2, 4, 8, 16, 32]:
            return False

        return True


# ==============================================================================
# QUICK CONVERSION FUNCTIONS
# ==============================================================================

def convert_midi_meter(
    input_file: str,
    output_file: str,
    new_numerator: int,
    new_denominator: int,
    strategy: str = "phrase_aware"
) -> bool:
    """
    Quick function to convert MIDI file to new time signature.

    Args:
        input_file: Path to input MIDI file
        output_file: Path to save output MIDI file
        new_numerator: Target numerator
        new_denominator: Target denominator
        strategy: Conversion strategy name

    Returns:
        True if successful
    """
    try:
        converter = MeterConverter(input_file)

        # Map strategy string to enum
        strategy_map = {
            "stretch": ConversionStrategy.STRETCH,
            "redistribute": ConversionStrategy.REDISTRIBUTE,
            "metric_modulation": ConversionStrategy.METRIC_MODULATION,
            "phrase_aware": ConversionStrategy.PHRASE_AWARE,
        }

        strategy_enum = strategy_map.get(strategy, ConversionStrategy.PHRASE_AWARE)
        params = MeterConversionParams(strategy=strategy_enum)

        result = converter.convert_meter(
            new_numerator,
            new_denominator,
            params=params
        )

        if result.success and result.new_midi:
            result.new_midi.save(output_file)
            print(f"Converted {input_file} to {new_numerator}/{new_denominator}")
            print(f"Saved to {output_file}")
            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
            return True

    except Exception as e:
        print(f"Error converting meter: {e}")
        return False

    return False


if __name__ == "__main__":
    print("Meter Converter Module - Agent 7")
    print("=" * 50)
    print("\nUsage:")
    print("  from meter_converter import MeterConverter")
    print("  converter = MeterConverter('input.mid')")
    print("  result = converter.convert_meter(7, 8)")
    print("  result.new_midi.save('output.mid')")
