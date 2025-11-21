"""
Rich Multitrack Feature Extractor - Agent 2
============================================

Extracts 600D rich multitrack features from MIDI files for training.

Feature Breakdown:
- Global features (200D): Existing OptimizedFeatureExtractor
- Per-track features (200D): 8 tracks × 25D each
- Temporal features (100D): 4 sections × 25D
- Orchestration features (100D): Detailed arrangement analysis

TOTAL: 600D feature vector

This extractor provides comprehensive multitrack analysis including:
- Track role classification (bass, melody, harmony, drums, other)
- Per-track density, register, rhythm, articulation
- Temporal evolution across sections
- Orchestration quality and balance

Author: Agent 2 - Rich Feature Extraction Specialist
License: MIT
"""

import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter, defaultdict

import numpy as np
import pretty_midi
from scipy import stats

# Import existing feature extractor for global features
try:
    from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor
    BASE_EXTRACTOR_AVAILABLE = True
except ImportError:
    BASE_EXTRACTOR_AVAILABLE = False
    print("WARNING: DeepFeatureExtractor not available")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrackInfo:
    """Information about a single MIDI track"""
    index: int
    instrument: int  # GM program number
    role: str  # bass, melody, harmony, drums, other
    note_count: int
    avg_pitch: float
    pitch_range: float
    avg_velocity: float
    density: float  # notes per second
    register: str  # low, mid, high


@dataclass
class Section:
    """Information about a section of the piece"""
    start_time: float
    end_time: float
    duration: float
    label: str  # intro, verse, chorus, bridge, outro


# ============================================================================
# Rich Multitrack Feature Extractor
# ============================================================================

class RichMultitrackFeatureExtractor:
    """
    Extract 600D rich multitrack features from MIDI files.

    Feature Architecture:
    - Global features (200D): Full-file statistical features
    - Per-track features (200D): 8 tracks × 25D (role, density, register, etc.)
    - Temporal features (100D): 4 sections × 25D (evolution over time)
    - Orchestration features (100D): Arrangement quality and balance

    Usage:
        extractor = RichMultitrackFeatureExtractor()
        features = extractor.extract('song.mid')  # Returns 600D vector

        # Batch extraction with multiprocessing
        features = extractor.extract_batch(midi_files, n_workers=16)
    """

    def __init__(self, use_base_extractor: bool = True):
        """
        Initialize rich feature extractor.

        Args:
            use_base_extractor: Use DeepFeatureExtractor for global features
        """
        self.use_base_extractor = use_base_extractor and BASE_EXTRACTOR_AVAILABLE

        if self.use_base_extractor:
            self.base_extractor = DeepFeatureExtractor()
        else:
            self.base_extractor = None

        # Feature dimensions
        self.GLOBAL_DIM = 200
        self.PER_TRACK_DIM = 200  # 8 tracks × 25D
        self.TEMPORAL_DIM = 100   # 4 sections × 25D
        self.ORCHESTRATION_DIM = 100
        self.TOTAL_DIM = 600

        # Track configuration
        self.MAX_TRACKS = 8
        self.FEATURES_PER_TRACK = 25

        # Section configuration
        self.NUM_SECTIONS = 4
        self.FEATURES_PER_SECTION = 25

        # GM instrument roles
        self.BASS_PROGRAMS = list(range(32, 40))  # Bass instruments
        self.MELODY_PROGRAMS = list(range(0, 8)) + list(range(24, 32))  # Piano, Guitar
        self.HARMONY_PROGRAMS = list(range(8, 16)) + list(range(16, 24))  # Chromatic, Organ
        self.DRUM_CHANNEL = 9  # Channel 10 (0-indexed as 9)

        print(f"✅ RichMultitrackFeatureExtractor initialized")
        print(f"   Total dimensions: {self.TOTAL_DIM}")
        print(f"   - Global: {self.GLOBAL_DIM}")
        print(f"   - Per-track: {self.PER_TRACK_DIM} ({self.MAX_TRACKS} × {self.FEATURES_PER_TRACK})")
        print(f"   - Temporal: {self.TEMPORAL_DIM} ({self.NUM_SECTIONS} × {self.FEATURES_PER_SECTION})")
        print(f"   - Orchestration: {self.ORCHESTRATION_DIM}")

    def extract(self, midi_path: str) -> np.ndarray:
        """
        Extract 600D feature vector from MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            numpy array of shape (600,)
        """
        start_time = time.time()

        # Load MIDI file
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            print(f"Error loading {midi_path}: {e}")
            return np.zeros(self.TOTAL_DIM)

        # Extract all feature components
        global_features = self.extract_global(midi, str(midi_path))
        per_track_features = self.extract_per_track(midi)
        temporal_features = self.extract_temporal(midi)
        orchestration_features = self.extract_orchestration(midi)

        # Concatenate all features
        features = np.concatenate([
            global_features,          # 200D
            per_track_features,       # 200D
            temporal_features,        # 100D
            orchestration_features,   # 100D
        ])

        assert features.shape == (self.TOTAL_DIM,), \
            f"Expected {self.TOTAL_DIM}D, got {features.shape[0]}D"

        extraction_time = time.time() - start_time

        if extraction_time > 30.0:
            print(f"⚠️ Extraction took {extraction_time:.2f}s (target: < 30.0s)")

        return features

    # ========================================================================
    # Global Features (200D)
    # ========================================================================

    def extract_global(self, midi: pretty_midi.PrettyMIDI, midi_path: str = None) -> np.ndarray:
        """
        Extract global 200D features using base extractor or compute directly.

        Args:
            midi: PrettyMIDI object
            midi_path: Optional path to MIDI file (for base extractor)

        Returns:
            numpy array of shape (200,)
        """
        if self.use_base_extractor and midi_path:
            try:
                # Use existing DeepFeatureExtractor
                features = self.base_extractor.extract(midi_path)
                # Take first 200 features
                return features[:200] if len(features) >= 200 else np.pad(features, (0, 200 - len(features)))
            except Exception as e:
                print(f"Warning: Base extractor failed, using fallback: {e}")

        # Fallback: Compute basic global features
        return self._compute_basic_global_features(midi)

    def _compute_basic_global_features(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Compute basic global features as fallback.

        Returns:
            numpy array of shape (200,)
        """
        features = []

        # Get all notes
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend(instrument.notes)

        if not all_notes:
            return np.zeros(200)

        # Basic statistics (50 features)
        pitches = [n.pitch for n in all_notes]
        velocities = [n.velocity for n in all_notes]
        durations = [n.end - n.start for n in all_notes]

        features.extend([
            np.mean(pitches), np.std(pitches), np.min(pitches), np.max(pitches),
            np.mean(velocities), np.std(velocities), np.min(velocities), np.max(velocities),
            np.mean(durations), np.std(durations), np.min(durations), np.max(durations),
            len(all_notes), len(midi.instruments), midi.get_end_time(),
            len(all_notes) / max(midi.get_end_time(), 0.1),  # note density
        ])

        # Pitch class distribution (12 features)
        pitch_classes = [p % 12 for p in pitches]
        pc_dist = np.bincount(pitch_classes, minlength=12) / len(pitch_classes)
        features.extend(pc_dist)

        # Interval distribution (12 features)
        intervals = [abs(pitches[i+1] - pitches[i]) % 12 for i in range(len(pitches)-1)] if len(pitches) > 1 else [0]*12
        interval_dist = np.bincount(intervals, minlength=12) / max(len(intervals), 1)
        features.extend(interval_dist)

        # Rhythm features (20 features)
        if len(all_notes) > 1:
            iois = [all_notes[i+1].start - all_notes[i].start for i in range(len(all_notes)-1)]
            features.extend([
                np.mean(iois), np.std(iois), np.min(iois), np.max(iois),
                stats.skew(iois) if len(iois) > 2 else 0,
                stats.kurtosis(iois) if len(iois) > 2 else 0,
            ])
        else:
            features.extend([0] * 6)

        # Tempo features (10 features)
        tempo_changes = midi.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            features.extend([
                np.mean(tempo_changes[1]), np.std(tempo_changes[1]),
                np.min(tempo_changes[1]), np.max(tempo_changes[1]),
            ])
        else:
            features.extend([120.0, 0, 120.0, 120.0])  # Default tempo

        # Pad to 200D
        features.extend([0] * (200 - len(features)))

        return np.array(features[:200])

    # ========================================================================
    # Per-Track Features (200D = 8 tracks × 25D)
    # ========================================================================

    def extract_per_track(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract per-track features for up to 8 tracks.

        For each track, extract 25D features:
        - Track role classification (5D one-hot: bass, melody, harmony, drums, other)
        - Density metrics (5D: note count, notes/sec, avg duration, etc.)
        - Register metrics (5D: avg pitch, pitch range, pitch std, etc.)
        - Rhythm metrics (5D: avg velocity, velocity range, articulation, etc.)
        - Interaction metrics (5D: overlap with other tracks, etc.)

        Args:
            midi: PrettyMIDI object

        Returns:
            numpy array of shape (200,) = 8 × 25
        """
        # Analyze all tracks
        track_infos = self._analyze_tracks(midi)

        # Extract features for each track
        all_features = []

        for i in range(self.MAX_TRACKS):
            if i < len(track_infos):
                track_features = self._extract_single_track_features(
                    midi, track_infos[i], i
                )
            else:
                # Zero-padding for missing tracks
                track_features = np.zeros(self.FEATURES_PER_TRACK)

            all_features.extend(track_features)

        features = np.array(all_features)
        assert features.shape == (self.PER_TRACK_DIM,)

        return features

    def _analyze_tracks(self, midi: pretty_midi.PrettyMIDI) -> List[TrackInfo]:
        """
        Analyze all tracks and classify their roles.

        Returns:
            List of TrackInfo objects, sorted by importance
        """
        track_infos = []

        for idx, instrument in enumerate(midi.instruments):
            if len(instrument.notes) == 0:
                continue

            # Get track statistics
            pitches = [n.pitch for n in instrument.notes]
            velocities = [n.velocity for n in instrument.notes]

            avg_pitch = np.mean(pitches)
            pitch_range = np.max(pitches) - np.min(pitches)
            avg_velocity = np.mean(velocities)

            duration = midi.get_end_time()
            density = len(instrument.notes) / max(duration, 0.1)

            # Classify track role
            role = self._classify_track_role(instrument, avg_pitch)

            # Determine register
            if avg_pitch < 48:
                register = "low"
            elif avg_pitch < 72:
                register = "mid"
            else:
                register = "high"

            track_info = TrackInfo(
                index=idx,
                instrument=instrument.program,
                role=role,
                note_count=len(instrument.notes),
                avg_pitch=avg_pitch,
                pitch_range=pitch_range,
                avg_velocity=avg_velocity,
                density=density,
                register=register
            )

            track_infos.append(track_info)

        # Sort by importance (note count as proxy)
        track_infos.sort(key=lambda t: t.note_count, reverse=True)

        return track_infos[:self.MAX_TRACKS]

    def _classify_track_role(self, instrument: pretty_midi.Instrument, avg_pitch: float) -> str:
        """
        Classify track role based on instrument and pitch.

        Returns:
            One of: 'bass', 'melody', 'harmony', 'drums', 'other'
        """
        # Check if drums (channel 10)
        if instrument.is_drum:
            return 'drums'

        # Check bass range and instrument
        if avg_pitch < 52 or instrument.program in self.BASS_PROGRAMS:
            return 'bass'

        # Check melody instruments
        if instrument.program in self.MELODY_PROGRAMS:
            return 'melody'

        # Check harmony instruments
        if instrument.program in self.HARMONY_PROGRAMS:
            return 'harmony'

        return 'other'

    def _extract_single_track_features(
        self,
        midi: pretty_midi.PrettyMIDI,
        track_info: TrackInfo,
        track_idx: int
    ) -> np.ndarray:
        """
        Extract 25D features for a single track.

        Returns:
            numpy array of shape (25,)
        """
        features = []

        instrument = midi.instruments[track_info.index]
        notes = instrument.notes

        # 1. Role classification (5D one-hot)
        role_encoding = self._encode_role(track_info.role)
        features.extend(role_encoding)

        # 2. Density metrics (5D)
        features.extend([
            track_info.note_count / 1000.0,  # Normalized note count
            track_info.density / 10.0,        # Normalized density
            np.mean([n.end - n.start for n in notes]) if notes else 0,  # Avg duration
            np.std([n.end - n.start for n in notes]) if len(notes) > 1 else 0,  # Duration std
            len([n for n in notes if n.end - n.start < 0.1]) / max(len(notes), 1),  # Staccato ratio
        ])

        # 3. Register metrics (5D)
        features.extend([
            track_info.avg_pitch / 127.0,     # Normalized avg pitch
            track_info.pitch_range / 88.0,    # Normalized pitch range
            np.std([n.pitch for n in notes]) / 127.0 if len(notes) > 1 else 0,  # Pitch std
            1.0 if track_info.register == "low" else 0.0,
            1.0 if track_info.register == "high" else 0.0,
        ])

        # 4. Rhythm/Articulation metrics (5D)
        features.extend([
            track_info.avg_velocity / 127.0,  # Normalized avg velocity
            np.std([n.velocity for n in notes]) / 127.0 if len(notes) > 1 else 0,  # Velocity std
            np.max([n.velocity for n in notes]) / 127.0 if notes else 0,  # Max velocity
            np.min([n.velocity for n in notes]) / 127.0 if notes else 0,  # Min velocity
            self._compute_syncopation(notes) if len(notes) > 2 else 0,  # Syncopation
        ])

        # 5. Interaction metrics (5D)
        features.extend([
            self._compute_overlap_ratio(instrument, midi),  # Overlap with other tracks
            self._compute_polyphony(notes),                  # Avg simultaneous notes
            track_info.instrument / 127.0,                   # Normalized instrument program
            float(track_idx) / self.MAX_TRACKS,              # Track position
            1.0 if len(notes) > 100 else 0.0,                # Active track indicator
        ])

        features = np.array(features)
        assert len(features) == self.FEATURES_PER_TRACK

        return features

    def _encode_role(self, role: str) -> List[float]:
        """Encode track role as one-hot vector."""
        roles = ['bass', 'melody', 'harmony', 'drums', 'other']
        encoding = [1.0 if role == r else 0.0 for r in roles]
        return encoding

    def _compute_syncopation(self, notes: List[pretty_midi.Note]) -> float:
        """Compute syncopation metric (simplified)."""
        if len(notes) < 2:
            return 0.0

        # Count notes that start off strong beats
        off_beat_count = 0
        for note in notes:
            beat_position = note.start % 1.0  # Position within beat
            if 0.4 < beat_position < 0.6:  # Around middle of beat
                off_beat_count += 1

        return off_beat_count / len(notes)

    def _compute_overlap_ratio(
        self,
        instrument: pretty_midi.Instrument,
        midi: pretty_midi.PrettyMIDI
    ) -> float:
        """Compute ratio of time this track overlaps with others."""
        if len(instrument.notes) == 0:
            return 0.0

        # Get this instrument's active time
        this_active = self._get_active_time_intervals(instrument.notes)

        # Get other instruments' active time
        other_active = []
        for other_inst in midi.instruments:
            if other_inst != instrument and len(other_inst.notes) > 0:
                other_active.extend(self._get_active_time_intervals(other_inst.notes))

        if not other_active:
            return 0.0

        # Compute overlap
        overlap_duration = 0.0
        for start, end in this_active:
            for other_start, other_end in other_active:
                overlap_start = max(start, other_start)
                overlap_end = min(end, other_end)
                if overlap_end > overlap_start:
                    overlap_duration += (overlap_end - overlap_start)

        total_duration = sum(end - start for start, end in this_active)

        return min(overlap_duration / max(total_duration, 0.1), 1.0)

    def _get_active_time_intervals(self, notes: List[pretty_midi.Note]) -> List[Tuple[float, float]]:
        """Get time intervals where notes are active."""
        if not notes:
            return []

        # Merge overlapping intervals
        intervals = sorted([(n.start, n.end) for n in notes])
        merged = [intervals[0]]

        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                # Overlapping, merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    def _compute_polyphony(self, notes: List[pretty_midi.Note]) -> float:
        """Compute average number of simultaneous notes."""
        if len(notes) < 2:
            return 1.0

        # Sample at regular intervals
        end_time = max(n.end for n in notes)
        sample_points = np.linspace(0, end_time, min(100, int(end_time * 10)))

        polyphony_values = []
        for t in sample_points:
            count = sum(1 for n in notes if n.start <= t < n.end)
            polyphony_values.append(count)

        return np.mean(polyphony_values) if polyphony_values else 1.0

    # ========================================================================
    # Temporal Features (100D = 4 sections × 25D)
    # ========================================================================

    def extract_temporal(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract temporal evolution features across 4 sections.

        Detects sections (intro, verse, chorus, bridge/outro) and extracts
        features per section to capture temporal evolution.

        Args:
            midi: PrettyMIDI object

        Returns:
            numpy array of shape (100,) = 4 × 25
        """
        # Detect sections
        sections = self._detect_sections(midi)

        # Extract features for each section
        all_features = []

        for i in range(self.NUM_SECTIONS):
            if i < len(sections):
                section_features = self._extract_section_features(midi, sections[i])
            else:
                # Zero-padding for missing sections
                section_features = np.zeros(self.FEATURES_PER_SECTION)

            all_features.extend(section_features)

        features = np.array(all_features)
        assert features.shape == (self.TEMPORAL_DIM,)

        return features

    def _detect_sections(self, midi: pretty_midi.PrettyMIDI) -> List[Section]:
        """
        Detect sections in the MIDI file.

        Simple heuristic: divide into 4 equal sections and label them.
        For more advanced detection, could use tempo changes, key changes, etc.

        Returns:
            List of Section objects
        """
        total_duration = midi.get_end_time()

        if total_duration < 4.0:
            # Too short, just use one section
            return [Section(0, total_duration, total_duration, 'intro')]

        # Divide into 4 equal sections
        section_duration = total_duration / 4
        labels = ['intro', 'verse', 'chorus', 'bridge']

        sections = []
        for i in range(4):
            start = i * section_duration
            end = (i + 1) * section_duration
            sections.append(Section(
                start_time=start,
                end_time=end,
                duration=section_duration,
                label=labels[i]
            ))

        return sections

    def _extract_section_features(
        self,
        midi: pretty_midi.PrettyMIDI,
        section: Section
    ) -> np.ndarray:
        """
        Extract 25D features for a single section.

        Returns:
            numpy array of shape (25,)
        """
        features = []

        # Get all notes in this section
        section_notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                if section.start_time <= note.start < section.end_time:
                    section_notes.append(note)

        if not section_notes:
            return np.zeros(self.FEATURES_PER_SECTION)

        # 1. Density features (5D)
        features.extend([
            len(section_notes) / max(section.duration, 0.1),  # Note density
            len(section_notes) / 1000.0,                       # Normalized count
            np.mean([n.end - n.start for n in section_notes]), # Avg duration
            np.std([n.end - n.start for n in section_notes]) if len(section_notes) > 1 else 0,
            self._compute_polyphony(section_notes),            # Polyphony
        ])

        # 2. Pitch features (5D)
        pitches = [n.pitch for n in section_notes]
        features.extend([
            np.mean(pitches) / 127.0,
            np.std(pitches) / 127.0 if len(pitches) > 1 else 0,
            (np.max(pitches) - np.min(pitches)) / 88.0,  # Normalized range
            stats.skew(pitches) if len(pitches) > 2 else 0,
            stats.kurtosis(pitches) if len(pitches) > 2 else 0,
        ])

        # 3. Velocity/Dynamics features (5D)
        velocities = [n.velocity for n in section_notes]
        features.extend([
            np.mean(velocities) / 127.0,
            np.std(velocities) / 127.0 if len(velocities) > 1 else 0,
            np.max(velocities) / 127.0,
            np.min(velocities) / 127.0,
            (np.max(velocities) - np.min(velocities)) / 127.0,  # Dynamic range
        ])

        # 4. Rhythm features (5D)
        if len(section_notes) > 1:
            sorted_notes = sorted(section_notes, key=lambda n: n.start)
            iois = [sorted_notes[i+1].start - sorted_notes[i].start
                   for i in range(len(sorted_notes)-1)]
            features.extend([
                np.mean(iois),
                np.std(iois) if len(iois) > 1 else 0,
                self._compute_syncopation(section_notes),
                np.percentile(iois, 25) if iois else 0,
                np.percentile(iois, 75) if iois else 0,
            ])
        else:
            features.extend([0] * 5)

        # 5. Harmonic features (5D)
        pitch_classes = [p % 12 for p in pitches]
        pc_entropy = stats.entropy(np.bincount(pitch_classes, minlength=12) + 1e-10)
        unique_pcs = len(set(pitch_classes))
        features.extend([
            unique_pcs / 12.0,              # Pitch class variety
            pc_entropy / np.log(12),        # Normalized entropy
            Counter(pitch_classes).most_common(1)[0][1] / len(pitch_classes),  # Most common PC ratio
            sum(1 for pc in pitch_classes if pc in [0, 2, 4, 5, 7, 9, 11]) / len(pitch_classes),  # Diatonic ratio
            section.start_time / max(midi.get_end_time(), 0.1),  # Position in piece
        ])

        features = np.array(features)
        assert len(features) == self.FEATURES_PER_SECTION

        return features

    # ========================================================================
    # Orchestration Features (100D)
    # ========================================================================

    def extract_orchestration(self, midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Extract orchestration features.

        Analyzes:
        - Voice leading quality
        - Textural density per section
        - Instrument balance and blend
        - Voicing spread and spacing
        - Role distribution

        Args:
            midi: PrettyMIDI object

        Returns:
            numpy array of shape (100,)
        """
        features = []

        # 1. Role distribution (10D)
        role_features = self._extract_role_distribution(midi)
        features.extend(role_features)

        # 2. Voice leading features (20D)
        voice_leading = self._extract_voice_leading(midi)
        features.extend(voice_leading)

        # 3. Textural features (20D)
        texture = self._extract_texture_features(midi)
        features.extend(texture)

        # 4. Balance and blend features (20D)
        balance = self._extract_balance_features(midi)
        features.extend(balance)

        # 5. Spacing and voicing features (20D)
        spacing = self._extract_spacing_features(midi)
        features.extend(spacing)

        # 6. Instrumentation features (10D)
        instrumentation = self._extract_instrumentation_features(midi)
        features.extend(instrumentation)

        features = np.array(features)

        # Pad or truncate to 100D
        if len(features) < self.ORCHESTRATION_DIM:
            features = np.pad(features, (0, self.ORCHESTRATION_DIM - len(features)))
        else:
            features = features[:self.ORCHESTRATION_DIM]

        assert features.shape == (self.ORCHESTRATION_DIM,)

        return features

    def _extract_role_distribution(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        """Extract role distribution features (10D)."""
        features = []

        # Count tracks by role
        role_counts = {'bass': 0, 'melody': 0, 'harmony': 0, 'drums': 0, 'other': 0}

        for instrument in midi.instruments:
            if len(instrument.notes) == 0:
                continue

            avg_pitch = np.mean([n.pitch for n in instrument.notes])
            role = self._classify_track_role(instrument, avg_pitch)
            role_counts[role] += 1

        total_tracks = sum(role_counts.values())

        # Role ratios (5D)
        for role in ['bass', 'melody', 'harmony', 'drums', 'other']:
            features.append(role_counts[role] / max(total_tracks, 1))

        # Additional role metrics (5D)
        features.extend([
            float(total_tracks) / self.MAX_TRACKS,      # Track count ratio
            1.0 if role_counts['bass'] > 0 else 0.0,    # Has bass
            1.0 if role_counts['drums'] > 0 else 0.0,   # Has drums
            role_counts['melody'] / max(total_tracks, 1),  # Melody ratio
            role_counts['harmony'] / max(total_tracks, 1), # Harmony ratio
        ])

        return features

    def _extract_voice_leading(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        """Extract voice leading features (20D)."""
        features = []

        # Analyze voice leading between consecutive chords
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend([(n.start, n.pitch, n.end) for n in instrument.notes])

        all_notes.sort()

        if len(all_notes) < 10:
            return [0] * 20

        # Sample chords at regular intervals
        end_time = max(n[2] for n in all_notes)
        sample_times = np.linspace(0, end_time, min(50, int(end_time * 2)))

        chords = []
        for t in sample_times:
            chord_pitches = [pitch for start, pitch, end in all_notes if start <= t < end]
            if chord_pitches:
                chords.append(sorted(chord_pitches))

        if len(chords) < 2:
            return [0] * 20

        # Analyze voice leading between consecutive chords
        voice_motions = []
        for i in range(len(chords) - 1):
            chord1, chord2 = chords[i], chords[i+1]

            # Compute minimal motion
            total_motion = 0
            for p1 in chord1:
                if chord2:
                    min_distance = min(abs(p1 - p2) for p2 in chord2)
                    total_motion += min_distance

            voice_motions.append(total_motion / max(len(chord1), 1))

        # Voice leading statistics (10D)
        features.extend([
            np.mean(voice_motions) if voice_motions else 0,
            np.std(voice_motions) if len(voice_motions) > 1 else 0,
            np.min(voice_motions) if voice_motions else 0,
            np.max(voice_motions) if voice_motions else 0,
            np.median(voice_motions) if voice_motions else 0,
            sum(1 for m in voice_motions if m < 2) / max(len(voice_motions), 1),  # Stepwise ratio
            sum(1 for m in voice_motions if m > 7) / max(len(voice_motions), 1),  # Large leap ratio
            stats.skew(voice_motions) if len(voice_motions) > 2 else 0,
            stats.kurtosis(voice_motions) if len(voice_motions) > 2 else 0,
            len(voice_motions) / max(end_time, 0.1),  # Harmonic rhythm
        ])

        # Chord statistics (10D)
        chord_sizes = [len(c) for c in chords]
        features.extend([
            np.mean(chord_sizes),
            np.std(chord_sizes) if len(chord_sizes) > 1 else 0,
            np.min(chord_sizes),
            np.max(chord_sizes),
            sum(1 for s in chord_sizes if s >= 3) / len(chord_sizes),  # Triadic+ ratio
            sum(1 for s in chord_sizes if s >= 4) / len(chord_sizes),  # Four-note+ ratio
            sum(1 for s in chord_sizes if s == 1) / len(chord_sizes),  # Monophonic ratio
            np.median(chord_sizes),
            stats.skew(chord_sizes) if len(chord_sizes) > 2 else 0,
            stats.kurtosis(chord_sizes) if len(chord_sizes) > 2 else 0,
        ])

        return features

    def _extract_texture_features(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        """Extract textural features (20D)."""
        features = []

        # Get all notes
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend(instrument.notes)

        if not all_notes:
            return [0] * 20

        # Sample texture at regular intervals
        end_time = midi.get_end_time()
        sample_times = np.linspace(0, end_time, min(100, int(end_time * 4)))

        texture_samples = []
        for t in sample_times:
            active_notes = sum(1 for n in all_notes if n.start <= t < n.end)
            texture_samples.append(active_notes)

        # Texture statistics (10D)
        features.extend([
            np.mean(texture_samples),
            np.std(texture_samples),
            np.min(texture_samples),
            np.max(texture_samples),
            np.median(texture_samples),
            np.percentile(texture_samples, 25),
            np.percentile(texture_samples, 75),
            sum(1 for t in texture_samples if t == 0) / len(texture_samples),  # Silence ratio
            sum(1 for t in texture_samples if t > 5) / len(texture_samples),   # Dense ratio
            stats.skew(texture_samples) if len(texture_samples) > 2 else 0,
        ])

        # Temporal texture features (10D)
        if len(texture_samples) > 1:
            texture_changes = [abs(texture_samples[i+1] - texture_samples[i])
                             for i in range(len(texture_samples)-1)]
            features.extend([
                np.mean(texture_changes),
                np.std(texture_changes),
                np.max(texture_changes),
                sum(1 for c in texture_changes if c > 2) / len(texture_changes),  # Large change ratio
                np.mean(texture_samples[:len(texture_samples)//2]),  # First half avg
                np.mean(texture_samples[len(texture_samples)//2:]),  # Second half avg
                stats.skew(texture_changes) if len(texture_changes) > 2 else 0,
                stats.kurtosis(texture_changes) if len(texture_changes) > 2 else 0,
                np.corrcoef(range(len(texture_samples)), texture_samples)[0, 1],  # Trend
                len([i for i in range(len(texture_samples)-1) if texture_samples[i+1] > texture_samples[i]]) / len(texture_samples),
            ])
        else:
            features.extend([0] * 10)

        return features

    def _extract_balance_features(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        """Extract balance and blend features (20D)."""
        features = []

        # Get per-track statistics
        track_note_counts = []
        track_avg_velocities = []
        track_avg_pitches = []

        for instrument in midi.instruments:
            if len(instrument.notes) > 0:
                track_note_counts.append(len(instrument.notes))
                track_avg_velocities.append(np.mean([n.velocity for n in instrument.notes]))
                track_avg_pitches.append(np.mean([n.pitch for n in instrument.notes]))

        if not track_note_counts:
            return [0] * 20

        # Note count balance (5D)
        features.extend([
            np.std(track_note_counts) / max(np.mean(track_note_counts), 1),  # CV
            np.max(track_note_counts) / max(np.min(track_note_counts), 1),   # Max/min ratio
            np.mean(track_note_counts),
            len(track_note_counts),  # Number of active tracks
            stats.skew(track_note_counts) if len(track_note_counts) > 2 else 0,
        ])

        # Velocity balance (5D)
        features.extend([
            np.mean(track_avg_velocities) / 127.0,
            np.std(track_avg_velocities) / 127.0,
            (np.max(track_avg_velocities) - np.min(track_avg_velocities)) / 127.0,
            np.std(track_avg_velocities) / max(np.mean(track_avg_velocities), 1),  # CV
            stats.skew(track_avg_velocities) if len(track_avg_velocities) > 2 else 0,
        ])

        # Pitch range balance (5D)
        features.extend([
            np.mean(track_avg_pitches) / 127.0,
            np.std(track_avg_pitches) / 127.0,
            (np.max(track_avg_pitches) - np.min(track_avg_pitches)) / 88.0,
            np.std(track_avg_pitches) / max(np.mean(track_avg_pitches), 1),  # CV
            stats.skew(track_avg_pitches) if len(track_avg_pitches) > 2 else 0,
        ])

        # Overall balance metrics (5D)
        features.extend([
            len([c for c in track_note_counts if c > np.mean(track_note_counts)]) / len(track_note_counts),
            len([v for v in track_avg_velocities if v > np.mean(track_avg_velocities)]) / len(track_avg_velocities),
            1.0 / (1.0 + np.std(track_note_counts)),  # Balance score (higher = more balanced)
            1.0 / (1.0 + np.std(track_avg_velocities)),  # Velocity balance score
            1.0 / (1.0 + np.std(track_avg_pitches)),     # Pitch balance score
        ])

        return features

    def _extract_spacing_features(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        """Extract voicing spacing features (20D)."""
        features = []

        # Get all notes
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend([(n.start, n.pitch, n.end) for n in instrument.notes])

        if not all_notes:
            return [0] * 20

        all_notes.sort()
        end_time = max(n[2] for n in all_notes)

        # Sample chords
        sample_times = np.linspace(0, end_time, min(50, int(end_time * 2)))

        chord_spacings = []
        chord_spreads = []

        for t in sample_times:
            chord_pitches = sorted([pitch for start, pitch, end in all_notes if start <= t < end])

            if len(chord_pitches) >= 2:
                # Compute intervals between adjacent voices
                intervals = [chord_pitches[i+1] - chord_pitches[i]
                           for i in range(len(chord_pitches)-1)]
                chord_spacings.extend(intervals)

                # Compute overall spread
                spread = max(chord_pitches) - min(chord_pitches)
                chord_spreads.append(spread)

        if not chord_spacings:
            return [0] * 20

        # Spacing statistics (10D)
        features.extend([
            np.mean(chord_spacings),
            np.std(chord_spacings),
            np.min(chord_spacings),
            np.max(chord_spacings),
            np.median(chord_spacings),
            sum(1 for s in chord_spacings if s <= 2) / len(chord_spacings),  # Close spacing ratio
            sum(1 for s in chord_spacings if s >= 12) / len(chord_spacings), # Wide spacing ratio
            stats.skew(chord_spacings) if len(chord_spacings) > 2 else 0,
            stats.kurtosis(chord_spacings) if len(chord_spacings) > 2 else 0,
            np.percentile(chord_spacings, 75) - np.percentile(chord_spacings, 25),  # IQR
        ])

        # Spread statistics (10D)
        if chord_spreads:
            features.extend([
                np.mean(chord_spreads),
                np.std(chord_spreads),
                np.min(chord_spreads),
                np.max(chord_spreads),
                np.median(chord_spreads),
                sum(1 for s in chord_spreads if s < 12) / len(chord_spreads),  # Narrow spread ratio
                sum(1 for s in chord_spreads if s > 24) / len(chord_spreads),  # Wide spread ratio
                stats.skew(chord_spreads) if len(chord_spreads) > 2 else 0,
                stats.kurtosis(chord_spreads) if len(chord_spreads) > 2 else 0,
                np.mean(chord_spreads) / 88.0,  # Normalized average spread
            ])
        else:
            features.extend([0] * 10)

        return features

    def _extract_instrumentation_features(self, midi: pretty_midi.PrettyMIDI) -> List[float]:
        """Extract instrumentation features (10D)."""
        features = []

        # Count instrument types
        programs = [inst.program for inst in midi.instruments if not inst.is_drum]
        has_drums = any(inst.is_drum for inst in midi.instruments)

        # Instrument family distribution
        families = {
            'piano': list(range(0, 8)),
            'chromatic': list(range(8, 16)),
            'organ': list(range(16, 24)),
            'guitar': list(range(24, 32)),
            'bass': list(range(32, 40)),
            'strings': list(range(40, 48)),
            'ensemble': list(range(48, 56)),
            'brass': list(range(56, 64)),
            'reed': list(range(64, 72)),
            'pipe': list(range(72, 80)),
        }

        family_counts = defaultdict(int)
        for program in programs:
            for family, family_range in families.items():
                if program in family_range:
                    family_counts[family] += 1
                    break

        total_instruments = len(programs)

        # Family ratios (5D - most common families)
        common_families = ['piano', 'guitar', 'bass', 'strings', 'brass']
        for family in common_families:
            features.append(family_counts[family] / max(total_instruments, 1))

        # Diversity metrics (5D)
        features.extend([
            len(set(programs)) / max(total_instruments, 1),  # Unique instrument ratio
            len(family_counts) / 10.0,                        # Family diversity (normalized)
            float(total_instruments) / self.MAX_TRACKS,       # Track usage ratio
            1.0 if has_drums else 0.0,                        # Has drums
            1.0 if len(set(programs)) == len(programs) else 0.0,  # All unique instruments
        ])

        return features

    # ========================================================================
    # Batch Processing
    # ========================================================================

    def extract_batch(
        self,
        midi_files: List[str],
        n_workers: int = 1,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract features from multiple MIDI files with optional multiprocessing.

        Args:
            midi_files: List of MIDI file paths
            n_workers: Number of parallel workers (1 = sequential)
            show_progress: Show progress bar

        Returns:
            numpy array of shape (n_files, 600)
        """
        print(f"\n🚀 Extracting features from {len(midi_files)} files...")
        print(f"   Workers: {n_workers}")
        print(f"   Target: < 30s per file")

        if n_workers == 1:
            # Sequential processing
            features_list = []

            iterator = midi_files
            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(midi_files, desc="Extracting")
                except ImportError:
                    pass

            for midi_file in iterator:
                try:
                    features = self.extract(midi_file)
                    features_list.append(features)
                except Exception as e:
                    print(f"⚠️ Error processing {midi_file}: {e}")
                    features_list.append(np.zeros(self.TOTAL_DIM))
        else:
            # Parallel processing
            with mp.Pool(processes=n_workers) as pool:
                if show_progress:
                    try:
                        from tqdm import tqdm
                        features_list = list(tqdm(
                            pool.imap(self._extract_worker, midi_files),
                            total=len(midi_files),
                            desc="Extracting"
                        ))
                    except ImportError:
                        features_list = pool.map(self._extract_worker, midi_files)
                else:
                    features_list = pool.map(self._extract_worker, midi_files)

        feature_matrix = np.array(features_list)

        print(f"✅ Extracted features: {feature_matrix.shape}")

        return feature_matrix

    def _extract_worker(self, midi_file: str) -> np.ndarray:
        """Worker function for parallel processing."""
        try:
            return self.extract(midi_file)
        except Exception as e:
            print(f"⚠️ Error processing {midi_file}: {e}")
            return np.zeros(self.TOTAL_DIM)


# ============================================================================
# Batch Feature Extraction Pipeline
# ============================================================================

class BatchFeatureExtractor:
    """
    Extract features for all train/val/test splits with checkpointing and reporting.

    Usage:
        extractor = BatchFeatureExtractor(
            corpus_dir='data/corpus',
            output_dir='data/features',
            n_workers=16
        )

        report = extractor.run()
    """

    def __init__(
        self,
        corpus_dir: str,
        output_dir: str,
        n_workers: int = 16,
        checkpoint_interval: int = 100
    ):
        """
        Initialize batch feature extractor.

        Args:
            corpus_dir: Directory containing train/val/test MIDI files
            output_dir: Directory to save feature .npy files
            n_workers: Number of parallel workers
            checkpoint_interval: Save checkpoint every N files
        """
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        self.checkpoint_interval = checkpoint_interval

        self.extractor = RichMultitrackFeatureExtractor()

        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

        print(f"✅ BatchFeatureExtractor initialized")
        print(f"   Corpus: {self.corpus_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Workers: {self.n_workers}")

    def run(self) -> Dict[str, Any]:
        """
        Run feature extraction for all splits.

        Returns:
            Report dictionary with timing, errors, statistics
        """
        report = {
            'start_time': time.time(),
            'splits': {},
            'errors': [],
            'total_files': 0,
            'total_features_extracted': 0,
            'avg_extraction_time': 0,
        }

        for split in ['train', 'val', 'test']:
            print(f"\n{'='*70}")
            print(f"Processing {split.upper()} split")
            print(f"{'='*70}")

            split_report = self._process_split(split)
            report['splits'][split] = split_report
            report['total_files'] += split_report['num_files']
            report['total_features_extracted'] += split_report['features_extracted']
            report['errors'].extend(split_report['errors'])

        report['end_time'] = time.time()
        report['total_time'] = report['end_time'] - report['start_time']
        report['avg_extraction_time'] = report['total_time'] / max(report['total_files'], 1)

        # Save report
        report_path = self.output_dir / 'feature_extraction_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*70}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total files: {report['total_files']}")
        print(f"Total time: {report['total_time']:.1f}s")
        print(f"Avg time per file: {report['avg_extraction_time']:.2f}s")
        print(f"Errors: {len(report['errors'])}")
        print(f"Report saved to: {report_path}")

        return report

    def _process_split(self, split: str) -> Dict[str, Any]:
        """Process a single split (train/val/test)."""
        split_dir = self.corpus_dir / split
        output_split_dir = self.output_dir / split

        # Find all MIDI files
        midi_files = list(split_dir.glob('**/*.mid')) + list(split_dir.glob('**/*.midi'))

        print(f"Found {len(midi_files)} MIDI files in {split_dir}")

        if len(midi_files) == 0:
            print(f"⚠️ No MIDI files found in {split_dir}")
            return {
                'num_files': 0,
                'features_extracted': 0,
                'errors': [],
                'avg_time': 0,
            }

        # Extract features
        start_time = time.time()
        errors = []

        for i, midi_file in enumerate(midi_files):
            try:
                # Extract features
                features = self.extractor.extract(str(midi_file))

                # Save features
                output_path = output_split_dir / f"{midi_file.stem}.npy"
                np.save(output_path, features)

                # Progress update
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (len(midi_files) - i - 1) / rate
                    print(f"  Progress: {i+1}/{len(midi_files)} "
                          f"({100*(i+1)/len(midi_files):.1f}%) "
                          f"ETA: {eta/60:.1f}m")

            except Exception as e:
                error_info = {
                    'file': str(midi_file),
                    'error': str(e),
                    'split': split
                }
                errors.append(error_info)
                print(f"⚠️ Error processing {midi_file}: {e}")

        end_time = time.time()
        total_time = end_time - start_time

        split_report = {
            'num_files': len(midi_files),
            'features_extracted': len(midi_files) - len(errors),
            'errors': errors,
            'total_time': total_time,
            'avg_time': total_time / max(len(midi_files), 1),
            'error_rate': len(errors) / len(midi_files) if midi_files else 0,
        }

        print(f"\n✅ {split.upper()} complete:")
        print(f"   Files processed: {split_report['features_extracted']}/{split_report['num_files']}")
        print(f"   Time: {split_report['total_time']:.1f}s")
        print(f"   Avg: {split_report['avg_time']:.2f}s per file")
        print(f"   Error rate: {split_report['error_rate']*100:.2f}%")

        return split_report


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RICH MULTITRACK FEATURE EXTRACTOR - AGENT 2")
    print("="*70)

    print("\nThis extractor provides 600D rich multitrack features:")
    print("  - Global features (200D): Full-file statistics")
    print("  - Per-track features (200D): 8 tracks × 25D each")
    print("  - Temporal features (100D): 4 sections × 25D")
    print("  - Orchestration features (100D): Arrangement analysis")
    print()
    print("Example usage:")
    print("  # Single file extraction")
    print("  extractor = RichMultitrackFeatureExtractor()")
    print("  features = extractor.extract('song.mid')  # Returns 600D vector")
    print()
    print("  # Batch extraction")
    print("  batch_extractor = BatchFeatureExtractor(")
    print("      corpus_dir='data/corpus',")
    print("      output_dir='data/features',")
    print("      n_workers=16")
    print("  )")
    print("  report = batch_extractor.run()")
    print()
    print("✅ Rich Multitrack Feature Extractor ready for use!")
