#!/usr/bin/env python3
"""
Factored MDL Pipeline - Full Multi-Factor Encoding
===================================================

This pipeline properly factors MIDI into all components:
- pitch_class: 0-11 (pitch mod 12)
- octave: 0-9 (pitch // 12)
- velocity: 0-7 (quantized)
- duration: timesteps
- onset_times: timing
- rhythm: binary onset pattern

Transform discovery uses product-space transforms:
    T = T_pitch × T_rhythm × T_velocity × T_duration

This enables discovering:
- Same rhythm, different pitches (harmonization)
- Same pitches, different rhythm (rhythmic variation)
- Velocity scaling (dynamics)
- Duration stretching (tempo)

Author: Factored Pipeline
"""

import os
import sys
import time
import json
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from core.primitives import (
    CompoundTransform,
    enumerate_compounds,
    apply_compound,
    find_transform,
)
from discovery.primitive_mdl import (
    extract_canonical_pairs,
    mine_transform_relations,
    mine_transform_relations_gpu,
    mine_cross_track_pairs_gpu,
    select_vocabulary_mdl,
)

# Level 3: Meta-pattern discovery
from scripts.level3_meta_patterns import (
    build_transform_lookup_gpu,
    extract_transform_sequences_gpu,
    run_meta_repair_gpu,
    interpret_meta_patterns,
)


# =============================================================================
# FACTORED NOTE REPRESENTATION
# =============================================================================

@dataclass
class FactoredNote:
    """A single note with all factors."""
    onset: int          # Onset time in ticks
    pitch: int          # MIDI pitch 0-127
    pitch_class: int    # pitch % 12
    octave: int         # pitch // 12
    velocity: int       # Quantized 0-7
    duration: int       # Duration in ticks


@dataclass
class FactoredTrack:
    """A track with factored note representation."""
    piece_id: str
    track_id: int
    notes: List[FactoredNote]

    # Factor arrays for efficient processing
    pitch_classes: np.ndarray    # (N,) int 0-11
    octaves: np.ndarray          # (N,) int 0-9
    velocities: np.ndarray       # (N,) int 0-7
    durations: np.ndarray        # (N,) int ticks
    onsets: np.ndarray           # (N,) int ticks

    # Derived
    rhythm_ioi: np.ndarray       # (N-1,) inter-onset intervals

    # Original MIDI pitches (for playback reconstruction) - optional with default
    pitches: np.ndarray = None   # (N,) int 0-127, original MIDI pitch

    # GM program number for instrument identity (0-127)
    gm_program: int = 0          # Default: Acoustic Grand Piano

    # Drum track flag (detected from MIDI channel 9/10)
    is_drum: bool = False        # True if track is drums (channel 9)

    def __len__(self):
        return len(self.notes)

    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        result = {
            'piece_id': self.piece_id,
            'track_id': self.track_id,
            'pitch_classes': self.pitch_classes.tolist(),
            'octaves': self.octaves.tolist(),
            'velocities': self.velocities.tolist(),
            'durations': self.durations.tolist(),
            'onsets': self.onsets.tolist(),
            'is_drum': self.is_drum,
            'gm_program': self.gm_program,
        }
        # Include original pitches if available
        if self.pitches is not None:
            result['pitches'] = self.pitches.tolist()
        return result


@dataclass
class PatternOccurrence:
    """A single occurrence of a pattern in the corpus."""
    piece_id: str       # Which piece
    track_id: int       # Which track
    onset_time: int     # When in the piece (ticks)
    position: int       # Position in sequence (note index)
    # Timing offsets for lossless reconstruction
    tau_offset: int = 480       # Base IOI in ticks (for rhythm scaling)
    duration_offset: int = 480  # Base duration in ticks
    v_offset: int = 4           # Base velocity (0-7 quantized)


@dataclass
class FactoredPattern:
    """A pattern with all factors preserved + metadata for meta-patterns."""
    pattern_id: int
    rule_id: str

    # All factors
    pitch_classes: List[int]     # 0-11
    octaves: List[int]           # 0-9 (or relative)
    velocities: List[int]        # 0-7
    durations: List[int]         # Relative durations
    rhythm_ioi: List[int]        # Inter-onset intervals

    # ALL occurrences of this pattern (for meta-pattern discovery)
    occurrences: List[PatternOccurrence] = field(default_factory=list)

    def __len__(self):
        return len(self.pitch_classes)

    # =========================================================================
    # NORMALIZED CONTOURS (T+τ+v normalization)
    # =========================================================================

    @property
    def pitch_intervals(self) -> List[int]:
        """
        T-normalized pitch contour: intervals from first note.
        [0, +4, +3] means C-E-G or D-F#-A (same interval structure).
        """
        if len(self.pitch_classes) < 2:
            return [0] if self.pitch_classes else []
        base = self.pitch_classes[0]
        return [(pc - base) % 12 for pc in self.pitch_classes]

    @property
    def rhythm_ratios(self) -> List[float]:
        """
        τ-normalized rhythm contour: IOI ratios relative to first IOI.
        [1.0, 0.5, 0.5] means quarter-eighth-eighth at any tempo.
        """
        if len(self.rhythm_ioi) < 1:
            return []
        base_ioi = self.rhythm_ioi[0] if self.rhythm_ioi[0] > 0 else 480
        return [float(ioi) / base_ioi for ioi in self.rhythm_ioi]

    @property
    def velocity_ratios(self) -> List[float]:
        """
        v-normalized velocity contour: ratios relative to first velocity.
        [1.0, 0.8, 0.6] means accent pattern independent of base velocity.
        """
        if len(self.velocities) < 1:
            return []
        base_vel = self.velocities[0] if self.velocities[0] > 0 else 4
        return [float(v) / base_vel for v in self.velocities]

    @property
    def T_offset(self) -> int:
        """Base pitch class for reconstruction (0-11)."""
        return self.pitch_classes[0] if self.pitch_classes else 0

    @property
    def tau_offset(self) -> int:
        """Base IOI for rhythm reconstruction (ticks)."""
        return self.rhythm_ioi[0] if self.rhythm_ioi else 480

    @property
    def v_offset(self) -> int:
        """Base velocity for reconstruction (0-7 quantized)."""
        return self.velocities[0] if self.velocities else 4

    @property
    def piece_id(self) -> Optional[str]:
        """First occurrence's piece_id (for backwards compat)."""
        return self.occurrences[0].piece_id if self.occurrences else None

    @property
    def track_id(self) -> Optional[int]:
        """First occurrence's track_id (for backwards compat)."""
        return self.occurrences[0].track_id if self.occurrences else None

    @property
    def onset_time(self) -> Optional[int]:
        """First occurrence's onset_time (for backwards compat)."""
        return self.occurrences[0].onset_time if self.occurrences else None

    def to_dict(self) -> Dict:
        # Use stored ratios if available (from extract_factored_canonical_patterns)
        rhythm_ratios = getattr(self, '_rhythm_ratios', None) or self.rhythm_ratios
        duration_ratios = getattr(self, '_duration_ratios', None) or [1.0] * len(self.durations)
        velocity_ratios = getattr(self, '_velocity_ratios', None) or self.velocity_ratios

        return {
            'pattern_id': self.pattern_id,
            'rule_id': self.rule_id,
            # Raw values
            'pitch_classes': self.pitch_classes,
            'octaves': self.octaves,
            'velocities': self.velocities,
            'durations': self.durations,
            'rhythm_ioi': self.rhythm_ioi,
            # Normalized contours (T+τ+v) - these are the KEY for lossless reconstruction
            'pitch_intervals': self.pitch_intervals,
            'rhythm_ratios': rhythm_ratios,
            'duration_ratios': duration_ratios,
            'velocity_ratios': velocity_ratios,
            # Reconstruction offsets (from first occurrence)
            'T_offset': self.T_offset,
            'tau_offset': self.tau_offset,
            'v_offset': self.v_offset,
            # All occurrences with per-occurrence timing offsets
            'occurrences': [
                {
                    'piece_id': occ.piece_id,
                    'track_id': occ.track_id,
                    'onset_time': occ.onset_time,
                    'position': occ.position,
                    # Per-occurrence timing offsets for lossless reconstruction
                    'tau_offset': getattr(occ, 'tau_offset', 480),
                    'duration_offset': getattr(occ, 'duration_offset', 480),
                    'v_offset': getattr(occ, 'v_offset', 4),
                }
                for occ in self.occurrences
            ],
        }


@dataclass
class PipelineStats:
    """Statistics from pipeline run."""
    n_files_loaded: int = 0
    n_files_failed: int = 0
    n_tracks: int = 0
    n_notes: int = 0
    n_pitch_sequences: int = 0
    n_grammar_rules: int = 0
    n_canonical_patterns: int = 0
    n_transform_vocabulary: int = 0
    d24_coverage: float = 0.0
    mdl_coverage: float = 0.0
    total_time: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# DRUM DETECTION HEURISTICS
# =============================================================================

def is_likely_drum_track(pitches: List[int]) -> bool:
    """
    Heuristically determine if a track is drums based on pitch distribution.

    Drums characteristics:
    - Use a fixed set of pitches (drum mappings, typically 35-81 in GM)
    - Low pitch variance within common drum patterns
    - Pitches cluster around specific drum kit notes
    - No melodic interval patterns

    Melodic tracks (including bass on wrong channel):
    - Use consecutive/scale-based intervals
    - Higher pitch variety with melodic patterns
    - May span wide range but with melodic movement
    """
    if len(pitches) < 5:
        return False

    pitches_arr = np.array(pitches)

    # Check interval distribution - drums have no melodic intervals
    intervals = np.diff(pitches_arr)
    unique_intervals = len(set(intervals.tolist()))

    # Melodic tracks have varied intervals (1, 2, 3, 4, 5, 7, 12, etc.)
    # Drum tracks tend to have repetitive "intervals" (same drum hit)
    interval_variety = unique_intervals / len(intervals) if len(intervals) > 0 else 0

    # Count unique pitches
    unique_pitches = len(set(pitches_arr.tolist()))
    pitch_variety = unique_pitches / len(pitches_arr)

    # Drum-like: low variety in both intervals and pitches
    # Most drum patterns use 5-15 different drum sounds
    if unique_pitches > 20 and interval_variety > 0.3:
        # Likely melodic - too many unique pitches and varied intervals
        return False

    # Check for melodic scale patterns (small consecutive intervals)
    small_intervals = np.abs(intervals)
    melodic_intervals = np.sum((small_intervals >= 1) & (small_intervals <= 5))
    melodic_ratio = melodic_intervals / len(intervals) if len(intervals) > 0 else 0

    if melodic_ratio > 0.5 and unique_pitches > 10:
        # Likely melodic - many stepwise/small intervals
        return False

    # Check pitch range - bass has narrow range, drums have specific GM range
    pitch_range = pitches_arr.max() - pitches_arr.min()

    # Bass typically: 28-55 (E1 to G3), narrow range, melodic intervals
    # Drums: 35-81 (GM spec), but clustered in specific notes
    if pitch_range < 24 and melodic_ratio > 0.4:
        # Narrow range with melodic movement = likely bass/melodic, not drums
        return False

    # Default: if on channel 9 and doesn't look melodic, treat as drums
    return True


# =============================================================================
# FACTORED MIDI LOADER
# =============================================================================

def load_midi_factored(midi_path: str) -> Optional[List[FactoredTrack]]:
    """
    Load MIDI file and extract FULL factored representation per track.

    Returns list of FactoredTrack with:
        - pitch_class: 0-11
        - octave: 0-9
        - velocity: 0-7 (quantized)
        - duration: ticks
        - onset: ticks
        - rhythm_ioi: inter-onset intervals
        - gm_program: GM instrument number (0-127)
    """
    try:
        import mido

        midi = mido.MidiFile(midi_path)
        piece_id = Path(midi_path).stem
        ticks_per_beat = midi.ticks_per_beat

        # Key: track_idx -> list of notes (each MIDI track = one instrument)
        tracks = defaultdict(list)
        # Key: track_idx -> GM program (first program_change in each track)
        track_programs = {}
        # Tracks that have notes on channel 9 (drums)
        drum_tracks = set()

        for track_idx, track in enumerate(midi.tracks):
            current_time = 0
            active = {}  # (channel, pitch) -> (onset, velocity)

            for msg in track:
                current_time += msg.time

                # Capture first program change in this track
                if msg.type == 'program_change' and track_idx not in track_programs:
                    track_programs[track_idx] = msg.program

                if msg.type == 'note_on' and msg.velocity > 0:
                    if msg.channel == 9:
                        drum_tracks.add(track_idx)
                    key = (msg.channel, msg.note)
                    active[key] = (current_time, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.channel, msg.note)
                    if key in active:
                        onset, velocity = active[key]
                        duration = current_time - onset

                        # Quantize velocity to 8 levels (0-7)
                        vel_quantized = min(7, velocity // 16)

                        tracks[track_idx].append(FactoredNote(
                            onset=onset,
                            pitch=msg.note,
                            pitch_class=msg.note % 12,
                            octave=msg.note // 12,
                            velocity=vel_quantized,
                            duration=max(1, duration),  # Ensure positive
                        ))
                        del active[key]

        # Build FactoredTrack objects - one per MIDI track
        results = []
        for track_idx, notes in tracks.items():
            if len(notes) < 3:  # Skip very short tracks
                continue

            # Sort by onset
            notes = sorted(notes, key=lambda n: n.onset)

            # Detect drum track from channel 9 (MIDI channel 10 in 1-indexed)
            is_drum = track_idx in drum_tracks

            # Build factor arrays
            pitch_classes = np.array([n.pitch_class for n in notes], dtype=np.int32)
            octaves = np.array([n.octave for n in notes], dtype=np.int32)
            velocities = np.array([n.velocity for n in notes], dtype=np.int32)
            durations = np.array([n.duration for n in notes], dtype=np.int32)
            onsets = np.array([n.onset for n in notes], dtype=np.int64)

            # Preserve original MIDI pitches for playback (0-127)
            pitches = np.array([n.pitch for n in notes], dtype=np.int32)

            # Compute inter-onset intervals (rhythm)
            if len(onsets) > 1:
                rhythm_ioi = np.diff(onsets).astype(np.int64)
            else:
                rhythm_ioi = np.array([], dtype=np.int64)

            # Get GM program from this track's program change
            # Channel 9 = drums = GM program 128 (special marker)
            if is_drum:
                gm_program = 128  # Special marker for drums
            else:
                gm_program = track_programs.get(track_idx, 0)  # Default: piano

            results.append(FactoredTrack(
                piece_id=piece_id,
                track_id=track_idx,
                notes=notes,
                pitch_classes=pitch_classes,
                octaves=octaves,
                velocities=velocities,
                durations=durations,
                onsets=onsets,
                pitches=pitches,
                rhythm_ioi=rhythm_ioi,
                gm_program=gm_program,
                is_drum=is_drum,
            ))

        return results if results else None

    except Exception as e:
        return None


# =============================================================================
# MULTI-FACTOR GRAMMAR INDUCTION
# =============================================================================

def encode_factored_token(pc: int, octave: int, velocity: int, duration_bucket: int) -> int:
    """
    Encode a factored note as a single token for grammar induction.

    Token = pc + 12*octave + 12*10*velocity + 12*10*8*duration_bucket

    This gives us 12 * 10 * 8 * 8 = 7680 possible tokens.
    """
    # Clamp values
    pc = pc % 12
    octave = min(9, max(0, octave))
    velocity = min(7, max(0, velocity))
    duration_bucket = min(7, max(0, duration_bucket))

    return pc + 12 * octave + 120 * velocity + 960 * duration_bucket


def decode_factored_token(token: int) -> Tuple[int, int, int, int]:
    """Decode token back to (pitch_class, octave, velocity, duration_bucket)."""
    pc = token % 12
    octave = (token // 12) % 10
    velocity = (token // 120) % 8
    duration_bucket = (token // 960) % 8
    return (pc, octave, velocity, duration_bucket)


def bucket_duration(duration: int, ticks_per_beat: int = 480) -> int:
    """
    Bucket duration into 8 categories:
    0: very short (< 1/16 beat)
    1: sixteenth
    2: eighth
    3: quarter
    4: half
    5: whole
    6: double-whole
    7: very long
    """
    ratio = duration / ticks_per_beat
    if ratio < 0.0625:
        return 0
    elif ratio < 0.125:
        return 1
    elif ratio < 0.25:
        return 2
    elif ratio < 0.5:
        return 3
    elif ratio < 1.0:
        return 4
    elif ratio < 2.0:
        return 5
    elif ratio < 4.0:
        return 6
    else:
        return 7


def factored_track_to_tokens(track: FactoredTrack, ticks_per_beat: int = 480) -> List[int]:
    """Convert a factored track to a sequence of encoded tokens."""
    tokens = []
    for i in range(len(track)):
        dur_bucket = bucket_duration(int(track.durations[i]), ticks_per_beat)
        token = encode_factored_token(
            int(track.pitch_classes[i]),
            int(track.octaves[i]),
            int(track.velocities[i]),
            dur_bucket
        )
        tokens.append(token)
    return tokens


# =============================================================================
# T+τ NORMALIZED RE-PAIR (Transposition & Tempo Invariant)
# =============================================================================

def compute_pitch_interval(pc1: int, pc2: int) -> int:
    """Compute pitch interval (0-11) between two pitch classes."""
    return (pc2 - pc1) % 12


def compute_rhythm_ratio_bucket(ioi1: int, ioi2: int) -> int:
    """
    Compute rhythm ratio bucket for τ-normalization.

    Ratios are quantized into 16 buckets:
    - 0-7: subdivisions (0.125, 0.25, 0.33, 0.5, 0.67, 0.75, 0.875, 1.0)
    - 8-15: multiples (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, >8.0)
    """
    if ioi1 <= 0:
        return 7  # Default to 1.0 ratio

    ratio = ioi2 / ioi1

    # Subdivision ratios
    if ratio <= 0.1875:
        return 0  # ~0.125 (eighth of the base)
    elif ratio <= 0.3125:
        return 1  # ~0.25 (quarter)
    elif ratio <= 0.415:
        return 2  # ~0.33 (triplet)
    elif ratio <= 0.585:
        return 3  # ~0.5 (half)
    elif ratio <= 0.71:
        return 4  # ~0.67 (dotted half of triplet)
    elif ratio <= 0.8125:
        return 5  # ~0.75 (dotted half)
    elif ratio <= 0.9375:
        return 6  # ~0.875
    elif ratio <= 1.25:
        return 7  # ~1.0 (same)
    # Multiple ratios
    elif ratio <= 1.75:
        return 8  # ~1.5 (dotted)
    elif ratio <= 2.25:
        return 9  # ~2.0 (double)
    elif ratio <= 2.75:
        return 10  # ~2.5
    elif ratio <= 3.5:
        return 11  # ~3.0 (triplet multiple)
    elif ratio <= 5.0:
        return 12  # ~4.0 (quadruple)
    elif ratio <= 7.0:
        return 13  # ~6.0
    elif ratio <= 12.0:
        return 14  # ~8.0
    else:
        return 15  # >8.0


def encode_normalized_pair(
    interval: int,
    rhythm_bucket: int,
    velocity_delta_bucket: int = 0
) -> int:
    """
    Encode a normalized pair as a single integer for counting.

    Encoding: interval + 12 * rhythm_bucket + 12 * 16 * vel_bucket
    This gives 12 * 16 * 8 = 1536 possible normalized pair types.
    """
    interval = interval % 12
    rhythm_bucket = min(15, max(0, rhythm_bucket))
    velocity_delta_bucket = min(7, max(0, velocity_delta_bucket))

    return interval + 12 * rhythm_bucket + 192 * velocity_delta_bucket


def decode_normalized_pair(encoded: int) -> Tuple[int, int, int]:
    """Decode normalized pair back to (interval, rhythm_bucket, vel_bucket)."""
    interval = encoded % 12
    rhythm_bucket = (encoded // 12) % 16
    vel_bucket = (encoded // 192) % 8
    return (interval, rhythm_bucket, vel_bucket)


@dataclass
class NormalizedContour:
    """
    A pattern's normalized contour (T+τ+v invariant).

    Two patterns with the same contour are related by:
    - Transposition (T): same intervals, different starting pitch
    - Tempo scaling (τ): same rhythm ratios, different base IOI
    - Dynamics scaling (v): same velocity ratios, different base velocity
    """
    # Pitch intervals from first note (length N)
    pitch_intervals: Tuple[int, ...]

    # Rhythm ratio buckets (length N-1, between consecutive notes)
    rhythm_buckets: Tuple[int, ...]

    # Velocity delta buckets (length N-1, optional)
    velocity_buckets: Tuple[int, ...]

    def __hash__(self):
        return hash((self.pitch_intervals, self.rhythm_buckets, self.velocity_buckets))

    def __eq__(self, other):
        if not isinstance(other, NormalizedContour):
            return False
        return (self.pitch_intervals == other.pitch_intervals and
                self.rhythm_buckets == other.rhythm_buckets and
                self.velocity_buckets == other.velocity_buckets)

    def to_tuple(self) -> Tuple:
        """Convert to hashable tuple for dictionary keys."""
        return (self.pitch_intervals, self.rhythm_buckets, self.velocity_buckets)

    @classmethod
    def from_track_slice(
        cls,
        track: FactoredTrack,
        start: int,
        length: int
    ) -> 'NormalizedContour':
        """Extract normalized contour from a track slice."""
        end = start + length

        # Pitch intervals from first note
        base_pc = int(track.pitch_classes[start])
        pitch_intervals = tuple(
            (int(track.pitch_classes[i]) - base_pc) % 12
            for i in range(start, end)
        )

        # Rhythm ratio buckets (between consecutive notes)
        if len(track.rhythm_ioi) > start and length > 1:
            base_ioi = int(track.rhythm_ioi[start]) if start < len(track.rhythm_ioi) else 480
            base_ioi = max(1, base_ioi)  # Avoid division by zero
            rhythm_buckets = []
            for i in range(start, min(end - 1, len(track.rhythm_ioi))):
                ioi = int(track.rhythm_ioi[i])
                bucket = compute_rhythm_ratio_bucket(base_ioi, ioi)
                rhythm_buckets.append(bucket)
            # Pad if needed
            while len(rhythm_buckets) < length - 1:
                rhythm_buckets.append(7)  # Default 1.0 ratio
            rhythm_buckets = tuple(rhythm_buckets)
        else:
            rhythm_buckets = tuple([7] * max(0, length - 1))

        # Velocity delta buckets (simplified - just track changes)
        if length > 1:
            base_vel = int(track.velocities[start])
            base_vel = max(1, base_vel)
            velocity_buckets = []
            for i in range(start + 1, end):
                vel = int(track.velocities[i])
                # Simple delta bucket: 0-7 based on ratio
                ratio = vel / base_vel if base_vel > 0 else 1.0
                if ratio <= 0.5:
                    bucket = 0
                elif ratio <= 0.75:
                    bucket = 1
                elif ratio <= 0.9:
                    bucket = 2
                elif ratio <= 1.1:
                    bucket = 3
                elif ratio <= 1.25:
                    bucket = 4
                elif ratio <= 1.5:
                    bucket = 5
                elif ratio <= 2.0:
                    bucket = 6
                else:
                    bucket = 7
                velocity_buckets.append(bucket)
            velocity_buckets = tuple(velocity_buckets)
        else:
            velocity_buckets = tuple()

        return cls(pitch_intervals, rhythm_buckets, velocity_buckets)


@dataclass
class NormalizedOccurrence:
    """An occurrence of a normalized pattern with its reconstruction offsets."""
    piece_id: str
    track_id: int
    position: int           # Note index in track
    onset_time: int         # Absolute onset time

    # Reconstruction offsets (to go from contour back to raw values)
    T_offset: int           # Base pitch class (0-11)
    tau_offset: int         # Base IOI (ticks)
    v_offset: int           # Base velocity (0-7)
    octave_offset: int      # Base octave (0-9)


def run_normalized_repair(
    tracks: List[FactoredTrack],
    min_length: int = 3,
    max_length: int = 32,
    min_count: int = 2,
    max_rules: int = 5000,
    verbose: bool = True
) -> Dict:
    """
    Run T+τ normalized Re-Pair on factored tracks.

    This discovers patterns that are the SAME under:
    - Transposition (T): C-E-G and D-F#-A are the same pattern [0,4,7]
    - Tempo scaling (τ): quarter-eighth-eighth at 120bpm and 60bpm are the same
    - Velocity scaling (v): f-mf-p and ff-f-mf are the same contour

    Returns:
        Dict with 'rules' containing normalized patterns and all their occurrences
    """
    import time
    t0 = time.time()

    if verbose:
        total_notes = sum(len(t) for t in tracks)
        print(f"\n[T+τ Normalized Re-Pair] {len(tracks)} tracks, {total_notes:,} notes", flush=True)
        print(f"  Length range: {min_length}-{max_length}, min_count: {min_count}", flush=True)

    # =========================================================================
    # STEP 1: Build contour index - find all n-grams by normalized contour
    # =========================================================================
    # contour -> list of NormalizedOccurrence
    contour_occurrences: Dict[NormalizedContour, List[NormalizedOccurrence]] = defaultdict(list)

    if verbose:
        print(f"  Building normalized contour index...", flush=True)

    for track_idx, track in enumerate(tracks):
        track_len = len(track)

        for length in range(min_length, min(max_length + 1, track_len + 1)):
            for start in range(track_len - length + 1):
                # Extract normalized contour
                contour = NormalizedContour.from_track_slice(track, start, length)

                # Record occurrence with offsets
                occurrence = NormalizedOccurrence(
                    piece_id=track.piece_id,
                    track_id=track.track_id,
                    position=start,
                    onset_time=int(track.onsets[start]),
                    T_offset=int(track.pitch_classes[start]),
                    tau_offset=int(track.rhythm_ioi[start]) if start < len(track.rhythm_ioi) else 480,
                    v_offset=int(track.velocities[start]),
                    octave_offset=int(track.octaves[start]),
                )

                contour_occurrences[contour].append(occurrence)

    if verbose:
        print(f"    Found {len(contour_occurrences):,} unique contours", flush=True)

    # =========================================================================
    # STEP 2: Filter by frequency and select top patterns
    # =========================================================================
    # Sort by count, keep top max_rules
    sorted_contours = sorted(
        contour_occurrences.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    # Filter by min_count
    frequent_contours = [
        (c, occs) for c, occs in sorted_contours
        if len(occs) >= min_count
    ]

    if verbose:
        print(f"    {len(frequent_contours):,} contours with count >= {min_count}", flush=True)

    # Take top max_rules
    selected_contours = frequent_contours[:max_rules]

    if verbose:
        print(f"    Selected top {len(selected_contours)} patterns", flush=True)

    # =========================================================================
    # STEP 3: Build output in v24 format
    # =========================================================================
    rules = {}

    for rule_idx, (contour, occurrences) in enumerate(selected_contours):
        rule_id = str(rule_idx)

        # Use first occurrence for canonical raw values
        first_occ = occurrences[0]
        first_track = None
        for t in tracks:
            if t.piece_id == first_occ.piece_id and t.track_id == first_occ.track_id:
                first_track = t
                break

        if first_track is None:
            continue

        length = len(contour.pitch_intervals)
        start = first_occ.position
        end = start + length

        # Extract raw values from first occurrence
        pitch_classes = first_track.pitch_classes[start:end].tolist()
        octaves = first_track.octaves[start:end].tolist()
        velocities = first_track.velocities[start:end].tolist()
        durations = first_track.durations[start:end].tolist()
        rhythm_ioi = first_track.rhythm_ioi[start:end-1].tolist() if start < len(first_track.rhythm_ioi) else []

        # Store pattern with normalized contour AND raw values
        rules[rule_id] = {
            # Normalized contour (for matching)
            'pitch_intervals': list(contour.pitch_intervals),
            'rhythm_buckets': list(contour.rhythm_buckets),
            'velocity_buckets': list(contour.velocity_buckets),

            # Canonical raw values (from first occurrence)
            'pitch_classes': pitch_classes,
            'octaves': octaves,
            'velocities': velocities,
            'duration_buckets': [bucket_duration(d) for d in durations],
            'rhythm_ioi': rhythm_ioi,

            # All occurrences with their offsets
            'occurrences': [
                {
                    'piece_id': occ.piece_id,
                    'track_id': occ.track_id,
                    'position': occ.position,
                    'onset_time': occ.onset_time,
                    'T_offset': occ.T_offset,
                    'tau_offset': occ.tau_offset,
                    'v_offset': occ.v_offset,
                    'octave_offset': occ.octave_offset,
                }
                for occ in occurrences
            ],

            # Count for sorting
            'count': len(occurrences),
        }

    elapsed = time.time() - t0

    if verbose:
        # Statistics
        total_occurrences = sum(r['count'] for r in rules.values())
        multi_piece = sum(
            1 for r in rules.values()
            if len(set(o['piece_id'] for o in r['occurrences'])) > 1
        )

        # Length distribution
        length_dist = defaultdict(int)
        for r in rules.values():
            length_dist[len(r['pitch_intervals'])] += 1

        print(f"\n[T+τ Normalized Re-Pair] Complete in {elapsed:.1f}s", flush=True)
        print(f"  Patterns: {len(rules)}", flush=True)
        print(f"  Total occurrences: {total_occurrences:,}", flush=True)
        print(f"  Multi-piece patterns: {multi_piece}", flush=True)
        print(f"  Length distribution:", flush=True)
        for length in sorted(length_dist.keys())[:10]:
            print(f"    {length}: {length_dist[length]}", flush=True)

    return {
        'rules': rules,
        'n_rules': len(rules),
        'n_tracks': len(tracks),
        'normalization': 'T+tau+v',
    }


def build_ngram_index(
    tracks: List[FactoredTrack],
    sequences: List[List[int]],
    max_n: int = 20,
    verbose: bool = True
) -> Dict[tuple, List[Dict]]:
    """
    Build a hash index of all n-grams in the corpus.

    This is O(corpus_size × max_n) done ONCE, then pattern lookups are O(1).

    Returns: {ngram_tuple -> [{'piece_id', 'track_id', 'onset_time', 'position'}, ...]}
    """
    if verbose:
        print(f"  Building n-gram index (n=3..{max_n})...", flush=True)

    index = defaultdict(list)

    for track, seq in zip(tracks, sequences):
        seq_len = len(seq)
        # Index all n-grams from length 3 to max_n
        for n in range(3, min(max_n + 1, seq_len + 1)):
            for i in range(seq_len - n + 1):
                ngram = tuple(seq[i:i+n])
                onset_time = int(track.onsets[i]) if i < len(track.onsets) else 0
                index[ngram].append({
                    'piece_id': track.piece_id,
                    'track_id': track.track_id,
                    'onset_time': onset_time,
                    'position': i,
                })

    if verbose:
        print(f"  Indexed {len(index):,} unique n-grams", flush=True)

    return dict(index)


def find_pattern_occurrences(
    pattern_tokens: List[int],
    tracks: List[FactoredTrack],
    sequences: List[List[int]],
    ngram_index: Optional[Dict[tuple, List[Dict]]] = None
) -> List[Dict]:
    """
    Find all occurrences of a pattern in the corpus.

    If ngram_index is provided, uses O(1) lookup.
    Otherwise falls back to O(seq_len) scan.

    Returns list of {piece_id, track_id, onset_time, position} dicts.
    """
    pattern_tuple = tuple(pattern_tokens)

    # Fast path: use pre-built index
    if ngram_index is not None:
        return ngram_index.get(pattern_tuple, [])

    # Slow path: scan corpus (fallback)
    occurrences = []
    pattern_len = len(pattern_tokens)

    for track, seq in zip(tracks, sequences):
        seq_len = len(seq)
        for i in range(seq_len - pattern_len + 1):
            if tuple(seq[i:i+pattern_len]) == pattern_tuple:
                onset_time = int(track.onsets[i]) if i < len(track.onsets) else 0
                occurrences.append({
                    'piece_id': track.piece_id,
                    'track_id': track.track_id,
                    'onset_time': onset_time,
                    'position': i,
                })

    return occurrences


def run_repair_gpu_factored(
    tracks: List[FactoredTrack],
    device: str = 'cuda',
    max_rules: int = 10000,
    verbose: bool = True
) -> Dict:
    """
    Run GPU-accelerated Re-Pair v2 on factored token sequences.

    Uses encoded tokens that preserve ALL factors.
    Also tracks WHERE each pattern occurs for meta-pattern discovery.
    """
    try:
        from grammar.v2.repair_gpu_v2 import build_repair_grammar_v2

        # Convert tracks to encoded token sequences (keep both for occurrence tracking)
        sequences = [factored_track_to_tokens(t) for t in tracks]

        if verbose:
            total_notes = sum(len(s) for s in sequences)
            print(f"  GPU Re-Pair v2 (factored): {len(sequences)} seqs, {total_notes:,} tokens", flush=True)
            print(f"  Token vocabulary size: 7680 (12 pc × 10 oct × 8 vel × 8 dur)", flush=True)

        # Build Re-Pair grammar on GPU
        grammar = build_repair_grammar_v2(
            sequences,
            device=device,
            max_rules=max_rules,
            verbose=verbose
        )

        # Extract patterns with full factor decoding
        rules = {}
        pattern_count = 0
        for i in range(grammar.n_rules):
            rule_id = grammar.n_terminals + i
            expansion = grammar.expand_rule(rule_id)

            # Only keep patterns of reasonable length
            if 3 <= len(expansion) <= 20:
                # Decode each token back to factors
                decoded = [decode_factored_token(t) for t in expansion]
                rules[str(rule_id)] = {
                    'tokens': list(expansion),  # Ensure it's a list
                    'pitch_classes': [d[0] for d in decoded],
                    'octaves': [d[1] for d in decoded],
                    'velocities': [d[2] for d in decoded],
                    'duration_buckets': [d[3] for d in decoded],
                    'occurrences': [],  # Will be filled below
                    # Timing ratios - computed from first occurrence
                    'rhythm_ratios': [],  # IOI ratios (n-1 values for n notes)
                    'duration_ratios': [],  # Duration ratios (n values)
                    'velocity_ratios': [],  # Velocity ratios (n values)
                }
                pattern_count += 1

        if verbose:
            print(f"  Re-Pair v2: {len(rules)} factored patterns (from {grammar.n_rules} rules), "
                  f"compression {grammar.compression_ratio():.2f}x", flush=True)

        # Find occurrences via hash lookup (O(1) per position per pattern length)
        if verbose:
            print(f"  Finding pattern occurrences (hash lookup for {len(rules)} patterns)...", flush=True)

        import time
        t0 = time.time()

        # Build hash sets per pattern length for O(1) lookup
        # pattern_tuple -> rule_id
        pattern_to_rule = {}
        lengths_present = set()
        for rule_id, rule_data in rules.items():
            pattern = tuple(rule_data['tokens'])
            pattern_to_rule[pattern] = rule_id
            lengths_present.add(len(pattern))
            rules[rule_id]['occurrences'] = []

        lengths_present = sorted(lengths_present)

        # For each sequence, check each position against pattern hash
        for seq_idx, (track, seq) in enumerate(zip(tracks, sequences)):
            seq_tuple = tuple(seq)
            seq_len = len(seq)
            track_onsets = track.onsets
            track_durations = track.durations
            track_velocities = track.velocities

            for pattern_len in lengths_present:
                if pattern_len > seq_len:
                    continue

                # Scan this sequence - O(1) hash lookup per position
                for i in range(seq_len - pattern_len + 1):
                    window = seq_tuple[i:i + pattern_len]

                    if window in pattern_to_rule:
                        rule_id = pattern_to_rule[window]
                        rule_data = rules[rule_id]

                        # Extract timing data from this occurrence
                        end_idx = min(i + pattern_len, len(track_onsets))
                        pattern_onsets = track_onsets[i:end_idx]
                        pattern_durations = track_durations[i:end_idx] if i < len(track_durations) else np.array([480] * pattern_len)
                        pattern_velocities = track_velocities[i:end_idx] if i < len(track_velocities) else np.array([4] * pattern_len)

                        onset_time = int(pattern_onsets[0]) if len(pattern_onsets) > 0 else 0

                        # Compute IOIs (inter-onset intervals) between consecutive notes
                        if len(pattern_onsets) > 1:
                            iois = np.diff(pattern_onsets)
                            base_ioi = int(iois[0]) if iois[0] > 0 else 480
                        else:
                            iois = np.array([])
                            base_ioi = 480

                        # Compute base duration and velocity
                        base_duration = int(pattern_durations[0]) if len(pattern_durations) > 0 and pattern_durations[0] > 0 else 480
                        base_velocity = int(pattern_velocities[0]) if len(pattern_velocities) > 0 and pattern_velocities[0] > 0 else 4

                        # Store occurrence with timing offsets
                        rules[rule_id]['occurrences'].append({
                            'piece_id': track.piece_id,
                            'track_id': track.track_id,
                            'onset_time': onset_time,
                            'position': i,
                            # Timing offsets for reconstruction
                            'tau_offset': base_ioi,  # Base IOI in ticks
                            'duration_offset': base_duration,  # Base duration in ticks
                            'v_offset': base_velocity,  # Base velocity (0-7 quantized)
                        })

                        # Set canonical ratios from FIRST occurrence only
                        # This establishes the pattern's normalized shape
                        if not rule_data['rhythm_ratios'] and len(iois) > 0:
                            # Rhythm ratios: IOI[i] / base_ioi
                            rule_data['rhythm_ratios'] = [float(ioi) / base_ioi for ioi in iois]

                        if not rule_data['duration_ratios'] and len(pattern_durations) > 0:
                            # Duration ratios: duration[i] / base_duration
                            rule_data['duration_ratios'] = [float(d) / base_duration for d in pattern_durations]

                        if not rule_data['velocity_ratios'] and len(pattern_velocities) > 0:
                            # Velocity ratios: velocity[i] / base_velocity
                            rule_data['velocity_ratios'] = [float(v) / base_velocity for v in pattern_velocities]

        if verbose:
            print(f"  Occurrence scan completed in {time.time() - t0:.1f}s", flush=True)

        if verbose:
            multi_piece = sum(
                1 for r in rules.values()
                if len(set(occ['piece_id'] for occ in r.get('occurrences', []))) > 1
            )
            print(f"  {multi_piece} patterns appear in multiple pieces", flush=True)

        return {
            'rules': rules,
            'n_rules': len(rules),
            'n_sequences': len(sequences),
            'compression_ratio': grammar.compression_ratio(),
        }

    except ImportError as e:
        if verbose:
            print(f"  Re-Pair v2 not available ({e}), falling back to n-grams", flush=True)
        return run_factored_ngram_extraction(tracks, verbose=verbose)
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
            print(f"  Re-Pair v2 failed ({e}), falling back to n-grams", flush=True)
        return run_factored_ngram_extraction(tracks, verbose=verbose)


def run_factored_ngram_extraction(
    tracks: List[FactoredTrack],
    min_n: int = 3,
    max_n: int = 12,
    min_freq: int = 3,
    max_patterns: int = 2000,
    verbose: bool = True
) -> Dict:
    """
    Fast n-gram extraction on factored tokens WITH occurrence tracking.

    Returns rules dict with:
        - tokens, pitch_classes, etc. (the pattern itself)
        - occurrences: list of (piece_id, track_id, onset_time, position)
    """
    from collections import Counter

    # Build sequences WITH track metadata for occurrence tracking
    sequences_with_meta = []
    for track in tracks:
        tokens = factored_track_to_tokens(track)
        sequences_with_meta.append({
            'tokens': tokens,
            'piece_id': track.piece_id,
            'track_id': track.track_id,
            'onsets': track.onsets.tolist(),  # onset time per note
        })

    if verbose:
        total_notes = sum(len(s['tokens']) for s in sequences_with_meta)
        print(f"  Factored n-gram extraction: {len(sequences_with_meta)} seqs, {total_notes:,} tokens", flush=True)

    # Track both count AND occurrences for each n-gram
    ngram_counts = Counter()
    ngram_occurrences = defaultdict(list)  # ngram -> list of occurrences

    for seq_data in sequences_with_meta:
        seq = seq_data['tokens']
        piece_id = seq_data['piece_id']
        track_id = seq_data['track_id']
        onsets = seq_data['onsets']
        seq_len = len(seq)

        for n in range(min_n, min(max_n + 1, seq_len + 1)):
            for i in range(seq_len - n + 1):
                ngram = tuple(seq[i:i+n])
                ngram_counts[ngram] += 1

                # Record this occurrence
                onset_time = onsets[i] if i < len(onsets) else 0
                ngram_occurrences[ngram].append({
                    'piece_id': piece_id,
                    'track_id': track_id,
                    'onset_time': onset_time,
                    'position': i,
                })

    # Filter and select by frequency
    frequent = [(ng, c) for ng, c in ngram_counts.most_common() if c >= min_freq]
    selected = frequent[:max_patterns]

    rules = {}
    for i, (ngram, count) in enumerate(selected):
        decoded = [decode_factored_token(t) for t in ngram]
        rules[str(i)] = {
            'tokens': list(ngram),
            'pitch_classes': [d[0] for d in decoded],
            'octaves': [d[1] for d in decoded],
            'velocities': [d[2] for d in decoded],
            'duration_buckets': [d[3] for d in decoded],
            # ALL occurrences of this pattern in the corpus
            'occurrences': ngram_occurrences[ngram],
        }

    if verbose:
        # Count patterns appearing in multiple pieces
        multi_piece = sum(
            1 for r in rules.values()
            if len(set(occ['piece_id'] for occ in r['occurrences'])) > 1
        )
        print(f"  Extracted {len(rules)} factored patterns (from {len(ngram_counts)} unique n-grams)", flush=True)
        print(f"  {multi_piece} patterns appear in multiple pieces", flush=True)

    return {
        'rules': rules,
        'n_rules': len(rules),
        'n_sequences': len(sequences_with_meta),
    }


# =============================================================================
# FACTORED TRANSFORM DISCOVERY
# =============================================================================

def extract_factored_canonical_patterns(
    grammar_rules: Dict,
    min_length: int = 3,
    max_length: int = 20
) -> List[FactoredPattern]:
    """
    Extract canonical patterns from grammar rules with full factor info.

    Each pattern includes ALL occurrences (piece_id, track_id, onset_time)
    for meta-pattern discovery (Level 3).

    Timing data (rhythm_ratios, duration_ratios) is now properly extracted
    from the first occurrence, enabling lossless reconstruction.
    """
    canonicals = []

    for rule_id, rule_data in grammar_rules.items():
        if isinstance(rule_data, dict):
            # Factored format with timing data
            pitch_classes = rule_data.get('pitch_classes', [])
            octaves = rule_data.get('octaves', [])
            velocities = rule_data.get('velocities', [])
            duration_buckets = rule_data.get('duration_buckets', [])
            raw_occurrences = rule_data.get('occurrences', [])
            # NEW: Real timing ratios from first occurrence
            rhythm_ratios = rule_data.get('rhythm_ratios', [])
            duration_ratios = rule_data.get('duration_ratios', [])
            velocity_ratios = rule_data.get('velocity_ratios', [])
        else:
            # Legacy format - just pitch classes
            pitch_classes = [int(x) % 12 for x in rule_data]
            octaves = [4] * len(pitch_classes)  # Default octave
            velocities = [4] * len(pitch_classes)  # Default velocity
            duration_buckets = [3] * len(pitch_classes)  # Default quarter note
            raw_occurrences = []
            rhythm_ratios = []
            duration_ratios = []
            velocity_ratios = []

        if not (min_length <= len(pitch_classes) <= max_length):
            continue

        # Compute rhythm IOI from ratios if available, otherwise use placeholder
        # rhythm_ioi is used for backwards compat - new code should use rhythm_ratios
        n_notes = len(pitch_classes)
        if rhythm_ratios and len(rhythm_ratios) == n_notes - 1:
            # Convert ratios back to representative IOIs (using 480 as base)
            rhythm_ioi = [int(r * 480) for r in rhythm_ratios]
        else:
            # Fallback: placeholder (indicates missing data)
            rhythm_ioi = [480] * max(0, n_notes - 1)

        # Build PatternOccurrence objects with timing offsets
        occurrences = [
            PatternOccurrence(
                piece_id=occ['piece_id'],
                track_id=occ['track_id'],
                onset_time=occ['onset_time'],
                position=occ['position'],
                tau_offset=occ.get('tau_offset', 480),
                duration_offset=occ.get('duration_offset', 480),
                v_offset=occ.get('v_offset', 4),
            )
            for occ in raw_occurrences
        ]

        pattern = FactoredPattern(
            pattern_id=len(canonicals),
            rule_id=rule_id,
            pitch_classes=pitch_classes,
            octaves=octaves,
            velocities=velocities,
            durations=duration_buckets,
            rhythm_ioi=rhythm_ioi,
            occurrences=occurrences,
        )

        # Store the actual ratios for proper reconstruction
        # These override the computed properties in FactoredPattern
        pattern._rhythm_ratios = rhythm_ratios
        pattern._duration_ratios = duration_ratios
        pattern._velocity_ratios = velocity_ratios

        canonicals.append(pattern)

    return canonicals


def extract_cross_track_pairs(
    canonicals: List[FactoredPattern],
    time_tolerance: int = 480,  # Within 1 beat
    max_pairs: int = 100000,    # Limit to prevent explosion
    verbose: bool = True
) -> List[Tuple[Dict, Dict]]:
    """
    Extract pairs of patterns that occur:
    1. In the SAME piece
    2. On DIFFERENT tracks
    3. At approximately the SAME time (within tolerance)

    OPTIMIZED: Uses fully vectorized numpy operations.
    Returns list of (pattern_dict_1, pattern_dict_2) pairs for MDL analysis.
    """
    import numpy as np
    import time as time_module
    t0 = time_module.time()

    # ================================================================
    # STEP 1: Build flat arrays directly (avoid Python list appends)
    # ================================================================
    # Count total occurrences first
    total_occs = sum(len(p.occurrences) for p in canonicals)
    if total_occs == 0:
        if verbose:
            print(f"  Cross-track pairs: 0 (no occurrences)")
        return []

    # First pass: collect all unique piece_ids and create mapping
    # (piece_id may be string filename, not int)
    unique_pieces = set()
    for p in canonicals:
        for occ in p.occurrences:
            unique_pieces.add(occ.piece_id)
    piece_to_idx = {pid: i for i, pid in enumerate(sorted(unique_pieces, key=str))}

    # Pre-allocate numpy arrays
    piece_ids = np.empty(total_occs, dtype=np.int32)
    track_ids = np.empty(total_occs, dtype=np.int32)
    onsets = np.empty(total_occs, dtype=np.int32)
    lengths = np.empty(total_occs, dtype=np.int32)
    pattern_idxs = np.empty(total_occs, dtype=np.int32)

    # Build pattern cache with numpy arrays for pitch classes
    pattern_cache = {}

    idx = 0
    for p_idx, p in enumerate(canonicals):
        p_len = len(p)
        pattern_cache[p_idx] = {
            'pattern_id': p.pattern_id,
            'pitch_classes': p.pitch_classes,
            'rule_id': p.rule_id,
        }

        n_occ = len(p.occurrences)
        if n_occ > 0:
            # Batch assign
            for occ in p.occurrences:
                piece_ids[idx] = piece_to_idx[occ.piece_id]  # Map string to int
                track_ids[idx] = occ.track_id
                onsets[idx] = occ.onset_time
                lengths[idx] = p_len
                pattern_idxs[idx] = p_idx
                idx += 1

    n = idx
    if verbose:
        print(f"    Processing {n} pattern occurrences...", flush=True)

    # ================================================================
    # STEP 2: Sort by (piece, onset) for efficient grouping
    # ================================================================
    # Create composite key for sorting
    sort_keys = piece_ids.astype(np.int64) * 10000000 + onsets
    sort_order = np.argsort(sort_keys)

    piece_ids = piece_ids[sort_order]
    track_ids = track_ids[sort_order]
    onsets = onsets[sort_order]
    lengths = lengths[sort_order]
    pattern_idxs = pattern_idxs[sort_order]

    # ================================================================
    # STEP 3: Find piece boundaries using diff
    # ================================================================
    piece_changes = np.diff(piece_ids) != 0
    piece_boundaries = np.concatenate([[0], np.where(piece_changes)[0] + 1, [n]])

    # ================================================================
    # STEP 4: Process each piece with vectorized operations
    # ================================================================
    # Collect all valid pairs first as arrays, then create dicts at end
    all_p1_idx = []
    all_p2_idx = []
    all_piece = []
    all_t1 = []
    all_t2 = []
    all_o1 = []
    all_o2 = []

    n_pairs_collected = 0

    for p in range(len(piece_boundaries) - 1):
        start = piece_boundaries[p]
        end = piece_boundaries[p + 1]
        n_piece = end - start

        if n_piece < 2 or n_pairs_collected >= max_pairs:
            continue

        # Extract slice
        p_tracks = track_ids[start:end]
        p_onsets = onsets[start:end]
        p_lengths = lengths[start:end]
        p_patterns = pattern_idxs[start:end]
        piece_id = piece_ids[start]

        # For small pieces, use triu_indices
        if n_piece <= 500:
            i_idx, j_idx = np.triu_indices(n_piece, k=1)
        else:
            # For large pieces, only consider nearby pairs (sorted by onset)
            # Window-based approach: only pairs within 50 positions
            window = 50
            i_list, j_list = [], []
            for i in range(n_piece):
                j_end = min(i + window, n_piece)
                for j in range(i + 1, j_end):
                    i_list.append(i)
                    j_list.append(j)
            i_idx = np.array(i_list, dtype=np.int32)
            j_idx = np.array(j_list, dtype=np.int32)

        if len(i_idx) == 0:
            continue

        # Vectorized filtering
        diff_track_mask = p_tracks[i_idx] != p_tracks[j_idx]
        same_length_mask = p_lengths[i_idx] == p_lengths[j_idx]
        time_diff = np.abs(p_onsets[i_idx] - p_onsets[j_idx])
        time_mask = time_diff <= time_tolerance

        valid_mask = diff_track_mask & same_length_mask & time_mask
        valid_i = i_idx[valid_mask]
        valid_j = j_idx[valid_mask]

        n_valid = len(valid_i)
        if n_valid == 0:
            continue

        # Limit pairs from this piece
        remaining = max_pairs - n_pairs_collected
        if n_valid > remaining:
            valid_i = valid_i[:remaining]
            valid_j = valid_j[:remaining]
            n_valid = remaining

        # Collect arrays (batch append)
        all_p1_idx.append(p_patterns[valid_i])
        all_p2_idx.append(p_patterns[valid_j])
        all_piece.append(np.full(n_valid, piece_id, dtype=np.int32))
        all_t1.append(p_tracks[valid_i])
        all_t2.append(p_tracks[valid_j])
        all_o1.append(p_onsets[valid_i])
        all_o2.append(p_onsets[valid_j])

        n_pairs_collected += n_valid

    # ================================================================
    # STEP 5: Concatenate all arrays and build pairs
    # ================================================================
    if not all_p1_idx:
        if verbose:
            print(f"  Cross-track pairs: 0", flush=True)
        return []

    # Concatenate
    p1_idxs = np.concatenate(all_p1_idx)
    p2_idxs = np.concatenate(all_p2_idx)
    pieces = np.concatenate(all_piece)
    t1s = np.concatenate(all_t1)
    t2s = np.concatenate(all_t2)
    o1s = np.concatenate(all_o1)
    o2s = np.concatenate(all_o2)

    # Build pairs (this is unavoidable Python, but now minimal)
    cross_track_pairs = []
    for i in range(len(p1_idxs)):
        dict1 = pattern_cache[p1_idxs[i]].copy()
        dict1['_piece_id'] = int(pieces[i])
        dict1['_track_id'] = int(t1s[i])
        dict1['_onset'] = int(o1s[i])

        dict2 = pattern_cache[p2_idxs[i]].copy()
        dict2['_piece_id'] = int(pieces[i])
        dict2['_track_id'] = int(t2s[i])
        dict2['_onset'] = int(o2s[i])

        cross_track_pairs.append((dict1, dict2))

    if verbose:
        print(f"  Cross-track pairs: {len(cross_track_pairs)} ({time_module.time()-t0:.1f}s)", flush=True)

    return cross_track_pairs


def run_factored_mdl_transform_discovery(
    canonicals: List[FactoredPattern],
    max_depth: int = 2,
    min_frequency: int = 3,
    enable_cross_track: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run MDL-based transform discovery on factored patterns.

    Discovers transforms from:
    1. All same-length pattern pairs (standard) - using GPU-native mining
    2. Cross-track pairs within same piece (if enable_cross_track=True)

    Currently discovers pitch transforms, but structure supports
    extending to rhythm/velocity/duration transforms.
    """
    if verbose:
        print(f"\n  MDL Transform Discovery on {len(canonicals)} factored patterns...")

    # Convert to legacy format for existing discovery code
    legacy_canonicals = [
        {
            'pattern_id': p.pattern_id,
            'pitch_classes': p.pitch_classes,
            'rule_id': p.rule_id,
            # Include factor info for future extension
            '_octaves': p.octaves,
            '_velocities': p.velocities,
            '_durations': p.durations,
        }
        for p in canonicals
    ]

    # Generate compound candidates
    candidates = enumerate_compounds(max_depth=max_depth)
    if verbose:
        print(f"  Generated {len(candidates)} compound candidates")

    # Use GPU-native mining for standard pairs (avoids O(n²) Python object creation)
    relations = mine_transform_relations_gpu(
        legacy_canonicals,
        candidates=candidates,
        verbose=verbose,
        device='cuda'
    )

    # Add cross-track pairs if enabled (these are much fewer)
    cross_track_pairs = []
    if enable_cross_track:
        if verbose:
            print(f"  Extracting cross-track pairs...", flush=True)
        cross_track_pairs = extract_cross_track_pairs(canonicals, verbose=verbose)
        if cross_track_pairs:
            # Mine cross-track relations using GPU-native function
            cross_track_relations = mine_cross_track_pairs_gpu(
                cross_track_pairs, candidates, verbose=verbose, device='cuda'
            )
            # Merge into main relations
            for transform, pairs in cross_track_relations.items():
                relations[transform].extend(pairs)
            if verbose:
                n_cross_found = sum(len(p) for p in cross_track_relations.values())
                print(f"  Cross-track relations merged: {n_cross_found}")

    if not relations:
        return {
            'vocabulary': [],
            'stats': {},
            'comparison': {},
            'factors_stored': True,
        }

    # Select vocabulary via MDL
    vocabulary, stats = select_vocabulary_mdl(relations, min_frequency, verbose=verbose)

    # Stats
    comparison = {
        'd24_coverage': 0.0,
        'd48_coverage': 0.0,
        'discovered_coverage': 0.0,
        'd24_count': 24,
        'd48_count': 48,
        'discovered_count': len(vocabulary),
        'cross_track_pairs': len(cross_track_pairs),
    }
    if verbose:
        print(f"\n(Skipping D24 comparison - using factored patterns)", flush=True)

    return {
        'vocabulary': [t.name for t in vocabulary],
        'stats': {
            name: {
                'frequency': s.frequency,
                'unique_sources': s.unique_sources,
                'mdl_benefit': s.mdl_benefit,
            }
            for name, s in stats.items()
        },
        'comparison': comparison,
        'factors_stored': True,  # Flag indicating full factors preserved
        'cross_track_enabled': enable_cross_track,
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_factored_checkpoint(
    path: str,
    canonicals: List[FactoredPattern],
    grammar: Dict,
    transform_discovery: Dict,
    stats: PipelineStats,
    meta_patterns: Dict = None,
    verbose: bool = True
):
    """Save checkpoint with full factored data."""

    # Convert patterns to serializable format
    canonicals_data = [p.to_dict() for p in canonicals]

    # Convert grammar rules - handle both dict and list formats
    grammar_rules_data = {}
    for rule_id, rule_data in grammar['rules'].items():
        if isinstance(rule_data, dict):
            grammar_rules_data[rule_id] = rule_data
        else:
            grammar_rules_data[rule_id] = {'tokens': rule_data}

    data = {
        'version': np.array(['factored_v1']),

        # Stats
        'n_files': np.array([stats.n_files_loaded]),
        'n_tracks': np.array([stats.n_tracks]),
        'n_notes': np.array([stats.n_notes]),
        'n_canonicals': np.array([len(canonicals)]),
        'n_grammar_rules': np.array([grammar['n_rules']]),
        'n_transform_vocab': np.array([len(transform_discovery['vocabulary'])]),
        'total_time': np.array([stats.total_time]),

        # Factored flag
        'is_factored': np.array([True]),
        'factors': np.array(['pitch_class,octave,velocity,duration']),

        # JSON data with full factors (convert numpy types for JSON serialization)
        'canonical_patterns_json': np.array([json.dumps(_convert_numpy_types(canonicals_data))]),
        'grammar_rules_json': np.array([json.dumps(_convert_numpy_types(grammar_rules_data))]),
        'transform_vocabulary_json': np.array([json.dumps(_convert_numpy_types(transform_discovery['vocabulary']))]),
        'transform_stats_json': np.array([json.dumps(_convert_numpy_types(transform_discovery['stats']))]),

        # Level 3 meta-patterns
        'meta_patterns_json': np.array([json.dumps(_convert_numpy_types(meta_patterns or {}))]),
    }

    np.savez_compressed(path, **data)

    if verbose:
        print(f"\n  Factored checkpoint saved to: {path}")
        print(f"  Factors stored: pitch_class, octave, velocity, duration")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_factored_pipeline(
    corpus_path: str,
    output_path: str = 'checkpoint_factored.npz',
    max_files: int = 500,
    verbose: bool = True
) -> PipelineStats:
    """
    Run the full factored MDL pipeline.
    """
    stats = PipelineStats()
    total_start = time.time()

    if verbose:
        print("=" * 70)
        print("FACTORED MDL PIPELINE")
        print("=" * 70)
        print("Encoding: pitch_class × octave × velocity × duration")

    # Phase 1: Load MIDI files with full factoring
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 1] Loading MIDI files (factored) from {corpus_path}...", flush=True)

    midi_files = sorted(glob.glob(str(Path(corpus_path) / "*.mid")))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "*.midi")))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "**/*.mid"), recursive=True))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "**/*.midi"), recursive=True))
    midi_files = list(dict.fromkeys(midi_files))

    if max_files:
        midi_files = midi_files[:max_files]

    if verbose:
        print(f"  Found {len(midi_files)} MIDI files", flush=True)

    # Parallel loading
    all_tracks: List[FactoredTrack] = []
    num_workers = min(8, max(1, len(midi_files)))

    if verbose:
        print(f"  Using {num_workers} parallel workers", flush=True)

    batch_size = 50
    total_batches = (len(midi_files) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(midi_files))
        batch_files = midi_files[batch_start:batch_end]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_midi_factored, f): f for f in batch_files}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result:
                        all_tracks.extend(result)
                        stats.n_files_loaded += 1
                    else:
                        stats.n_files_failed += 1
                except Exception:
                    stats.n_files_failed += 1

        if verbose:
            elapsed = time.time() - phase_start
            rate = batch_end / elapsed if elapsed > 0 else 0
            print(f"    [{batch_end}/{len(midi_files)}] {stats.n_files_loaded} loaded, "
                  f"{stats.n_files_failed} failed, {len(all_tracks)} tracks "
                  f"({rate:.1f} files/s)", flush=True)

    stats.n_tracks = len(all_tracks)
    stats.n_notes = sum(len(t) for t in all_tracks)
    stats.phase_times['loading'] = time.time() - phase_start

    if verbose:
        print(f"  ✓ Loaded {stats.n_files_loaded} files, {stats.n_tracks} tracks, {stats.n_notes:,} notes", flush=True)
        print(f"  ✓ Each note has: pitch_class, octave, velocity, duration", flush=True)
        print(f"  Time: {stats.phase_times['loading']:.1f}s", flush=True)

    # Build track_instruments lookup: (piece_id, track_id) -> GM program
    track_instruments: Dict[Tuple[str, int], int] = {}
    for track in all_tracks:
        track_instruments[(track.piece_id, track.track_id)] = track.gm_program

    if verbose:
        # Count unique instruments
        unique_gm = set(track_instruments.values())
        print(f"  ✓ Built instrument lookup: {len(unique_gm)} unique GM programs", flush=True)

    # Phase 2: Factored grammar induction
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 2] Running factored grammar induction on {len(all_tracks)} sequences...", flush=True)

    stats.n_pitch_sequences = len(all_tracks)
    grammar = run_repair_gpu_factored(all_tracks, device='cuda', verbose=verbose)
    stats.n_grammar_rules = grammar['n_rules']
    stats.phase_times['grammar'] = time.time() - phase_start

    if verbose:
        print(f"  ✓ Induced {stats.n_grammar_rules} factored grammar rules", flush=True)
        print(f"  Time: {stats.phase_times['grammar']:.1f}s", flush=True)

    # Phase 3: Extract factored canonical patterns
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 3] Extracting factored canonical patterns...", flush=True)

    canonicals = extract_factored_canonical_patterns(grammar['rules'])
    stats.n_canonical_patterns = len(canonicals)
    stats.phase_times['canonicals'] = time.time() - phase_start

    if verbose:
        print(f"  ✓ Extracted {stats.n_canonical_patterns} factored patterns", flush=True)
        print(f"  Time: {stats.phase_times['canonicals']:.1f}s", flush=True)

    # Phase 4: MDL Transform Discovery on factored patterns
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 4] MDL Transform Discovery on {stats.n_canonical_patterns} factored patterns...", flush=True)

    transform_discovery = run_factored_mdl_transform_discovery(
        canonicals,
        max_depth=2,
        min_frequency=3,
        verbose=verbose
    )
    stats.n_transform_vocabulary = len(transform_discovery['vocabulary'])
    stats.phase_times['transforms'] = time.time() - phase_start

    if verbose:
        print(f"  ✓ Discovered {stats.n_transform_vocabulary} transforms", flush=True)
        print(f"  Time: {stats.phase_times['transforms']:.1f}s", flush=True)

    # Phase 5: Level 3 Meta-Pattern Discovery
    phase_start = time.time()
    meta_patterns = {}
    if verbose:
        print(f"\n[Phase 5] Level 3 Meta-Pattern Discovery...", flush=True)

    if stats.n_transform_vocabulary > 0 and stats.n_canonical_patterns > 10:
        try:
            # Convert canonicals to dict format for Level 3
            patterns_for_l3 = [p.to_dict() for p in canonicals]

            # Parse transform vocab for Level 3
            from scripts.level3_meta_patterns import parse_transform_name, aggregate_orchestration_rules
            transform_vocab_parsed = [
                parse_transform_name(t) if isinstance(t, str) else t
                for t in transform_discovery['vocabulary']
            ]

            # Step 1: Build transform lookup table
            if verbose:
                print(f"  Building GPU transform lookup table...", flush=True)
            transform_lookup = build_transform_lookup_gpu(
                patterns_for_l3, transform_vocab_parsed, device='cuda'
            )
            if verbose:
                print(f"    Lookup table: {transform_lookup.shape}", flush=True)

            # Step 2: Extract transform sequences
            if verbose:
                print(f"  Extracting transform sequences per piece...", flush=True)
            sequences = extract_transform_sequences_gpu(
                patterns_for_l3, transform_lookup, device='cuda'
            )
            if verbose:
                print(f"    {len(sequences)} sequences extracted", flush=True)

            # Step 3: Run meta Re-Pair
            if len(sequences) >= 10:
                if verbose:
                    print(f"  Running GPU Re-Pair on transform sequences...", flush=True)
                meta_result = run_meta_repair_gpu(
                    sequences,
                    n_transforms=len(transform_vocab_parsed),
                    device='cuda',
                    verbose=verbose
                )

                # Step 4: Interpret
                if meta_result['n_rules'] > 0:
                    interpreted = interpret_meta_patterns(
                        meta_result['rules'],
                        transform_vocab_parsed,
                        verbose=verbose
                    )
                    meta_patterns = {
                        'rules': meta_result['rules'],
                        'interpreted': interpreted,
                        'compression_ratio': meta_result.get('compression_ratio', 0),
                        'n_sequences': len(sequences),
                    }
                    if verbose:
                        print(f"  ✓ Discovered {meta_result['n_rules']} horizontal meta-patterns", flush=True)
            else:
                if verbose:
                    print(f"  Too few sequences for meta-pattern discovery ({len(sequences)})", flush=True)

            # Step 5: Orchestration Rule Aggregation (vertical/cross-track relations)
            if verbose:
                print(f"  Aggregating orchestration rules (vertical slices)...", flush=True)
            orchestration_result = aggregate_orchestration_rules(
                patterns_for_l3,
                transform_vocab_parsed,
                min_confidence=0.3,
                min_frequency=5,
                verbose=verbose,
                track_instruments=track_instruments,
            )
            if orchestration_result['n_rules'] > 0:
                meta_patterns['orchestration_rules'] = orchestration_result['rules']
                meta_patterns['n_orchestration_rules'] = orchestration_result['n_rules']
                meta_patterns['n_vertical_slices'] = orchestration_result['n_slices']
                if verbose:
                    print(f"  ✓ Found {orchestration_result['n_rules']} orchestration rules from {orchestration_result['n_slices']} vertical slices", flush=True)

        except Exception as e:
            if verbose:
                print(f"  Level 3 discovery failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
    else:
        if verbose:
            print(f"  Skipping (need transforms and patterns)", flush=True)

    stats.phase_times['meta_patterns'] = time.time() - phase_start
    if verbose:
        print(f"  Time: {stats.phase_times['meta_patterns']:.1f}s", flush=True)

    # Phase 6: Save factored checkpoint
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 6] Saving factored checkpoint to {output_path}...", flush=True)

    stats.total_time = time.time() - total_start

    save_factored_checkpoint(
        output_path,
        canonicals,
        grammar,
        transform_discovery,
        stats,
        meta_patterns=meta_patterns,
        verbose=verbose
    )
    stats.phase_times['checkpoint'] = time.time() - phase_start

    # Final summary
    if verbose:
        print(f"\n{'=' * 70}", flush=True)
        print("✓ FACTORED PIPELINE COMPLETE", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"  Files: {stats.n_files_loaded} loaded, {stats.n_files_failed} failed", flush=True)
        print(f"  Tracks: {stats.n_tracks}", flush=True)
        print(f"  Notes: {stats.n_notes:,}", flush=True)
        print(f"  Grammar rules: {stats.n_grammar_rules} (factored tokens)", flush=True)
        print(f"  Canonical patterns: {stats.n_canonical_patterns}", flush=True)
        print(f"  Transform vocabulary: {stats.n_transform_vocabulary}", flush=True)
        print(f"  Factors stored: pitch_class, octave, velocity, duration", flush=True)
        print(f"  Total time: {stats.total_time:.1f}s", flush=True)
        print(f"\n  Checkpoint: {output_path}", flush=True)

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Factored MDL Pipeline')
    parser.add_argument('--corpus', type=str, required=True, help='Path to MIDI corpus')
    parser.add_argument('--output', type=str, default='checkpoint_factored.npz', help='Output checkpoint')
    parser.add_argument('--max-files', type=int, default=500, help='Max files to process')
    args = parser.parse_args()

    run_factored_pipeline(
        corpus_path=args.corpus,
        output_path=args.output,
        max_files=args.max_files,
        verbose=True
    )
