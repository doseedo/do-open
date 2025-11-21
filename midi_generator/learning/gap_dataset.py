"""
Gap Dataset Creation - Agent 4
======================================

Creates training data for semantic feature discovery by computing
reconstruction gaps between original MIDI and parameter-based reconstructions.

This module provides:
1. GapAnalyzer - Computes reconstruction gaps
2. ParameterMIDIGenerator - Generates approximate MIDI from parameters
3. GapCache - Efficient caching system (5-10 GB)
4. GapDataset - PyTorch Dataset for training

Author: Agent 4 - Gap Dataset Creation
Phase: 2 (Neural Infrastructure)
Duration: 7-8 days
License: MIT
"""

import json
import pickle
import hashlib
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Import existing extractors
from midi_generator.feature_selection.optimized_feature_extractor import (
    OptimizedFeatureExtractor,
    FeatureNormalizer
)
from midi_generator.parameters.hierarchical_extractor_v2 import (
    HierarchicalParameterExtractorV2
)

# Optional imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available. Install with: pip install mido")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ReconstructionGap:
    """
    Represents the gap between original and reconstructed MIDI features.

    This is the training signal for semantic feature discovery:
    - Large gaps indicate missing parameters
    - Small gaps indicate sufficient reconstruction
    """
    file_id: str
    file_path: str

    # Original MIDI data
    original_features: np.ndarray  # 200D feature vector
    original_parameters: Dict[str, Any]  # 50 parameters

    # Reconstructed MIDI data (from parameters)
    reconstructed_features: np.ndarray  # 200D feature vector
    reconstructed_parameters: Dict[str, Any]  # 50 parameters

    # Gap analysis
    feature_gaps: np.ndarray  # |original - reconstructed| for each feature
    parameter_gaps: Dict[str, float]  # gaps for each parameter

    # Summary statistics
    total_gap: float  # Overall reconstruction error
    max_gap_indices: List[int]  # Indices of largest gaps
    max_gap_values: List[float]  # Values of largest gaps

    # Metadata
    computation_time: float = 0.0
    success: bool = True
    error_message: str = ""

    def get_top_gaps(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get indices and values of top k gaps"""
        top_indices = np.argsort(self.feature_gaps)[-k:][::-1]
        return [(int(idx), float(self.feature_gaps[idx])) for idx in top_indices]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'file_id': self.file_id,
            'file_path': self.file_path,
            'total_gap': float(self.total_gap),
            'feature_gaps': self.feature_gaps.tolist(),
            'parameter_gaps': self.parameter_gaps,
            'max_gap_indices': [int(x) for x in self.max_gap_indices],
            'max_gap_values': [float(x) for x in self.max_gap_values],
            'computation_time': self.computation_time,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class CorpusGapStatistics:
    """
    Aggregated gap statistics across entire corpus.

    Helps identify:
    - Which features are consistently under-reconstructed
    - Which parameters are most important
    - Overall reconstruction quality
    """
    n_files: int
    total_files_processed: int
    failed_files: int

    # Per-feature statistics
    mean_feature_gaps: np.ndarray  # Mean gap for each of 200 features
    std_feature_gaps: np.ndarray   # Std of gaps
    max_feature_gaps: np.ndarray   # Max gap observed

    # Per-parameter statistics
    mean_parameter_gaps: Dict[str, float]
    std_parameter_gaps: Dict[str, float]
    max_parameter_gaps: Dict[str, float]

    # Overall statistics
    mean_total_gap: float
    std_total_gap: float
    percentiles: Dict[str, float]  # 25th, 50th, 75th, 95th percentiles

    # Top problematic features
    top_gap_features: List[Tuple[int, float]]  # (index, mean_gap)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'n_files': self.n_files,
            'total_files_processed': self.total_files_processed,
            'failed_files': self.failed_files,
            'mean_feature_gaps': self.mean_feature_gaps.tolist(),
            'std_feature_gaps': self.std_feature_gaps.tolist(),
            'max_feature_gaps': self.max_feature_gaps.tolist(),
            'mean_parameter_gaps': self.mean_parameter_gaps,
            'std_parameter_gaps': self.std_parameter_gaps,
            'max_parameter_gaps': self.max_parameter_gaps,
            'mean_total_gap': float(self.mean_total_gap),
            'std_total_gap': float(self.std_total_gap),
            'percentiles': self.percentiles,
            'top_gap_features': [(int(idx), float(gap)) for idx, gap in self.top_gap_features]
        }


# ============================================================================
# ParameterMIDIGenerator
# ============================================================================

class ParameterMIDIGenerator:
    """
    Generates approximate MIDI from parameters (Option C - Risk Mitigation).

    This is a simplified generator that produces musically valid MIDI
    based on the 50 extracted parameters. It's used to compute reconstruction
    gaps without requiring the full generation system.

    Strategy:
    - Use parameters to create a simple but musically coherent MIDI
    - Focus on capturing the aspects that parameters represent
    - Ensure generated MIDI can be feature-extracted

    Note: This is intentionally approximate. The goal is NOT perfect
    reconstruction, but rather to identify what's MISSING from the
    current parameter set.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        if not MIDO_AVAILABLE:
            raise RuntimeError("mido is required. Install with: pip install mido")

    def generate(
        self,
        parameters: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Optional[mido.MidiFile]:
        """
        Generate approximate MIDI from parameters.

        Args:
            parameters: Dictionary with level1, level2, level3 parameters
            output_path: Optional path to save MIDI file

        Returns:
            mido.MidiFile object, or None if generation fails
        """
        try:
            # Extract parameters
            level1 = parameters.get('level1_global', {})
            level2 = parameters.get('level2_universal', {})
            level3 = parameters.get('level3_genre_specific', {})

            # Create MIDI file
            midi = mido.MidiFile(type=1)  # Type 1: multiple tracks

            # Get basic timing info
            tempo_bpm = level1.get('tempo.bpm', 120.0)
            tempo = mido.bpm2tempo(tempo_bpm)
            ticks_per_beat = 480

            # Create tempo track
            tempo_track = mido.MidiTrack()
            midi.tracks.append(tempo_track)
            tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

            # Extract time signature
            time_sig = level1.get('time_signature.string', '4/4')
            numerator, denominator = self._parse_time_signature(time_sig)
            tempo_track.append(mido.MetaMessage(
                'time_signature',
                numerator=numerator,
                denominator=denominator,
                time=0
            ))

            # Generate melody track
            if 'melody' in level2:
                melody_track = self._generate_melody_track(
                    level2['melody'],
                    ticks_per_beat,
                    tempo_bpm
                )
                midi.tracks.append(melody_track)

            # Generate harmony track
            if 'harmony' in level2:
                harmony_track = self._generate_harmony_track(
                    level2['harmony'],
                    level3.get('harmony', {}),
                    ticks_per_beat,
                    tempo_bpm
                )
                midi.tracks.append(harmony_track)

            # Generate rhythm track (drums)
            if 'rhythm' in level2:
                rhythm_track = self._generate_rhythm_track(
                    level2['rhythm'],
                    ticks_per_beat,
                    tempo_bpm
                )
                midi.tracks.append(rhythm_track)

            # Generate bass track
            if 'bass' in level2:
                bass_track = self._generate_bass_track(
                    level2['bass'],
                    level3.get('bass', {}),
                    ticks_per_beat,
                    tempo_bpm
                )
                midi.tracks.append(bass_track)

            # Save if requested
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                midi.save(output_path)
                if self.verbose:
                    print(f"✅ Generated MIDI saved to {output_path}")

            return midi

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Generation failed: {e}")
            return None

    def _parse_time_signature(self, time_sig_str: str) -> Tuple[int, int]:
        """Parse time signature string like '4/4' to (4, 4)"""
        try:
            parts = time_sig_str.split('/')
            return int(parts[0]), int(parts[1])
        except:
            return 4, 4

    def _generate_melody_track(
        self,
        melody_params: Dict,
        ticks_per_beat: int,
        tempo_bpm: float
    ) -> mido.MidiTrack:
        """Generate melody track from parameters"""
        track = mido.MidiTrack()
        track.append(mido.Message('program_change', program=0, time=0))  # Piano

        # Extract melody parameters
        pitch_range_min = melody_params.get('pitch_range.min', 60)
        pitch_range_max = melody_params.get('pitch_range.max', 84)
        avg_pitch = melody_params.get('pitch_statistics.mean', 72.0)
        note_density = melody_params.get('note_density.notes_per_beat', 2.0)

        # Generate simple melodic sequence
        duration = 16  # 16 beats
        notes_per_beat = max(0.25, min(4.0, note_density))
        note_duration_beats = 1.0 / notes_per_beat
        note_duration_ticks = int(note_duration_beats * ticks_per_beat)

        current_time = 0
        current_pitch = int(avg_pitch)

        for beat in range(int(duration * notes_per_beat)):
            # Simple melodic contour (sine wave pattern)
            pitch_offset = int(8 * np.sin(beat * 0.3))
            pitch = int(np.clip(
                current_pitch + pitch_offset,
                pitch_range_min,
                pitch_range_max
            ))

            velocity = 80

            # Note on
            track.append(mido.Message(
                'note_on',
                note=pitch,
                velocity=velocity,
                time=current_time
            ))

            # Note off
            track.append(mido.Message(
                'note_off',
                note=pitch,
                velocity=0,
                time=note_duration_ticks
            ))

            current_time = 0

        return track

    def _generate_harmony_track(
        self,
        harmony_params: Dict,
        harmony_genre_params: Dict,
        ticks_per_beat: int,
        tempo_bpm: float
    ) -> mido.MidiTrack:
        """Generate harmony/chord track from parameters"""
        track = mido.MidiTrack()
        track.append(mido.Message('program_change', program=0, time=0))  # Piano

        # Extract harmony parameters
        chord_complexity = harmony_params.get('chord_complexity.avg_notes', 3.0)
        chord_change_rate = harmony_params.get('chord_change_rate.changes_per_beat', 0.5)

        # Generate chord progression
        duration_beats = 16
        beats_per_chord = max(1, int(1.0 / chord_change_rate))

        # Simple chord progression: I-IV-V-I
        root_notes = [60, 65, 67, 60]  # C, F, G, C
        chord_idx = 0

        current_time = 0
        chord_duration_ticks = beats_per_chord * ticks_per_beat

        for _ in range(int(duration_beats / beats_per_chord)):
            root = root_notes[chord_idx % len(root_notes)]
            chord_notes = self._build_chord(root, int(chord_complexity))

            # Chord on
            for i, note in enumerate(chord_notes):
                track.append(mido.Message(
                    'note_on',
                    note=note,
                    velocity=60,
                    time=current_time if i == 0 else 0
                ))

            # Chord off
            for i, note in enumerate(chord_notes):
                track.append(mido.Message(
                    'note_off',
                    note=note,
                    velocity=0,
                    time=chord_duration_ticks if i == 0 else 0
                ))

            current_time = 0
            chord_idx += 1

        return track

    def _build_chord(self, root: int, num_notes: int) -> List[int]:
        """Build a chord from root note"""
        # Major triad intervals
        intervals = [0, 4, 7, 11, 12]  # Root, M3, P5, M7, Octave
        return [root + intervals[i % len(intervals)] for i in range(num_notes)]

    def _generate_rhythm_track(
        self,
        rhythm_params: Dict,
        ticks_per_beat: int,
        tempo_bpm: float
    ) -> mido.MidiTrack:
        """Generate rhythm/drum track from parameters"""
        track = mido.MidiTrack()

        # Drums are on channel 9 (10 in 1-indexed)
        kick = 36
        snare = 38
        hihat = 42

        # Extract rhythm parameters
        density = rhythm_params.get('note_density.notes_per_beat', 4.0)
        syncopation = rhythm_params.get('syncopation.degree', 0.3)

        # Generate basic drum pattern
        duration_beats = 16
        current_time = 0

        for beat in range(duration_beats):
            # Kick on 1 and 3
            if beat % 4 in [0, 2]:
                track.append(mido.Message(
                    'note_on',
                    note=kick,
                    velocity=90,
                    time=current_time,
                    channel=9
                ))
                track.append(mido.Message(
                    'note_off',
                    note=kick,
                    velocity=0,
                    time=ticks_per_beat // 4,
                    channel=9
                ))
                current_time = 0

            # Snare on 2 and 4
            if beat % 4 in [1, 3]:
                track.append(mido.Message(
                    'note_on',
                    note=snare,
                    velocity=85,
                    time=current_time,
                    channel=9
                ))
                track.append(mido.Message(
                    'note_off',
                    note=snare,
                    velocity=0,
                    time=ticks_per_beat // 4,
                    channel=9
                ))
                current_time = 0

            # Hi-hat every quarter note
            for _ in range(4):
                track.append(mido.Message(
                    'note_on',
                    note=hihat,
                    velocity=70,
                    time=current_time,
                    channel=9
                ))
                track.append(mido.Message(
                    'note_off',
                    note=hihat,
                    velocity=0,
                    time=ticks_per_beat // 8,
                    channel=9
                ))
                current_time = ticks_per_beat // 4 - ticks_per_beat // 8

        return track

    def _generate_bass_track(
        self,
        bass_params: Dict,
        bass_genre_params: Dict,
        ticks_per_beat: int,
        tempo_bpm: float
    ) -> mido.MidiTrack:
        """Generate bass track from parameters"""
        track = mido.MidiTrack()
        track.append(mido.Message('program_change', program=32, time=0))  # Acoustic Bass

        # Extract bass parameters
        pitch_range_min = bass_params.get('pitch_range.min', 28)
        pitch_range_max = bass_params.get('pitch_range.max', 55)
        avg_pitch = bass_params.get('pitch_statistics.mean', 40.0)

        # Generate bass line following chord roots
        root_notes = [36, 41, 43, 36]  # C2, F2, G2, C2
        duration_beats = 16
        beats_per_note = 2

        current_time = 0
        note_duration_ticks = beats_per_note * ticks_per_beat

        for i in range(int(duration_beats / beats_per_note)):
            note = root_notes[i % len(root_notes)]

            track.append(mido.Message(
                'note_on',
                note=note,
                velocity=75,
                time=current_time
            ))
            track.append(mido.Message(
                'note_off',
                note=note,
                velocity=0,
                time=note_duration_ticks
            ))
            current_time = 0

        return track


# ============================================================================
# GapAnalyzer
# ============================================================================

class GapAnalyzer:
    """
    Analyzes reconstruction gaps between original and reconstructed MIDI.

    Process:
    1. Extract 200D features + 50 params from original MIDI
    2. Generate approximate MIDI from 50 params
    3. Extract 200D features from reconstructed MIDI
    4. Compute gaps: |original_features - reconstructed_features|
    5. Identify largest gaps (missing information)

    These gaps are the training signal for semantic feature discovery.
    """

    def __init__(
        self,
        feature_extractor: OptimizedFeatureExtractor,
        parameter_extractor: HierarchicalParameterExtractorV2,
        midi_generator: ParameterMIDIGenerator,
        verbose: bool = False
    ):
        self.feature_extractor = feature_extractor
        self.parameter_extractor = parameter_extractor
        self.midi_generator = midi_generator
        self.verbose = verbose

    def compute_gap(
        self,
        midi_file: Path,
        temp_dir: Optional[Path] = None
    ) -> ReconstructionGap:
        """
        Compute reconstruction gap for a single MIDI file.

        Args:
            midi_file: Path to original MIDI file
            temp_dir: Directory for temporary files

        Returns:
            ReconstructionGap object
        """
        start_time = time.time()

        try:
            # Step 1: Extract from original MIDI
            original_features = self.feature_extractor.extract(midi_file)
            extraction_result = self.parameter_extractor.extract_complete(str(midi_file))
            original_parameters = extraction_result['parameters']

            # Step 2: Generate reconstructed MIDI
            if temp_dir is None:
                temp_dir = Path('/tmp/gap_analysis')
            temp_dir.mkdir(parents=True, exist_ok=True)

            reconstructed_path = temp_dir / f"reconstructed_{midi_file.stem}.mid"
            reconstructed_midi = self.midi_generator.generate(
                original_parameters,
                output_path=reconstructed_path
            )

            if reconstructed_midi is None:
                raise RuntimeError("MIDI generation failed")

            # Step 3: Extract features from reconstructed MIDI
            reconstructed_features = self.feature_extractor.extract(reconstructed_path)
            reconstructed_extraction = self.parameter_extractor.extract_complete(
                str(reconstructed_path)
            )
            reconstructed_parameters = reconstructed_extraction['parameters']

            # Step 4: Compute gaps
            feature_gaps = np.abs(original_features - reconstructed_features)
            total_gap = float(np.mean(feature_gaps))

            # Compute parameter gaps
            parameter_gaps = self._compute_parameter_gaps(
                original_parameters,
                reconstructed_parameters
            )

            # Find largest gaps
            top_k = 20
            top_indices = np.argsort(feature_gaps)[-top_k:][::-1]
            max_gap_indices = top_indices.tolist()
            max_gap_values = feature_gaps[top_indices].tolist()

            # Clean up temporary file
            if reconstructed_path.exists():
                reconstructed_path.unlink()

            computation_time = time.time() - start_time

            return ReconstructionGap(
                file_id=midi_file.stem,
                file_path=str(midi_file),
                original_features=original_features,
                original_parameters=original_parameters,
                reconstructed_features=reconstructed_features,
                reconstructed_parameters=reconstructed_parameters,
                feature_gaps=feature_gaps,
                parameter_gaps=parameter_gaps,
                total_gap=total_gap,
                max_gap_indices=max_gap_indices,
                max_gap_values=max_gap_values,
                computation_time=computation_time,
                success=True,
                error_message=""
            )

        except Exception as e:
            computation_time = time.time() - start_time

            if self.verbose:
                print(f"⚠️ Gap computation failed for {midi_file}: {e}")

            # Return error gap
            return ReconstructionGap(
                file_id=midi_file.stem,
                file_path=str(midi_file),
                original_features=np.zeros(200),
                original_parameters={},
                reconstructed_features=np.zeros(200),
                reconstructed_parameters={},
                feature_gaps=np.zeros(200),
                parameter_gaps={},
                total_gap=0.0,
                max_gap_indices=[],
                max_gap_values=[],
                computation_time=computation_time,
                success=False,
                error_message=str(e)
            )

    def _compute_parameter_gaps(
        self,
        original: Dict[str, Any],
        reconstructed: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute gaps for parameters"""
        gaps = {}

        for level_key in ['level1_global', 'level2_universal', 'level3_genre_specific']:
            orig_level = original.get(level_key, {})
            recon_level = reconstructed.get(level_key, {})

            # Flatten nested dictionaries
            orig_flat = self._flatten_dict(orig_level)
            recon_flat = self._flatten_dict(recon_level)

            for param_name, orig_value in orig_flat.items():
                recon_value = recon_flat.get(param_name, 0.0)

                # Compute gap (handle different types)
                if isinstance(orig_value, (int, float)) and isinstance(recon_value, (int, float)):
                    gap = abs(float(orig_value) - float(recon_value))
                elif isinstance(orig_value, str) and isinstance(recon_value, str):
                    gap = 0.0 if orig_value == recon_value else 1.0
                else:
                    gap = 0.0

                gaps[f"{level_key}.{param_name}"] = gap

        return gaps

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def analyze_corpus_gaps(
        self,
        midi_files: List[Path],
        output_dir: Optional[Path] = None,
        show_progress: bool = True
    ) -> CorpusGapStatistics:
        """
        Analyze reconstruction gaps across entire corpus.

        Args:
            midi_files: List of MIDI file paths
            output_dir: Optional directory to save individual gaps
            show_progress: Show progress bar

        Returns:
            CorpusGapStatistics object
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Analyzing reconstruction gaps for {len(midi_files)} files")
            print(f"{'='*70}\n")

        gaps_list = []
        all_feature_gaps = []
        all_parameter_gaps = defaultdict(list)
        all_total_gaps = []
        failed_count = 0

        iterator = midi_files
        if show_progress and TQDM_AVAILABLE:
            iterator = tqdm(midi_files, desc="Computing gaps")

        for midi_file in iterator:
            gap = self.compute_gap(midi_file)
            gaps_list.append(gap)

            if gap.success:
                all_feature_gaps.append(gap.feature_gaps)
                all_total_gaps.append(gap.total_gap)

                for param_name, param_gap in gap.parameter_gaps.items():
                    all_parameter_gaps[param_name].append(param_gap)
            else:
                failed_count += 1

        # Save individual gaps if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            gaps_file = output_dir / 'individual_gaps.json'
            with open(gaps_file, 'w') as f:
                json.dump([g.to_dict() for g in gaps_list], f, indent=2)

            if self.verbose:
                print(f"✅ Saved individual gaps to {gaps_file}")

        # Compute corpus-wide statistics
        if len(all_feature_gaps) == 0:
            raise RuntimeError("No successful gap computations")

        all_feature_gaps = np.array(all_feature_gaps)  # (n_files, 200)

        mean_feature_gaps = np.mean(all_feature_gaps, axis=0)
        std_feature_gaps = np.std(all_feature_gaps, axis=0)
        max_feature_gaps = np.max(all_feature_gaps, axis=0)

        # Parameter statistics
        mean_parameter_gaps = {
            name: float(np.mean(vals))
            for name, vals in all_parameter_gaps.items()
        }
        std_parameter_gaps = {
            name: float(np.std(vals))
            for name, vals in all_parameter_gaps.items()
        }
        max_parameter_gaps = {
            name: float(np.max(vals))
            for name, vals in all_parameter_gaps.items()
        }

        # Overall statistics
        mean_total_gap = float(np.mean(all_total_gaps))
        std_total_gap = float(np.std(all_total_gaps))
        percentiles = {
            'p25': float(np.percentile(all_total_gaps, 25)),
            'p50': float(np.percentile(all_total_gaps, 50)),
            'p75': float(np.percentile(all_total_gaps, 75)),
            'p95': float(np.percentile(all_total_gaps, 95))
        }

        # Top problematic features
        top_k = 50
        top_indices = np.argsort(mean_feature_gaps)[-top_k:][::-1]
        top_gap_features = [
            (int(idx), float(mean_feature_gaps[idx]))
            for idx in top_indices
        ]

        statistics = CorpusGapStatistics(
            n_files=len(midi_files),
            total_files_processed=len(gaps_list),
            failed_files=failed_count,
            mean_feature_gaps=mean_feature_gaps,
            std_feature_gaps=std_feature_gaps,
            max_feature_gaps=max_feature_gaps,
            mean_parameter_gaps=mean_parameter_gaps,
            std_parameter_gaps=std_parameter_gaps,
            max_parameter_gaps=max_parameter_gaps,
            mean_total_gap=mean_total_gap,
            std_total_gap=std_total_gap,
            percentiles=percentiles,
            top_gap_features=top_gap_features
        )

        # Save statistics
        if output_dir:
            stats_file = output_dir / 'corpus_gap_statistics.json'
            with open(stats_file, 'w') as f:
                json.dump(statistics.to_dict(), f, indent=2)

            if self.verbose:
                print(f"✅ Saved corpus statistics to {stats_file}")

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Gap Analysis Summary:")
            print(f"  Files processed: {statistics.total_files_processed}/{statistics.n_files}")
            print(f"  Failed: {statistics.failed_files}")
            print(f"  Mean total gap: {statistics.mean_total_gap:.4f} ± {statistics.std_total_gap:.4f}")
            print(f"  Top 5 problematic features:")
            for idx, gap in statistics.top_gap_features[:5]:
                print(f"    Feature {idx}: {gap:.4f}")
            print(f"{'='*70}\n")

        return statistics


# ============================================================================
# GapCache
# ============================================================================

class GapCache:
    """
    Efficient caching system for gap computations.

    Speeds up retraining by caching:
    - Computed gaps
    - Feature extractions
    - Parameter extractions

    Cache size: 5-10 GB (configurable)
    Cache key: SHA256 hash of MIDI file
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 10.0,
        verbose: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.verbose = verbose

        # Cache metadata
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'entries': {},
            'total_size_bytes': 0,
            'hit_count': 0,
            'miss_count': 0
        }

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_cache_path(self, file_hash: str) -> Path:
        """Get cache file path for hash"""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = file_hash[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{file_hash}.pkl"

    def get(self, midi_file: Path) -> Optional[ReconstructionGap]:
        """
        Get cached gap for MIDI file.

        Args:
            midi_file: Path to MIDI file

        Returns:
            ReconstructionGap if cached, None otherwise
        """
        file_hash = self._compute_file_hash(midi_file)
        cache_path = self._get_cache_path(file_hash)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    gap = pickle.load(f)

                self.metadata['hit_count'] += 1

                if self.verbose:
                    print(f"✅ Cache hit for {midi_file.name}")

                return gap
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Cache read error: {e}")
                return None

        self.metadata['miss_count'] += 1
        return None

    def put(self, midi_file: Path, gap: ReconstructionGap):
        """
        Cache gap computation.

        Args:
            midi_file: Path to MIDI file
            gap: ReconstructionGap object to cache
        """
        file_hash = self._compute_file_hash(midi_file)
        cache_path = self._get_cache_path(file_hash)

        # Check if we need to evict old entries
        self._evict_if_needed()

        # Save gap
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(gap, f)

            # Update metadata
            file_size = cache_path.stat().st_size
            self.metadata['entries'][file_hash] = {
                'file_path': str(midi_file),
                'cache_path': str(cache_path),
                'size_bytes': file_size,
                'timestamp': time.time()
            }
            self.metadata['total_size_bytes'] += file_size
            self._save_metadata()

            if self.verbose:
                print(f"✅ Cached gap for {midi_file.name} ({file_size / 1024:.1f} KB)")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Cache write error: {e}")

    def _evict_if_needed(self):
        """Evict oldest entries if cache is too large"""
        while self.metadata['total_size_bytes'] > self.max_size_bytes:
            # Find oldest entry
            oldest_hash = None
            oldest_time = float('inf')

            for file_hash, entry in self.metadata['entries'].items():
                if entry['timestamp'] < oldest_time:
                    oldest_time = entry['timestamp']
                    oldest_hash = file_hash

            if oldest_hash is None:
                break

            # Remove oldest entry
            entry = self.metadata['entries'][oldest_hash]
            cache_path = Path(entry['cache_path'])

            if cache_path.exists():
                cache_path.unlink()

            self.metadata['total_size_bytes'] -= entry['size_bytes']
            del self.metadata['entries'][oldest_hash]

            if self.verbose:
                print(f"⚠️ Evicted cache entry {oldest_hash}")

        self._save_metadata()

    def clear(self):
        """Clear entire cache"""
        for entry in self.metadata['entries'].values():
            cache_path = Path(entry['cache_path'])
            if cache_path.exists():
                cache_path.unlink()

        self.metadata = {
            'entries': {},
            'total_size_bytes': 0,
            'hit_count': 0,
            'miss_count': 0
        }
        self._save_metadata()

        if self.verbose:
            print("✅ Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_hits = self.metadata['hit_count']
        total_misses = self.metadata['miss_count']
        total_requests = total_hits + total_misses

        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

        return {
            'entries': len(self.metadata['entries']),
            'total_size_gb': self.metadata['total_size_bytes'] / (1024**3),
            'max_size_gb': self.max_size_bytes / (1024**3),
            'utilization': self.metadata['total_size_bytes'] / self.max_size_bytes,
            'hit_count': total_hits,
            'miss_count': total_misses,
            'hit_rate': hit_rate
        }


# ============================================================================
# GapDataset (PyTorch Dataset)
# ============================================================================

if TORCH_AVAILABLE:
    class GapDataset(Dataset):
        """
        PyTorch Dataset for semantic feature discovery training.

        Provides:
        - 200D feature vectors (INPUT)
        - Reconstruction gaps (TRAINING SIGNAL)
        - Original parameters (LABELS)
        - Locality information (from Agent 1)

        This dataset is used by Agent 5's GapDiscoveryTrainer.
        """

        def __init__(
            self,
            midi_files: List[Path],
            gap_analyzer: GapAnalyzer,
            cache: Optional[GapCache] = None,
            precompute: bool = True,
            normalize_features: bool = True,
            verbose: bool = False
        ):
            """
            Initialize Gap Dataset.

            Args:
                midi_files: List of MIDI file paths
                gap_analyzer: GapAnalyzer instance
                cache: Optional GapCache for efficient loading
                precompute: Precompute all gaps (recommended)
                normalize_features: Normalize features to zero mean, unit variance
                verbose: Print progress
            """
            self.midi_files = midi_files
            self.gap_analyzer = gap_analyzer
            self.cache = cache
            self.precompute = precompute
            self.normalize_features = normalize_features
            self.verbose = verbose

            # Storage for precomputed gaps
            self.gaps: List[ReconstructionGap] = []
            self.valid_indices: List[int] = []

            # Feature normalizer
            self.feature_normalizer = None

            if precompute:
                self._precompute_gaps()
                if normalize_features:
                    self._fit_normalizer()

        def _precompute_gaps(self):
            """Precompute all gaps and cache"""
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Precomputing gaps for {len(self.midi_files)} files")
                print(f"{'='*70}\n")

            iterator = self.midi_files
            if TQDM_AVAILABLE:
                iterator = tqdm(self.midi_files, desc="Precomputing gaps")

            for idx, midi_file in enumerate(iterator):
                # Try cache first
                gap = None
                if self.cache:
                    gap = self.cache.get(midi_file)

                # Compute if not cached
                if gap is None:
                    gap = self.gap_analyzer.compute_gap(midi_file)

                    # Cache the result
                    if self.cache and gap.success:
                        self.cache.put(midi_file, gap)

                # Store gap and index if successful
                if gap.success:
                    self.gaps.append(gap)
                    self.valid_indices.append(idx)
                else:
                    if self.verbose:
                        print(f"⚠️ Skipping {midi_file.name}: {gap.error_message}")

            if self.verbose:
                print(f"\n✅ Precomputed {len(self.gaps)}/{len(self.midi_files)} gaps")

                # Cache stats
                if self.cache:
                    stats = self.cache.get_stats()
                    print(f"\nCache Statistics:")
                    print(f"  Entries: {stats['entries']}")
                    print(f"  Size: {stats['total_size_gb']:.2f} GB / {stats['max_size_gb']:.2f} GB")
                    print(f"  Hit rate: {stats['hit_rate']:.2%}")

        def _fit_normalizer(self):
            """Fit feature normalizer on dataset"""
            if len(self.gaps) == 0:
                return

            # Collect all features
            all_features = np.array([gap.original_features for gap in self.gaps])

            # Fit normalizer
            self.feature_normalizer = FeatureNormalizer()
            self.feature_normalizer.fit(all_features)

            if self.verbose:
                print(f"✅ Feature normalizer fitted on {len(all_features)} samples")

        def __len__(self) -> int:
            """Dataset length"""
            if self.precompute:
                return len(self.gaps)
            return len(self.midi_files)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            """
            Get dataset item.

            Returns:
                Dictionary with:
                - 'features': Original 200D features (INPUT)
                - 'gaps': Reconstruction gaps (TRAINING SIGNAL)
                - 'parameters_flat': Flattened 50 parameters (LABELS)
                - 'file_id': File identifier
            """
            if self.precompute:
                gap = self.gaps[idx]
            else:
                midi_file = self.midi_files[idx]

                # Try cache
                gap = None
                if self.cache:
                    gap = self.cache.get(midi_file)

                # Compute if needed
                if gap is None:
                    gap = self.gap_analyzer.compute_gap(midi_file)
                    if self.cache and gap.success:
                        self.cache.put(midi_file, gap)

                if not gap.success:
                    raise RuntimeError(f"Gap computation failed for {midi_file}")

            # Normalize features if requested
            features = gap.original_features
            if self.normalize_features and self.feature_normalizer:
                features = self.feature_normalizer.transform(features.reshape(1, -1))[0]

            # Flatten parameters to single vector
            parameters_flat = self._flatten_parameters(gap.original_parameters)

            return {
                'features': torch.FloatTensor(features),
                'gaps': torch.FloatTensor(gap.feature_gaps),
                'parameters_flat': torch.FloatTensor(parameters_flat),
                'file_id': gap.file_id
            }

        def _flatten_parameters(self, parameters: Dict[str, Any]) -> np.ndarray:
            """
            Flatten hierarchical parameters to single vector.

            Expected structure:
            - level1_global: 8 params
            - level2_universal: 20 params (nested)
            - level3_genre_specific: 22 params (nested)

            Total: 50 params
            """
            flat_values = []

            # Process each level
            for level_key in ['level1_global', 'level2_universal', 'level3_genre_specific']:
                level_dict = parameters.get(level_key, {})
                level_flat = self._flatten_dict_to_array(level_dict)
                flat_values.extend(level_flat)

            # Pad or truncate to exactly 50 values
            if len(flat_values) < 50:
                flat_values.extend([0.0] * (50 - len(flat_values)))
            elif len(flat_values) > 50:
                flat_values = flat_values[:50]

            return np.array(flat_values, dtype=np.float32)

        def _flatten_dict_to_array(self, d: Dict) -> List[float]:
            """Recursively flatten dictionary to list of float values"""
            values = []

            for key in sorted(d.keys()):  # Sort for consistency
                value = d[key]

                if isinstance(value, dict):
                    values.extend(self._flatten_dict_to_array(value))
                elif isinstance(value, (int, float)):
                    values.append(float(value))
                elif isinstance(value, bool):
                    values.append(1.0 if value else 0.0)
                elif isinstance(value, str):
                    # Hash string to float in [0, 1]
                    hash_val = hash(value) % 1000000
                    values.append(hash_val / 1000000.0)
                elif isinstance(value, list):
                    # Use length as feature
                    values.append(float(len(value)))

            return values

        def get_dataloader(
            self,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0
        ) -> DataLoader:
            """Create PyTorch DataLoader"""
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_gap_dataset_from_directory(
    midi_dir: Path,
    cache_dir: Path,
    output_dir: Optional[Path] = None,
    max_files: Optional[int] = None,
    normalize: bool = True,
    verbose: bool = True
) -> 'GapDataset':
    """
    Create gap dataset from directory of MIDI files.

    Args:
        midi_dir: Directory containing MIDI files
        cache_dir: Directory for gap cache
        output_dir: Optional directory for saving statistics
        max_files: Maximum number of files to process
        normalize: Normalize features
        verbose: Print progress

    Returns:
        GapDataset ready for training
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required. Install with: pip install torch")

    # Find MIDI files
    midi_files = list(Path(midi_dir).glob('**/*.mid'))
    if max_files:
        midi_files = midi_files[:max_files]

    if verbose:
        print(f"Found {len(midi_files)} MIDI files in {midi_dir}")

    # Initialize components
    feature_extractor = OptimizedFeatureExtractor.from_selection_file(
        Path(__file__).parent.parent / 'feature_selection/output/selected_features_200_template.json'
    )

    parameter_extractor = HierarchicalParameterExtractorV2(verbose=False)

    midi_generator = ParameterMIDIGenerator(verbose=False)

    gap_analyzer = GapAnalyzer(
        feature_extractor=feature_extractor,
        parameter_extractor=parameter_extractor,
        midi_generator=midi_generator,
        verbose=verbose
    )

    # Create cache
    cache = GapCache(
        cache_dir=cache_dir,
        max_size_gb=10.0,
        verbose=verbose
    )

    # Analyze corpus gaps (optional, for statistics)
    if output_dir:
        statistics = gap_analyzer.analyze_corpus_gaps(
            midi_files,
            output_dir=output_dir,
            show_progress=True
        )

    # Create dataset
    dataset = GapDataset(
        midi_files=midi_files,
        gap_analyzer=gap_analyzer,
        cache=cache,
        precompute=True,
        normalize_features=normalize,
        verbose=verbose
    )

    return dataset


# ============================================================================
# Main / Demo
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GAP DATASET CREATION - AGENT 4")
    print("="*70)

    print("\nThis module provides:")
    print("  1. GapAnalyzer - Computes reconstruction gaps")
    print("  2. ParameterMIDIGenerator - Generates approximate MIDI from parameters")
    print("  3. GapCache - Efficient caching system")
    print("  4. GapDataset - PyTorch Dataset for training")
    print()

    print("Example usage:")
    print()
    print("  # Create dataset from directory")
    print("  dataset = create_gap_dataset_from_directory(")
    print("      midi_dir=Path('data/midi_corpus'),")
    print("      cache_dir=Path('data/gap_cache'),")
    print("      output_dir=Path('output/gap_analysis'),")
    print("      max_files=100,")
    print("      normalize=True")
    print("  )")
    print()
    print("  # Use in training loop (Agent 5)")
    print("  dataloader = dataset.get_dataloader(batch_size=32, shuffle=True)")
    print("  for batch in dataloader:")
    print("      features = batch['features']  # (batch_size, 200)")
    print("      gaps = batch['gaps']  # (batch_size, 200)")
    print("      parameters = batch['parameters_flat']  # (batch_size, 50)")
    print("      # Train semantic encoder...")
    print()

    print("✅ Gap Dataset system ready for use!")
    print()
    print("Integration points:")
    print("  - Uses: OptimizedFeatureExtractor (200D features)")
    print("  - Uses: HierarchicalParameterExtractorV2 (50 params)")
    print("  - Feeds: Agent 3 (SemanticFeatureEncoder)")
    print("  - Feeds: Agent 5 (GapDiscoveryTrainer)")
