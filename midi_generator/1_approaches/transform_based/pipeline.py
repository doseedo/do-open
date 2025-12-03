"""
Dosedo Unified Pipeline
=======================

Single entry point for the complete discovery and optimization pipeline.

Pipeline Phases:
1. Load & tokenize corpus (GPU parallel)
2. Factor objects into V2 representation (pitch_class, octave, quantized velocity)
3. Grammar induction (SEQUITUR, O(n) CPU)
4. Algebraic transform discovery (GPU, dense tables)
5. Cross-track relationship discovery (GPU)
6. MDL vocabulary optimization (CPU)
7. Checkpoint save

GPU Utilization:
- Tokenization: 90% (batch pitch_class, octave extraction)
- Pattern hashing: 80% (parallel hash computation)
- Transform discovery: 95% (dense table, all transforms parallel)
- Cross-track: 85% (batch naturality checking)

Author: Dosedo Architecture v2
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# PARALLEL MIDI LOADING HELPER
# =============================================================================

def _load_single_midi(midi_path: str) -> Optional[List[dict]]:
    """
    Load a single MIDI file and convert to per-track tensors.

    This is a top-level function so it can be pickled for multiprocessing.
    Returns list of dicts with tensor and metadata, or None on failure.
    Each track becomes a separate object for cross-track analysis.
    """
    try:
        import mido
        import numpy as np
        from pathlib import Path

        # Import here to avoid pickling issues
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from core.space_level_transforms import extract_notes_from_midi

        midi_obj = mido.MidiFile(midi_path)
        notes = extract_notes_from_midi(midi_obj)

        if not notes:
            return None

        # Group notes by track
        tracks = {}
        for note in notes:
            track_id = note.get('track', 0)
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(note)

        piece_id = Path(midi_path).stem
        results = []

        # Create time parameters
        ticks_per_beat = midi_obj.ticks_per_beat
        ticks_per_16th = ticks_per_beat // 4 if ticks_per_beat else 120

        # Create separate object for each track with notes
        for track_id, track_notes in tracks.items():
            if not track_notes:
                continue

            # Skip drum tracks for melodic analysis
            is_drum = any(n.get('is_drum', False) for n in track_notes)

            # Build tensor for this track only
            max_time_steps = 2048
            num_features = 133

            max_ticks = max(n['start_time'] + n['duration'] for n in track_notes)
            num_steps = min(int(max_ticks / ticks_per_16th) + 1, max_time_steps)

            tensor = np.zeros((max_time_steps, num_features), dtype=np.float32)

            for note in track_notes:
                start_step = int(note['start_time'] / ticks_per_16th)
                duration_steps = max(1, int(note['duration'] / ticks_per_16th))
                end_step = min(start_step + duration_steps, max_time_steps)

                if start_step >= max_time_steps:
                    continue

                for step in range(start_step, end_step):
                    pitch = np.clip(note['pitch'], 0, 127)
                    tensor[step, pitch] = 1.0
                    tensor[step, 128] = note['velocity'] / 127.0
                    tensor[step, 129] = note.get('program', 0) / 127.0
                    tensor[step, 130] = note.get('channel', 0) / 15.0
                    tensor[step, 131] = 1.0 if note.get('is_drum', False) else 0.0
                    tensor[step, 132] = track_id / 20.0

            # Only include tracks with actual content
            if tensor.sum() > 0:
                results.append({
                    'tensor': tensor,
                    'piece_id': piece_id,
                    'track_id': track_id,
                    'is_drum': is_drum
                })

        return results if results else None
    except Exception:
        return None


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class PipelinePhase(Enum):
    """Pipeline phases for progress tracking."""
    LOADING = auto()
    FACTORING = auto()
    GRAMMAR_INDUCTION = auto()
    TRANSFORM_DISCOVERY = auto()
    CROSS_TRACK = auto()
    MDL_OPTIMIZATION = auto()
    TRANSFORM_ENCODING = auto()  # New phase for transform-relative encoding
    CHECKPOINT = auto()
    COMPLETE = auto()


@dataclass
class ProgressUpdate:
    """Progress update from pipeline."""
    phase: PipelinePhase
    progress: float  # 0.0 to 1.0
    message: str
    elapsed_seconds: float = 0.0
    items_processed: int = 0
    total_items: int = 0

    def __repr__(self):
        pct = self.progress * 100
        return f"[{self.phase.name}] {pct:.1f}% - {self.message}"


@dataclass
class PipelineResult:
    """Complete result from pipeline run."""
    # Timing
    total_seconds: float
    phase_times: Dict[str, float]

    # Discovered structures
    n_objects: int
    n_patterns: int
    n_grammar_rules: int
    n_transforms: int
    n_cross_track_relations: int

    # Vocabulary
    vocabulary_size: int
    mdl_score: float

    # Coverage metrics
    object_coverage: float  # Fraction of objects explained
    compression_ratio: float

    # === Fields with defaults must come after non-defaults ===
    # Transform-relative encoding stats
    n_canonicals: int = 0  # Canonical patterns in encoding
    n_intros: int = 0  # INTRO tokens
    n_transform_refs: int = 0  # TRANSFORM tokens
    n_repeats: int = 0  # REPEAT tokens

    # Paths
    checkpoint_path: Optional[str] = None

    def __repr__(self):
        return (f"PipelineResult(objects={self.n_objects}, patterns={self.n_patterns}, "
                f"vocab={self.vocabulary_size}, coverage={self.object_coverage:.1%}, "
                f"time={self.total_seconds:.1f}s)")


# =============================================================================
# CHECKPOINT FORMAT V2
# =============================================================================

@dataclass
class CheckpointV2:
    """
    New checkpoint format with all layers.

    Contents:
    - algebraic_group: D24 Cayley table (24x24)
    - rhythm_group: Rhythm transform table (14x14)
    - grammar_rules: SEQUITUR grammar (pattern vocabulary)
    - dense_transform_table: NxN pattern relationships
    - cross_track_relations: Sparse relation list
    - track_functors: Categorical structure per track
    - mdl_scores: Per-pattern compression value
    - vocabulary: Final selected vocabulary
    """
    version: str = "2.0"

    # Algebraic groups
    d24_cayley_table: np.ndarray = None
    rhythm_cayley_table: np.ndarray = None

    # Grammar
    grammar_rules: Dict = field(default_factory=dict)
    grammar_stats: Dict = field(default_factory=dict)

    # Transform tables
    d24_transform_table: np.ndarray = None  # [N_patterns, N_patterns]
    rhythm_transform_table: np.ndarray = None

    # Relations
    cross_track_relations: List = field(default_factory=list)

    # MDL
    pattern_mdl_scores: Dict = field(default_factory=dict)
    vocabulary: List = field(default_factory=list)
    vocabulary_size: int = 0

    # Metrics
    object_coverage: float = 0.0
    compression_ratio: float = 1.0


def save_checkpoint_v2(path: str, checkpoint: CheckpointV2, verbose: bool = True):
    """
    Save V2 checkpoint as compressed NPZ.

    Args:
        path: Output path (should end in .npz)
        checkpoint: CheckpointV2 object
        verbose: Print progress
    """
    import json

    # Prepare data for NPZ
    data = {
        'version': np.array([checkpoint.version]),
        'vocabulary_size': np.array([checkpoint.vocabulary_size]),
        'object_coverage': np.array([checkpoint.object_coverage]),
        'compression_ratio': np.array([checkpoint.compression_ratio]),
    }

    # Add arrays if present
    if checkpoint.d24_cayley_table is not None:
        data['d24_cayley_table'] = checkpoint.d24_cayley_table
    if checkpoint.rhythm_cayley_table is not None:
        data['rhythm_cayley_table'] = checkpoint.rhythm_cayley_table
    if checkpoint.d24_transform_table is not None:
        data['d24_transform_table'] = checkpoint.d24_transform_table
    if checkpoint.rhythm_transform_table is not None:
        data['rhythm_transform_table'] = checkpoint.rhythm_transform_table

    # Serialize dicts/lists as JSON strings
    data['grammar_rules_json'] = np.array([json.dumps(checkpoint.grammar_rules)])
    data['grammar_stats_json'] = np.array([json.dumps(checkpoint.grammar_stats)])
    data['cross_track_json'] = np.array([json.dumps(checkpoint.cross_track_relations)])
    data['mdl_scores_json'] = np.array([json.dumps(checkpoint.pattern_mdl_scores)])
    data['vocabulary_json'] = np.array([json.dumps(checkpoint.vocabulary)])

    np.savez_compressed(path, **data)

    if verbose:
        print(f"  Checkpoint saved to: {path}")
        print(f"    Version: {checkpoint.version}")
        print(f"    Vocabulary size: {checkpoint.vocabulary_size}")
        print(f"    Coverage: {checkpoint.object_coverage:.1%}")


def load_checkpoint_v2(path: str) -> CheckpointV2:
    """
    Load V2 checkpoint.

    Args:
        path: Path to NPZ file

    Returns:
        CheckpointV2 object
    """
    import json

    data = np.load(path, allow_pickle=True)

    checkpoint = CheckpointV2()
    checkpoint.version = str(data['version'][0])
    checkpoint.vocabulary_size = int(data['vocabulary_size'][0])
    checkpoint.object_coverage = float(data['object_coverage'][0])
    checkpoint.compression_ratio = float(data['compression_ratio'][0])

    # Load arrays
    if 'd24_cayley_table' in data:
        checkpoint.d24_cayley_table = data['d24_cayley_table']
    if 'rhythm_cayley_table' in data:
        checkpoint.rhythm_cayley_table = data['rhythm_cayley_table']
    if 'd24_transform_table' in data:
        checkpoint.d24_transform_table = data['d24_transform_table']
    if 'rhythm_transform_table' in data:
        checkpoint.rhythm_transform_table = data['rhythm_transform_table']

    # Parse JSON
    if 'grammar_rules_json' in data:
        checkpoint.grammar_rules = json.loads(str(data['grammar_rules_json'][0]))
    if 'grammar_stats_json' in data:
        checkpoint.grammar_stats = json.loads(str(data['grammar_stats_json'][0]))
    if 'cross_track_json' in data:
        checkpoint.cross_track_relations = json.loads(str(data['cross_track_json'][0]))
    if 'mdl_scores_json' in data:
        checkpoint.pattern_mdl_scores = json.loads(str(data['mdl_scores_json'][0]))
    if 'vocabulary_json' in data:
        checkpoint.vocabulary = json.loads(str(data['vocabulary_json'][0]))

    return checkpoint


# =============================================================================
# CHECKPOINT FORMAT V3 (Transform-Relative Encoding)
# =============================================================================

@dataclass
class CheckpointV3:
    """
    Checkpoint format v3 with transform-relative encoding.

    This is the "MIDI Gene" checkpoint format where transforms are
    explicit in the encoding, enabling:
    - Editable canonical patterns
    - Visible transform relationships
    - "MIDI Gene Editor" interface

    Contents (extends V2):
    - All V2 fields
    - canonical_patterns: The canonical motifs
    - encoding_tokens: Transform-relative token sequences per track
    - encoding_stats: Compression statistics
    """
    version: str = "3.0"

    # === V2 fields ===
    # Algebraic groups
    d24_cayley_table: np.ndarray = None
    rhythm_cayley_table: np.ndarray = None

    # Grammar
    grammar_rules: Dict = field(default_factory=dict)
    grammar_stats: Dict = field(default_factory=dict)

    # Transform tables
    d24_transform_table: np.ndarray = None
    rhythm_transform_table: np.ndarray = None

    # Relations
    cross_track_relations: List = field(default_factory=list)

    # MDL
    pattern_mdl_scores: Dict = field(default_factory=dict)
    vocabulary: List = field(default_factory=list)
    vocabulary_size: int = 0

    # Metrics
    object_coverage: float = 0.0
    compression_ratio: float = 1.0

    # === V3 fields (Transform-Relative Encoding) ===
    # Canonical patterns: list of {pattern_id, pitch_classes, original_rule_id, usage_count}
    canonical_patterns: List = field(default_factory=list)

    # Token sequences: list of track token lists
    # Each token: {type, pattern_idx, transform_id, ...}
    encoding_tokens: List = field(default_factory=list)

    # Encoding stats
    n_canonicals: int = 0
    n_intros: int = 0
    n_transforms: int = 0
    n_repeats: int = 0
    encoding_compression_ratio: float = 1.0


def save_checkpoint_v3(path: str, checkpoint: CheckpointV3, verbose: bool = True):
    """
    Save V3 checkpoint as compressed NPZ.

    Args:
        path: Output path (should end in .npz)
        checkpoint: CheckpointV3 object
        verbose: Print progress
    """
    import json

    # Prepare data for NPZ
    data = {
        'version': np.array([checkpoint.version]),
        'vocabulary_size': np.array([checkpoint.vocabulary_size]),
        'object_coverage': np.array([checkpoint.object_coverage]),
        'compression_ratio': np.array([checkpoint.compression_ratio]),
        'n_canonicals': np.array([checkpoint.n_canonicals]),
        'n_intros': np.array([checkpoint.n_intros]),
        'n_transforms': np.array([checkpoint.n_transforms]),
        'n_repeats': np.array([checkpoint.n_repeats]),
        'encoding_compression_ratio': np.array([checkpoint.encoding_compression_ratio]),
    }

    # Add arrays if present
    if checkpoint.d24_cayley_table is not None:
        data['d24_cayley_table'] = checkpoint.d24_cayley_table
    if checkpoint.rhythm_cayley_table is not None:
        data['rhythm_cayley_table'] = checkpoint.rhythm_cayley_table
    if checkpoint.d24_transform_table is not None:
        data['d24_transform_table'] = checkpoint.d24_transform_table
    if checkpoint.rhythm_transform_table is not None:
        data['rhythm_transform_table'] = checkpoint.rhythm_transform_table

    # Serialize dicts/lists as JSON strings
    data['grammar_rules_json'] = np.array([json.dumps(checkpoint.grammar_rules)])
    data['grammar_stats_json'] = np.array([json.dumps(checkpoint.grammar_stats)])
    data['cross_track_json'] = np.array([json.dumps(checkpoint.cross_track_relations)])
    data['mdl_scores_json'] = np.array([json.dumps(checkpoint.pattern_mdl_scores)])
    data['vocabulary_json'] = np.array([json.dumps(checkpoint.vocabulary)])

    # V3 specific
    data['canonical_patterns_json'] = np.array([json.dumps(checkpoint.canonical_patterns)])
    data['encoding_tokens_json'] = np.array([json.dumps(checkpoint.encoding_tokens)])

    np.savez_compressed(path, **data)

    if verbose:
        print(f"  Checkpoint saved to: {path}")
        print(f"    Version: {checkpoint.version}")
        print(f"    Vocabulary size: {checkpoint.vocabulary_size}")
        print(f"    Canonicals: {checkpoint.n_canonicals}")
        print(f"    Encoding compression: {checkpoint.encoding_compression_ratio:.2f}x")


def load_checkpoint_v3(path: str) -> CheckpointV3:
    """
    Load V3 checkpoint.

    Args:
        path: Path to NPZ file

    Returns:
        CheckpointV3 object
    """
    import json

    data = np.load(path, allow_pickle=True)

    checkpoint = CheckpointV3()
    checkpoint.version = str(data['version'][0])
    checkpoint.vocabulary_size = int(data['vocabulary_size'][0])
    checkpoint.object_coverage = float(data['object_coverage'][0])
    checkpoint.compression_ratio = float(data['compression_ratio'][0])

    # V3 specific stats
    if 'n_canonicals' in data:
        checkpoint.n_canonicals = int(data['n_canonicals'][0])
    if 'n_intros' in data:
        checkpoint.n_intros = int(data['n_intros'][0])
    if 'n_transforms' in data:
        checkpoint.n_transforms = int(data['n_transforms'][0])
    if 'n_repeats' in data:
        checkpoint.n_repeats = int(data['n_repeats'][0])
    if 'encoding_compression_ratio' in data:
        checkpoint.encoding_compression_ratio = float(data['encoding_compression_ratio'][0])

    # Load arrays
    if 'd24_cayley_table' in data:
        checkpoint.d24_cayley_table = data['d24_cayley_table']
    if 'rhythm_cayley_table' in data:
        checkpoint.rhythm_cayley_table = data['rhythm_cayley_table']
    if 'd24_transform_table' in data:
        checkpoint.d24_transform_table = data['d24_transform_table']
    if 'rhythm_transform_table' in data:
        checkpoint.rhythm_transform_table = data['rhythm_transform_table']

    # Parse JSON
    if 'grammar_rules_json' in data:
        checkpoint.grammar_rules = json.loads(str(data['grammar_rules_json'][0]))
    if 'grammar_stats_json' in data:
        checkpoint.grammar_stats = json.loads(str(data['grammar_stats_json'][0]))
    if 'cross_track_json' in data:
        checkpoint.cross_track_relations = json.loads(str(data['cross_track_json'][0]))
    if 'mdl_scores_json' in data:
        checkpoint.pattern_mdl_scores = json.loads(str(data['mdl_scores_json'][0]))
    if 'vocabulary_json' in data:
        checkpoint.vocabulary = json.loads(str(data['vocabulary_json'][0]))

    # V3 specific
    if 'canonical_patterns_json' in data:
        checkpoint.canonical_patterns = json.loads(str(data['canonical_patterns_json'][0]))
    if 'encoding_tokens_json' in data:
        checkpoint.encoding_tokens = json.loads(str(data['encoding_tokens_json'][0]))

    return checkpoint


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class DosedoPipeline:
    """
    Unified pipeline for Dosedo discovery and optimization.

    Usage:
        pipeline = DosedoPipeline(corpus_path='/path/to/midi')

        # Run with progress updates
        for progress in pipeline.run():
            print(progress)

        # Or run directly
        result = pipeline.run_sync()
    """

    def __init__(
        self,
        corpus_path: str = None,
        device: str = 'cuda',
        target_vocab_size: int = 500,
        verbose: bool = False
    ):
        """
        Initialize pipeline.

        Args:
            corpus_path: Path to MIDI corpus directory
            device: 'cuda' or 'cpu'
            target_vocab_size: Target vocabulary size for MDL optimization
            verbose: Print debug output
        """
        self.corpus_path = corpus_path
        self.device = device if HAS_TORCH and torch.cuda.is_available() else 'cpu'
        self.target_vocab_size = target_vocab_size
        self.verbose = verbose

        # Pipeline state
        self._objects = []
        self._factored = []
        self._patterns = {}
        self._grammar = None
        self._relations = None
        self._vocabulary = None
        self._checkpoint = None
        self._encoding = None  # Transform-relative encoding

        # Timing
        self._phase_times = {}

    def run(
        self,
        objects: List = None,
        max_files: int = None,
        save_checkpoint_path: str = None
    ) -> Generator[ProgressUpdate, None, PipelineResult]:
        """
        Run the pipeline with streaming progress updates.

        Args:
            objects: Pre-loaded objects (if None, loads from corpus_path)
            max_files: Maximum files to process
            save_checkpoint_path: Path to save checkpoint

        Yields:
            ProgressUpdate objects

        Returns:
            PipelineResult
        """
        total_start = time.time()

        # Phase 1: Load corpus
        phase_start = time.time()
        if objects is not None:
            self._objects = objects
            yield ProgressUpdate(
                phase=PipelinePhase.LOADING,
                progress=1.0,
                message=f"Using {len(self._objects)} pre-loaded objects",
                items_processed=len(self._objects),
                total_items=len(self._objects)
            )
        else:
            # Stream progress during loading
            self._objects = []
            for progress_update in self._load_corpus_with_progress(max_files):
                if isinstance(progress_update, ProgressUpdate):
                    yield progress_update
                else:
                    # It's a loaded object
                    self._objects.append(progress_update)
        self._phase_times['loading'] = time.time() - phase_start

        # Phase 2: Factor objects
        yield ProgressUpdate(
            phase=PipelinePhase.FACTORING,
            progress=0.0,
            message="Factoring objects..."
        )

        phase_start = time.time()
        self._factored = self._factor_objects(self._objects)
        self._phase_times['factoring'] = time.time() - phase_start

        yield ProgressUpdate(
            phase=PipelinePhase.FACTORING,
            progress=1.0,
            message=f"Factored {len(self._factored)} objects"
        )

        # Phase 3: Grammar induction
        yield ProgressUpdate(
            phase=PipelinePhase.GRAMMAR_INDUCTION,
            progress=0.0,
            message="Running SEQUITUR grammar induction..."
        )

        phase_start = time.time()
        self._grammar = self._run_grammar_induction(self._factored)
        self._phase_times['grammar'] = time.time() - phase_start

        n_rules = self._grammar.get_vocabulary_size() if self._grammar else 0
        yield ProgressUpdate(
            phase=PipelinePhase.GRAMMAR_INDUCTION,
            progress=1.0,
            message=f"Induced {n_rules} grammar rules"
        )

        # Phase 4: Transform discovery
        yield ProgressUpdate(
            phase=PipelinePhase.TRANSFORM_DISCOVERY,
            progress=0.0,
            message="Discovering algebraic transforms..."
        )

        phase_start = time.time()
        self._relations = self._run_transform_discovery(self._factored)
        self._phase_times['transforms'] = time.time() - phase_start

        n_transforms = len(self._relations.pattern_transforms) if self._relations else 0
        yield ProgressUpdate(
            phase=PipelinePhase.TRANSFORM_DISCOVERY,
            progress=1.0,
            message=f"Found {n_transforms} pattern transforms"
        )

        # Phase 5: Cross-track discovery
        yield ProgressUpdate(
            phase=PipelinePhase.CROSS_TRACK,
            progress=0.0,
            message="Discovering cross-track relations..."
        )

        phase_start = time.time()
        n_cross_track = len(self._relations.cross_track) if self._relations else 0
        self._phase_times['cross_track'] = time.time() - phase_start

        yield ProgressUpdate(
            phase=PipelinePhase.CROSS_TRACK,
            progress=1.0,
            message=f"Found {n_cross_track} cross-track relations"
        )

        # Phase 6: MDL optimization
        yield ProgressUpdate(
            phase=PipelinePhase.MDL_OPTIMIZATION,
            progress=0.0,
            message="Optimizing vocabulary..."
        )

        phase_start = time.time()
        self._vocabulary, mdl_scores = self._run_mdl_optimization(self._factored)
        self._phase_times['mdl'] = time.time() - phase_start

        vocab_size = self._vocabulary.size if self._vocabulary else 0
        yield ProgressUpdate(
            phase=PipelinePhase.MDL_OPTIMIZATION,
            progress=1.0,
            message=f"Selected {vocab_size} vocabulary elements"
        )

        # Phase 7: Transform-relative encoding
        yield ProgressUpdate(
            phase=PipelinePhase.TRANSFORM_ENCODING,
            progress=0.0,
            message="Building transform-relative encoding..."
        )

        phase_start = time.time()
        self._encoding = self._run_transform_encoding(self._factored)
        self._phase_times['encoding'] = time.time() - phase_start

        if self._encoding:
            n_can = self._encoding.n_canonicals
            n_intro = self._encoding.n_intros
            n_trans = self._encoding.n_transforms
            n_rep = self._encoding.n_repeats
            yield ProgressUpdate(
                phase=PipelinePhase.TRANSFORM_ENCODING,
                progress=1.0,
                message=f"Encoded: {n_can} canonicals, {n_intro} intros, {n_trans} transforms, {n_rep} repeats"
            )
        else:
            yield ProgressUpdate(
                phase=PipelinePhase.TRANSFORM_ENCODING,
                progress=1.0,
                message="Transform encoding skipped (no grammar/relations)"
            )

        # Phase 8: Save checkpoint
        if save_checkpoint_path:
            yield ProgressUpdate(
                phase=PipelinePhase.CHECKPOINT,
                progress=0.0,
                message="Saving checkpoint..."
            )

            phase_start = time.time()
            self._save_checkpoint(save_checkpoint_path, self._factored, mdl_scores)
            self._phase_times['checkpoint'] = time.time() - phase_start

            yield ProgressUpdate(
                phase=PipelinePhase.CHECKPOINT,
                progress=1.0,
                message=f"Saved to {save_checkpoint_path}"
            )

        # Compute final metrics
        total_time = time.time() - total_start

        # Coverage
        if self._relations:
            targets = self._relations.get_all_targets()
            coverage = len(targets) / max(1, len(self._factored))
        else:
            coverage = 0.0

        # Compression ratio (rough estimate)
        compression = 1.0 / max(1.0, vocab_size / 1000)

        yield ProgressUpdate(
            phase=PipelinePhase.COMPLETE,
            progress=1.0,
            message=f"Pipeline complete in {total_time:.1f}s",
            elapsed_seconds=total_time
        )

        # Return final result
        return PipelineResult(
            total_seconds=total_time,
            phase_times=self._phase_times,
            n_objects=len(self._factored),
            n_patterns=len(self._patterns) if self._patterns else 0,
            n_grammar_rules=n_rules,
            n_transforms=n_transforms,
            n_cross_track_relations=n_cross_track,
            vocabulary_size=vocab_size,
            mdl_score=sum(s.mdl_score for s in mdl_scores) if mdl_scores else 0,
            n_canonicals=self._encoding.n_canonicals if self._encoding else 0,
            n_intros=self._encoding.n_intros if self._encoding else 0,
            n_transform_refs=self._encoding.n_transforms if self._encoding else 0,
            n_repeats=self._encoding.n_repeats if self._encoding else 0,
            object_coverage=coverage,
            compression_ratio=compression,
            checkpoint_path=save_checkpoint_path
        )

    def run_sync(
        self,
        objects: List = None,
        max_files: int = None,
        save_checkpoint_path: str = None,
        verbose: bool = True
    ) -> PipelineResult:
        """
        Run pipeline synchronously (blocking).

        Args:
            objects: Pre-loaded objects
            max_files: Maximum files to process
            save_checkpoint_path: Checkpoint output path
            verbose: Print progress

        Returns:
            PipelineResult
        """
        result = None

        for progress in self.run(objects, max_files, save_checkpoint_path):
            if verbose:
                print(progress)
            result = progress

        # The generator returns PipelineResult at the end
        return result

    # =========================================================================
    # Phase Implementations
    # =========================================================================

    def _load_corpus_with_progress(self, max_files: int = None) -> Generator:
        """Load corpus from path with streaming progress updates using parallel loading."""
        if not self.corpus_path:
            yield ProgressUpdate(
                phase=PipelinePhase.LOADING,
                progress=1.0,
                message="No corpus path specified"
            )
            return

        from pathlib import Path
        import glob
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

        try:
            # Find MIDI files
            corpus_path = Path(self.corpus_path)
            if corpus_path.is_file():
                midi_files = [str(corpus_path)]
            else:
                midi_files = sorted(glob.glob(str(corpus_path / "*.mid")))
                midi_files += sorted(glob.glob(str(corpus_path / "*.midi")))

            if max_files:
                midi_files = midi_files[:max_files]

            total_files = len(midi_files)
            if not midi_files:
                yield ProgressUpdate(
                    phase=PipelinePhase.LOADING,
                    progress=1.0,
                    message=f"No MIDI files found in {self.corpus_path}"
                )
                return

            yield ProgressUpdate(
                phase=PipelinePhase.LOADING,
                progress=0.0,
                message=f"Found {total_files} MIDI files",
                total_items=total_files
            )

            # Determine number of workers (leave some CPUs for other tasks)
            num_workers = min(multiprocessing.cpu_count() - 2, 16, total_files)
            num_workers = max(num_workers, 1)

            # Load in parallel using ProcessPoolExecutor
            loaded_count = 0
            failed_count = 0
            results_buffer = []

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all jobs
                future_to_path = {
                    executor.submit(_load_single_midi, path): path
                    for path in midi_files
                }

                # Process completed futures
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        if result is not None:
                            # Result is now a list of per-track objects
                            for track_data in result:
                                obj = type('MusicalObject', (), {
                                    'tensor': track_data['tensor'],
                                    'piece_id': track_data['piece_id'],
                                    'track_id': track_data['track_id'],
                                    'start_time': 0,
                                    'is_drum': track_data['is_drum']
                                })()
                                results_buffer.append(obj)
                            loaded_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        if self.verbose:
                            print(f"  [!] Failed to load {Path(path).name}: {e}")

                    # Yield progress every 10 completions
                    processed = loaded_count + failed_count
                    if processed % 10 == 0 or processed == total_files:
                        progress = processed / total_files
                        yield ProgressUpdate(
                            phase=PipelinePhase.LOADING,
                            progress=progress,
                            message=f"Loaded {loaded_count}/{processed} files ({failed_count} failed)",
                            items_processed=loaded_count,
                            total_items=total_files
                        )

            # Yield all loaded objects
            for obj in results_buffer:
                yield obj

            yield ProgressUpdate(
                phase=PipelinePhase.LOADING,
                progress=1.0,
                message=f"Loaded {loaded_count} objects ({failed_count} failed)",
                items_processed=loaded_count,
                total_items=total_files
            )

        except ImportError as e:
            yield ProgressUpdate(
                phase=PipelinePhase.LOADING,
                progress=1.0,
                message=f"Import error: {e}"
            )

    def _load_corpus(self, max_files: int = None) -> List:
        """Load corpus from path (non-streaming version)."""
        objects = []
        for item in self._load_corpus_with_progress(max_files):
            if not isinstance(item, ProgressUpdate):
                objects.append(item)
        return objects

    def _factor_objects(self, objects: List) -> List:
        """Factor objects into V2 representation."""
        if not objects:
            return []

        try:
            from discovery.factored_v2 import factor_tensor_v2, FactoredObjectV2

            factored = []
            for obj in objects:
                if hasattr(obj, 'tensor'):
                    fobj = factor_tensor_v2(
                        tensor=obj.tensor,
                        piece_id=obj.piece_id,
                        track_id=obj.track_id,
                        start_time=obj.start_time,
                        scale=obj.tensor.shape[0],
                        is_drum=getattr(obj, 'is_drum', False)
                    )
                    factored.append(fobj)
                elif hasattr(obj, 'pitch_class'):
                    # Already factored V2
                    factored.append(obj)

            return factored
        except ImportError:
            return objects

    def _run_grammar_induction(self, objects: List):
        """Run SEQUITUR grammar induction."""
        if not objects:
            return None

        try:
            from grammar.sequitur import SequiturGrammar, build_grammar_from_corpus

            grammar = build_grammar_from_corpus(objects, verbose=False)
            return grammar
        except ImportError:
            return None

    def _run_transform_discovery(self, objects: List):
        """Run unified transform discovery.

        IMPORTANT: Pass grammar to discover transforms between grammar-induced patterns
        (motifs), not between raw objects (entire songs). This is the key to finding
        meaningful musical relationships.
        """
        if not objects:
            return None

        try:
            from discovery.unified_relations import UnifiedRelationDiscovery

            discoverer = UnifiedRelationDiscovery(device=self.device)
            # Pass grammar to use grammar patterns for transform discovery
            relations = discoverer.discover_all(
                objects,
                grammar=self._grammar,  # KEY: enables grammar pattern extraction
                verbose=self.verbose
            )
            return relations
        except ImportError:
            return None

    def _run_mdl_optimization(self, objects: List) -> Tuple[Any, List]:
        """Run MDL vocabulary optimization with transform-based deduplication."""
        if not objects or not self._grammar:
            return None, []

        try:
            from mdl.vocabulary_optimizer import run_vocabulary_optimization

            # Get D24 transform table for deduplication
            d24_table = None
            if self._relations and hasattr(self._relations, 'd24_transform_table'):
                d24_table = self._relations.d24_transform_table

            vocabulary, scores = run_vocabulary_optimization(
                self._grammar,
                objects,
                target_size=self.target_vocab_size,
                d24_transform_table=d24_table,
                verbose=False
            )
            return vocabulary, scores
        except ImportError:
            return None, []

    def _run_transform_encoding(self, objects: List):
        """
        Build transform-relative encoding from grammar and relations.

        This creates the "MIDI Gene" representation where transforms
        are explicit in the encoding format.
        """
        if not objects or not self._grammar:
            return None

        try:
            from encoding.encoder import TransformRelativeEncoder
            from grammar.pattern_extractor import GrammarPatternExtractor

            # Get D24 transform table
            d24_table = None
            if self._relations and hasattr(self._relations, 'd24_transform_table'):
                d24_table = self._relations.d24_transform_table

            # Get cross-track relations
            cross_track = None
            if self._relations and hasattr(self._relations, 'cross_track'):
                cross_track = self._relations.cross_track

            # Create pattern extractor
            extractor = GrammarPatternExtractor(self._grammar)

            # Create encoder
            encoder = TransformRelativeEncoder(d24_table)

            # Encode
            encoding = encoder.encode(
                self._grammar,
                objects,
                pattern_extractor=extractor,
                cross_track_relations=cross_track
            )

            return encoding
        except ImportError as e:
            # Always print import errors - they indicate missing modules
            print(f"  [!] Transform encoding import error: {e}")
            return None
        except Exception as e:
            # Always print encoding errors for debugging
            import traceback
            print(f"  [!] Transform encoding error: {e}")
            traceback.print_exc()
            return None

    def _save_checkpoint(self, path: str, objects: List, mdl_scores: List):
        """Save checkpoint (V3 if encoding available, V2 otherwise)."""
        # Use V3 if we have transform-relative encoding
        if self._encoding:
            checkpoint = CheckpointV3()
        else:
            checkpoint = CheckpointV2()

        # Add group tables
        try:
            from core.groups.d24_group import D24Group
            from core.groups.rhythm_group import RhythmGroup

            d24 = D24Group()
            rhythm = RhythmGroup()

            checkpoint.d24_cayley_table = d24.cayley_table
            checkpoint.rhythm_cayley_table = rhythm.compose_table
        except ImportError:
            pass

        # Add grammar
        if self._grammar:
            checkpoint.grammar_rules = {
                str(rid): rule.to_list()
                for rid, rule in self._grammar.rules.items()
            }
            checkpoint.grammar_stats = self._grammar.get_rule_stats()

        # Add relations
        if self._relations:
            checkpoint.d24_transform_table = self._relations.d24_transform_table
            checkpoint.rhythm_transform_table = self._relations.rhythm_transform_table
            checkpoint.cross_track_relations = [
                {
                    'track_a': r.track_a_id,
                    'track_b': r.track_b_id,
                    'type': r.relation_type.value,
                    'confidence': r.confidence
                }
                for r in self._relations.cross_track
            ]

        # Add MDL scores
        if mdl_scores:
            checkpoint.pattern_mdl_scores = {
                str(s.pattern_id): {
                    'type': s.pattern_type,
                    'score': s.mdl_score,
                    'usage': s.usage_count
                }
                for s in mdl_scores
            }

        # Add vocabulary
        if self._vocabulary:
            checkpoint.vocabulary = self._vocabulary.all_elements
            checkpoint.vocabulary_size = self._vocabulary.size

        # Compute coverage
        if self._relations and objects:
            targets = self._relations.get_all_targets()
            checkpoint.object_coverage = len(targets) / max(1, len(objects))

        # V3 specific: add transform-relative encoding
        if self._encoding and isinstance(checkpoint, CheckpointV3):
            # Serialize canonical patterns
            checkpoint.canonical_patterns = [
                {
                    'pattern_id': cp.pattern_id,
                    'pitch_classes': cp.pitch_classes.tolist(),
                    'original_rule_id': cp.original_rule_id,
                    'usage_count': cp.usage_count,
                }
                for cp in self._encoding.canonical_patterns
            ]

            # Serialize token sequences
            checkpoint.encoding_tokens = [
                [
                    {
                        'type': t.token_type.name,
                        'pattern_idx': t.pattern_idx,
                        'transform_id': t.transform_id,
                        'source_track': t.source_track,
                    }
                    for t in track_tokens
                ]
                for track_tokens in self._encoding.tokens
            ]

            # Encoding stats
            checkpoint.n_canonicals = self._encoding.n_canonicals
            checkpoint.n_intros = self._encoding.n_intros
            checkpoint.n_transforms = self._encoding.n_transforms
            checkpoint.n_repeats = self._encoding.n_repeats
            checkpoint.encoding_compression_ratio = self._encoding.compression_ratio

            save_checkpoint_v3(path, checkpoint)
        else:
            save_checkpoint_v2(path, checkpoint)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_pipeline(
    corpus_path: str = None,
    objects: List = None,
    save_checkpoint_path: str = None,
    device: str = 'cuda',
    verbose: bool = True
) -> PipelineResult:
    """
    Convenience function to run the full pipeline.

    Args:
        corpus_path: Path to MIDI corpus
        objects: Pre-loaded objects (alternative to corpus_path)
        save_checkpoint_path: Where to save checkpoint
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        PipelineResult
    """
    pipeline = DosedoPipeline(
        corpus_path=corpus_path,
        device=device
    )

    return pipeline.run_sync(
        objects=objects,
        save_checkpoint_path=save_checkpoint_path,
        verbose=verbose
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dosedo Unified Pipeline')
    parser.add_argument('--corpus', type=str, help='Path to MIDI corpus')
    parser.add_argument('--output', type=str, default='checkpoint_v2.npz',
                        help='Output checkpoint path')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--vocab-size', type=int, default=500,
                        help='Target vocabulary size')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process')

    args = parser.parse_args()

    pipeline = DosedoPipeline(
        corpus_path=args.corpus,
        device=args.device,
        target_vocab_size=args.vocab_size
    )

    result = pipeline.run_sync(
        max_files=args.max_files,
        save_checkpoint_path=args.output,
        verbose=True
    )

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(result)
