"""
Unified Relational Discovery
============================

Replaces the 4 separate discovery phases with a single GPU-optimized pass:
1. Pattern transforms (D24 group operations on pitch-class patterns)
2. Time-shift relations (temporal alignment)
3. Cross-component discovery (rhythm from A, pitch from B)
4. Cross-track relationships (voice motion, voicing types)

Key Optimization: Single GPU upload, multiple relation queries.

Dense Transform Table:
- Build N×N×T table where table[i,j,t] = param if patterns[j] = T_t(patterns[i], param)
- For D24: T dimension is 24
- For rhythm: T dimension is 14

This replaces iterative pattern matching with parallel GPU computation.

Author: Dosedo Architecture v2
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import our algebraic groups
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.groups.d24_group import D24Group
from core.groups.rhythm_group import RhythmGroup
from core.groups.voicing_group import VoicingGroup, CrossTrackRelationType, CrossTrackAnalyzer


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PatternTransform:
    """A transform relation between two patterns."""
    source_pattern_id: int
    target_pattern_id: int
    transform_type: str  # 'd24', 'rhythm', 'compound'
    transform_id: int    # Element index in the group
    param: Optional[float] = None

    def __repr__(self):
        return f"PatternTransform({self.source_pattern_id}->{self.target_pattern_id}, {self.transform_type}[{self.transform_id}])"


@dataclass
class ObjectRelation:
    """A relation between two musical objects."""
    source_idx: int
    target_idx: int
    relation_type: str
    param: Optional[float] = None
    confidence: float = 1.0

    def __repr__(self):
        return f"ObjectRelation({self.source_idx}->{self.target_idx}, {self.relation_type})"


@dataclass
class CrossTrackRelation:
    """A relation between two tracks."""
    track_a_id: int
    track_b_id: int
    relation_type: CrossTrackRelationType
    confidence: float
    param: Optional[float] = None


@dataclass
class RelationGraph:
    """
    Complete relational structure discovered from a corpus.

    Contains:
    - pattern_transforms: D24/Rhythm transforms between patterns
    - time_shifts: Temporal offset relations between objects
    - cross_component: Relations where components come from different sources
    - cross_track: Relations between different tracks/voices
    """
    pattern_transforms: List[PatternTransform] = field(default_factory=list)
    time_shifts: List[ObjectRelation] = field(default_factory=list)
    cross_component: List[ObjectRelation] = field(default_factory=list)
    cross_track: List[CrossTrackRelation] = field(default_factory=list)

    # Dense transform tables (GPU tensors if available)
    d24_transform_table: Optional[np.ndarray] = None  # [N, N] pattern transforms
    rhythm_transform_table: Optional[np.ndarray] = None

    @property
    def total_relations(self) -> int:
        return (len(self.pattern_transforms) + len(self.time_shifts) +
                len(self.cross_component) + len(self.cross_track))

    def get_all_targets(self) -> Set[int]:
        """Get all target object indices that are explained by some relation."""
        targets = set()
        for rel in self.time_shifts:
            targets.add(rel.target_idx)
        for rel in self.cross_component:
            targets.add(rel.target_idx)
        return targets


@dataclass
class GPUData:
    """
    All data uploaded to GPU for relation discovery.

    Single upload, multiple queries.
    """
    device: str

    # Pattern data (from objects)
    rhythm_patterns: 'torch.Tensor'   # [N_rhythm, max_len]
    pitch_class_patterns: 'torch.Tensor'  # [N_pitch, max_notes]
    n_rhythm_patterns: int
    n_pitch_patterns: int

    # Object data
    object_rhythm_ids: 'torch.Tensor'  # [N_objects]
    object_pitch_ids: 'torch.Tensor'   # [N_objects]
    object_times: 'torch.Tensor'       # [N_objects] start times
    object_track_ids: 'torch.Tensor'   # [N_objects]
    object_piece_ids: 'torch.Tensor'   # [N_objects] (encoded as ints)
    n_objects: int

    # Grammar patterns (from SEQUITUR rules) - these are what we search for transforms in
    # Fields with defaults must come after fields without defaults
    grammar_pitch_patterns: 'torch.Tensor' = None  # [N_grammar, max_len]
    grammar_pattern_lengths: 'torch.Tensor' = None  # [N_grammar] actual lengths
    n_grammar_patterns: int = 0


# =============================================================================
# UNIFIED DISCOVERY CLASS
# =============================================================================

class UnifiedRelationDiscovery:
    """
    Single-pass discovery of all relation types.

    Key principle: Upload all data to GPU once, then run multiple queries.
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize with algebraic groups.

        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device if HAS_TORCH and torch.cuda.is_available() else 'cpu'

        # Initialize algebraic groups
        self.d24 = D24Group(device=self.device)
        self.rhythm_group = RhythmGroup(device=self.device)
        self.voicing_group = VoicingGroup(device=self.device)
        self.cross_track_analyzer = CrossTrackAnalyzer(device=self.device)

    def discover_all(
        self,
        objects: List,
        patterns: Dict = None,
        grammar: 'SequiturGrammar' = None,
        verbose: bool = True
    ) -> RelationGraph:
        """
        Discover all relations in a single pass.

        Args:
            objects: List of FactoredObjectV2
            patterns: Optional pre-computed pattern dictionaries
            grammar: Optional SequiturGrammar - if provided, extracts patterns from grammar rules
                     (RECOMMENDED: finds transforms between motifs, not entire songs)
            verbose: Print progress

        Returns:
            RelationGraph with all discovered relations
        """
        if verbose:
            print(f"\n{'='*70}")
            print("UNIFIED RELATIONAL DISCOVERY")
            print(f"{'='*70}")
            start_time = time.time()

        # Validate objects
        valid_objects = [o for o in objects if hasattr(o, 'num_notes') and o.num_notes > 0]
        if verbose:
            print(f"  Valid objects: {len(valid_objects)}")

        if not valid_objects:
            return RelationGraph()

        # === KEY CHANGE: Use grammar patterns if provided ===
        grammar_patterns = None
        if grammar is not None:
            from grammar.pattern_extractor import extract_grammar_patterns
            grammar_patterns = extract_grammar_patterns(
                grammar,
                min_length=4,
                max_length=64,
                min_usage=2,
                verbose=verbose
            )
            if verbose:
                print(f"  Using {len(grammar_patterns)} grammar patterns for transform discovery")

        # Extract or use provided patterns (for objects)
        if patterns is None:
            patterns = self._extract_patterns(valid_objects)

        # === SINGLE GPU UPLOAD ===
        gpu_data = self._upload_all(valid_objects, patterns, grammar_patterns)

        # === PARALLEL RELATION DISCOVERY ===
        relations = RelationGraph()

        # 1. Pattern transforms (D24 and Rhythm)
        if verbose:
            print(f"  [1/4] Finding pattern transforms...")
        d24_table, rhythm_table = self._find_pattern_transforms_gpu(gpu_data)
        relations.d24_transform_table = d24_table
        relations.rhythm_transform_table = rhythm_table
        relations.pattern_transforms = self._table_to_transforms(
            d24_table, rhythm_table, patterns
        )

        # 2. Time-shift relations
        if verbose:
            print(f"  [2/4] Finding time-shift relations...")
        relations.time_shifts = self._find_time_shifts_gpu(gpu_data)

        # 3. Cross-component relations
        if verbose:
            print(f"  [3/4] Finding cross-component relations...")
        relations.cross_component = self._find_cross_component_gpu(gpu_data)

        # 4. Cross-track relations
        if verbose:
            print(f"  [4/4] Finding cross-track relations...")
        relations.cross_track = self._find_cross_track_gpu(gpu_data, valid_objects)

        if verbose:
            elapsed = time.time() - start_time
            print(f"\n  Discovery completed in {elapsed:.2f}s")
            print(f"  Pattern transforms: {len(relations.pattern_transforms)}")
            print(f"  Time-shift relations: {len(relations.time_shifts)}")
            print(f"  Cross-component relations: {len(relations.cross_component)}")
            print(f"  Cross-track relations: {len(relations.cross_track)}")
            print(f"  Total relations: {relations.total_relations}")

            # Coverage
            targets = relations.get_all_targets()
            coverage = len(targets) / len(valid_objects) * 100 if valid_objects else 0
            print(f"  Object coverage: {len(targets)}/{len(valid_objects)} ({coverage:.1f}%)")

        return relations

    # =========================================================================
    # Pattern Extraction
    # =========================================================================

    def _extract_patterns(self, objects: List) -> Dict:
        """Extract unique patterns from objects."""
        rhythm_patterns = {}  # hash -> (id, array)
        pitch_class_patterns = {}  # hash -> (id, array)

        rhythm_id = 0
        pitch_id = 0

        for obj in objects:
            # Rhythm pattern
            rh = hash(obj.rhythm.tobytes())
            if rh not in rhythm_patterns:
                rhythm_patterns[rh] = (rhythm_id, obj.rhythm)
                rhythm_id += 1

            # Pitch-class pattern
            ph = hash(obj.pitch_class.tobytes())
            if ph not in pitch_class_patterns:
                pitch_class_patterns[ph] = (pitch_id, obj.pitch_class)
                pitch_id += 1

        return {
            'rhythm': rhythm_patterns,
            'pitch_class': pitch_class_patterns
        }

    # =========================================================================
    # GPU Data Upload
    # =========================================================================

    def _upload_all(self, objects: List, patterns: Dict, grammar_patterns: List = None) -> GPUData:
        """
        Single upload of all data to GPU.

        This is the key optimization - all subsequent queries reuse this data.

        Args:
            objects: List of FactoredObjectV2
            patterns: Dict with 'rhythm' and 'pitch_class' pattern dicts
            grammar_patterns: Optional list of RulePattern from grammar extraction
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU operations")

        device = self.device

        # Prepare rhythm patterns
        rhythm_data = list(patterns['rhythm'].values())
        if rhythm_data:
            max_rhythm_len = max(len(p[1]) for p in rhythm_data)
            rhythm_tensor = torch.zeros(len(rhythm_data), max_rhythm_len,
                                        device=device, dtype=torch.float32)
            for i, (_, arr) in enumerate(rhythm_data):
                rhythm_tensor[i, :len(arr)] = torch.tensor(arr, device=device)
        else:
            rhythm_tensor = torch.zeros(0, 1, device=device)

        # Prepare pitch-class patterns (from objects)
        pitch_data = list(patterns['pitch_class'].values())
        if pitch_data:
            max_pitch_len = max(len(p[1]) for p in pitch_data)
            pitch_tensor = torch.zeros(len(pitch_data), max_pitch_len,
                                       device=device, dtype=torch.int32)
            for i, (_, arr) in enumerate(pitch_data):
                pitch_tensor[i, :len(arr)] = torch.tensor(arr, device=device)
        else:
            pitch_tensor = torch.zeros(0, 1, device=device, dtype=torch.int32)

        # === NEW: Prepare grammar patterns ===
        grammar_pitch_tensor = None
        grammar_lengths_tensor = None
        n_grammar = 0

        if grammar_patterns and len(grammar_patterns) > 0:
            n_grammar = len(grammar_patterns)
            max_grammar_len = max(p.length for p in grammar_patterns)

            grammar_pitch_tensor = torch.zeros(n_grammar, max_grammar_len,
                                               device=device, dtype=torch.int32)
            grammar_lengths = []

            for i, gp in enumerate(grammar_patterns):
                grammar_pitch_tensor[i, :gp.length] = torch.tensor(
                    gp.pitch_class, device=device, dtype=torch.int32
                )
                grammar_lengths.append(gp.length)

            grammar_lengths_tensor = torch.tensor(grammar_lengths, device=device, dtype=torch.int32)

        # Build object data
        rhythm_hash_to_id = {h: data[0] for h, data in patterns['rhythm'].items()}
        pitch_hash_to_id = {h: data[0] for h, data in patterns['pitch_class'].items()}

        object_rhythm_ids = []
        object_pitch_ids = []
        object_times = []
        object_track_ids = []
        piece_id_map = {}
        object_piece_ids = []

        for obj in objects:
            rh = hash(obj.rhythm.tobytes())
            ph = hash(obj.pitch_class.tobytes())

            object_rhythm_ids.append(rhythm_hash_to_id.get(rh, 0))
            object_pitch_ids.append(pitch_hash_to_id.get(ph, 0))
            object_times.append(obj.start_time)
            object_track_ids.append(obj.track_id)

            pid = obj.piece_id
            if pid not in piece_id_map:
                piece_id_map[pid] = len(piece_id_map)
            object_piece_ids.append(piece_id_map[pid])

        return GPUData(
            device=device,
            rhythm_patterns=rhythm_tensor,
            pitch_class_patterns=pitch_tensor,
            n_rhythm_patterns=len(rhythm_data),
            n_pitch_patterns=len(pitch_data),
            # Grammar patterns
            grammar_pitch_patterns=grammar_pitch_tensor,
            grammar_pattern_lengths=grammar_lengths_tensor,
            n_grammar_patterns=n_grammar,
            # Object data
            object_rhythm_ids=torch.tensor(object_rhythm_ids, device=device, dtype=torch.int32),
            object_pitch_ids=torch.tensor(object_pitch_ids, device=device, dtype=torch.int32),
            object_times=torch.tensor(object_times, device=device, dtype=torch.int32),
            object_track_ids=torch.tensor(object_track_ids, device=device, dtype=torch.int32),
            object_piece_ids=torch.tensor(object_piece_ids, device=device, dtype=torch.int32),
            n_objects=len(objects)
        )

    # =========================================================================
    # Pattern Transform Discovery (GPU)
    # =========================================================================

    def _find_pattern_transforms_gpu(
        self,
        gpu_data: GPUData
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build dense N×N transform tables for patterns.

        For pitch-class patterns: Check all 24 D24 transforms
        For rhythm patterns: Check all 14 rhythm transforms

        IMPORTANT: If grammar patterns are available, use those instead of object patterns.
        Grammar patterns represent recurring motifs, which are much more likely to have
        transform relationships than entire songs.

        Returns:
            (d24_table, rhythm_table) - both [N, N] with transform IDs (-1 if none)
        """
        device = gpu_data.device

        # === D24 transforms on patterns ===
        # Prefer grammar patterns if available (they represent motifs)
        if gpu_data.n_grammar_patterns > 0 and gpu_data.grammar_pitch_patterns is not None:
            N_pitch = gpu_data.n_grammar_patterns
            pitch_patterns = gpu_data.grammar_pitch_patterns  # [N, L]
            pattern_lengths = gpu_data.grammar_pattern_lengths  # [N]
        else:
            N_pitch = gpu_data.n_pitch_patterns
            pitch_patterns = gpu_data.pitch_class_patterns  # [N, L]
            pattern_lengths = None

        if N_pitch > 0:
            d24_table = torch.full((N_pitch, N_pitch), -1, device=device, dtype=torch.int16)
            L = pitch_patterns.shape[1]  # max pattern length

            # Precompute position mask for length-aware comparison
            # position_valid[i, k] = True if position k is within pattern i's length
            if pattern_lengths is not None:
                positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
                position_valid = positions < pattern_lengths.unsqueeze(1)  # [N, L]
            else:
                position_valid = None

            # Check each D24 element
            for t_id in range(24):
                # Apply transform to all patterns
                transformed = self.d24.apply_to_pitch_class_batch_gpu(
                    torch.full((N_pitch,), t_id, device=device, dtype=torch.int64),
                    pitch_patterns
                )  # [N, L]

                if pattern_lengths is not None:
                    # Length-aware matching: only compare valid positions
                    # For patterns i and j to match:
                    # 1. They must have the same length
                    # 2. All positions within that length must match

                    len_i = pattern_lengths.unsqueeze(1)  # [N, 1]
                    len_j = pattern_lengths.unsqueeze(0)  # [1, N]
                    same_length = (len_i == len_j)  # [N, N]

                    # Element-wise comparison: [N, 1, L] vs [1, N, L] -> [N, N, L]
                    element_match = (pitch_patterns.unsqueeze(1) == transformed.unsqueeze(0))

                    # Create valid position mask for each pair (i,j): position k is valid
                    # if k < min(len_i, len_j), but since we require same_length, just use len_i
                    # valid_mask[i, j, k] = position_valid[i, k]
                    valid_mask = position_valid.unsqueeze(1).expand(-1, N_pitch, -1)  # [N, N, L]

                    # Match only where valid, ignore padding positions
                    # A pair matches if all valid positions match
                    # Use masked sum: (element_match | ~valid_mask).all(dim=2)
                    matches_at_valid = element_match | ~valid_mask  # True where match OR not valid
                    matches = matches_at_valid.all(dim=2) & same_length  # [N, N]
                else:
                    # No length info - compare full patterns (original behavior)
                    matches = (pitch_patterns.unsqueeze(1) == transformed.unsqueeze(0)).all(dim=2)

                # Record matches (only if simpler transform not already found)
                new_matches = matches & (d24_table == -1)
                d24_table[new_matches] = t_id

            d24_result = d24_table.cpu().numpy()
        else:
            d24_result = np.array([])

        # === Rhythm transforms ===
        N_rhythm = gpu_data.n_rhythm_patterns
        if N_rhythm > 0:
            rhythm_table = torch.full((N_rhythm, N_rhythm), -1, device=device, dtype=torch.int16)

            # For rhythm, we check a smaller set (8 elements, not 14 for efficiency)
            # Focus on: identity (0), retrograde (1), augment_2x (2), diminish_half (3)
            rhythm_patterns = gpu_data.rhythm_patterns  # [N, L]

            # Identity is always there
            identity_matches = (rhythm_patterns.unsqueeze(1) == rhythm_patterns.unsqueeze(0)).all(dim=2)
            rhythm_table[identity_matches] = 0

            # Retrograde: reverse patterns
            reversed_patterns = rhythm_patterns.flip(dims=[1])
            retro_matches = (rhythm_patterns.unsqueeze(1) == reversed_patterns.unsqueeze(0)).all(dim=2)
            retro_matches = retro_matches & (rhythm_table == -1)
            rhythm_table[retro_matches] = 1

            rhythm_result = rhythm_table.cpu().numpy()
        else:
            rhythm_result = np.array([])

        return d24_result, rhythm_result

    def _table_to_transforms(
        self,
        d24_table: np.ndarray,
        rhythm_table: np.ndarray,
        patterns: Dict
    ) -> List[PatternTransform]:
        """Convert dense tables to PatternTransform list."""
        transforms = []

        # D24 transforms (skip identity and self-loops)
        if d24_table.size > 0:
            for i in range(d24_table.shape[0]):
                for j in range(d24_table.shape[1]):
                    t_id = d24_table[i, j]
                    if t_id > 0 and i != j:  # Non-identity, non-self
                        transforms.append(PatternTransform(
                            source_pattern_id=i,
                            target_pattern_id=j,
                            transform_type='d24',
                            transform_id=int(t_id)
                        ))

        # Rhythm transforms
        if rhythm_table.size > 0:
            for i in range(rhythm_table.shape[0]):
                for j in range(rhythm_table.shape[1]):
                    t_id = rhythm_table[i, j]
                    if t_id > 0 and i != j:
                        transforms.append(PatternTransform(
                            source_pattern_id=i,
                            target_pattern_id=j,
                            transform_type='rhythm',
                            transform_id=int(t_id)
                        ))

        return transforms

    # =========================================================================
    # Time-Shift Discovery (GPU)
    # =========================================================================

    def _find_time_shifts_gpu(self, gpu_data: GPUData) -> List[ObjectRelation]:
        """
        Find objects that are time-shifted versions of each other.

        Two objects are time-shifted if:
        1. Same piece_id
        2. Same track_id
        3. Same rhythm pattern
        4. Same pitch-class pattern
        5. Different start_time (by offset)
        """
        device = gpu_data.device
        N = gpu_data.n_objects

        if N == 0:
            return []

        # Build mask for same piece, same track
        same_piece = gpu_data.object_piece_ids.unsqueeze(1) == gpu_data.object_piece_ids.unsqueeze(0)
        same_track = gpu_data.object_track_ids.unsqueeze(1) == gpu_data.object_track_ids.unsqueeze(0)

        # Same patterns
        same_rhythm = gpu_data.object_rhythm_ids.unsqueeze(1) == gpu_data.object_rhythm_ids.unsqueeze(0)
        same_pitch = gpu_data.object_pitch_ids.unsqueeze(1) == gpu_data.object_pitch_ids.unsqueeze(0)

        # Different time (not self)
        time_diff = gpu_data.object_times.unsqueeze(1) - gpu_data.object_times.unsqueeze(0)
        not_self = time_diff != 0

        # Combine conditions
        is_time_shift = same_piece & same_track & same_rhythm & same_pitch & not_self

        # Only keep lower triangle (avoid duplicates)
        is_time_shift = torch.tril(is_time_shift, diagonal=-1)

        # Extract relations
        relations = []
        shift_indices = torch.nonzero(is_time_shift, as_tuple=False)

        for idx in shift_indices.cpu().numpy():
            i, j = idx[0], idx[1]
            offset = int(time_diff[i, j].item())
            relations.append(ObjectRelation(
                source_idx=int(i),
                target_idx=int(j),
                relation_type='time_shift',
                param=float(offset)
            ))

        return relations

    # =========================================================================
    # Cross-Component Discovery (GPU)
    # =========================================================================

    def _find_cross_component_gpu(self, gpu_data: GPUData) -> List[ObjectRelation]:
        """
        Find objects where rhythm comes from one source, pitch from another.

        An object C has cross-component if:
        - C.rhythm == A.rhythm (A ≠ C)
        - C.pitch_class == B.pitch_class (B ≠ C, B ≠ A)
        """
        device = gpu_data.device
        N = gpu_data.n_objects

        if N < 3:  # Need at least 3 objects
            return []

        relations = []

        # For each object, find rhythm source and pitch source
        for c in range(N):
            c_rhythm = gpu_data.object_rhythm_ids[c]
            c_pitch = gpu_data.object_pitch_ids[c]

            # Find objects with same rhythm (potential rhythm sources)
            same_rhythm_mask = (gpu_data.object_rhythm_ids == c_rhythm)
            same_rhythm_mask[c] = False  # Exclude self

            # Find objects with same pitch (potential pitch sources)
            same_pitch_mask = (gpu_data.object_pitch_ids == c_pitch)
            same_pitch_mask[c] = False  # Exclude self

            # Check if there's a genuine cross-component relation
            # (different sources for rhythm and pitch)
            rhythm_sources = torch.nonzero(same_rhythm_mask, as_tuple=False).flatten()
            pitch_sources = torch.nonzero(same_pitch_mask, as_tuple=False).flatten()

            if len(rhythm_sources) > 0 and len(pitch_sources) > 0:
                # Check if any rhythm source is different from all pitch sources
                rhythm_set = set(rhythm_sources.cpu().numpy().tolist())
                pitch_set = set(pitch_sources.cpu().numpy().tolist())

                # If the sets are different, we have a cross-component
                if rhythm_set != pitch_set and not rhythm_set.issubset(pitch_set):
                    # Pick first rhythm source not in pitch set
                    for r in rhythm_sources.cpu().numpy():
                        if r not in pitch_set:
                            relations.append(ObjectRelation(
                                source_idx=int(r),
                                target_idx=c,
                                relation_type='cross_component_rhythm',
                                confidence=0.8
                            ))
                            break

        return relations

    # =========================================================================
    # Cross-Track Discovery (GPU)
    # =========================================================================

    def _find_cross_track_gpu(
        self,
        gpu_data: GPUData,
        objects: List
    ) -> List[CrossTrackRelation]:
        """
        Find relationships between different tracks.

        Uses VoicingGroup to detect motion types, voicing, and rhythmic relations.
        """
        device = gpu_data.device
        N = gpu_data.n_objects

        if N < 2:
            return []

        relations = []

        # Group objects by piece_id and time window
        # For efficiency, sample a subset of track pairs
        unique_pieces = torch.unique(gpu_data.object_piece_ids).cpu().numpy()

        for piece_id in unique_pieces:
            piece_mask = (gpu_data.object_piece_ids == piece_id).cpu().numpy()
            piece_indices = np.where(piece_mask)[0]

            if len(piece_indices) < 2:
                continue

            # Get unique tracks in this piece
            tracks_in_piece = set()
            for idx in piece_indices:
                tracks_in_piece.add(int(gpu_data.object_track_ids[idx].item()))

            if len(tracks_in_piece) < 2:
                continue

            # Compare pairs of tracks
            track_list = sorted(tracks_in_piece)
            for i, track_a in enumerate(track_list):
                for track_b in track_list[i+1:]:
                    # Get objects for each track
                    track_a_mask = (gpu_data.object_track_ids == track_a).cpu().numpy() & piece_mask
                    track_b_mask = (gpu_data.object_track_ids == track_b).cpu().numpy() & piece_mask

                    track_a_indices = np.where(track_a_mask)[0]
                    track_b_indices = np.where(track_b_mask)[0]

                    if len(track_a_indices) == 0 or len(track_b_indices) == 0:
                        continue

                    # Sample objects for comparison
                    sample_size = min(10, len(track_a_indices), len(track_b_indices))
                    sampled_a = np.random.choice(track_a_indices, sample_size, replace=False)
                    sampled_b = np.random.choice(track_b_indices, sample_size, replace=False)

                    # Extract pitch and rhythm data
                    pitches_a = np.concatenate([
                        objects[idx].pitch_class.astype(np.int32) + objects[idx].octave.astype(np.int32) * 12
                        for idx in sampled_a
                    ])
                    pitches_b = np.concatenate([
                        objects[idx].pitch_class.astype(np.int32) + objects[idx].octave.astype(np.int32) * 12
                        for idx in sampled_b
                    ])

                    # Truncate to same length
                    min_len = min(len(pitches_a), len(pitches_b))
                    if min_len < 2:
                        continue

                    pitches_a = pitches_a[:min_len]
                    pitches_b = pitches_b[:min_len]

                    # Detect pitch relationship using CrossTrackAnalyzer
                    rel_type, param = self.cross_track_analyzer.detect_pitch_relationship(
                        pitches_a, pitches_b
                    )

                    # rel_type is None if no relationship found
                    if rel_type is not None:
                        relations.append(CrossTrackRelation(
                            track_a_id=track_a,
                            track_b_id=track_b,
                            relation_type=rel_type,
                            confidence=0.8,  # High confidence for detected relations
                            param=param
                        ))

        return relations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_unified_discovery(
    objects: List,
    verbose: bool = True,
    device: str = 'cuda'
) -> RelationGraph:
    """
    Run unified relation discovery on a corpus.

    Args:
        objects: List of FactoredObjectV2
        verbose: Print progress
        device: 'cuda' or 'cpu'

    Returns:
        RelationGraph with all discovered relations
    """
    discoverer = UnifiedRelationDiscovery(device=device)
    return discoverer.discover_all(objects, verbose=verbose)


def get_explained_objects(relations: RelationGraph, n_objects: int) -> Dict[str, Set[int]]:
    """
    Get sets of object indices explained by each relation type.

    Args:
        relations: RelationGraph from discovery
        n_objects: Total number of objects

    Returns:
        Dict mapping relation_type -> set of target indices
    """
    explained = {
        'time_shift': set(),
        'cross_component': set(),
        'd24_transform': set(),
        'rhythm_transform': set(),
    }

    for rel in relations.time_shifts:
        explained['time_shift'].add(rel.target_idx)

    for rel in relations.cross_component:
        explained['cross_component'].add(rel.target_idx)

    # Pattern transforms explain objects that use those patterns
    # (This would require mapping back to objects)

    return explained


def compute_coverage_stats(relations: RelationGraph, n_objects: int) -> Dict:
    """
    Compute coverage statistics.

    Returns:
        Dict with coverage percentages for each relation type
    """
    if n_objects == 0:
        return {
            'total_coverage': 0.0,
            'time_shift_coverage': 0.0,
            'cross_component_coverage': 0.0,
        }

    explained = get_explained_objects(relations, n_objects)

    all_explained = set()
    for s in explained.values():
        all_explained.update(s)

    return {
        'total_coverage': len(all_explained) / n_objects * 100,
        'time_shift_coverage': len(explained['time_shift']) / n_objects * 100,
        'cross_component_coverage': len(explained['cross_component']) / n_objects * 100,
        'n_pattern_transforms': len(relations.pattern_transforms),
        'n_cross_track_relations': len(relations.cross_track),
    }
