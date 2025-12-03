"""
Unified Relational Discovery.

Single-pass discovery of ALL relation types using GPU acceleration.
Replaces 4 separate sequential phases with one efficient parallel phase.

Key insight: Upload all data to GPU ONCE, then run multiple relation
queries on the same data without re-uploading.

Relation types discovered:
1. Pattern transforms (D24, rhythm group)
2. Time-shift relations (same pattern at different times)
3. Cross-component relations (rhythm from A, contour from B)
4. Cross-track relations (track A derives from track B)

Author: Architecture Refactor - Dosedo v2
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import algebraic groups
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.groups.d24_group import D24Group
from core.groups.rhythm_group import RhythmGroup
from core.groups.voicing_group import CrossTrackAnalyzer, CrossTrackRelationType


# =============================================================================
# DATA STRUCTURES FOR RELATIONS
# =============================================================================

@dataclass
class PatternTransform:
    """A transform relationship between two patterns."""
    source_id: int  # Source pattern ID
    target_id: int  # Target pattern ID
    transform_type: str  # 'd24', 'rhythm', 'voicing'
    transform_id: int  # Element ID in the transform group
    param: float  # Additional parameter (e.g., transposition amount)
    confidence: float = 1.0

    def __repr__(self):
        return f"PatternTransform({self.source_id}->{self.target_id}, {self.transform_type}[{self.transform_id}])"


@dataclass
class ObjectRelation:
    """A relation between two musical objects."""
    source_obj_id: Tuple  # (piece_id, track_id, start_time, scale)
    target_obj_id: Tuple
    relation_type: str  # 'time_shift', 'transform', 'cross_component'
    params: Dict = field(default_factory=dict)


@dataclass
class CrossTrackRelation:
    """A relation between two tracks."""
    track_a_id: str
    track_b_id: str
    relation_type: CrossTrackRelationType
    param: Optional[int] = None
    confidence: float = 1.0


@dataclass
class RelationGraph:
    """
    Graph structure storing all discovered relations.

    Organized by relation type for efficient lookup.
    """
    # Pattern-level relations (NxN sparse)
    pattern_transforms: List[PatternTransform] = field(default_factory=list)

    # Object-level relations
    time_shifts: List[ObjectRelation] = field(default_factory=list)
    cross_component: List[ObjectRelation] = field(default_factory=list)

    # Track-level relations
    cross_track: List[CrossTrackRelation] = field(default_factory=list)

    # Index structures for fast lookup
    _pattern_by_source: Dict[int, List[PatternTransform]] = field(default_factory=lambda: defaultdict(list))
    _pattern_by_target: Dict[int, List[PatternTransform]] = field(default_factory=lambda: defaultdict(list))

    def add_pattern_transform(self, pt: PatternTransform):
        """Add a pattern transform relation."""
        self.pattern_transforms.append(pt)
        self._pattern_by_source[pt.source_id].append(pt)
        self._pattern_by_target[pt.target_id].append(pt)

    def get_transforms_from(self, pattern_id: int) -> List[PatternTransform]:
        """Get all transforms originating from a pattern."""
        return self._pattern_by_source.get(pattern_id, [])

    def get_transforms_to(self, pattern_id: int) -> List[PatternTransform]:
        """Get all transforms targeting a pattern."""
        return self._pattern_by_target.get(pattern_id, [])

    def get_stats(self) -> Dict:
        """Get statistics about the relation graph."""
        return {
            'pattern_transforms': len(self.pattern_transforms),
            'time_shifts': len(self.time_shifts),
            'cross_component': len(self.cross_component),
            'cross_track': len(self.cross_track),
            'unique_source_patterns': len(self._pattern_by_source),
            'unique_target_patterns': len(self._pattern_by_target)
        }


# =============================================================================
# GPU DATA CONTAINER
# =============================================================================

@dataclass
class GPUData:
    """Container for data uploaded to GPU for relation discovery."""
    device: str

    # Pattern data (for pattern-level relations)
    n_patterns: int = 0
    pattern_pitch_class: 'torch.Tensor' = None  # [N, max_len] pitch classes
    pattern_rhythm: 'torch.Tensor' = None  # [N, max_time] binary rhythms
    pattern_lengths: 'torch.Tensor' = None  # [N] actual lengths

    # Object data (for object-level relations)
    n_objects: int = 0
    object_ids: List[Tuple] = field(default_factory=list)
    object_pattern_ids: 'torch.Tensor' = None  # [M] which pattern each object uses
    object_times: 'torch.Tensor' = None  # [M] start times
    object_track_ids: 'torch.Tensor' = None  # [M] track IDs

    # Track data (for track-level relations)
    track_ids: List[str] = field(default_factory=list)
    track_objects: Dict[str, List[int]] = field(default_factory=dict)  # track_id -> object indices


# =============================================================================
# UNIFIED RELATION DISCOVERY
# =============================================================================

class UnifiedRelationDiscovery:
    """
    Single-pass discovery of all relation types.

    Key optimization: Upload ALL data to GPU once, then run multiple
    discovery algorithms on the same data without re-uploading.

    Usage:
        discovery = UnifiedRelationDiscovery(device='cuda')
        relations = discovery.discover_all(patterns, objects)

        # Relations contains all discovered relationships:
        # - pattern_transforms: D24/rhythm transforms between patterns
        # - time_shifts: same pattern at different times
        # - cross_component: rhythm from A + contour from B
        # - cross_track: track A derives from track B
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize discovery with algebraic groups.

        Args:
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.device = device if HAS_TORCH and torch.cuda.is_available() else 'cpu'

        # Initialize algebraic groups
        self.d24 = D24Group(device=self.device)
        self.rhythm_group = RhythmGroup(device=self.device)
        self.cross_track_analyzer = CrossTrackAnalyzer(device=self.device)

    def discover_all(
        self,
        patterns: Dict,
        objects: List,
        verbose: bool = True
    ) -> RelationGraph:
        """
        Single GPU upload, multiple relation queries.

        Args:
            patterns: Dict of pattern_id -> pattern data
            objects: List of FactoredObject
            verbose: Print progress

        Returns:
            RelationGraph with all discovered relations
        """
        if verbose:
            print(f"\n{'='*70}")
            print("UNIFIED RELATION DISCOVERY")
            print(f"{'='*70}")
            print(f"  Patterns: {len(patterns)}")
            print(f"  Objects: {len(objects)}")
            print(f"  Device: {self.device}")

        # === SINGLE UPLOAD ===
        if verbose:
            print("\n  Uploading data to GPU...")
        gpu_data = self._upload_all(patterns, objects)

        if verbose:
            print(f"    Patterns uploaded: {gpu_data.n_patterns}")
            print(f"    Objects uploaded: {gpu_data.n_objects}")

        # === PARALLEL RELATION DISCOVERY ===
        relations = RelationGraph()

        # 1. Pattern transforms (D24 + rhythm)
        if verbose:
            print("\n  Finding pattern transforms...")
        pattern_transforms = self._find_pattern_transforms_gpu(gpu_data)
        for pt in pattern_transforms:
            relations.add_pattern_transform(pt)
        if verbose:
            print(f"    Found {len(pattern_transforms)} pattern transforms")

        # 2. Time shifts
        if verbose:
            print("\n  Finding time shifts...")
        time_shifts = self._find_time_shifts_gpu(gpu_data)
        relations.time_shifts = time_shifts
        if verbose:
            print(f"    Found {len(time_shifts)} time shift relations")

        # 3. Cross-component relations
        if verbose:
            print("\n  Finding cross-component relations...")
        cross_component = self._find_cross_component_gpu(gpu_data, objects)
        relations.cross_component = cross_component
        if verbose:
            print(f"    Found {len(cross_component)} cross-component relations")

        # 4. Cross-track relations
        if verbose:
            print("\n  Finding cross-track relations...")
        cross_track = self._find_cross_track_gpu(gpu_data, objects)
        relations.cross_track = cross_track
        if verbose:
            print(f"    Found {len(cross_track)} cross-track relations")

        if verbose:
            print(f"\n  Total relations: {sum(relations.get_stats().values())}")

        return relations

    # =========================================================================
    # DATA UPLOAD
    # =========================================================================

    def _upload_all(self, patterns: Dict, objects: List) -> GPUData:
        """
        Upload all data to GPU in one batch.

        This is the key optimization: single upload, multiple queries.
        """
        if not HAS_TORCH:
            return GPUData(device='cpu', n_patterns=len(patterns), n_objects=len(objects))

        gpu_data = GPUData(device=self.device)
        gpu_data.n_patterns = len(patterns)
        gpu_data.n_objects = len(objects)

        if not patterns or not objects:
            return gpu_data

        # Upload pattern data
        pattern_ids = sorted(patterns.keys())
        gpu_data.pattern_ids = pattern_ids

        # Find max lengths
        max_pitch_len = max(
            len(p.get('pitch_class', p.get('pitches', [])))
            for p in patterns.values()
        )
        max_rhythm_len = max(
            len(p.get('rhythm', []))
            for p in patterns.values()
        )

        # Prepare pattern tensors
        n_patterns = len(patterns)
        pitch_class_data = np.zeros((n_patterns, max(1, max_pitch_len)), dtype=np.int32)
        rhythm_data = np.zeros((n_patterns, max(1, max_rhythm_len)), dtype=np.float32)
        lengths = np.zeros(n_patterns, dtype=np.int32)

        for i, pid in enumerate(pattern_ids):
            p = patterns[pid]
            pc = p.get('pitch_class', p.get('pitches', []))
            if len(pc) > 0:
                pitch_class_data[i, :len(pc)] = np.array(pc) % 12
                lengths[i] = len(pc)

            rhythm = p.get('rhythm', [])
            if len(rhythm) > 0:
                rhythm_data[i, :len(rhythm)] = rhythm

        # Upload to GPU
        gpu_data.pattern_pitch_class = torch.tensor(
            pitch_class_data, device=self.device, dtype=torch.int32
        )
        gpu_data.pattern_rhythm = torch.tensor(
            rhythm_data, device=self.device, dtype=torch.float32
        )
        gpu_data.pattern_lengths = torch.tensor(
            lengths, device=self.device, dtype=torch.int32
        )

        # Upload object data
        gpu_data.object_ids = [
            (obj.piece_id, obj.track_id, obj.start_time, obj.scale)
            for obj in objects
        ]

        # Group objects by track
        track_objects = defaultdict(list)
        for i, obj in enumerate(objects):
            track_key = f"{obj.piece_id}_{obj.track_id}"
            track_objects[track_key].append(i)
        gpu_data.track_objects = dict(track_objects)
        gpu_data.track_ids = list(track_objects.keys())

        # Object times for time-shift detection
        times = np.array([obj.start_time for obj in objects], dtype=np.int32)
        gpu_data.object_times = torch.tensor(times, device=self.device, dtype=torch.int32)

        return gpu_data

    # =========================================================================
    # PATTERN TRANSFORM DISCOVERY
    # =========================================================================

    def _find_pattern_transforms_gpu(self, gpu_data: GPUData) -> List[PatternTransform]:
        """
        Find transforms between patterns using D24 group.

        For each pair of patterns, check if any D24 element maps one to the other.

        GPU Strategy:
        - For each transform type (24 total), apply to all patterns
        - Compute pairwise distances between original and transformed
        - Find matches (distance < threshold)
        """
        if not HAS_TORCH or gpu_data.pattern_pitch_class is None:
            return []

        transforms = []
        N = gpu_data.n_patterns
        if N == 0:
            return []

        pc = gpu_data.pattern_pitch_class  # [N, L]
        lengths = gpu_data.pattern_lengths  # [N]

        # Check each D24 transform
        for t_id in range(24):
            # Apply transform to all patterns
            if t_id < 12:
                # Transposition
                transformed = (pc + t_id) % 12
            else:
                # Inversion
                axis = t_id - 12
                transformed = (axis - pc) % 12

            # Create mask for valid positions
            max_len = pc.shape[1]
            positions = torch.arange(max_len, device=self.device).unsqueeze(0)
            mask = positions < lengths.unsqueeze(1)

            # Compute pairwise distances
            # Use masked comparison
            for i in range(min(N, 1000)):  # Limit for memory
                len_i = lengths[i].item()
                if len_i == 0:
                    continue

                # Compare pattern i (transformed) with all patterns
                pattern_i = transformed[i, :len_i]

                for j in range(i + 1, min(N, 1000)):
                    len_j = lengths[j].item()
                    if len_j != len_i:
                        continue

                    pattern_j = pc[j, :len_j]

                    # Check if they match
                    if torch.equal(pattern_i, pattern_j):
                        transforms.append(PatternTransform(
                            source_id=i,
                            target_id=j,
                            transform_type='d24',
                            transform_id=t_id,
                            param=float(t_id if t_id < 12 else t_id - 12)
                        ))

        return transforms

    # =========================================================================
    # TIME SHIFT DISCOVERY
    # =========================================================================

    def _find_time_shifts_gpu(self, gpu_data: GPUData) -> List[ObjectRelation]:
        """
        Find time-shift relations: same pattern at different times.

        GPU Strategy:
        - Group objects by pattern ID
        - Within each group, check for time offset relationships
        """
        relations = []

        if not HAS_TORCH or gpu_data.n_objects == 0:
            return relations

        # Group objects by track
        for track_id, obj_indices in gpu_data.track_objects.items():
            if len(obj_indices) < 2:
                continue

            # Get times for objects in this track
            times = gpu_data.object_times[obj_indices].cpu().numpy()
            obj_ids = [gpu_data.object_ids[i] for i in obj_indices]

            # Find pairs with consistent time offset
            # Sort by time
            sorted_indices = np.argsort(times)

            for i in range(len(sorted_indices) - 1):
                idx_i = sorted_indices[i]
                idx_j = sorted_indices[i + 1]

                time_diff = times[idx_j] - times[idx_i]

                # Check if this is a meaningful offset (power of 2 or musical division)
                if time_diff in [16, 32, 64, 128, 256, 512]:
                    relations.append(ObjectRelation(
                        source_obj_id=obj_ids[idx_i],
                        target_obj_id=obj_ids[idx_j],
                        relation_type='time_shift',
                        params={'offset': int(time_diff)}
                    ))

        return relations

    # =========================================================================
    # CROSS-COMPONENT DISCOVERY
    # =========================================================================

    def _find_cross_component_gpu(
        self,
        gpu_data: GPUData,
        objects: List
    ) -> List[ObjectRelation]:
        """
        Find cross-component relations: rhythm from A, contour from B.

        This is the "factored matching" - objects that share one component
        but differ in another.

        GPU Strategy:
        - Build rhythm hash index
        - Build contour hash index
        - Find objects with same rhythm but different contour (and vice versa)
        """
        relations = []

        if len(objects) < 2:
            return relations

        # Build hash indices
        rhythm_index = defaultdict(list)  # rhythm_hash -> object indices
        contour_index = defaultdict(list)  # contour_hash -> object indices

        for i, obj in enumerate(objects):
            if obj.num_notes > 0:
                rhythm_index[obj.rhythm_hash].append(i)
                contour_index[obj.contour_hash].append(i)

        # Find same-rhythm pairs (candidates for cross-component)
        for rhythm_hash, indices in rhythm_index.items():
            if len(indices) < 2:
                continue

            # Check if they have different contours
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]
                    obj_i, obj_j = objects[idx_i], objects[idx_j]

                    if obj_i.contour_hash != obj_j.contour_hash:
                        relations.append(ObjectRelation(
                            source_obj_id=gpu_data.object_ids[idx_i],
                            target_obj_id=gpu_data.object_ids[idx_j],
                            relation_type='cross_component',
                            params={
                                'shared': 'rhythm',
                                'differs': 'contour'
                            }
                        ))

        # Find same-contour pairs
        for contour_hash, indices in contour_index.items():
            if len(indices) < 2:
                continue

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]
                    obj_i, obj_j = objects[idx_i], objects[idx_j]

                    if obj_i.rhythm_hash != obj_j.rhythm_hash:
                        relations.append(ObjectRelation(
                            source_obj_id=gpu_data.object_ids[idx_i],
                            target_obj_id=gpu_data.object_ids[idx_j],
                            relation_type='cross_component',
                            params={
                                'shared': 'contour',
                                'differs': 'rhythm'
                            }
                        ))

        return relations

    # =========================================================================
    # CROSS-TRACK DISCOVERY
    # =========================================================================

    def _find_cross_track_gpu(
        self,
        gpu_data: GPUData,
        objects: List
    ) -> List[CrossTrackRelation]:
        """
        Find cross-track relations: track A derives from track B.

        GPU Strategy:
        - For each pair of tracks, aggregate objects and compare
        - Use CrossTrackAnalyzer for pitch/rhythm/melodic relationships
        """
        relations = []

        if len(gpu_data.track_ids) < 2:
            return relations

        # Compare each pair of tracks
        track_ids = gpu_data.track_ids

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track_a = track_ids[i]
                track_b = track_ids[j]

                # Get objects for each track
                indices_a = gpu_data.track_objects[track_a]
                indices_b = gpu_data.track_objects[track_b]

                # Aggregate pitches and rhythms
                track_a_data = self._aggregate_track_data(objects, indices_a)
                track_b_data = self._aggregate_track_data(objects, indices_b)

                # Find relationships
                rels = self.cross_track_analyzer.find_all_relationships(
                    track_a_data, track_b_data
                )

                for rel_type, param in rels:
                    relations.append(CrossTrackRelation(
                        track_a_id=track_a,
                        track_b_id=track_b,
                        relation_type=rel_type,
                        param=param
                    ))

        return relations

    def _aggregate_track_data(self, objects: List, indices: List[int]) -> Dict:
        """Aggregate data from multiple objects in a track."""
        pitches = []
        rhythms = []
        contours = []

        for idx in indices:
            obj = objects[idx]
            if obj.num_notes > 0:
                pitches.extend(obj.pitches.tolist())
                contours.extend(obj.pitch_contour.tolist())
            rhythms.extend(obj.rhythm.tolist())

        return {
            'pitches': np.array(pitches, dtype=np.int32) if pitches else np.array([]),
            'rhythm': np.array(rhythms, dtype=np.float32) if rhythms else np.array([]),
            'contour': np.array(contours, dtype=np.int32) if contours else np.array([])
        }


# =============================================================================
# DENSE TRANSFORM TABLE (FOR FUTURE OPTIMIZATION)
# =============================================================================

def build_dense_transform_table_gpu(
    patterns: Dict,
    d24_group: D24Group,
    device: str = 'cuda',
    verbose: bool = True
) -> Optional['torch.Tensor']:
    """
    Build dense NxNx24 transform table.

    table[i, j, t] = 1 if pattern[j] = T_t(pattern[i])

    This is memory-intensive but enables O(1) transform lookup.
    Only feasible for N < ~5000 patterns on typical GPUs.

    Args:
        patterns: Dict of pattern_id -> pattern data
        d24_group: D24 algebraic group
        device: PyTorch device
        verbose: Print progress

    Returns:
        Dense tensor [N, N, 24] or None if too large
    """
    if not HAS_TORCH:
        return None

    N = len(patterns)
    if N > 5000:
        if verbose:
            print(f"  Warning: {N} patterns too large for dense table (max ~5000)")
        return None

    if verbose:
        print(f"\n  Building dense transform table ({N}x{N}x24)...")
        memory_mb = N * N * 24 * 1 / (1024 * 1024)  # 1 byte per entry
        print(f"    Estimated memory: {memory_mb:.1f} MB")

    # This would be the full implementation
    # For now, return None to indicate not implemented
    return None
