"""
Meta-pattern discovery via anti-unification.

Discovers parameterized transform patterns like:
    sax_voicing = λ(X): Transpose(X)(melody)

by finding common structure across instance transforms:
    sax_1 = Transpose(+7)(melody)
    sax_2 = Transpose(+3)(melody)
    sax_3 = Transpose(0)(melody)

Architecture:
    Stage 1: Instance discovery (hierarchical track-level)
    Stage 2: Grouping by source
    Stage 3: Anti-unification within groups
    Stage 4: Meta-pattern abstraction

Author: Agent - Meta-Pattern Discovery
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class TransformInstance:
    """Single transform instance: target = Transform(source)"""
    piece_id: str
    source_track: str
    target_track: str
    transform_name: str
    transform_amount: float
    reconstruction_error: float


@dataclass
class MetaPattern:
    """
    Parameterized transform pattern.

    Example:
        source = 'melody'
        transform_type = 'transpose_semitone'
        parameter_values = [+7, +3, 0, -5]
        target_tracks = ['sax_0', 'sax_1', 'sax_2', 'sax_3']
        instances = [TransformInstance(...), ...]
    """
    source_track: str
    transform_type: str
    parameter_values: List[float]
    target_tracks: List[str]
    instances: List[TransformInstance]
    frequency: int  # How many pieces exhibit this pattern

    def __repr__(self):
        return (f"MetaPattern({self.transform_type}(X) applied to {self.source_track} "
                f"with X ∈ {self.parameter_values}, freq={self.frequency})")


class MetaPatternDiscovery:
    """
    Discovers meta-patterns via anti-unification.

    Pipeline:
        1. Find all track-to-track transforms (instances)
        2. Group by source track
        3. Group by transform type
        4. Anti-unify to find parameterized patterns
    """

    def __init__(self, min_instances: int = 3, max_error: float = 0.01):
        """
        Args:
            min_instances: Minimum instances required to form meta-pattern
            max_error: Maximum reconstruction error for valid transform
        """
        self.min_instances = min_instances
        self.max_error = max_error

    def discover_track_instances(
        self,
        corpus: Dict,
        transforms: List[Dict],
        verbose: bool = True
    ) -> List[TransformInstance]:
        """
        Stage 1: Discover all track-to-track transform instances.

        For each piece, for each track pair (source, target):
            Find best transform T where target ≈ T(source)

        Args:
            corpus: Hierarchical corpus from HierarchicalMIDICorpus
            transforms: List of available transforms
            verbose: Print progress

        Returns:
            instances: List of TransformInstance objects
        """
        if verbose:
            print(f"\n{'='*70}")
            print("STAGE 1: INSTANCE DISCOVERY (Track-Level)")
            print(f"{'='*70}")
            print(f"Testing {len(corpus)} pieces...")

        instances = []

        for piece_id, piece_data in corpus.items():
            tracks = piece_data['tracks']

            # Try all track pairs
            for source_name, source_tensor in tracks.items():
                for target_name, target_tensor in tracks.items():
                    if source_name == target_name:
                        continue

                    # Find best transform
                    best_transform, error = self._find_best_transform(
                        source_tensor,
                        target_tensor,
                        transforms
                    )

                    if error < self.max_error and best_transform:
                        instances.append(TransformInstance(
                            piece_id=piece_id,
                            source_track=source_name,
                            target_track=target_name,
                            transform_name=best_transform['name'],
                            transform_amount=best_transform['amount'],
                            reconstruction_error=error
                        ))

        if verbose:
            print(f"✓ Found {len(instances)} valid transform instances")
            print(f"  Average error: {np.mean([i.reconstruction_error for i in instances]):.6f}")

        return instances

    def _find_best_transform(
        self,
        source: np.ndarray,
        target: np.ndarray,
        transforms: List[Dict]
    ) -> Tuple[Optional[Dict], float]:
        """
        Find best transform T minimizing ||target - T(source)||²

        Args:
            source: (T, F) source tensor
            target: (T, F) target tensor
            transforms: Available transforms

        Returns:
            (best_transform, error)
        """
        from core.numpy_transforms import NumpyTransformLibrary

        lib = NumpyTransformLibrary()
        best_transform = None
        best_error = float('inf')

        for transform in transforms:
            try:
                # Apply transform
                source_expanded = np.expand_dims(source, 0)  # (1, T, F)
                transformed = lib.apply_transform(
                    source_expanded,
                    transform['name'],
                    transform['amount']
                )[0]  # Back to (T, F)

                # Compute MSE
                error = np.mean((target - transformed) ** 2)

                if error < best_error:
                    best_error = error
                    best_transform = transform

            except Exception:
                continue

        return best_transform, best_error

    def discover_meta_patterns(
        self,
        instances: List[TransformInstance],
        verbose: bool = True
    ) -> List[MetaPattern]:
        """
        Stage 2-4: Group instances and anti-unify to meta-patterns.

        Pipeline:
            1. Group by (piece_id, source_track)
            2. Group by transform_type
            3. Anti-unify parameter values
            4. Create MetaPattern if >= min_instances

        Args:
            instances: List of transform instances
            verbose: Print progress

        Returns:
            meta_patterns: List of discovered meta-patterns
        """
        if verbose:
            print(f"\n{'='*70}")
            print("STAGE 2-4: META-PATTERN ABSTRACTION (Anti-Unification)")
            print(f"{'='*70}")

        # Group by (piece_id, source_track)
        by_piece_source = defaultdict(list)
        for inst in instances:
            key = (inst.piece_id, inst.source_track)
            by_piece_source[key].append(inst)

        if verbose:
            print(f"Found {len(by_piece_source)} unique (piece, source_track) combinations")

        # Find meta-patterns within each group
        meta_patterns = []

        for (piece_id, source_track), group_instances in by_piece_source.items():
            if len(group_instances) < self.min_instances:
                continue

            # Group by transform type
            by_transform_type = defaultdict(list)
            for inst in group_instances:
                by_transform_type[inst.transform_name].append(inst)

            # Anti-unify each transform type
            for transform_type, type_instances in by_transform_type.items():
                if len(type_instances) >= self.min_instances:
                    # Extract parameter values
                    param_values = [inst.transform_amount for inst in type_instances]
                    target_tracks = [inst.target_track for inst in type_instances]

                    meta_pattern = MetaPattern(
                        source_track=source_track,
                        transform_type=transform_type,
                        parameter_values=param_values,
                        target_tracks=target_tracks,
                        instances=type_instances,
                        frequency=1  # Within single piece
                    )

                    meta_patterns.append(meta_pattern)

        # Find cross-piece meta-patterns
        meta_patterns = self._find_cross_piece_patterns(meta_patterns, verbose)

        if verbose:
            print(f"\n✓ Discovered {len(meta_patterns)} meta-patterns")
            print(f"\nTop 10 meta-patterns by frequency:")
            for i, pattern in enumerate(sorted(meta_patterns, key=lambda x: x.frequency, reverse=True)[:10]):
                print(f"  {i+1}. {pattern}")

        return meta_patterns

    def _find_cross_piece_patterns(
        self,
        single_piece_patterns: List[MetaPattern],
        verbose: bool
    ) -> List[MetaPattern]:
        """
        Find meta-patterns that appear across multiple pieces.

        Groups similar patterns and counts frequency.
        """
        # Group by (source_track, transform_type, parameter_signature)
        pattern_groups = defaultdict(list)

        for pattern in single_piece_patterns:
            # Create signature from sorted parameters (rounded)
            param_sig = tuple(sorted([round(p, 1) for p in pattern.parameter_values]))
            key = (pattern.source_track, pattern.transform_type, param_sig)
            pattern_groups[key].append(pattern)

        # Create cross-piece meta-patterns
        cross_piece_patterns = []

        for key, group in pattern_groups.items():
            if len(group) < 2:  # Need at least 2 pieces
                continue

            source_track, transform_type, param_sig = key

            # Merge instances from all pieces
            all_instances = []
            all_targets = []
            for p in group:
                all_instances.extend(p.instances)
                all_targets.extend(p.target_tracks)

            meta_pattern = MetaPattern(
                source_track=source_track,
                transform_type=transform_type,
                parameter_values=list(param_sig),
                target_tracks=list(set(all_targets)),  # Unique targets
                instances=all_instances,
                frequency=len(group)  # Number of pieces
            )

            cross_piece_patterns.append(meta_pattern)

        if verbose:
            print(f"  Found {len(cross_piece_patterns)} cross-piece patterns")

        return cross_piece_patterns

    def run_full_discovery(
        self,
        corpus: Dict,
        transforms: List[Dict],
        verbose: bool = True
    ) -> Dict:
        """
        Complete meta-pattern discovery pipeline.

        Args:
            corpus: Hierarchical MIDI corpus
            transforms: Available primitive transforms
            verbose: Print progress

        Returns:
            results: {
                'instances': List[TransformInstance],
                'meta_patterns': List[MetaPattern],
                'statistics': Dict
            }
        """
        # Stage 1: Instance discovery
        instances = self.discover_track_instances(corpus, transforms, verbose)

        # Stage 2-4: Meta-pattern abstraction
        meta_patterns = self.discover_meta_patterns(instances, verbose)

        # Compute statistics
        statistics = {
            'total_instances': len(instances),
            'total_meta_patterns': len(meta_patterns),
            'avg_instances_per_pattern': (
                np.mean([len(p.instances) for p in meta_patterns])
                if meta_patterns else 0
            ),
            'avg_pattern_frequency': (
                np.mean([p.frequency for p in meta_patterns])
                if meta_patterns else 0
            ),
            'transform_types': len(set(p.transform_type for p in meta_patterns))
        }

        if verbose:
            print(f"\n{'='*70}")
            print("DISCOVERY STATISTICS")
            print(f"{'='*70}")
            for key, value in statistics.items():
                print(f"  {key}: {value}")

        return {
            'instances': instances,
            'meta_patterns': meta_patterns,
            'statistics': statistics
        }
