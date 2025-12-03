"""
Unified Relational Discovery Module.

Replaces 4 separate discovery phases with a single-pass GPU-accelerated
discovery that finds all relation types simultaneously.

Components:
- UnifiedRelationDiscovery: Main discovery class
- RelationGraph: Graph structure for storing discovered relations
- DenseTransformTable: NxNxT transform lookup table

Author: Architecture Refactor - Dosedo v2
"""

from .unified_relations import (
    UnifiedRelationDiscovery,
    RelationGraph,
    PatternTransform,
    ObjectRelation,
    CrossTrackRelation
)

from .dense_transform_table import (
    DenseTransformTable,
    build_pattern_transform_table,
    find_transforms_for_pattern_pair
)

__all__ = [
    'UnifiedRelationDiscovery',
    'RelationGraph',
    'PatternTransform',
    'ObjectRelation',
    'CrossTrackRelation',
    'DenseTransformTable',
    'build_pattern_transform_table',
    'find_transforms_for_pattern_pair'
]
