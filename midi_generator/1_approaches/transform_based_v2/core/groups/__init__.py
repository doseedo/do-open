"""
Algebraic Transform Groups for Musical Transformations.

This module implements finite group structures with precomputed Cayley tables
for O(1) composition operations.

Groups:
- D24Group: Dihedral group of order 24 for triadic harmony (T₀-T₁₁, I₀-I₁₁)
- RhythmGroup: Multiplicative group for rhythmic transforms
- VoicingGroup: Voice-leading transforms (drop-2, drop-3, etc.)

Author: Architecture Refactor - Dosedo v2
"""

from .d24_group import D24Group
from .rhythm_group import RhythmGroup
from .voicing_group import VoicingGroup, CrossTrackRelationType

__all__ = ['D24Group', 'RhythmGroup', 'VoicingGroup', 'CrossTrackRelationType']
