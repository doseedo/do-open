"""
Synthesis Module
================

This module contains inverse synthesis and analysis tools for Musical Program Synthesis.

Modules:
- deep_feature_extractor: Extract 1000+ musical features from MIDI files (Agent 8)
- inverse_analyzer: Inverse MIDI analysis (coming soon)
- gap_detector: Intelligent gap detection (coming soon)

Author: Musical Program Synthesis Team
License: MIT
"""

from .deep_feature_extractor import DeepFeatureExtractor, extract_features

__all__ = [
    'DeepFeatureExtractor',
    'extract_features',
]
