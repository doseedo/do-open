#!/usr/bin/env python3
"""
Big Band Style Profiles
=======================

Style profiles for different big band composers and arrangers.

Each style profile defines characteristics such as:
- Orchestration preferences
- Harmonic complexity
- Voicing choices
- Articulation usage
- Dynamic range
- Form preferences
- Texture density

Implemented Styles:
- Duke Ellington: Exotic harmonies, plunger mutes, rich orchestration
- Count Basie: Simple riffs, powerful rhythm section, sparse piano
- Thad Jones: Modern voicings, angular melodies, quartal harmony
- Maria Schneider: Orchestral colors, impressionistic approach

Author: Agent 13 - Duke Ellington Style Analyzer
"""

from .ellington_profile import ELLINGTON_STYLE, EllingtonStyleConfig

# Import arranger only if dependencies are available
try:
    from .ellington_arranger import EllingtonArranger
    __all__ = [
        'ELLINGTON_STYLE',
        'EllingtonStyleConfig',
        'EllingtonArranger',
    ]
except ImportError as e:
    # Mido not available, skip arranger import
    __all__ = [
        'ELLINGTON_STYLE',
        'EllingtonStyleConfig',
    ]
