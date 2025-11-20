#!/usr/bin/env python3
"""
Big Band Style Profiles
========================

This module contains style profiles for different big band arrangers and eras.
Each profile defines the characteristic arranging techniques and aesthetic
choices of legendary big band composers and arrangers.

Available Styles:
-----------------
- Duke Ellington: Exotic harmonies, plunger mutes, rich orchestration
- Count Basie: Simple riffs, powerful rhythm section, sparse piano
- Thad Jones: Modern voicings, angular melodies, quartal harmony (future)
- Maria Schneider: Impressionistic, orchestral colors (future)

Author: Agents 13 & 14 - Style Analyzers
Date: 2025
License: MIT
"""

# Duke Ellington Style (Agent 13)
from .ellington_profile import ELLINGTON_STYLE, EllingtonStyleConfig

# Count Basie Style (Agent 14)
from .basie_profile import BASIE_STYLE, BasieStyleConfig

# Import arrangers only if dependencies are available
_arrangers_available = []

try:
    from .ellington_arranger import EllingtonArranger
    _arrangers_available.append('EllingtonArranger')
except ImportError:
    pass

try:
    from .basie_arranger import BasieArranger, BasieRiffGenerator
    _arrangers_available.extend(['BasieArranger', 'BasieRiffGenerator'])
except ImportError:
    pass

# Export list
__all__ = [
    # Profiles
    'ELLINGTON_STYLE',
    'EllingtonStyleConfig',
    'BASIE_STYLE',
    'BasieStyleConfig',
] + _arrangers_available
