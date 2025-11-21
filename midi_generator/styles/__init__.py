#!/usr/bin/env python3
"""
Big Band Style Profiles Module
===============================

This module provides style profiles for different big band arrangers and eras.
Each profile defines arranging characteristics, voicing preferences, harmonic
complexity, and orchestration techniques.

Available Profiles:
------------------
- Duke Ellington: Exotic harmonies, plunger mutes, rich orchestration
- Count Basie: Simple riffs, powerful rhythm section, sparse piano
- Thad Jones: Modern, angular, quartal harmony
- Maria Schneider: Orchestral colors, impressionistic
- Gordon Goodwin: High energy, contemporary swing

Authors: Agents 13, 14, 15 - Style Analyzers
Date: 2025
License: MIT
"""

# Classic Big Band Styles (Agents 13 & 14)
try:
    from .ellington_profile import ELLINGTON_STYLE, EllingtonStyleConfig
    _ellington_exports = ['ELLINGTON_STYLE', 'EllingtonStyleConfig']
except ImportError:
    _ellington_exports = []

try:
    from .basie_profile import BASIE_STYLE, BasieStyleConfig
    _basie_exports = ['BASIE_STYLE', 'BasieStyleConfig']
except ImportError:
    _basie_exports = []

# Modern Big Band Styles (Agent 15)
from .modern_profiles import (
    THAD_JONES_STYLE,
    MARIA_SCHNEIDER_STYLE,
    GORDON_GOODWIN_STYLE,
    StyleProfile,
    ModernBigBandArranger
)

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
    # Modern Profiles
    'THAD_JONES_STYLE',
    'MARIA_SCHNEIDER_STYLE',
    'GORDON_GOODWIN_STYLE',
    'StyleProfile',
    'ModernBigBandArranger',
]

# Add classic profiles if available
__all__.extend(_ellington_exports)
__all__.extend(_basie_exports)
__all__.extend(_arrangers_available)
