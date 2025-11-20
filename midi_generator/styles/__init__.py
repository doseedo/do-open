"""
Big Band Style Profiles Module
===============================

This module provides style profiles for different big band arrangers and eras.
Each profile defines arranging characteristics, voicing preferences, harmonic
complexity, and orchestration techniques.

Available Profiles:
------------------
- Thad Jones: Modern, angular, quartal harmony
- Maria Schneider: Orchestral colors, impressionistic
- Gordon Goodwin: High energy, contemporary swing

Author: Agent 15 - Modern Big Band Style Analyzer
"""

from .modern_profiles import (
    THAD_JONES_STYLE,
    MARIA_SCHNEIDER_STYLE,
    GORDON_GOODWIN_STYLE,
    StyleProfile,
    ModernBigBandArranger
)

__all__ = [
    'THAD_JONES_STYLE',
    'MARIA_SCHNEIDER_STYLE',
    'GORDON_GOODWIN_STYLE',
    'StyleProfile',
    'ModernBigBandArranger'
]
