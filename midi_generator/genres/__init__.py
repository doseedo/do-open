"""
Genre-specific music generators

Available genres:
- Blues (Delta, Chicago, Texas)
- Country (Traditional, Outlaw, Bluegrass)
- Electronic (House, Techno, Trance, etc.)
- Gospel (Traditional, Contemporary)
- Reggae (Roots, Dancehall, Dub)
- Metal (Thrash, Death, Black, Progressive, Djent, etc.)
- World Music (see world/ subdirectory)
"""

# Import only the metal module for now to avoid breaking existing imports
from .metal import (
    MetalGenerator,
    MetalSubgenre,
    DropTuning,
    BlastBeatType,
    RiffTechnique,
    MetalScales,
    MetalRiff,
    DrumPattern
)

__all__ = [
    # Metal
    'MetalGenerator',
    'MetalSubgenre',
    'DropTuning',
    'BlastBeatType',
    'RiffTechnique',
    'MetalScales',
    'MetalRiff',
    'DrumPattern',
]
