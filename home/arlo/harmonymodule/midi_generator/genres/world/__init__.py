"""
World Music Generators

This package provides authentic world music generation for various traditions:
- Indian Classical (Hindustani and Carnatic)
- Arabic/Middle Eastern (Maqam system)
- African (West African, Afro-Cuban rhythms)
- Expanded World Music (Flamenco, Klezmer, Gamelan, Celtic, Bossa Nova, Tango)
"""

from .indian import IndianClassicalGenerator
from .arabic import ArabicMusicGenerator
from .african import AfricanMusicGenerator
from .expanded import ExpandedWorldMusic

__all__ = [
    'IndianClassicalGenerator',
    'ArabicMusicGenerator',
    'AfricanMusicGenerator',
    'ExpandedWorldMusic',
]
