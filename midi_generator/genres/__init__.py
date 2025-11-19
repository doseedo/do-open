"""
Genre Modules for MIDI Generator

This package contains genre-specific music generators.

Available genres:
- Blues (blues.py)
- Classic Rock (classic_rock.py)
- Country (country.py)
- Electronic (electronic.py)
- Gospel (gospel.py)
- Reggae (reggae.py)
- World Music (world/)
"""

# Import genre modules for easy access
from . import blues
from . import classic_rock
from . import country
from . import electronic
from . import gospel
from . import reggae

__all__ = [
    'blues',
    'classic_rock',
    'country',
    'electronic',
    'gospel',
    'reggae',
]
