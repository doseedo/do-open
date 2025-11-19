"""
Genre modules for the MIDI Generator.

Each module implements a specific musical genre with:
- Style-specific patterns and progressions
- Characteristic rhythms and instrumentation
- Research-based musical authenticity

Available genres:
- blues: Delta, Chicago, Texas blues
- country: Traditional, bluegrass, country rock
- electronic: Ambient, IDM, glitch, breakcore
- gospel: Traditional, contemporary, quartet
- reggae: Roots, dancehall, dub
- hiphop: Boom bap, trap, lo-fi, drill, conscious, G-funk (NEW!)
"""

# Import genre modules
from . import blues
from . import country
from . import electronic
from . import gospel
from . import reggae
from . import hiphop

# Import main generator classes (available classes)
try:
    from .blues import BluesGenerator, BluesStyle
except ImportError:
    pass

try:
    from .country import CountryGenerator, CountryStyle
except ImportError:
    pass

try:
    from .gospel import GospelGenerator, GospelStyle
except ImportError:
    pass

try:
    from .reggae import ReggaeGenerator, ReggaeStyle
except ImportError:
    pass

try:
    from .hiphop import HipHopGenerator, HipHopStyle
except ImportError:
    pass

__all__ = [
    # Modules
    'blues',
    'country',
    'electronic',
    'gospel',
    'reggae',
    'hiphop',
]
