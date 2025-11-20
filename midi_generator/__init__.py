"""
Unified Music Generation Library
=================================

Combines comprehensive music theory, algorithmic composition, and genre-specific
generation into one unified system.

Features:
- 35+ music genres with authentic implementations
- Advanced harmony (modal, neo-Riemannian, microtonal)
- Algorithmic composition (L-systems, cellular automata, constraints)
- Style fusion and context-aware generation
- Professional orchestration and arrangement
- MIDI analysis and pattern extraction
- Machine learning from corpus

Quick Start:
    from midi_generator.api import UnifiedMusicGenerator
    from midi_generator.generators import AdvancedHarmonyGenerator
    from midi_generator.core import Mode, Triad

Version: 2.0.0 (Unified)
"""

__version__ = "2.0.0"

# Core music theory
from .core.modal_harmony import Mode, ModalProgressionGenerator, ModalInterchange
from .core.neo_riemannian import Triad, TriadQuality, NeoRiemannianTransformations
from .core.microtonality import ArabicMaqam, IndianRaga, TurkishMakam
from .core.instrument_library import Instrument, InstrumentFamily

# Generators
from .generators.advanced_harmony_generator import AdvancedHarmonyGenerator
from .generators.form_generator import FormGenerator, MusicalForm
from .generators.orchestrator import Orchestrator

# High-level API
try:
    from .api.unified_api import UnifiedMusicGenerator
    __all_exports = ['UnifiedMusicGenerator']
except ImportError:
    __all_exports = []

__all__ = [
    # Version
    '__version__',

    # Core
    'Mode', 'ModalProgressionGenerator', 'ModalInterchange',
    'Triad', 'TriadQuality', 'NeoRiemannianTransformations',
    'ArabicMaqam', 'IndianRaga', 'TurkishMakam',
    'Instrument', 'InstrumentFamily',

    # Generators
    'AdvancedHarmonyGenerator', 'FormGenerator', 'MusicalForm',
    'Orchestrator',
] + __all_exports
