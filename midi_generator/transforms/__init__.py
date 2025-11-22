"""
Transforms Module - Space-Level Musical Transforms
===================================================

Universal transform system for MIDI DNA encoding/decoding.

Architecture:
- 40 transforms across 5 musical dimensions
- Each transform has a continuous [0,1] parameter
- Compositional and interpretable
- Real-time editing capable

Dimensions:
- Pitch (8 transforms): transpose, interval scale, voice spread, register, range, contour, doubling, detune
- Rhythm (8 transforms): tempo, swing, syncopation, density, quantize, groove, rubato, polyrhythm
- Harmony (8 transforms): complexity, tension, extensions, voice leading, modality, chromaticism, harmonic rhythm, substitution
- Texture (8 transforms): polyphony, voice spacing, doubling, layering, articulation, dynamics, density, timbre
- Form (8 transforms): repetition, development, contrast, variation, symmetry, fragmentation, continuity, recapitulation

Usage:
    from midi_generator.transforms import get_transform_registry

    # Get registry
    registry = get_transform_registry()

    # Encode MIDI to 40D transform space
    dna = registry.encode(midi_file)  # → 40D vector

    # Decode 40D vector back to MIDI
    reconstructed = registry.decode(dna)

    # Get specific transform
    transpose = registry.get_transform('transpose')
    transposed_midi = transpose.apply(midi_file, amount=0.75)

    # Interpolate between two pieces
    morphed = registry.interpolate_midis(midi1, midi2, t=0.5)

Author: Agent 8 - Transform Architecture
Phase: 1 (Foundation)
"""

from .space_level_transforms import (
    SpaceLevelTransform,
    TransformMetadata,
    TransformChain,
    extract_notes_from_midi,
    notes_to_midi,
    compute_tempo_bpm,
    set_tempo_bpm,
    interpolate_transforms
)

from .transform_registry import (
    TransformRegistry,
    get_transform_registry
)

# Pitch transforms
from .pitch_transforms import (
    TransposeTransform,
    IntervalScaleTransform,
    VoiceSpreadTransform,
    RegisterShiftTransform,
    PitchRangeTransform,
    MelodicContourTransform,
    OctaveDoublingTransform,
    MicrotonalDetuneTransform
)

# Rhythm transforms
from .rhythm_transforms import (
    TempoTransform,
    SwingTransform,
    SyncopationTransform,
    NoteDensityTransform,
    QuantizeTransform,
    GrooveTransform,
    RubatoTransform,
    PolyrhythmTransform
)

# Harmony transforms
from .harmony_transforms import (
    HarmonicComplexityTransform,
    TensionTransform,
    ChordExtensionsTransform,
    VoiceLeadingTransform,
    ModalityTransform,
    ChromaticismTransform,
    HarmonicRhythmTransform,
    SubstitutionTransform
)

# Texture transforms
from .texture_transforms import (
    PolyphonyTransform,
    VoiceSpacingTransform,
    DoublingTransform,
    LayeringTransform,
    ArticulationTransform,
    DynamicRangeTransform,
    TextureDensityTransform,
    TimbreVarietyTransform
)

# Form transforms
from .form_transforms import (
    RepetitionTransform,
    DevelopmentTransform,
    ContrastTransform,
    VariationTransform,
    SymmetryTransform,
    FragmentationTransform,
    ContinuityTransform,
    RecapitulationTransform
)


__all__ = [
    # Core classes
    'SpaceLevelTransform',
    'TransformMetadata',
    'TransformChain',
    'TransformRegistry',
    'get_transform_registry',

    # Utilities
    'extract_notes_from_midi',
    'notes_to_midi',
    'compute_tempo_bpm',
    'set_tempo_bpm',
    'interpolate_transforms',

    # Pitch transforms
    'TransposeTransform',
    'IntervalScaleTransform',
    'VoiceSpreadTransform',
    'RegisterShiftTransform',
    'PitchRangeTransform',
    'MelodicContourTransform',
    'OctaveDoublingTransform',
    'MicrotonalDetuneTransform',

    # Rhythm transforms
    'TempoTransform',
    'SwingTransform',
    'SyncopationTransform',
    'NoteDensityTransform',
    'QuantizeTransform',
    'GrooveTransform',
    'RubatoTransform',
    'PolyrhythmTransform',

    # Harmony transforms
    'HarmonicComplexityTransform',
    'TensionTransform',
    'ChordExtensionsTransform',
    'VoiceLeadingTransform',
    'ModalityTransform',
    'ChromaticismTransform',
    'HarmonicRhythmTransform',
    'SubstitutionTransform',

    # Texture transforms
    'PolyphonyTransform',
    'VoiceSpacingTransform',
    'DoublingTransform',
    'LayeringTransform',
    'ArticulationTransform',
    'DynamicRangeTransform',
    'TextureDensityTransform',
    'TimbreVarietyTransform',

    # Form transforms
    'RepetitionTransform',
    'DevelopmentTransform',
    'ContrastTransform',
    'VariationTransform',
    'SymmetryTransform',
    'FragmentationTransform',
    'ContinuityTransform',
    'RecapitulationTransform',
]


# Version info
__version__ = '0.1.0'
__author__ = 'Agent 8 - Data Pipeline & Transform Architecture'
