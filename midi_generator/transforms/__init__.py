"""
Transforms Module - Space-Level Musical Transforms
===================================================

Universal transform system for MIDI DNA encoding/decoding.

Architecture:
- 60 transforms across 6 musical dimensions (expandable to 200-400)
- Each transform has a continuous [0,1] parameter
- Compositional and interpretable
- Real-time editing capable

Dimensions:
- Pitch (8 transforms): transpose, interval scale, voice spread, register, range, contour, doubling, detune
- Rhythm (11 transforms): tempo, swing, syncopation, density, quantize, groove, rubato, polyrhythm + 3 advanced
- Harmony (11 transforms): complexity, tension, extensions, voice leading, modality, chromaticism + 3 advanced
- Texture (10 transforms): polyphony, voice spacing, doubling, layering, articulation, dynamics + 2 advanced
- Form (8 transforms): repetition, development, contrast, variation, symmetry, fragmentation, continuity, recapitulation
- Expression (12 transforms): dynamics contour, phrasing, accents, legato, velocity curve, attack, pedaling + 5 more

Total: 60 theory-based transforms
Target: Expand to 200-400 with sparse dictionary learning + gap discovery

Usage:
    from midi_generator.transforms import get_transform_registry

    # Get registry
    registry = get_transform_registry()

    # Encode MIDI to 60D transform space
    dna = registry.encode(midi_file)  # → 60D vector

    # Decode 60D vector back to MIDI
    reconstructed = registry.decode(dna)

    # Get specific transform
    transpose = registry.get_transform('transpose')
    transposed_midi = transpose.apply(midi_file, amount=0.75)

    # Interpolate between two pieces
    morphed = registry.interpolate_midis(midi1, midi2, t=0.5)

Author: Agent 8 - Transform Architecture
Phase: 1+ (Foundation Extended)
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

# Expression transforms
from .expression_transforms import (
    DynamicsContourTransform,
    PhrasingTransform,
    AccentPatternTransform,
    LegatoStaccatoTransform,
    VelocityCurveTransform,
    AttackDecayTransform,
    PedalingTransform,
    VibratoTransform,
    TremoloTransform,
    BendTransform,
    GlissandoTransform,
    OrnamentationTransform
)

# Advanced transforms
from .advanced_transforms import (
    MetricModulationTransform,
    ModalMixtureTransform,
    CounterpointDensityTransform,
    TexturalEvolutionTransform,
    HarmonicModulationTransform,
    PolymeterTransform,
    MicrorhythmTransform,
    SpectralDensityTransform
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

    # Expression transforms (12)
    'DynamicsContourTransform',
    'PhrasingTransform',
    'AccentPatternTransform',
    'LegatoStaccatoTransform',
    'VelocityCurveTransform',
    'AttackDecayTransform',
    'PedalingTransform',
    'VibratoTransform',
    'TremoloTransform',
    'BendTransform',
    'GlissandoTransform',
    'OrnamentationTransform',

    # Advanced transforms (8)
    'MetricModulationTransform',
    'ModalMixtureTransform',
    'CounterpointDensityTransform',
    'TexturalEvolutionTransform',
    'HarmonicModulationTransform',
    'PolymeterTransform',
    'MicrorhythmTransform',
    'SpectralDensityTransform',
]


# Version info
__version__ = '0.2.0'  # Updated to 60 transforms
__author__ = 'Agent 8 - Data Pipeline & Transform Architecture'
