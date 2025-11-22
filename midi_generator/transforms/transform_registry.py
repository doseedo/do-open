"""
Transform Registry
==================

Central registry and management system for all space-level transforms.

Provides:
- Registration of all 40 transforms across 5 dimensions
- Lookup by name, dimension, or level
- Transform composition and chaining
- DNA encoding/decoding interface

Author: Agent 8 - Transform Architecture
"""

from typing import List, Dict, Optional, Any
import numpy as np
import mido

from .space_level_transforms import SpaceLevelTransform, TransformChain

# Import all transforms
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


# ============================================================================
# Transform Registry
# ============================================================================

class TransformRegistry:
    """
    Central registry of all space-level transforms.

    Manages 40 transforms across 5 dimensions:
    - Pitch (8 transforms)
    - Rhythm (8 transforms)
    - Harmony (8 transforms)
    - Texture (8 transforms)
    - Form (8 transforms)

    Example:
        registry = TransformRegistry()

        # Get transform by name
        transpose = registry.get_transform('transpose')

        # Get all pitch transforms
        pitch_transforms = registry.get_by_dimension('pitch')

        # Get transform vector for MIDI
        dna = registry.encode(midi_file)  # 40D vector

        # Apply transform vector to reconstruct
        reconstructed = registry.decode(dna)
    """

    def __init__(self):
        """Initialize registry and register all transforms"""
        self.transforms: Dict[str, SpaceLevelTransform] = {}
        self.dimensions: Dict[str, List[str]] = {
            'pitch': [],
            'rhythm': [],
            'harmony': [],
            'texture': [],
            'form': []
        }
        self.transform_order: List[str] = []  # Ordered list of transform names

        self._register_all_transforms()

    def _register_all_transforms(self):
        """Register all 40 transforms"""

        # Pitch transforms (8)
        self._register(TransposeTransform())
        self._register(IntervalScaleTransform())
        self._register(VoiceSpreadTransform())
        self._register(RegisterShiftTransform())
        self._register(PitchRangeTransform())
        self._register(MelodicContourTransform())
        self._register(OctaveDoublingTransform())
        self._register(MicrotonalDetuneTransform())

        # Rhythm transforms (8)
        self._register(TempoTransform())
        self._register(SwingTransform())
        self._register(SyncopationTransform())
        self._register(NoteDensityTransform())
        self._register(QuantizeTransform())
        self._register(GrooveTransform())
        self._register(RubatoTransform())
        self._register(PolyrhythmTransform())

        # Harmony transforms (8)
        self._register(HarmonicComplexityTransform())
        self._register(TensionTransform())
        self._register(ChordExtensionsTransform())
        self._register(VoiceLeadingTransform())
        self._register(ModalityTransform())
        self._register(ChromaticismTransform())
        self._register(HarmonicRhythmTransform())
        self._register(SubstitutionTransform())

        # Texture transforms (8)
        self._register(PolyphonyTransform())
        self._register(VoiceSpacingTransform())
        self._register(DoublingTransform())
        self._register(LayeringTransform())
        self._register(ArticulationTransform())
        self._register(DynamicRangeTransform())
        self._register(TextureDensityTransform())
        self._register(TimbreVarietyTransform())

        # Form transforms (8)
        self._register(RepetitionTransform())
        self._register(DevelopmentTransform())
        self._register(ContrastTransform())
        self._register(VariationTransform())
        self._register(SymmetryTransform())
        self._register(FragmentationTransform())
        self._register(ContinuityTransform())
        self._register(RecapitulationTransform())

    def _register(self, transform: SpaceLevelTransform):
        """Register a single transform"""
        name = transform.name
        dimension = transform.dimension

        if name in self.transforms:
            raise ValueError(f"Transform '{name}' already registered")

        self.transforms[name] = transform
        self.dimensions[dimension].append(name)
        self.transform_order.append(name)

    def get_transform(self, name: str) -> SpaceLevelTransform:
        """Get transform by name"""
        if name not in self.transforms:
            raise KeyError(f"Transform '{name}' not found")
        return self.transforms[name]

    def get_by_dimension(self, dimension: str) -> List[SpaceLevelTransform]:
        """Get all transforms for a dimension"""
        if dimension not in self.dimensions:
            raise KeyError(f"Dimension '{dimension}' not found")
        return [self.transforms[name] for name in self.dimensions[dimension]]

    def get_by_level(self, level: str) -> List[SpaceLevelTransform]:
        """Get all transforms for an abstraction level"""
        return [t for t in self.transforms.values() if t.level == level]

    def list_transforms(self) -> List[str]:
        """Get list of all transform names in order"""
        return self.transform_order.copy()

    def count_transforms(self) -> int:
        """Get total number of transforms"""
        return len(self.transforms)

    # ========================================================================
    # Encoding/Decoding
    # ========================================================================

    def encode(self, midi: mido.MidiFile) -> np.ndarray:
        """
        Extract transform coefficients from MIDI.

        This is the "analysis" direction: given MIDI, what are the
        values of all transform parameters?

        Args:
            midi: MIDI file to analyze

        Returns:
            40D vector of transform coefficients (each in [0,1])
        """
        coefficients = np.zeros(len(self.transform_order))

        for i, name in enumerate(self.transform_order):
            transform = self.transforms[name]
            try:
                value = transform.get_current_value(midi)
                coefficients[i] = value
            except Exception as e:
                # If extraction fails, use default
                coefficients[i] = transform.metadata.default_value

        return coefficients

    def decode(
        self,
        coefficients: np.ndarray,
        base_template: Optional[mido.MidiFile] = None
    ) -> mido.MidiFile:
        """
        Apply transform coefficients to reconstruct MIDI.

        This is the "synthesis" direction: given transform parameters,
        generate MIDI.

        Args:
            coefficients: 40D vector of transform amounts
            base_template: Starting MIDI template (if None, creates neutral template)

        Returns:
            Reconstructed MIDI file
        """
        if len(coefficients) != len(self.transform_order):
            raise ValueError(
                f"Expected {len(self.transform_order)} coefficients, "
                f"got {len(coefficients)}"
            )

        # Start with base template
        if base_template is None:
            midi = self._create_neutral_template()
        else:
            midi = base_template

        # Apply transforms in order
        for i, name in enumerate(self.transform_order):
            transform = self.transforms[name]
            amount = coefficients[i]

            try:
                midi = transform.apply(midi, amount)
            except Exception as e:
                # If transform fails, skip it
                print(f"Warning: Transform '{name}' failed: {e}")
                continue

        return midi

    def _create_neutral_template(self) -> mido.MidiFile:
        """
        Create neutral starting MIDI template.

        This is a simple C major scale that serves as a blank canvas
        for transforms to modify.

        Returns:
            Neutral MIDI file
        """
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Add tempo (120 BPM)
        track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))

        # C major scale: C4-C5
        scale = [60, 62, 64, 65, 67, 69, 71, 72]
        velocity = 64
        note_duration = 480  # Quarter note

        current_time = 0
        for pitch in scale:
            # Note on
            track.append(mido.Message(
                'note_on',
                note=pitch,
                velocity=velocity,
                time=current_time
            ))

            # Note off
            track.append(mido.Message(
                'note_off',
                note=pitch,
                velocity=0,
                time=note_duration
            ))

            current_time = 0  # Next note follows immediately

        # End of track
        track.append(mido.MetaMessage('end_of_track', time=0))

        return midi

    # ========================================================================
    # Utilities
    # ========================================================================

    def get_transform_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a transform"""
        transform = self.get_transform(name)
        metadata = transform.metadata

        return {
            'name': metadata.name,
            'dimension': metadata.dimension,
            'level': metadata.level,
            'description': metadata.description,
            'parameter_range': metadata.parameter_range,
            'default_value': metadata.default_value,
            'is_invertible': metadata.is_invertible
        }

    def print_summary(self):
        """Print registry summary"""
        print(f"\n{'='*70}")
        print(f"Transform Registry Summary")
        print(f"{'='*70}")
        print(f"Total transforms: {self.count_transforms()}")
        print()

        for dimension, names in self.dimensions.items():
            print(f"{dimension.capitalize()} transforms ({len(names)}):")
            for name in names:
                transform = self.transforms[name]
                print(f"  - {name}: {transform.description}")
            print()

        print(f"{'='*70}\n")

    def create_transform_chain(
        self,
        transform_names: List[str]
    ) -> TransformChain:
        """
        Create a chain of transforms.

        Args:
            transform_names: List of transform names to chain

        Returns:
            TransformChain object
        """
        transforms = [self.get_transform(name) for name in transform_names]
        return TransformChain(transforms)

    def interpolate_midis(
        self,
        midi1: mido.MidiFile,
        midi2: mido.MidiFile,
        t: float
    ) -> mido.MidiFile:
        """
        Interpolate between two MIDI files in transform space.

        Args:
            midi1: First MIDI file
            midi2: Second MIDI file
            t: Interpolation parameter (0=midi1, 1=midi2)

        Returns:
            Interpolated MIDI file
        """
        # Encode both
        coeffs1 = self.encode(midi1)
        coeffs2 = self.encode(midi2)

        # Interpolate
        coeffs_interp = (1 - t) * coeffs1 + t * coeffs2

        # Decode
        return self.decode(coeffs_interp, base_template=midi1)


# ============================================================================
# Global Registry Instance
# ============================================================================

# Create global singleton instance
_global_registry = None


def get_transform_registry() -> TransformRegistry:
    """
    Get global transform registry instance (singleton).

    Returns:
        Global TransformRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = TransformRegistry()
    return _global_registry
