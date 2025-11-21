#!/usr/bin/env python3
"""
Musical Locality Functions - Demo & Integration Examples
=========================================================

This script demonstrates the usage of Agent 1's musical locality functions
and shows integration patterns for Agent 2 (Semantic Feature Representations).

Author: Agent 1 - Musical Locality Functions
Date: 2025-11-21
"""

from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.learning.musical_locality import (
    MusicalLocalityFunctions,
    MusicalTransform,
    LocalityType,
    create_random_transform
)
from mido import MidiFile
import numpy as np


# ==============================================================================
# DEMO 1: Basic Transformation
# ==============================================================================

def demo_basic_transformation():
    """
    Demonstrate basic transformation usage.
    """
    print("=" * 70)
    print("DEMO 1: Basic Transformation")
    print("=" * 70)

    # Initialize transformer
    transformer = MusicalLocalityFunctions()

    # Load a MIDI file
    midi_path = Path("path/to/your/song.mid")
    if not midi_path.exists():
        print(f"⚠️  MIDI file not found: {midi_path}")
        print("   Using placeholder for demonstration")
        return

    midi = MidiFile(str(midi_path))

    # Create a transpose transformation
    transform = MusicalTransform(
        transform_type=LocalityType.TRANSPOSE,
        parameters={"semitones": 5}
    )

    print(f"Original file: {midi_path.name}")
    print(f"Transformation: {transform.transform_type.value}")
    print(f"Parameters: {transform.parameters}")

    # Apply transformation
    transformed = transformer.apply_transform(midi, transform)

    # Save result
    output_path = Path("output/transposed_song.mid")
    output_path.parent.mkdir(exist_ok=True)
    transformed.save(str(output_path))

    print(f"✅ Transformed file saved: {output_path}")
    print()


# ==============================================================================
# DEMO 2: Invertibility Testing
# ==============================================================================

def demo_invertibility():
    """
    Demonstrate invertibility validation.
    """
    print("=" * 70)
    print("DEMO 2: Invertibility Testing")
    print("=" * 70)

    transformer = MusicalLocalityFunctions()

    # Test different transformations for invertibility
    test_transforms = [
        MusicalTransform(
            transform_type=LocalityType.TRANSPOSE,
            parameters={"semitones": 7}
        ),
        MusicalTransform(
            transform_type=LocalityType.INVERT,
            parameters={"pivot_pitch": 60}
        ),
        MusicalTransform(
            transform_type=LocalityType.AUGMENT,
            parameters={"factor": 1.5}
        ),
        MusicalTransform(
            transform_type=LocalityType.RETROGRADE,
            parameters={}
        ),
    ]

    midi_path = Path("path/to/your/song.mid")
    if not midi_path.exists():
        print("⚠️  MIDI file not found for testing")
        return

    midi = MidiFile(str(midi_path))

    print(f"Testing invertibility on: {midi_path.name}")
    print()

    for transform in test_transforms:
        is_invertible, metrics = transformer.validate_invertibility(
            midi, transform
        )

        print(f"Transform: {transform.transform_type.value}")
        print(f"  Invertible: {is_invertible}")
        print(f"  Note mismatches: {metrics['note_mismatch_count']}")
        print(f"  Max time error: {metrics['max_time_error']}")
        print(f"  Max pitch error: {metrics['max_pitch_error']}")
        print()


# ==============================================================================
# DEMO 3: All 12 Transformations
# ==============================================================================

def demo_all_transformations():
    """
    Apply all 12 transformations to a MIDI file.
    """
    print("=" * 70)
    print("DEMO 3: All 12 Transformations")
    print("=" * 70)

    transformer = MusicalLocalityFunctions()
    rng = np.random.RandomState(42)

    midi_path = Path("path/to/your/song.mid")
    if not midi_path.exists():
        print("⚠️  MIDI file not found")
        return

    midi = MidiFile(str(midi_path))
    output_dir = Path("output/transformations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {midi_path.name}")
    print(f"Output directory: {output_dir}")
    print()

    for transform_type in LocalityType:
        # Create random transform
        transform = create_random_transform(transform_type, rng)

        # Handle voice permutation (needs correct track count)
        if transform_type == LocalityType.VOICE_PERMUTATION:
            n_tracks = len(midi.tracks)
            transform.parameters = {"permutation": list(range(n_tracks))}

        # Apply transform
        try:
            result = transformer.apply_transform(midi, transform)

            # Save result
            output_name = f"{midi_path.stem}_{transform_type.value}.mid"
            output_path = output_dir / output_name
            result.save(str(output_path))

            print(f"✅ {transform_type.value:20s} → {output_name}")

        except Exception as e:
            print(f"❌ {transform_type.value:20s} → Error: {e}")

    print()


# ==============================================================================
# DEMO 4: Composition of Transformations
# ==============================================================================

def demo_composition():
    """
    Demonstrate composition of multiple transformations.
    """
    print("=" * 70)
    print("DEMO 4: Transformation Composition")
    print("=" * 70)

    transformer = MusicalLocalityFunctions()

    midi_path = Path("path/to/your/song.mid")
    if not midi_path.exists():
        print("⚠️  MIDI file not found")
        return

    midi = MidiFile(str(midi_path))

    # Define a sequence of transformations
    transform_sequence = [
        MusicalTransform(
            transform_type=LocalityType.TRANSPOSE,
            parameters={"semitones": 3}
        ),
        MusicalTransform(
            transform_type=LocalityType.AUGMENT,
            parameters={"factor": 1.2}
        ),
        MusicalTransform(
            transform_type=LocalityType.VELOCITY_SCALE,
            parameters={"factor": 0.8}
        ),
    ]

    print(f"Input: {midi_path.name}")
    print(f"Applying {len(transform_sequence)} transformations:")
    print()

    result = midi
    for i, transform in enumerate(transform_sequence, 1):
        print(f"  {i}. {transform.transform_type.value}")
        print(f"     Parameters: {transform.parameters}")
        result = transformer.apply_transform(result, transform)

    # Save final result
    output_path = Path("output/composed_transformations.mid")
    output_path.parent.mkdir(exist_ok=True)
    result.save(str(output_path))

    print()
    print(f"✅ Result saved: {output_path}")
    print()


# ==============================================================================
# DEMO 5: Data Augmentation for Training
# ==============================================================================

def demo_data_augmentation():
    """
    Demonstrate data augmentation for training.
    This is the pattern Agent 2 will use.
    """
    print("=" * 70)
    print("DEMO 5: Data Augmentation for Training")
    print("=" * 70)

    transformer = MusicalLocalityFunctions()
    rng = np.random.RandomState(42)

    midi_path = Path("path/to/your/song.mid")
    if not midi_path.exists():
        print("⚠️  MIDI file not found")
        return

    midi = MidiFile(str(midi_path))

    # Generate multiple augmented versions
    n_augmentations = 10
    output_dir = Path("output/augmented")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_augmentations} augmented versions")
    print(f"Original: {midi_path.name}")
    print()

    for i in range(n_augmentations):
        # Randomly select transformation type
        transform_type = rng.choice(list(LocalityType))
        transform = create_random_transform(transform_type, rng)

        # Handle voice permutation
        if transform_type == LocalityType.VOICE_PERMUTATION:
            n_tracks = len(midi.tracks)
            perm = list(range(n_tracks))
            rng.shuffle(perm)
            transform.parameters = {"permutation": perm}

        # Apply transformation
        augmented = transformer.apply_transform(midi, transform)

        # Save
        output_path = output_dir / f"aug_{i:02d}_{transform_type.value}.mid"
        augmented.save(str(output_path))

        print(f"  {i+1:2d}. {transform_type.value:20s} → {output_path.name}")

    print()
    print(f"✅ Generated {n_augmentations} augmented files in {output_dir}")
    print()


# ==============================================================================
# DEMO 6: Integration Pattern for Agent 2
# ==============================================================================

def demo_agent2_integration_pattern():
    """
    Demonstrate the integration pattern that Agent 2 will use.
    This is a template/pseudocode example.
    """
    print("=" * 70)
    print("DEMO 6: Agent 2 Integration Pattern (Pseudocode)")
    print("=" * 70)

    print("""
# This is how Agent 2 will use Agent 1's transformations:

from midi_generator.learning.musical_locality import (
    MusicalLocalityFunctions,
    MusicalTransform,
    LocalityType,
    create_random_transform
)

class SemanticFeature:
    '''
    A semantic feature that should be invariant under certain transformations.
    '''

    def __init__(
        self,
        name: str,
        invariant_transforms: List[LocalityType],
        extractor_function: Callable
    ):
        self.name = name
        self.invariant_transforms = invariant_transforms
        self.extractor = extractor_function
        self.transformer = MusicalLocalityFunctions()

    def generate_variants(
        self,
        midi: MidiFile,
        n_variants: int = 10
    ) -> List[Tuple[MidiFile, MusicalTransform]]:
        '''
        Generate transformed variants of a MIDI file.
        '''
        variants = []
        rng = np.random.RandomState()

        for _ in range(n_variants):
            # Create random transform
            transform_type = rng.choice(self.invariant_transforms)
            transform = create_random_transform(transform_type, rng)

            # Apply transform
            variant = self.transformer.apply_transform(midi, transform)
            variants.append((variant, transform))

        return variants

    def test_invariance(self, midi: MidiFile, threshold: float = 0.1) -> bool:
        '''
        Test if the feature is invariant under expected transformations.
        '''
        # Get original activation
        activation_original = self.extractor(midi)

        # Test each invariant transformation
        for transform_type in self.invariant_transforms:
            transform = create_random_transform(transform_type)
            transformed = self.transformer.apply_transform(midi, transform)
            activation_transformed = self.extractor(transformed)

            # Check if activations are similar
            difference = abs(activation_original - activation_transformed)
            if difference > threshold:
                return False

        return True

    def get_activation_strength(self, midi: MidiFile) -> float:
        '''
        Get the activation strength of this feature on a MIDI file.
        '''
        return self.extractor(midi)


class SemanticFeatureBank:
    '''
    Collection of semantic features for parameter extraction.
    '''

    def __init__(self):
        self.features = []
        self.transformer = MusicalLocalityFunctions()

    def add_feature(self, feature: SemanticFeature):
        '''Add a feature to the bank.'''
        self.features.append(feature)

    def get_activations(
        self,
        midi: MidiFile,
        use_augmentation: bool = True
    ) -> np.ndarray:
        '''
        Get activation vector for all features.

        If use_augmentation=True, test with transformed variants
        to improve robustness.
        '''
        activations = []

        for feature in self.features:
            if use_augmentation:
                # Average activation across variants
                variants = feature.generate_variants(midi, n_variants=5)
                variant_activations = [
                    feature.get_activation_strength(v[0])
                    for v in variants
                ]
                activation = np.mean(variant_activations)
            else:
                activation = feature.get_activation_strength(midi)

            activations.append(activation)

        return np.array(activations)


# Example: Define a pitch-class feature (octave-invariant)
def extract_c_major_presence(midi: MidiFile) -> float:
    '''Extract presence of C major pitch classes.'''
    # Count C, E, G notes (pitch classes 0, 4, 7)
    # ... implementation ...
    return 0.5  # placeholder

c_major_feature = SemanticFeature(
    name="c_major_presence",
    invariant_transforms=[
        LocalityType.OCTAVE_SHIFT,    # Should be octave-invariant
        LocalityType.TIME_SHIFT,       # Should be timing-invariant
        LocalityType.VELOCITY_SCALE,   # Should be dynamics-invariant
    ],
    extractor_function=extract_c_major_presence
)

# Test the feature
midi = MidiFile("song.mid")
is_invariant = c_major_feature.test_invariance(midi)
print(f"C major feature is invariant: {is_invariant}")

# Generate training data with augmentation
variants = c_major_feature.generate_variants(midi, n_variants=20)
print(f"Generated {len(variants)} augmented variants for training")
    """)

    print("=" * 70)
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """
    Run all demonstrations.
    """
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  MUSICAL LOCALITY FUNCTIONS - DEMONSTRATION & EXAMPLES  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Agent 1 - Musical Parameter Discovery".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Run demos
    demos = [
        ("Basic Transformation", demo_basic_transformation),
        ("Invertibility Testing", demo_invertibility),
        ("All 12 Transformations", demo_all_transformations),
        ("Transformation Composition", demo_composition),
        ("Data Augmentation", demo_data_augmentation),
        ("Agent 2 Integration Pattern", demo_agent2_integration_pattern),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"⚠️  Demo failed: {e}")
            import traceback
            traceback.print_exc()

        if i < len(demos):
            input("Press Enter to continue to next demo...")
            print("\n" * 2)

    print()
    print("=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
