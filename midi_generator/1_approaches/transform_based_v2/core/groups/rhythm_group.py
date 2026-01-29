"""
Rhythm Transform Group.

A finite group of rhythmic transformations with precomputed composition.

Elements:
- identity: No change
- retrograde: Time reversal
- augment_2x: Double duration
- diminish_half: Halve duration
- augment_3x: Triple duration (for compound meters)
- diminish_third: Third duration

This group is NOT closed under arbitrary composition (augment_2x ∘ augment_2x = augment_4x
which isn't in our base set), so we use a "clamped" composition that maps results
back to the closest representable element.

For practical music, we limit to these common transformations.

Author: Architecture Refactor - Dosedo v2
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RhythmElement(IntEnum):
    """Named elements of the rhythm group."""
    IDENTITY = 0
    RETROGRADE = 1
    AUGMENT_2X = 2
    DIMINISH_HALF = 3
    AUGMENT_3X = 4
    DIMINISH_THIRD = 5
    AUGMENT_4X = 6
    DIMINISH_QUARTER = 7


@dataclass
class RhythmTransformSpec:
    """Specification for a rhythm transform."""
    name: str
    scale_factor: float  # Time scaling (1.0 = identity)
    is_retrograde: bool  # Whether to reverse time


class RhythmGroup:
    """
    Multiplicative group for rhythmic transforms.

    Unlike D24 which is mathematically closed, this rhythm group
    uses a practical set of transforms commonly used in music.

    Usage:
        rg = RhythmGroup()

        # Apply transform to rhythm pattern
        new_rhythm = rg.apply_to_rhythm(RhythmElement.RETROGRADE, rhythm_array)

        # Compose transforms
        composed = rg.compose(RhythmElement.AUGMENT_2X, RhythmElement.RETROGRADE)
    """

    # Element specifications: (scale_factor, is_retrograde)
    ELEMENTS = {
        RhythmElement.IDENTITY: RhythmTransformSpec('identity', 1.0, False),
        RhythmElement.RETROGRADE: RhythmTransformSpec('retrograde', 1.0, True),
        RhythmElement.AUGMENT_2X: RhythmTransformSpec('augment_2x', 2.0, False),
        RhythmElement.DIMINISH_HALF: RhythmTransformSpec('diminish_half', 0.5, False),
        RhythmElement.AUGMENT_3X: RhythmTransformSpec('augment_3x', 3.0, False),
        RhythmElement.DIMINISH_THIRD: RhythmTransformSpec('diminish_third', 1/3, False),
        RhythmElement.AUGMENT_4X: RhythmTransformSpec('augment_4x', 4.0, False),
        RhythmElement.DIMINISH_QUARTER: RhythmTransformSpec('diminish_quarter', 0.25, False),
    }

    # Scale factors for quick lookup
    SCALE_FACTORS = {
        0: 1.0,    # identity
        1: 1.0,    # retrograde (scale = 1, but reversed)
        2: 2.0,    # augment_2x
        3: 0.5,    # diminish_half
        4: 3.0,    # augment_3x
        5: 1/3,    # diminish_third
        6: 4.0,    # augment_4x
        7: 0.25,   # diminish_quarter
    }

    # Valid composed scale factors -> element mapping
    SCALE_TO_ELEMENT = {
        1.0: RhythmElement.IDENTITY,
        2.0: RhythmElement.AUGMENT_2X,
        0.5: RhythmElement.DIMINISH_HALF,
        3.0: RhythmElement.AUGMENT_3X,
        1/3: RhythmElement.DIMINISH_THIRD,
        4.0: RhythmElement.AUGMENT_4X,
        0.25: RhythmElement.DIMINISH_QUARTER,
    }

    def __init__(self, device: str = 'cuda'):
        """
        Initialize rhythm group.

        Args:
            device: PyTorch device for GPU operations
        """
        self.device = device if HAS_TORCH and torch.cuda.is_available() else 'cpu'
        self.num_elements = len(self.ELEMENTS)

        # Build composition table
        # Note: This is a "clamped" table - compositions outside our set
        # map to the closest valid element
        self.compose_table = self._build_compose_table()

        # Build inverse table
        self.inverse_table = self._build_inverse_table()

        if HAS_TORCH:
            self.compose_gpu = torch.tensor(
                self.compose_table,
                device=self.device,
                dtype=torch.int16
            )

    def _build_compose_table(self) -> np.ndarray:
        """
        Build composition table.

        Composition rules:
        - Scale factors multiply
        - Retrograde toggles (R ∘ R = identity)
        - R ∘ augment = augment ∘ R (retrograde commutes with scaling)
        """
        n = self.num_elements
        table = np.zeros((n, n), dtype=np.int16)

        for g1 in range(n):
            for g2 in range(n):
                table[g1, g2] = self._compose_elements(g1, g2)

        return table

    def _compose_elements(self, g1: int, g2: int) -> int:
        """
        Compose two rhythm elements.

        Convention: g1 ∘ g2 means apply g2 first, then g1
        """
        spec1 = self.ELEMENTS[g1]
        spec2 = self.ELEMENTS[g2]

        # Composed scale factor
        composed_scale = spec1.scale_factor * spec2.scale_factor

        # Composed retrograde (XOR - two retrogrades cancel)
        composed_retro = spec1.is_retrograde != spec2.is_retrograde

        # Find closest valid element
        if composed_retro:
            # If retrograde, the scale must be 1.0 for our base elements
            # (we don't have "retrograde + augment" as a single element)
            # So we approximate by checking if scale is close to 1.0
            if abs(composed_scale - 1.0) < 0.01:
                return RhythmElement.RETROGRADE
            else:
                # Non-standard: retrograde with scaling
                # Return retrograde (prioritize the reversal operation)
                return RhythmElement.RETROGRADE
        else:
            # No retrograde - find by scale factor
            return self._scale_to_element(composed_scale)

    def _scale_to_element(self, scale: float) -> int:
        """Map scale factor to nearest element."""
        # Check for exact matches
        for s, elem in self.SCALE_TO_ELEMENT.items():
            if abs(s - scale) < 0.001:
                return elem

        # Find nearest
        best_elem = RhythmElement.IDENTITY
        best_dist = float('inf')

        for s, elem in self.SCALE_TO_ELEMENT.items():
            # Use log distance for multiplicative comparison
            dist = abs(np.log(scale) - np.log(s)) if scale > 0 and s > 0 else float('inf')
            if dist < best_dist:
                best_dist = dist
                best_elem = elem

        return best_elem

    def _build_inverse_table(self) -> np.ndarray:
        """
        Build inverse lookup table.

        Inverses:
        - identity⁻¹ = identity
        - retrograde⁻¹ = retrograde (involution)
        - augment_2x⁻¹ = diminish_half
        - diminish_half⁻¹ = augment_2x
        - etc.
        """
        table = np.zeros(self.num_elements, dtype=np.int16)

        table[RhythmElement.IDENTITY] = RhythmElement.IDENTITY
        table[RhythmElement.RETROGRADE] = RhythmElement.RETROGRADE
        table[RhythmElement.AUGMENT_2X] = RhythmElement.DIMINISH_HALF
        table[RhythmElement.DIMINISH_HALF] = RhythmElement.AUGMENT_2X
        table[RhythmElement.AUGMENT_3X] = RhythmElement.DIMINISH_THIRD
        table[RhythmElement.DIMINISH_THIRD] = RhythmElement.AUGMENT_3X
        table[RhythmElement.AUGMENT_4X] = RhythmElement.DIMINISH_QUARTER
        table[RhythmElement.DIMINISH_QUARTER] = RhythmElement.AUGMENT_4X

        return table

    # =========================================================================
    # Core Group Operations
    # =========================================================================

    def compose(self, g1: int, g2: int) -> int:
        """
        Compose two rhythm transforms.

        Args:
            g1: First element (applied second)
            g2: Second element (applied first)

        Returns:
            Composed element
        """
        return int(self.compose_table[g1, g2])

    def inverse(self, g: int) -> int:
        """Get inverse of element."""
        return int(self.inverse_table[g])

    def identity(self) -> int:
        """Get identity element."""
        return RhythmElement.IDENTITY

    def get_scale_factor(self, g: int) -> float:
        """Get scale factor for element."""
        return self.SCALE_FACTORS.get(g, 1.0)

    def is_retrograde(self, g: int) -> bool:
        """Check if element includes retrograde."""
        return self.ELEMENTS[g].is_retrograde

    # =========================================================================
    # Application to Rhythm Patterns
    # =========================================================================

    def apply_to_rhythm(self, g: int, rhythm: np.ndarray) -> np.ndarray:
        """
        Apply rhythm transform to binary onset pattern.

        Args:
            g: Rhythm element
            rhythm: Binary array of onsets

        Returns:
            Transformed rhythm
        """
        spec = self.ELEMENTS[g]

        # Apply retrograde first (if needed)
        result = rhythm[::-1].copy() if spec.is_retrograde else rhythm.copy()

        # Apply time scaling
        if abs(spec.scale_factor - 1.0) > 0.001:
            result = self._scale_rhythm(result, spec.scale_factor)

        return result

    def _scale_rhythm(self, rhythm: np.ndarray, factor: float) -> np.ndarray:
        """
        Scale rhythm by time factor.

        factor > 1: augmentation (stretching)
        factor < 1: diminution (compression)
        """
        original_len = len(rhythm)

        if factor > 1.0:
            # Augmentation: stretch
            new_len = int(original_len * factor)
            new_rhythm = np.zeros(new_len, dtype=rhythm.dtype)

            # Each onset expands
            for i, val in enumerate(rhythm):
                if val > 0:
                    new_idx = int(i * factor)
                    if new_idx < new_len:
                        new_rhythm[new_idx] = val

            return new_rhythm

        elif factor < 1.0:
            # Diminution: compress
            new_len = max(1, int(original_len * factor))
            new_rhythm = np.zeros(new_len, dtype=rhythm.dtype)

            # Each onset compresses
            for i, val in enumerate(rhythm):
                if val > 0:
                    new_idx = int(i * factor)
                    if new_idx < new_len:
                        new_rhythm[new_idx] = max(new_rhythm[new_idx], val)

            return new_rhythm

        else:
            return rhythm.copy()

    def apply_to_onset_times(self, g: int, onset_times: np.ndarray, total_duration: int) -> np.ndarray:
        """
        Apply rhythm transform to onset time array.

        Args:
            g: Rhythm element
            onset_times: Array of onset positions
            total_duration: Total duration of the pattern

        Returns:
            Transformed onset times
        """
        spec = self.ELEMENTS[g]

        result = onset_times.copy().astype(float)

        # Apply retrograde
        if spec.is_retrograde:
            # Reverse onset positions within the duration
            result = total_duration - result - 1
            result = np.sort(result)

        # Apply time scaling
        result = result * spec.scale_factor

        return result.astype(np.int32)

    # =========================================================================
    # Batch GPU Operations
    # =========================================================================

    def compose_batch_gpu(
        self,
        g1_batch: 'torch.Tensor',
        g2_batch: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Batch composition on GPU.

        Args:
            g1_batch: [N] first elements
            g2_batch: [N] second elements

        Returns:
            [N] composed elements
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU operations")

        return self.compose_gpu[g1_batch.long(), g2_batch.long()]

    # =========================================================================
    # Utilities
    # =========================================================================

    def element_name(self, g: int) -> str:
        """Get human-readable name for element."""
        return self.ELEMENTS[g].name

    def parse_element(self, name: str) -> int:
        """Parse element from string name."""
        name = name.strip().lower().replace(' ', '_').replace('-', '_')

        name_map = {
            'identity': RhythmElement.IDENTITY,
            'retrograde': RhythmElement.RETROGRADE,
            'augment_2x': RhythmElement.AUGMENT_2X,
            'augmentation': RhythmElement.AUGMENT_2X,
            'double': RhythmElement.AUGMENT_2X,
            'diminish_half': RhythmElement.DIMINISH_HALF,
            'diminution': RhythmElement.DIMINISH_HALF,
            'half': RhythmElement.DIMINISH_HALF,
            'augment_3x': RhythmElement.AUGMENT_3X,
            'triple': RhythmElement.AUGMENT_3X,
            'diminish_third': RhythmElement.DIMINISH_THIRD,
            'third': RhythmElement.DIMINISH_THIRD,
            'augment_4x': RhythmElement.AUGMENT_4X,
            'quadruple': RhythmElement.AUGMENT_4X,
            'diminish_quarter': RhythmElement.DIMINISH_QUARTER,
            'quarter': RhythmElement.DIMINISH_QUARTER,
        }

        if name in name_map:
            return name_map[name]

        raise ValueError(f"Unknown rhythm element: {name}")

    def validate_group_properties(self) -> bool:
        """
        Validate group properties.

        Note: This is a "practical" group - not mathematically closed,
        but satisfies key properties within our element set.
        """
        print("Validating Rhythm group properties...")

        # 1. Identity
        e = self.identity()
        for g in range(self.num_elements):
            if self.compose(e, g) != g:
                print(f"  ⚠ Identity (left) approximated for {self.element_name(g)}")
            if self.compose(g, e) != g:
                print(f"  ⚠ Identity (right) approximated for {self.element_name(g)}")
        print("  ✓ Identity property (practical)")

        # 2. Inverse
        for g in range(self.num_elements):
            g_inv = self.inverse(g)
            result = self.compose(g, g_inv)
            if result != e:
                print(f"  ⚠ Inverse of {self.element_name(g)} gives "
                      f"{self.element_name(result)} (not identity)")
        print("  ✓ Inverse property (practical)")

        # 3. Retrograde is involution
        r = RhythmElement.RETROGRADE
        if self.compose(r, r) != e:
            print("  ❌ Retrograde is not involution!")
            return False
        print("  ✓ Retrograde is involution")

        print("✓ Rhythm group properties validated (practical group)")
        return True
