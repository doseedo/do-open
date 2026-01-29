"""
Transform Groups with Precomputed Cayley Tables.

Implements algebraic transform groups for O(1) composition:
- D24Group: Dihedral group of order 24 for triadic harmony (12 transpositions + 12 inversions)
- RhythmGroup: Multiplicative group for rhythmic transforms

The key insight: By precomputing the Cayley multiplication table,
group composition becomes a single array lookup - O(1) instead of O(n).

GPU Optimization:
- Cayley tables uploaded to GPU as constant tensors
- Batch composition for N patterns in single kernel launch
- All transforms applied in parallel

Author: Doseedoo Architecture V2
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from functools import lru_cache

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# D24 DIHEDRAL GROUP FOR TRIADIC HARMONY
# =============================================================================

class D24Group:
    """
    Dihedral group of order 24 for triadic harmony.

    Elements:
        - 0-11: Transpositions T₀ through T₁₁
        - 12-23: Inversions I₀ through I₁₁

    Group Structure:
        - T_n ∘ T_m = T_{(n+m) mod 12}
        - I_n ∘ I_m = T_{(n-m) mod 12}
        - T_n ∘ I_m = I_{(n+m) mod 12}
        - I_n ∘ T_m = I_{(n-m) mod 12}

    This is the dihedral group D₁₂ extended with pitch-class operations.
    Also includes PLR operations (Neo-Riemannian).
    """

    # Group size
    ORDER = 24

    # Named elements for clarity
    T = list(range(12))      # Transpositions T₀-T₁₁
    I = list(range(12, 24))  # Inversions I₀-I₁₁

    # PLR operations (indices into the 24 elements)
    # These are specific inversions that correspond to Neo-Riemannian operations
    P = 12  # Parallel: I₀ (invert at 0)
    L = 16  # Leading-tone: I₄
    R = 21  # Relative: I₉

    def __init__(self, device: str = 'cpu'):
        """
        Initialize D24 group with precomputed Cayley table.

        Args:
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.device = device

        # Precompute 24×24 Cayley multiplication table
        self.cayley_table = self._build_cayley_table()

        # Precompute inverse table
        self.inverse_table = self._build_inverse_table()

        # Upload to GPU if requested and available
        self.cayley_gpu = None
        self.inverse_gpu = None
        if device == 'cuda' and HAS_TORCH and torch.cuda.is_available():
            self.cayley_gpu = torch.tensor(
                self.cayley_table, device='cuda', dtype=torch.int16
            )
            self.inverse_gpu = torch.tensor(
                self.inverse_table, device='cuda', dtype=torch.int16
            )

    def _build_cayley_table(self) -> np.ndarray:
        """
        Build 24×24 Cayley multiplication table.

        table[g1, g2] = g1 ∘ g2 (g1 applied after g2)

        Returns:
            [24, 24] int array of group element indices
        """
        table = np.zeros((24, 24), dtype=np.int16)

        for g1 in range(24):
            for g2 in range(24):
                table[g1, g2] = self._compose_elements(g1, g2)

        return table

    def _compose_elements(self, g1: int, g2: int) -> int:
        """
        Compose two group elements.

        Convention: g1 ∘ g2 means apply g2 first, then g1.

        Args:
            g1: First element (0-23)
            g2: Second element (0-23)

        Returns:
            Composed element index (0-23)
        """
        # Decode elements
        is_t1 = g1 < 12
        is_t2 = g2 < 12
        n1 = g1 if is_t1 else g1 - 12
        n2 = g2 if is_t2 else g2 - 12

        if is_t1 and is_t2:
            # T_n1 ∘ T_n2 = T_{(n1+n2) mod 12}
            return (n1 + n2) % 12
        elif not is_t1 and not is_t2:
            # I_n1 ∘ I_n2 = T_{(n1-n2) mod 12}
            return (n1 - n2) % 12
        elif is_t1 and not is_t2:
            # T_n1 ∘ I_n2 = I_{(n1+n2) mod 12}
            return 12 + (n1 + n2) % 12
        else:
            # I_n1 ∘ T_n2 = I_{(n1-n2) mod 12}
            return 12 + (n1 - n2) % 12

    def _build_inverse_table(self) -> np.ndarray:
        """
        Build inverse lookup table.

        Returns:
            [24] int array where inverse_table[g] = g⁻¹
        """
        table = np.zeros(24, dtype=np.int16)

        for g in range(24):
            table[g] = self._compute_inverse(g)

        return table

    def _compute_inverse(self, g: int) -> int:
        """
        Compute inverse of group element.

        T_n⁻¹ = T_{-n mod 12}
        I_n⁻¹ = I_n (inversions are involutory)

        Args:
            g: Element index (0-23)

        Returns:
            Inverse element index
        """
        if g < 12:
            # T_n inverse is T_{-n mod 12}
            return (-g) % 12
        else:
            # I_n is self-inverse (involution)
            return g

    def compose(self, g1: int, g2: int) -> int:
        """
        O(1) group composition via table lookup.

        Args:
            g1: First element index
            g2: Second element index

        Returns:
            Composed element index
        """
        return int(self.cayley_table[g1, g2])

    def inverse(self, g: int) -> int:
        """
        Return inverse element.

        Args:
            g: Element index

        Returns:
            Inverse element index
        """
        return int(self.inverse_table[g])

    def apply_to_pitch_class(self, g: int, pc: int) -> int:
        """
        Apply group element to pitch class.

        Args:
            g: Group element index (0-23)
            pc: Pitch class (0-11)

        Returns:
            Transformed pitch class (0-11)
        """
        if g < 12:
            # Transposition: T_n(pc) = (pc + n) mod 12
            return (pc + g) % 12
        else:
            # Inversion: I_n(pc) = (n - pc) mod 12
            return ((g - 12) - pc) % 12

    def apply_to_pitch_class_batch(self, g: int, pcs: np.ndarray) -> np.ndarray:
        """
        Apply group element to array of pitch classes.

        Args:
            g: Group element index
            pcs: Array of pitch classes

        Returns:
            Transformed pitch classes
        """
        if g < 12:
            return (pcs + g) % 12
        else:
            return ((g - 12) - pcs) % 12

    def compose_batch_gpu(
        self, g1_batch: 'torch.Tensor', g2_batch: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Batch composition on GPU.

        Args:
            g1_batch: [N] tensor of first elements
            g2_batch: [N] tensor of second elements

        Returns:
            [N] tensor of composed elements
        """
        if self.cayley_gpu is None:
            raise RuntimeError("GPU not initialized. Use device='cuda'")

        return self.cayley_gpu[g1_batch, g2_batch]

    def element_name(self, g: int) -> str:
        """Get human-readable name for element."""
        if g < 12:
            return f"T{g}"
        else:
            return f"I{g-12}"

    def find_transform(self, source_pc: int, target_pc: int) -> int:
        """
        Find transposition that maps source to target pitch class.

        Args:
            source_pc: Source pitch class (0-11)
            target_pc: Target pitch class (0-11)

        Returns:
            Transposition element (0-11) such that T[source] = target
        """
        return (target_pc - source_pc) % 12

    def find_all_transforms(
        self, source_pcs: np.ndarray, target_pcs: np.ndarray
    ) -> List[int]:
        """
        Find all group elements that map source to target pitch class sequence.

        Returns empty list if no consistent transform exists.

        Args:
            source_pcs: Source pitch classes
            target_pcs: Target pitch classes

        Returns:
            List of valid transform element indices
        """
        if len(source_pcs) != len(target_pcs) or len(source_pcs) == 0:
            return []

        valid = []

        # Check all 24 transforms
        for g in range(24):
            transformed = self.apply_to_pitch_class_batch(g, source_pcs)
            if np.array_equal(transformed, target_pcs):
                valid.append(g)

        return valid

    # Neo-Riemannian Operations
    def parallel(self, root_pc: int) -> int:
        """P operation: Major <-> Minor (same root)."""
        return self.apply_to_pitch_class(self.P, root_pc)

    def leading_tone(self, root_pc: int) -> int:
        """L operation: Leading-tone exchange."""
        return self.apply_to_pitch_class(self.L, root_pc)

    def relative(self, root_pc: int) -> int:
        """R operation: Relative major/minor."""
        return self.apply_to_pitch_class(self.R, root_pc)

    def __repr__(self) -> str:
        return f"D24Group(device='{self.device}')"


# =============================================================================
# RHYTHM GROUP
# =============================================================================

class RhythmGroup:
    """
    Multiplicative group for rhythmic transforms.

    Elements:
        0: identity (×1.0)
        1: augment_2x (×2.0)
        2: diminish_half (×0.5)
        3: augment_3x (×3.0)
        4: diminish_third (×1/3)
        5: augment_4x (×4.0)
        6: diminish_quarter (×0.25)
        7: retrograde (reversal - special)

    The first 7 elements form a multiplicative group over duration scaling.
    Retrograde is special: it's an involution that reverses time order.
    """

    ORDER = 8

    # Element definitions: (name, scale_factor)
    # scale_factor = -1 indicates retrograde (special case)
    ELEMENTS = {
        0: ('identity', 1.0),
        1: ('augment_2x', 2.0),
        2: ('diminish_half', 0.5),
        3: ('augment_3x', 3.0),
        4: ('diminish_third', 1/3),
        5: ('augment_4x', 4.0),
        6: ('diminish_quarter', 0.25),
        7: ('retrograde', -1),  # Special: reversal
    }

    def __init__(self, device: str = 'cpu'):
        """Initialize rhythm group with Cayley table."""
        self.device = device

        # Build Cayley table for the group
        self.cayley_table = self._build_cayley_table()
        self.inverse_table = self._build_inverse_table()

        # GPU tensors
        self.cayley_gpu = None
        if device == 'cuda' and HAS_TORCH and torch.cuda.is_available():
            self.cayley_gpu = torch.tensor(
                self.cayley_table, device='cuda', dtype=torch.int16
            )

    def _build_cayley_table(self) -> np.ndarray:
        """Build 8×8 Cayley table."""
        table = np.zeros((8, 8), dtype=np.int16)

        # Non-retrograde compositions (multiplicative)
        for g1 in range(7):
            for g2 in range(7):
                _, scale1 = self.ELEMENTS[g1]
                _, scale2 = self.ELEMENTS[g2]
                composed_scale = scale1 * scale2
                # Find closest element
                table[g1, g2] = self._find_closest_scale(composed_scale)

        # Retrograde compositions
        # retrograde ∘ retrograde = identity
        table[7, 7] = 0

        # retrograde ∘ scale = retrograde_scale (we approximate as retrograde)
        # scale ∘ retrograde = scale_retrograde
        for g in range(7):
            table[7, g] = 7  # retrograde first, then scale -> still retrograde
            table[g, 7] = 7  # scale first, then retrograde -> still retrograde

        return table

    def _find_closest_scale(self, target_scale: float) -> int:
        """Find element index with closest scale factor."""
        best_idx = 0
        best_diff = float('inf')

        for idx in range(7):  # Exclude retrograde
            _, scale = self.ELEMENTS[idx]
            diff = abs(scale - target_scale)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

        return best_idx

    def _build_inverse_table(self) -> np.ndarray:
        """Build inverse lookup table."""
        table = np.zeros(8, dtype=np.int16)

        # Multiplicative inverses
        table[0] = 0  # identity
        table[1] = 2  # 2x ↔ 0.5x
        table[2] = 1
        table[3] = 4  # 3x ↔ 1/3x
        table[4] = 3
        table[5] = 6  # 4x ↔ 0.25x
        table[6] = 5
        table[7] = 7  # retrograde is self-inverse

        return table

    def compose(self, g1: int, g2: int) -> int:
        """O(1) composition via table lookup."""
        return int(self.cayley_table[g1, g2])

    def inverse(self, g: int) -> int:
        """Return inverse element."""
        return int(self.inverse_table[g])

    def get_scale_factor(self, g: int) -> float:
        """Get scale factor for element (returns -1 for retrograde)."""
        return self.ELEMENTS[g][1]

    def apply_to_rhythm(self, g: int, rhythm: np.ndarray) -> np.ndarray:
        """
        Apply transform to rhythm pattern.

        Args:
            g: Group element index
            rhythm: Binary onset pattern [T]

        Returns:
            Transformed rhythm pattern
        """
        name, scale = self.ELEMENTS[g]

        if name == 'identity':
            return rhythm.copy()

        if name == 'retrograde':
            return rhythm[::-1].copy()

        # Time scaling
        T = len(rhythm)
        new_T = int(T * scale)

        if new_T <= 0:
            return np.zeros(1, dtype=rhythm.dtype)

        if scale > 1:
            # Augmentation: stretch
            result = np.zeros(new_T, dtype=rhythm.dtype)
            for i, val in enumerate(rhythm):
                if val > 0:
                    new_idx = int(i * scale)
                    if new_idx < new_T:
                        result[new_idx] = val
            return result
        else:
            # Diminution: compress
            result = np.zeros(new_T, dtype=rhythm.dtype)
            for i in range(new_T):
                old_idx = int(i / scale)
                if old_idx < T:
                    result[i] = rhythm[old_idx]
            return result

    def element_name(self, g: int) -> str:
        """Get human-readable name for element."""
        return self.ELEMENTS[g][0]

    def __repr__(self) -> str:
        return f"RhythmGroup(device='{self.device}')"


# =============================================================================
# VELOCITY GROUP (Optional extension)
# =============================================================================

class VelocityGroup:
    """
    Group for velocity/dynamics transforms.

    Velocity is typically treated as a multiplicative scaling.
    We also support quantization to standard dynamic levels.
    """

    # Standard dynamic levels (MIDI velocity approximations)
    DYNAMICS = {
        'ppp': 16,
        'pp': 32,
        'p': 48,
        'mp': 64,
        'mf': 80,
        'f': 96,
        'ff': 112,
        'fff': 127,
    }

    # Number of quantization levels
    N_LEVELS = 8

    @staticmethod
    def quantize_velocity(velocity: int, n_levels: int = 8) -> int:
        """
        Quantize MIDI velocity to reduced levels.

        Args:
            velocity: MIDI velocity (0-127)
            n_levels: Number of quantization levels (default 8)

        Returns:
            Quantized level (0 to n_levels-1)
        """
        # Map 0-127 to 0-(n_levels-1)
        return min(int(velocity * n_levels / 128), n_levels - 1)

    @staticmethod
    def dequantize_velocity(level: int, n_levels: int = 8) -> int:
        """
        Convert quantized level back to MIDI velocity.

        Args:
            level: Quantized level (0 to n_levels-1)
            n_levels: Number of quantization levels

        Returns:
            MIDI velocity (0-127)
        """
        # Map level to center of velocity range for that level
        bin_size = 128 / n_levels
        return int((level + 0.5) * bin_size)

    @staticmethod
    def quantize_batch(velocities: np.ndarray, n_levels: int = 8) -> np.ndarray:
        """Batch quantize velocities."""
        return np.minimum((velocities * n_levels / 128).astype(np.int32), n_levels - 1)

    @staticmethod
    def dequantize_batch(levels: np.ndarray, n_levels: int = 8) -> np.ndarray:
        """Batch dequantize levels to velocities."""
        bin_size = 128 / n_levels
        return ((levels + 0.5) * bin_size).astype(np.int32)


# =============================================================================
# CROSS-TRACK RELATIONSHIP TYPES
# =============================================================================

@dataclass
class CrossTrackRelation:
    """
    Enumeration of cross-track relationship types.

    These describe how one track relates to another musically.
    """
    # Melodic motion types
    PARALLEL_MOTION = 0      # Same direction, same interval
    CONTRARY_MOTION = 1      # Opposite directions
    SIMILAR_MOTION = 2       # Same direction, different interval
    OBLIQUE_MOTION = 3       # One voice stationary

    # Rhythmic relationships
    RHYTHMIC_UNISON = 4      # Same rhythm
    RHYTHMIC_OFFSET = 5      # Same rhythm, time-shifted
    RHYTHMIC_AUGMENT = 6     # Proportional rhythm (2x, 0.5x, etc.)
    RHYTHMIC_COMPLEMENT = 7  # Fills gaps in other voice

    # Voicing transforms
    VOICING_CLOSE = 8        # Close position
    VOICING_OPEN = 9         # Open position
    VOICING_DROP2 = 10       # Drop-2 voicing
    VOICING_DROP3 = 11       # Drop-3 voicing

    # Octave relationships
    OCTAVE_DOUBLE = 12       # Doubling at octave
    OCTAVE_DOUBLE_2 = 13     # Doubling at 2 octaves

    # Harmonic relationships
    THIRD_HARMONIZE = 14     # Parallel thirds
    SIXTH_HARMONIZE = 15     # Parallel sixths
    TENTH_HARMONIZE = 16     # Parallel tenths

    # Derived relationships
    INVERSION_DERIVE = 17    # Inverted version
    RETROGRADE_DERIVE = 18   # Retrograde version

    # Identity
    IDENTITY = 19            # Same content (copy)

    N_TYPES = 20


# =============================================================================
# COMBINED TRANSFORM ELEMENT
# =============================================================================

@dataclass
class TransformElement:
    """
    A combined transform element from all groups.

    This represents a single atomic transform that can be applied to
    a musical object, combining pitch-class, octave, rhythm, and
    velocity components.
    """
    pitch_class_transform: int = 0   # D24 element (0-23)
    octave_shift: int = 0            # Octave transposition (-2 to +2 typically)
    rhythm_transform: int = 0         # RhythmGroup element (0-7)
    velocity_scale: float = 1.0       # Velocity multiplier

    def compose(self, other: 'TransformElement',
                d24: D24Group, rhythm_group: RhythmGroup) -> 'TransformElement':
        """Compose two transform elements."""
        return TransformElement(
            pitch_class_transform=d24.compose(
                self.pitch_class_transform, other.pitch_class_transform
            ),
            octave_shift=self.octave_shift + other.octave_shift,
            rhythm_transform=rhythm_group.compose(
                self.rhythm_transform, other.rhythm_transform
            ),
            velocity_scale=self.velocity_scale * other.velocity_scale
        )

    def inverse(self, d24: D24Group, rhythm_group: RhythmGroup) -> 'TransformElement':
        """Get inverse transform."""
        return TransformElement(
            pitch_class_transform=d24.inverse(self.pitch_class_transform),
            octave_shift=-self.octave_shift,
            rhythm_transform=rhythm_group.inverse(self.rhythm_transform),
            velocity_scale=1.0 / self.velocity_scale if self.velocity_scale != 0 else 1.0
        )

    def is_identity(self) -> bool:
        """Check if this is the identity transform."""
        return (
            self.pitch_class_transform == 0 and
            self.octave_shift == 0 and
            self.rhythm_transform == 0 and
            abs(self.velocity_scale - 1.0) < 0.01
        )

    def __repr__(self) -> str:
        parts = []
        if self.pitch_class_transform != 0:
            parts.append(f"pc:T{self.pitch_class_transform}" if self.pitch_class_transform < 12
                        else f"pc:I{self.pitch_class_transform-12}")
        if self.octave_shift != 0:
            parts.append(f"oct:{self.octave_shift:+d}")
        if self.rhythm_transform != 0:
            parts.append(f"r:{RhythmGroup.ELEMENTS[self.rhythm_transform][0]}")
        if abs(self.velocity_scale - 1.0) > 0.01:
            parts.append(f"v:×{self.velocity_scale:.2f}")
        return f"Transform({', '.join(parts) if parts else 'identity'})"
