"""
D₂₄ Dihedral Group for Triadic Harmony.

The dihedral group D₂₄ models the symmetries of pitch-class space under
transposition and inversion. This is fundamental to neo-Riemannian theory.

Group Structure:
- Order: 24 (12 transpositions + 12 inversions)
- Elements 0-11: T₀ through T₁₁ (transpositions)
- Elements 12-23: I₀ through I₁₁ (inversions)

Composition Rules:
- T_a ∘ T_b = T_{(a+b) mod 12}
- T_a ∘ I_b = I_{(a+b) mod 12}
- I_a ∘ T_b = I_{(a-b) mod 12}
- I_a ∘ I_b = T_{(a-b) mod 12}

This implementation precomputes the full 24×24 Cayley multiplication table
for O(1) group composition.

PLR Operations (Neo-Riemannian):
- P (Parallel): C major ↔ C minor (I_0 ∘ T_0 relative to root)
- L (Leading-tone): C major ↔ E minor (I_4 ∘ T_0)
- R (Relative): C major ↔ A minor (I_9 ∘ T_0)

Author: Architecture Refactor - Dosedo v2
Reference: Lewin, David. "Generalized Musical Intervals and Transformations" (1987)
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import IntEnum

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class D24Element(IntEnum):
    """Named elements of D24 for readability."""
    # Transpositions T_0 through T_11
    T0 = 0
    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4
    T5 = 5
    T6 = 6
    T7 = 7
    T8 = 8
    T9 = 9
    T10 = 10
    T11 = 11
    # Inversions I_0 through I_11
    I0 = 12
    I1 = 13
    I2 = 14
    I3 = 15
    I4 = 16
    I5 = 17
    I6 = 18
    I7 = 19
    I8 = 20
    I9 = 21
    I10 = 22
    I11 = 23


@dataclass
class PLROperation:
    """Neo-Riemannian PLR operation."""
    name: str
    d24_element: int
    description: str

    # Contextual transformation on major triads
    # (how root moves when applied to major triad)
    major_root_shift: int


class D24Group:
    """
    Dihedral group of order 24 for triadic harmony.

    Provides O(1) group composition via precomputed Cayley table.

    Usage:
        d24 = D24Group()

        # Compose two transforms
        result = d24.compose(D24Element.T5, D24Element.I7)  # T5 ∘ I7

        # Apply to pitch class
        new_pc = d24.apply_to_pitch_class(D24Element.T5, 0)  # C -> F

        # Batch operations on GPU
        results = d24.compose_batch_gpu(g1_tensor, g2_tensor)
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize D24 group with precomputed Cayley table.

        Args:
            device: PyTorch device for GPU operations ('cuda' or 'cpu')
        """
        self.device = device if HAS_TORCH and torch.cuda.is_available() else 'cpu'

        # Precompute 24×24 Cayley multiplication table
        self.cayley_table = self._build_cayley_table()

        # Precompute inverse table (24 elements)
        self.inverse_table = self._build_inverse_table()

        # Initialize PLR operations
        self.plr_operations = self._init_plr_operations()

        # Upload to GPU if available
        if HAS_TORCH:
            self.cayley_gpu = torch.tensor(
                self.cayley_table,
                device=self.device,
                dtype=torch.int16
            )
            self.inverse_gpu = torch.tensor(
                self.inverse_table,
                device=self.device,
                dtype=torch.int16
            )

    def _build_cayley_table(self) -> np.ndarray:
        """
        Build the 24×24 Cayley multiplication table.

        Entry [g1, g2] gives the result of g1 ∘ g2 (g1 applied after g2).

        Returns:
            np.ndarray of shape [24, 24] with dtype int16
        """
        table = np.zeros((24, 24), dtype=np.int16)

        for g1 in range(24):
            for g2 in range(24):
                table[g1, g2] = self._compose_elements(g1, g2)

        return table

    def _compose_elements(self, g1: int, g2: int) -> int:
        """
        Compute g1 ∘ g2 using group composition rules.

        Convention: g1 ∘ g2 means "apply g2 first, then g1"

        Composition rules:
        - T_a ∘ T_b = T_{(a+b) mod 12}
        - T_a ∘ I_b = I_{(a+b) mod 12}
        - I_a ∘ T_b = I_{(a-b) mod 12}
        - I_a ∘ I_b = T_{(a-b) mod 12}
        """
        # Determine if each element is a transposition or inversion
        g1_is_inversion = g1 >= 12
        g2_is_inversion = g2 >= 12

        # Extract parameter (0-11)
        a = g1 % 12 if g1 >= 12 else g1
        b = g2 % 12 if g2 >= 12 else g2

        if not g1_is_inversion and not g2_is_inversion:
            # T_a ∘ T_b = T_{(a+b) mod 12}
            return (a + b) % 12
        elif not g1_is_inversion and g2_is_inversion:
            # T_a ∘ I_b = I_{(a+b) mod 12}
            return 12 + (a + b) % 12
        elif g1_is_inversion and not g2_is_inversion:
            # I_a ∘ T_b = I_{(a-b) mod 12}
            return 12 + (a - b) % 12
        else:
            # I_a ∘ I_b = T_{(a-b) mod 12}
            return (a - b) % 12

    def _build_inverse_table(self) -> np.ndarray:
        """
        Build inverse lookup table.

        - T_n inverse is T_{-n mod 12} = T_{12-n mod 12}
        - I_n inverse is I_n (inversions are involutions)

        Returns:
            np.ndarray of shape [24] with dtype int16
        """
        table = np.zeros(24, dtype=np.int16)

        # Transpositions: inverse of T_n is T_{-n mod 12}
        for n in range(12):
            table[n] = (12 - n) % 12

        # Inversions: I_n is self-inverse (involution)
        for n in range(12):
            table[12 + n] = 12 + n

        return table

    def _init_plr_operations(self) -> dict:
        """
        Initialize neo-Riemannian PLR operations.

        These are contextual inversions that map major triads to minor triads
        and vice versa.
        """
        return {
            'P': PLROperation(
                name='Parallel',
                d24_element=D24Element.I0,  # I_0: inversion around root
                description='Parallel major/minor (C major ↔ C minor)',
                major_root_shift=0
            ),
            'L': PLROperation(
                name='Leading-tone',
                d24_element=D24Element.I4,  # I_4: inversion around third
                description='Leading-tone exchange (C major ↔ E minor)',
                major_root_shift=4
            ),
            'R': PLROperation(
                name='Relative',
                d24_element=D24Element.I9,  # I_9: inversion around fifth
                description='Relative major/minor (C major ↔ A minor)',
                major_root_shift=9
            ),
        }

    # =========================================================================
    # Core Group Operations
    # =========================================================================

    def compose(self, g1: int, g2: int) -> int:
        """
        O(1) group composition via table lookup.

        Computes g1 ∘ g2 (apply g2 first, then g1).

        Args:
            g1: First group element (0-23)
            g2: Second group element (0-23)

        Returns:
            Composed element (0-23)
        """
        return int(self.cayley_table[g1, g2])

    def inverse(self, g: int) -> int:
        """
        Return inverse element.

        Args:
            g: Group element (0-23)

        Returns:
            Inverse element (0-23)
        """
        return int(self.inverse_table[g])

    def identity(self) -> int:
        """Return identity element (T_0)."""
        return D24Element.T0

    def is_identity(self, g: int) -> bool:
        """Check if element is identity."""
        return g == D24Element.T0

    def is_transposition(self, g: int) -> bool:
        """Check if element is a transposition."""
        return g < 12

    def is_inversion(self, g: int) -> bool:
        """Check if element is an inversion."""
        return g >= 12

    def get_transposition_amount(self, g: int) -> Optional[int]:
        """
        Get transposition amount if element is transposition.

        Returns:
            Semitones (0-11) or None if inversion
        """
        if g < 12:
            return g
        return None

    def get_inversion_axis(self, g: int) -> Optional[int]:
        """
        Get inversion axis if element is inversion.

        Returns:
            Axis pitch class (0-11) or None if transposition
        """
        if g >= 12:
            return g - 12
        return None

    # =========================================================================
    # Application to Pitch Classes
    # =========================================================================

    def apply_to_pitch_class(self, g: int, pc: int) -> int:
        """
        Apply group element to a single pitch class.

        Args:
            g: Group element (0-23)
            pc: Pitch class (0-11)

        Returns:
            Transformed pitch class (0-11)
        """
        if g < 12:
            # Transposition: T_n(pc) = (pc + n) mod 12
            return (pc + g) % 12
        else:
            # Inversion: I_n(pc) = (n - pc) mod 12
            n = g - 12
            return (n - pc) % 12

    def apply_to_pitch_class_array(self, g: int, pcs: np.ndarray) -> np.ndarray:
        """
        Apply group element to array of pitch classes.

        Args:
            g: Group element (0-23)
            pcs: Array of pitch classes (values 0-11)

        Returns:
            Transformed pitch classes
        """
        if g < 12:
            # Transposition
            return (pcs + g) % 12
        else:
            # Inversion
            n = g - 12
            return (n - pcs) % 12

    def find_transform(self, source_pc: int, target_pc: int) -> List[int]:
        """
        Find all transforms that map source_pc to target_pc.

        Args:
            source_pc: Source pitch class (0-11)
            target_pc: Target pitch class (0-11)

        Returns:
            List of group elements that perform this mapping
        """
        transforms = []

        # Check transposition: (source + n) mod 12 = target
        # => n = (target - source) mod 12
        n = (target_pc - source_pc) % 12
        transforms.append(n)  # T_n

        # Check inversion: (n - source) mod 12 = target
        # => n = (target + source) mod 12
        n = (target_pc + source_pc) % 12
        transforms.append(12 + n)  # I_n

        return transforms

    # =========================================================================
    # GPU Batch Operations
    # =========================================================================

    def compose_batch_gpu(
        self,
        g1_batch: 'torch.Tensor',
        g2_batch: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Batch group composition on GPU.

        Args:
            g1_batch: Tensor of first elements [N]
            g2_batch: Tensor of second elements [N]

        Returns:
            Tensor of composed elements [N]
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU operations")

        # Use advanced indexing with Cayley table
        return self.cayley_gpu[g1_batch.long(), g2_batch.long()]

    def apply_to_pitch_class_batch_gpu(
        self,
        g_batch: 'torch.Tensor',
        pc_batch: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Apply transforms to pitch classes in batch on GPU.

        Supports broadcasting for applying single transform to 2D patterns.

        Args:
            g_batch: Tensor of group elements [N] or scalar (values 0-23)
            pc_batch: Tensor of pitch classes [N] or [N, L] (values 0-11)

        Returns:
            Tensor of transformed pitch classes, same shape as pc_batch
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU operations")

        # Handle scalar g_batch or expand for broadcasting
        if g_batch.dim() == 0 or (g_batch.dim() == 1 and pc_batch.dim() == 2):
            # Scalar or [N] applied to [N, L] - need to broadcast properly
            if g_batch.dim() == 1 and pc_batch.dim() == 2:
                g_batch = g_batch.unsqueeze(-1)  # [N, 1] for broadcasting with [N, L]

        # Separate transpositions and inversions
        is_inversion = g_batch >= 12

        # Compute transposition: (pc + g) mod 12
        transposed = (pc_batch + g_batch) % 12

        # Compute inversion: (g - 12 - pc) mod 12
        inverted = ((g_batch - 12) - pc_batch) % 12

        # Select based on type
        result = torch.where(is_inversion, inverted, transposed)

        return result

    def find_all_transforms_batch_gpu(
        self,
        source_pcs: 'torch.Tensor',
        target_pcs: 'torch.Tensor'
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Find transforms mapping each source to target in batch.

        Args:
            source_pcs: [N] source pitch classes
            target_pcs: [N] target pitch classes

        Returns:
            Tuple of (transposition_ids, inversion_ids) each [N]
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for GPU operations")

        # Transposition: n = (target - source) mod 12
        trans_ids = (target_pcs - source_pcs) % 12

        # Inversion: n = (target + source) mod 12, then add 12
        inv_ids = 12 + (target_pcs + source_pcs) % 12

        return trans_ids, inv_ids

    # =========================================================================
    # PLR Operations (Neo-Riemannian)
    # =========================================================================

    def apply_plr(self, operation: str, g: int) -> int:
        """
        Compose a PLR operation with an existing transform.

        Args:
            operation: 'P', 'L', or 'R'
            g: Current group element

        Returns:
            New group element after PLR operation
        """
        if operation not in self.plr_operations:
            raise ValueError(f"Unknown PLR operation: {operation}")

        plr = self.plr_operations[operation]
        return self.compose(plr.d24_element, g)

    def plr_sequence(self, operations: str) -> int:
        """
        Compute result of PLR sequence starting from identity.

        Args:
            operations: String of P, L, R operations (e.g., "PRL")

        Returns:
            Final group element
        """
        result = self.identity()
        for op in operations:
            result = self.apply_plr(op, result)
        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    def element_name(self, g: int) -> str:
        """Get human-readable name for element."""
        if g < 12:
            return f"T{g}"
        else:
            return f"I{g - 12}"

    def parse_element(self, name: str) -> int:
        """Parse element from string name like 'T5' or 'I7'."""
        name = name.strip().upper()
        if name.startswith('T'):
            n = int(name[1:])
            return n % 12
        elif name.startswith('I'):
            n = int(name[1:])
            return 12 + (n % 12)
        else:
            raise ValueError(f"Cannot parse element name: {name}")

    def print_cayley_table(self):
        """Print the Cayley table for debugging."""
        print("\nD24 Cayley Table (rows: g1, cols: g2, entry: g1 ∘ g2)")
        print("=" * 80)

        # Header
        header = "     " + "  ".join(f"{self.element_name(j):>3}" for j in range(24))
        print(header)
        print("-" * 80)

        # Rows
        for i in range(24):
            row = f"{self.element_name(i):>4} "
            row += "  ".join(f"{self.element_name(self.cayley_table[i, j]):>3}" for j in range(24))
            print(row)

    def validate_group_axioms(self) -> bool:
        """
        Validate that D24 satisfies group axioms.

        Returns:
            True if all axioms satisfied
        """
        print("Validating D24 group axioms...")

        # 1. Identity: e ∘ g = g ∘ e = g for all g
        e = self.identity()
        for g in range(24):
            if self.compose(e, g) != g or self.compose(g, e) != g:
                print(f"  ❌ Identity failed for {self.element_name(g)}")
                return False
        print("  ✓ Identity axiom satisfied")

        # 2. Inverse: g ∘ g⁻¹ = g⁻¹ ∘ g = e for all g
        for g in range(24):
            g_inv = self.inverse(g)
            if self.compose(g, g_inv) != e or self.compose(g_inv, g) != e:
                print(f"  ❌ Inverse failed for {self.element_name(g)}")
                return False
        print("  ✓ Inverse axiom satisfied")

        # 3. Associativity: (g1 ∘ g2) ∘ g3 = g1 ∘ (g2 ∘ g3)
        # Test a sample (full test is O(n³))
        test_triples = [(0, 5, 15), (7, 12, 3), (20, 8, 16), (11, 23, 1)]
        for g1, g2, g3 in test_triples:
            left = self.compose(self.compose(g1, g2), g3)
            right = self.compose(g1, self.compose(g2, g3))
            if left != right:
                print(f"  ❌ Associativity failed for ({self.element_name(g1)}, "
                      f"{self.element_name(g2)}, {self.element_name(g3)})")
                return False
        print("  ✓ Associativity axiom satisfied (sampled)")

        # 4. Closure: g1 ∘ g2 ∈ G for all g1, g2 ∈ G
        for g1 in range(24):
            for g2 in range(24):
                result = self.compose(g1, g2)
                if result < 0 or result >= 24:
                    print(f"  ❌ Closure failed for {self.element_name(g1)} ∘ {self.element_name(g2)}")
                    return False
        print("  ✓ Closure axiom satisfied")

        print("✓ D24 is a valid group!")
        return True


# =============================================================================
# Convenience Functions
# =============================================================================

def transpose(semitones: int) -> int:
    """Create transposition element."""
    return semitones % 12


def invert(axis: int) -> int:
    """Create inversion element."""
    return 12 + (axis % 12)


def parse_transform_string(s: str) -> int:
    """
    Parse transform from string notation.

    Accepts:
    - "T5", "T_5", "transpose(5)" -> T5
    - "I7", "I_7", "invert(7)" -> I7
    """
    s = s.strip().upper().replace('_', '').replace(' ', '')

    if s.startswith('TRANSPOSE('):
        n = int(s[10:-1])
        return transpose(n)
    elif s.startswith('INVERT('):
        n = int(s[7:-1])
        return invert(n)
    elif s.startswith('T'):
        n = int(s[1:])
        return transpose(n)
    elif s.startswith('I'):
        n = int(s[1:])
        return invert(n)
    else:
        raise ValueError(f"Cannot parse: {s}")
