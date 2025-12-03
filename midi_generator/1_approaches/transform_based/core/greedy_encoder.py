"""
Greedy Matching Pursuit for stable, fast sparse coding.

Approximates: minimize ||X - D*a||² + λ||a||₁

Using greedy selection instead of iterative optimization.
Numerically stable, fast, good enough for MDL discovery.
"""

import numpy as np
from typing import Tuple, Dict


class GreedyEncoder:
    """
    Greedy matching pursuit for sparse coding.

    Algorithm:
    1. Start with residual = piece
    2. Find transform that best reduces residual
    3. Subtract its contribution
    4. Repeat max_atoms times or until no improvement

    Benefits:
    - No iterative optimization (no divergence)
    - Single pass (very fast)
    - Numerically stable
    - Good approximation of L1-regularized sparse coding
    """

    def __init__(self, max_atoms: int = 5, min_improvement: float = 0.01, composition_bonus: float = 0.0):
        """
        Args:
            max_atoms: Maximum number of transforms to use per piece
            min_improvement: Stop if improvement < this fraction of residual
            composition_bonus: Bonus factor for compositions (e.g., 0.2 = 20% error reduction bonus)
        """
        self.max_atoms = max_atoms
        self.min_improvement = min_improvement
        self.composition_bonus = composition_bonus

    def encode_batch(
        self,
        pieces_batch: np.ndarray,
        transforms_dict: np.ndarray,
        verbose: bool = False,
        transform_metadata: list = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Encode batch using greedy matching pursuit.

        Args:
            pieces_batch: (B, T, F) - corpus pieces
            transforms_dict: (M, T, F) - transform dictionary
            verbose: Print progress

        Returns:
            codes: (B, M) - sparse coefficients
            metrics: dict with reconstruction error and sparsity
        """
        B, T, F = pieces_batch.shape
        M = transforms_dict.shape[0]

        if verbose:
            print(f"Greedy encoding: {B} pieces × {M} transforms (max {self.max_atoms} atoms/piece)")

        codes = np.zeros((B, M))
        reconstruction_errors = np.zeros(B)

        # Precompute transform norms for efficiency
        transform_norms = np.sum(transforms_dict ** 2, axis=(1, 2))

        for b in range(B):
            codes[b] = self._encode_single(
                pieces_batch[b],
                transforms_dict,
                transform_norms,
                transform_metadata
            )

            # Compute final reconstruction
            reconstruction = np.einsum('m,mtf->tf', codes[b], transforms_dict)
            reconstruction_errors[b] = np.sum((pieces_batch[b] - reconstruction) ** 2)

            if verbose and (b + 1) % 10 == 0:
                print(f"  Encoded {b+1}/{B} pieces...")

        metrics = {
            'reconstruction_error_mean': reconstruction_errors.mean(),
            'reconstruction_error_std': reconstruction_errors.std(),
            'sparsity_mean': np.sum(np.abs(codes) > 1e-6, axis=1).mean(),
            'sparsity_std': np.sum(np.abs(codes) > 1e-6, axis=1).std(),
        }

        if verbose:
            print(f"  Reconstruction error (MSE): {metrics['reconstruction_error_mean']:.8f}")
            print(f"  Sparsity: {metrics['sparsity_mean']:.1f} transforms/piece")

        return codes, metrics

    def _encode_single(
        self,
        piece: np.ndarray,
        transforms_dict: np.ndarray,
        transform_norms: np.ndarray,
        transform_metadata: list = None
    ) -> np.ndarray:
        """
        Encode single piece using greedy selection.

        Args:
            piece: (T, F) - single piece
            transforms_dict: (M, T, F) - dictionary
            transform_norms: (M,) - precomputed ||transform_i||²

        Returns:
            codes: (M,) - sparse coefficients
        """
        M = transforms_dict.shape[0]
        codes = np.zeros(M)
        residual = piece.copy()
        residual_energy = np.sum(residual ** 2)

        for atom in range(self.max_atoms):
            best_idx = None
            best_error = float('inf')
            best_coeff = 0

            # Try each transform
            for i in range(M):
                if transform_norms[i] < 1e-10:
                    continue

                transform = transforms_dict[i]

                # Optimal coefficient for this transform
                # coeff = <residual, transform> / ||transform||²
                numerator = np.sum(residual * transform)
                coeff = numerator / transform_norms[i]

                # Error after using this transform
                projection = coeff * transform
                new_residual = residual - projection
                error = np.sum(new_residual ** 2)

                # Apply composition bonus (bias toward compositions)
                if self.composition_bonus > 0 and transform_metadata is not None:
                    if i < len(transform_metadata):
                        name = transform_metadata[i].get('name', '')
                        if '_o_' in name:  # This is a composition
                            error *= (1.0 - self.composition_bonus)  # Reduce effective error

                if error < best_error:
                    best_error = error
                    best_idx = i
                    best_coeff = coeff

            # Check if we got improvement
            if best_idx is None:
                break

            improvement = (residual_energy - best_error) / (residual_energy + 1e-10)

            if improvement < self.min_improvement:
                # Not worth adding another atom
                break

            # Accept this atom
            codes[best_idx] += best_coeff
            residual -= best_coeff * transforms_dict[best_idx]
            residual_energy = best_error

        return codes

    def compute_reconstruction_error(
        self,
        pieces_batch: np.ndarray,
        encodings: np.ndarray,
        transforms_dict: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-piece reconstruction error.

        Returns:
            errors: (B,) - MSE per piece
        """
        reconstruction = np.einsum('bm,mtf->btf', encodings, transforms_dict)
        mse = np.sum((pieces_batch - reconstruction) ** 2, axis=(1, 2))
        return mse

    def compute_reconstruction_quality(
        self,
        pieces_batch: np.ndarray,
        encodings: np.ndarray,
        transforms_dict: np.ndarray
    ) -> np.ndarray:
        """
        Compute reconstruction quality: 1 - (MSE / energy)

        Returns:
            quality: (B,) - quality per piece (0-1, higher is better)
        """
        reconstruction = np.einsum('bm,mtf->btf', encodings, transforms_dict)
        mse = np.sum((pieces_batch - reconstruction) ** 2, axis=(1, 2))
        energy = np.sum(pieces_batch ** 2, axis=(1, 2))
        quality = 1.0 - (mse / (energy + 1e-8))
        return np.clip(quality, 0, 1)
