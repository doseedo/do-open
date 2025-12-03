"""
CPU-based sparse coding for MDL discovery.

Implements FISTA algorithm for: minimize ||X - D*a||² + λ||a||₁

Where:
    X: corpus (B, T, F)
    D: transform dictionary (M, T, F)
    a: sparse coefficients (B, M)
"""

import numpy as np
from typing import Tuple, Dict


class CPUSparseEncoder:
    """
    Sparse coding using FISTA algorithm on CPU/numpy.

    Slower than GPU version but works without CUDA.
    """

    def __init__(
        self,
        lambda_sparsity: float = 0.1,
        max_iters: int = 100,
        tolerance: float = 1e-4
    ):
        """
        Args:
            lambda_sparsity: L1 regularization strength (higher = sparser)
            max_iters: Maximum FISTA iterations
            tolerance: Convergence tolerance
        """
        self.lambda_sparsity = lambda_sparsity
        self.max_iters = max_iters
        self.tolerance = tolerance

    def soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding: sign(x) * max(|x| - λ, 0)"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def encode_batch(
        self,
        pieces_batch: np.ndarray,
        transforms_dict: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Encode batch using FISTA sparse coding.

        Args:
            pieces_batch: (B, T, F) - corpus pieces
            transforms_dict: (M, T, F) - transform dictionary
            verbose: Print progress

        Returns:
            encodings: (B, M) - sparse coefficients
            metrics: dict with reconstruction error and sparsity
        """
        B, T, F = pieces_batch.shape
        M = transforms_dict.shape[0]

        if verbose:
            print(f"Sparse coding: {B} pieces × {M} transforms")

        # Normalize pieces for numerical stability
        piece_norms = np.linalg.norm(pieces_batch.reshape(B, -1), axis=1, keepdims=True)
        piece_norms = np.maximum(piece_norms, 1e-10)  # Avoid division by zero
        pieces_normalized = pieces_batch / piece_norms.reshape(B, 1, 1)

        # Initialize coefficients
        a = np.zeros((B, M))

        # FISTA momentum
        y = a.copy()
        t = 1.0

        # Lipschitz constant (step size)
        # L ≈ largest eigenvalue of D^T D
        # Use safe estimate to avoid numerical issues
        dict_norm = np.linalg.norm(transforms_dict.reshape(M, -1), axis=1).max()
        if dict_norm < 1e-10:
            # Dictionary is zero, return zero encodings
            return a, {'iterations': 0, 'reconstruction_error_mean': 0.0,
                      'reconstruction_error_std': 0.0, 'sparsity_mean': 0.0, 'sparsity_std': 0.0}

        L = dict_norm ** 2
        step_size = 0.9 / L

        prev_loss = float('inf')
        divergence_count = 0

        for iteration in range(self.max_iters):
            # Reconstruction: (B, M) @ (M, T, F) -> (B, T, F)
            reconstruction = np.einsum('bm,mtf->btf', y, transforms_dict)

            # Residual (using normalized pieces)
            residual = pieces_normalized - reconstruction

            # Gradient: -2 * D^T * residual
            grad = -2 * np.einsum('mtf,btf->bm', transforms_dict, residual)

            # Gradient step with clipping for numerical stability
            a_new = y - step_size * grad
            a_new = np.clip(a_new, -1e3, 1e3)  # Prevent explosion

            # Proximal step (soft thresholding for L1)
            a_new = self.soft_threshold(a_new, step_size * self.lambda_sparsity)

            # FISTA momentum update
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = a_new + ((t - 1) / t_new) * (a_new - a)
            y = np.clip(y, -1e3, 1e3)  # Clip momentum too

            a = a_new
            t = t_new

            # Check convergence every 10 iterations
            if iteration % 10 == 0:
                reconstruction = np.einsum('bm,mtf->btf', a, transforms_dict)
                data_loss = np.sum((pieces_normalized - reconstruction) ** 2)
                sparsity_loss = self.lambda_sparsity * np.sum(np.abs(a))
                total_loss = data_loss + sparsity_loss

                if verbose and iteration % 20 == 0:
                    sparsity = np.sum(np.abs(a) > 1e-6) / (B * M) * 100
                    print(f"  Iter {iteration:3d}: Loss={total_loss:.2e}, Sparsity={sparsity:.1f}%")

                # Check for divergence
                if not np.isfinite(total_loss) or total_loss > 1e10:
                    if verbose:
                        print(f"  Warning: Diverged at iteration {iteration}, using last stable state")
                    break

                if abs(prev_loss - total_loss) < self.tolerance:
                    if verbose:
                        print(f"  Converged at iteration {iteration}")
                    break

                # Check if loss increased significantly (divergence sign)
                if total_loss > 10 * prev_loss:
                    divergence_count += 1
                    if divergence_count >= 3:
                        if verbose:
                            print(f"  Warning: Loss increasing, stopping early at iteration {iteration}")
                        break
                else:
                    divergence_count = 0

                prev_loss = total_loss

        # Final reconstruction (denormalize)
        final_reconstruction = np.einsum('bm,mtf->btf', a, transforms_dict)
        final_reconstruction = final_reconstruction * piece_norms.reshape(B, 1, 1)

        # Compute metrics (on original scale)
        reconstruction_error = np.sum((pieces_batch - final_reconstruction) ** 2, axis=(1, 2))
        sparsity_per_piece = np.sum(np.abs(a) > 1e-6, axis=1)

        metrics = {
            'iterations': iteration + 1,
            'reconstruction_error_mean': np.mean(reconstruction_error),
            'reconstruction_error_std': np.std(reconstruction_error),
            'sparsity_mean': np.mean(sparsity_per_piece),
            'sparsity_std': np.std(sparsity_per_piece),
        }

        return a, metrics

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


def build_transform_dictionary(
    transforms: list,
    corpus_shape: Tuple[int, int, int],
    transform_lib
) -> np.ndarray:
    """
    Build transform dictionary from transform list.

    Args:
        transforms: List of transform dicts with 'name' and 'amount'
        corpus_shape: (B, T, F) shape for creating example
        transform_lib: NumpyTransformLibrary instance

    Returns:
        dictionary: (M, T, F) - each transform applied to identity
    """
    B, T, F = corpus_shape
    M = len(transforms)

    dictionary = np.zeros((M, T, F))

    # Create identity input (all zeros except one note)
    identity = np.zeros((1, T, F))
    identity[0, 0, 60] = 1.0  # Single middle C note

    for i, transform in enumerate(transforms):
        try:
            result = transform_lib.apply_transform(
                identity.copy(),
                transform['name'],
                transform['amount']
            )
            dictionary[i] = result[0]

            # Normalize each transform to unit norm for numerical stability
            norm = np.linalg.norm(dictionary[i])
            if norm > 1e-10:
                dictionary[i] /= norm
        except Exception as e:
            print(f"Warning: Failed to apply {transform['name']}: {e}")
            dictionary[i] = identity[0]  # Fallback to identity

    return dictionary
