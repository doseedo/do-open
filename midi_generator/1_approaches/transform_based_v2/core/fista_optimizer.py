"""
FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for L1-regularized sparse coding.

Solves: minimize ||X - D*a||² + λ||a||₁

Where:
- X: data (piece)
- D: dictionary (transforms)
- a: sparse coefficients
- λ: sparsity penalty

FISTA properly penalizes sparsity, unlike greedy matching pursuit which only minimizes reconstruction error.
This makes compositions attractive: using 1 composition is preferred over 2 primitives if they give similar reconstruction.
"""

import numpy as np
from typing import Tuple, Dict


class FISTAOptimizer:
    """
    FISTA optimizer for L1-regularized sparse coding.

    Benefits over greedy matching pursuit:
    - Explicitly minimizes ||a||₁ (L1 norm = sparsity)
    - Global optimization (not greedy)
    - Compositions are preferred if they reduce atom count

    Trade-off:
    - ~10-20× slower than greedy (iterative optimization)
    - Worth it for discovery where we need correct MDL ranking
    """

    def __init__(self, lambda_sparsity: float = 0.0001, max_iterations: int = 200, tolerance: float = 1e-6):
        """
        Args:
            lambda_sparsity: L1 regularization strength (higher = sparser)
            max_iterations: Max FISTA iterations
            tolerance: Convergence threshold
        """
        self.lambda_sparsity = lambda_sparsity
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def encode_batch(
        self,
        pieces_batch: np.ndarray,
        transforms_dict: np.ndarray,
        verbose: bool = False,
        transform_metadata: list = None  # Ignored for FISTA (for API compatibility)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Encode batch using FISTA.

        Args:
            pieces_batch: (B, T, F) - corpus pieces
            transforms_dict: (M, T, F) - transform dictionary
            verbose: Print progress
            transform_metadata: Ignored (for API compatibility with GreedyEncoder)

        Returns:
            codes: (B, M) - sparse coefficients
            metrics: dict with reconstruction error and sparsity
        """
        B, T, F = pieces_batch.shape
        M = transforms_dict.shape[0]

        if verbose:
            print(f"FISTA encoding: {B} pieces × {M} transforms (λ={self.lambda_sparsity})")

        codes = np.zeros((B, M))
        reconstruction_errors = np.zeros(B)

        # Precompute D^T D for efficiency
        D_flat = transforms_dict.reshape(M, -1)  # (M, T*F)

        # Normalize dictionary columns for numerical stability
        D_norms = np.linalg.norm(D_flat, axis=1, keepdims=True)
        D_flat_normalized = D_flat / (D_norms + 1e-10)

        DtD = D_flat_normalized @ D_flat_normalized.T  # (M, M)

        # Lipschitz constant for gradient step (largest eigenvalue of D^T D)
        L = np.linalg.norm(DtD, ord=2)
        step_size = 1.0 / (L + 1e-8)

        for b in range(B):
            piece_flat = pieces_batch[b].reshape(-1)  # (T*F,)
            piece_norm = np.linalg.norm(piece_flat) + 1e-10

            # Normalize piece
            piece_normalized = piece_flat / piece_norm

            # Run FISTA in normalized space
            codes_normalized = self._fista_single(
                piece_normalized,
                D_flat_normalized,
                DtD,
                step_size
            )

            # Debug output for first piece
            if b == 0 and verbose:
                print(f"[DEBUG] codes_norm: min={codes_normalized.min():.2e}, "
                      f"max={codes_normalized.max():.2e}, "
                      f"nonzero={np.sum(np.abs(codes_normalized) > 1e-6)}/{M}, "
                      f"L={1.0/step_size:.2f}, "
                      f"threshold={0.1 * step_size:.6f}")

            # Compute error in normalized space
            reconstruction_normalized = D_flat_normalized.T @ codes_normalized
            error_normalized = np.sum((piece_normalized - reconstruction_normalized) ** 2)

            # Scale error back to original space
            reconstruction_errors[b] = error_normalized * (piece_norm ** 2)

            # Denormalize codes for output (so they work with original transforms)
            codes[b] = codes_normalized * (piece_norm / (D_norms.flatten() + 1e-10))

            if verbose and (b + 1) % 100 == 0:
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

    def _fista_single(
        self,
        x: np.ndarray,
        D: np.ndarray,
        DtD: np.ndarray,
        step_size: float
    ) -> np.ndarray:
        """
        FISTA for single sample.

        Args:
            x: (T*F,) - flattened piece
            D: (M, T*F) - flattened dictionary
            DtD: (M, M) - precomputed D^T D
            step_size: Gradient step size (1/L)

        Returns:
            a: (M,) - sparse coefficients
        """
        M = D.shape[0]

        # Initialize
        a = np.zeros(M)
        y = a.copy()
        t = 1.0

        # Precompute D^T x
        Dtx = D @ x  # (M,)

        for iteration in range(self.max_iterations):
            a_old = a.copy()

            # Gradient of (1/2)||x - D.T @ a||² w.r.t. a is: DtD @ a - Dtx
            gradient = DtD @ y - Dtx

            # Gradient step
            z = y - step_size * gradient

            # Soft thresholding (proximal operator for L1)
            threshold = self.lambda_sparsity * step_size
            a = self._soft_threshold(z, threshold)

            # FISTA momentum
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            y = a + ((t - 1.0) / t_new) * (a - a_old)
            t = t_new

            # Check convergence
            change = np.linalg.norm(a - a_old)
            if change < self.tolerance:
                break

        return a

    @staticmethod
    def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Soft thresholding operator (proximal operator for L1 norm).

        prox_{λ||·||₁}(x) = sign(x) * max(|x| - λ, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

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
