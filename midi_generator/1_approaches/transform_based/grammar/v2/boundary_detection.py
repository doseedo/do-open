"""
Step 4: Information-Theoretic Boundary Detection (GPU-Optimized)
================================================================

Find phrase boundaries without relying on meter using two approaches:

Option A: Self-Similarity Matrix + Novelty Kernel
- Compute pitch/rhythm similarity between all time positions
- Convolve with checkerboard kernel at multiple scales
- Peaks indicate boundaries at that scale

Option B: Information Content Spikes
- Train simple Markov model on corpus
- Compute surprise (information content) at each note
- Boundaries occur at surprise spikes

GPU Optimizations for A100 40GB:
- Batch self-similarity computation (O(n²) -> batched matrix ops)
- FFT-based convolution for novelty kernel
- Parallel Markov probability computation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time


@dataclass
class Boundary:
    """A detected phrase boundary."""
    position: int           # Position in sequence
    strength: float         # Boundary strength (0-1)
    scale: int = 0         # Scale at which boundary was detected
    boundary_type: str = 'novelty'  # 'novelty' or 'surprise'


@dataclass
class BoundaryResult:
    """Result of boundary detection."""
    boundaries: List[Boundary]
    sequence_length: int
    n_segments: int

    # Metrics
    avg_segment_length: float
    boundary_strength_mean: float
    boundary_strength_std: float

    def get_segment_indices(self) -> List[Tuple[int, int]]:
        """Get (start, end) indices for each segment."""
        segments = []
        positions = sorted([0] + [b.position for b in self.boundaries] + [self.sequence_length])

        for i in range(len(positions) - 1):
            if positions[i] < positions[i+1]:
                segments.append((positions[i], positions[i+1]))

        return segments


class SelfSimilarityBoundaryDetector:
    """
    Boundary detection using self-similarity matrix and novelty kernel.

    Algorithm:
    1. Compute feature vectors for each time position (pitch, duration, etc.)
    2. Build self-similarity matrix S[i,j] = similarity(feature[i], feature[j])
    3. Convolve with checkerboard novelty kernel at multiple scales
    4. Peaks in novelty curve indicate boundaries
    """

    def __init__(
        self,
        device: str = 'cuda',
        scales: List[int] = [4, 8, 16, 32],  # Kernel sizes
        threshold: float = 0.3,              # Novelty threshold
        min_segment_length: int = 4,         # Minimum notes between boundaries
        feature_dim: int = 24,               # Feature dimensionality
        verbose: bool = False,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.scales = scales
        self.threshold = threshold
        self.min_segment_length = min_segment_length
        self.feature_dim = feature_dim
        self.verbose = verbose

    def detect(
        self,
        pitch_sequence: List[int],
        duration_sequence: Optional[List[int]] = None,
        velocity_sequence: Optional[List[int]] = None,
    ) -> BoundaryResult:
        """
        Detect boundaries in a single sequence.

        Args:
            pitch_sequence: Pitch class sequence (0-11)
            duration_sequence: Optional duration sequence
            velocity_sequence: Optional velocity sequence

        Returns:
            BoundaryResult with detected boundaries
        """
        start_time = time.time()
        n = len(pitch_sequence)

        if n < self.min_segment_length * 2:
            return BoundaryResult(
                boundaries=[],
                sequence_length=n,
                n_segments=1,
                avg_segment_length=n,
                boundary_strength_mean=0.0,
                boundary_strength_std=0.0,
            )

        # Extract features
        features = self._extract_features(
            pitch_sequence, duration_sequence, velocity_sequence
        )

        # Compute self-similarity matrix (GPU)
        ssm = self._compute_ssm_gpu(features)

        # Compute novelty curve at multiple scales
        novelty_curves = []
        for scale in self.scales:
            if scale <= n:
                novelty = self._compute_novelty_gpu(ssm, scale)
                novelty_curves.append((scale, novelty))

        # Combine novelty curves
        combined_novelty = self._combine_novelty_curves(novelty_curves, n)

        # Find peaks (boundary candidates)
        boundaries = self._find_peaks(combined_novelty)

        # Filter by minimum segment length
        boundaries = self._filter_boundaries(boundaries, n)

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"[Boundary] Found {len(boundaries)} boundaries in {elapsed:.2f}s")

        # Compute statistics
        segment_lengths = self._compute_segment_lengths(boundaries, n)

        return BoundaryResult(
            boundaries=boundaries,
            sequence_length=n,
            n_segments=len(boundaries) + 1,
            avg_segment_length=np.mean(segment_lengths) if segment_lengths else n,
            boundary_strength_mean=np.mean([b.strength for b in boundaries]) if boundaries else 0,
            boundary_strength_std=np.std([b.strength for b in boundaries]) if boundaries else 0,
        )

    def detect_batch(
        self,
        sequences: List[List[int]],
        duration_sequences: Optional[List[List[int]]] = None,
    ) -> List[BoundaryResult]:
        """Detect boundaries in multiple sequences."""
        results = []

        for i, seq in enumerate(sequences):
            dur = duration_sequences[i] if duration_sequences else None
            result = self.detect(seq, dur)
            results.append(result)

            if self.verbose and i % 100 == 0:
                print(f"[Boundary] Processed {i+1}/{len(sequences)} sequences")

        return results

    def _extract_features(
        self,
        pitches: List[int],
        durations: Optional[List[int]],
        velocities: Optional[List[int]],
    ) -> torch.Tensor:
        """
        Extract feature vectors for each time position.

        Features:
        - One-hot pitch class (12 dims)
        - One-hot duration bucket (8 dims)
        - Normalized velocity (1 dim)
        - Interval from previous (1 dim)
        - Running pitch histogram (optional)
        """
        n = len(pitches)
        features = torch.zeros(n, self.feature_dim, device=self.device)

        for i, p in enumerate(pitches):
            # Pitch class one-hot (0-11)
            features[i, p % 12] = 1.0

            # Duration bucket (12-19)
            if durations:
                d_bucket = min(durations[i] // 120, 7)
                features[i, 12 + d_bucket] = 1.0

            # Velocity (20)
            if velocities:
                features[i, 20] = velocities[i] / 127.0

            # Interval from previous (21-23)
            if i > 0:
                interval = (pitches[i] - pitches[i-1]) % 12
                if interval <= 2:  # Stepwise
                    features[i, 21] = 1.0
                elif interval <= 5:  # Small leap
                    features[i, 22] = 1.0
                else:  # Large leap
                    features[i, 23] = 1.0

        return features

    def _compute_ssm_gpu(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute self-similarity matrix using GPU.

        SSM[i,j] = cosine_similarity(features[i], features[j])
        """
        # Normalize features
        features_norm = F.normalize(features, dim=1)

        # Compute similarity matrix (cosine similarity)
        ssm = torch.mm(features_norm, features_norm.t())

        # Ensure diagonal is 1
        ssm.fill_diagonal_(1.0)

        return ssm

    def _compute_novelty_gpu(
        self,
        ssm: torch.Tensor,
        kernel_size: int,
    ) -> torch.Tensor:
        """
        Compute novelty curve using checkerboard kernel convolution.

        The checkerboard kernel detects transitions between similar regions.
        """
        n = ssm.shape[0]

        # Create checkerboard kernel
        # [+1 -1]
        # [-1 +1]
        half = kernel_size // 2
        kernel = torch.ones(kernel_size, kernel_size, device=self.device)
        kernel[:half, :half] = 1.0
        kernel[half:, half:] = 1.0
        kernel[:half, half:] = -1.0
        kernel[half:, :half] = -1.0

        # Normalize kernel
        kernel = kernel / (kernel_size * kernel_size)

        # Reshape for conv2d: (batch, channel, H, W)
        ssm_4d = ssm.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

        # Pad SSM
        pad = kernel_size // 2
        ssm_padded = F.pad(ssm_4d, (pad, pad, pad, pad), mode='reflect')

        # Convolve
        novelty_2d = F.conv2d(ssm_padded, kernel_4d)

        # Extract diagonal (novelty at each position)
        novelty_2d = novelty_2d.squeeze()
        novelty = torch.diag(novelty_2d)

        # Take absolute value and normalize
        novelty = torch.abs(novelty)
        if novelty.max() > 0:
            novelty = novelty / novelty.max()

        return novelty

    def _combine_novelty_curves(
        self,
        novelty_curves: List[Tuple[int, torch.Tensor]],
        n: int,
    ) -> np.ndarray:
        """Combine novelty curves from multiple scales."""
        combined = np.zeros(n)

        for scale, novelty in novelty_curves:
            # Resize to full length
            novelty_np = novelty.cpu().numpy()

            # Pad or truncate
            if len(novelty_np) < n:
                # Pad with zeros
                padded = np.zeros(n)
                offset = (n - len(novelty_np)) // 2
                padded[offset:offset+len(novelty_np)] = novelty_np
                novelty_np = padded
            elif len(novelty_np) > n:
                novelty_np = novelty_np[:n]

            # Weight by scale (longer scales = more global boundaries)
            weight = np.log2(scale + 1)
            combined += weight * novelty_np

        # Normalize
        if combined.max() > 0:
            combined = combined / combined.max()

        return combined

    def _find_peaks(self, novelty: np.ndarray) -> List[Boundary]:
        """Find peaks in novelty curve above threshold."""
        boundaries = []
        n = len(novelty)

        for i in range(1, n - 1):
            # Local maximum
            if novelty[i] > novelty[i-1] and novelty[i] > novelty[i+1]:
                # Above threshold
                if novelty[i] > self.threshold:
                    boundaries.append(Boundary(
                        position=i,
                        strength=float(novelty[i]),
                        boundary_type='novelty',
                    ))

        return boundaries

    def _filter_boundaries(
        self,
        boundaries: List[Boundary],
        n: int,
    ) -> List[Boundary]:
        """Filter boundaries by minimum segment length."""
        if not boundaries:
            return []

        # Sort by position
        boundaries = sorted(boundaries, key=lambda b: b.position)

        # Greedy filtering
        filtered = []
        last_pos = 0

        for b in boundaries:
            if b.position - last_pos >= self.min_segment_length:
                if n - b.position >= self.min_segment_length:
                    filtered.append(b)
                    last_pos = b.position

        return filtered

    def _compute_segment_lengths(
        self,
        boundaries: List[Boundary],
        n: int,
    ) -> List[int]:
        """Compute lengths of segments."""
        positions = [0] + [b.position for b in boundaries] + [n]
        lengths = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        return lengths


class MarkovSurpriseBoundaryDetector:
    """
    Boundary detection using information content (surprise) from Markov model.

    Algorithm:
    1. Train n-gram Markov model on corpus
    2. Compute P(note | context) at each position
    3. Information content = -log(P)
    4. Boundaries occur at high surprise (low probability) events
    """

    def __init__(
        self,
        device: str = 'cuda',
        n_gram: int = 3,                     # Markov order
        surprise_threshold: float = 0.7,     # Percentile threshold
        min_segment_length: int = 4,
        verbose: bool = False,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.n_gram = n_gram
        self.surprise_threshold = surprise_threshold
        self.min_segment_length = min_segment_length
        self.verbose = verbose

        # Markov model (will be built during training)
        self.transition_counts = None
        self.total_counts = None

    def train(self, sequences: List[List[int]]):
        """
        Train Markov model on corpus.

        Args:
            sequences: List of pitch class sequences
        """
        if self.verbose:
            print(f"[Markov] Training {self.n_gram}-gram model on {len(sequences)} sequences")

        # Count transitions
        # Context -> next symbol -> count
        self.transition_counts = {}
        self.total_counts = {}

        for seq in sequences:
            for i in range(len(seq) - self.n_gram):
                context = tuple(seq[i:i+self.n_gram])
                next_symbol = seq[i+self.n_gram]

                if context not in self.transition_counts:
                    self.transition_counts[context] = {}
                    self.total_counts[context] = 0

                if next_symbol not in self.transition_counts[context]:
                    self.transition_counts[context][next_symbol] = 0

                self.transition_counts[context][next_symbol] += 1
                self.total_counts[context] += 1

        if self.verbose:
            print(f"[Markov] Learned {len(self.transition_counts)} contexts")

    def detect(self, pitch_sequence: List[int]) -> BoundaryResult:
        """
        Detect boundaries using surprise (information content).
        """
        if self.transition_counts is None:
            raise RuntimeError("Model not trained. Call train() first.")

        n = len(pitch_sequence)
        if n <= self.n_gram:
            return BoundaryResult(
                boundaries=[],
                sequence_length=n,
                n_segments=1,
                avg_segment_length=n,
                boundary_strength_mean=0.0,
                boundary_strength_std=0.0,
            )

        # Compute surprise at each position
        surprises = np.zeros(n)

        for i in range(self.n_gram, n):
            context = tuple(pitch_sequence[i-self.n_gram:i])
            next_symbol = pitch_sequence[i]

            # Compute probability
            prob = self._get_probability(context, next_symbol)

            # Information content (surprise)
            surprises[i] = -np.log2(prob + 1e-10)

        # Normalize
        if surprises.max() > 0:
            surprises = surprises / surprises.max()

        # Find peaks above threshold
        threshold = np.percentile(surprises[self.n_gram:], self.surprise_threshold * 100)
        boundaries = []

        for i in range(self.n_gram + 1, n - 1):
            # Local maximum above threshold
            if (surprises[i] > surprises[i-1] and
                surprises[i] > surprises[i+1] and
                surprises[i] > threshold):
                boundaries.append(Boundary(
                    position=i,
                    strength=float(surprises[i]),
                    boundary_type='surprise',
                ))

        # Filter by minimum segment length
        boundaries = self._filter_boundaries(boundaries, n)

        # Compute statistics
        segment_lengths = self._compute_segment_lengths(boundaries, n)

        return BoundaryResult(
            boundaries=boundaries,
            sequence_length=n,
            n_segments=len(boundaries) + 1,
            avg_segment_length=np.mean(segment_lengths) if segment_lengths else n,
            boundary_strength_mean=np.mean([b.strength for b in boundaries]) if boundaries else 0,
            boundary_strength_std=np.std([b.strength for b in boundaries]) if boundaries else 0,
        )

    def _get_probability(self, context: Tuple[int, ...], next_symbol: int) -> float:
        """Get probability of next symbol given context."""
        if context not in self.transition_counts:
            # Uniform fallback
            return 1.0 / 12.0

        counts = self.transition_counts[context]
        total = self.total_counts[context]

        if next_symbol in counts:
            return counts[next_symbol] / total
        else:
            # Smoothing
            return 0.1 / total

    def _filter_boundaries(self, boundaries: List[Boundary], n: int) -> List[Boundary]:
        """Filter by minimum segment length."""
        if not boundaries:
            return []

        boundaries = sorted(boundaries, key=lambda b: b.position)
        filtered = []
        last_pos = 0

        for b in boundaries:
            if b.position - last_pos >= self.min_segment_length:
                if n - b.position >= self.min_segment_length:
                    filtered.append(b)
                    last_pos = b.position

        return filtered

    def _compute_segment_lengths(self, boundaries: List[Boundary], n: int) -> List[int]:
        """Compute segment lengths."""
        positions = [0] + [b.position for b in boundaries] + [n]
        return [positions[i+1] - positions[i] for i in range(len(positions)-1)]


def detect_boundaries(
    pitch_sequence: List[int],
    duration_sequence: Optional[List[int]] = None,
    method: str = 'ssm',  # 'ssm' or 'markov'
    device: str = 'cuda',
    verbose: bool = False,
    **kwargs,
) -> BoundaryResult:
    """
    Convenience function to detect boundaries.

    Args:
        pitch_sequence: Pitch class sequence
        duration_sequence: Optional duration sequence
        method: 'ssm' (self-similarity) or 'markov' (surprise)
        device: 'cuda' or 'cpu'
        verbose: Print progress
        **kwargs: Additional arguments for detector

    Returns:
        BoundaryResult
    """
    if method == 'ssm':
        detector = SelfSimilarityBoundaryDetector(device=device, verbose=verbose, **kwargs)
        return detector.detect(pitch_sequence, duration_sequence)
    elif method == 'markov':
        detector = MarkovSurpriseBoundaryDetector(device=device, verbose=verbose, **kwargs)
        detector.train([pitch_sequence])  # Self-train on single sequence
        return detector.detect(pitch_sequence)
    else:
        raise ValueError(f"Unknown method: {method}")


def segment_sequences(
    pitch_sequences: List[List[int]],
    duration_sequences: Optional[List[List[int]]] = None,
    method: str = 'ssm',
    device: str = 'cuda',
    verbose: bool = False,
    **kwargs,
) -> List[List[List[int]]]:
    """
    Segment multiple sequences and return segments.

    Returns list of list of segments (each segment is a list of pitch classes).
    """
    all_segments = []

    if method == 'markov':
        # Train on full corpus first
        detector = MarkovSurpriseBoundaryDetector(device=device, verbose=verbose, **kwargs)
        detector.train(pitch_sequences)

        for i, seq in enumerate(pitch_sequences):
            result = detector.detect(seq)
            segments = []
            for start, end in result.get_segment_indices():
                segments.append(seq[start:end])
            all_segments.append(segments)
    else:
        detector = SelfSimilarityBoundaryDetector(device=device, verbose=verbose, **kwargs)
        for i, seq in enumerate(pitch_sequences):
            dur = duration_sequences[i] if duration_sequences else None
            result = detector.detect(seq, dur)
            segments = []
            for start, end in result.get_segment_indices():
                segments.append(seq[start:end])
            all_segments.append(segments)

    return all_segments


if __name__ == '__main__':
    print("Testing Boundary Detection...")

    # Test sequence with clear structure
    # Pattern A repeated, then Pattern B repeated
    pattern_a = [0, 2, 4, 5, 7]  # C major ascending
    pattern_b = [7, 5, 4, 2, 0]  # C major descending

    test_seq = pattern_a * 3 + pattern_b * 3 + pattern_a * 2

    print(f"Test sequence length: {len(test_seq)}")
    print(f"Expected boundaries around positions 15 and 30")

    # Test SSM method
    print("\n--- Self-Similarity Method ---")
    result = detect_boundaries(test_seq, method='ssm', device='cuda', verbose=True)
    print(f"Boundaries: {[b.position for b in result.boundaries]}")
    print(f"Segments: {result.get_segment_indices()}")

    # Test Markov method
    print("\n--- Markov Surprise Method ---")
    result = detect_boundaries(test_seq, method='markov', device='cpu', verbose=True)
    print(f"Boundaries: {[b.position for b in result.boundaries]}")
    print(f"Segments: {result.get_segment_indices()}")
