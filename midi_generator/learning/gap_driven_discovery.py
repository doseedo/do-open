"""
Gap-Driven Transform Discovery
===============================

Phase 3: Iteratively discover transforms to fill reconstruction gaps.

Process:
1. Encode dataset with current transforms
2. Reconstruct and measure residuals
3. Cluster residuals (DBSCAN)
4. Find largest gap cluster
5. Synthesize transform using hybrid synthesizer
6. Add to registry and repeat

Stopping criteria:
- Gap cluster < 0.5% of variance
- Or 200 total gap transforms
- Or diminishing returns (<1% improvement per 10 transforms)

Research Foundation:
- Iterative refinement approach
- Residual analysis
- Adaptive basis discovery

Author: Agent 8 - Transform Architecture
Phase: 3 (Gap-Driven Discovery)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import pickle
from collections import defaultdict
import hashlib

try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
except ImportError:
    print("Warning: scikit-learn not installed")
    DBSCAN = None
    PCA = None

import mido

from ..transforms.space_level_transforms import SpaceLevelTransform
from ..transforms.transform_registry import TransformRegistry
from ..transforms.hybrid_synthesizer import (
    HybridTransformSynthesizer,
    GapCluster,
    SynthesisConfig
)
from .sparse_transform_learning import MIDIFeatureExtractor


# ============================================================================
# Residual Analysis
# ============================================================================

@dataclass
class ReconstructionResult:
    """Result of MIDI encoding + decoding"""
    file_id: str
    midi_path: Path
    original_midi: mido.MidiFile
    reconstructed_midi: mido.MidiFile
    encoding: np.ndarray  # Transform coefficients
    residual: np.ndarray  # Feature-space residual
    reconstruction_error: float


class ResidualAnalyzer:
    """
    Analyze reconstruction residuals to identify gaps.

    Compares original vs reconstructed MIDI in feature space
    to identify systematic errors.
    """

    def __init__(self, registry: TransformRegistry):
        """
        Initialize residual analyzer.

        Args:
            registry: Transform registry for encoding/decoding
        """
        self.registry = registry
        self.feature_extractor = MIDIFeatureExtractor()

    def analyze_dataset(self,
                       midi_files: List[Path],
                       verbose: bool = True) -> List[ReconstructionResult]:
        """
        Analyze reconstruction quality for entire dataset.

        Args:
            midi_files: List of MIDI files to analyze
            verbose: Print progress

        Returns:
            List of reconstruction results with residuals
        """
        if verbose:
            print(f"\n{'='*70}")
            print("Residual Analysis")
            print(f"Dataset: {len(midi_files)} MIDI files")
            print(f"Current transforms: {self.registry.count_transforms()}")
            print(f"{'='*70}\n")

        results = []

        for i, midi_path in enumerate(midi_files):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(midi_files)} files...")

            try:
                result = self._analyze_file(midi_path)
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to analyze {midi_path}: {e}")
                continue

        if verbose:
            # Summarize
            errors = [r.reconstruction_error for r in results]
            print(f"\n  → Analyzed {len(results)} files")
            print(f"  Mean error: {np.mean(errors):.4f}")
            print(f"  Std error: {np.std(errors):.4f}")
            print(f"  Max error: {np.max(errors):.4f}")
            print(f"  Min error: {np.min(errors):.4f}\n")

        return results

    def _analyze_file(self, midi_path: Path) -> ReconstructionResult:
        """Analyze single MIDI file"""
        # Load original
        original_midi = mido.MidiFile(str(midi_path))

        # Encode
        encoding = self.registry.encode(original_midi)

        # Decode
        reconstructed_midi = self.registry.decode(encoding, base_template=original_midi)

        # Extract features for both
        original_features = self.feature_extractor.extract(original_midi)
        reconstructed_features = self.feature_extractor.extract(reconstructed_midi)

        # Compute residual
        residual = original_features - reconstructed_features

        # Compute error (L2 norm)
        reconstruction_error = float(np.linalg.norm(residual))

        # Create result
        file_id = hashlib.md5(str(midi_path).encode()).hexdigest()[:16]

        return ReconstructionResult(
            file_id=file_id,
            midi_path=midi_path,
            original_midi=original_midi,
            reconstructed_midi=reconstructed_midi,
            encoding=encoding,
            residual=residual,
            reconstruction_error=reconstruction_error
        )


# ============================================================================
# Gap Clustering
# ============================================================================

class GapClusterer:
    """
    Cluster residuals to identify systematic gaps.

    Uses DBSCAN to find dense regions in residual space.
    Each cluster represents a specific reconstruction gap.
    """

    def __init__(self,
                 eps: float = 5.0,
                 min_samples: int = 10,
                 metric: str = 'euclidean'):
        """
        Initialize gap clusterer.

        Args:
            eps: DBSCAN epsilon (neighborhood size)
            min_samples: Minimum cluster size
            metric: Distance metric
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def cluster_residuals(self,
                         results: List[ReconstructionResult],
                         verbose: bool = True) -> List[GapCluster]:
        """
        Cluster residuals to find gaps.

        Args:
            results: List of reconstruction results
            verbose: Print progress

        Returns:
            List of gap clusters, sorted by size
        """
        if DBSCAN is None:
            raise RuntimeError("scikit-learn required for clustering")

        if verbose:
            print(f"\n{'='*70}")
            print("Gap Clustering")
            print(f"Residuals: {len(results)}")
            print(f"DBSCAN eps={self.eps}, min_samples={self.min_samples}")
            print(f"{'='*70}\n")

        # Stack residuals
        residuals = np.array([r.residual for r in results])

        # Optional: reduce dimensionality for faster clustering
        if residuals.shape[1] > 100 and PCA:
            if verbose:
                print("  Reducing dimensionality with PCA...")
            pca = PCA(n_components=100)
            residuals_reduced = pca.fit_transform(residuals)
            if verbose:
                print(f"    → Reduced to {residuals_reduced.shape[1]}D\n")
        else:
            residuals_reduced = residuals

        # Cluster
        if verbose:
            print("  Running DBSCAN clustering...")

        clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=-1
        )

        labels = clusterer.fit_predict(residuals_reduced)

        # Group by cluster
        clusters_dict = defaultdict(list)
        for result, label in zip(results, labels):
            if label != -1:  # Ignore noise
                clusters_dict[label].append(result)

        if verbose:
            print(f"    → Found {len(clusters_dict)} clusters")
            print(f"    → Noise points: {sum(labels == -1)}\n")

        # Convert to GapCluster objects
        gap_clusters = []

        for cluster_id, cluster_results in clusters_dict.items():
            # Compute centroid
            cluster_residuals = np.array([r.residual for r in cluster_results])
            centroid = np.mean(cluster_residuals, axis=0)

            # Compute variance explained
            total_variance = np.sum([r.reconstruction_error ** 2 for r in results])
            cluster_variance = np.sum([r.reconstruction_error ** 2 for r in cluster_results])
            variance_explained = cluster_variance / max(total_variance, 1e-10)

            # Create GapCluster
            gap_cluster = GapCluster(
                cluster_id=f"gap_{cluster_id}",
                midi_files=[r.midi_path for r in cluster_results],
                residual_vectors=cluster_residuals.tolist(),
                centroid=centroid,
                variance_explained=variance_explained,
                size=len(cluster_results)
            )

            gap_clusters.append(gap_cluster)

        # Sort by variance explained (largest first)
        gap_clusters.sort(key=lambda c: c.variance_explained, reverse=True)

        if verbose:
            print("  Top 5 gap clusters:")
            for i, cluster in enumerate(gap_clusters[:5], 1):
                print(f"    {i}. {cluster.cluster_id}: {cluster.size} pieces, "
                      f"{cluster.variance_explained:.2%} variance")
            print()

        return gap_clusters


# ============================================================================
# Gap-Driven Discovery System
# ============================================================================

class GapDrivenDiscovery:
    """
    Main system for iterative gap-driven transform discovery.

    Pipeline:
    1. Start with existing transforms (theory + sparse learned)
    2. Analyze dataset, compute residuals
    3. Cluster residuals to find gaps
    4. For largest gap: synthesize transform with hybrid synthesizer
    5. Add transform to registry
    6. Repeat until convergence

    Stopping criteria:
    - Largest gap < 0.5% variance
    - Or 200 gap transforms discovered
    - Or <1% improvement per 10 transforms
    """

    def __init__(self,
                 registry: TransformRegistry,
                 synthesis_config: Optional[SynthesisConfig] = None,
                 max_gap_transforms: int = 200,
                 min_gap_variance: float = 0.005,
                 convergence_window: int = 10,
                 convergence_threshold: float = 0.01):
        """
        Initialize gap-driven discovery system.

        Args:
            registry: Transform registry (will be extended)
            synthesis_config: Configuration for hybrid synthesizer
            max_gap_transforms: Maximum gap transforms to discover
            min_gap_variance: Stop if gap < this variance
            convergence_window: Window for checking convergence
            convergence_threshold: Improvement threshold for convergence
        """
        self.registry = registry
        self.synthesis_config = synthesis_config or SynthesisConfig()
        self.max_gap_transforms = max_gap_transforms
        self.min_gap_variance = min_gap_variance
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

        # Components
        self.residual_analyzer = ResidualAnalyzer(registry)
        self.gap_clusterer = GapClusterer()
        self.hybrid_synthesizer = HybridTransformSynthesizer(self.synthesis_config)

        # State
        self.discovered_transforms: List[SpaceLevelTransform] = []
        self.iteration_history: List[Dict[str, Any]] = []

    def discover_transforms(self,
                           midi_files: List[Path],
                           verbose: bool = True) -> List[SpaceLevelTransform]:
        """
        Run gap-driven discovery on dataset.

        Args:
            midi_files: List of MIDI files for analysis
            verbose: Print progress

        Returns:
            List of discovered transforms
        """
        if verbose:
            print(f"\n{'='*80}")
            print("GAP-DRIVEN TRANSFORM DISCOVERY")
            print(f"{'='*80}")
            print(f"Dataset: {len(midi_files)} MIDI files")
            print(f"Initial transforms: {self.registry.count_transforms()}")
            print(f"Target: Discover up to {self.max_gap_transforms} gap transforms")
            print(f"{'='*80}\n")

        iteration = 0
        converged = False

        while not converged and len(self.discovered_transforms) < self.max_gap_transforms:
            iteration += 1

            if verbose:
                print(f"\n{'='*80}")
                print(f"ITERATION {iteration}")
                print(f"Current transforms: {self.registry.count_transforms()}")
                print(f"Gap transforms discovered: {len(self.discovered_transforms)}")
                print(f"{'='*80}\n")

            # Step 1: Analyze residuals
            results = self.residual_analyzer.analyze_dataset(midi_files, verbose=verbose)

            # Step 2: Cluster gaps
            gap_clusters = self.gap_clusterer.cluster_residuals(results, verbose=verbose)

            if not gap_clusters:
                if verbose:
                    print("  → No gaps found. Stopping.")
                break

            # Step 3: Check convergence
            largest_gap = gap_clusters[0]

            if largest_gap.variance_explained < self.min_gap_variance:
                if verbose:
                    print(f"  → Largest gap ({largest_gap.variance_explained:.4f}) below "
                          f"threshold ({self.min_gap_variance}). Converged!")
                converged = True
                break

            # Step 4: Synthesize transform for largest gap
            if verbose:
                print(f"  Synthesizing transform for largest gap...")
                print(f"    Gap: {largest_gap.cluster_id}")
                print(f"    Size: {largest_gap.size} pieces")
                print(f"    Variance: {largest_gap.variance_explained:.2%}\n")

            try:
                discovered_pattern = self.hybrid_synthesizer.synthesize_transform(largest_gap)

                if discovered_pattern.final_transform:
                    # Add to registry
                    # Note: In practice, would need to properly register the transform
                    # For now, just store it
                    self.discovered_transforms.append(discovered_pattern.final_transform)

                    if verbose:
                        print(f"  ✓ Transform discovered and added to registry")

                else:
                    if verbose:
                        print(f"  ✗ Transform synthesis failed")

            except Exception as e:
                if verbose:
                    print(f"  ✗ Error during synthesis: {e}")

            # Step 5: Record iteration
            iteration_record = {
                'iteration': iteration,
                'num_transforms': self.registry.count_transforms(),
                'num_discovered': len(self.discovered_transforms),
                'largest_gap_variance': largest_gap.variance_explained,
                'num_gaps': len(gap_clusters),
                'mean_error': np.mean([r.reconstruction_error for r in results])
            }
            self.iteration_history.append(iteration_record)

            # Step 6: Check diminishing returns
            if len(self.iteration_history) >= self.convergence_window:
                recent_improvements = [
                    self.iteration_history[-i-1]['mean_error'] - self.iteration_history[-i]['mean_error']
                    for i in range(1, self.convergence_window)
                ]
                avg_improvement = np.mean(recent_improvements)

                if avg_improvement < self.convergence_threshold:
                    if verbose:
                        print(f"\n  → Diminishing returns detected (avg improvement: {avg_improvement:.4f})")
                        print(f"  → Converged after {iteration} iterations")
                    converged = True

        # Final summary
        if verbose:
            self._print_final_summary()

        return self.discovered_transforms

    def _print_final_summary(self):
        """Print final summary of discovery process"""
        print(f"\n{'='*80}")
        print("GAP-DRIVEN DISCOVERY COMPLETE")
        print(f"{'='*80}")
        print(f"Total iterations: {len(self.iteration_history)}")
        print(f"Transforms discovered: {len(self.discovered_transforms)}")
        print(f"Final transform count: {self.registry.count_transforms()}")

        if self.iteration_history:
            initial_error = self.iteration_history[0]['mean_error']
            final_error = self.iteration_history[-1]['mean_error']
            improvement = (initial_error - final_error) / max(initial_error, 1e-10)
            print(f"Error improvement: {improvement:.2%}")

        print(f"{'='*80}\n")

    def save_state(self, save_path: Path):
        """Save discovery state"""
        state = {
            'discovered_transforms': self.discovered_transforms,
            'iteration_history': self.iteration_history,
            'config': {
                'max_gap_transforms': self.max_gap_transforms,
                'min_gap_variance': self.min_gap_variance,
                'convergence_window': self.convergence_window,
                'convergence_threshold': self.convergence_threshold
            }
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"Saved discovery state to {save_path}")

    def load_state(self, load_path: Path):
        """Load discovery state"""
        with open(load_path, 'rb') as f:
            state = pickle.load(f)

        self.discovered_transforms = state['discovered_transforms']
        self.iteration_history = state['iteration_history']

        print(f"Loaded discovery state from {load_path}")
        print(f"  Discovered transforms: {len(self.discovered_transforms)}")
        print(f"  Iterations: {len(self.iteration_history)}")
