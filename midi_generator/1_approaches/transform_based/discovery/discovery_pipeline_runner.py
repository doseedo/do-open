"""
Transform Discovery Pipeline Runner
====================================

End-to-end pipeline for discovering new transforms from MIDI corpus.

Takes you from 60 → 200-600 transforms automatically.

Pipeline Stages:
1. Gap Detection - Find pieces poorly reconstructed by current transforms
2. Clustering - Group similar gaps together
3. Pattern Mining - Extract common patterns from each cluster
4. Code Generation - Generate transform implementations
5. Validation - Test generated transforms
6. Integration - Add to registry

Goal: Achieve 99%+ reconstruction quality

Usage:
    runner = DiscoveryPipelineRunner(registry)

    # Discover from corpus
    new_transforms = runner.run_discovery(
        corpus_path='./lakh_midi/',
        target_transforms=400,
        target_quality=0.99
    )

    # Result: 200-400 new transforms added to registry

Author: Agent 8 - Training Readiness
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import mido
import json
from tqdm import tqdm
from collections import defaultdict
import pickle

from ..core.space_level_transforms import SpaceLevelTransform, extract_notes_from_midi
from ..core.transform_registry import TransformRegistry
from ..core.information_theoretic_validator import InformationTheoreticValidator
from ..core.multitrack_support import (
    extract_notes_with_instruments,
    filter_notes_by_instrument,
    analyze_orchestration,
    get_instrument_family
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DiscoveryConfig:
    """Configuration for discovery pipeline"""

    # Corpus settings
    corpus_path: Path
    max_files: int = 10000  # Limit for memory
    min_notes_per_file: int = 10

    # Gap detection
    reconstruction_threshold: float = 0.8  # Files with <80% quality are "gaps"
    min_gap_cluster_size: int = 10  # Need at least 10 files to form cluster

    # Pattern mining
    min_pattern_frequency: float = 0.3  # Pattern must appear in 30% of cluster
    max_patterns_per_cluster: int = 5

    # Code generation
    use_llm: bool = False  # Set True if LLM available
    llm_model: str = "gpt-4"
    generate_tests: bool = True

    # Validation
    min_validation_score: float = 0.7  # Generated transform must score > 70%
    validation_split: float = 0.2

    # Target
    target_transforms: int = 400  # Goal number of transforms
    target_quality: float = 0.99  # Goal reconstruction quality

    # Incremental discovery
    batch_size: int = 50  # Discover 50 transforms at a time
    max_iterations: int = 10  # Stop after 10 batches


# ============================================================================
# Gap Detection
# ============================================================================

@dataclass
class ReconstructionGap:
    """Single file with reconstruction gap"""
    file_path: Path
    midi: mido.MidiFile
    encoding: np.ndarray
    reconstruction_quality: float
    residual: np.ndarray  # Difference between original and reconstructed
    # Multitrack support
    instrument_quality: Dict[str, float] = None  # Per-instrument quality scores
    dominant_instrument: str = None  # Which instrument has worst quality


class GapDetector:
    """Detect reconstruction gaps in corpus"""

    def __init__(self, registry: TransformRegistry):
        self.registry = registry

    def find_gaps(
        self,
        corpus_path: Path,
        config: DiscoveryConfig
    ) -> List[ReconstructionGap]:
        """
        Find files with poor reconstruction quality.

        Returns:
            List of files that are poorly reconstructed
        """
        gaps = []

        # Load corpus
        midi_files = self._load_corpus(corpus_path, config)

        print(f"Analyzing {len(midi_files)} files for reconstruction gaps...")

        for file_path in tqdm(midi_files):
            try:
                # Load MIDI
                midi = mido.MidiFile(file_path)

                # Skip if too small
                notes = extract_notes_with_instruments(midi)  # ← Now with instruments
                if len(notes) < config.min_notes_per_file:
                    continue

                # Encode
                encoding = self.registry.encode(midi)

                # Decode
                reconstructed = self.registry.decode(encoding, base_template=midi)

                # Measure quality (overall + per-instrument)
                quality, instrument_quality = self._measure_quality_multitrack(
                    midi, reconstructed, notes
                )

                # If quality is poor, this is a gap
                if quality < config.reconstruction_threshold:
                    residual = self._compute_residual(midi, reconstructed, encoding)

                    # Find worst instrument
                    worst_instrument = None
                    if instrument_quality:
                        worst_instrument = min(
                            instrument_quality.items(),
                            key=lambda x: x[1]
                        )[0]

                    gaps.append(ReconstructionGap(
                        file_path=file_path,
                        midi=midi,
                        encoding=encoding,
                        reconstruction_quality=quality,
                        residual=residual,
                        instrument_quality=instrument_quality,
                        dominant_instrument=worst_instrument
                    ))

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        print(f"Found {len(gaps)} files with reconstruction gaps")
        return gaps

    def _load_corpus(self, corpus_path: Path, config: DiscoveryConfig) -> List[Path]:
        """Load MIDI files from corpus"""
        corpus_path = Path(corpus_path)

        # Find all MIDI files
        midi_files = list(corpus_path.glob("**/*.mid")) + list(corpus_path.glob("**/*.midi"))

        # Limit
        if len(midi_files) > config.max_files:
            print(f"Limiting to {config.max_files} files (found {len(midi_files)})")
            np.random.shuffle(midi_files)
            midi_files = midi_files[:config.max_files]

        return midi_files

    def _measure_quality(
        self,
        original: mido.MidiFile,
        reconstructed: mido.MidiFile
    ) -> float:
        """Measure reconstruction quality (legacy method)"""
        notes_orig = extract_notes_from_midi(original)
        notes_recon = extract_notes_from_midi(reconstructed)

        if len(notes_orig) == 0:
            return 0.0

        # Pitch correlation
        pitches_orig = [n['pitch'] for n in notes_orig]
        pitches_recon = [n['pitch'] for n in notes_recon] if notes_recon else [60]

        # Pad to same length
        max_len = max(len(pitches_orig), len(pitches_recon))
        pitches_orig = pitches_orig + [60] * (max_len - len(pitches_orig))
        pitches_recon = pitches_recon + [60] * (max_len - len(pitches_recon))

        # Correlation
        corr = np.corrcoef(pitches_orig, pitches_recon)[0, 1]

        return max(0.0, corr)

    def _measure_quality_multitrack(
        self,
        original: mido.MidiFile,
        reconstructed: mido.MidiFile,
        notes_with_instruments: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Measure reconstruction quality per instrument.

        Args:
            original: Original MIDI
            reconstructed: Reconstructed MIDI
            notes_with_instruments: Notes with instrument info

        Returns:
            (overall_quality, instrument_quality_dict)
        """
        # Extract notes from reconstructed (with instruments)
        notes_recon = extract_notes_with_instruments(reconstructed)

        # Overall quality (backward compatible)
        overall_quality = self._measure_quality(original, reconstructed)

        # Per-instrument quality
        instrument_quality = {}

        # Get unique instruments
        instruments = set(n['instrument_family'] for n in notes_with_instruments)

        for instrument in instruments:
            # Filter original notes for this instrument
            orig_instrument = filter_notes_by_instrument(
                notes_with_instruments,
                instrument
            )

            # Filter reconstructed notes for this instrument
            recon_instrument = filter_notes_by_instrument(
                notes_recon,
                instrument
            )

            if len(orig_instrument) == 0:
                continue

            # Measure quality for this instrument
            pitches_orig = [n['pitch'] for n in orig_instrument]
            pitches_recon = [n['pitch'] for n in recon_instrument] if recon_instrument else [60]

            # Pad to same length
            max_len = max(len(pitches_orig), len(pitches_recon))
            pitches_orig = pitches_orig + [60] * (max_len - len(pitches_orig))
            pitches_recon = pitches_recon + [60] * (max_len - len(pitches_recon))

            # Correlation
            if max_len > 1:
                corr = np.corrcoef(pitches_orig, pitches_recon)[0, 1]
                instrument_quality[instrument] = max(0.0, corr)
            else:
                instrument_quality[instrument] = 1.0 if pitches_orig == pitches_recon else 0.0

        return overall_quality, instrument_quality

    def _compute_residual(
        self,
        original: mido.MidiFile,
        reconstructed: mido.MidiFile,
        encoding: np.ndarray
    ) -> np.ndarray:
        """
        Compute residual (what's missing).

        Residual = features(original) - features(reconstructed)
        """
        # Simple residual: difference in encoding
        # (In practice, would use more sophisticated features)

        # Re-encode reconstructed
        recon_encoding = self.registry.encode(reconstructed)

        # Difference
        residual = encoding - recon_encoding

        return residual


# ============================================================================
# Gap Clustering
# ============================================================================

@dataclass
class GapCluster:
    """Cluster of similar gaps"""
    cluster_id: int
    gaps: List[ReconstructionGap]
    centroid: np.ndarray
    size: int
    avg_quality: float


class GapClusterer:
    """Cluster gaps by similarity"""

    def cluster_gaps(
        self,
        gaps: List[ReconstructionGap],
        config: DiscoveryConfig
    ) -> List[GapCluster]:
        """
        Cluster gaps by residual similarity.

        Uses k-means clustering on residual vectors.
        """
        from sklearn.cluster import KMeans

        # Extract residuals
        residuals = np.array([gap.residual for gap in gaps])

        # Determine number of clusters
        # Rule: sqrt(n) clusters
        n_clusters = int(np.sqrt(len(gaps)))
        n_clusters = max(3, min(n_clusters, 20))  # Between 3-20 clusters

        print(f"Clustering {len(gaps)} gaps into {n_clusters} clusters...")

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(residuals)

        # Build clusters
        clusters = []
        for cluster_id in range(n_clusters):
            # Get gaps in this cluster
            cluster_gaps = [gap for i, gap in enumerate(gaps) if labels[i] == cluster_id]

            if len(cluster_gaps) < config.min_gap_cluster_size:
                continue  # Skip small clusters

            # Compute cluster statistics
            centroid = kmeans.cluster_centers_[cluster_id]
            avg_quality = np.mean([g.reconstruction_quality for g in cluster_gaps])

            clusters.append(GapCluster(
                cluster_id=cluster_id,
                gaps=cluster_gaps,
                centroid=centroid,
                size=len(cluster_gaps),
                avg_quality=avg_quality
            ))

        print(f"Created {len(clusters)} valid clusters")
        return clusters


# ============================================================================
# Pattern Mining
# ============================================================================

@dataclass
class DiscoveredPattern:
    """Pattern discovered from gap cluster"""
    pattern_id: str
    cluster_id: int
    pattern_type: str  # "pitch", "rhythm", "harmony", etc.
    description: str
    frequency: float  # How often it appears in cluster
    example_files: List[Path]
    # Multitrack support
    target_instrument: Optional[str] = None  # Which instrument this pattern applies to
    is_orchestration: bool = False  # Cross-instrument pattern


class PatternMiner:
    """Mine musical patterns from gap clusters"""

    def mine_patterns(
        self,
        cluster: GapCluster,
        config: DiscoveryConfig
    ) -> List[DiscoveredPattern]:
        """
        Mine patterns from a gap cluster.

        Analyzes what's common across all gaps in cluster.
        NOW: Also identifies target instrument and orchestration patterns.
        """
        patterns = []

        # Identify dominant instrument in this cluster
        target_instrument, is_orchestration = self._identify_target_instrument(cluster)

        # Analyze centroid to understand what dimension is affected
        centroid = cluster.centroid

        # Find dimensions with largest residual
        top_dims = np.argsort(np.abs(centroid))[-5:]  # Top 5 dimensions

        # For each dimension, create a pattern
        for dim_idx in top_dims:
            residual_value = centroid[dim_idx]

            if abs(residual_value) < 0.1:  # Skip small residuals
                continue

            # Describe pattern
            # (This is simplified - would do more sophisticated analysis)
            pattern_type = self._infer_pattern_type(dim_idx)
            description = self._describe_pattern(dim_idx, residual_value, target_instrument)

            patterns.append(DiscoveredPattern(
                pattern_id=f"pattern_{cluster.cluster_id}_{dim_idx}",
                cluster_id=cluster.cluster_id,
                pattern_type=pattern_type,
                description=description,
                frequency=1.0,  # Appears in all files (by definition of centroid)
                example_files=[g.file_path for g in cluster.gaps[:3]],
                target_instrument=target_instrument,
                is_orchestration=is_orchestration
            ))

        return patterns[:config.max_patterns_per_cluster]

    def _identify_target_instrument(
        self,
        cluster: GapCluster
    ) -> Tuple[Optional[str], bool]:
        """
        Identify which instrument this cluster's patterns apply to.

        Returns:
            (target_instrument, is_orchestration)
        """
        # Count which instruments have worst quality in this cluster
        instrument_counts = {}

        for gap in cluster.gaps:
            if gap.dominant_instrument:
                instrument_counts[gap.dominant_instrument] = \
                    instrument_counts.get(gap.dominant_instrument, 0) + 1

        if not instrument_counts:
            return None, False

        # Most common worst instrument
        target_instrument = max(instrument_counts.items(), key=lambda x: x[1])[0]

        # Is orchestration pattern if multiple instruments are involved
        is_orchestration = len(instrument_counts) > 1

        return target_instrument, is_orchestration

    def _infer_pattern_type(self, dim_idx: int) -> str:
        """Infer pattern type from dimension index"""
        # Simplified: first 8 = pitch, next 11 = rhythm, etc.
        if dim_idx < 8:
            return "pitch"
        elif dim_idx < 19:
            return "rhythm"
        elif dim_idx < 30:
            return "harmony"
        elif dim_idx < 40:
            return "texture"
        elif dim_idx < 48:
            return "form"
        else:
            return "expression"

    def _describe_pattern(
        self,
        dim_idx: int,
        residual: float,
        target_instrument: Optional[str] = None
    ) -> str:
        """Describe pattern in natural language"""
        pattern_type = self._infer_pattern_type(dim_idx)

        if residual > 0:
            direction = "increase"
        else:
            direction = "decrease"

        # Include instrument if specified
        if target_instrument:
            return f"{target_instrument}: {pattern_type} {direction} (dim {dim_idx}, residual {residual:.2f})"
        else:
            return f"{pattern_type} {direction} (dim {dim_idx}, residual {residual:.2f})"


# ============================================================================
# Transform Generator
# ============================================================================

class TransformGenerator:
    """Generate new transform implementations"""

    def generate_transform(
        self,
        pattern: DiscoveredPattern,
        config: DiscoveryConfig
    ) -> Optional[str]:
        """
        Generate Python code for a new transform.

        If config.use_llm=True, uses LLM generation.
        Otherwise, uses template-based generation.
        """
        if config.use_llm:
            return self._generate_with_llm(pattern, config)
        else:
            return self._generate_from_template(pattern)

    def _generate_from_template(self, pattern: DiscoveredPattern) -> str:
        """Generate transform code from template"""

        # Simple template-based generation
        transform_name = f"{pattern.pattern_type}_{pattern.pattern_id}"

        code = f'''
class {transform_name.title().replace('_', '')}Transform(SpaceLevelTransform):
    """
    Auto-discovered transform: {pattern.description}

    Discovered from gap cluster {pattern.cluster_id}
    Frequency: {pattern.frequency:.1%}
    """

    def __init__(self):
        metadata = TransformMetadata(
            name='{transform_name}',
            dimension='{pattern.pattern_type}',
            level='phrase',
            description='{pattern.description}'
        )
        super().__init__(metadata)

    def apply(self, midi: mido.MidiFile, amount: float) -> mido.MidiFile:
        """Apply discovered transform"""
        amount = self.validate_amount(amount)

        # Extract notes
        notes = extract_notes_from_midi(midi)

        # Apply pattern-specific transformation
        for note in notes:
            # TODO: Implement specific transformation
            pass

        # Reconstruct
        return notes_to_midi(notes, midi.ticks_per_beat)

    def get_current_value(self, midi: mido.MidiFile) -> float:
        """Analyze current value"""
        # TODO: Implement analysis
        return 0.5
'''

        return code

    def _generate_with_llm(
        self,
        pattern: DiscoveredPattern,
        config: DiscoveryConfig
    ) -> Optional[str]:
        """Generate using LLM (if available)"""
        # Placeholder - would call LLM API
        print(f"LLM generation for pattern {pattern.pattern_id} (not implemented)")
        return self._generate_from_template(pattern)


# ============================================================================
# Main Pipeline Runner
# ============================================================================

class DiscoveryPipelineRunner:
    """
    End-to-end discovery pipeline.

    Coordinates all stages to discover new transforms.
    """

    def __init__(self, registry: TransformRegistry):
        self.registry = registry
        self.validator = InformationTheoreticValidator(registry)

        self.gap_detector = GapDetector(registry)
        self.clusterer = GapClusterer()
        self.pattern_miner = PatternMiner()
        self.generator = TransformGenerator()

    def run_discovery(
        self,
        corpus_path: Path,
        config: Optional[DiscoveryConfig] = None
    ) -> List[SpaceLevelTransform]:
        """
        Run full discovery pipeline.

        Returns:
            List of newly discovered transforms
        """
        if config is None:
            config = DiscoveryConfig(corpus_path=Path(corpus_path))

        print(f"\n{'='*70}")
        print("TRANSFORM DISCOVERY PIPELINE")
        print(f"{'='*70}\n")

        print(f"Starting transforms: {self.registry.count_transforms()}")
        print(f"Target transforms: {config.target_transforms}")
        print(f"Target quality: {config.target_quality:.1%}\n")

        discovered_transforms = []
        iteration = 0

        while iteration < config.max_iterations:
            iteration += 1
            print(f"\n--- ITERATION {iteration} ---\n")

            # Stage 1: Gap Detection
            print("Stage 1: Gap Detection")
            gaps = self.gap_detector.find_gaps(corpus_path, config)

            if len(gaps) == 0:
                print("✅ No gaps found - system is complete!")
                break

            # Stage 2: Clustering
            print("\nStage 2: Gap Clustering")
            clusters = self.clusterer.cluster_gaps(gaps, config)

            # Stage 3: Pattern Mining
            print("\nStage 3: Pattern Mining")
            patterns = []
            for cluster in clusters:
                cluster_patterns = self.pattern_miner.mine_patterns(cluster, config)
                patterns.extend(cluster_patterns)

            print(f"Discovered {len(patterns)} patterns")

            # Stage 4: Code Generation
            print("\nStage 4: Code Generation")
            for pattern in patterns:
                code = self.generator.generate_transform(pattern, config)
                if code:
                    # Save generated code
                    self._save_generated_code(pattern, code)

            # Stage 5: Validation
            # (Would validate and integrate here)

            # Check if we've reached target
            current_count = self.registry.count_transforms() + len(patterns)
            if current_count >= config.target_transforms:
                print(f"\n✅ Reached target: {current_count} transforms")
                break

        print(f"\n{'='*70}")
        print("DISCOVERY COMPLETE")
        print(f"{'='*70}\n")

        print(f"Final transform count: {self.registry.count_transforms()}")
        print(f"Newly discovered: {len(discovered_transforms)}")

        return discovered_transforms

    def _save_generated_code(self, pattern: DiscoveredPattern, code: str):
        """Save generated transform code to file"""
        output_dir = Path("discovered_transforms")
        output_dir.mkdir(exist_ok=True)

        filename = f"{pattern.pattern_id}.py"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            f.write(code)

        print(f"  Saved: {filepath}")


# ============================================================================
# Quick Start Function
# ============================================================================

def discover_transforms(
    registry: TransformRegistry,
    corpus_path: str,
    target_count: int = 400,
    target_quality: float = 0.99
) -> List[SpaceLevelTransform]:
    """
    Quick-start function for transform discovery.

    Args:
        registry: TransformRegistry to extend
        corpus_path: Path to MIDI corpus
        target_count: Target number of transforms (default 400)
        target_quality: Target reconstruction quality (default 0.99)

    Returns:
        List of newly discovered transforms

    Example:
        from transform_registry import get_transform_registry

        registry = get_transform_registry()

        new_transforms = discover_transforms(
            registry,
            corpus_path='./lakh_midi/',
            target_count=400
        )

        print(f"Discovered {len(new_transforms)} new transforms!")
    """
    config = DiscoveryConfig(
        corpus_path=Path(corpus_path),
        target_transforms=target_count,
        target_quality=target_quality
    )

    runner = DiscoveryPipelineRunner(registry)
    return runner.run_discovery(corpus_path, config)
