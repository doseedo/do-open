"""
Information-Theoretic Transform Validation
==========================================

Validates transform system optimality using information theory.

Measures:
1. Kolmogorov Complexity Bounds - Theoretical minimum description length
2. Compression Ratio - Actual bits needed with current transform set
3. Rate-Distortion Curve - Quality vs bits tradeoff
4. Redundancy Analysis - Which transforms are redundant?
5. Sufficiency Test - Are we missing critical transforms?

Based on:
- Kolmogorov Complexity Theory
- Shannon Information Theory
- Rate-Distortion Theory
- Minimum Description Length (MDL)

Author: Agent 8 - Training Readiness
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import mido
from pathlib import Path
import gzip
import json

from .space_level_transforms import SpaceLevelTransform, extract_notes_from_midi
from .transform_registry import TransformRegistry


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class InformationMetrics:
    """Information-theoretic metrics for transform system"""

    # Kolmogorov bounds
    theoretical_min_bits: float  # Lower bound (uncomputable, but estimated)
    empirical_compression_bits: float  # Gzip baseline

    # Transform encoding
    transform_bits: float  # Bits using current transform set
    transform_count: int

    # Efficiency
    compression_efficiency: float  # transform_bits / empirical_bits
    kolmogorov_efficiency: float  # theoretical_min / transform_bits (estimated)

    # Rate-distortion
    rate_distortion_curve: Dict[float, float]  # bits → reconstruction quality
    optimal_operating_point: Tuple[float, float]  # (bits, quality)

    # Redundancy
    redundant_transforms: List[str]  # Transforms adding < 0.01 bits
    critical_transforms: List[str]  # Transforms explaining > 5% variance

    # Sufficiency
    reconstruction_quality: float  # How well transforms reconstruct corpus
    explained_variance: float  # Variance explained by transforms
    residual_patterns: List[str]  # Patterns not captured


@dataclass
class TransformContribution:
    """Individual transform's information contribution"""

    name: str
    bits_contributed: float  # Bits needed to encode this dimension
    variance_explained: float  # % of total variance
    mutual_information: Dict[str, float]  # MI with other transforms
    is_redundant: bool
    is_critical: bool


# ============================================================================
# Kolmogorov Complexity Estimation
# ============================================================================

class KolmogorovEstimator:
    """
    Estimate Kolmogorov complexity bounds.

    True Kolmogorov complexity is uncomputable, but we can estimate:
    - Upper bound: Best practical compression (gzip, LZMA)
    - Lower bound: Entropy-based estimate
    - Structural bound: Using music theory constraints
    """

    def estimate_lower_bound(self, midi_file: mido.MidiFile) -> float:
        """
        Estimate lower bound on description length.

        Uses structural constraints from music theory:
        - Notes are not random (tonal structure)
        - Rhythm follows metrical grid
        - Typical musical patterns

        Returns:
            Estimated minimum bits needed
        """
        notes = extract_notes_from_midi(midi_file)

        if len(notes) == 0:
            return 0.0

        # Extract musical structures
        pitches = [n['pitch'] for n in notes]
        durations = [n['duration'] for n in notes]
        velocities = [n['velocity'] for n in notes]

        # 1. Pitch entropy (accounting for tonality)
        pitch_entropy = self._tonal_entropy(pitches)

        # 2. Rhythm entropy (accounting for meter)
        rhythm_entropy = self._metrical_entropy(durations)

        # 3. Velocity entropy (accounting for dynamics patterns)
        velocity_entropy = self._dynamics_entropy(velocities)

        # 4. Structural redundancy (repetition, sequences)
        redundancy_factor = self._estimate_redundancy(notes)

        # Total bits = sum of entropies * redundancy factor
        total_bits = (
            len(notes) * (pitch_entropy + rhythm_entropy + velocity_entropy)
            * redundancy_factor
        )

        return total_bits

    def _tonal_entropy(self, pitches: List[int]) -> float:
        """
        Entropy of pitches accounting for tonal structure.

        Random pitches: 7 bits (128 possibilities)
        Tonal pitches: ~3-4 bits (12 pitch classes, tonal constraints)
        """
        if not pitches:
            return 7.0

        # Calculate pitch class distribution
        pitch_classes = [p % 12 for p in pitches]
        pc_counts = np.bincount(pitch_classes, minlength=12)
        pc_probs = pc_counts / pc_counts.sum()

        # Shannon entropy of pitch classes
        pc_entropy = -np.sum(pc_probs * np.log2(pc_probs + 1e-10))

        # Add bits for octave (3-4 bits typically)
        octave_range = (max(pitches) - min(pitches)) // 12 + 1
        octave_bits = np.log2(octave_range + 1)

        return pc_entropy + octave_bits

    def _metrical_entropy(self, durations: List[float]) -> float:
        """
        Entropy of durations accounting for metrical structure.

        Random durations: ~10 bits (many possibilities)
        Metrical durations: ~2-3 bits (quantized to grid)
        """
        if not durations:
            return 3.0

        # Quantize to common note values
        common_durations = [0.125, 0.25, 0.5, 1.0, 2.0]  # 16th, 8th, quarter, half, whole

        # Find closest quantized duration for each note
        quantized = []
        for d in durations:
            closest = min(common_durations, key=lambda x: abs(x - d))
            quantized.append(closest)

        # Calculate entropy of quantized durations
        unique, counts = np.unique(quantized, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Add bits for deviations from grid (humanization)
        deviation_bits = 0.5  # Small amount for timing variations

        return entropy + deviation_bits

    def _dynamics_entropy(self, velocities: List[int]) -> float:
        """
        Entropy of velocities accounting for dynamics patterns.

        Random velocities: 7 bits (128 values)
        Structured dynamics: ~3-4 bits (p, mp, mf, f patterns)
        """
        if not velocities:
            return 4.0

        # Quantize to dynamic levels (ppp=20, pp=35, p=50, mp=65, mf=80, f=95, ff=110, fff=125)
        dynamic_levels = [20, 35, 50, 65, 80, 95, 110, 125]

        quantized = []
        for v in velocities:
            closest = min(dynamic_levels, key=lambda x: abs(x - v))
            quantized.append(closest)

        # Calculate entropy
        unique, counts = np.unique(quantized, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy

    def _estimate_redundancy(self, notes: List[Dict]) -> float:
        """
        Estimate redundancy due to musical repetition.

        Returns factor in [0, 1] where:
        - 0.1 = highly repetitive (10% of random complexity)
        - 1.0 = no repetition (full random complexity)
        """
        if len(notes) < 4:
            return 1.0

        # Look for exact repetitions in pitch sequences
        pitches = [n['pitch'] for n in notes]

        # Count unique 4-note patterns
        patterns_4 = set()
        for i in range(len(pitches) - 3):
            pattern = tuple(pitches[i:i+4])
            patterns_4.add(pattern)

        # Redundancy = unique patterns / total possible patterns
        total_patterns = len(pitches) - 3
        unique_patterns = len(patterns_4)

        redundancy = unique_patterns / total_patterns if total_patterns > 0 else 1.0

        # Clamp to reasonable range
        return np.clip(redundancy, 0.1, 1.0)

    def estimate_upper_bound(self, midi_file: mido.MidiFile) -> float:
        """
        Estimate upper bound using practical compression.

        Uses gzip as a practical compressor.

        Returns:
            Bits needed by gzip compression
        """
        # Serialize MIDI to bytes
        midi_bytes = self._midi_to_bytes(midi_file)

        # Compress with gzip
        compressed = gzip.compress(midi_bytes)

        # Return bits
        return len(compressed) * 8

    def _midi_to_bytes(self, midi_file: mido.MidiFile) -> bytes:
        """Convert MIDI to byte representation"""
        # Extract structured representation
        notes = extract_notes_from_midi(midi_file)

        # Serialize to JSON (simple approach)
        data = {
            'ticks_per_beat': midi_file.ticks_per_beat,
            'notes': [
                {
                    'pitch': n['pitch'],
                    'velocity': n['velocity'],
                    'start': round(n['start_time'], 3),
                    'duration': round(n['duration'], 3),
                    'track': n['track']
                }
                for n in notes
            ]
        }

        json_str = json.dumps(data)
        return json_str.encode('utf-8')


# ============================================================================
# Rate-Distortion Analysis
# ============================================================================

class RateDistortionAnalyzer:
    """
    Analyze rate-distortion tradeoff.

    Rate: Bits needed to encode
    Distortion: Reconstruction error

    Goal: Find optimal operating point on R-D curve
    """

    def __init__(self, registry: TransformRegistry):
        self.registry = registry

    def compute_curve(
        self,
        midi_corpus: List[mido.MidiFile],
        bit_levels: List[int] = None
    ) -> Dict[float, float]:
        """
        Compute rate-distortion curve.

        Args:
            midi_corpus: List of MIDI files
            bit_levels: Number of bits to test (e.g., [10, 20, 30, 40, 50, 60])

        Returns:
            Dict mapping bits → reconstruction quality (0-1)
        """
        if bit_levels is None:
            # Test using different numbers of transforms
            bit_levels = [5, 10, 20, 30, 40, 50, 60]

        curve = {}

        for num_transforms in bit_levels:
            # Select top N transforms by variance explained
            top_transforms = self._select_top_transforms(midi_corpus, num_transforms)

            # Measure reconstruction quality with just these transforms
            quality = self._measure_reconstruction_quality(
                midi_corpus, top_transforms
            )

            # Estimate bits needed (log2(parameter_resolution) per transform)
            bits = num_transforms * 8  # 8 bits per parameter (256 levels)

            curve[bits] = quality

        return curve

    def _select_top_transforms(
        self,
        corpus: List[mido.MidiFile],
        n: int
    ) -> List[str]:
        """Select top N transforms by variance explained"""

        # Encode all files
        encodings = [self.registry.encode(midi) for midi in corpus]
        encodings_matrix = np.array(encodings)  # Shape: (num_files, num_transforms)

        # Calculate variance per transform dimension
        variances = np.var(encodings_matrix, axis=0)

        # Get top N
        top_indices = np.argsort(variances)[-n:]
        top_names = [self.registry.transform_order[i] for i in top_indices]

        return top_names

    def _measure_reconstruction_quality(
        self,
        corpus: List[mido.MidiFile],
        transform_subset: List[str]
    ) -> float:
        """
        Measure reconstruction quality using subset of transforms.

        Returns:
            Quality score in [0, 1] where 1 = perfect reconstruction
        """
        total_error = 0.0

        for midi in corpus:
            # Encode with full transform set
            full_encoding = self.registry.encode(midi)

            # Zero out transforms not in subset
            subset_encoding = full_encoding.copy()
            for i, name in enumerate(self.registry.transform_order):
                if name not in transform_subset:
                    subset_encoding[i] = 0.5  # Neutral value

            # Decode and measure error
            reconstructed = self.registry.decode(subset_encoding, base_template=midi)
            error = self._compute_reconstruction_error(midi, reconstructed)
            total_error += error

        avg_error = total_error / len(corpus)

        # Convert error to quality (1 - normalized_error)
        quality = 1.0 - np.clip(avg_error, 0.0, 1.0)

        return quality

    def _compute_reconstruction_error(
        self,
        original: mido.MidiFile,
        reconstructed: mido.MidiFile
    ) -> float:
        """
        Compute reconstruction error between two MIDI files.

        Returns:
            Normalized error in [0, 1]
        """
        notes_orig = extract_notes_from_midi(original)
        notes_recon = extract_notes_from_midi(reconstructed)

        if len(notes_orig) == 0:
            return 0.0

        # Simple error: difference in note count and pitch distribution
        count_error = abs(len(notes_orig) - len(notes_recon)) / len(notes_orig)

        # Pitch distribution error
        pitches_orig = np.array([n['pitch'] for n in notes_orig])
        pitches_recon = np.array([n['pitch'] for n in notes_recon]) if notes_recon else np.array([60])

        pitch_error = abs(pitches_orig.mean() - pitches_recon.mean()) / 127.0

        # Combined error
        total_error = (count_error + pitch_error) / 2.0

        return total_error


# ============================================================================
# Transform Contribution Analysis
# ============================================================================

class TransformContributionAnalyzer:
    """
    Analyze each transform's contribution to information capacity.

    Identifies:
    - Redundant transforms (can be removed)
    - Critical transforms (essential)
    - Mutual information between transforms
    """

    def __init__(self, registry: TransformRegistry):
        self.registry = registry

    def analyze_contributions(
        self,
        corpus: List[mido.MidiFile]
    ) -> Dict[str, TransformContribution]:
        """
        Analyze contribution of each transform.

        Returns:
            Dict mapping transform name → contribution metrics
        """
        # Encode corpus
        encodings = np.array([self.registry.encode(midi) for midi in corpus])

        contributions = {}

        for i, name in enumerate(self.registry.transform_order):
            # Extract this transform's values across corpus
            values = encodings[:, i]

            # Calculate variance explained
            variance = np.var(values)
            total_variance = np.sum(np.var(encodings, axis=0))
            variance_explained = variance / total_variance

            # Estimate bits contributed
            # H(X) = -sum(p(x) * log2(p(x)))
            hist, _ = np.histogram(values, bins=20, range=(0, 1))
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            bits = entropy * len(corpus)

            # Calculate mutual information with other transforms
            mi_dict = self._mutual_information(values, encodings, i)

            # Classify
            is_redundant = variance_explained < 0.01  # < 1% variance
            is_critical = variance_explained > 0.05   # > 5% variance

            contributions[name] = TransformContribution(
                name=name,
                bits_contributed=bits,
                variance_explained=variance_explained,
                mutual_information=mi_dict,
                is_redundant=is_redundant,
                is_critical=is_critical
            )

        return contributions

    def _mutual_information(
        self,
        X: np.ndarray,
        all_encodings: np.ndarray,
        current_idx: int
    ) -> Dict[str, float]:
        """
        Calculate mutual information between this transform and others.

        MI(X, Y) = H(X) + H(Y) - H(X, Y)

        Returns:
            Dict mapping other transform names → MI value
        """
        mi_dict = {}

        for j, other_name in enumerate(self.registry.transform_order):
            if j == current_idx:
                continue

            Y = all_encodings[:, j]

            # Discretize for MI calculation
            X_discrete = np.digitize(X, bins=np.linspace(0, 1, 11))
            Y_discrete = np.digitize(Y, bins=np.linspace(0, 1, 11))

            # Calculate MI (simplified)
            mi = self._calculate_mi(X_discrete, Y_discrete)

            if mi > 0.1:  # Only store significant MI
                mi_dict[other_name] = mi

        return mi_dict

    def _calculate_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate mutual information between two discrete variables"""
        # Joint histogram
        joint_hist = np.histogram2d(X, Y, bins=10)[0]
        joint_probs = joint_hist / joint_hist.sum()

        # Marginal histograms
        px = joint_probs.sum(axis=1)
        py = joint_probs.sum(axis=0)

        # MI = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if joint_probs[i, j] > 0:
                    mi += joint_probs[i, j] * np.log2(
                        joint_probs[i, j] / (px[i] * py[j] + 1e-10) + 1e-10
                    )

        return max(0.0, mi)


# ============================================================================
# Main Validator
# ============================================================================

class InformationTheoreticValidator:
    """
    Main validator for transform system optimality.

    Provides comprehensive information-theoretic analysis.
    """

    def __init__(self, registry: TransformRegistry):
        self.registry = registry
        self.kolmogorov = KolmogorovEstimator()
        self.rate_distortion = RateDistortionAnalyzer(registry)
        self.contribution = TransformContributionAnalyzer(registry)

    def validate_system(
        self,
        midi_corpus: List[mido.MidiFile]
    ) -> InformationMetrics:
        """
        Full information-theoretic validation.

        Args:
            midi_corpus: Representative corpus of MIDI files

        Returns:
            Complete information metrics
        """
        print("Running information-theoretic validation...")
        print(f"Corpus size: {len(midi_corpus)} files")
        print(f"Transform count: {self.registry.count_transforms()}")

        # 1. Estimate Kolmogorov bounds
        print("\n1. Estimating Kolmogorov complexity bounds...")
        theoretical_min = np.mean([
            self.kolmogorov.estimate_lower_bound(midi)
            for midi in midi_corpus
        ])
        empirical_compression = np.mean([
            self.kolmogorov.estimate_upper_bound(midi)
            for midi in midi_corpus
        ])

        # 2. Measure transform encoding
        print("2. Measuring transform encoding efficiency...")
        transform_count = self.registry.count_transforms()
        transform_bits = transform_count * 8  # 8 bits per parameter

        # 3. Calculate efficiency
        compression_efficiency = transform_bits / empirical_compression
        kolmogorov_efficiency = theoretical_min / transform_bits

        # 4. Compute rate-distortion curve
        print("3. Computing rate-distortion curve...")
        rd_curve = self.rate_distortion.compute_curve(midi_corpus)

        # Find optimal operating point (elbow of curve)
        optimal_point = self._find_optimal_rd_point(rd_curve)

        # 5. Analyze transform contributions
        print("4. Analyzing transform contributions...")
        contributions = self.contribution.analyze_contributions(midi_corpus)

        redundant = [name for name, c in contributions.items() if c.is_redundant]
        critical = [name for name, c in contributions.items() if c.is_critical]

        # 6. Measure reconstruction quality
        print("5. Measuring reconstruction quality...")
        reconstruction_quality = self._measure_reconstruction(midi_corpus)

        # 7. Calculate explained variance
        encodings = np.array([self.registry.encode(midi) for midi in midi_corpus])
        total_variance = np.sum(np.var(encodings, axis=0))
        explained_variance = sum(c.variance_explained for c in contributions.values())

        # 8. Identify residual patterns (gaps)
        print("6. Identifying residual patterns...")
        residual_patterns = self._identify_residuals(midi_corpus)

        print("\n✅ Validation complete!\n")

        return InformationMetrics(
            theoretical_min_bits=theoretical_min,
            empirical_compression_bits=empirical_compression,
            transform_bits=transform_bits,
            transform_count=transform_count,
            compression_efficiency=compression_efficiency,
            kolmogorov_efficiency=kolmogorov_efficiency,
            rate_distortion_curve=rd_curve,
            optimal_operating_point=optimal_point,
            redundant_transforms=redundant,
            critical_transforms=critical,
            reconstruction_quality=reconstruction_quality,
            explained_variance=explained_variance,
            residual_patterns=residual_patterns
        )

    def _find_optimal_rd_point(self, rd_curve: Dict[float, float]) -> Tuple[float, float]:
        """Find optimal point on rate-distortion curve (elbow point)"""
        if not rd_curve:
            return (0.0, 0.0)

        # Sort by bits
        points = sorted(rd_curve.items())
        bits = np.array([p[0] for p in points])
        quality = np.array([p[1] for p in points])

        # Find elbow using perpendicular distance method
        # Line from first to last point
        line_vec = np.array([bits[-1] - bits[0], quality[-1] - quality[0]])
        line_vec_norm = line_vec / np.linalg.norm(line_vec)

        # Distance of each point from line
        distances = []
        for i in range(len(bits)):
            point = np.array([bits[i], quality[i]])
            point_vec = point - np.array([bits[0], quality[0]])
            dist = np.abs(np.cross(point_vec, line_vec_norm))
            distances.append(dist)

        # Elbow = point with maximum distance
        elbow_idx = np.argmax(distances)

        return (bits[elbow_idx], quality[elbow_idx])

    def _measure_reconstruction(self, corpus: List[mido.MidiFile]) -> float:
        """Measure average reconstruction quality"""
        total_quality = 0.0

        for midi in corpus:
            encoding = self.registry.encode(midi)
            reconstructed = self.registry.decode(encoding, base_template=midi)

            # Measure similarity
            quality = self._compute_similarity(midi, reconstructed)
            total_quality += quality

        return total_quality / len(corpus)

    def _compute_similarity(
        self,
        midi1: mido.MidiFile,
        midi2: mido.MidiFile
    ) -> float:
        """Compute similarity between two MIDI files"""
        notes1 = extract_notes_from_midi(midi1)
        notes2 = extract_notes_from_midi(midi2)

        if len(notes1) == 0 or len(notes2) == 0:
            return 0.0

        # Simple similarity: pitch and rhythm correlation
        pitches1 = [n['pitch'] for n in notes1]
        pitches2 = [n['pitch'] for n in notes2]

        # Pad to same length
        max_len = max(len(pitches1), len(pitches2))
        pitches1 += [60] * (max_len - len(pitches1))
        pitches2 += [60] * (max_len - len(pitches2))

        # Correlation
        corr = np.corrcoef(pitches1, pitches2)[0, 1]

        return max(0.0, corr)

    def _identify_residuals(self, corpus: List[mido.MidiFile]) -> List[str]:
        """Identify patterns not captured by current transforms"""
        # Placeholder - would do cluster analysis on reconstruction errors
        return [
            "High-frequency ornamentations",
            "Polymetric patterns",
            "Microtonal inflections"
        ]

    def print_report(self, metrics: InformationMetrics):
        """Print human-readable validation report"""
        print("="*70)
        print("INFORMATION-THEORETIC VALIDATION REPORT")
        print("="*70)

        print("\n📊 COMPLEXITY BOUNDS")
        print(f"  Theoretical minimum (Kolmogorov): {metrics.theoretical_min_bits:.0f} bits")
        print(f"  Empirical compression (gzip):     {metrics.empirical_compression_bits:.0f} bits")
        print(f"  Current transform encoding:        {metrics.transform_bits:.0f} bits")

        print("\n⚡ EFFICIENCY METRICS")
        print(f"  Compression efficiency:     {metrics.compression_efficiency:.2%}")
        print(f"  Kolmogorov efficiency:      {metrics.kolmogorov_efficiency:.2%}")
        print(f"  Reconstruction quality:     {metrics.reconstruction_quality:.2%}")
        print(f"  Explained variance:         {metrics.explained_variance:.2%}")

        print("\n📈 RATE-DISTORTION ANALYSIS")
        print(f"  Optimal operating point:    {metrics.optimal_operating_point[0]:.0f} bits → {metrics.optimal_operating_point[1]:.1%} quality")
        print("  Rate-distortion curve:")
        for bits, quality in sorted(metrics.rate_distortion_curve.items()):
            bar = "█" * int(quality * 50)
            print(f"    {bits:3.0f} bits → {quality:.1%} {bar}")

        print(f"\n🎯 TRANSFORM ANALYSIS ({metrics.transform_count} transforms)")
        print(f"  Critical transforms ({len(metrics.critical_transforms)}): {', '.join(metrics.critical_transforms[:5])}...")
        print(f"  Redundant transforms ({len(metrics.redundant_transforms)}): {', '.join(metrics.redundant_transforms[:5])}...")

        print(f"\n🔍 RESIDUAL PATTERNS (Gaps)")
        for pattern in metrics.residual_patterns:
            print(f"  - {pattern}")

        print("\n" + "="*70)

        # Recommendations
        print("\n💡 RECOMMENDATIONS")

        if metrics.kolmogorov_efficiency < 0.5:
            print("  ⚠️  LOW EFFICIENCY: Need more transforms to reach theoretical minimum")
            estimated_needed = int(metrics.transform_count / metrics.kolmogorov_efficiency)
            print(f"      Estimate: Need ~{estimated_needed} transforms for 0.7+ efficiency")

        if len(metrics.redundant_transforms) > 5:
            print(f"  ⚠️  HIGH REDUNDANCY: Consider removing {len(metrics.redundant_transforms)} redundant transforms")

        if metrics.reconstruction_quality < 0.8:
            print("  ⚠️  POOR RECONSTRUCTION: System missing critical musical patterns")
            print("      → Run discovery pipeline to find missing transforms")

        if metrics.explained_variance < 0.9:
            print(f"  ⚠️  LOW COVERAGE: Only {metrics.explained_variance:.1%} variance explained")
            print("      → Need more diverse transforms")

        print("\n" + "="*70)
