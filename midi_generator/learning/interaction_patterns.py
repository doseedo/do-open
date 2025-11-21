#!/usr/bin/env python3
"""
Interaction Pattern Discovery - Agent 7
========================================

This module discovers and analyzes interaction patterns between musical dimensions.

Key capabilities:
1. Detect cross-dimensional correlations
2. Identify causal relationships (e.g., harmony changes → texture changes)
3. Discover temporal patterns (e.g., section boundaries → orchestration shifts)
4. Find statistical dependencies between dimensions

Author: Agent 7 - Cross-Dimensional Pattern Discoverer
Date: November 21, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from pathlib import Path
import json


@dataclass
class InteractionPattern:
    """Represents a discovered interaction pattern between dimensions"""

    dimension_a: str
    dimension_b: str
    pattern_type: str  # 'correlation', 'causation', 'temporal', 'coupling'
    strength: float  # 0.0 to 1.0
    direction: Optional[str] = None  # 'a_to_b', 'b_to_a', 'bidirectional'
    lag: int = 0  # Temporal lag in measures
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dimension_a': self.dimension_a,
            'dimension_b': self.dimension_b,
            'pattern_type': self.pattern_type,
            'strength': float(self.strength),
            'direction': self.direction,
            'lag': self.lag,
            'description': self.description
        }


class InteractionPatternDiscoverer:
    """
    Discovers interaction patterns between musical dimensions.

    This class analyzes outputs from the 5 dimension encoders to find:
    1. Statistical correlations
    2. Causal relationships
    3. Temporal dependencies
    4. Structural couplings
    """

    def __init__(
        self,
        min_correlation: float = 0.3,
        min_samples: int = 10,
        max_lag: int = 4
    ):
        """
        Initialize pattern discoverer.

        Args:
            min_correlation: Minimum correlation to consider significant
            min_samples: Minimum number of samples required
            max_lag: Maximum temporal lag to check (in measures)
        """
        self.min_correlation = min_correlation
        self.min_samples = min_samples
        self.max_lag = max_lag

        self.discovered_patterns: List[InteractionPattern] = []

        # Dimension names and sizes
        self.dimensions = {
            'harmony': 30,
            'rhythm': 20,
            'form': 15,
            'orchestration': 25,
            'texture': 20
        }

    def discover_patterns(
        self,
        dimension_features: Dict[str, np.ndarray],
        temporal_data: Optional[Dict[str, List[np.ndarray]]] = None
    ) -> List[InteractionPattern]:
        """
        Discover all interaction patterns from dimension features.

        Args:
            dimension_features: Dictionary mapping dimension names to feature arrays
                                Each array is [num_samples, feature_dim]
            temporal_data: Optional temporal sequences for each dimension
                          Dict[dimension_name, List[feature_arrays_over_time]]

        Returns:
            List of discovered interaction patterns
        """
        self.discovered_patterns = []

        # 1. Discover correlation patterns
        correlation_patterns = self._discover_correlations(dimension_features)
        self.discovered_patterns.extend(correlation_patterns)

        # 2. Discover causal patterns
        causal_patterns = self._discover_causal_patterns(dimension_features)
        self.discovered_patterns.extend(causal_patterns)

        # 3. Discover temporal patterns (if temporal data provided)
        if temporal_data is not None:
            temporal_patterns = self._discover_temporal_patterns(temporal_data)
            self.discovered_patterns.extend(temporal_patterns)

        # 4. Discover coupling patterns
        coupling_patterns = self._discover_coupling_patterns(dimension_features)
        self.discovered_patterns.extend(coupling_patterns)

        # Sort by strength
        self.discovered_patterns.sort(key=lambda p: p.strength, reverse=True)

        return self.discovered_patterns

    def _discover_correlations(
        self,
        dimension_features: Dict[str, np.ndarray]
    ) -> List[InteractionPattern]:
        """
        Discover correlation patterns between dimensions.

        Computes correlation between aggregate statistics of each dimension pair.
        """
        patterns = []
        dimension_names = list(dimension_features.keys())

        for i, dim_a in enumerate(dimension_names):
            for dim_b in dimension_names[i+1:]:
                # Compute aggregate statistics for each dimension
                stats_a = self._compute_aggregate_stats(dimension_features[dim_a])
                stats_b = self._compute_aggregate_stats(dimension_features[dim_b])

                # Compute correlation
                if len(stats_a) >= self.min_samples and len(stats_b) >= self.min_samples:
                    correlation, p_value = stats.pearsonr(stats_a, stats_b)

                    if abs(correlation) >= self.min_correlation and p_value < 0.05:
                        pattern = InteractionPattern(
                            dimension_a=dim_a,
                            dimension_b=dim_b,
                            pattern_type='correlation',
                            strength=abs(correlation),
                            direction='bidirectional',
                            description=f"{dim_a} and {dim_b} are {'positively' if correlation > 0 else 'negatively'} correlated (r={correlation:.3f}, p={p_value:.4f})"
                        )
                        patterns.append(pattern)

        return patterns

    def _discover_causal_patterns(
        self,
        dimension_features: Dict[str, np.ndarray]
    ) -> List[InteractionPattern]:
        """
        Discover causal patterns using Granger causality.

        Tests if one dimension helps predict another.
        """
        patterns = []
        dimension_names = list(dimension_features.keys())

        for dim_a in dimension_names:
            for dim_b in dimension_names:
                if dim_a == dim_b:
                    continue

                # Compute aggregate statistics
                stats_a = self._compute_aggregate_stats(dimension_features[dim_a])
                stats_b = self._compute_aggregate_stats(dimension_features[dim_b])

                if len(stats_a) >= self.min_samples * 2:
                    # Simple Granger causality test
                    # Check if past values of A help predict current values of B
                    causality_strength = self._compute_granger_causality(stats_a, stats_b)

                    if causality_strength >= self.min_correlation:
                        pattern = InteractionPattern(
                            dimension_a=dim_a,
                            dimension_b=dim_b,
                            pattern_type='causation',
                            strength=causality_strength,
                            direction='a_to_b',
                            description=f"{dim_a} causally influences {dim_b} (strength={causality_strength:.3f})"
                        )
                        patterns.append(pattern)

        return patterns

    def _discover_temporal_patterns(
        self,
        temporal_data: Dict[str, List[np.ndarray]]
    ) -> List[InteractionPattern]:
        """
        Discover temporal lag patterns.

        Checks if changes in one dimension predict changes in another at various lags.
        """
        patterns = []
        dimension_names = list(temporal_data.keys())

        for dim_a in dimension_names:
            for dim_b in dimension_names:
                if dim_a == dim_b:
                    continue

                # Find optimal lag
                best_lag, max_correlation = self._find_optimal_lag(
                    temporal_data[dim_a],
                    temporal_data[dim_b]
                )

                if max_correlation >= self.min_correlation:
                    pattern = InteractionPattern(
                        dimension_a=dim_a,
                        dimension_b=dim_b,
                        pattern_type='temporal',
                        strength=max_correlation,
                        direction='a_to_b',
                        lag=best_lag,
                        description=f"{dim_a} predicts {dim_b} with {best_lag}-measure lag (r={max_correlation:.3f})"
                    )
                    patterns.append(pattern)

        return patterns

    def _discover_coupling_patterns(
        self,
        dimension_features: Dict[str, np.ndarray]
    ) -> List[InteractionPattern]:
        """
        Discover coupling patterns (mutual information).

        Measures how much knowing one dimension reduces uncertainty about another.
        """
        patterns = []
        dimension_names = list(dimension_features.keys())

        for i, dim_a in enumerate(dimension_names):
            for dim_b in dimension_names[i+1:]:
                # Compute mutual information
                stats_a = self._compute_aggregate_stats(dimension_features[dim_a])
                stats_b = self._compute_aggregate_stats(dimension_features[dim_b])

                if len(stats_a) >= self.min_samples:
                    mi = self._compute_mutual_information(stats_a, stats_b)

                    # Normalize MI to [0, 1]
                    mi_normalized = min(mi / 2.0, 1.0)

                    if mi_normalized >= self.min_correlation:
                        pattern = InteractionPattern(
                            dimension_a=dim_a,
                            dimension_b=dim_b,
                            pattern_type='coupling',
                            strength=mi_normalized,
                            direction='bidirectional',
                            description=f"{dim_a} and {dim_b} are coupled (MI={mi:.3f})"
                        )
                        patterns.append(pattern)

        return patterns

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _compute_aggregate_stats(self, features: np.ndarray) -> np.ndarray:
        """
        Compute aggregate statistics for a dimension.

        Args:
            features: [num_samples, feature_dim]

        Returns:
            Aggregated statistics [num_samples]
        """
        # Use mean across features as aggregate
        return features.mean(axis=1)

    def _compute_granger_causality(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Compute simplified Granger causality.

        Returns strength of A predicting B (0.0 to 1.0).
        """
        if len(series_a) < lag + self.min_samples:
            return 0.0

        # Prepare lagged data
        X = series_a[:-lag] if lag > 0 else series_a
        Y = series_b[lag:] if lag > 0 else series_b

        # Ensure same length
        min_len = min(len(X), len(Y))
        X = X[:min_len]
        Y = Y[:min_len]

        if len(X) < self.min_samples:
            return 0.0

        # Compute correlation of lagged series
        correlation, p_value = stats.pearsonr(X, Y)

        if p_value < 0.05:
            return abs(correlation)
        else:
            return 0.0

    def _find_optimal_lag(
        self,
        temporal_a: List[np.ndarray],
        temporal_b: List[np.ndarray]
    ) -> Tuple[int, float]:
        """
        Find optimal temporal lag between two dimensions.

        Returns:
            (best_lag, max_correlation)
        """
        # Convert to aggregate statistics over time
        stats_a = [arr.mean() for arr in temporal_a]
        stats_b = [arr.mean() for arr in temporal_b]

        best_lag = 0
        max_correlation = 0.0

        for lag in range(self.max_lag + 1):
            if len(stats_a) < lag + self.min_samples:
                continue

            # Compute correlation at this lag
            X = stats_a[:-lag] if lag > 0 else stats_a
            Y = stats_b[lag:] if lag > 0 else stats_b

            min_len = min(len(X), len(Y))
            X = X[:min_len]
            Y = Y[:min_len]

            if len(X) >= self.min_samples:
                correlation, p_value = stats.pearsonr(X, Y)

                if p_value < 0.05 and abs(correlation) > abs(max_correlation):
                    max_correlation = correlation
                    best_lag = lag

        return best_lag, abs(max_correlation)

    def _compute_mutual_information(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Compute mutual information between two series.

        Returns mutual information in nats.
        """
        # Discretize into bins
        a_binned = np.digitize(series_a, bins=np.linspace(series_a.min(), series_a.max(), bins))
        b_binned = np.digitize(series_b, bins=np.linspace(series_b.min(), series_b.max(), bins))

        # Compute joint and marginal histograms
        joint_hist, _, _ = np.histogram2d(a_binned, b_binned, bins=bins)
        joint_hist = joint_hist / joint_hist.sum()  # Normalize

        a_hist = joint_hist.sum(axis=1)
        b_hist = joint_hist.sum(axis=0)

        # Compute MI
        mi = 0.0
        for i in range(len(a_hist)):
            for j in range(len(b_hist)):
                if joint_hist[i, j] > 0 and a_hist[i] > 0 and b_hist[j] > 0:
                    mi += joint_hist[i, j] * np.log(
                        joint_hist[i, j] / (a_hist[i] * b_hist[j])
                    )

        return mi

    # =============================================================================
    # Export and Visualization
    # =============================================================================

    def export_patterns(self, output_path: Path):
        """Export discovered patterns to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        patterns_dict = {
            'num_patterns': len(self.discovered_patterns),
            'patterns': [p.to_dict() for p in self.discovered_patterns]
        }

        with open(output_path, 'w') as f:
            json.dump(patterns_dict, f, indent=2)

        print(f"✅ Exported {len(self.discovered_patterns)} patterns to {output_path}")

    def generate_report(self) -> str:
        """Generate human-readable report of discovered patterns"""
        report = []
        report.append("=" * 80)
        report.append("INTERACTION PATTERN DISCOVERY REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal patterns discovered: {len(self.discovered_patterns)}\n")

        # Group by pattern type
        by_type = {}
        for pattern in self.discovered_patterns:
            by_type.setdefault(pattern.pattern_type, []).append(pattern)

        for pattern_type, patterns in by_type.items():
            report.append(f"\n{pattern_type.upper()} PATTERNS ({len(patterns)}):")
            report.append("-" * 80)

            for i, pattern in enumerate(patterns[:10], 1):  # Top 10
                report.append(f"\n{i}. {pattern.description}")
                report.append(f"   Strength: {pattern.strength:.3f}")
                if pattern.direction:
                    report.append(f"   Direction: {pattern.direction}")
                if pattern.lag > 0:
                    report.append(f"   Lag: {pattern.lag} measures")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Interaction Pattern Discovery - Agent 7")
    print("=" * 80)

    # Create discoverer
    print("\n1. Creating pattern discoverer...")
    discoverer = InteractionPatternDiscoverer(
        min_correlation=0.3,
        min_samples=20,
        max_lag=4
    )
    print("   ✅ Pattern discoverer initialized")

    # Generate synthetic test data
    print("\n2. Generating synthetic dimension features...")
    np.random.seed(42)

    num_samples = 100
    dimension_features = {
        'harmony': np.random.randn(num_samples, 30),
        'rhythm': np.random.randn(num_samples, 20),
        'form': np.random.randn(num_samples, 15),
        'orchestration': np.random.randn(num_samples, 25),
        'texture': np.random.randn(num_samples, 20)
    }

    # Add some correlations for testing
    # Make texture correlate with harmony
    harmony_mean = dimension_features['harmony'].mean(axis=1)
    dimension_features['texture'][:, 0] = harmony_mean + np.random.randn(num_samples) * 0.1

    # Make orchestration causally follow form
    form_mean = dimension_features['form'].mean(axis=1)
    dimension_features['orchestration'][:, 0] = np.roll(form_mean, 2) + np.random.randn(num_samples) * 0.1

    print("   ✅ Generated test data with synthetic patterns")

    # Discover patterns
    print("\n3. Discovering interaction patterns...")
    patterns = discoverer.discover_patterns(dimension_features)
    print(f"   ✅ Discovered {len(patterns)} patterns")

    # Generate report
    print("\n4. Generating report...")
    report = discoverer.generate_report()
    print(report)

    # Export patterns
    print("\n5. Exporting patterns...")
    output_path = Path("/tmp/interaction_patterns.json")
    discoverer.export_patterns(output_path)

    print("\n" + "=" * 80)
    print("✅ Pattern discovery complete!")
    print("=" * 80)
