"""
Feature Correlation Analyzer - Agent 25
========================================

Analyzes correlations between the 1,000 musical features extracted by Agent 8
to optimize model training and identify redundant features.

This analyzer provides:
1. Pairwise correlation analysis (Pearson, Spearman, Kendall)
2. Redundant feature identification
3. Feature subset suggestion for each parameter
4. Feature interaction detection
5. Correlation visualizations (heatmaps, dendrograms)
6. Dimensionality reduction recommendations
7. Feature importance rankings per parameter
8. Automated feature selection strategies

Goals:
- Identify highly correlated features (r > 0.95) for removal
- Find optimal feature subsets per parameter (reduce from 1000 to ~100)
- Detect important feature interactions
- Improve model training efficiency
- Reduce overfitting through feature selection

Author: Agent 25 - Feature Correlation Analyzer
License: MIT
"""

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed, some correlation methods unavailable")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("WARNING: matplotlib/seaborn not installed, visualizations unavailable")

try:
    from sklearn.feature_selection import (
        mutual_info_regression,
        mutual_info_classif,
        SelectKBest,
        f_regression,
        f_classif
    )
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed, some feature selection methods unavailable")

warnings.filterwarnings('ignore')


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    feature1: str
    feature2: str
    correlation: float
    p_value: Optional[float] = None
    method: str = 'pearson'


@dataclass
class RedundantFeaturePair:
    """Pair of redundant features"""
    feature1: str
    feature2: str
    correlation: float
    recommendation: str  # Which feature to keep


@dataclass
class FeatureSubset:
    """Recommended feature subset for a parameter"""
    parameter_name: str
    selected_features: List[str]
    importance_scores: Dict[str, float]
    selection_method: str
    n_features: int
    expected_performance: Optional[float] = None


@dataclass
class FeatureInteraction:
    """Detected feature interaction"""
    feature1: str
    feature2: str
    interaction_strength: float
    parameter_relevance: Optional[str] = None


@dataclass
class CorrelationAnalysisReport:
    """Complete correlation analysis report"""
    total_features: int
    correlation_matrix: np.ndarray
    feature_names: List[str]
    redundant_pairs: List[RedundantFeaturePair]
    feature_subsets: Dict[str, FeatureSubset]
    interactions: List[FeatureInteraction]
    analysis_timestamp: str
    recommendations: List[str]


# ============================================================================
# Feature Correlation Analyzer
# ============================================================================

class FeatureCorrelationAnalyzer:
    """
    Analyzes correlations between musical features to optimize model training.

    This class:
    1. Computes correlation matrices (Pearson, Spearman, Kendall)
    2. Identifies redundant features (high correlation)
    3. Suggests optimal feature subsets per parameter
    4. Detects feature interactions
    5. Generates visualizations
    6. Provides feature selection recommendations

    Usage:
        analyzer = FeatureCorrelationAnalyzer()
        analyzer.fit(feature_matrix, feature_names)
        redundant = analyzer.identify_redundant_features(threshold=0.95)
        subset = analyzer.suggest_feature_subset('harmony.chord_complexity', max_features=100)
    """

    def __init__(
        self,
        correlation_method: str = 'pearson',
        redundancy_threshold: float = 0.95,
        interaction_threshold: float = 0.3,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize Feature Correlation Analyzer.

        Args:
            correlation_method: 'pearson', 'spearman', or 'kendall'
            redundancy_threshold: Correlation above which features are redundant
            interaction_threshold: Minimum correlation for feature interactions
            cache_dir: Directory to cache correlation matrices
        """
        self.correlation_method = correlation_method
        self.redundancy_threshold = redundancy_threshold
        self.interaction_threshold = interaction_threshold
        self.cache_dir = cache_dir or Path('.feature_correlation_cache')
        self.cache_dir.mkdir(exist_ok=True)

        # Internal state
        self.feature_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.correlation_matrix: Optional[np.ndarray] = None
        self.p_values: Optional[np.ndarray] = None
        self.is_fitted = False

        # Cached results
        self._redundant_pairs: Optional[List[RedundantFeaturePair]] = None
        self._feature_clusters: Optional[List[List[str]]] = None
        self._importance_cache: Dict[str, Dict[str, float]] = {}

    def fit(
        self,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        compute_p_values: bool = False
    ) -> 'FeatureCorrelationAnalyzer':
        """
        Fit the analyzer on a feature matrix.

        Args:
            feature_matrix: Array of shape (n_samples, n_features)
            feature_names: List of feature names (length n_features)
            compute_p_values: Whether to compute p-values (slower)

        Returns:
            self for chaining
        """
        if feature_matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature matrix has {feature_matrix.shape[1]} features "
                f"but {len(feature_names)} names provided"
            )

        print(f"Fitting FeatureCorrelationAnalyzer on {feature_matrix.shape[0]} samples, "
              f"{feature_matrix.shape[1]} features...")

        self.feature_matrix = feature_matrix
        self.feature_names = feature_names

        # Compute correlation matrix
        self._compute_correlation_matrix(compute_p_values)

        self.is_fitted = True
        print(f"✅ Correlation analysis complete")

        return self

    def _compute_correlation_matrix(self, compute_p_values: bool = False):
        """Compute correlation matrix using specified method"""
        n_features = len(self.feature_names)
        self.correlation_matrix = np.zeros((n_features, n_features))

        if compute_p_values:
            self.p_values = np.ones((n_features, n_features))

        print(f"Computing {self.correlation_method} correlations...")

        if self.correlation_method == 'pearson':
            # Use pandas for fast pearson correlation
            df = pd.DataFrame(self.feature_matrix, columns=self.feature_names)
            self.correlation_matrix = df.corr().values

            if compute_p_values and HAS_SCIPY:
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        _, p = stats.pearsonr(
                            self.feature_matrix[:, i],
                            self.feature_matrix[:, j]
                        )
                        self.p_values[i, j] = p
                        self.p_values[j, i] = p

        elif self.correlation_method == 'spearman' and HAS_SCIPY:
            corr, p_matrix = stats.spearmanr(self.feature_matrix)
            self.correlation_matrix = corr
            if compute_p_values:
                self.p_values = p_matrix

        elif self.correlation_method == 'kendall' and HAS_SCIPY:
            for i in range(n_features):
                for j in range(i, n_features):
                    if i == j:
                        self.correlation_matrix[i, j] = 1.0
                    else:
                        tau, p = stats.kendalltau(
                            self.feature_matrix[:, i],
                            self.feature_matrix[:, j]
                        )
                        self.correlation_matrix[i, j] = tau
                        self.correlation_matrix[j, i] = tau
                        if compute_p_values:
                            self.p_values[i, j] = p
                            self.p_values[j, i] = p

        else:
            raise ValueError(f"Unknown correlation method: {self.correlation_method}")

    def identify_redundant_features(
        self,
        threshold: Optional[float] = None
    ) -> List[RedundantFeaturePair]:
        """
        Identify pairs of highly correlated (redundant) features.

        Args:
            threshold: Correlation threshold (default: self.redundancy_threshold)

        Returns:
            List of redundant feature pairs with recommendations
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        if self._redundant_pairs is not None and threshold is None:
            return self._redundant_pairs

        threshold = threshold or self.redundancy_threshold
        redundant_pairs = []

        n_features = len(self.feature_names)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(self.correlation_matrix[i, j])
                if corr >= threshold:
                    # Recommend keeping the feature with lower average correlation
                    avg_corr_i = np.mean(np.abs(self.correlation_matrix[i, :]))
                    avg_corr_j = np.mean(np.abs(self.correlation_matrix[j, :]))

                    if avg_corr_i < avg_corr_j:
                        recommendation = f"Keep {self.feature_names[i]}, remove {self.feature_names[j]}"
                    else:
                        recommendation = f"Keep {self.feature_names[j]}, remove {self.feature_names[i]}"

                    redundant_pairs.append(RedundantFeaturePair(
                        feature1=self.feature_names[i],
                        feature2=self.feature_names[j],
                        correlation=corr,
                        recommendation=recommendation
                    ))

        if threshold == self.redundancy_threshold:
            self._redundant_pairs = redundant_pairs

        print(f"Found {len(redundant_pairs)} redundant feature pairs (|r| > {threshold})")
        return redundant_pairs

    def get_feature_clusters(
        self,
        n_clusters: Optional[int] = None,
        linkage_method: str = 'average'
    ) -> List[List[str]]:
        """
        Cluster features based on correlation using hierarchical clustering.

        Args:
            n_clusters: Number of clusters (default: auto-detect)
            linkage_method: 'single', 'complete', 'average', 'ward'

        Returns:
            List of feature clusters
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        if not HAS_SCIPY:
            print("WARNING: scipy required for clustering")
            return [[name] for name in self.feature_names]

        # Convert correlation to distance
        distance_matrix = 1 - np.abs(self.correlation_matrix)
        condensed_dist = squareform(distance_matrix)

        # Hierarchical clustering
        linkage_matrix = hierarchy.linkage(condensed_dist, method=linkage_method)

        if n_clusters is None:
            # Auto-detect number of clusters using elbow method
            n_clusters = min(20, len(self.feature_names) // 50)

        cluster_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Group features by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(self.feature_names[idx])

        self._feature_clusters = list(clusters.values())
        print(f"Clustered {len(self.feature_names)} features into {len(self._feature_clusters)} groups")

        return self._feature_clusters

    def suggest_feature_subset(
        self,
        parameter_name: str,
        parameter_values: Optional[np.ndarray] = None,
        max_features: int = 100,
        method: str = 'correlation'
    ) -> FeatureSubset:
        """
        Suggest optimal feature subset for predicting a specific parameter.

        Args:
            parameter_name: Name of the parameter to predict
            parameter_values: Target values for the parameter (if available)
            max_features: Maximum number of features to select
            method: Selection method ('correlation', 'mutual_info', 'f_test', 'variance')

        Returns:
            FeatureSubset with selected features and importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        print(f"Selecting top {max_features} features for parameter: {parameter_name}")

        if parameter_values is not None:
            # Supervised feature selection
            importance_scores = self._compute_feature_importance(
                parameter_values, method
            )
        else:
            # Unsupervised: use variance or cached importance
            if parameter_name in self._importance_cache:
                importance_scores = self._importance_cache[parameter_name]
            else:
                # Fall back to variance
                importance_scores = {
                    name: np.var(self.feature_matrix[:, i])
                    for i, name in enumerate(self.feature_names)
                }

        # Sort by importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Remove redundant features from top features
        selected_features = []
        selected_set = set()

        for feature, score in sorted_features:
            if len(selected_features) >= max_features:
                break

            # Check if this feature is redundant with already selected features
            feature_idx = self.feature_names.index(feature)
            is_redundant = False

            for selected_feature in selected_features:
                selected_idx = self.feature_names.index(selected_feature)
                corr = abs(self.correlation_matrix[feature_idx, selected_idx])
                if corr > self.redundancy_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                selected_features.append(feature)
                selected_set.add(feature)

        # Create subset
        subset = FeatureSubset(
            parameter_name=parameter_name,
            selected_features=selected_features,
            importance_scores={f: importance_scores[f] for f in selected_features},
            selection_method=method,
            n_features=len(selected_features)
        )

        print(f"✅ Selected {len(selected_features)} features for {parameter_name}")

        return subset

    def _compute_feature_importance(
        self,
        target_values: np.ndarray,
        method: str
    ) -> Dict[str, float]:
        """Compute feature importance scores for target prediction"""
        n_features = len(self.feature_names)
        scores = np.zeros(n_features)

        if method == 'correlation':
            # Pearson correlation with target
            for i in range(n_features):
                if HAS_SCIPY:
                    corr, _ = stats.pearsonr(self.feature_matrix[:, i], target_values)
                    scores[i] = abs(corr)
                else:
                    scores[i] = abs(np.corrcoef(self.feature_matrix[:, i], target_values)[0, 1])

        elif method == 'mutual_info' and HAS_SKLEARN:
            # Mutual information
            is_classification = len(np.unique(target_values)) < 20
            if is_classification:
                scores = mutual_info_classif(self.feature_matrix, target_values)
            else:
                scores = mutual_info_regression(self.feature_matrix, target_values)

        elif method == 'f_test' and HAS_SKLEARN:
            # F-test
            is_classification = len(np.unique(target_values)) < 20
            if is_classification:
                scores, _ = f_classif(self.feature_matrix, target_values)
            else:
                scores, _ = f_regression(self.feature_matrix, target_values)

        elif method == 'variance':
            # Simple variance
            scores = np.var(self.feature_matrix, axis=0)

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        return {
            self.feature_names[i]: float(scores[i])
            for i in range(n_features)
        }

    def analyze_feature_interactions(
        self,
        threshold: Optional[float] = None
    ) -> List[FeatureInteraction]:
        """
        Identify important feature interactions (correlated feature pairs).

        Args:
            threshold: Minimum correlation for interaction (default: self.interaction_threshold)

        Returns:
            List of feature interactions
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        threshold = threshold or self.interaction_threshold
        interactions = []

        n_features = len(self.feature_names)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(self.correlation_matrix[i, j])
                if threshold < corr < self.redundancy_threshold:
                    # Moderate correlation = potential interaction
                    interactions.append(FeatureInteraction(
                        feature1=self.feature_names[i],
                        feature2=self.feature_names[j],
                        interaction_strength=corr
                    ))

        # Sort by interaction strength
        interactions.sort(key=lambda x: x.interaction_strength, reverse=True)

        print(f"Found {len(interactions)} feature interactions "
              f"({threshold} < |r| < {self.redundancy_threshold})")

        return interactions

    def get_uncorrelated_features(
        self,
        max_avg_correlation: float = 0.3
    ) -> List[str]:
        """
        Get features with low average correlation to other features.

        Args:
            max_avg_correlation: Maximum average absolute correlation

        Returns:
            List of relatively uncorrelated feature names
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        uncorrelated = []
        for i, name in enumerate(self.feature_names):
            avg_corr = np.mean(np.abs(self.correlation_matrix[i, :]))
            if avg_corr <= max_avg_correlation:
                uncorrelated.append(name)

        print(f"Found {len(uncorrelated)} features with avg correlation <= {max_avg_correlation}")
        return uncorrelated

    def visualize_correlation_matrix(
        self,
        output_path: Optional[Path] = None,
        max_features: int = 100,
        figsize: Tuple[int, int] = (20, 18)
    ):
        """
        Generate correlation heatmap visualization.

        Args:
            output_path: Where to save the plot (if None, display only)
            max_features: Maximum features to display (for readability)
            figsize: Figure size in inches
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        if not HAS_PLOTTING:
            print("WARNING: matplotlib/seaborn required for visualization")
            return

        # Subsample features if too many
        if len(self.feature_names) > max_features:
            indices = np.linspace(0, len(self.feature_names) - 1, max_features, dtype=int)
            plot_names = [self.feature_names[i] for i in indices]
            plot_corr = self.correlation_matrix[np.ix_(indices, indices)]
        else:
            plot_names = self.feature_names
            plot_corr = self.correlation_matrix

        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            plot_corr,
            xticklabels=plot_names,
            yticklabels=plot_names,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f'Feature Correlation Matrix ({self.correlation_method.capitalize()})',
                  fontsize=16, pad=20)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved correlation heatmap to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_feature_importance(
        self,
        feature_subset: FeatureSubset,
        output_path: Optional[Path] = None,
        top_n: int = 20
    ):
        """
        Plot feature importance for a parameter.

        Args:
            feature_subset: FeatureSubset from suggest_feature_subset()
            output_path: Where to save the plot
            top_n: Number of top features to display
        """
        if not HAS_PLOTTING:
            print("WARNING: matplotlib required for visualization")
            return

        # Get top N features
        sorted_features = sorted(
            feature_subset.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        features = [f[0] for f in sorted_features]
        scores = [f[1] for f in sorted_features]

        # Create bar plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Features for {feature_subset.parameter_name}',
                  fontsize=14, pad=15)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved feature importance plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_dendrogram(
        self,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot hierarchical clustering dendrogram of features.

        Args:
            output_path: Where to save the plot
            figsize: Figure size in inches
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        if not HAS_SCIPY or not HAS_PLOTTING:
            print("WARNING: scipy and matplotlib required for dendrogram")
            return

        # Compute linkage
        distance_matrix = 1 - np.abs(self.correlation_matrix)
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = hierarchy.linkage(condensed_dist, method='average')

        # Plot dendrogram
        plt.figure(figsize=figsize)
        hierarchy.dendrogram(
            linkage_matrix,
            labels=self.feature_names,
            leaf_rotation=90,
            leaf_font_size=8
        )
        plt.title('Feature Correlation Dendrogram', fontsize=14, pad=15)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Distance (1 - |correlation|)', fontsize=12)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved dendrogram to {output_path}")
        else:
            plt.show()

        plt.close()

    def generate_report(
        self,
        output_path: Optional[Path] = None
    ) -> CorrelationAnalysisReport:
        """
        Generate comprehensive correlation analysis report.

        Args:
            output_path: Where to save JSON report

        Returns:
            CorrelationAnalysisReport
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        from datetime import datetime

        redundant_pairs = self.identify_redundant_features()
        interactions = self.analyze_feature_interactions()

        # Generate recommendations
        recommendations = []

        if len(redundant_pairs) > 0:
            recommendations.append(
                f"Remove {len(redundant_pairs)} redundant feature pairs to reduce dimensionality"
            )

        avg_corr = np.mean(np.abs(self.correlation_matrix))
        recommendations.append(
            f"Average absolute correlation: {avg_corr:.3f}"
        )

        if avg_corr > 0.5:
            recommendations.append(
                "High average correlation detected - consider PCA or feature selection"
            )

        if len(interactions) > 100:
            recommendations.append(
                f"Found {len(interactions)} feature interactions - consider interaction terms in models"
            )

        report = CorrelationAnalysisReport(
            total_features=len(self.feature_names),
            correlation_matrix=self.correlation_matrix,
            feature_names=self.feature_names,
            redundant_pairs=redundant_pairs,
            feature_subsets={},  # To be filled externally
            interactions=interactions[:100],  # Top 100 interactions
            analysis_timestamp=datetime.now().isoformat(),
            recommendations=recommendations
        )

        if output_path:
            # Save report (without full correlation matrix for space)
            report_dict = {
                'total_features': report.total_features,
                'feature_names': report.feature_names,
                'redundant_pairs': [
                    {
                        'feature1': p.feature1,
                        'feature2': p.feature2,
                        'correlation': float(p.correlation),
                        'recommendation': p.recommendation
                    }
                    for p in report.redundant_pairs
                ],
                'top_interactions': [
                    {
                        'feature1': i.feature1,
                        'feature2': i.feature2,
                        'strength': float(i.interaction_strength)
                    }
                    for i in report.interactions
                ],
                'analysis_timestamp': report.analysis_timestamp,
                'recommendations': report.recommendations
            }

            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)

            print(f"✅ Saved correlation report to {output_path}")

        return report

    def save_correlation_matrix(self, output_path: Path):
        """Save correlation matrix to file"""
        if not self.is_fitted:
            raise RuntimeError("Analyzer not fitted. Call fit() first.")

        np.save(output_path, self.correlation_matrix)
        print(f"✅ Saved correlation matrix to {output_path}")

    def load_correlation_matrix(self, matrix_path: Path, feature_names: List[str]):
        """Load pre-computed correlation matrix"""
        self.correlation_matrix = np.load(matrix_path)
        self.feature_names = feature_names
        self.is_fitted = True
        print(f"✅ Loaded correlation matrix from {matrix_path}")


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_correlation_analysis(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    output_dir: Optional[Path] = None
) -> CorrelationAnalysisReport:
    """
    Perform quick correlation analysis with default settings.

    Args:
        feature_matrix: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        output_dir: Directory to save outputs

    Returns:
        CorrelationAnalysisReport
    """
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.fit(feature_matrix, feature_names)

    report = analyzer.generate_report()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save visualizations
        analyzer.visualize_correlation_matrix(
            output_path=output_dir / 'correlation_heatmap.png'
        )
        analyzer.plot_dendrogram(
            output_path=output_dir / 'feature_dendrogram.png'
        )

        # Save report
        analyzer.generate_report(
            output_path=output_dir / 'correlation_report.json'
        )

        # Save correlation matrix
        analyzer.save_correlation_matrix(
            output_path=output_dir / 'correlation_matrix.npy'
        )

    return report


def find_best_features_for_parameter(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    parameter_values: np.ndarray,
    parameter_name: str,
    max_features: int = 100
) -> FeatureSubset:
    """
    Find best feature subset for predicting a parameter.

    Args:
        feature_matrix: Feature matrix
        feature_names: Feature names
        parameter_values: Target parameter values
        parameter_name: Parameter name
        max_features: Maximum features to select

    Returns:
        FeatureSubset
    """
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.fit(feature_matrix, feature_names)

    subset = analyzer.suggest_feature_subset(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        max_features=max_features,
        method='mutual_info' if HAS_SKLEARN else 'correlation'
    )

    return subset


# ============================================================================
# Module Interface
# ============================================================================

__all__ = [
    'FeatureCorrelationAnalyzer',
    'CorrelationResult',
    'RedundantFeaturePair',
    'FeatureSubset',
    'FeatureInteraction',
    'CorrelationAnalysisReport',
    'quick_correlation_analysis',
    'find_best_features_for_parameter',
]


if __name__ == "__main__":
    print("=" * 80)
    print("FEATURE CORRELATION ANALYZER - AGENT 25")
    print("=" * 80)

    # Demo with synthetic data
    print("\n📊 Creating synthetic feature data for demonstration...")
    n_samples = 1000
    n_features = 100  # Subset for demo

    # Create correlated features
    np.random.seed(42)
    base_features = np.random.randn(n_samples, 50)
    redundant_features = base_features + np.random.randn(n_samples, 50) * 0.1
    feature_matrix = np.column_stack([base_features, redundant_features])

    feature_names = [f"feature_{i:03d}" for i in range(n_features)]

    # Initialize analyzer
    print("\n🔧 Initializing Feature Correlation Analyzer...")
    analyzer = FeatureCorrelationAnalyzer(
        correlation_method='pearson',
        redundancy_threshold=0.95,
        interaction_threshold=0.3
    )

    # Fit analyzer
    analyzer.fit(feature_matrix, feature_names)

    # Identify redundant features
    print("\n🔍 Identifying redundant features...")
    redundant = analyzer.identify_redundant_features()
    print(f"   Found {len(redundant)} redundant pairs")

    # Find feature interactions
    print("\n🔗 Analyzing feature interactions...")
    interactions = analyzer.analyze_feature_interactions()
    print(f"   Found {len(interactions)} feature interactions")

    # Suggest feature subset
    print("\n✨ Suggesting feature subset for parameter...")
    target = np.random.randn(n_samples)
    subset = analyzer.suggest_feature_subset(
        parameter_name='demo_parameter',
        parameter_values=target,
        max_features=20,
        method='correlation'
    )
    print(f"   Selected {subset.n_features} features")

    # Generate report
    print("\n📝 Generating correlation analysis report...")
    report = analyzer.generate_report()
    print(f"   Total features analyzed: {report.total_features}")
    print(f"   Redundant pairs: {len(report.redundant_pairs)}")
    print(f"   Recommendations: {len(report.recommendations)}")

    print("\n✅ Agent 25 demonstration complete!")
    print("\nFor production use:")
    print("  from midi_generator.analysis import FeatureCorrelationAnalyzer")
    print("  from midi_generator.synthesis import extract_features")
    print()
    print("  # Extract features from MIDI")
    print("  features = extract_features('song.mid')")
    print()
    print("  # Analyze correlations")
    print("  analyzer = FeatureCorrelationAnalyzer()")
    print("  analyzer.fit(features, feature_names)")
    print("  redundant = analyzer.identify_redundant_features()")
