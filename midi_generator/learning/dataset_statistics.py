#!/usr/bin/env python3
"""
Dataset Statistics and Visualization Dashboard
Agent 03: Metadata & Labeling Manager

Comprehensive statistics and visualizations for labeled MIDI datasets.

Features:
    - Parameter distribution analysis
    - Genre-wise statistics
    - Correlation analysis
    - Inter-rater reliability metrics
    - Quality assessment
    - Visualization generation

Author: Agent 03 - Metadata & Labeling Manager
License: MIT
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass

# Optional imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("WARNING: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Import dataset utilities
try:
    from midi_generator.learning.dataset_utils import LabeledDatasetEntry, LabeledDatasetLoader
    DATASET_UTILS_AVAILABLE = True
except ImportError:
    DATASET_UTILS_AVAILABLE = False
    print("WARNING: dataset_utils not available")


# ==============================================================================
# STATISTICS CALCULATOR
# ==============================================================================

class DatasetStatistics:
    """Calculate comprehensive statistics for labeled datasets."""

    def __init__(self, entries: List[LabeledDatasetEntry]):
        """Initialize with dataset entries."""
        self.entries = entries
        self.stats = {}

    def compute_all_statistics(self) -> Dict[str, Any]:
        """Compute all statistics."""
        print("Computing dataset statistics...")

        self.stats['basic'] = self._compute_basic_stats()
        self.stats['parameters'] = self._compute_parameter_stats()
        self.stats['genres'] = self._compute_genre_stats()
        self.stats['correlations'] = self._compute_correlations()
        self.stats['quality'] = self._compute_quality_stats()

        print("✓ Statistics computed")

        return self.stats

    def _compute_basic_stats(self) -> Dict:
        """Compute basic dataset statistics."""
        stats = {
            'total_files': len(self.entries),
            'auto_labeled': sum(1 for e in self.entries if e.auto_labeled),
            'manually_labeled': sum(1 for e in self.entries if e.manually_labeled),
            'flagged': sum(1 for e in self.entries if e.flagged)
        }

        # Labeler distribution
        labeler_counts = Counter(e.labeler_id for e in self.entries if e.labeler_id)
        stats['labeler_distribution'] = dict(labeler_counts)

        return stats

    def _compute_parameter_stats(self) -> Dict:
        """Compute statistics for each parameter."""
        param_stats = {}

        # Collect all parameters
        all_params = set()
        for entry in self.entries:
            all_params.update(entry.get_all_labels_flat().keys())

        # Compute stats for each parameter
        for param in sorted(all_params):
            values = []
            for entry in self.entries:
                value = entry.get_all_labels_flat().get(param)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)

            if values:
                param_stats[param] = {
                    'count': len(values),
                    'missing': len(self.entries) - len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75))
                }
            else:
                # Categorical parameter
                cat_values = []
                for entry in self.entries:
                    value = entry.get_all_labels_flat().get(param)
                    if value is not None:
                        cat_values.append(value)

                if cat_values:
                    value_counts = Counter(cat_values)
                    param_stats[param] = {
                        'type': 'categorical',
                        'count': len(cat_values),
                        'missing': len(self.entries) - len(cat_values),
                        'unique_values': len(value_counts),
                        'distribution': dict(value_counts)
                    }

        return param_stats

    def _compute_genre_stats(self) -> Dict:
        """Compute per-genre statistics."""
        # Group by genre
        genre_groups = defaultdict(list)
        for entry in self.entries:
            genre = entry.level1_labels.get('genre.primary', 'unknown')
            genre_groups[genre].append(entry)

        genre_stats = {}
        for genre, genre_entries in genre_groups.items():
            stats = {
                'count': len(genre_entries),
                'percentage': (len(genre_entries) / len(self.entries)) * 100
            }

            # Average parameter values for this genre
            param_averages = {}
            all_params = set()
            for entry in genre_entries:
                all_params.update(entry.get_all_labels_flat().keys())

            for param in all_params:
                values = []
                for entry in genre_entries:
                    value = entry.get_all_labels_flat().get(param)
                    if value is not None and isinstance(value, (int, float)):
                        values.append(value)

                if values:
                    param_averages[param] = float(np.mean(values))

            stats['parameter_averages'] = param_averages

            genre_stats[genre] = stats

        return genre_stats

    def _compute_correlations(self) -> Dict:
        """Compute parameter correlations."""
        # Collect all continuous parameters
        continuous_params = []
        for entry in self.entries:
            for param, value in entry.get_all_labels_flat().items():
                if isinstance(value, (int, float)) and param not in ['tempo.bpm']:
                    continuous_params.append(param)

        continuous_params = sorted(set(continuous_params))

        # Build data matrix
        data_matrix = []
        for entry in self.entries:
            row = []
            for param in continuous_params:
                value = entry.get_all_labels_flat().get(param)
                if value is not None and isinstance(value, (int, float)):
                    row.append(value)
                else:
                    row.append(np.nan)
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        # Compute correlation matrix (ignoring NaNs)
        correlation_matrix = np.full((len(continuous_params), len(continuous_params)), np.nan)

        for i in range(len(continuous_params)):
            for j in range(len(continuous_params)):
                # Get valid pairs
                valid_mask = ~(np.isnan(data_matrix[:, i]) | np.isnan(data_matrix[:, j]))
                if valid_mask.sum() > 10:  # Need at least 10 pairs
                    corr = np.corrcoef(data_matrix[valid_mask, i], data_matrix[valid_mask, j])[0, 1]
                    correlation_matrix[i, j] = corr

        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(continuous_params)):
            for j in range(i + 1, len(continuous_params)):
                corr = correlation_matrix[i, j]
                if not np.isnan(corr) and abs(corr) > 0.7:
                    high_correlations.append({
                        'param1': continuous_params[i],
                        'param2': continuous_params[j],
                        'correlation': float(corr)
                    })

        return {
            'parameters': continuous_params,
            'correlation_matrix': correlation_matrix.tolist(),
            'high_correlations': sorted(high_correlations, key=lambda x: abs(x['correlation']), reverse=True)
        }

    def _compute_quality_stats(self) -> Dict:
        """Compute quality-related statistics."""
        stats = {}

        # Quality scores
        quality_scores = [e.quality_score for e in self.entries if e.quality_score is not None]
        if quality_scores:
            stats['quality_scores'] = {
                'mean': float(np.mean(quality_scores)),
                'std': float(np.std(quality_scores)),
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores))
            }

        # Missing value analysis
        missing_by_param = defaultdict(int)
        for entry in self.entries:
            for param, value in entry.get_all_labels_flat().items():
                if value is None:
                    missing_by_param[param] += 1

        stats['missing_values'] = {
            param: {
                'count': count,
                'percentage': (count / len(self.entries)) * 100
            }
            for param, count in missing_by_param.items()
        }

        # Files with notes/flags
        stats['files_with_notes'] = sum(1 for e in self.entries if e.notes)
        stats['files_flagged'] = sum(1 for e in self.entries if e.flagged)

        return stats

    def save_report(self, output_file: Path):
        """Save statistics report to JSON."""
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"✓ Statistics saved to {output_file}")

    def print_summary(self):
        """Print summary of statistics."""
        print("\n" + "=" * 80)
        print("DATASET STATISTICS SUMMARY")
        print("=" * 80)

        # Basic stats
        basic = self.stats.get('basic', {})
        print(f"\nTotal Files: {basic.get('total_files', 0)}")
        print(f"  Auto-labeled: {basic.get('auto_labeled', 0)}")
        print(f"  Manually-labeled: {basic.get('manually_labeled', 0)}")
        print(f"  Flagged: {basic.get('flagged', 0)}")

        # Genre distribution
        genres = self.stats.get('genres', {})
        print("\nGenre Distribution:")
        for genre, stats in sorted(genres.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {genre}: {stats['count']} ({stats['percentage']:.1f}%)")

        # High correlations
        corr = self.stats.get('correlations', {})
        high_corr = corr.get('high_correlations', [])[:5]
        if high_corr:
            print("\nTop Correlated Parameters:")
            for item in high_corr:
                print(f"  {item['param1']} <-> {item['param2']}: {item['correlation']:.3f}")

        # Quality
        quality = self.stats.get('quality', {})
        if 'quality_scores' in quality:
            qs = quality['quality_scores']
            print(f"\nQuality Scores: mean={qs['mean']:.2f}, std={qs['std']:.2f}")

        print("\n" + "=" * 80)


# ==============================================================================
# VISUALIZATION GENERATOR
# ==============================================================================

class DatasetVisualizer:
    """Generate visualizations for dataset analysis."""

    def __init__(self, entries: List[LabeledDatasetEntry], output_dir: Path):
        """Initialize visualizer."""
        self.entries = entries
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if PLOTTING_AVAILABLE:
            sns.set_style('whitegrid')
            sns.set_palette('husl')

    def generate_all_plots(self):
        """Generate all visualizations."""
        if not PLOTTING_AVAILABLE:
            print("ERROR: Plotting not available. Install matplotlib and seaborn.")
            return

        print("Generating visualizations...")

        self.plot_genre_distribution()
        self.plot_parameter_distributions()
        self.plot_correlation_heatmap()
        self.plot_genre_comparison()
        self.plot_missing_values()

        print(f"✓ Visualizations saved to {self.output_dir}")

    def plot_genre_distribution(self):
        """Plot genre distribution pie chart."""
        genre_counts = Counter(e.level1_labels.get('genre.primary', 'unknown') for e in self.entries)

        plt.figure(figsize=(10, 6))
        plt.pie(genre_counts.values(), labels=genre_counts.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('Genre Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'genre_distribution.png', dpi=150)
        plt.close()

        print("  ✓ Genre distribution plot")

    def plot_parameter_distributions(self):
        """Plot distributions for key continuous parameters."""
        # Select key parameters
        key_params = [
            'energy.level', 'complexity.overall', 'harmony.tension',
            'melody.note_density', 'rhythm.syncopation', 'dynamics.overall_level'
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, param in enumerate(key_params):
            values = []
            for entry in self.entries:
                value = entry.get_all_labels_flat().get(param)
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)

            if values:
                axes[idx].hist(values, bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(param, fontweight='bold')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.2f}')
                axes[idx].legend()

        plt.suptitle('Parameter Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_distributions.png', dpi=150)
        plt.close()

        print("  ✓ Parameter distributions plot")

    def plot_correlation_heatmap(self):
        """Plot correlation heatmap."""
        # Get continuous parameters
        continuous_params = [
            'energy.level', 'complexity.overall', 'harmony.chord_density',
            'harmony.complexity', 'harmony.chromaticism', 'melody.note_density',
            'rhythm.syncopation', 'dynamics.overall_level'
        ]

        # Build data matrix
        data_matrix = []
        for entry in self.entries:
            row = []
            for param in continuous_params:
                value = entry.get_all_labels_flat().get(param)
                if value is not None and isinstance(value, (int, float)):
                    row.append(value)
                else:
                    row.append(np.nan)
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        # Compute correlation matrix
        correlation_matrix = np.full((len(continuous_params), len(continuous_params)), np.nan)

        for i in range(len(continuous_params)):
            for j in range(len(continuous_params)):
                valid_mask = ~(np.isnan(data_matrix[:, i]) | np.isnan(data_matrix[:, j]))
                if valid_mask.sum() > 10:
                    corr = np.corrcoef(data_matrix[valid_mask, i], data_matrix[valid_mask, j])[0, 1]
                    correlation_matrix[i, j] = corr

        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=[p.split('.')[-1] for p in continuous_params],
                   yticklabels=[p.split('.')[-1] for p in continuous_params],
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title('Parameter Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=150)
        plt.close()

        print("  ✓ Correlation heatmap")

    def plot_genre_comparison(self):
        """Plot parameter comparison across genres."""
        # Group by genre
        genre_groups = defaultdict(list)
        for entry in self.entries:
            genre = entry.level1_labels.get('genre.primary', 'unknown')
            genre_groups[genre].append(entry)

        # Compare key parameters
        key_params = ['energy.level', 'complexity.overall', 'harmony.chord_density']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, param in enumerate(key_params):
            genre_values = {}
            for genre, entries in genre_groups.items():
                values = []
                for entry in entries:
                    value = entry.get_all_labels_flat().get(param)
                    if value is not None and isinstance(value, (int, float)):
                        values.append(value)
                if values:
                    genre_values[genre] = values

            # Box plot
            if genre_values:
                axes[idx].boxplot(genre_values.values(), labels=genre_values.keys())
                axes[idx].set_title(param, fontweight='bold')
                axes[idx].set_ylabel('Value')
                axes[idx].tick_params(axis='x', rotation=45)

        plt.suptitle('Parameter Comparison Across Genres', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'genre_comparison.png', dpi=150)
        plt.close()

        print("  ✓ Genre comparison plot")

    def plot_missing_values(self):
        """Plot missing value analysis."""
        # Count missing values per parameter
        missing_counts = defaultdict(int)
        for entry in self.entries:
            for param, value in entry.get_all_labels_flat().items():
                if value is None:
                    missing_counts[param] += 1

        if not missing_counts:
            print("  ⊘ No missing values to plot")
            return

        # Sort by count
        sorted_params = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        params = [p[0].split('.')[-1] for p in sorted_params]
        counts = [p[1] for p in sorted_params]
        percentages = [(c / len(self.entries)) * 100 for c in counts]

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.barh(params, percentages, edgecolor='black')
        plt.xlabel('Missing Values (%)', fontsize=12)
        plt.ylabel('Parameter', fontsize=12)
        plt.title('Missing Values by Parameter', fontsize=16, fontweight='bold')
        plt.xlim(0, 100)

        # Color bars by severity
        for bar, pct in zip(bars, percentages):
            if pct > 50:
                bar.set_color('red')
            elif pct > 20:
                bar.set_color('orange')
            else:
                bar.set_color('green')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_values.png', dpi=150)
        plt.close()

        print("  ✓ Missing values plot")


# ==============================================================================
# INTER-RATER RELIABILITY
# ==============================================================================

class InterRaterReliability:
    """Calculate inter-rater reliability metrics."""

    @staticmethod
    def calculate_agreement(
            labels1: List[float],
            labels2: List[float],
            tolerance: float = 0.15
    ) -> Dict[str, float]:
        """
        Calculate agreement between two raters.

        Args:
            labels1: Labels from rater 1
            labels2: Labels from rater 2
            tolerance: Tolerance for agreement on continuous scale

        Returns:
            Dictionary with agreement metrics
        """
        assert len(labels1) == len(labels2), "Label lists must have same length"

        # Mean Absolute Difference
        mad = np.mean(np.abs(np.array(labels1) - np.array(labels2)))

        # Percentage within tolerance
        within_tolerance = np.sum(np.abs(np.array(labels1) - np.array(labels2)) <= tolerance)
        agreement_rate = within_tolerance / len(labels1)

        # Correlation
        correlation = np.corrcoef(labels1, labels2)[0, 1]

        return {
            'mean_absolute_difference': float(mad),
            'agreement_rate': float(agreement_rate),
            'correlation': float(correlation),
            'tolerance': tolerance
        }

    @staticmethod
    def calculate_cohens_kappa(labels1: List, labels2: List) -> float:
        """Calculate Cohen's Kappa for categorical agreement."""
        assert len(labels1) == len(labels2)

        # Observed agreement
        agreements = sum(1 for a, b in zip(labels1, labels2) if a == b)
        po = agreements / len(labels1)

        # Expected agreement
        categories = set(labels1 + labels2)
        pe = 0
        for cat in categories:
            p1 = labels1.count(cat) / len(labels1)
            p2 = labels2.count(cat) / len(labels2)
            pe += p1 * p2

        # Kappa
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

        return kappa


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_statistics.py <dataset.json> [output_dir]")
        sys.exit(1)

    dataset_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('dataset_analysis')

    # Load dataset
    if DATASET_UTILS_AVAILABLE:
        entries = LabeledDatasetLoader.load_from_json(dataset_file)
    else:
        print("ERROR: dataset_utils not available")
        sys.exit(1)

    # Compute statistics
    stats_calc = DatasetStatistics(entries)
    stats = stats_calc.compute_all_statistics()
    stats_calc.print_summary()

    # Save report
    stats_calc.save_report(output_dir / 'statistics_report.json')

    # Generate visualizations
    if PLOTTING_AVAILABLE:
        visualizer = DatasetVisualizer(entries, output_dir)
        visualizer.generate_all_plots()
