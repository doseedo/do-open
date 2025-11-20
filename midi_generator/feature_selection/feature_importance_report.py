"""
Feature Importance Analysis and Reporting - Agent 04
====================================================

Generates comprehensive feature importance reports showing:
- Top features per parameter
- Feature selection method comparisons
- Category-wise feature importance
- Correlation analysis
- Performance impact analysis

Author: Agent 04 - Feature Selection Optimizer
License: MIT
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FeatureImportanceReport:
    """Comprehensive feature importance report"""
    report_title: str
    generation_date: str
    summary: Dict[str, Any]
    method_comparisons: Dict[str, Dict[str, Any]]
    top_features_overall: List[Tuple[str, float]]
    category_analysis: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureImportanceAnalyzer:
    """
    Analyzes and reports on feature importance across different selection methods.

    This class:
    1. Aggregates results from multiple feature selection methods
    2. Identifies most important features across methods
    3. Analyzes feature importance by category
    4. Generates comprehensive markdown reports
    5. Creates visualizations of importance scores

    Usage:
        analyzer = FeatureImportanceAnalyzer()
        analyzer.add_method_result('xgboost', result_xgb)
        analyzer.add_method_result('lasso', result_lasso)

        report = analyzer.generate_report()
        analyzer.save_report(report, 'feature_importance_report.md')
    """

    def __init__(self):
        """Initialize analyzer"""
        self.method_results: Dict[str, Dict] = {}
        self.feature_scores_by_method: Dict[str, Dict[str, float]] = {}
        self.selected_features_by_method: Dict[str, List[str]] = {}

    def add_method_result(
        self,
        method_name: str,
        result: Dict[str, Any]
    ):
        """
        Add feature selection result from a method.

        Args:
            method_name: Name of the method
            result: Result dictionary with 'selected_features' and 'feature_scores'
        """
        self.method_results[method_name] = result
        self.selected_features_by_method[method_name] = result.get('selected_features', [])
        self.feature_scores_by_method[method_name] = result.get('feature_scores', {})

        print(f"✅ Added results from method: {method_name}")

    def compute_overall_importance(
        self,
        aggregation: str = 'mean'
    ) -> Dict[str, float]:
        """
        Compute overall feature importance across all methods.

        Args:
            aggregation: 'mean', 'max', 'median'

        Returns:
            Dictionary of feature -> importance score
        """
        all_features = set()
        for scores in self.feature_scores_by_method.values():
            all_features.update(scores.keys())

        overall_scores = {}

        for feature in all_features:
            scores = []
            for method_scores in self.feature_scores_by_method.values():
                if feature in method_scores:
                    # Normalize score to [0, 1]
                    max_score = max(method_scores.values()) if method_scores.values() else 1.0
                    normalized = method_scores[feature] / max_score if max_score > 0 else 0.0
                    scores.append(normalized)

            if scores:
                if aggregation == 'mean':
                    overall_scores[feature] = np.mean(scores)
                elif aggregation == 'max':
                    overall_scores[feature] = np.max(scores)
                elif aggregation == 'median':
                    overall_scores[feature] = np.median(scores)

        return overall_scores

    def get_top_features(
        self,
        n: int = 20,
        aggregation: str = 'mean'
    ) -> List[Tuple[str, float]]:
        """
        Get top N most important features overall.

        Args:
            n: Number of top features
            aggregation: Aggregation method

        Returns:
            List of (feature_name, importance_score) tuples
        """
        overall_scores = self.compute_overall_importance(aggregation)
        sorted_features = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def analyze_feature_categories(
        self,
        aggregation: str = 'mean'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze feature importance by category.

        Returns:
            Dictionary of category -> analysis
        """
        overall_scores = self.compute_overall_importance(aggregation)

        categories = defaultdict(list)
        for feature, score in overall_scores.items():
            category = self._infer_category(feature)
            categories[category].append((feature, score))

        analysis = {}
        for category, features_scores in categories.items():
            scores = [s for f, s in features_scores]
            analysis[category] = {
                'n_features': len(features_scores),
                'mean_importance': float(np.mean(scores)),
                'max_importance': float(np.max(scores)),
                'top_features': sorted(features_scores, key=lambda x: x[1], reverse=True)[:10]
            }

        return analysis

    def _infer_category(self, feature_name: str) -> str:
        """Infer feature category from name"""
        fname_lower = feature_name.lower()

        if any(kw in fname_lower for kw in ['chord', 'harmony', 'voicing', 'progression', 'voice']):
            return 'harmony'
        elif any(kw in fname_lower for kw in ['melody', 'melodic', 'interval', 'contour', 'pitch']):
            return 'melody'
        elif any(kw in fname_lower for kw in ['rhythm', 'beat', 'tempo', 'syncopation', 'groove', 'swing']):
            return 'rhythm'
        elif any(kw in fname_lower for kw in ['dynamic', 'velocity', 'accent', 'articulation']):
            return 'dynamics'
        elif any(kw in fname_lower for kw in ['texture', 'density', 'polyphony', 'layer']):
            return 'texture'
        elif any(kw in fname_lower for kw in ['structure', 'form', 'section', 'repetition']):
            return 'structure'
        else:
            return 'other'

    def compare_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare feature selection methods.

        Returns:
            Dictionary of method -> comparison metrics
        """
        comparisons = {}

        all_selected = set()
        for features in self.selected_features_by_method.values():
            all_selected.update(features)

        for method_name, selected_features in self.selected_features_by_method.items():
            selected_set = set(selected_features)

            # Compute overlap with each other method
            overlaps = {}
            for other_method, other_features in self.selected_features_by_method.items():
                if other_method != method_name:
                    other_set = set(other_features)
                    overlap = len(selected_set & other_set)
                    overlap_ratio = overlap / len(selected_set) if selected_set else 0
                    overlaps[other_method] = {
                        'overlap_count': overlap,
                        'overlap_ratio': float(overlap_ratio)
                    }

            # Average scores for selected features
            if method_name in self.feature_scores_by_method:
                scores = self.feature_scores_by_method[method_name]
                selected_scores = [scores.get(f, 0.0) for f in selected_features if f in scores]
                avg_score = float(np.mean(selected_scores)) if selected_scores else 0.0
            else:
                avg_score = 0.0

            comparisons[method_name] = {
                'n_features': len(selected_features),
                'unique_features': len(selected_set - (all_selected - selected_set)),
                'avg_score': avg_score,
                'overlaps': overlaps
            }

        return comparisons

    def generate_recommendations(
        self,
        top_features: List[Tuple[str, float]],
        category_analysis: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Check category balance
        category_counts = {cat: info['n_features'] for cat, info in category_analysis.items()}
        total = sum(category_counts.values())

        if category_counts.get('harmony', 0) / total < 0.15:
            recommendations.append(
                "Consider including more harmony features - they are typically important for music generation"
            )

        if category_counts.get('rhythm', 0) / total < 0.15:
            recommendations.append(
                "Consider including more rhythm features - they are critical for timing and groove"
            )

        # Check for very low importance features
        low_importance = [f for f, s in top_features[-20:] if s < 0.1]
        if len(low_importance) > 10:
            recommendations.append(
                f"Consider removing {len(low_importance)} features with very low importance (< 0.1)"
            )

        # Check method agreement
        if len(self.method_results) >= 3:
            recommendations.append(
                f"Feature selection based on {len(self.method_results)} methods - good ensemble diversity"
            )

        return recommendations

    def generate_report(
        self,
        report_title: str = "Feature Importance Analysis",
        aggregation: str = 'mean'
    ) -> FeatureImportanceReport:
        """
        Generate comprehensive feature importance report.

        Args:
            report_title: Title for the report
            aggregation: Aggregation method for overall importance

        Returns:
            FeatureImportanceReport
        """
        print("\n📊 Generating feature importance report...")

        # Compute top features
        top_features = self.get_top_features(n=50, aggregation=aggregation)

        # Analyze categories
        category_analysis = self.analyze_feature_categories(aggregation=aggregation)

        # Compare methods
        method_comparisons = self.compare_methods()

        # Generate recommendations
        recommendations = self.generate_recommendations(top_features, category_analysis)

        # Summary statistics
        all_features = set()
        for features in self.selected_features_by_method.values():
            all_features.update(features)

        summary = {
            'n_methods': len(self.method_results),
            'n_features_total': len(all_features),
            'n_categories': len(category_analysis),
            'aggregation_method': aggregation
        }

        report = FeatureImportanceReport(
            report_title=report_title,
            generation_date=datetime.now().isoformat(),
            summary=summary,
            method_comparisons=method_comparisons,
            top_features_overall=top_features,
            category_analysis=category_analysis,
            recommendations=recommendations
        )

        print("✅ Report generated successfully!")

        return report

    def save_report(
        self,
        report: FeatureImportanceReport,
        output_path: Path,
        format: str = 'markdown'
    ):
        """
        Save report to file.

        Args:
            report: FeatureImportanceReport instance
            output_path: Output file path
            format: 'markdown' or 'json'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'markdown':
            self._save_markdown_report(report, output_path)
        elif format == 'json':
            self._save_json_report(report, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"✅ Saved report to {output_path}")

    def _save_markdown_report(
        self,
        report: FeatureImportanceReport,
        output_path: Path
    ):
        """Save report as Markdown"""
        lines = []

        # Title and metadata
        lines.append(f"# {report.report_title}\n")
        lines.append(f"**Generated:** {report.generation_date}\n")
        lines.append(f"**Agent:** Agent 04 - Feature Selection Optimizer\n")
        lines.append("\n---\n")

        # Summary
        lines.append("## Summary\n")
        lines.append(f"- **Methods analyzed:** {report.summary['n_methods']}")
        lines.append(f"- **Total unique features:** {report.summary['n_features_total']}")
        lines.append(f"- **Feature categories:** {report.summary['n_categories']}")
        lines.append(f"- **Aggregation method:** {report.summary['aggregation_method']}")
        lines.append("\n")

        # Top features
        lines.append("## Top 50 Most Important Features\n")
        lines.append("| Rank | Feature Name | Importance Score |")
        lines.append("|------|--------------|------------------|")
        for i, (feature, score) in enumerate(report.top_features_overall[:50], 1):
            lines.append(f"| {i} | `{feature}` | {score:.4f} |")
        lines.append("\n")

        # Category analysis
        lines.append("## Feature Importance by Category\n")
        for category in ['harmony', 'melody', 'rhythm', 'dynamics', 'texture', 'structure', 'other']:
            if category in report.category_analysis:
                info = report.category_analysis[category]
                lines.append(f"### {category.capitalize()}\n")
                lines.append(f"- **Total features:** {info['n_features']}")
                lines.append(f"- **Mean importance:** {info['mean_importance']:.4f}")
                lines.append(f"- **Max importance:** {info['max_importance']:.4f}\n")

                lines.append("**Top 10 features:**")
                for j, (feat, score) in enumerate(info['top_features'][:10], 1):
                    lines.append(f"{j}. `{feat}` ({score:.4f})")
                lines.append("\n")

        # Method comparisons
        lines.append("## Method Comparisons\n")
        for method_name, comparison in report.method_comparisons.items():
            lines.append(f"### {method_name}\n")
            lines.append(f"- **Features selected:** {comparison['n_features']}")
            lines.append(f"- **Unique features:** {comparison['unique_features']}")
            lines.append(f"- **Average score:** {comparison['avg_score']:.4f}\n")

            if comparison['overlaps']:
                lines.append("**Overlap with other methods:**")
                for other_method, overlap_info in comparison['overlaps'].items():
                    overlap_pct = overlap_info['overlap_ratio'] * 100
                    lines.append(f"- {other_method}: {overlap_info['overlap_count']} features ({overlap_pct:.1f}%)")
                lines.append("\n")

        # Recommendations
        lines.append("## Recommendations\n")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("\n")

        # Footer
        lines.append("---\n")
        lines.append("*Report generated by Agent 04: Feature Selection Optimizer*\n")

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def _save_json_report(
        self,
        report: FeatureImportanceReport,
        output_path: Path
    ):
        """Save report as JSON"""
        data = {
            'report_title': report.report_title,
            'generation_date': report.generation_date,
            'summary': report.summary,
            'method_comparisons': report.method_comparisons,
            'top_features_overall': [
                {'feature': f, 'importance': s}
                for f, s in report.top_features_overall
            ],
            'category_analysis': {
                cat: {
                    'n_features': info['n_features'],
                    'mean_importance': info['mean_importance'],
                    'max_importance': info['max_importance'],
                    'top_features': [
                        {'feature': f, 'importance': s}
                        for f, s in info['top_features']
                    ]
                }
                for cat, info in report.category_analysis.items()
            },
            'recommendations': report.recommendations,
            'metadata': report.metadata
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_importance_report_from_selection_results(
    results_dir: Path,
    output_path: Path,
    format: str = 'markdown'
):
    """
    Generate importance report from directory of selection results.

    Args:
        results_dir: Directory containing selection result JSON files
        output_path: Output report path
        format: Output format ('markdown' or 'json')
    """
    results_dir = Path(results_dir)

    analyzer = FeatureImportanceAnalyzer()

    # Load all result files
    for result_file in results_dir.glob('*_features.json'):
        method_name = result_file.stem.replace('_features', '')

        with open(result_file, 'r') as f:
            result = json.load(f)

        analyzer.add_method_result(method_name, result)

    # Generate and save report
    report = analyzer.generate_report()
    analyzer.save_report(report, output_path, format=format)

    return report


if __name__ == "__main__":
    print("="*70)
    print("FEATURE IMPORTANCE ANALYZER - AGENT 04")
    print("="*70)

    print("\nExample usage:")
    print()
    print("  # Create analyzer")
    print("  analyzer = FeatureImportanceAnalyzer()")
    print()
    print("  # Add method results")
    print("  analyzer.add_method_result('xgboost', xgb_result)")
    print("  analyzer.add_method_result('lasso', lasso_result)")
    print("  analyzer.add_method_result('rfe', rfe_result)")
    print()
    print("  # Generate report")
    print("  report = analyzer.generate_report()")
    print()
    print("  # Save as markdown")
    print("  analyzer.save_report(report, 'feature_importance.md', format='markdown')")
    print()
    print("  # Or generate from directory")
    print("  generate_importance_report_from_selection_results(")
    print("      results_dir='output/',")
    print("      output_path='importance_report.md'")
    print("  )")

    print("\n✅ Feature Importance Analyzer ready!")
