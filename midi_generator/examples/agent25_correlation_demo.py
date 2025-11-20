"""
Agent 25: Feature Correlation Analyzer - Demonstration
=======================================================

This script demonstrates the Feature Correlation Analyzer, which analyzes
correlations between the 1,000 musical features extracted by Agent 8 to
optimize model training.

Key Features Demonstrated:
1. Feature correlation matrix computation
2. Redundant feature identification
3. Feature subset suggestion for parameters
4. Feature interaction detection
5. Visualization generation (heatmaps, dendrograms)
6. Comprehensive correlation reports

Author: Agent 25 - Feature Correlation Analyzer
License: MIT
"""

import sys
from pathlib import Path
import numpy as np
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_generator.analysis import (
    FeatureCorrelationAnalyzer,
    quick_correlation_analysis,
    find_best_features_for_parameter
)
from midi_generator.synthesis import extract_features, DeepFeatureExtractor


def demo_basic_correlation_analysis():
    """Demo 1: Basic correlation analysis with synthetic data"""
    print("=" * 80)
    print("DEMO 1: Basic Correlation Analysis")
    print("=" * 80)

    # Create synthetic feature matrix
    print("\n📊 Creating synthetic feature data...")
    n_samples = 1000
    n_features = 200

    np.random.seed(42)

    # Create base features
    base_features = np.random.randn(n_samples, 100)

    # Add some redundant features (highly correlated)
    redundant_features = base_features[:, :50] + np.random.randn(n_samples, 50) * 0.05

    # Add some independent features
    independent_features = np.random.randn(n_samples, 50)

    # Combine all features
    feature_matrix = np.column_stack([base_features, redundant_features, independent_features])
    feature_names = [f"feature_{i:03d}" for i in range(n_features)]

    # Initialize analyzer
    print("\n🔧 Initializing Feature Correlation Analyzer...")
    analyzer = FeatureCorrelationAnalyzer(
        correlation_method='pearson',
        redundancy_threshold=0.95,
        interaction_threshold=0.3
    )

    # Fit analyzer
    print("\n⚙️ Fitting analyzer on feature matrix...")
    analyzer.fit(feature_matrix, feature_names)

    # Identify redundant features
    print("\n🔍 Identifying redundant features...")
    redundant_pairs = analyzer.identify_redundant_features()

    print(f"\n📋 Found {len(redundant_pairs)} redundant feature pairs:")
    for i, pair in enumerate(redundant_pairs[:5]):  # Show first 5
        print(f"   {i+1}. {pair.feature1} <-> {pair.feature2}")
        print(f"      Correlation: {pair.correlation:.4f}")
        print(f"      Recommendation: {pair.recommendation}")

    if len(redundant_pairs) > 5:
        print(f"   ... and {len(redundant_pairs) - 5} more")

    # Find uncorrelated features
    print("\n🎯 Finding uncorrelated features...")
    uncorrelated = analyzer.get_uncorrelated_features(max_avg_correlation=0.3)
    print(f"   Found {len(uncorrelated)} relatively uncorrelated features")

    print("\n✅ Demo 1 complete!")


def demo_feature_subset_suggestion():
    """Demo 2: Suggest optimal feature subsets for parameters"""
    print("\n" + "=" * 80)
    print("DEMO 2: Feature Subset Suggestion for Parameters")
    print("=" * 80)

    # Create feature matrix
    print("\n📊 Creating feature matrix and target parameter...")
    n_samples = 500
    n_features = 150

    np.random.seed(123)
    feature_matrix = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i:03d}" for i in range(n_features)]

    # Create target parameter (influenced by some features)
    important_features = [0, 5, 10, 20, 35, 50, 75, 100]
    target_param = np.sum(feature_matrix[:, important_features], axis=1) + \
                   np.random.randn(n_samples) * 0.5

    # Initialize and fit analyzer
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.fit(feature_matrix, feature_names)

    # Suggest feature subset
    print("\n✨ Suggesting optimal feature subset for 'harmony.chord_complexity'...")
    subset = analyzer.suggest_feature_subset(
        parameter_name='harmony.chord_complexity',
        parameter_values=target_param,
        max_features=30,
        method='correlation'
    )

    print(f"\n📋 Selected Feature Subset:")
    print(f"   Parameter: {subset.parameter_name}")
    print(f"   Number of features: {subset.n_features}")
    print(f"   Selection method: {subset.selection_method}")

    print(f"\n   Top 10 most important features:")
    top_features = sorted(
        subset.importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for i, (feature, score) in enumerate(top_features, 1):
        print(f"      {i:2d}. {feature:20s} - importance: {score:.4f}")

    print("\n✅ Demo 2 complete!")


def demo_feature_interactions():
    """Demo 3: Detect feature interactions"""
    print("\n" + "=" * 80)
    print("DEMO 3: Feature Interaction Detection")
    print("=" * 80)

    # Create feature matrix with interactions
    print("\n📊 Creating feature matrix with interactions...")
    n_samples = 800
    n_features = 100

    np.random.seed(456)

    # Base features
    features = np.random.randn(n_samples, 80)

    # Add interaction features
    interaction_1 = features[:, 0] * features[:, 1] + np.random.randn(n_samples) * 0.1
    interaction_2 = features[:, 5] + features[:, 10] + np.random.randn(n_samples) * 0.2
    interaction_features = np.column_stack([
        interaction_1, interaction_2,
        np.random.randn(n_samples, 18)
    ])

    feature_matrix = np.column_stack([features, interaction_features])
    feature_names = [f"feature_{i:03d}" for i in range(n_features)]

    # Initialize and fit
    analyzer = FeatureCorrelationAnalyzer(interaction_threshold=0.25)
    analyzer.fit(feature_matrix, feature_names)

    # Detect interactions
    print("\n🔗 Detecting feature interactions...")
    interactions = analyzer.analyze_feature_interactions()

    print(f"\n📋 Found {len(interactions)} feature interactions:")
    print(f"   (showing top 10)")

    for i, interaction in enumerate(interactions[:10], 1):
        print(f"   {i:2d}. {interaction.feature1:20s} <-> {interaction.feature2:20s}")
        print(f"       Interaction strength: {interaction.interaction_strength:.4f}")

    print("\n✅ Demo 3 complete!")


def demo_feature_clustering():
    """Demo 4: Hierarchical feature clustering"""
    print("\n" + "=" * 80)
    print("DEMO 4: Hierarchical Feature Clustering")
    print("=" * 80)

    # Create feature matrix with clusters
    print("\n📊 Creating feature matrix with natural clusters...")
    n_samples = 600
    np.random.seed(789)

    # Create 5 distinct feature clusters
    cluster1 = np.random.randn(n_samples, 15) + np.random.randn(n_samples, 1)
    cluster2 = np.random.randn(n_samples, 20) + np.random.randn(n_samples, 1) * 2
    cluster3 = np.random.randn(n_samples, 12) + np.random.randn(n_samples, 1) * 0.5
    cluster4 = np.random.randn(n_samples, 18)
    cluster5 = np.random.randn(n_samples, 10) + np.random.randn(n_samples, 1) * 1.5

    feature_matrix = np.column_stack([cluster1, cluster2, cluster3, cluster4, cluster5])
    n_features = feature_matrix.shape[1]
    feature_names = [f"feature_{i:03d}" for i in range(n_features)]

    # Initialize and fit
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.fit(feature_matrix, feature_names)

    # Cluster features
    print("\n🌳 Performing hierarchical clustering...")
    clusters = analyzer.get_feature_clusters(n_clusters=5, linkage_method='average')

    print(f"\n📋 Feature Clusters:")
    for i, cluster in enumerate(clusters, 1):
        print(f"   Cluster {i}: {len(cluster)} features")
        print(f"      {', '.join(cluster[:5])}", end='')
        if len(cluster) > 5:
            print(f", ... and {len(cluster)-5} more")
        else:
            print()

    print("\n✅ Demo 4 complete!")


def demo_correlation_report():
    """Demo 5: Generate comprehensive correlation report"""
    print("\n" + "=" * 80)
    print("DEMO 5: Comprehensive Correlation Report")
    print("=" * 80)

    # Create feature matrix
    print("\n📊 Creating feature matrix...")
    n_samples = 1000
    n_features = 100

    np.random.seed(321)
    feature_matrix = np.random.randn(n_samples, n_features)

    # Add some structure
    feature_matrix[:, 10:20] = feature_matrix[:, 0:10] + \
                               np.random.randn(n_samples, 10) * 0.1

    feature_names = [f"feature_{i:03d}" for i in range(n_features)]

    # Initialize and fit
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.fit(feature_matrix, feature_names)

    # Generate report
    print("\n📝 Generating comprehensive correlation analysis report...")
    report = analyzer.generate_report()

    print(f"\n📋 Correlation Analysis Report:")
    print(f"   Total features: {report.total_features}")
    print(f"   Redundant pairs found: {len(report.redundant_pairs)}")
    print(f"   Feature interactions: {len(report.interactions)}")
    print(f"   Analysis timestamp: {report.analysis_timestamp}")

    print(f"\n   Recommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"      {i}. {rec}")

    print("\n✅ Demo 5 complete!")


def demo_quick_analysis():
    """Demo 6: Quick correlation analysis with outputs"""
    print("\n" + "=" * 80)
    print("DEMO 6: Quick Correlation Analysis (Convenience Function)")
    print("=" * 80)

    # Create feature matrix
    print("\n📊 Creating feature matrix...")
    n_samples = 500
    n_features = 80

    np.random.seed(654)
    feature_matrix = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i:03d}" for i in range(n_features)]

    # Quick analysis
    print("\n⚡ Running quick correlation analysis...")
    output_dir = Path("correlation_analysis_output")

    report = quick_correlation_analysis(
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        output_dir=output_dir
    )

    print(f"\n✅ Analysis complete!")
    print(f"   Outputs saved to: {output_dir}/")
    print(f"   - correlation_heatmap.png")
    print(f"   - feature_dendrogram.png")
    print(f"   - correlation_report.json")
    print(f"   - correlation_matrix.npy")

    print("\n✅ Demo 6 complete!")


def demo_with_agent8_features():
    """Demo 7: Integration with Agent 8 feature extraction"""
    print("\n" + "=" * 80)
    print("DEMO 7: Integration with Agent 8 (Deep Feature Extractor)")
    print("=" * 80)

    # Check if there are any MIDI files to test with
    midi_files = list(Path("midi_generator").glob("*.mid"))[:5]

    if not midi_files:
        print("\n⚠️ No MIDI files found in midi_generator/ directory")
        print("   Skipping Agent 8 integration demo")
        print("   To run this demo, add MIDI files to the midi_generator/ directory")
        return

    print(f"\n📂 Found {len(midi_files)} MIDI files for testing")

    # Extract features from MIDI files
    print("\n🎵 Extracting features from MIDI files using Agent 8...")
    extractor = DeepFeatureExtractor()
    feature_vectors = []

    for midi_file in midi_files:
        try:
            print(f"   Extracting from: {midi_file.name}")
            features = extractor.extract(midi_file)
            feature_vectors.append(features)
        except Exception as e:
            print(f"   ❌ Error extracting from {midi_file.name}: {e}")

    if len(feature_vectors) == 0:
        print("\n⚠️ No features extracted successfully")
        return

    # Stack feature vectors
    feature_matrix = np.vstack(feature_vectors)
    feature_names = extractor.feature_names

    print(f"\n✅ Extracted features:")
    print(f"   Shape: {feature_matrix.shape}")
    print(f"   Samples: {feature_matrix.shape[0]}")
    print(f"   Features: {feature_matrix.shape[1]}")

    # Analyze correlations
    print("\n🔍 Analyzing feature correlations...")
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.fit(feature_matrix, feature_names)

    # Identify redundant features
    redundant = analyzer.identify_redundant_features(threshold=0.95)
    print(f"\n   Found {len(redundant)} redundant feature pairs")

    # Get feature clusters
    clusters = analyzer.get_feature_clusters(n_clusters=10)
    print(f"   Clustered features into {len(clusters)} groups")

    print("\n✅ Demo 7 complete!")


def demo_best_features_for_parameter():
    """Demo 8: Find best features for predicting a parameter"""
    print("\n" + "=" * 80)
    print("DEMO 8: Best Features for Parameter Prediction")
    print("=" * 80)

    # Create feature matrix and target
    print("\n📊 Creating feature matrix and target parameter...")
    n_samples = 1000
    n_features = 200

    np.random.seed(999)
    feature_matrix = np.random.randn(n_samples, n_features)

    # Target influenced by specific features
    influential_features = [5, 15, 25, 50, 75, 100, 125, 150]
    weights = [2.0, 1.5, 1.2, 0.8, 0.6, 0.5, 0.4, 0.3]

    target = np.zeros(n_samples)
    for feat_idx, weight in zip(influential_features, weights):
        target += feature_matrix[:, feat_idx] * weight
    target += np.random.randn(n_samples) * 0.5

    feature_names = [f"feature_{i:03d}" for i in range(n_features)]

    # Find best features using convenience function
    print("\n⚡ Finding best features for 'rhythm.swing_intensity'...")
    subset = find_best_features_for_parameter(
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        parameter_values=target,
        parameter_name='rhythm.swing_intensity',
        max_features=50
    )

    print(f"\n📋 Best Feature Subset:")
    print(f"   Parameter: {subset.parameter_name}")
    print(f"   Features selected: {subset.n_features}")
    print(f"   Selection method: {subset.selection_method}")

    print(f"\n   Top 15 features:")
    top_features = sorted(
        subset.importance_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    for i, (feature, score) in enumerate(top_features, 1):
        print(f"      {i:2d}. {feature:20s} - {score:.6f}")

    print("\n✅ Demo 8 complete!")


def main():
    """Run all demonstrations"""
    print("=" * 80)
    print("AGENT 25: FEATURE CORRELATION ANALYZER - COMPREHENSIVE DEMO")
    print("=" * 80)
    print()
    print("This demonstration showcases the Feature Correlation Analyzer's")
    print("capabilities for analyzing the 1,000 musical features from Agent 8.")
    print()
    print("Topics covered:")
    print("  1. Basic correlation analysis")
    print("  2. Feature subset suggestion")
    print("  3. Feature interaction detection")
    print("  4. Hierarchical clustering")
    print("  5. Comprehensive reports")
    print("  6. Quick analysis")
    print("  7. Agent 8 integration")
    print("  8. Best features for parameters")
    print()

    try:
        demo_basic_correlation_analysis()
        demo_feature_subset_suggestion()
        demo_feature_interactions()
        demo_feature_clustering()
        demo_correlation_report()
        demo_quick_analysis()
        demo_with_agent8_features()
        demo_best_features_for_parameter()

        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETE!")
        print("=" * 80)
        print()
        print("✅ Agent 25 Feature Correlation Analyzer is fully operational")
        print()
        print("Next Steps:")
        print("  1. Use correlation analyzer to optimize feature selection")
        print("  2. Integrate with Agent 9 (Feature-Parameter Mapper)")
        print("  3. Improve XGBoost training efficiency")
        print("  4. Reduce model training time by selecting optimal feature subsets")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
