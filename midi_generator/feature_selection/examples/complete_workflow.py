"""
Complete Feature Selection Workflow Example
============================================

This script demonstrates the complete workflow for feature selection
and optimized extraction using Agent 04's feature selection system.

**Prerequisites:**
- Labeled dataset from Agent 03 (labeled_dataset.json)
- Hierarchical parameters from Agent 01 (hierarchical_parameters.json)

Author: Agent 04 - Feature Selection Optimizer
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.feature_selection.feature_selector import ComprehensiveFeatureSelector
from midi_generator.feature_selection.optimized_feature_extractor import (
    OptimizedFeatureExtractor,
    FeatureNormalizer,
    BatchFeatureProcessor
)
from midi_generator.feature_selection.feature_importance_report import (
    FeatureImportanceAnalyzer
)


def main():
    """Run complete feature selection workflow"""

    print("="*70)
    print("FEATURE SELECTION WORKFLOW - AGENT 04")
    print("="*70)

    # ========================================================================
    # PHASE 1: FEATURE SELECTION
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 1: FEATURE SELECTION")
    print("="*70)

    # Step 1: Load labeled dataset
    print("\n📂 Step 1: Loading labeled dataset...")

    # NOTE: This will be available after Agent 03 completes
    labeled_dataset_path = Path('midi_generator/labeled_dataset.json')

    if not labeled_dataset_path.exists():
        print("⚠️ Labeled dataset not found!")
        print("   Creating synthetic data for demonstration...")

        # Create synthetic data
        n_samples = 750
        n_features = 1000
        X = np.random.randn(n_samples, n_features)
        feature_names = [f"feature_{i:04d}" for i in range(n_features)]
        y = np.random.randn(n_samples)  # Target parameter values
        parameter_name = 'demo_parameter'

    else:
        with open(labeled_dataset_path, 'r') as f:
            dataset = json.load(f)

        X = np.array(dataset['features'])
        feature_names = dataset['feature_names']
        y = np.array(dataset['labels']['tempo'])  # Example parameter
        parameter_name = 'tempo'

    print(f"✅ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")

    # Step 2: Run feature selection
    print("\n🎯 Step 2: Running feature selection methods...")

    selector = ComprehensiveFeatureSelector(
        feature_matrix=X,
        feature_names=feature_names,
        target_n_features=200
    )

    # Run all methods (use subset for speed in demo)
    results = selector.run_all_methods(
        target_values=y,
        parameter_name=parameter_name,
        methods=['filter', 'univariate', 'lasso', 'domain']
    )

    # Step 3: Ensemble selection
    print("\n🎲 Step 3: Running ensemble selection...")

    ensemble_result = selector.ensemble_selection(
        results=results,
        min_votes=2  # At least 2 methods must agree
    )

    # Step 4: Save selected features
    print("\n💾 Step 4: Saving selected features...")

    output_dir = Path('midi_generator/feature_selection/output')
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_features_path = output_dir / 'selected_features_200.json'
    selector.save_selected_features(
        ensemble_result,
        selected_features_path,
        include_scores=True
    )

    # Step 5: Generate importance report
    print("\n📊 Step 5: Generating feature importance report...")

    analyzer = FeatureImportanceAnalyzer()

    # Add all method results
    for method_name, result in results.items():
        analyzer.add_method_result(method_name, {
            'selected_features': result.selected_features,
            'feature_scores': result.feature_scores
        })

    # Generate report
    report = analyzer.generate_report(
        report_title=f"Feature Importance Analysis for {parameter_name}"
    )

    # Save report
    report_path = output_dir / 'feature_importance_report.md'
    analyzer.save_report(report, report_path, format='markdown')

    # Print summary
    selector.print_summary()

    # ========================================================================
    # PHASE 2: OPTIMIZED EXTRACTION
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 2: OPTIMIZED EXTRACTION")
    print("="*70)

    # Step 6: Create optimized extractor
    print("\n🚀 Step 6: Creating optimized feature extractor...")

    # NOTE: This requires selected_features_200.json from Phase 1
    if selected_features_path.exists():
        extractor = OptimizedFeatureExtractor.from_selection_file(
            selected_features_path
        )
        print("✅ Optimized extractor created!")

        # Demonstrate extraction (if MIDI files available)
        print("\n   To extract features from MIDI:")
        print("   >>> features = extractor.extract('song.mid')")
        print("   >>> # Returns: numpy array of shape (200,)")

    # Step 7: Feature normalization
    print("\n📐 Step 7: Creating feature normalizer...")

    # Simulate extracted features
    print("   Simulating feature extraction for normalization...")
    simulated_features = np.random.randn(n_samples, 200)

    normalizer = FeatureNormalizer()
    normalizer.fit(simulated_features)

    # Save normalizer
    normalizer_path = output_dir / 'normalizer.json'
    normalizer.save(normalizer_path)

    print("✅ Normalizer fitted and saved!")

    # Step 8: Batch processing example
    print("\n🔄 Step 8: Batch processing setup...")

    processor = BatchFeatureProcessor(
        extractor=extractor,
        normalizer=normalizer
    )

    print("✅ Batch processor ready!")
    print("\n   To process a directory:")
    print("   >>> features = processor.process_directory('midi_corpus/')")
    print("   >>> # Returns: numpy array of shape (n_files, 200)")

    # ========================================================================
    # PHASE 3: RESULTS SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 3: RESULTS SUMMARY")
    print("="*70)

    print("\n📋 Generated Files:")
    print(f"   1. {selected_features_path}")
    print(f"   2. {report_path}")
    print(f"   3. {normalizer_path}")

    print("\n📊 Feature Selection Results:")
    print(f"   - Original features: {X.shape[1]}")
    print(f"   - Selected features: {ensemble_result.n_features_selected}")
    print(f"   - Reduction: {(1 - ensemble_result.n_features_selected/X.shape[1])*100:.1f}%")
    print(f"   - Methods used: {len(results)}")

    print("\n🎯 Next Steps:")
    print("   1. Review feature_importance_report.md")
    print("   2. Validate selected features make musical sense")
    print("   3. Test extraction speed with optimized extractor")
    print("   4. Pass selected_features_200.json to Agent 05 (Hierarchical MTL)")

    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
