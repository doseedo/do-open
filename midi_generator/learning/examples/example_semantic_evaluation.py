#!/usr/bin/env python3
"""
AGENT 9: EXAMPLE USAGE - SEMANTIC EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Demonstrates how to use the SemanticEvaluator to assess
the quality of a trained semantic feature discovery system.

This example shows:
1. Loading a trained encoder and feature bank
2. Creating a test dataset
3. Running comprehensive evaluation
4. Generating quality reports
5. Running ablation studies

Author: Agent 9 - Testing & Validation Specialist
"""

import torch
import numpy as np
from pathlib import Path

# Import evaluation framework
from midi_generator.learning.semantic_evaluation import (
    SemanticEvaluator,
    EvaluationMetrics,
)
from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder
from midi_generator.learning.semantic_features import SemanticFeature, SemanticFeatureBank
from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline


def example_1_basic_evaluation():
    """Example 1: Basic evaluation of a trained encoder."""

    print("=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)

    # Step 1: Create or load a trained encoder
    print("\n[1/4] Creating encoder...")
    encoder = SemanticFeatureEncoder(input_dim=200, output_dim=30)

    # In practice, you would load a trained encoder:
    # encoder.load_state_dict(torch.load('trained_encoder.pth'))

    # Step 2: Create a test dataset
    print("[2/4] Creating test dataset...")

    class SimpleTestDataset:
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                'features': torch.randn(200),
                'gap': torch.randn(200) * 0.1,
            }

    test_dataset = SimpleTestDataset(size=100)

    # Step 3: Create evaluator
    print("[3/4] Creating evaluator...")
    evaluator = SemanticEvaluator(
        encoder=encoder,
        test_dataset=test_dataset,
        device='cpu'
    )

    # Step 4: Run evaluation
    print("[4/4] Running evaluation...")
    metrics = evaluator.evaluate_all(verbose=True)

    # Print results
    print(f"\n✅ Evaluation complete!")
    print(f"   Reconstruction R²: {metrics.reconstruction_r2:.4f}")
    print(f"   Overall Quality:   {metrics.overall_quality_score:.4f}")


def example_2_with_feature_bank():
    """Example 2: Evaluation with discovered features."""

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Evaluation with Feature Bank")
    print("=" * 70)

    # Step 1: Create encoder
    print("\n[1/5] Creating encoder...")
    encoder = SemanticFeatureEncoder(input_dim=200, output_dim=30)

    # Step 2: Create feature bank with discovered features
    print("[2/5] Creating feature bank with discovered features...")
    feature_bank = SemanticFeatureBank()

    # Simulate discovered features
    feature_names = [
        'harmonic_complexity',
        'rhythmic_syncopation',
        'tonal_clarity',
        'voice_leading_smoothness',
        'textural_density',
        'dynamic_range',
        'articulation_sharpness',
        'tempo_stability',
        'harmonic_tension',
        'melodic_contour',
    ]

    for i, name in enumerate(feature_names):
        feature = SemanticFeature(
            index=i,
            name=name,
            description=f"Discovered semantic parameter: {name}",
        )
        feature.musical_validity_score = np.random.uniform(0.75, 0.95)
        feature_bank.add_feature(feature)

    print(f"   Created {len(feature_bank.features)} features")

    # Step 3: Create test dataset
    print("[3/5] Creating test dataset...")

    class TestDataset:
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {'features': torch.randn(200)}

    test_dataset = TestDataset()

    # Step 4: Create evaluator
    print("[4/5] Creating evaluator with feature bank...")
    evaluator = SemanticEvaluator(
        encoder=encoder,
        feature_bank=feature_bank,
        test_dataset=test_dataset,
        device='cpu'
    )

    # Step 5: Run evaluation
    print("[5/5] Running evaluation...")
    metrics = evaluator.evaluate_all(verbose=False)

    # Print interpretability results
    print("\n📊 INTERPRETABILITY RESULTS:")
    print(f"   Named Parameters:       {metrics.named_parameters_ratio:.1%}")
    print(f"   Musical Validity:       {metrics.musical_validity_score:.4f}")
    print(f"   Interpretability Score: {metrics.interpretability_score:.4f}")


def example_3_generate_reports():
    """Example 3: Generate comprehensive quality reports."""

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Generate Quality Reports")
    print("=" * 70)

    # Create complete system
    print("\n[1/5] Setting up evaluation system...")

    encoder = SemanticFeatureEncoder(input_dim=200, output_dim=30)

    feature_bank = SemanticFeatureBank()
    for i in range(20):
        feature = SemanticFeature(
            index=i,
            name=f"discovered_param_{i}",
            description=f"Parameter {i}",
        )
        feature.musical_validity_score = np.random.uniform(0.7, 0.9)
        feature_bank.add_feature(feature)

    class TestDataset:
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {'features': torch.randn(200)}

    test_dataset = TestDataset()

    # Create evaluator
    print("[2/5] Creating evaluator...")
    evaluator = SemanticEvaluator(
        encoder=encoder,
        feature_bank=feature_bank,
        test_dataset=test_dataset,
        device='cpu'
    )

    # Run evaluation
    print("[3/5] Running comprehensive evaluation...")
    metrics = evaluator.evaluate_all(verbose=False)

    # Run ablation study
    print("[4/5] Running ablation study...")
    ablation_results = evaluator.ablation_study()

    # Generate reports
    print("[5/5] Generating reports...")
    output_dir = Path('evaluation_reports')
    evaluator.generate_report(output_dir)

    print(f"\n✅ Reports generated in: {output_dir}/")
    print(f"   • evaluation_metrics.json")
    print(f"   • quality_report.html")
    print(f"   • evaluation_summary.txt")
    print(f"   • ablation_results.json")


def example_4_custom_metrics():
    """Example 4: Access individual metrics."""

    print("\n" + "=" * 70)
    print("EXAMPLE 4: Access Individual Metrics")
    print("=" * 70)

    # Setup
    encoder = SemanticFeatureEncoder(input_dim=200, output_dim=30)

    class TestDataset:
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {'features': torch.randn(200)}

    evaluator = SemanticEvaluator(
        encoder=encoder,
        test_dataset=TestDataset(),
        device='cpu'
    )

    # Run individual evaluations
    print("\n[1/5] Evaluating reconstruction quality...")
    evaluator._evaluate_reconstruction_quality()
    print(f"   MSE: {evaluator.metrics.reconstruction_mse:.6f}")
    print(f"   R²:  {evaluator.metrics.reconstruction_r2:.4f}")

    print("\n[2/5] Evaluating locality preservation...")
    evaluator._evaluate_locality_preservation()
    print(f"   Preservation: {evaluator.metrics.locality_preservation_score:.4f}")
    print(f"   Violations:   {evaluator.metrics.invariance_violations}")

    print("\n[3/5] Evaluating sparsity...")
    evaluator._evaluate_sparsity_orthogonality()
    print(f"   Sparsity:      {evaluator.metrics.activation_sparsity:.4f}")
    print(f"   Orthogonality: {evaluator.metrics.feature_orthogonality:.4f}")

    print("\n[4/5] Evaluating performance...")
    evaluator._evaluate_performance()
    print(f"   Extraction time: {evaluator.metrics.extraction_time_ms:.2f} ms")

    print("\n[5/5] Computing overall quality...")
    evaluator._compute_overall_quality()
    print(f"   Overall quality: {evaluator.metrics.overall_quality_score:.4f}")


def example_5_pipeline_integration():
    """Example 5: Integration with SemanticDiscoveryPipeline."""

    print("\n" + "=" * 70)
    print("EXAMPLE 5: Pipeline Integration")
    print("=" * 70)

    # Create pipeline
    print("\n[1/3] Creating semantic discovery pipeline...")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = SemanticDiscoveryPipeline(
            cache_dir=tmpdir,
            feature_dim=200,
            semantic_dim=30,
            device='cpu'
        )

        # Get trained encoder from pipeline
        encoder = pipeline.encoder

        # Create test dataset
        print("[2/3] Creating test dataset...")

        class TestDataset:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {'features': torch.randn(200)}

        test_dataset = TestDataset()

        # Create evaluator with pipeline components
        print("[3/3] Evaluating pipeline...")
        evaluator = SemanticEvaluator(
            encoder=encoder,
            test_dataset=test_dataset,
            device='cpu'
        )

        metrics = evaluator.evaluate_all(verbose=False)

        print(f"\n✅ Pipeline evaluation complete!")
        print(f"   Reconstruction R²: {metrics.reconstruction_r2:.4f}")
        print(f"   Overall quality:   {metrics.overall_quality_score:.4f}")


def example_6_ablation_study():
    """Example 6: Detailed ablation study."""

    print("\n" + "=" * 70)
    print("EXAMPLE 6: Ablation Study")
    print("=" * 70)

    # Setup
    print("\n[1/2] Setting up system...")
    encoder = SemanticFeatureEncoder(input_dim=200, output_dim=30)

    class TestDataset:
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {'features': torch.randn(200)}

    evaluator = SemanticEvaluator(
        encoder=encoder,
        test_dataset=TestDataset(),
        device='cpu'
    )

    # First, establish baseline
    evaluator.evaluate_all(verbose=False)
    baseline = evaluator.metrics.reconstruction_r2

    print(f"   Baseline R²: {baseline:.4f}")

    # Run ablation study
    print("\n[2/2] Running ablation study...")
    components = [
        'locality_loss',
        'sparsity_loss',
        'orthogonality_loss'
    ]

    results = evaluator.ablation_study(components=components)

    print("\n📊 ABLATION RESULTS:")
    for result in results:
        print(f"   {result.component_name:20s}: "
              f"importance = {result.importance:.4f}")


def main():
    """Run all examples."""

    print("\n" + "═" * 70)
    print("AGENT 9: SEMANTIC EVALUATION - USAGE EXAMPLES")
    print("═" * 70)

    examples = [
        ("Basic Evaluation", example_1_basic_evaluation),
        ("Evaluation with Feature Bank", example_2_with_feature_bank),
        ("Generate Quality Reports", example_3_generate_reports),
        ("Access Individual Metrics", example_4_custom_metrics),
        ("Pipeline Integration", example_5_pipeline_integration),
        ("Ablation Study", example_6_ablation_study),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")

    print("\n" + "═" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("═" * 70)


if __name__ == "__main__":
    main()
