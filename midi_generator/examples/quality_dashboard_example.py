#!/usr/bin/env python3
"""
Quality Metrics Dashboard Example - Agent 31
============================================

Demonstration of the Quality Metrics Dashboard for monitoring
the Musical Program Synthesis system.

This example shows:
1. Setting up the dashboard
2. Registering parameters from the universal registry
3. Simulating model training and validation
4. Monitoring reconstruction quality
5. Generating reports and visualizations
6. Identifying improvement opportunities
"""

import sys
from pathlib import Path
import time
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.quality_dashboard import QualityMetricsDashboard


def simulate_training_session():
    """Simulate a complete training session with quality monitoring"""

    print("=" * 80)
    print("QUALITY METRICS DASHBOARD - DEMONSTRATION")
    print("Agent 31: Real-time Quality Monitoring")
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 1: Initialize Dashboard
    # =========================================================================
    print("STEP 1: Initializing Quality Metrics Dashboard")
    print("-" * 80)

    dashboard = QualityMetricsDashboard(
        storage_dir=Path("./demo_monitoring_data"),
        alert_threshold_r2=0.60,
        enable_visualization=True
    )

    print()

    # =========================================================================
    # STEP 2: Register Parameters
    # =========================================================================
    print("STEP 2: Registering Parameters from Universal Registry")
    print("-" * 80)

    # Sample parameters from different categories
    parameters = [
        # Harmony parameters
        ("voicing_type", "harmony.voicing.type", "harmony"),
        ("voicing_spread", "harmony.voicing.spread", "harmony"),
        ("voicing_density", "harmony.voicing.density", "harmony"),
        ("extension_complexity", "harmony.extensions.complexity", "harmony"),
        ("tritone_sub_prob", "harmony.substitution.tritone_probability", "harmony"),

        # Melody parameters
        ("note_density", "melody.density", "melody"),
        ("interval_distribution", "melody.intervals.distribution", "melody"),
        ("phrase_length", "melody.phrasing.length", "melody"),
        ("contour_shape", "melody.contour.shape", "melody"),
        ("ornamentation_rate", "melody.ornamentation.rate", "melody"),

        # Rhythm parameters
        ("swing_ratio", "rhythm.swing.ratio", "rhythm"),
        ("syncopation_level", "rhythm.syncopation.level", "rhythm"),
        ("groove_intensity", "rhythm.groove.intensity", "rhythm"),
        ("subdivision_complexity", "rhythm.subdivision.complexity", "rhythm"),
        ("polyrhythm_density", "rhythm.polyrhythm.density", "rhythm"),

        # Bass parameters
        ("walking_pattern", "bass.walking.pattern", "bass"),
        ("register_preference", "bass.register.preference", "bass"),
        ("rhythmic_activity", "bass.rhythm.activity", "bass"),

        # Drum parameters
        ("kick_density", "drums.kick.density", "drums"),
        ("snare_pattern", "drums.snare.pattern", "drums"),
        ("hihat_subdivision", "drums.hihat.subdivision", "drums"),
    ]

    for name, path, category in parameters:
        dashboard.register_parameter(name, path, category)

    print(f"✓ Registered {len(parameters)} parameters")
    print()

    # =========================================================================
    # STEP 3: Simulate Model Training
    # =========================================================================
    print("STEP 3: Simulating XGBoost Model Training and Validation")
    print("-" * 80)

    print("Training models for each parameter...")
    print()

    for i, (name, path, category) in enumerate(parameters, 1):
        # Simulate training
        training_time = random.uniform(15, 60)  # 15-60 seconds
        training_samples = random.randint(500, 2500)

        # Simulate varying quality based on category
        # Harmony and melody models tend to perform better
        if category in ["harmony", "melody"]:
            base_r2 = random.uniform(0.75, 0.95)
        elif category in ["rhythm", "bass"]:
            base_r2 = random.uniform(0.70, 0.90)
        else:  # drums
            base_r2 = random.uniform(0.65, 0.85)

        # Add some variation
        r2_score = base_r2 + random.gauss(0, 0.05)
        r2_score = max(0.4, min(0.98, r2_score))  # Clamp to reasonable range

        # Correlated metrics
        mae = (1.0 - r2_score) * 0.2 + random.uniform(0.01, 0.05)
        rmse = mae * 1.2 + random.uniform(0.01, 0.03)
        correlation = r2_score ** 0.5

        # Simulate feature importance
        feature_importance = {
            f"feature_{j}": random.random()
            for j in range(10)
        }
        # Normalize
        total = sum(feature_importance.values())
        feature_importance = {k: v/total for k, v in feature_importance.items()}

        # Update dashboard
        dashboard.update_parameter_metrics(
            parameter_path=path,
            r2_score=r2_score,
            mae=mae,
            rmse=rmse,
            correlation=correlation,
            training_samples=training_samples,
            training_time=training_time,
            inference_time=random.uniform(1.0, 5.0),
            feature_importance=feature_importance
        )

        print(f"  [{i:2d}/{len(parameters)}] {path:40s} R²={r2_score:.3f}")

        time.sleep(0.1)  # Small delay for realistic timing

    print()
    print("✓ Model training complete")
    print()

    # =========================================================================
    # STEP 4: Simulate Reconstruction Testing
    # =========================================================================
    print("STEP 4: Testing MIDI Reconstruction Quality")
    print("-" * 80)

    print("Running reconstruction tests (MIDI → features → parameters → MIDI)...")
    print()

    for test_num in range(20):
        # Simulate reconstruction quality
        reconstruction_accuracy = random.uniform(0.75, 0.95)
        parameter_recovery_rate = random.uniform(0.80, 0.95)
        note_accuracy = random.uniform(0.85, 0.98)
        rhythm_accuracy = random.uniform(0.80, 0.95)
        harmony_accuracy = random.uniform(0.75, 0.92)

        # Simulate errors
        errors = {
            'missing_notes': random.randint(0, 5),
            'extra_notes': random.randint(0, 3),
            'pitch_errors': random.randint(0, 4),
            'timing_errors': random.randint(0, 6)
        }

        dashboard.update_reconstruction_metrics(
            reconstruction_accuracy=reconstruction_accuracy,
            parameter_recovery_rate=parameter_recovery_rate,
            note_accuracy=note_accuracy,
            rhythm_accuracy=rhythm_accuracy,
            harmony_accuracy=harmony_accuracy,
            errors=errors
        )

        if (test_num + 1) % 5 == 0:
            print(f"  Completed {test_num + 1}/20 reconstruction tests")

    print()
    print("✓ Reconstruction testing complete")
    print()

    # =========================================================================
    # STEP 5: Compute System Health
    # =========================================================================
    print("STEP 5: Computing System Health Metrics")
    print("-" * 80)

    health = dashboard.compute_system_health()

    print(f"Overall Health Score: {health.health_score:.1f}/100")
    print()
    print(f"Parameters:")
    print(f"  Total: {health.total_parameters}")
    print(f"  Trained: {health.trained_parameters} ({health.trained_parameters/health.total_parameters*100:.1f}%)")
    print()
    print(f"Model Quality:")
    print(f"  Excellent (R²>0.90): {health.excellent_models:2d} ({health.excellent_models/health.trained_parameters*100:.1f}%)")
    print(f"  Good (R²>0.75):      {health.good_models:2d} ({health.good_models/health.trained_parameters*100:.1f}%)")
    print(f"  Poor (R²<0.60):      {health.poor_models:2d} ({health.poor_models/health.trained_parameters*100:.1f}%)")
    print()
    print(f"R² Statistics:")
    print(f"  Average: {health.avg_r2_score:.3f}")
    print(f"  Median:  {health.median_r2_score:.3f}")
    print(f"  Range:   [{health.min_r2_score:.3f}, {health.max_r2_score:.3f}]")
    print()
    print(f"Reconstruction: {health.avg_reconstruction_accuracy:.1%} accuracy")
    print(f"Active Alerts: {health.active_alerts} ({health.critical_alerts} critical)")
    print()

    # =========================================================================
    # STEP 6: Check Alerts
    # =========================================================================
    print("STEP 6: Reviewing Active Alerts")
    print("-" * 80)

    active_alerts = dashboard.get_active_alerts()

    if active_alerts:
        print(f"Found {len(active_alerts)} active alerts:")
        print()

        for i, alert in enumerate(active_alerts[:5], 1):  # Show first 5
            print(f"Alert {i}: [{alert.severity.value.upper()}] {alert.message}")
            if alert.parameter_name:
                print(f"  Parameter: {alert.parameter_name}")
            print(f"  Time: {alert.datetime_str}")
            print()
    else:
        print("✓ No active alerts - system performing well!")
        print()

    # =========================================================================
    # STEP 7: Identify Improvement Opportunities
    # =========================================================================
    print("STEP 7: Identifying Improvement Opportunities")
    print("-" * 80)

    opportunities = dashboard.identify_improvement_opportunities(top_n=10)

    if opportunities:
        print(f"Top {len(opportunities)} parameters that could benefit from improvement:")
        print()

        for i, opp in enumerate(opportunities, 1):
            print(f"{i:2d}. {opp['parameter']}")
            print(f"    Current R²: {opp['current_r2']:.3f}")
            print(f"    Quality: {opp['quality']} | Trend: {opp['trend']}")
            print(f"    Priority Score: {opp['improvement_score']:.2f}")
            print(f"    Training Samples: {opp['training_samples']}")
            print(f"    Recommendations:")
            for rec in opp['recommendations']:
                print(f"      → {rec}")
            print()
    else:
        print("✓ All parameters performing well!")
        print()

    # =========================================================================
    # STEP 8: Category Analysis
    # =========================================================================
    print("STEP 8: Analyzing Performance by Category")
    print("-" * 80)

    categories = ["harmony", "melody", "rhythm", "bass", "drums"]

    for category in categories:
        summary = dashboard.get_category_summary(category)
        if "error" not in summary:
            print(f"{category.upper()}:")
            print(f"  Parameters: {summary['trained_parameters']}/{summary['total_parameters']}")
            print(f"  Average R²: {summary['average_r2']:.3f}")
            print(f"  Quality: {summary['quality_distribution']}")
            print()

    # =========================================================================
    # STEP 9: Detect Anomalies
    # =========================================================================
    print("STEP 9: Detecting Anomalies")
    print("-" * 80)

    anomalies = dashboard.detect_anomalies(window_size=20, std_threshold=2.5)

    if anomalies:
        print(f"⚠️  Detected {len(anomalies)} anomalies:")
        print()
        for anom in anomalies[:5]:  # Show first 5
            print(f"  Parameter: {anom['parameter']}")
            print(f"  Value: {anom['value']:.3f} (expected ~{anom['expected']:.3f})")
            print(f"  Severity: {anom['severity']}")
            print()
    else:
        print("✓ No anomalies detected")
        print()

    # =========================================================================
    # STEP 10: Generate Reports and Visualizations
    # =========================================================================
    print("STEP 10: Generating Reports and Visualizations")
    print("-" * 80)

    # Generate text report
    report_path = dashboard.storage_dir / "quality_report.txt"
    report = dashboard.generate_report(report_type="full", save_path=report_path)
    print(f"✓ Full report saved to: {report_path}")

    # Generate visualizations
    if dashboard.enable_visualization:
        print()
        print("Generating visualizations...")

        # System health dashboard
        health_plot = dashboard.plot_system_health()
        if health_plot:
            print(f"  ✓ System health dashboard: {health_plot}")

        # Category comparison
        category_plot = dashboard.plot_category_comparison()
        if category_plot:
            print(f"  ✓ Category comparison: {category_plot}")

        # Individual parameter plots (sample a few)
        sample_params = [
            "harmony.voicing.type",
            "melody.density",
            "rhythm.swing.ratio"
        ]

        for param in sample_params:
            if param in dashboard.parameter_metrics:
                plot_path = dashboard.plot_parameter_history(param)
                if plot_path:
                    print(f"  ✓ Parameter history ({param}): {plot_path}")

    print()

    # =========================================================================
    # STEP 11: Save State
    # =========================================================================
    print("STEP 11: Saving Dashboard State")
    print("-" * 80)

    dashboard.save_state()
    print(f"✓ Dashboard state saved to: {dashboard.storage_dir}")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  • Monitored {len(parameters)} parameters across {len(categories)} categories")
    print(f"  • System Health Score: {health.health_score:.1f}/100")
    print(f"  • Average R² Score: {health.avg_r2_score:.3f}")
    print(f"  • Reconstruction Accuracy: {health.avg_reconstruction_accuracy:.1%}")
    print(f"  • Active Alerts: {health.active_alerts}")
    print()
    print("All reports and visualizations saved to:")
    print(f"  {dashboard.storage_dir.absolute()}")
    print()

    return dashboard


def example_integration_with_registry():
    """Example showing integration with parameter registry"""
    print("\n" + "=" * 80)
    print("BONUS: Integration with Universal Parameter Registry")
    print("=" * 80)
    print()

    try:
        from parameters.universal_registry import UniversalParameterRegistry

        print("Loading Universal Parameter Registry...")
        registry = UniversalParameterRegistry()

        print(f"✓ Registry loaded with {len(registry.parameters)} parameters")
        print()

        # Create dashboard
        dashboard = QualityMetricsDashboard(
            storage_dir=Path("./full_system_monitoring"),
            enable_visualization=False
        )

        print("Auto-registering all parameters from registry...")

        # Register all parameters
        registered = 0
        for param_path, param_def in registry.parameters.items():
            dashboard.register_parameter(
                parameter_name=param_def.name,
                parameter_path=param_path,
                category=param_def.category.value if param_def.category else "unknown"
            )
            registered += 1

        print(f"✓ Registered {registered} parameters from universal registry")
        print()
        print("Dashboard is now ready to monitor the complete system!")

    except ImportError as e:
        print(f"⚠️  Could not import UniversalParameterRegistry: {e}")
        print("   (This is normal if the registry module is not yet available)")


if __name__ == "__main__":
    # Run main demonstration
    dashboard = simulate_training_session()

    # Show integration example
    example_integration_with_registry()

    print()
    print("✓ Quality Metrics Dashboard demonstration complete!")
    print()
