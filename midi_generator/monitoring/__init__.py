"""
Monitoring Module - Agent 31
============================

Real-time quality monitoring and metrics dashboard for the Musical Program Synthesis system.

This module provides comprehensive quality tracking for:
- Per-parameter XGBoost model performance (R² scores, MAE, RMSE)
- MIDI reconstruction quality (accuracy, error rates)
- System health indicators
- Anomaly detection
- Trend analysis
- Visual reporting

Usage:
    from midi_generator.monitoring import QualityMetricsDashboard

    # Create dashboard
    dashboard = QualityMetricsDashboard(storage_dir="./monitoring_data")

    # Register parameters
    dashboard.register_parameter("voicing_type", "harmony.voicing.type", "harmony")

    # Update metrics
    dashboard.update_parameter_metrics(
        "harmony.voicing.type",
        r2_score=0.85,
        mae=0.12,
        training_samples=1000
    )

    # Generate reports
    health = dashboard.compute_system_health()
    report = dashboard.generate_report()

    # Create visualizations
    dashboard.plot_system_health()

Author: Agent 31 - Quality Metrics Dashboard
"""

from .quality_dashboard import (
    QualityMetricsDashboard,
    QualityLevel,
    MetricType,
    AlertSeverity,
    TrendDirection,
    ParameterMetrics,
    ReconstructionMetrics,
    SystemHealthMetrics,
    QualityAlert,
    MetricDataPoint
)

__version__ = "1.0.0"

__all__ = [
    "QualityMetricsDashboard",
    "QualityLevel",
    "MetricType",
    "AlertSeverity",
    "TrendDirection",
    "ParameterMetrics",
    "ReconstructionMetrics",
    "SystemHealthMetrics",
    "QualityAlert",
    "MetricDataPoint"
]
