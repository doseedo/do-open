#!/usr/bin/env python3
"""
AGENT 31: QUALITY METRICS DASHBOARD
====================================

Real-time quality monitoring system for the Musical Program Synthesis framework.

This comprehensive dashboard provides:
- Per-parameter R² score tracking for all 165+ XGBoost models
- Reconstruction quality metrics (MIDI → features → parameters → MIDI)
- Real-time performance monitoring
- Historical trend analysis
- Anomaly detection and alerting
- Visual reporting and insights

The dashboard ensures the self-expanding system maintains quality as it grows
from 165 → 515 → 800+ parameters.

Author: Agent 31 - Quality Metrics Dashboard
Integration: Monitors outputs from ALL agents (1-35)
Architecture: Modular metrics per XGBoost model (no retraining needed)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import warnings
import time
from enum import Enum, auto

# Optional imports for visualization and statistics
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available - some statistical features will be limited")

try:
    from scipy import stats
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - advanced statistics will be limited")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available - visualization will be disabled")


# ==============================================================================
# ENUMERATIONS
# ==============================================================================

class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"  # R² > 0.90
    GOOD = "good"            # R² > 0.75
    ACCEPTABLE = "acceptable"  # R² > 0.60
    POOR = "poor"            # R² > 0.40
    CRITICAL = "critical"    # R² <= 0.40


class MetricType(Enum):
    """Types of metrics tracked"""
    R_SQUARED = "r_squared"
    MAE = "mae"  # Mean Absolute Error
    RMSE = "rmse"  # Root Mean Square Error
    CORRELATION = "correlation"
    RECONSTRUCTION_ERROR = "reconstruction_error"
    FEATURE_IMPORTANCE = "feature_importance"
    PREDICTION_VARIANCE = "prediction_variance"
    OUTLIER_RATE = "outlier_rate"
    TRAINING_TIME = "training_time"
    INFERENCE_TIME = "inference_time"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TrendDirection(Enum):
    """Direction of metric trends"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class MetricDataPoint:
    """Single metric measurement"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def datetime_str(self) -> str:
        """Human-readable timestamp"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class ParameterMetrics:
    """Complete metrics for a single parameter"""
    parameter_name: str
    parameter_path: str  # e.g., "harmony.voicing.type"
    category: str  # e.g., "harmony", "rhythm"

    # Current metrics
    current_r2: Optional[float] = None
    current_mae: Optional[float] = None
    current_rmse: Optional[float] = None
    current_correlation: Optional[float] = None

    # Historical tracking (last 1000 measurements)
    r2_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    mae_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    reconstruction_error_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Statistics
    training_samples: int = 0
    last_trained: Optional[float] = None
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None

    # Quality indicators
    quality_level: QualityLevel = QualityLevel.ACCEPTABLE
    trend_direction: TrendDirection = TrendDirection.STABLE

    # Feature importance (from XGBoost)
    feature_importance_scores: Dict[str, float] = field(default_factory=dict)

    # Flags
    needs_retraining: bool = False
    has_anomalies: bool = False
    is_monitored: bool = True

    def update_quality_level(self):
        """Update quality level based on current R²"""
        if self.current_r2 is None:
            self.quality_level = QualityLevel.ACCEPTABLE
        elif self.current_r2 > 0.90:
            self.quality_level = QualityLevel.EXCELLENT
        elif self.current_r2 > 0.75:
            self.quality_level = QualityLevel.GOOD
        elif self.current_r2 > 0.60:
            self.quality_level = QualityLevel.ACCEPTABLE
        elif self.current_r2 > 0.40:
            self.quality_level = QualityLevel.POOR
        else:
            self.quality_level = QualityLevel.CRITICAL

    def get_recent_trend(self, window: int = 20) -> TrendDirection:
        """Calculate trend from recent R² measurements"""
        if len(self.r2_history) < window:
            return TrendDirection.STABLE

        recent = list(self.r2_history)[-window:]
        values = [dp.value for dp in recent]

        if not NUMPY_AVAILABLE:
            # Simple trend detection without numpy
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            diff = avg_second - avg_first
            std = self._simple_std(values)

            if abs(diff) < 0.02:  # Small change
                return TrendDirection.STABLE
            elif diff > 0:
                return TrendDirection.IMPROVING
            else:
                return TrendDirection.DECLINING

        # With numpy: use linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        residuals = values - (slope * x + intercept)
        volatility = np.std(residuals)

        if volatility > 0.1:
            return TrendDirection.VOLATILE
        elif abs(slope) < 0.001:
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DECLINING

    @staticmethod
    def _simple_std(values: List[float]) -> float:
        """Calculate standard deviation without numpy"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


@dataclass
class ReconstructionMetrics:
    """Metrics for MIDI reconstruction quality"""
    timestamp: float

    # Overall quality
    reconstruction_accuracy: float  # 0.0 to 1.0
    parameter_recovery_rate: float  # % of parameters correctly recovered

    # Detailed metrics
    note_accuracy: float  # % notes in correct position
    rhythm_accuracy: float  # % rhythms preserved
    harmony_accuracy: float  # % chords preserved

    # Errors
    missing_notes: int = 0
    extra_notes: int = 0
    pitch_errors: int = 0
    timing_errors: int = 0

    # Complexity
    input_complexity: float = 0.0  # Measure of input MIDI complexity
    output_complexity: float = 0.0


@dataclass
class SystemHealthMetrics:
    """Overall system health indicators"""
    timestamp: float

    # Model statistics
    total_parameters: int
    trained_parameters: int
    excellent_models: int  # R² > 0.90
    good_models: int       # R² > 0.75
    poor_models: int       # R² < 0.60

    # Performance
    avg_r2_score: float
    median_r2_score: float
    min_r2_score: float
    max_r2_score: float

    # Reconstruction
    avg_reconstruction_accuracy: float
    reconstruction_samples: int

    # Alerts
    active_alerts: int
    critical_alerts: int

    # Resource usage
    total_training_time_hours: float = 0.0
    avg_inference_time_ms: float = 0.0

    @property
    def health_score(self) -> float:
        """Overall health score 0-100"""
        # Weighted combination of metrics
        r2_score = min(100, self.avg_r2_score * 100)
        reconstruction_score = self.avg_reconstruction_accuracy * 100
        model_coverage = (self.trained_parameters / max(1, self.total_parameters)) * 100

        # Penalties
        alert_penalty = min(20, self.critical_alerts * 5)
        poor_model_penalty = min(20, self.poor_models * 2)

        health = (r2_score * 0.4 +
                 reconstruction_score * 0.3 +
                 model_coverage * 0.3 -
                 alert_penalty -
                 poor_model_penalty)

        return max(0, min(100, health))


@dataclass
class QualityAlert:
    """Alert for quality issues"""
    timestamp: float
    severity: AlertSeverity
    category: str
    parameter_name: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None

    @property
    def age_seconds(self) -> float:
        """How long this alert has been active"""
        if self.resolved:
            return self.resolved_at - self.timestamp
        return time.time() - self.timestamp

    @property
    def datetime_str(self) -> str:
        """Human-readable timestamp"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


# ==============================================================================
# MAIN DASHBOARD CLASS
# ==============================================================================

class QualityMetricsDashboard:
    """
    Comprehensive quality monitoring dashboard for the Musical Program Synthesis system.

    Features:
    - Per-parameter R² tracking for all XGBoost models
    - Real-time reconstruction quality monitoring
    - Trend analysis and anomaly detection
    - Visual reporting and alerting
    - Historical data persistence
    """

    def __init__(self,
                 storage_dir: Optional[Path] = None,
                 max_history_size: int = 10000,
                 alert_threshold_r2: float = 0.60,
                 enable_visualization: bool = True):
        """
        Initialize the quality dashboard.

        Args:
            storage_dir: Directory to store metrics and reports
            max_history_size: Maximum number of historical data points to keep
            alert_threshold_r2: R² threshold for generating alerts
            enable_visualization: Whether to generate visualizations
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path("./monitoring_data")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_history_size = max_history_size
        self.alert_threshold_r2 = alert_threshold_r2
        self.enable_visualization = enable_visualization and MATPLOTLIB_AVAILABLE

        # Core data structures
        self.parameter_metrics: Dict[str, ParameterMetrics] = {}
        self.reconstruction_history: deque = deque(maxlen=max_history_size)
        self.system_health_history: deque = deque(maxlen=max_history_size)
        self.active_alerts: List[QualityAlert] = []
        self.alert_history: deque = deque(maxlen=max_history_size)

        # Statistics
        self.total_measurements = 0
        self.dashboard_start_time = time.time()

        # Load existing data if available
        self._load_state()

        print(f"📊 Quality Metrics Dashboard initialized")
        print(f"   Storage: {self.storage_dir}")
        print(f"   Visualization: {'enabled' if self.enable_visualization else 'disabled'}")
        print(f"   Tracking: {len(self.parameter_metrics)} parameters")

    # ==========================================================================
    # PARAMETER REGISTRATION & UPDATES
    # ==========================================================================

    def register_parameter(self,
                          parameter_name: str,
                          parameter_path: str,
                          category: str = "unknown") -> ParameterMetrics:
        """
        Register a new parameter for monitoring.

        Args:
            parameter_name: Short name (e.g., "voicing_type")
            parameter_path: Full path (e.g., "harmony.voicing.type")
            category: Category (e.g., "harmony", "rhythm")

        Returns:
            ParameterMetrics object for this parameter
        """
        if parameter_path not in self.parameter_metrics:
            metrics = ParameterMetrics(
                parameter_name=parameter_name,
                parameter_path=parameter_path,
                category=category
            )
            self.parameter_metrics[parameter_path] = metrics
            print(f"✓ Registered parameter: {parameter_path}")

        return self.parameter_metrics[parameter_path]

    def update_parameter_metrics(self,
                                parameter_path: str,
                                r2_score: Optional[float] = None,
                                mae: Optional[float] = None,
                                rmse: Optional[float] = None,
                                correlation: Optional[float] = None,
                                training_samples: Optional[int] = None,
                                training_time: Optional[float] = None,
                                inference_time: Optional[float] = None,
                                feature_importance: Optional[Dict[str, float]] = None):
        """
        Update metrics for a parameter.

        Args:
            parameter_path: Full parameter path
            r2_score: R² score from model validation
            mae: Mean Absolute Error
            rmse: Root Mean Square Error
            correlation: Correlation coefficient
            training_samples: Number of training samples
            training_time: Training time in seconds
            inference_time: Inference time in milliseconds
            feature_importance: Feature importance scores from XGBoost
        """
        if parameter_path not in self.parameter_metrics:
            # Auto-register if not already registered
            parts = parameter_path.split('.')
            name = parts[-1] if parts else parameter_path
            category = parts[0] if len(parts) > 1 else "unknown"
            self.register_parameter(name, parameter_path, category)

        metrics = self.parameter_metrics[parameter_path]
        timestamp = time.time()

        # Update current values
        if r2_score is not None:
            metrics.current_r2 = r2_score
            metrics.r2_history.append(MetricDataPoint(timestamp, r2_score))
            metrics.update_quality_level()

            # Check for alerts
            if r2_score < self.alert_threshold_r2:
                self._create_alert(
                    severity=AlertSeverity.WARNING if r2_score > 0.40 else AlertSeverity.CRITICAL,
                    category="model_performance",
                    parameter_name=parameter_path,
                    message=f"R² score {r2_score:.3f} below threshold {self.alert_threshold_r2}",
                    details={"r2": r2_score, "threshold": self.alert_threshold_r2}
                )

        if mae is not None:
            metrics.current_mae = mae
            metrics.mae_history.append(MetricDataPoint(timestamp, mae))

        if rmse is not None:
            metrics.current_rmse = rmse

        if correlation is not None:
            metrics.current_correlation = correlation

        if training_samples is not None:
            metrics.training_samples = training_samples

        if training_time is not None:
            metrics.training_time_seconds = training_time
            metrics.last_trained = timestamp

        if inference_time is not None:
            metrics.inference_time_ms = inference_time

        if feature_importance is not None:
            metrics.feature_importance_scores = feature_importance

        # Update trend
        metrics.trend_direction = metrics.get_recent_trend()

        self.total_measurements += 1

    def update_reconstruction_metrics(self,
                                     reconstruction_accuracy: float,
                                     parameter_recovery_rate: float,
                                     note_accuracy: float = 0.0,
                                     rhythm_accuracy: float = 0.0,
                                     harmony_accuracy: float = 0.0,
                                     errors: Optional[Dict[str, int]] = None):
        """
        Update reconstruction quality metrics.

        Args:
            reconstruction_accuracy: Overall reconstruction accuracy (0-1)
            parameter_recovery_rate: % of parameters correctly recovered
            note_accuracy: % of notes preserved
            rhythm_accuracy: % of rhythms preserved
            harmony_accuracy: % of harmonies preserved
            errors: Dict with keys: missing_notes, extra_notes, pitch_errors, timing_errors
        """
        timestamp = time.time()

        metrics = ReconstructionMetrics(
            timestamp=timestamp,
            reconstruction_accuracy=reconstruction_accuracy,
            parameter_recovery_rate=parameter_recovery_rate,
            note_accuracy=note_accuracy,
            rhythm_accuracy=rhythm_accuracy,
            harmony_accuracy=harmony_accuracy
        )

        if errors:
            metrics.missing_notes = errors.get('missing_notes', 0)
            metrics.extra_notes = errors.get('extra_notes', 0)
            metrics.pitch_errors = errors.get('pitch_errors', 0)
            metrics.timing_errors = errors.get('timing_errors', 0)

        self.reconstruction_history.append(metrics)

        # Alert on poor reconstruction
        if reconstruction_accuracy < 0.70:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                category="reconstruction",
                parameter_name=None,
                message=f"Reconstruction accuracy {reconstruction_accuracy:.2%} is low",
                details={
                    "accuracy": reconstruction_accuracy,
                    "parameter_recovery": parameter_recovery_rate,
                    "note_accuracy": note_accuracy
                }
            )

    # ==========================================================================
    # ANALYSIS & REPORTING
    # ==========================================================================

    def compute_system_health(self) -> SystemHealthMetrics:
        """
        Compute overall system health metrics.

        Returns:
            SystemHealthMetrics with current system state
        """
        timestamp = time.time()

        # Count models by quality
        total = len(self.parameter_metrics)
        trained = sum(1 for m in self.parameter_metrics.values() if m.current_r2 is not None)
        excellent = sum(1 for m in self.parameter_metrics.values()
                       if m.quality_level == QualityLevel.EXCELLENT)
        good = sum(1 for m in self.parameter_metrics.values()
                  if m.quality_level == QualityLevel.GOOD)
        poor = sum(1 for m in self.parameter_metrics.values()
                  if m.quality_level in [QualityLevel.POOR, QualityLevel.CRITICAL])

        # R² statistics
        r2_scores = [m.current_r2 for m in self.parameter_metrics.values()
                    if m.current_r2 is not None]

        if r2_scores:
            if NUMPY_AVAILABLE:
                avg_r2 = float(np.mean(r2_scores))
                median_r2 = float(np.median(r2_scores))
                min_r2 = float(np.min(r2_scores))
                max_r2 = float(np.max(r2_scores))
            else:
                avg_r2 = sum(r2_scores) / len(r2_scores)
                sorted_r2 = sorted(r2_scores)
                median_r2 = sorted_r2[len(sorted_r2) // 2]
                min_r2 = min(r2_scores)
                max_r2 = max(r2_scores)
        else:
            avg_r2 = median_r2 = min_r2 = max_r2 = 0.0

        # Reconstruction metrics
        if self.reconstruction_history:
            recent_recon = list(self.reconstruction_history)[-100:]  # Last 100
            avg_recon = sum(r.reconstruction_accuracy for r in recent_recon) / len(recent_recon)
            recon_samples = len(self.reconstruction_history)
        else:
            avg_recon = 0.0
            recon_samples = 0

        # Alerts
        active = len([a for a in self.active_alerts if not a.resolved])
        critical = len([a for a in self.active_alerts
                       if not a.resolved and a.severity == AlertSeverity.CRITICAL])

        # Performance stats
        total_training_time = sum(m.training_time_seconds or 0
                                 for m in self.parameter_metrics.values()) / 3600.0

        inference_times = [m.inference_time_ms for m in self.parameter_metrics.values()
                          if m.inference_time_ms is not None]
        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0.0

        health = SystemHealthMetrics(
            timestamp=timestamp,
            total_parameters=total,
            trained_parameters=trained,
            excellent_models=excellent,
            good_models=good,
            poor_models=poor,
            avg_r2_score=avg_r2,
            median_r2_score=median_r2,
            min_r2_score=min_r2,
            max_r2_score=max_r2,
            avg_reconstruction_accuracy=avg_recon,
            reconstruction_samples=recon_samples,
            active_alerts=active,
            critical_alerts=critical,
            total_training_time_hours=total_training_time,
            avg_inference_time_ms=avg_inference
        )

        self.system_health_history.append(health)

        return health

    def get_parameter_summary(self, parameter_path: str) -> Dict[str, Any]:
        """
        Get detailed summary for a parameter.

        Args:
            parameter_path: Full parameter path

        Returns:
            Dict with comprehensive parameter metrics
        """
        if parameter_path not in self.parameter_metrics:
            return {"error": "Parameter not found"}

        metrics = self.parameter_metrics[parameter_path]

        # Get trend statistics
        recent_r2 = [dp.value for dp in list(metrics.r2_history)[-50:]]

        summary = {
            "parameter_name": metrics.parameter_name,
            "parameter_path": metrics.parameter_path,
            "category": metrics.category,
            "quality_level": metrics.quality_level.value,
            "trend": metrics.trend_direction.value,
            "current_metrics": {
                "r2": metrics.current_r2,
                "mae": metrics.current_mae,
                "rmse": metrics.current_rmse,
                "correlation": metrics.current_correlation
            },
            "training_info": {
                "samples": metrics.training_samples,
                "last_trained": datetime.fromtimestamp(metrics.last_trained).isoformat() if metrics.last_trained else None,
                "training_time_seconds": metrics.training_time_seconds,
                "inference_time_ms": metrics.inference_time_ms
            },
            "statistics": {
                "measurements": len(metrics.r2_history),
                "recent_avg_r2": sum(recent_r2) / len(recent_r2) if recent_r2 else 0.0,
                "recent_min_r2": min(recent_r2) if recent_r2 else 0.0,
                "recent_max_r2": max(recent_r2) if recent_r2 else 0.0
            },
            "flags": {
                "needs_retraining": metrics.needs_retraining,
                "has_anomalies": metrics.has_anomalies,
                "is_monitored": metrics.is_monitored
            },
            "top_features": dict(list(sorted(metrics.feature_importance_scores.items(),
                                            key=lambda x: x[1], reverse=True))[:10])
        }

        return summary

    def get_category_summary(self, category: str) -> Dict[str, Any]:
        """
        Get summary for all parameters in a category.

        Args:
            category: Category name (e.g., "harmony", "rhythm")

        Returns:
            Dict with category-wide metrics
        """
        category_params = [m for m in self.parameter_metrics.values()
                          if m.category == category]

        if not category_params:
            return {"error": f"No parameters found in category '{category}'"}

        # Aggregate statistics
        r2_scores = [m.current_r2 for m in category_params if m.current_r2 is not None]

        quality_counts = defaultdict(int)
        for m in category_params:
            quality_counts[m.quality_level.value] += 1

        trend_counts = defaultdict(int)
        for m in category_params:
            trend_counts[m.trend_direction.value] += 1

        summary = {
            "category": category,
            "total_parameters": len(category_params),
            "trained_parameters": len(r2_scores),
            "average_r2": sum(r2_scores) / len(r2_scores) if r2_scores else 0.0,
            "quality_distribution": dict(quality_counts),
            "trend_distribution": dict(trend_counts),
            "parameters": [
                {
                    "path": m.parameter_path,
                    "name": m.parameter_name,
                    "r2": m.current_r2,
                    "quality": m.quality_level.value,
                    "trend": m.trend_direction.value
                }
                for m in sorted(category_params,
                              key=lambda x: x.current_r2 or 0,
                              reverse=True)
            ]
        }

        return summary

    def detect_anomalies(self,
                        window_size: int = 50,
                        std_threshold: float = 2.5) -> List[Dict[str, Any]]:
        """
        Detect anomalies in parameter performance.

        Args:
            window_size: Size of sliding window for analysis
            std_threshold: Number of standard deviations for anomaly threshold

        Returns:
            List of detected anomalies with details
        """
        anomalies = []

        for param_path, metrics in self.parameter_metrics.items():
            if len(metrics.r2_history) < window_size:
                continue

            recent = list(metrics.r2_history)[-window_size:]
            values = [dp.value for dp in recent]

            if NUMPY_AVAILABLE:
                mean = np.mean(values)
                std = np.std(values)

                # Check for sudden drops
                for i, dp in enumerate(recent[-10:]):  # Check last 10
                    z_score = abs(dp.value - mean) / std if std > 0 else 0
                    if z_score > std_threshold and dp.value < mean:
                        anomalies.append({
                            "parameter": param_path,
                            "timestamp": dp.timestamp,
                            "datetime": dp.datetime_str,
                            "value": dp.value,
                            "expected": mean,
                            "z_score": z_score,
                            "severity": "high" if z_score > 3.5 else "medium"
                        })

                        # Mark in metrics
                        metrics.has_anomalies = True
            else:
                # Simple anomaly detection without numpy
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std = variance ** 0.5

                for dp in recent[-10:]:
                    if std > 0 and abs(dp.value - mean) / std > std_threshold and dp.value < mean:
                        anomalies.append({
                            "parameter": param_path,
                            "timestamp": dp.timestamp,
                            "datetime": dp.datetime_str,
                            "value": dp.value,
                            "expected": mean,
                            "deviation": abs(dp.value - mean),
                            "severity": "medium"
                        })
                        metrics.has_anomalies = True

        return anomalies

    def identify_improvement_opportunities(self,
                                          top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Identify parameters that would benefit most from improvement.

        Args:
            top_n: Number of top opportunities to return

        Returns:
            List of improvement opportunities sorted by priority
        """
        opportunities = []

        for param_path, metrics in self.parameter_metrics.items():
            if metrics.current_r2 is None:
                continue

            # Calculate improvement score
            # Factors:
            # 1. Current performance (lower is worse)
            # 2. Declining trend (negative trend is worse)
            # 3. Category importance (some categories more critical)
            # 4. Training samples (more samples = more potential)

            performance_score = max(0, 1.0 - metrics.current_r2)  # Higher = needs more improvement

            trend_score = 0.0
            if metrics.trend_direction == TrendDirection.DECLINING:
                trend_score = 0.3
            elif metrics.trend_direction == TrendDirection.VOLATILE:
                trend_score = 0.2

            category_weight = {
                "harmony": 1.2,
                "melody": 1.1,
                "rhythm": 1.1,
                "bass": 1.0,
                "voice": 1.0,
                "drums": 0.9,
                "timbre": 0.8,
                "dynamics": 0.8
            }.get(metrics.category, 1.0)

            sample_factor = min(1.0, metrics.training_samples / 1000.0) if metrics.training_samples > 0 else 0.5

            improvement_score = (performance_score + trend_score) * category_weight * sample_factor

            if improvement_score > 0.3:  # Threshold for inclusion
                opportunities.append({
                    "parameter": param_path,
                    "name": metrics.parameter_name,
                    "category": metrics.category,
                    "current_r2": metrics.current_r2,
                    "quality": metrics.quality_level.value,
                    "trend": metrics.trend_direction.value,
                    "improvement_score": improvement_score,
                    "training_samples": metrics.training_samples,
                    "recommendations": self._generate_recommendations(metrics)
                })

        # Sort by improvement score
        opportunities.sort(key=lambda x: x['improvement_score'], reverse=True)

        return opportunities[:top_n]

    def _generate_recommendations(self, metrics: ParameterMetrics) -> List[str]:
        """Generate recommendations for improving a parameter's model"""
        recommendations = []

        if metrics.current_r2 and metrics.current_r2 < 0.60:
            recommendations.append("Model performance is low - consider retraining with more data")

        if metrics.training_samples < 500:
            recommendations.append(f"Only {metrics.training_samples} training samples - gather more data")

        if metrics.trend_direction == TrendDirection.DECLINING:
            recommendations.append("Performance is declining - investigate recent changes")

        if metrics.trend_direction == TrendDirection.VOLATILE:
            recommendations.append("High variance - consider feature engineering or regularization")

        if metrics.has_anomalies:
            recommendations.append("Anomalies detected - review training data quality")

        if not recommendations:
            recommendations.append("Model is performing well")

        return recommendations

    # ==========================================================================
    # ALERTING
    # ==========================================================================

    def _create_alert(self,
                     severity: AlertSeverity,
                     category: str,
                     parameter_name: Optional[str],
                     message: str,
                     details: Dict[str, Any]):
        """Create a new alert"""
        # Check if similar alert already exists
        for alert in self.active_alerts:
            if (not alert.resolved and
                alert.parameter_name == parameter_name and
                alert.category == category):
                return  # Don't create duplicate

        alert = QualityAlert(
            timestamp=time.time(),
            severity=severity,
            category=category,
            parameter_name=parameter_name,
            message=message,
            details=details
        )

        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Print to console
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }

        print(f"\n{severity_emoji[severity]} ALERT [{severity.value.upper()}]: {message}")
        if parameter_name:
            print(f"   Parameter: {parameter_name}")
        print(f"   Category: {category}")

    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved"""
        if 0 <= alert_index < len(self.active_alerts):
            alert = self.active_alerts[alert_index]
            alert.resolved = True
            alert.resolved_at = time.time()

    def get_active_alerts(self,
                         severity: Optional[AlertSeverity] = None) -> List[QualityAlert]:
        """Get list of active alerts, optionally filtered by severity"""
        alerts = [a for a in self.active_alerts if not a.resolved]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================

    def plot_parameter_history(self,
                               parameter_path: str,
                               metric_type: MetricType = MetricType.R_SQUARED,
                               save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plot historical metrics for a parameter.

        Args:
            parameter_path: Full parameter path
            metric_type: Type of metric to plot
            save_path: Where to save the plot (auto-generated if None)

        Returns:
            Path to saved plot, or None if visualization disabled
        """
        if not self.enable_visualization:
            return None

        if parameter_path not in self.parameter_metrics:
            print(f"Parameter {parameter_path} not found")
            return None

        metrics = self.parameter_metrics[parameter_path]

        # Get appropriate history
        if metric_type == MetricType.R_SQUARED:
            history = metrics.r2_history
            ylabel = "R² Score"
            title = f"R² Score History: {parameter_path}"
        elif metric_type == MetricType.MAE:
            history = metrics.mae_history
            ylabel = "Mean Absolute Error"
            title = f"MAE History: {parameter_path}"
        else:
            print(f"Metric type {metric_type} not supported for plotting")
            return None

        if not history:
            print(f"No history data for {parameter_path}")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        timestamps = [datetime.fromtimestamp(dp.timestamp) for dp in history]
        values = [dp.value for dp in history]

        ax.plot(timestamps, values, linewidth=2, label="Actual")

        # Add trend line if enough data
        if len(values) > 10 and NUMPY_AVAILABLE:
            x = np.arange(len(values))
            z = np.polyfit(x, values, 2)  # Quadratic fit
            p = np.poly1d(z)
            ax.plot(timestamps, p(x), "--", alpha=0.5, label="Trend")

        # Add quality threshold line for R²
        if metric_type == MetricType.R_SQUARED:
            ax.axhline(y=self.alert_threshold_r2, color='r', linestyle='--',
                      alpha=0.5, label=f"Alert Threshold ({self.alert_threshold_r2})")
            ax.axhline(y=0.75, color='g', linestyle='--',
                      alpha=0.3, label="Good Threshold (0.75)")

        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()

        # Save
        if save_path is None:
            save_path = self.storage_dir / f"{parameter_path.replace('.', '_')}_history.png"

        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        return save_path

    def plot_system_health(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plot overall system health over time.

        Args:
            save_path: Where to save the plot

        Returns:
            Path to saved plot, or None if visualization disabled
        """
        if not self.enable_visualization or not self.system_health_history:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        history = list(self.system_health_history)
        timestamps = [datetime.fromtimestamp(h.timestamp) for h in history]

        # Plot 1: Average R² over time
        ax = axes[0, 0]
        avg_r2 = [h.avg_r2_score for h in history]
        ax.plot(timestamps, avg_r2, linewidth=2, color='blue')
        ax.set_title("Average R² Score Over Time")
        ax.set_ylabel("R² Score")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Plot 2: Model quality distribution
        ax = axes[0, 1]
        excellent = [h.excellent_models for h in history]
        good = [h.good_models for h in history]
        poor = [h.poor_models for h in history]

        ax.plot(timestamps, excellent, label="Excellent (R²>0.90)", color='green', linewidth=2)
        ax.plot(timestamps, good, label="Good (R²>0.75)", color='blue', linewidth=2)
        ax.plot(timestamps, poor, label="Poor (R²<0.60)", color='red', linewidth=2)
        ax.set_title("Model Quality Distribution")
        ax.set_ylabel("Number of Models")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Plot 3: Reconstruction accuracy
        ax = axes[1, 0]
        if self.reconstruction_history:
            recon_times = [datetime.fromtimestamp(r.timestamp) for r in self.reconstruction_history]
            recon_acc = [r.reconstruction_accuracy for r in self.reconstruction_history]
            ax.plot(recon_times, recon_acc, linewidth=2, color='purple')
            ax.set_title("Reconstruction Accuracy Over Time")
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Plot 4: Overall health score
        ax = axes[1, 1]
        health_scores = [h.health_score for h in history]
        ax.plot(timestamps, health_scores, linewidth=2, color='darkgreen')
        ax.fill_between(timestamps, 0, health_scores, alpha=0.3, color='green')
        ax.set_title("Overall System Health Score")
        ax.set_ylabel("Health Score (0-100)")
        ax.set_ylim([0, 100])
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.3, label="Good (80)")
        ax.axhline(y=60, color='orange', linestyle='--', alpha=0.3, label="Warning (60)")
        ax.axhline(y=40, color='r', linestyle='--', alpha=0.3, label="Critical (40)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Format all x-axes
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)

        if save_path is None:
            save_path = self.storage_dir / "system_health_dashboard.png"

        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"📊 System health dashboard saved to {save_path}")

        return save_path

    def plot_category_comparison(self, save_path: Optional[Path] = None) -> Optional[Path]:
        """
        Plot comparison of different parameter categories.

        Returns:
            Path to saved plot
        """
        if not self.enable_visualization:
            return None

        # Group by category
        categories = defaultdict(list)
        for metrics in self.parameter_metrics.values():
            if metrics.current_r2 is not None:
                categories[metrics.category].append(metrics.current_r2)

        if not categories:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Box plot of R² by category
        cat_names = list(categories.keys())
        cat_data = [categories[cat] for cat in cat_names]

        ax1.boxplot(cat_data, labels=cat_names)
        ax1.set_title("R² Score Distribution by Category")
        ax1.set_ylabel("R² Score")
        ax1.set_xlabel("Category")
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Bar chart of average R² by category
        cat_avgs = {cat: sum(scores)/len(scores) for cat, scores in categories.items()}
        sorted_cats = sorted(cat_avgs.items(), key=lambda x: x[1], reverse=True)

        names = [c[0] for c in sorted_cats]
        avgs = [c[1] for c in sorted_cats]
        colors = ['green' if a > 0.75 else 'orange' if a > 0.60 else 'red' for a in avgs]

        ax2.bar(names, avgs, color=colors, alpha=0.7)
        ax2.set_title("Average R² by Category")
        ax2.set_ylabel("Average R² Score")
        ax2.set_xlabel("Category")
        ax2.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label="Good (0.75)")
        ax2.axhline(y=0.60, color='orange', linestyle='--', alpha=0.5, label="Acceptable (0.60)")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)

        if save_path is None:
            save_path = self.storage_dir / "category_comparison.png"

        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        return save_path

    # ==========================================================================
    # PERSISTENCE
    # ==========================================================================

    def save_state(self):
        """Save dashboard state to disk"""
        state = {
            "dashboard_metadata": {
                "version": "1.0",
                "last_updated": time.time(),
                "total_measurements": self.total_measurements,
                "dashboard_start_time": self.dashboard_start_time
            },
            "parameters": {
                path: {
                    "name": m.parameter_name,
                    "path": m.parameter_path,
                    "category": m.category,
                    "current_r2": m.current_r2,
                    "current_mae": m.current_mae,
                    "current_rmse": m.current_rmse,
                    "current_correlation": m.current_correlation,
                    "training_samples": m.training_samples,
                    "last_trained": m.last_trained,
                    "training_time_seconds": m.training_time_seconds,
                    "inference_time_ms": m.inference_time_ms,
                    "quality_level": m.quality_level.value,
                    "trend_direction": m.trend_direction.value,
                    "r2_history": [{"timestamp": dp.timestamp, "value": dp.value}
                                  for dp in list(m.r2_history)[-100:]],  # Last 100
                    "feature_importance": m.feature_importance_scores
                }
                for path, m in self.parameter_metrics.items()
            },
            "system_health_history": [
                {
                    "timestamp": h.timestamp,
                    "total_parameters": h.total_parameters,
                    "avg_r2_score": h.avg_r2_score,
                    "health_score": h.health_score
                }
                for h in list(self.system_health_history)[-100:]
            ],
            "active_alerts": [
                {
                    "timestamp": a.timestamp,
                    "severity": a.severity.value,
                    "category": a.category,
                    "parameter_name": a.parameter_name,
                    "message": a.message,
                    "resolved": a.resolved
                }
                for a in self.active_alerts
            ]
        }

        state_file = self.storage_dir / "dashboard_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"💾 Dashboard state saved to {state_file}")

    def _load_state(self):
        """Load dashboard state from disk"""
        state_file = self.storage_dir / "dashboard_state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore metadata
            meta = state.get("dashboard_metadata", {})
            self.total_measurements = meta.get("total_measurements", 0)
            self.dashboard_start_time = meta.get("dashboard_start_time", time.time())

            # Restore parameters
            for path, data in state.get("parameters", {}).items():
                metrics = ParameterMetrics(
                    parameter_name=data["name"],
                    parameter_path=data["path"],
                    category=data["category"],
                    current_r2=data.get("current_r2"),
                    current_mae=data.get("current_mae"),
                    current_rmse=data.get("current_rmse"),
                    current_correlation=data.get("current_correlation"),
                    training_samples=data.get("training_samples", 0),
                    last_trained=data.get("last_trained"),
                    training_time_seconds=data.get("training_time_seconds"),
                    inference_time_ms=data.get("inference_time_ms")
                )

                # Restore history
                for dp in data.get("r2_history", []):
                    metrics.r2_history.append(MetricDataPoint(dp["timestamp"], dp["value"]))

                metrics.feature_importance_scores = data.get("feature_importance", {})
                metrics.quality_level = QualityLevel(data.get("quality_level", "acceptable"))
                metrics.trend_direction = TrendDirection(data.get("trend_direction", "stable"))

                self.parameter_metrics[path] = metrics

            print(f"✓ Loaded {len(self.parameter_metrics)} parameters from saved state")

        except Exception as e:
            print(f"⚠️  Could not load saved state: {e}")

    # ==========================================================================
    # REPORTING
    # ==========================================================================

    def generate_report(self,
                       report_type: str = "full",
                       save_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive text report.

        Args:
            report_type: "full", "summary", or "problems"
            save_path: Where to save the report

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("QUALITY METRICS DASHBOARD - REPORT")
        lines.append("Musical Program Synthesis - Agent 31")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Dashboard Uptime: {(time.time() - self.dashboard_start_time) / 3600:.1f} hours")
        lines.append("")

        # System health
        health = self.compute_system_health()
        lines.append("SYSTEM HEALTH")
        lines.append("-" * 80)
        lines.append(f"Overall Health Score: {health.health_score:.1f}/100")
        lines.append(f"Total Parameters: {health.total_parameters}")
        lines.append(f"Trained Models: {health.trained_parameters} ({health.trained_parameters/max(1,health.total_parameters)*100:.1f}%)")
        lines.append(f"")
        lines.append(f"Model Quality Distribution:")
        lines.append(f"  Excellent (R²>0.90): {health.excellent_models:3d} ({health.excellent_models/max(1,health.trained_parameters)*100:.1f}%)")
        lines.append(f"  Good (R²>0.75):      {health.good_models:3d} ({health.good_models/max(1,health.trained_parameters)*100:.1f}%)")
        lines.append(f"  Poor (R²<0.60):      {health.poor_models:3d} ({health.poor_models/max(1,health.trained_parameters)*100:.1f}%)")
        lines.append(f"")
        lines.append(f"R² Statistics:")
        lines.append(f"  Average: {health.avg_r2_score:.3f}")
        lines.append(f"  Median:  {health.median_r2_score:.3f}")
        lines.append(f"  Range:   [{health.min_r2_score:.3f}, {health.max_r2_score:.3f}]")
        lines.append(f"")
        lines.append(f"Reconstruction: {health.avg_reconstruction_accuracy:.1%} accuracy ({health.reconstruction_samples} samples)")
        lines.append(f"Alerts: {health.active_alerts} active ({health.critical_alerts} critical)")
        lines.append("")

        if report_type in ["full", "problems"]:
            # Active alerts
            active_alerts = self.get_active_alerts()
            if active_alerts:
                lines.append("ACTIVE ALERTS")
                lines.append("-" * 80)
                for alert in active_alerts[:20]:  # Top 20
                    lines.append(f"[{alert.severity.value.upper()}] {alert.message}")
                    if alert.parameter_name:
                        lines.append(f"  Parameter: {alert.parameter_name}")
                    lines.append(f"  Time: {alert.datetime_str} ({alert.age_seconds/3600:.1f}h ago)")
                    lines.append("")

            # Improvement opportunities
            opportunities = self.identify_improvement_opportunities(top_n=15)
            if opportunities:
                lines.append("TOP IMPROVEMENT OPPORTUNITIES")
                lines.append("-" * 80)
                for i, opp in enumerate(opportunities, 1):
                    lines.append(f"{i}. {opp['parameter']}")
                    lines.append(f"   Current R²: {opp['current_r2']:.3f} | Quality: {opp['quality']} | Trend: {opp['trend']}")
                    lines.append(f"   Priority Score: {opp['improvement_score']:.2f}")
                    for rec in opp['recommendations']:
                        lines.append(f"   → {rec}")
                    lines.append("")

        if report_type == "full":
            # Category summaries
            categories = set(m.category for m in self.parameter_metrics.values())
            lines.append("CATEGORY SUMMARIES")
            lines.append("-" * 80)
            for category in sorted(categories):
                summary = self.get_category_summary(category)
                if "error" not in summary:
                    lines.append(f"{category.upper()}: {summary['trained_parameters']}/{summary['total_parameters']} trained, avg R² = {summary['average_r2']:.3f}")

        report_text = "\n".join(lines)

        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to {save_path}")

        return report_text


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def create_demo_dashboard():
    """Create a demo dashboard with sample data"""
    dashboard = QualityMetricsDashboard(
        storage_dir=Path("./demo_monitoring"),
        enable_visualization=True
    )

    # Register some parameters
    parameters = [
        ("voicing_type", "harmony.voicing.type", "harmony"),
        ("voicing_spread", "harmony.voicing.spread", "harmony"),
        ("swing_ratio", "rhythm.swing.ratio", "rhythm"),
        ("note_density", "melody.density", "melody"),
        ("walking_pattern", "bass.walking.pattern", "bass"),
    ]

    for name, path, category in parameters:
        dashboard.register_parameter(name, path, category)

    # Simulate some measurements
    import random
    for i in range(100):
        for name, path, category in parameters:
            # Simulate improving model
            base_r2 = 0.70 + random.random() * 0.2
            noise = random.gauss(0, 0.05)
            improvement = i * 0.001
            r2 = min(0.95, base_r2 + improvement + noise)

            dashboard.update_parameter_metrics(
                parameter_path=path,
                r2_score=r2,
                mae=random.uniform(0.05, 0.15),
                training_samples=500 + i * 5
            )

    # Add some reconstruction metrics
    for i in range(50):
        dashboard.update_reconstruction_metrics(
            reconstruction_accuracy=0.80 + random.random() * 0.15,
            parameter_recovery_rate=0.85 + random.random() * 0.10,
            note_accuracy=0.90 + random.random() * 0.08
        )

    return dashboard


if __name__ == "__main__":
    print("Quality Metrics Dashboard - Agent 31")
    print("=" * 80)
    print()
    print("Creating demo dashboard...")

    dashboard = create_demo_dashboard()

    print("\nGenerating reports and visualizations...")

    # Generate health report
    health = dashboard.compute_system_health()
    print(f"\n✓ System Health Score: {health.health_score:.1f}/100")

    # Generate full report
    report = dashboard.generate_report(report_type="full")
    print("\n" + report)

    # Generate visualizations
    if MATPLOTLIB_AVAILABLE:
        dashboard.plot_system_health()
        dashboard.plot_category_comparison()
        dashboard.plot_parameter_history("harmony.voicing.type")
        print("\n✓ Visualizations generated")

    # Save state
    dashboard.save_state()

    print("\n✓ Demo complete!")
    print(f"   Data saved to: {dashboard.storage_dir}")
