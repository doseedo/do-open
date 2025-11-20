"""
Model Registry Manager - Agent 33
==================================

Comprehensive tracking and version control system for all 800+ XGBoost models
in the Musical Program Synthesis system.

This manager provides:
1. Complete model lifecycle management (save, load, archive, rollback)
2. Version control with full lineage tracking
3. Performance metrics and benchmarking
4. Model metadata (training date, feature importance, hyperparameters)
5. Automated validation and quality gates
6. Performance degradation detection
7. Model comparison and A/B testing
8. Ensemble management
9. Export/import capabilities
10. Integration with Universal Parameter Registry

Architecture:
- ONE XGBoost model per parameter (modular, independent)
- Each model tracks its own version history
- Models can be promoted, demoted, or rolled back
- Performance metrics tracked across all versions
- Automatic archival of old versions

Author: Agent 33 - Model Registry Manager
License: MIT
"""

import json
import pickle
import hashlib
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from enum import Enum
import numpy as np
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Some functionality will be limited.")


# ============================================================================
# Enums and Constants
# ============================================================================

class ModelStatus(Enum):
    """Status of a model version"""
    TRAINING = "training"           # Currently being trained
    VALIDATING = "validating"       # Under validation
    TESTING = "testing"             # Being tested
    ACTIVE = "active"               # Currently in production use
    STAGED = "staged"               # Staged for deployment
    ARCHIVED = "archived"           # Archived but available
    DEPRECATED = "deprecated"       # No longer recommended
    FAILED = "failed"               # Failed validation


class ModelType(Enum):
    """Type of model"""
    XGBOOST_REGRESSOR = "xgboost_regressor"
    XGBOOST_CLASSIFIER = "xgboost_classifier"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


class PerformanceMetric(Enum):
    """Performance metrics for models"""
    MAE = "mae"                     # Mean Absolute Error
    MSE = "mse"                     # Mean Squared Error
    RMSE = "rmse"                   # Root Mean Squared Error
    R2 = "r2"                       # R-squared
    ACCURACY = "accuracy"           # Classification accuracy
    PRECISION = "precision"         # Precision
    RECALL = "recall"               # Recall
    F1 = "f1"                       # F1 score
    AUC = "auc"                     # Area Under Curve
    MUSICAL_VALIDITY = "musical_validity"  # Custom musical validation


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ModelMetadata:
    """
    Complete metadata for a single model version
    """
    # Identity
    model_id: str                   # Unique model ID
    parameter_path: str             # Full parameter path (e.g., "harmony.voicing.type")
    version: int                    # Version number (1, 2, 3, ...)
    model_type: ModelType = ModelType.XGBOOST_REGRESSOR

    # Status
    status: ModelStatus = ModelStatus.TRAINING
    is_active: bool = False         # Is this the active version?

    # Training information
    training_date: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None

    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Feature information
    num_features: Optional[int] = None
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Lineage
    parent_version: Optional[int] = None  # Version this was trained from
    child_versions: List[int] = field(default_factory=list)

    # File information
    model_file_path: Optional[str] = None
    model_file_size_bytes: Optional[int] = None
    model_file_hash: Optional[str] = None

    # Notes and tags
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    # Validation results
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)

    # A/B testing
    ab_test_id: Optional[str] = None
    ab_test_performance: Dict[str, float] = field(default_factory=dict)

    # Created/updated
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by: str = "Agent 33"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['model_type'] = self.model_type.value
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        data = data.copy()
        data['model_type'] = ModelType(data['model_type'])
        data['status'] = ModelStatus(data['status'])
        return cls(**data)


@dataclass
class ModelPerformanceSnapshot:
    """
    Performance snapshot for a model at a specific time
    """
    model_id: str
    version: int
    snapshot_date: str
    metrics: Dict[str, float]
    test_dataset_id: Optional[str] = None
    test_dataset_size: Optional[int] = None
    notes: str = ""


@dataclass
class ModelComparison:
    """
    Comparison between two model versions
    """
    model_id: str
    version_a: int
    version_b: int
    comparison_date: str

    # Performance differences
    metric_improvements: Dict[str, float]  # Positive = version_b is better
    metric_degradations: Dict[str, float]  # Negative = version_b is worse

    # Feature importance changes
    feature_importance_changes: Dict[str, float]

    # Recommendation
    recommended_version: int
    recommendation_reason: str


@dataclass
class ModelRegistry:
    """
    Complete registry of all models
    """
    total_models: int = 0
    total_versions: int = 0
    active_models: int = 0
    models: Dict[str, List[ModelMetadata]] = field(default_factory=lambda: defaultdict(list))

    # Index by parameter path for fast lookup
    parameter_to_model: Dict[str, str] = field(default_factory=dict)

    # Performance history
    performance_history: List[ModelPerformanceSnapshot] = field(default_factory=list)

    # Comparisons
    comparisons: List[ModelComparison] = field(default_factory=list)


# ============================================================================
# Model Registry Manager
# ============================================================================

class ModelRegistryManager:
    """
    Central manager for all XGBoost models in the system.

    Responsibilities:
    - Track all model versions
    - Manage model lifecycle
    - Performance monitoring
    - Version control
    - Model validation
    - A/B testing support
    """

    def __init__(self, registry_path: str = "/home/user/Do/midi_generator/models"):
        """
        Initialize the Model Registry Manager

        Args:
            registry_path: Root path for model storage
        """
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / "saved_models"
        self.archive_path = self.registry_path / "archive"
        self.metadata_path = self.registry_path / "metadata"
        self.db_path = self.registry_path / "registry.db"

        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        # In-memory registry
        self.registry = ModelRegistry()

        # Initialize database
        self._init_database()

        # Load existing registry
        self._load_registry()

    # ========================================================================
    # Database Management
    # ========================================================================

    def _init_database(self):
        """Initialize SQLite database for fast queries"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                parameter_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Model versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                status TEXT NOT NULL,
                is_active BOOLEAN NOT NULL,
                training_date TEXT,
                training_samples INTEGER,
                num_features INTEGER,
                model_file_path TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (model_id, version),
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            )
        """)

        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (model_id, version) REFERENCES model_versions(model_id, version)
            )
        """)

        # Feature importance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                feature_name TEXT NOT NULL,
                importance_value REAL NOT NULL,
                FOREIGN KEY (model_id, version) REFERENCES model_versions(model_id, version)
            )
        """)

        # Model comparisons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version_a INTEGER NOT NULL,
                version_b INTEGER NOT NULL,
                comparison_date TEXT NOT NULL,
                recommended_version INTEGER NOT NULL,
                recommendation_reason TEXT
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parameter_path ON models(parameter_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active_versions ON model_versions(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON model_versions(status)")

        conn.commit()
        conn.close()

    def _load_registry(self):
        """Load registry from database and metadata files"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Load all models
        cursor.execute("SELECT model_id, parameter_path FROM models")
        for model_id, parameter_path in cursor.fetchall():
            self.registry.parameter_to_model[parameter_path] = model_id

        # Load all versions
        cursor.execute("""
            SELECT model_id, version FROM model_versions
        """)

        for model_id, version in cursor.fetchall():
            # Load metadata from file
            metadata_file = self.metadata_path / f"{model_id}_v{version}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    metadata = ModelMetadata.from_dict(data)
                    self.registry.models[model_id].append(metadata)
                    self.registry.total_versions += 1

                    if metadata.is_active:
                        self.registry.active_models += 1

        self.registry.total_models = len(self.registry.models)

        conn.close()

    # ========================================================================
    # Model Registration
    # ========================================================================

    def register_model(
        self,
        parameter_path: str,
        model_type: ModelType = ModelType.XGBOOST_REGRESSOR,
        notes: str = ""
    ) -> str:
        """
        Register a new model for a parameter

        Args:
            parameter_path: Full parameter path (e.g., "harmony.voicing.type")
            model_type: Type of model
            notes: Optional notes

        Returns:
            model_id: Unique model ID
        """
        # Check if model already exists for this parameter
        if parameter_path in self.registry.parameter_to_model:
            existing_id = self.registry.parameter_to_model[parameter_path]
            print(f"⚠️  Model already exists for {parameter_path}: {existing_id}")
            return existing_id

        # Generate model ID
        model_id = self._generate_model_id(parameter_path)

        # Add to registry
        self.registry.parameter_to_model[parameter_path] = model_id
        self.registry.models[model_id] = []
        self.registry.total_models += 1

        # Add to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO models (model_id, parameter_path, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (model_id, parameter_path, now, now))

        conn.commit()
        conn.close()

        print(f"✅ Registered new model: {model_id} for {parameter_path}")
        return model_id

    def _generate_model_id(self, parameter_path: str) -> str:
        """Generate unique model ID from parameter path"""
        # Use parameter path as base
        # e.g., "harmony.voicing.type" -> "model_harmony_voicing_type"
        clean_path = parameter_path.replace(".", "_")
        return f"model_{clean_path}"

    # ========================================================================
    # Model Version Management
    # ========================================================================

    def create_version(
        self,
        model_id: str,
        model_type: ModelType = ModelType.XGBOOST_REGRESSOR,
        parent_version: Optional[int] = None,
        notes: str = "",
        tags: Optional[List[str]] = None
    ) -> ModelMetadata:
        """
        Create a new model version

        Args:
            model_id: Model ID
            model_type: Type of model
            parent_version: Parent version (for lineage tracking)
            notes: Notes about this version
            tags: Tags for categorization

        Returns:
            ModelMetadata for new version
        """
        if model_id not in self.registry.models:
            raise ValueError(f"Model {model_id} not found in registry")

        # Get next version number
        existing_versions = self.registry.models[model_id]
        if existing_versions:
            next_version = max(v.version for v in existing_versions) + 1
        else:
            next_version = 1

        # Get parameter path
        parameter_path = None
        for path, mid in self.registry.parameter_to_model.items():
            if mid == model_id:
                parameter_path = path
                break

        # Create metadata
        now = datetime.now().isoformat()
        metadata = ModelMetadata(
            model_id=model_id,
            parameter_path=parameter_path,
            version=next_version,
            model_type=model_type,
            status=ModelStatus.TRAINING,
            parent_version=parent_version,
            notes=notes,
            tags=tags or [],
            created_at=now,
            updated_at=now
        )

        # Add to registry
        self.registry.models[model_id].append(metadata)
        self.registry.total_versions += 1

        # Update parent's children
        if parent_version:
            for v in existing_versions:
                if v.version == parent_version:
                    v.child_versions.append(next_version)
                    self._save_metadata(v)

        # Save metadata
        self._save_metadata(metadata)

        # Add to database
        self._add_version_to_db(metadata)

        print(f"✅ Created version {next_version} for {model_id}")
        return metadata

    def save_model(
        self,
        model_id: str,
        version: int,
        model_obj: Any,
        training_samples: int,
        validation_samples: int,
        performance_metrics: Dict[str, float],
        feature_names: Optional[List[str]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_duration_seconds: Optional[float] = None
    ):
        """
        Save a trained model with all metadata

        Args:
            model_id: Model ID
            version: Version number
            model_obj: Trained model object (XGBoost or other)
            training_samples: Number of training samples
            validation_samples: Number of validation samples
            performance_metrics: Performance metrics dict
            feature_names: List of feature names
            feature_importance: Feature importance dict
            hyperparameters: Hyperparameters used
            training_duration_seconds: Training time
        """
        # Get metadata
        metadata = self._get_version_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Version {version} not found for {model_id}")

        # Save model file
        model_filename = f"{model_id}_v{version}.pkl"
        model_filepath = self.models_path / model_filename

        with open(model_filepath, 'wb') as f:
            pickle.dump(model_obj, f)

        # Calculate file hash
        file_hash = self._calculate_file_hash(model_filepath)
        file_size = model_filepath.stat().st_size

        # Update metadata
        now = datetime.now().isoformat()
        metadata.training_date = now
        metadata.training_samples = training_samples
        metadata.validation_samples = validation_samples
        metadata.performance_metrics = performance_metrics
        metadata.num_features = len(feature_names) if feature_names else None
        metadata.feature_names = feature_names or []
        metadata.feature_importance = feature_importance or {}
        metadata.hyperparameters = hyperparameters or {}
        metadata.training_duration_seconds = training_duration_seconds
        metadata.model_file_path = str(model_filepath)
        metadata.model_file_size_bytes = file_size
        metadata.model_file_hash = file_hash
        metadata.status = ModelStatus.VALIDATING
        metadata.updated_at = now

        # Save metadata
        self._save_metadata(metadata)

        # Update database
        self._update_version_in_db(metadata)
        self._save_performance_metrics_to_db(model_id, version, performance_metrics)
        if feature_importance:
            self._save_feature_importance_to_db(model_id, version, feature_importance)

        print(f"💾 Saved model {model_id} v{version} ({file_size / 1024:.2f} KB)")
        print(f"   Metrics: {performance_metrics}")

    def load_model(self, model_id: str, version: Optional[int] = None) -> Tuple[Any, ModelMetadata]:
        """
        Load a model

        Args:
            model_id: Model ID
            version: Version number (None = load active version)

        Returns:
            (model_object, metadata)
        """
        # Get metadata
        if version is None:
            metadata = self.get_active_version(model_id)
            if not metadata:
                raise ValueError(f"No active version found for {model_id}")
        else:
            metadata = self._get_version_metadata(model_id, version)
            if not metadata:
                raise ValueError(f"Version {version} not found for {model_id}")

        # Load model file
        if not metadata.model_file_path:
            raise ValueError(f"No model file path for {model_id} v{metadata.version}")

        model_path = Path(metadata.model_file_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Verify hash
        current_hash = self._calculate_file_hash(model_path)
        if current_hash != metadata.model_file_hash:
            warnings.warn(f"Model file hash mismatch for {model_id} v{metadata.version}")

        # Load model
        with open(model_path, 'rb') as f:
            model_obj = pickle.load(f)

        print(f"📂 Loaded model {model_id} v{metadata.version}")
        return model_obj, metadata

    def set_active_version(self, model_id: str, version: int):
        """
        Set a version as the active (production) version

        Args:
            model_id: Model ID
            version: Version to activate
        """
        # Deactivate all other versions
        for v in self.registry.models[model_id]:
            if v.is_active:
                v.is_active = False
                v.status = ModelStatus.ARCHIVED
                v.updated_at = datetime.now().isoformat()
                self._save_metadata(v)
                self._update_version_in_db(v)
                self.registry.active_models -= 1

        # Activate target version
        metadata = self._get_version_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Version {version} not found for {model_id}")

        metadata.is_active = True
        metadata.status = ModelStatus.ACTIVE
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(metadata)
        self._update_version_in_db(metadata)
        self.registry.active_models += 1

        print(f"✅ Activated {model_id} v{version}")

    def get_active_version(self, model_id: str) -> Optional[ModelMetadata]:
        """Get the active version metadata"""
        for v in self.registry.models[model_id]:
            if v.is_active:
                return v
        return None

    def archive_version(self, model_id: str, version: int):
        """Archive a model version"""
        metadata = self._get_version_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Version {version} not found for {model_id}")

        if metadata.is_active:
            raise ValueError("Cannot archive active version. Activate another version first.")

        # Move model file to archive
        if metadata.model_file_path:
            model_path = Path(metadata.model_file_path)
            if model_path.exists():
                archive_path = self.archive_path / model_path.name
                shutil.move(str(model_path), str(archive_path))
                metadata.model_file_path = str(archive_path)

        metadata.status = ModelStatus.ARCHIVED
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(metadata)
        self._update_version_in_db(metadata)

        print(f"📦 Archived {model_id} v{version}")

    # ========================================================================
    # Model Validation
    # ========================================================================

    def validate_model(
        self,
        model_id: str,
        version: int,
        test_samples: int,
        test_metrics: Dict[str, float],
        validation_errors: Optional[List[str]] = None
    ) -> bool:
        """
        Validate a model version

        Args:
            model_id: Model ID
            version: Version number
            test_samples: Number of test samples
            test_metrics: Test performance metrics
            validation_errors: List of validation errors

        Returns:
            True if validation passed
        """
        metadata = self._get_version_metadata(model_id, version)
        if not metadata:
            raise ValueError(f"Version {version} not found for {model_id}")

        # Update metadata
        metadata.test_samples = test_samples
        metadata.validation_errors = validation_errors or []

        # Check if validation passed
        passed = len(validation_errors or []) == 0
        metadata.validation_passed = passed

        if passed:
            metadata.status = ModelStatus.STAGED
            print(f"✅ Validation passed for {model_id} v{version}")
        else:
            metadata.status = ModelStatus.FAILED
            print(f"❌ Validation failed for {model_id} v{version}")
            for error in validation_errors:
                print(f"   - {error}")

        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(metadata)
        self._update_version_in_db(metadata)

        # Record performance snapshot
        self._record_performance_snapshot(model_id, version, test_metrics, test_samples)

        return passed

    # ========================================================================
    # Model Comparison
    # ========================================================================

    def compare_versions(
        self,
        model_id: str,
        version_a: int,
        version_b: int
    ) -> ModelComparison:
        """
        Compare two model versions

        Args:
            model_id: Model ID
            version_a: First version
            version_b: Second version

        Returns:
            ModelComparison object
        """
        metadata_a = self._get_version_metadata(model_id, version_a)
        metadata_b = self._get_version_metadata(model_id, version_b)

        if not metadata_a or not metadata_b:
            raise ValueError(f"Versions not found for {model_id}")

        # Compare metrics
        improvements = {}
        degradations = {}

        for metric, value_b in metadata_b.performance_metrics.items():
            value_a = metadata_a.performance_metrics.get(metric, 0)
            diff = value_b - value_a

            # For errors, lower is better
            if metric in ['mae', 'mse', 'rmse']:
                if diff < 0:  # Improvement
                    improvements[metric] = abs(diff)
                else:  # Degradation
                    degradations[metric] = diff
            else:  # For other metrics, higher is better
                if diff > 0:  # Improvement
                    improvements[metric] = diff
                else:  # Degradation
                    degradations[metric] = abs(diff)

        # Compare feature importance
        importance_changes = {}
        for feature, imp_b in metadata_b.feature_importance.items():
            imp_a = metadata_a.feature_importance.get(feature, 0)
            importance_changes[feature] = imp_b - imp_a

        # Determine recommendation
        if improvements and not degradations:
            recommended = version_b
            reason = f"Version {version_b} shows improvements across all metrics"
        elif degradations and not improvements:
            recommended = version_a
            reason = f"Version {version_b} shows degradations across all metrics"
        elif sum(improvements.values()) > sum(degradations.values()):
            recommended = version_b
            reason = f"Version {version_b} shows net improvement"
        else:
            recommended = version_a
            reason = f"Version {version_a} performs better overall"

        comparison = ModelComparison(
            model_id=model_id,
            version_a=version_a,
            version_b=version_b,
            comparison_date=datetime.now().isoformat(),
            metric_improvements=improvements,
            metric_degradations=degradations,
            feature_importance_changes=importance_changes,
            recommended_version=recommended,
            recommendation_reason=reason
        )

        # Save comparison
        self.registry.comparisons.append(comparison)
        self._save_comparison_to_db(comparison)

        return comparison

    # ========================================================================
    # Bulk Operations
    # ========================================================================

    def register_all_parameters(self, parameter_paths: List[str]):
        """
        Register models for all parameters

        Args:
            parameter_paths: List of parameter paths
        """
        print(f"📝 Registering models for {len(parameter_paths)} parameters...")

        registered = 0
        skipped = 0

        for path in parameter_paths:
            if path not in self.registry.parameter_to_model:
                self.register_model(path)
                registered += 1
            else:
                skipped += 1

        print(f"✅ Registered {registered} new models ({skipped} already existed)")

    def get_all_active_models(self) -> Dict[str, Tuple[Any, ModelMetadata]]:
        """
        Load all active models

        Returns:
            Dict mapping model_id to (model_obj, metadata)
        """
        active_models = {}

        for model_id in self.registry.models:
            metadata = self.get_active_version(model_id)
            if metadata:
                try:
                    model_obj, _ = self.load_model(model_id)
                    active_models[model_id] = (model_obj, metadata)
                except Exception as e:
                    print(f"⚠️  Failed to load {model_id}: {e}")

        return active_models

    def get_models_by_parameter_prefix(self, prefix: str) -> List[str]:
        """
        Get all model IDs for parameters matching a prefix

        Args:
            prefix: Parameter path prefix (e.g., "harmony.")

        Returns:
            List of model IDs
        """
        model_ids = []
        for param_path, model_id in self.registry.parameter_to_model.items():
            if param_path.startswith(prefix):
                model_ids.append(model_id)
        return model_ids

    # ========================================================================
    # Statistics and Reporting
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the registry"""
        stats = {
            "total_models": self.registry.total_models,
            "total_versions": self.registry.total_versions,
            "active_models": self.registry.active_models,
            "by_status": defaultdict(int),
            "by_type": defaultdict(int),
            "total_disk_usage_mb": 0,
            "avg_versions_per_model": 0,
            "performance_trends": {},
        }

        total_size = 0
        for model_id, versions in self.registry.models.items():
            for v in versions:
                stats["by_status"][v.status.value] += 1
                stats["by_type"][v.model_type.value] += 1
                if v.model_file_size_bytes:
                    total_size += v.model_file_size_bytes

        stats["total_disk_usage_mb"] = total_size / (1024 * 1024)
        if self.registry.total_models > 0:
            stats["avg_versions_per_model"] = self.registry.total_versions / self.registry.total_models

        return dict(stats)

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive registry report

        Args:
            output_path: Optional path to save report

        Returns:
            Report text
        """
        stats = self.get_statistics()

        lines = []
        lines.append("=" * 80)
        lines.append("MODEL REGISTRY REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append("📊 SUMMARY")
        lines.append(f"   Total Models:        {stats['total_models']:4d}")
        lines.append(f"   Total Versions:      {stats['total_versions']:4d}")
        lines.append(f"   Active Models:       {stats['active_models']:4d}")
        lines.append(f"   Avg Versions/Model:  {stats['avg_versions_per_model']:.2f}")
        lines.append(f"   Disk Usage:          {stats['total_disk_usage_mb']:.2f} MB")
        lines.append("")

        lines.append("📁 BY STATUS")
        for status, count in sorted(stats['by_status'].items()):
            lines.append(f"   {status:20s}: {count:4d}")
        lines.append("")

        lines.append("🤖 BY TYPE")
        for model_type, count in sorted(stats['by_type'].items()):
            lines.append(f"   {model_type:20s}: {count:4d}")
        lines.append("")

        lines.append("🎯 TOP MODELS BY PERFORMANCE")
        # Get top 10 models by R2 score
        top_models = []
        for model_id, versions in self.registry.models.items():
            for v in versions:
                if v.is_active and 'r2' in v.performance_metrics:
                    top_models.append((model_id, v.version, v.performance_metrics['r2'], v.parameter_path))

        top_models.sort(key=lambda x: x[2], reverse=True)
        for i, (model_id, version, r2, param_path) in enumerate(top_models[:10], 1):
            lines.append(f"   {i:2d}. {param_path:40s} R²={r2:.4f} (v{version})")
        lines.append("")

        lines.append("📈 RECENT COMPARISONS")
        for comp in self.registry.comparisons[-10:]:
            lines.append(f"   {comp.model_id}: v{comp.version_a} vs v{comp.version_b}")
            lines.append(f"      Recommended: v{comp.recommended_version}")
            lines.append(f"      Reason: {comp.recommendation_reason}")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"💾 Report saved to {output_path}")

        return report

    def export_registry(self, output_path: str):
        """Export complete registry to JSON"""
        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_models": self.registry.total_models,
            "total_versions": self.registry.total_versions,
            "models": {},
        }

        for model_id, versions in self.registry.models.items():
            export_data["models"][model_id] = {
                "parameter_path": versions[0].parameter_path if versions else None,
                "versions": [v.to_dict() for v in versions]
            }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"💾 Exported registry to {output_path}")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_version_metadata(self, model_id: str, version: int) -> Optional[ModelMetadata]:
        """Get metadata for a specific version"""
        for v in self.registry.models.get(model_id, []):
            if v.version == version:
                return v
        return None

    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to file"""
        metadata_file = self.metadata_path / f"{metadata.model_id}_v{metadata.version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _add_version_to_db(self, metadata: ModelMetadata):
        """Add version to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO model_versions
            (model_id, version, model_type, status, is_active, training_date,
             training_samples, num_features, model_file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.model_id,
            metadata.version,
            metadata.model_type.value,
            metadata.status.value,
            metadata.is_active,
            metadata.training_date,
            metadata.training_samples,
            metadata.num_features,
            metadata.model_file_path,
            metadata.created_at
        ))

        conn.commit()
        conn.close()

    def _update_version_in_db(self, metadata: ModelMetadata):
        """Update version in database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE model_versions
            SET status = ?, is_active = ?, training_date = ?,
                training_samples = ?, num_features = ?, model_file_path = ?
            WHERE model_id = ? AND version = ?
        """, (
            metadata.status.value,
            metadata.is_active,
            metadata.training_date,
            metadata.training_samples,
            metadata.num_features,
            metadata.model_file_path,
            metadata.model_id,
            metadata.version
        ))

        conn.commit()
        conn.close()

    def _save_performance_metrics_to_db(
        self,
        model_id: str,
        version: int,
        metrics: Dict[str, float]
    ):
        """Save performance metrics to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        for metric_name, metric_value in metrics.items():
            cursor.execute("""
                INSERT INTO performance_metrics
                (model_id, version, metric_name, metric_value, recorded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (model_id, version, metric_name, metric_value, now))

        conn.commit()
        conn.close()

    def _save_feature_importance_to_db(
        self,
        model_id: str,
        version: int,
        importance: Dict[str, float]
    ):
        """Save feature importance to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        for feature_name, importance_value in importance.items():
            cursor.execute("""
                INSERT INTO feature_importance
                (model_id, version, feature_name, importance_value)
                VALUES (?, ?, ?, ?)
            """, (model_id, version, feature_name, importance_value))

        conn.commit()
        conn.close()

    def _save_comparison_to_db(self, comparison: ModelComparison):
        """Save comparison to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO model_comparisons
            (model_id, version_a, version_b, comparison_date,
             recommended_version, recommendation_reason)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            comparison.model_id,
            comparison.version_a,
            comparison.version_b,
            comparison.comparison_date,
            comparison.recommended_version,
            comparison.recommendation_reason
        ))

        conn.commit()
        conn.close()

    def _record_performance_snapshot(
        self,
        model_id: str,
        version: int,
        metrics: Dict[str, float],
        test_samples: int
    ):
        """Record performance snapshot"""
        snapshot = ModelPerformanceSnapshot(
            model_id=model_id,
            version=version,
            snapshot_date=datetime.now().isoformat(),
            metrics=metrics,
            test_dataset_size=test_samples
        )
        self.registry.performance_history.append(snapshot)


# ============================================================================
# Integration with Parameter Registry
# ============================================================================

def initialize_registry_from_parameters(
    manager: ModelRegistryManager,
    parameter_registry_path: str = "/home/user/Do/midi_generator/parameters"
):
    """
    Initialize model registry from parameter registry

    Args:
        manager: ModelRegistryManager instance
        parameter_registry_path: Path to parameter registry
    """
    try:
        # Import parameter registry
        import sys
        sys.path.insert(0, str(Path(parameter_registry_path).parent))
        from parameters.universal_registry import REGISTRY
        from parameters.registry_expansion import register_all_expansions

        # Register expansions
        register_all_expansions()

        # Get all parameter paths
        parameter_paths = REGISTRY.get_all_parameters()

        # Register models for all parameters
        manager.register_all_parameters(parameter_paths)

        print(f"\n✅ Initialized registry with {len(parameter_paths)} parameters")

    except Exception as e:
        print(f"⚠️  Could not initialize from parameter registry: {e}")
        print("   You can manually register models using register_model()")


# ============================================================================
# Main / Testing
# ============================================================================

    # ========================================================================
    # Ensemble Management
    # ========================================================================

    def create_ensemble(
        self,
        ensemble_name: str,
        model_versions: List[Tuple[str, int]],
        ensemble_strategy: str = "averaging",
        weights: Optional[List[float]] = None,
        notes: str = ""
    ) -> str:
        """
        Create an ensemble of multiple model versions

        Args:
            ensemble_name: Name for the ensemble
            model_versions: List of (model_id, version) tuples
            ensemble_strategy: Strategy (averaging, weighted, voting, stacking)
            weights: Optional weights for weighted averaging
            notes: Notes about the ensemble

        Returns:
            ensemble_id: ID of created ensemble
        """
        # Validate all models exist
        for model_id, version in model_versions:
            if not self._get_version_metadata(model_id, version):
                raise ValueError(f"Model {model_id} v{version} not found")

        # Create ensemble model ID
        ensemble_id = f"ensemble_{ensemble_name}"

        # Register ensemble
        metadata = ModelMetadata(
            model_id=ensemble_id,
            parameter_path=f"ensemble.{ensemble_name}",
            version=1,
            model_type=ModelType.ENSEMBLE,
            status=ModelStatus.ACTIVE,
            notes=f"{notes}\nStrategy: {ensemble_strategy}\nModels: {model_versions}",
            tags=["ensemble", ensemble_strategy],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

        # Store ensemble configuration
        metadata.hyperparameters = {
            "strategy": ensemble_strategy,
            "model_versions": model_versions,
            "weights": weights
        }

        self.registry.models[ensemble_id] = [metadata]
        self._save_metadata(metadata)

        print(f"✅ Created ensemble {ensemble_id} with {len(model_versions)} models")
        return ensemble_id

    def predict_ensemble(
        self,
        ensemble_id: str,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions using an ensemble

        Args:
            ensemble_id: Ensemble ID
            features: Input features

        Returns:
            Predictions
        """
        metadata = self._get_version_metadata(ensemble_id, 1)
        if not metadata or metadata.model_type != ModelType.ENSEMBLE:
            raise ValueError(f"Ensemble {ensemble_id} not found")

        strategy = metadata.hyperparameters['strategy']
        model_versions = metadata.hyperparameters['model_versions']
        weights = metadata.hyperparameters.get('weights')

        # Load all models and make predictions
        predictions = []
        for model_id, version in model_versions:
            model_obj, _ = self.load_model(model_id, version)
            pred = model_obj.predict(features)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Apply ensemble strategy
        if strategy == "averaging":
            return np.mean(predictions, axis=0)
        elif strategy == "weighted":
            if weights is None:
                raise ValueError("Weights required for weighted averaging")
            weights = np.array(weights).reshape(-1, 1)
            return np.sum(predictions * weights, axis=0)
        elif strategy == "voting":
            # Majority voting for classification
            from scipy.stats import mode
            return mode(predictions, axis=0)[0].flatten()
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")

    # ========================================================================
    # Performance Degradation Detection
    # ========================================================================

    def detect_performance_degradation(
        self,
        model_id: str,
        current_metrics: Dict[str, float],
        threshold: float = 0.05
    ) -> Tuple[bool, List[str]]:
        """
        Detect if model performance has degraded

        Args:
            model_id: Model ID
            current_metrics: Current performance metrics
            threshold: Degradation threshold (5% by default)

        Returns:
            (is_degraded, list_of_degradations)
        """
        # Get active version
        active_metadata = self.get_active_version(model_id)
        if not active_metadata:
            return False, []

        degradations = []
        baseline_metrics = active_metadata.performance_metrics

        for metric, current_value in current_metrics.items():
            if metric not in baseline_metrics:
                continue

            baseline_value = baseline_metrics[metric]

            # For error metrics (lower is better)
            if metric in ['mae', 'mse', 'rmse']:
                if current_value > baseline_value * (1 + threshold):
                    pct_increase = ((current_value - baseline_value) / baseline_value) * 100
                    degradations.append(
                        f"{metric.upper()}: {baseline_value:.4f} → {current_value:.4f} "
                        f"(+{pct_increase:.1f}%)"
                    )
            # For other metrics (higher is better)
            else:
                if current_value < baseline_value * (1 - threshold):
                    pct_decrease = ((baseline_value - current_value) / baseline_value) * 100
                    degradations.append(
                        f"{metric.upper()}: {baseline_value:.4f} → {current_value:.4f} "
                        f"(-{pct_decrease:.1f}%)"
                    )

        return len(degradations) > 0, degradations

    def trigger_retraining(
        self,
        model_id: str,
        reason: str = "Performance degradation detected"
    ) -> ModelMetadata:
        """
        Trigger retraining for a model

        Args:
            model_id: Model ID
            reason: Reason for retraining

        Returns:
            New version metadata
        """
        # Get current active version
        active_metadata = self.get_active_version(model_id)
        parent_version = active_metadata.version if active_metadata else None

        # Create new version
        new_metadata = self.create_version(
            model_id=model_id,
            parent_version=parent_version,
            notes=f"Retraining triggered: {reason}",
            tags=["retraining", "auto_trigger"]
        )

        print(f"🔄 Triggered retraining for {model_id}")
        print(f"   Reason: {reason}")
        print(f"   New version: {new_metadata.version}")

        return new_metadata

    # ========================================================================
    # Model Audit Trail
    # ========================================================================

    def get_model_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get complete history for a model

        Args:
            model_id: Model ID

        Returns:
            List of history events
        """
        history = []

        versions = sorted(self.registry.models.get(model_id, []), key=lambda v: v.version)

        for v in versions:
            history.append({
                "event": "version_created",
                "timestamp": v.created_at,
                "version": v.version,
                "status": v.status.value,
                "is_active": v.is_active,
                "notes": v.notes
            })

            if v.training_date:
                history.append({
                    "event": "training_completed",
                    "timestamp": v.training_date,
                    "version": v.version,
                    "samples": v.training_samples,
                    "metrics": v.performance_metrics
                })

        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])

        return history

    def get_audit_log(
        self,
        model_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit log for models

        Args:
            model_id: Optional model ID to filter
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)

        Returns:
            List of audit events
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = """
            SELECT model_id, version, status, is_active, created_at
            FROM model_versions
            WHERE 1=1
        """
        params = []

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)

        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)

        audit_log = []
        for row in cursor.fetchall():
            audit_log.append({
                "model_id": row[0],
                "version": row[1],
                "status": row[2],
                "is_active": row[3],
                "timestamp": row[4]
            })

        conn.close()
        return audit_log

    # ========================================================================
    # Model Rollback
    # ========================================================================

    def rollback_to_version(
        self,
        model_id: str,
        target_version: int,
        reason: str = "Manual rollback"
    ):
        """
        Rollback to a previous version

        Args:
            model_id: Model ID
            target_version: Version to rollback to
            reason: Reason for rollback
        """
        # Validate target version exists
        target_metadata = self._get_version_metadata(model_id, target_version)
        if not target_metadata:
            raise ValueError(f"Version {target_version} not found for {model_id}")

        # Deactivate current version
        current_active = self.get_active_version(model_id)
        if current_active:
            current_active.is_active = False
            current_active.status = ModelStatus.ARCHIVED
            current_active.notes += f"\nRolled back: {reason}"
            current_active.updated_at = datetime.now().isoformat()
            self._save_metadata(current_active)
            self._update_version_in_db(current_active)

        # Activate target version
        target_metadata.is_active = True
        target_metadata.status = ModelStatus.ACTIVE
        target_metadata.notes += f"\nRolled back from v{current_active.version}: {reason}"
        target_metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(target_metadata)
        self._update_version_in_db(target_metadata)

        print(f"⏮️  Rolled back {model_id} from v{current_active.version} to v{target_version}")
        print(f"   Reason: {reason}")

    def safe_rollback_with_validation(
        self,
        model_id: str,
        target_version: int,
        validation_func: callable,
        reason: str = "Safe rollback with validation"
    ) -> bool:
        """
        Rollback with validation check

        Args:
            model_id: Model ID
            target_version: Version to rollback to
            validation_func: Function to validate rollback (returns bool)
            reason: Reason for rollback

        Returns:
            True if rollback succeeded
        """
        # Load target version
        model_obj, metadata = self.load_model(model_id, target_version)

        # Validate
        try:
            is_valid = validation_func(model_obj, metadata)
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

        if not is_valid:
            print(f"❌ Validation failed for v{target_version}")
            return False

        # Perform rollback
        self.rollback_to_version(model_id, target_version, reason)
        print(f"✅ Safe rollback completed with validation")
        return True

    # ========================================================================
    # Model Serving Utilities
    # ========================================================================

    def get_model_for_parameter(
        self,
        parameter_path: str,
        version: Optional[int] = None
    ) -> Tuple[Any, ModelMetadata]:
        """
        Get model for a specific parameter

        Args:
            parameter_path: Parameter path (e.g., "harmony.voicing.type")
            version: Optional version (None = active version)

        Returns:
            (model_obj, metadata)
        """
        model_id = self.registry.parameter_to_model.get(parameter_path)
        if not model_id:
            raise ValueError(f"No model registered for parameter: {parameter_path}")

        return self.load_model(model_id, version)

    def batch_predict(
        self,
        parameter_paths: List[str],
        features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for multiple parameters

        Args:
            parameter_paths: List of parameter paths
            features: Input features (same for all models)

        Returns:
            Dict mapping parameter_path to predictions
        """
        predictions = {}

        for param_path in parameter_paths:
            try:
                model_obj, metadata = self.get_model_for_parameter(param_path)
                pred = model_obj.predict(features)
                predictions[param_path] = pred
            except Exception as e:
                print(f"⚠️  Failed to predict for {param_path}: {e}")
                predictions[param_path] = None

        return predictions

    # ========================================================================
    # Advanced Search and Query
    # ========================================================================

    def search_models(
        self,
        status: Optional[ModelStatus] = None,
        min_r2: Optional[float] = None,
        has_tag: Optional[str] = None,
        parameter_prefix: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        Search for models matching criteria

        Args:
            status: Filter by status
            min_r2: Minimum R2 score
            has_tag: Filter by tag
            parameter_prefix: Filter by parameter path prefix

        Returns:
            List of matching metadata
        """
        results = []

        for model_id, versions in self.registry.models.items():
            for v in versions:
                # Apply filters
                if status and v.status != status:
                    continue

                if min_r2 and v.performance_metrics.get('r2', 0) < min_r2:
                    continue

                if has_tag and has_tag not in v.tags:
                    continue

                if parameter_prefix and not v.parameter_path.startswith(parameter_prefix):
                    continue

                results.append(v)

        return results

    def get_best_models_by_metric(
        self,
        metric: str,
        top_n: int = 10,
        minimize: bool = False
    ) -> List[Tuple[str, int, float, str]]:
        """
        Get top N models by a performance metric

        Args:
            metric: Metric name (e.g., 'r2', 'mae')
            top_n: Number of top models to return
            minimize: True if lower is better (for error metrics)

        Returns:
            List of (model_id, version, metric_value, parameter_path)
        """
        models_with_metric = []

        for model_id, versions in self.registry.models.items():
            for v in versions:
                if v.is_active and metric in v.performance_metrics:
                    models_with_metric.append(
                        (model_id, v.version, v.performance_metrics[metric], v.parameter_path)
                    )

        # Sort
        models_with_metric.sort(key=lambda x: x[2], reverse=not minimize)

        return models_with_metric[:top_n]

    # ========================================================================
    # Health Checks
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on registry

        Returns:
            Health check report
        """
        report = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "warnings": []
        }

        # Check for models without active versions
        models_without_active = []
        for model_id in self.registry.models:
            if not self.get_active_version(model_id):
                models_without_active.append(model_id)

        if models_without_active:
            report["warnings"].append(
                f"{len(models_without_active)} models without active version"
            )

        # Check for missing model files
        missing_files = []
        for model_id, versions in self.registry.models.items():
            for v in versions:
                if v.model_file_path:
                    if not Path(v.model_file_path).exists():
                        missing_files.append(f"{model_id} v{v.version}")

        if missing_files:
            report["issues"].append(f"{len(missing_files)} missing model files")
            report["status"] = "degraded"

        # Check database integrity
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM models")
            db_model_count = cursor.fetchone()[0]
            conn.close()

            if db_model_count != self.registry.total_models:
                report["issues"].append("Database model count mismatch")
                report["status"] = "degraded"
        except Exception as e:
            report["issues"].append(f"Database error: {e}")
            report["status"] = "critical"

        return report


# ============================================================================
# Main / Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MODEL REGISTRY MANAGER - Agent 33")
    print("=" * 80)
    print()

    # Initialize manager
    manager = ModelRegistryManager()

    # Initialize from parameter registry
    print("🔧 Initializing from parameter registry...")
    initialize_registry_from_parameters(manager)

    # Print statistics
    print("\n" + "=" * 80)
    print("REGISTRY STATISTICS")
    print("=" * 80)
    stats = manager.get_statistics()
    print(f"Total Models:        {stats['total_models']}")
    print(f"Total Versions:      {stats['total_versions']}")
    print(f"Active Models:       {stats['active_models']}")
    print(f"Disk Usage:          {stats['total_disk_usage_mb']:.2f} MB")

    print("\nBy Status:")
    for status, count in sorted(stats['by_status'].items()):
        print(f"  {status:20s}: {count:4d}")

    # Perform health check
    print("\n" + "=" * 80)
    print("HEALTH CHECK")
    print("=" * 80)
    health = manager.health_check()
    print(f"Status: {health['status'].upper()}")
    if health['warnings']:
        print("\nWarnings:")
        for warning in health['warnings']:
            print(f"  ⚠️  {warning}")
    if health['issues']:
        print("\nIssues:")
        for issue in health['issues']:
            print(f"  ❌ {issue}")

    # Generate report
    print("\n" + "=" * 80)
    report_path = manager.registry_path / "registry_report.txt"
    manager.generate_report(str(report_path))

    # Export registry
    export_path = manager.registry_path / "registry_export.json"
    manager.export_registry(str(export_path))

    print("\n" + "=" * 80)
    print("✅ Model Registry Manager initialized successfully!")
    print(f"   Registry path: {manager.registry_path}")
    print(f"   Database: {manager.db_path}")
    print(f"   Ready to track {stats['total_models']} models")
    print("=" * 80)
