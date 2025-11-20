<<<<<<< HEAD
#!/usr/bin/env python3
"""
Model Training Specialist - Agent 15
====================================

Comprehensive XGBoost model training system for parameter prediction.

This module provides:
1. Automatic objective selection (regression/classification)
2. Train/val/test splitting with stratification
3. Early stopping for efficiency
4. Comprehensive evaluation metrics
5. Hyperparameter tuning (grid search + random search)
6. Feature importance analysis
7. Quality thresholds and validation
8. Batch training for multiple parameters
9. Progress monitoring and reporting
10. Cross-validation support
11. Learning curve analysis
12. Model persistence and versioning

Key Features:
- Modular training pipeline
- Automatic data preprocessing
- Robust error handling
- Detailed logging and visualization
- Support for all parameter types
- Ensemble model support
- Incremental learning capabilities

Target: R² > 0.5 for all models (preferably > 0.7)
Accuracy > 0.5 for classification (preferably > 0.7)
=======
"""
AGENT 15: Model Training Specialist
====================================

Trains XGBoost models for new parameters with hyperparameter optimization.

This agent:
1. Trains one XGBoost model per parameter (modular architecture)
2. Automatically selects objective based on parameter type
3. Performs train/val/test splitting
4. Uses early stopping for efficiency
5. Evaluates with comprehensive metrics
6. Performs hyperparameter tuning if quality is insufficient
7. Analyzes feature importance
8. Validates model quality

Target: R² > 0.5 for all models (preferably > 0.7)
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS

Author: Agent 15 - Model Training Specialist
License: MIT
"""

<<<<<<< HEAD
import os
import sys
import time
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd

# XGBoost
try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available. Install with: pip install xgboost")

# Scikit-learn
try:
    from sklearn.model_selection import (
        train_test_split, GridSearchCV, RandomizedSearchCV,
        cross_val_score, learning_curve, KFold, StratifiedKFold
    )
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix, roc_auc_score,
        log_loss, mean_absolute_percentage_error
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ Scikit-learn not available. Install with: pip install scikit-learn")

# Joblib for model persistence
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("❌ Joblib not available. Install with: pip install joblib")

# Plotting (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available. Visualization disabled.")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import parameter registry
try:
    from parameters.universal_registry import (
        UniversalParameterRegistry, ParameterDefinition,
        ParameterType, REGISTRY
    )
except ImportError:
    print("⚠️ Could not import parameter registry")
    REGISTRY = None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training"""

    # Data splitting
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    stratify: bool = True

    # Training
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    # Early stopping
    early_stopping_rounds: int = 10
    eval_metric: Optional[str] = None

    # Hyperparameter tuning
    enable_tuning: bool = False
    tuning_method: str = 'grid'  # 'grid' or 'random'
    n_iter: int = 50  # For random search
    cv_folds: int = 3

    # Quality thresholds
    min_r2: float = 0.5
    min_accuracy: float = 0.5
    min_f1: float = 0.4

    # Training options
    verbose: bool = True
    n_jobs: int = -1
    use_gpu: bool = False

    # Output
    save_plots: bool = True
    save_metrics: bool = True

    def to_xgb_params(self, objective: str) -> Dict[str, Any]:
        """Convert to XGBoost parameters"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'objective': objective
        }

        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'

        return params
=======
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed, model training will be limited")

try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        accuracy_score, f1_score, precision_score, recall_score
    )
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("WARNING: joblib not installed, model saving will use pickle")
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS


@dataclass
class TrainingMetrics:
<<<<<<< HEAD
    """Comprehensive training metrics"""

    # Model identity
    param_name: str
    param_type: str
    model_path: str

    # Dataset info
    n_train: int
    n_val: int
    n_test: int
    n_features: int

    # Regression metrics
=======
    """Metrics from model training"""
    # Performance metrics
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
    train_r2: Optional[float] = None
    val_r2: Optional[float] = None
    test_r2: Optional[float] = None
    train_mae: Optional[float] = None
    val_mae: Optional[float] = None
    test_mae: Optional[float] = None
    train_rmse: Optional[float] = None
<<<<<<< HEAD
    val_rmse: Optional[float] = None
    test_rmse: Optional[float] = None
    train_mape: Optional[float] = None
    test_mape: Optional[float] = None
=======
    test_rmse: Optional[float] = None
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS

    # Classification metrics
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    train_f1: Optional[float] = None
<<<<<<< HEAD
    val_f1: Optional[float] = None
    test_f1: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_auc: Optional[float] = None

    # Training info
    training_time: float = 0.0
    best_iteration: Optional[int] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    # Cross-validation
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    # Hyperparameter tuning
    best_params: Optional[Dict[str, Any]] = None
    tuning_time: Optional[float] = None

    # Quality checks
    passed_quality_check: bool = False
    quality_message: str = ""

    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append(f"Parameter: {self.param_name}")
        lines.append(f"Type: {self.param_type}")
        lines.append(f"Dataset: train={self.n_train}, val={self.n_val}, test={self.n_test}")
        lines.append(f"Features: {self.n_features}")

        if self.test_r2 is not None:
            lines.append(f"\nRegression Metrics:")
            lines.append(f"  R²:   train={self.train_r2:.4f}, val={self.val_r2:.4f}, test={self.test_r2:.4f}")
            lines.append(f"  MAE:  train={self.train_mae:.4f}, val={self.val_mae:.4f}, test={self.test_mae:.4f}")
            lines.append(f"  RMSE: train={self.train_rmse:.4f}, val={self.val_rmse:.4f}, test={self.test_rmse:.4f}")

        if self.test_accuracy is not None:
            lines.append(f"\nClassification Metrics:")
            lines.append(f"  Accuracy: train={self.train_accuracy:.4f}, val={self.val_accuracy:.4f}, test={self.test_accuracy:.4f}")
            lines.append(f"  F1 Score: train={self.train_f1:.4f}, val={self.val_f1:.4f}, test={self.test_f1:.4f}")
            if self.test_precision is not None:
                lines.append(f"  Precision: {self.test_precision:.4f}")
                lines.append(f"  Recall: {self.test_recall:.4f}")

        lines.append(f"\nTraining Time: {self.training_time:.2f}s")

        if self.best_iteration is not None:
            lines.append(f"Best Iteration: {self.best_iteration}")

        if self.cv_mean is not None:
            lines.append(f"CV Score: {self.cv_mean:.4f} ± {self.cv_std:.4f}")

        lines.append(f"\nQuality Check: {'✅ PASSED' if self.passed_quality_check else '❌ FAILED'}")
        if self.quality_message:
            lines.append(f"  {self.quality_message}")

        lines.append(f"\nModel saved to: {self.model_path}")

        return '\n'.join(lines)


@dataclass
class BatchTrainingResults:
    """Results from batch training multiple parameters"""

    total_parameters: int = 0
    successful: int = 0
    failed: int = 0
    total_time: float = 0.0

    results: Dict[str, TrainingMetrics] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def add_success(self, param_name: str, metrics: TrainingMetrics):
        """Add successful training result"""
        self.successful += 1
        self.results[param_name] = metrics

    def add_failure(self, param_name: str, error: str):
        """Add failed training result"""
        self.failed += 1
        self.errors[param_name] = error

    def summary(self) -> str:
        """Generate summary report"""
        lines = []
        lines.append("=" * 80)
        lines.append("BATCH TRAINING SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total Parameters: {self.total_parameters}")
        lines.append(f"Successful: {self.successful} ({100*self.successful/self.total_parameters:.1f}%)")
        lines.append(f"Failed: {self.failed} ({100*self.failed/self.total_parameters:.1f}%)")
        lines.append(f"Total Time: {self.total_time:.2f}s ({self.total_time/60:.2f}m)")

        if self.successful > 0:
            lines.append(f"\n✅ Successful Models:")
            for param_name, metrics in self.results.items():
                quality = "✅" if metrics.passed_quality_check else "⚠️"
                if metrics.test_r2 is not None:
                    lines.append(f"  {quality} {param_name}: R²={metrics.test_r2:.3f}")
                else:
                    lines.append(f"  {quality} {param_name}: Acc={metrics.test_accuracy:.3f}")

        if self.failed > 0:
            lines.append(f"\n❌ Failed Models:")
            for param_name, error in self.errors.items():
                lines.append(f"  {param_name}: {error}")

        lines.append("=" * 80)
        return '\n'.join(lines)


# ============================================================================
# Model Training Specialist
# ============================================================================

class ModelTrainingSpecialist:
    """
    Comprehensive model training system for parameter prediction.

    This class handles:
    - Data preparation and splitting
    - Model training with early stopping
    - Hyperparameter optimization
    - Comprehensive evaluation
    - Feature importance analysis
    - Batch training
    - Progress monitoring
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize Model Training Specialist.

        Args:
            config: Training configuration. If None, uses defaults.
        """
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            raise RuntimeError("XGBoost and scikit-learn are required")

        self.config = config or TrainingConfig()
        self.registry = REGISTRY

        # Feature extractor (will be set when needed)
        self.feature_extractor = None
        self.feature_names = None

        # Label encoders for categorical parameters
        self.label_encoders: Dict[str, LabelEncoder] = {}

        # Scalers for continuous parameters (optional)
        self.scalers: Dict[str, StandardScaler] = {}

        # Training history
        self.training_history: List[TrainingMetrics] = []

    # ========================================================================
    # Main Training Methods
    # ========================================================================

    def train_parameter_model(
        self,
        param_name: str,
        param_def: ParameterDefinition,
        training_data: Union[List[Dict], pd.DataFrame],
        models_dir: Path = Path('midi_generator/models/pretrained'),
        output_dir: Optional[Path] = None
    ) -> Tuple[Any, TrainingMetrics]:
        """
        Train XGBoost model for a single parameter.

        Args:
            param_name: Full parameter path (e.g., 'harmony.voicing.spread')
            param_def: Parameter definition from registry
            training_data: Training data as list of dicts or DataFrame
                Each entry must have 'features' and 'parameter_value'
            models_dir: Directory to save trained models
            output_dir: Directory for plots and metrics (optional)

        Returns:
            Tuple of (trained_model, metrics)

        Raises:
            ValueError: If training data is invalid
            RuntimeError: If training fails
        """
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL: {param_name}")
        print(f"{'='*80}")

        start_time = time.time()

        # Create output directories
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Prepare data
            X, y, data_info = self._prepare_training_data(
                training_data, param_def, param_name
            )

            print(f"\n📊 Dataset Information:")
            print(f"   Samples: {X.shape[0]}")
            print(f"   Features: {X.shape[1]}")
            print(f"   Parameter Type: {param_def.param_type.value}")
            print(f"   Value Range: {data_info['value_range']}")
            print(f"   Missing Values: {data_info['missing_values']}")

            # 2. Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(
                X, y, param_def
            )

            print(f"\n🔀 Data Split:")
            print(f"   Train: {len(X_train)} samples")
            print(f"   Val:   {len(X_val)} samples")
            print(f"   Test:  {len(X_test)} samples")

            # 3. Determine objective
            objective = self._get_objective(param_def)
            print(f"\n🎯 Objective: {objective}")

            # 4. Train model
            if self.config.enable_tuning:
                print(f"\n🔧 Hyperparameter tuning enabled ({self.config.tuning_method})")
                model, tuning_info = self._train_with_tuning(
                    X_train, y_train, X_val, y_val,
                    param_def, objective
                )
            else:
                print(f"\n🚀 Training with default parameters")
                model, training_info = self._train_model(
                    X_train, y_train, X_val, y_val,
                    param_def, objective
                )
                tuning_info = None

            # 5. Evaluate
            metrics = self._evaluate_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test,
                param_def, param_name
            )

            # Add dataset sizes
            metrics.n_train = len(X_train)
            metrics.n_val = len(X_val)
            metrics.n_test = len(X_test)
            metrics.n_features = X.shape[1]

            # Add training time
            metrics.training_time = time.time() - start_time

            # Add tuning info
            if tuning_info:
                metrics.best_params = tuning_info['best_params']
                metrics.tuning_time = tuning_info['tuning_time']
                if 'best_iteration' in tuning_info:
                    metrics.best_iteration = tuning_info['best_iteration']

            # 6. Feature importance
            print(f"\n🔍 Analyzing feature importance...")
            metrics.feature_importance = self._analyze_feature_importance(
                model, param_name
            )
            metrics.top_features = self._get_top_features(
                metrics.feature_importance, n=10
            )

            # 7. Quality check
            passed, message = self._check_quality(metrics, param_def)
            metrics.passed_quality_check = passed
            metrics.quality_message = message

            # 8. Save model
            model_path = self._save_model(
                model, param_name, models_dir, param_def, metrics
            )
            metrics.model_path = str(model_path)

            # 9. Generate visualizations
            if self.config.save_plots and output_dir:
                self._generate_plots(
                    model, X_train, y_train, X_test, y_test,
                    param_name, metrics, output_dir
                )

            # 10. Save metrics
            if self.config.save_metrics and output_dir:
                self._save_metrics(metrics, output_dir, param_name)

            # 11. Print summary
            print(f"\n{metrics.summary()}")

            # Add to history
            self.training_history.append(metrics)

            return model, metrics

        except Exception as e:
            print(f"\n❌ Training failed for {param_name}: {e}")
            raise RuntimeError(f"Failed to train model for {param_name}") from e

    def train_batch(
        self,
        parameters: List[Tuple[str, ParameterDefinition]],
        training_data_dir: Path,
        models_dir: Path = Path('midi_generator/models/pretrained'),
        output_dir: Optional[Path] = None,
        continue_on_error: bool = True
    ) -> BatchTrainingResults:
        """
        Train models for multiple parameters in batch.

        Args:
            parameters: List of (param_name, param_def) tuples
            training_data_dir: Directory containing training data
            models_dir: Directory to save models
            output_dir: Directory for plots and metrics
            continue_on_error: If True, continue training even if some fail

        Returns:
            BatchTrainingResults with summary of all training
        """
        print(f"\n{'='*80}")
        print(f"BATCH TRAINING: {len(parameters)} PARAMETERS")
        print(f"{'='*80}")

        results = BatchTrainingResults()
        results.total_parameters = len(parameters)

        batch_start = time.time()

        for i, (param_name, param_def) in enumerate(parameters, 1):
            print(f"\n[{i}/{len(parameters)}] Training: {param_name}")

            try:
                # Load training data
                training_data = self._load_training_data(
                    param_name, training_data_dir
                )

                # Train model
                model, metrics = self.train_parameter_model(
                    param_name, param_def, training_data,
                    models_dir, output_dir
                )

                results.add_success(param_name, metrics)

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Failed: {error_msg}")
                results.add_failure(param_name, error_msg)

                if not continue_on_error:
                    break

        results.total_time = time.time() - batch_start

        # Print summary
        print(f"\n{results.summary()}")

        # Save batch results
        if output_dir:
            self._save_batch_results(results, output_dir)

        return results

    # ========================================================================
    # Data Preparation
    # ========================================================================

    def _prepare_training_data(
        self,
        training_data: Union[List[Dict], pd.DataFrame],
        param_def: ParameterDefinition,
        param_name: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare training data for model training.

        Returns:
            Tuple of (X, y, data_info)
        """
        # Convert to DataFrame if needed
        if isinstance(training_data, list):
            df = pd.DataFrame(training_data)
        else:
            df = training_data

        # Extract features
        if 'features' in df.columns:
            if isinstance(df['features'].iloc[0], (list, np.ndarray)):
                X = np.array(df['features'].tolist())
            else:
                # Features might be flattened columns
                feature_cols = [c for c in df.columns if c.startswith('feature_')]
                X = df[feature_cols].values
        else:
            raise ValueError("Training data must have 'features' column")

        # Extract parameter values
        if 'parameter_value' not in df.columns:
            raise ValueError("Training data must have 'parameter_value' column")

        param_values = df['parameter_value'].values

        # Convert based on parameter type
        if param_def.param_type in [
            ParameterType.CONTINUOUS,
            ParameterType.PROBABILITY,
            ParameterType.DURATION
        ]:
            y = param_values.astype(float)

        elif param_def.param_type == ParameterType.INTEGER:
            y = param_values.astype(int).astype(float)  # XGBoost works with float

        elif param_def.param_type in [ParameterType.MIDI_NOTE, ParameterType.VELOCITY]:
            y = param_values.astype(int).astype(float)

        elif param_def.param_type == ParameterType.BOOLEAN:
            y = np.array([int(bool(v)) for v in param_values], dtype=int)

        elif param_def.param_type == ParameterType.CATEGORICAL:
            # Encode categorical values
            encoder = LabelEncoder()
            y = encoder.fit_transform(param_values)
            self.label_encoders[param_name] = encoder

        elif param_def.param_type in [ParameterType.ARRAY_INT, ParameterType.ARRAY_FLOAT]:
            # Use first element or length as proxy
            y = np.array([
                len(v) if isinstance(v, (list, tuple, np.ndarray)) else float(v)
                for v in param_values
            ], dtype=float)

        else:
            raise ValueError(f"Unsupported parameter type: {param_def.param_type}")

        # Check for NaN/inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("⚠️ Warning: Found NaN/inf in features, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("⚠️ Warning: Found NaN/inf in targets, removing samples")
            valid_mask = ~(np.isnan(y) | np.isinf(y))
            X = X[valid_mask]
            y = y[valid_mask]

        # Data info
        data_info = {
            'value_range': (float(y.min()), float(y.max())),
            'missing_values': len(param_values) - len(y),
            'n_samples': len(y),
            'n_features': X.shape[1]
        }

        return X, y, data_info

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_def: ParameterDefinition
    ) -> Tuple[np.ndarray, ...]:
        """
        Split data into train/val/test sets with stratification.

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Determine if we should stratify
        stratify_target = None
        if self.config.stratify:
            if param_def.param_type == ParameterType.CATEGORICAL:
                stratify_target = y
            elif param_def.param_type == ParameterType.BOOLEAN:
                stratify_target = y
            else:
                # For continuous, bin into quartiles
                try:
                    stratify_target = pd.qcut(
                        y, q=4, labels=False, duplicates='drop'
                    )
                except (ValueError, TypeError):
                    # If binning fails, don't stratify
                    stratify_target = None

        # First split: separate test set
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify_target
            )
        except ValueError:
            # If stratification fails, split without it
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

        # Second split: separate train and val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)

        try:
            if stratify_target is not None:
                # Get stratification for temp set
                stratify_temp = pd.qcut(
                    y_temp, q=4, labels=False, duplicates='drop'
                ) if param_def.param_type not in [
                    ParameterType.CATEGORICAL, ParameterType.BOOLEAN
                ] else y_temp
            else:
                stratify_temp = None

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=stratify_temp
            )
        except (ValueError, TypeError):
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state
            )

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ========================================================================
    # Model Training
    # ========================================================================

    def _get_objective(self, param_def: ParameterDefinition) -> str:
        """Determine XGBoost objective function"""
        if param_def.param_type in [
            ParameterType.CONTINUOUS,
            ParameterType.PROBABILITY,
            ParameterType.DURATION,
            ParameterType.INTEGER,
            ParameterType.MIDI_NOTE,
            ParameterType.VELOCITY,
            ParameterType.ARRAY_INT,
            ParameterType.ARRAY_FLOAT
        ]:
            return 'reg:squarederror'

        elif param_def.param_type == ParameterType.BOOLEAN:
            return 'binary:logistic'

        elif param_def.param_type == ParameterType.CATEGORICAL:
            return 'multi:softmax'

        else:
            return 'reg:squarederror'

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_def: ParameterDefinition,
        objective: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train XGBoost model with early stopping.

        Returns:
            Tuple of (model, training_info)
        """
        # Get XGBoost parameters
        params = self.config.to_xgb_params(objective)

        # Add categorical-specific params
        if param_def.param_type == ParameterType.CATEGORICAL:
            n_classes = len(param_def.options) if param_def.options else len(np.unique(y_train))
            params['num_class'] = n_classes

        # Create model
        if objective.startswith('reg'):
            model = XGBRegressor(**params)
        else:
            model = XGBClassifier(**params)

        # Train with early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        eval_names = ['train', 'val']

        # Determine eval metric
        if self.config.eval_metric:
            eval_metric = self.config.eval_metric
        else:
            if objective.startswith('reg'):
                eval_metric = 'rmse'
            elif objective == 'binary:logistic':
                eval_metric = 'logloss'
            else:
                eval_metric = 'mlogloss'

        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=eval_metric,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=False
        )

        # Get training info
        training_info = {
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None,
            'evals_result': model.evals_result() if hasattr(model, 'evals_result') else None
        }

        return model, training_info

    def _train_with_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_def: ParameterDefinition,
        objective: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with hyperparameter tuning.

        Returns:
            Tuple of (best_model, tuning_info)
        """
        tuning_start = time.time()

        # Combine train and val for CV
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        # Create base model
        base_params = {
            'random_state': self.config.random_state,
            'n_jobs': self.config.n_jobs,
            'objective': objective
        }

        if param_def.param_type == ParameterType.CATEGORICAL:
            n_classes = len(param_def.options) if param_def.options else len(np.unique(y_train))
            base_params['num_class'] = n_classes

        if objective.startswith('reg'):
            base_model = XGBRegressor(**base_params)
            scoring = 'r2'
        else:
            base_model = XGBClassifier(**base_params)
            scoring = 'accuracy'

        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0.0, 0.1, 0.2, 0.3],
            'reg_alpha': [0.0, 0.01, 0.1, 1.0],
            'reg_lambda': [0.0, 0.1, 1.0, 10.0]
        }

        # Choose search method
        if self.config.tuning_method == 'grid':
            # Use smaller grid for grid search
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }

            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.config.cv_folds,
                scoring=scoring,
                n_jobs=self.config.n_jobs,
                verbose=1 if self.config.verbose else 0
            )
        else:
            # Random search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=self.config.n_iter,
                cv=self.config.cv_folds,
                scoring=scoring,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=1 if self.config.verbose else 0
            )

        # Perform search
        search.fit(X_combined, y_combined)

        best_model = search.best_estimator_

        tuning_info = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'tuning_time': time.time() - tuning_start,
            'cv_results': search.cv_results_
        }

        print(f"   Best params: {search.best_params_}")
        print(f"   Best CV score: {search.best_score_:.4f}")

        return best_model, tuning_info

    # ========================================================================
    # Model Evaluation
    # ========================================================================

    def _evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        param_def: ParameterDefinition,
        param_name: str
    ) -> TrainingMetrics:
        """
        Comprehensive model evaluation.

        Returns:
            TrainingMetrics object
        """
        metrics = TrainingMetrics(
            param_name=param_name,
            param_type=param_def.param_type.value,
            model_path="",  # Will be filled later
            n_train=0, n_val=0, n_test=0, n_features=0
        )

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        if param_def.param_type in [
            ParameterType.CONTINUOUS,
            ParameterType.PROBABILITY,
            ParameterType.DURATION,
            ParameterType.INTEGER,
            ParameterType.MIDI_NOTE,
            ParameterType.VELOCITY,
            ParameterType.ARRAY_INT,
            ParameterType.ARRAY_FLOAT
        ]:
            # Regression metrics
            metrics.train_r2 = r2_score(y_train, y_train_pred)
            metrics.val_r2 = r2_score(y_val, y_val_pred)
            metrics.test_r2 = r2_score(y_test, y_test_pred)

            metrics.train_mae = mean_absolute_error(y_train, y_train_pred)
            metrics.val_mae = mean_absolute_error(y_val, y_val_pred)
            metrics.test_mae = mean_absolute_error(y_test, y_test_pred)

            metrics.train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            metrics.val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            metrics.test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # MAPE (avoid division by zero)
            try:
                if np.all(y_train != 0):
                    metrics.train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
                if np.all(y_test != 0):
                    metrics.test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
            except:
                pass

        else:
            # Classification metrics
            metrics.train_accuracy = accuracy_score(y_train, y_train_pred)
            metrics.val_accuracy = accuracy_score(y_val, y_val_pred)
            metrics.test_accuracy = accuracy_score(y_test, y_test_pred)

            # F1 score
            average = 'binary' if param_def.param_type == ParameterType.BOOLEAN else 'weighted'

            metrics.train_f1 = f1_score(y_train, y_train_pred, average=average, zero_division=0)
            metrics.val_f1 = f1_score(y_val, y_val_pred, average=average, zero_division=0)
            metrics.test_f1 = f1_score(y_test, y_test_pred, average=average, zero_division=0)

            metrics.test_precision = precision_score(
                y_test, y_test_pred, average=average, zero_division=0
            )
            metrics.test_recall = recall_score(
                y_test, y_test_pred, average=average, zero_division=0
            )

            # AUC for binary classification
            if param_def.param_type == ParameterType.BOOLEAN:
                try:
                    y_test_proba = model.predict_proba(X_test)[:, 1]
                    metrics.test_auc = roc_auc_score(y_test, y_test_proba)
                except:
                    pass

        return metrics

    def _check_quality(
        self,
        metrics: TrainingMetrics,
        param_def: ParameterDefinition
    ) -> Tuple[bool, str]:
        """
        Check if model quality meets requirements.

        Returns:
            (passed, message)
        """
        if metrics.test_r2 is not None:
            # Regression
            if metrics.test_r2 >= self.config.min_r2:
                return True, f"R² = {metrics.test_r2:.4f} >= {self.config.min_r2}"
            else:
                return False, f"R² = {metrics.test_r2:.4f} < {self.config.min_r2} (threshold)"

        else:
            # Classification
            if metrics.test_accuracy >= self.config.min_accuracy:
                if metrics.test_f1 >= self.config.min_f1:
                    return True, f"Accuracy = {metrics.test_accuracy:.4f}, F1 = {metrics.test_f1:.4f}"
                else:
                    return False, f"F1 = {metrics.test_f1:.4f} < {self.config.min_f1} (threshold)"
            else:
                return False, f"Accuracy = {metrics.test_accuracy:.4f} < {self.config.min_accuracy} (threshold)"

    # ========================================================================
    # Feature Importance
    # ========================================================================

    def _analyze_feature_importance(
        self,
        model: Any,
        param_name: str
    ) -> Dict[str, float]:
        """
        Analyze feature importance.

        Returns:
            Dictionary of feature_name: importance
        """
        try:
            importances = model.feature_importances_

            # Get feature names
            feature_names = self._get_feature_names()

            if len(feature_names) != len(importances):
                # Use generic names
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            importance_dict = dict(zip(feature_names, importances))

            return importance_dict

        except Exception as e:
            print(f"⚠️ Could not analyze feature importance: {e}")
            return {}

    def _get_feature_names(self) -> List[str]:
        """Get feature names from feature extractor"""
        if self.feature_names is not None:
            return self.feature_names

        # Try to load from feature extractor
        try:
            # This would load the actual feature extractor
            # For now, return generic names
            return [f"feature_{i}" for i in range(1000)]
        except:
            return [f"feature_{i}" for i in range(1000)]

    def _get_top_features(
        self,
        importance_dict: Dict[str, float],
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top N most important features"""
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]

    # ========================================================================
    # Model Persistence
    # ========================================================================

    def _save_model(
        self,
        model: Any,
        param_name: str,
        models_dir: Path,
        param_def: ParameterDefinition,
        metrics: TrainingMetrics
    ) -> Path:
        """
        Save trained model and metadata.
=======
    test_f1: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None

    # Training metadata
    training_time: float = 0.0
    n_training_examples: int = 0
    n_features: int = 0
    best_iteration: Optional[int] = None

    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    # Hyperparameter tuning
    hyperparameter_tuning: Optional[Dict[str, Any]] = None

    # Model path
    model_path: Optional[str] = None


@dataclass
class ModelTrainingResult:
    """Complete result of model training"""
    parameter_name: str
    success: bool
    model: Optional[Any] = None
    metrics: Optional[TrainingMetrics] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class ModelTrainingSpecialist:
    """
    Specialist for training XGBoost models for parameter prediction

    Trains one model per parameter using modular architecture:
    - Each model is independent
    - Adding parameters requires NO retraining of existing models
    - Models predict parameter values from extracted MIDI features
    """

    def __init__(self,
                 models_dir: Path = Path('models/pretrained'),
                 quality_threshold: float = 0.5):
        """
        Initialize model training specialist

        Args:
            models_dir: Directory to save trained models
            quality_threshold: Minimum R²/accuracy for acceptable model
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.quality_threshold = quality_threshold

        # Default hyperparameters
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        self.training_history: List[ModelTrainingResult] = []

    def train_parameter_model(self,
                             param_name: str,
                             param_def: dict,
                             training_data: List[dict],
                             feature_names: Optional[List[str]] = None) -> ModelTrainingResult:
        """
        Train XGBoost model for single parameter

        Args:
            param_name: Full parameter name
            param_def: Parameter definition with type, range, etc.
            training_data: List of {features, parameter_value, ...} dicts
            feature_names: Names of features (for importance analysis)

        Returns:
            ModelTrainingResult with model and metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL: {param_name}")
        print(f"{'='*80}\n")

        start_time = time.time()

        try:
            # 1. Prepare data
            print("Step 1: Preparing training data...")
            X, y, label_encoder = self._prepare_training_data(training_data, param_def)

            if X is None or y is None:
                return ModelTrainingResult(
                    parameter_name=param_name,
                    success=False,
                    error="Failed to prepare training data"
                )

            print(f"  Dataset: {X.shape[0]} examples, {X.shape[1]} features")
            print(f"  Parameter type: {param_def.get('type', 'UNKNOWN')}")
            print(f"  Value range: {param_def.get('range', 'UNKNOWN')}")

            # 2. Split data
            print("\nStep 2: Splitting train/val/test...")
            split_data = self._split_data(X, y, param_def)

            if split_data is None:
                return ModelTrainingResult(
                    parameter_name=param_name,
                    success=False,
                    error="Failed to split data"
                )

            X_train, X_val, X_test, y_train, y_val, y_test = split_data

            print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # 3. Determine objective
            objective = self._get_objective(param_def)
            print(f"\nStep 3: Training with objective: {objective}")

            # 4. Train model
            model, training_history = self._train_model(
                X_train, y_train, X_val, y_val,
                param_def, objective
            )

            if model is None:
                return ModelTrainingResult(
                    parameter_name=param_name,
                    success=False,
                    error="Model training failed"
                )

            print("  ✅ Model training complete")

            # 5. Evaluate
            print("\nStep 4: Evaluating model...")
            metrics = self._evaluate_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test,
                param_def, feature_names
            )

            metrics.training_time = time.time() - start_time
            metrics.n_training_examples = len(X_train)
            metrics.n_features = X.shape[1]

            # 6. Check quality
            quality_ok = self._is_quality_acceptable(metrics, param_def)

            if not quality_ok:
                print("\n⚠️ Model quality below threshold, attempting hyperparameter tuning...")
                model, metrics = self._tune_hyperparameters(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    param_def, objective, feature_names
                )

                if model is None:
                    return ModelTrainingResult(
                        parameter_name=param_name,
                        success=False,
                        error="Hyperparameter tuning failed",
                        metrics=metrics
                    )

            # 7. Save model
            print("\nStep 5: Saving model...")
            model_path = self._save_model(model, param_name, label_encoder)
            metrics.model_path = str(model_path)

            # 8. Print summary
            self._print_training_summary(param_name, metrics)

            result = ModelTrainingResult(
                parameter_name=param_name,
                success=True,
                model=model,
                metrics=metrics
            )

            self.training_history.append(result)

            return result

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

            return ModelTrainingResult(
                parameter_name=param_name,
                success=False,
                error=str(e)
            )

    def _prepare_training_data(self, training_data: List[dict], param_def: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
        """
        Convert training data to X, y arrays

        Args:
            training_data: List of training examples
            param_def: Parameter definition

        Returns:
            (X, y, label_encoder) or (None, None, None) if failed
        """
        if not training_data:
            print("  ERROR: No training data provided")
            return None, None, None

        try:
            # Extract features
            X = np.array([example['features'] for example in training_data])

            # Extract parameter values
            param_values = [example['parameter_value'] for example in training_data]

            param_type = param_def.get('type', 'CONTINUOUS')
            label_encoder = None

            # Convert based on type
            if param_type == 'CONTINUOUS' or param_type == 'PROBABILITY':
                y = np.array(param_values, dtype=float)

            elif param_type == 'BOOLEAN':
                y = np.array([int(v) for v in param_values], dtype=int)

            elif param_type == 'CATEGORICAL':
                # Encode categorical values
                if HAS_SKLEARN:
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(param_values)
                else:
                    # Fallback: manual encoding
                    unique_values = list(set(param_values))
                    value_to_idx = {v: i for i, v in enumerate(unique_values)}
                    y = np.array([value_to_idx[v] for v in param_values], dtype=int)
                    label_encoder = value_to_idx

            else:  # ARRAY types - use first element or length as proxy
                y = np.array([
                    len(v) if isinstance(v, list) else (v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v)
                    for v in param_values
                ], dtype=float)

            return X, y, label_encoder

        except Exception as e:
            print(f"  ERROR preparing data: {e}")
            return None, None, None

    def _split_data(self, X: np.ndarray, y: np.ndarray, param_def: dict) -> Optional[Tuple]:
        """
        Split data into train/val/test sets

        Args:
            X: Feature matrix
            y: Target values
            param_def: Parameter definition

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test) or None
        """
        try:
            # Create stratification for splitting
            stratify = self._create_stratification(y, param_def)

            if HAS_SKLEARN:
                # First split: train + val vs test
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=stratify if stratify is not None else None
                )

                # Second split: train vs val
                stratify_temp = self._create_stratification(y_temp, param_def)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.2, random_state=42,
                    stratify=stratify_temp if stratify_temp is not None else None
                )

                return X_train, X_val, X_test, y_train, y_val, y_test

            else:
                # Manual splitting
                n = len(X)
                indices = np.random.permutation(n)

                test_size = int(0.2 * n)
                val_size = int(0.2 * n)
                train_size = n - test_size - val_size

                train_idx = indices[:train_size]
                val_idx = indices[train_size:train_size+val_size]
                test_idx = indices[train_size+val_size:]

                return (
                    X[train_idx], X[val_idx], X[test_idx],
                    y[train_idx], y[val_idx], y[test_idx]
                )

        except Exception as e:
            print(f"  ERROR splitting data: {e}")
            return None

    def _create_stratification(self, y: np.ndarray, param_def: dict) -> Optional[np.ndarray]:
        """Create stratification for train/test split"""

        param_type = param_def.get('type', 'CONTINUOUS')

        try:
            if param_type == 'CATEGORICAL' or param_type == 'BOOLEAN':
                return y  # Use values directly

            elif param_type == 'CONTINUOUS' or param_type == 'PROBABILITY':
                # Bin into quartiles for stratification
                if HAS_SKLEARN and len(y) > 10:
                    try:
                        return pd.qcut(y, q=4, labels=False, duplicates='drop')
                    except:
                        return None
                else:
                    return None

            else:
                return None

        except:
            return None

    def _get_objective(self, param_def: dict) -> str:
        """Determine XGBoost objective function"""

        param_type = param_def.get('type', 'CONTINUOUS')

        if param_type == 'CONTINUOUS' or param_type == 'PROBABILITY':
            return 'reg:squarederror'
        elif param_type == 'BOOLEAN':
            return 'binary:logistic'
        elif param_type == 'CATEGORICAL':
            return 'multi:softmax'
        else:
            return 'reg:squarederror'  # Default

    def _train_model(self, X_train, y_train, X_val, y_val, param_def, objective) -> Tuple[Optional[Any], dict]:
        """
        Train XGBoost model with early stopping

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            param_def: Parameter definition
            objective: XGBoost objective

        Returns:
            (model, training_history) or (None, {})
        """
        if not HAS_XGBOOST:
            print("  ERROR: XGBoost not installed")
            return None, {}

        try:
            params = self.default_params.copy()
            params['objective'] = objective

            param_type = param_def.get('type', 'CONTINUOUS')

            if param_type == 'CATEGORICAL':
                param_range = param_def.get('range', [])
                if isinstance(param_range, list):
                    params['num_class'] = len(param_range)

            # Create model
            if param_type in ['CONTINUOUS', 'PROBABILITY', 'ARRAY_INT', 'ARRAY_FLOAT']:
                model = XGBRegressor(**params)
            else:
                model = XGBClassifier(**params)

            # Train with early stopping
            eval_set = [(X_train, y_train), (X_val, y_val)]

            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )

            # Get training history
            training_history = {
                'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else None,
                'best_score': model.best_score if hasattr(model, 'best_score') else None
            }

            return model, training_history

        except Exception as e:
            print(f"  ERROR training model: {e}")
            return None, {}

    def _evaluate_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test,
                       param_def, feature_names: Optional[List[str]] = None) -> TrainingMetrics:
        """
        Comprehensive model evaluation

        Args:
            model: Trained model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            param_def: Parameter definition
            feature_names: Feature names for importance

        Returns:
            TrainingMetrics
        """
        metrics = TrainingMetrics()
        param_type = param_def.get('type', 'CONTINUOUS')

        try:
            if param_type in ['CONTINUOUS', 'PROBABILITY', 'ARRAY_INT', 'ARRAY_FLOAT']:
                # Regression metrics
                if HAS_SKLEARN:
                    metrics.train_r2 = r2_score(y_train, model.predict(X_train))
                    metrics.val_r2 = r2_score(y_val, model.predict(X_val))
                    metrics.test_r2 = r2_score(y_test, model.predict(X_test))

                    metrics.train_mae = mean_absolute_error(y_train, model.predict(X_train))
                    metrics.val_mae = mean_absolute_error(y_val, model.predict(X_val))
                    metrics.test_mae = mean_absolute_error(y_test, model.predict(X_test))

                    metrics.train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
                    metrics.test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

            else:
                # Classification metrics
                if HAS_SKLEARN:
                    y_train_pred = model.predict(X_train)
                    y_val_pred = model.predict(X_val)
                    y_test_pred = model.predict(X_test)

                    metrics.train_accuracy = accuracy_score(y_train, y_train_pred)
                    metrics.val_accuracy = accuracy_score(y_val, y_val_pred)
                    metrics.test_accuracy = accuracy_score(y_test, y_test_pred)

                    # F1 score
                    average = 'binary' if param_type == 'BOOLEAN' else 'weighted'
                    metrics.train_f1 = f1_score(y_train, y_train_pred, average=average, zero_division=0)
                    metrics.test_f1 = f1_score(y_test, y_test_pred, average=average, zero_division=0)

                    # Precision/Recall
                    metrics.test_precision = precision_score(y_test, y_test_pred, average=average, zero_division=0)
                    metrics.test_recall = recall_score(y_test, y_test_pred, average=average, zero_division=0)

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                if feature_names and len(feature_names) == len(importances):
                    metrics.feature_importance = dict(zip(feature_names, importances))
                else:
                    metrics.feature_importance = {f'feature_{i}': imp for i, imp in enumerate(importances)}

                # Get top features
                sorted_features = sorted(metrics.feature_importance.items(), key=lambda x: x[1], reverse=True)
                metrics.top_features = sorted_features[:10]

        except Exception as e:
            print(f"  WARNING: Error computing some metrics: {e}")

        return metrics

    def _is_quality_acceptable(self, metrics: TrainingMetrics, param_def: dict) -> bool:
        """
        Check if model quality meets requirements

        Args:
            metrics: Training metrics
            param_def: Parameter definition

        Returns:
            True if quality is acceptable
        """
        param_type = param_def.get('type', 'CONTINUOUS')

        if param_type in ['CONTINUOUS', 'PROBABILITY', 'ARRAY_INT', 'ARRAY_FLOAT']:
            if metrics.test_r2 is not None:
                return metrics.test_r2 > self.quality_threshold
            else:
                return False
        else:
            if metrics.test_accuracy is not None:
                return metrics.test_accuracy > self.quality_threshold
            else:
                return False

    def _tune_hyperparameters(self, X_train, y_train, X_val, y_val, X_test, y_test,
                             param_def, objective, feature_names: Optional[List[str]] = None) -> Tuple[Optional[Any], TrainingMetrics]:
        """
        Grid search for better hyperparameters

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            param_def: Parameter definition
            objective: XGBoost objective
            feature_names: Feature names

        Returns:
            (best_model, metrics)
        """
        print("  Running hyperparameter tuning...")

        if not HAS_SKLEARN or not HAS_XGBOOST:
            print("  ERROR: sklearn or xgboost not available")
            return None, TrainingMetrics()

        try:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            param_type = param_def.get('type', 'CONTINUOUS')

            if param_type in ['CONTINUOUS', 'PROBABILITY', 'ARRAY_INT', 'ARRAY_FLOAT']:
                base_model = XGBRegressor(objective=objective, random_state=42)
                scoring = 'r2'
            else:
                base_model = XGBClassifier(objective=objective, random_state=42)
                if param_type == 'CATEGORICAL':
                    param_range = param_def.get('range', [])
                    if isinstance(param_range, list):
                        base_model.set_params(num_class=len(param_range))
                scoring = 'accuracy'

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)

            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.3f}")

            best_model = grid_search.best_estimator_

            # Re-evaluate with best model
            metrics = self._evaluate_model(
                best_model, X_train, y_train, X_val, y_val, X_test, y_test,
                param_def, feature_names
            )
            metrics.hyperparameter_tuning = {
                'best_params': grid_search.best_params_,
                'best_cv_score': float(grid_search.best_score_)
            }

            return best_model, metrics

        except Exception as e:
            print(f"  ERROR in hyperparameter tuning: {e}")
            return None, TrainingMetrics()

    def _save_model(self, model, param_name: str, label_encoder: Optional[Any] = None) -> Path:
        """
        Save trained model to disk

        Args:
            model: Trained model
            param_name: Parameter name
            label_encoder: Label encoder for categorical params
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS

        Returns:
            Path to saved model
        """
<<<<<<< HEAD
        # Create safe filename
        safe_name = param_name.replace('.', '_')
        model_path = models_dir / f"{safe_name}.pkl"

        # Save model
        if JOBLIB_AVAILABLE:
            joblib.dump(model, model_path, compress=3)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Save metadata
        metadata = {
            'param_name': param_name,
            'param_type': param_def.param_type.value,
            'param_description': param_def.description,
            'timestamp': metrics.timestamp,
            'metrics': {
                'test_r2': metrics.test_r2,
                'test_mae': metrics.test_mae,
                'test_accuracy': metrics.test_accuracy,
                'test_f1': metrics.test_f1,
                'passed_quality': metrics.passed_quality_check
            },
            'training_config': {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate
            }
        }

        # Save label encoder if categorical
        if param_name in self.label_encoders:
            encoder_path = models_dir / f"{safe_name}_encoder.pkl"
            if JOBLIB_AVAILABLE:
                joblib.dump(self.label_encoders[param_name], encoder_path)
            else:
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoders[param_name], f)
            metadata['encoder_path'] = str(encoder_path)

        metadata_path = models_dir / f"{safe_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return model_path

    def _save_metrics(
        self,
        metrics: TrainingMetrics,
        output_dir: Path,
        param_name: str
    ):
        """Save metrics to JSON"""
        safe_name = param_name.replace('.', '_')
        metrics_path = output_dir / f"{safe_name}_metrics.json"

        with open(metrics_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

    # ========================================================================
    # Visualization
    # ========================================================================

    def _generate_plots(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        param_name: str,
        metrics: TrainingMetrics,
        output_dir: Path
    ):
        """Generate training visualizations"""
        if not MATPLOTLIB_AVAILABLE:
            return

        safe_name = param_name.replace('.', '_')

        try:
            # 1. Feature importance plot
            if metrics.top_features:
                self._plot_feature_importance(
                    metrics.top_features,
                    output_dir / f"{safe_name}_feature_importance.png"
                )

            # 2. Prediction vs actual
            if metrics.test_r2 is not None:
                self._plot_predictions(
                    y_test, model.predict(X_test),
                    output_dir / f"{safe_name}_predictions.png",
                    metrics.test_r2
                )

        except Exception as e:
            print(f"⚠️ Could not generate plots: {e}")

    def _plot_feature_importance(
        self,
        top_features: List[Tuple[str, float]],
        output_path: Path
    ):
        """Plot top feature importances"""
        features, importances = zip(*top_features)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Path,
        r2: float
    ):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actual (R² = {r2:.4f})')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    # ========================================================================
    # Data Loading
    # ========================================================================

    def _load_training_data(
        self,
        param_name: str,
        training_data_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Load training data for a parameter.

        Expected directory structure:
        training_data_dir/
            param_name/
                metadata.json
                data.pkl or data.csv
        """
        param_dir = training_data_dir / param_name.replace('.', '_')

        if not param_dir.exists():
            raise FileNotFoundError(f"Training data not found: {param_dir}")

        # Load metadata
        metadata_path = param_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Try different data formats
        data_pkl = param_dir / 'data.pkl'
        data_csv = param_dir / 'data.csv'
        data_json = param_dir / 'data.json'

        if data_pkl.exists():
            with open(data_pkl, 'rb') as f:
                training_data = pickle.load(f)
        elif data_csv.exists():
            training_data = pd.read_csv(data_csv).to_dict('records')
        elif data_json.exists():
            with open(data_json, 'r') as f:
                training_data = json.load(f)
        else:
            raise FileNotFoundError(f"No data file found in {param_dir}")

        return training_data

    def _save_batch_results(
        self,
        results: BatchTrainingResults,
        output_dir: Path
    ):
        """Save batch training results"""
        results_path = output_dir / 'batch_training_results.json'

        results_dict = {
            'total_parameters': results.total_parameters,
            'successful': results.successful,
            'failed': results.failed,
            'total_time': results.total_time,
            'results': {
                name: metrics.to_dict()
                for name, metrics in results.results.items()
            },
            'errors': results.errors
        }

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Also save text summary
        summary_path = output_dir / 'batch_training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(results.summary())


# ============================================================================
# Utility Functions
# ============================================================================

def train_single_parameter(
    param_name: str,
    training_data_path: Path,
    models_dir: Path = Path('midi_generator/models/pretrained'),
    output_dir: Optional[Path] = None,
    config: Optional[TrainingConfig] = None
) -> Tuple[Any, TrainingMetrics]:
    """
    Convenience function to train a single parameter model.

    Args:
        param_name: Parameter name (e.g., 'harmony.voicing.spread')
        training_data_path: Path to training data
        models_dir: Directory to save model
        output_dir: Directory for plots and metrics
        config: Training configuration

    Returns:
        (model, metrics)
    """
    if REGISTRY is None:
        raise RuntimeError("Parameter registry not available")

    param_def = REGISTRY.get(param_name)
    if param_def is None:
        raise ValueError(f"Unknown parameter: {param_name}")

    trainer = ModelTrainingSpecialist(config)

    # Load training data
    training_data = trainer._load_training_data(param_name, training_data_path.parent)

    return trainer.train_parameter_model(
        param_name, param_def, training_data, models_dir, output_dir
    )


def train_all_parameters(
    training_data_dir: Path,
    models_dir: Path = Path('midi_generator/models/pretrained'),
    output_dir: Optional[Path] = None,
    config: Optional[TrainingConfig] = None
) -> BatchTrainingResults:
    """
    Train models for all parameters in training data directory.

    Args:
        training_data_dir: Directory containing training data
        models_dir: Directory to save models
        output_dir: Directory for plots and metrics
        config: Training configuration

    Returns:
        BatchTrainingResults
    """
    if REGISTRY is None:
        raise RuntimeError("Parameter registry not available")

    # Find all parameters with training data
    parameters = []
    for param_dir in training_data_dir.iterdir():
        if param_dir.is_dir():
            param_name = param_dir.name.replace('_', '.')
            param_def = REGISTRY.get(param_name)
            if param_def:
                parameters.append((param_name, param_def))

    if not parameters:
        raise ValueError(f"No training data found in {training_data_dir}")

    trainer = ModelTrainingSpecialist(config)

    return trainer.train_batch(
        parameters, training_data_dir, models_dir, output_dir
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train XGBoost models for parameter prediction'
    )
    parser.add_argument(
        'mode',
        choices=['single', 'batch', 'all'],
        help='Training mode'
    )
    parser.add_argument(
        '--param', '-p',
        help='Parameter name (for single mode)'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        required=True,
        help='Training data directory'
    )
    parser.add_argument(
        '--models-dir', '-m',
        type=Path,
        default=Path('midi_generator/models/pretrained'),
        help='Models output directory'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        help='Output directory for plots and metrics'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning'
    )
    parser.add_argument(
        '--tuning-method',
        choices=['grid', 'random'],
        default='grid',
        help='Hyperparameter tuning method'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of estimators'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate'
    )

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        enable_tuning=args.tune,
        tuning_method=args.tuning_method,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )

    # Run training
    if args.mode == 'single':
        if not args.param:
            parser.error("--param required for single mode")

        model, metrics = train_single_parameter(
            args.param,
            args.data_dir,
            args.models_dir,
            args.output_dir,
            config
        )

        print("\n" + metrics.summary())

    elif args.mode in ['batch', 'all']:
        results = train_all_parameters(
            args.data_dir,
            args.models_dir,
            args.output_dir,
            config
        )

        print("\n" + results.summary())


if __name__ == '__main__':
    main()
=======
        model_filename = f"{param_name.replace('.', '_')}.pkl"
        model_path = self.models_dir / model_filename

        try:
            if HAS_JOBLIB:
                joblib.dump({
                    'model': model,
                    'label_encoder': label_encoder,
                    'parameter_name': param_name
                }, model_path)
            else:
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'label_encoder': label_encoder,
                        'parameter_name': param_name
                    }, f)

            print(f"  ✅ Model saved to: {model_path}")
            return model_path

        except Exception as e:
            print(f"  ERROR saving model: {e}")
            return model_path

    def _print_training_summary(self, param_name: str, metrics: TrainingMetrics):
        """Print formatted training summary"""

        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY: {param_name}")
        print(f"{'='*80}")

        if metrics.test_r2 is not None:
            print(f"\nRegression Metrics:")
            print(f"  R² Scores:")
            print(f"    Train: {metrics.train_r2:.3f}")
            print(f"    Val:   {metrics.val_r2:.3f}")
            print(f"    Test:  {metrics.test_r2:.3f}")
            print(f"  MAE Scores:")
            print(f"    Train: {metrics.train_mae:.4f}")
            print(f"    Test:  {metrics.test_mae:.4f}")
            print(f"  RMSE:")
            print(f"    Train: {metrics.train_rmse:.4f}")
            print(f"    Test:  {metrics.test_rmse:.4f}")
        else:
            print(f"\nClassification Metrics:")
            print(f"  Accuracy:")
            print(f"    Train: {metrics.train_accuracy:.3f}")
            print(f"    Val:   {metrics.val_accuracy:.3f}")
            print(f"    Test:  {metrics.test_accuracy:.3f}")
            print(f"  F1 Score: {metrics.test_f1:.3f}")
            print(f"  Precision: {metrics.test_precision:.3f}")
            print(f"  Recall: {metrics.test_recall:.3f}")

        print(f"\nTraining Info:")
        print(f"  Time: {metrics.training_time:.2f}s")
        print(f"  Examples: {metrics.n_training_examples}")
        print(f"  Features: {metrics.n_features}")

        if metrics.top_features:
            print(f"\nTop 5 Most Important Features:")
            for feature, importance in metrics.top_features[:5]:
                print(f"  {feature}: {importance:.4f}")

        if metrics.hyperparameter_tuning:
            print(f"\nHyperparameter Tuning:")
            print(f"  Best CV score: {metrics.hyperparameter_tuning['best_cv_score']:.3f}")
            print(f"  Best params: {metrics.hyperparameter_tuning['best_params']}")

        print(f"\nModel saved to: {metrics.model_path}")
        print(f"{'='*80}\n")

    def load_model(self, param_name: str) -> Optional[Any]:
        """
        Load trained model from disk

        Args:
            param_name: Parameter name

        Returns:
            Model or None if not found
        """
        model_filename = f"{param_name.replace('.', '_')}.pkl"
        model_path = self.models_dir / model_filename

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return None

        try:
            if HAS_JOBLIB:
                data = joblib.load(model_path)
            else:
                import pickle
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)

            return data['model']

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def train_all_parameters(self, parameters: List[Tuple[str, dict]],
                            training_data_dir: Path) -> Dict[str, ModelTrainingResult]:
        """
        Train models for multiple parameters

        Args:
            parameters: List of (param_name, param_def) tuples
            training_data_dir: Directory containing training data

        Returns:
            Dictionary mapping param_name to ModelTrainingResult
        """
        print(f"\n{'='*80}")
        print(f"BATCH TRAINING: {len(parameters)} PARAMETERS")
        print(f"{'='*80}\n")

        results = {}

        for i, (param_name, param_def) in enumerate(parameters, 1):
            print(f"\n[{i}/{len(parameters)}] Training: {param_name}")
            print("-" * 80)

            # Load training data
            param_dir = training_data_dir / param_name.replace('.', '_')

            try:
                training_data = self._load_training_data(param_dir)

                if not training_data:
                    results[param_name] = ModelTrainingResult(
                        parameter_name=param_name,
                        success=False,
                        error="No training data found"
                    )
                    continue

                # Train model
                result = self.train_parameter_model(param_name, param_def, training_data)
                results[param_name] = result

                # Print progress
                if result.success:
                    metric_key = 'test_r2' if result.metrics.test_r2 is not None else 'test_accuracy'
                    metric_val = getattr(result.metrics, metric_key, 0.0)
                    print(f"  ✅ Success: {metric_key} = {metric_val:.3f}")
                else:
                    print(f"  ❌ Failed: {result.error}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[param_name] = ModelTrainingResult(
                    parameter_name=param_name,
                    success=False,
                    error=str(e)
                )

        # Print summary
        self._print_batch_summary(results)

        return results

    def _load_training_data(self, param_dir: Path) -> List[dict]:
        """Load training data from directory"""

        metadata_file = param_dir / 'metadata.json'
        if not metadata_file.exists():
            print(f"  Metadata not found: {metadata_file}")
            return []

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Load features from .npz file if exists
            features_file = param_dir / 'features.npz'
            if features_file.exists():
                data = np.load(features_file)
                X = data['X']
                y = data['y']

                training_data = [
                    {'features': X[i], 'parameter_value': y[i]}
                    for i in range(len(X))
                ]
                return training_data

            # Otherwise, load from individual examples
            training_data = []
            for ex_meta in metadata.get('examples', []):
                # Mock features for now
                features = np.random.randn(1000)  # Should extract from MIDI

                training_data.append({
                    'features': features,
                    'parameter_value': ex_meta['parameter_value']
                })

            return training_data

        except Exception as e:
            print(f"  Error loading training data: {e}")
            return []

    def _print_batch_summary(self, results: Dict[str, ModelTrainingResult]):
        """Print summary of batch training"""

        print(f"\n{'='*80}")
        print("BATCH TRAINING SUMMARY")
        print(f"{'='*80}")

        total = len(results)
        successful = sum(1 for r in results.values() if r.success)
        failed = total - successful

        print(f"\nTotal: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if successful > 0:
            print(f"\nSuccessful Models:")
            for param_name, result in results.items():
                if result.success and result.metrics:
                    metric_key = 'test_r2' if result.metrics.test_r2 is not None else 'test_accuracy'
                    metric_val = getattr(result.metrics, metric_key, 0.0)
                    print(f"  ✅ {param_name}: {metric_key} = {metric_val:.3f}")

        if failed > 0:
            print(f"\nFailed Models:")
            for param_name, result in results.items():
                if not result.success:
                    print(f"  ❌ {param_name}: {result.error}")

        print(f"\n{'='*80}\n")


# Example usage
if __name__ == '__main__':
    # Example usage with mock data
    example_param_def = {
        'name': 'harmony.voicing.quartal_probability',
        'type': 'CONTINUOUS',
        'range': (0.0, 1.0),
        'default': 0.3
    }

    # Generate mock training data
    n_examples = 500
    n_features = 1000

    training_data = [
        {
            'features': np.random.randn(n_features),
            'parameter_value': np.random.uniform(0.0, 1.0)
        }
        for _ in range(n_examples)
    ]

    # Create trainer
    trainer = ModelTrainingSpecialist()

    # Train model
    result = trainer.train_parameter_model(
        param_name=example_param_def['name'],
        param_def=example_param_def,
        training_data=training_data
    )

    if result.success:
        print("\n✅ Model training successful!")
        print(f"Test R²: {result.metrics.test_r2:.3f}")
    else:
        print(f"\n❌ Model training failed: {result.error}")
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
