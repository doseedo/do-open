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

Author: Agent 15 - Model Training Specialist
License: MIT
"""

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


@dataclass
class TrainingMetrics:
    """Metrics from model training"""
    # Performance metrics
    train_r2: Optional[float] = None
    val_r2: Optional[float] = None
    test_r2: Optional[float] = None
    train_mae: Optional[float] = None
    val_mae: Optional[float] = None
    test_mae: Optional[float] = None
    train_rmse: Optional[float] = None
    test_rmse: Optional[float] = None

    # Classification metrics
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    train_f1: Optional[float] = None
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

        Returns:
            Path to saved model
        """
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
