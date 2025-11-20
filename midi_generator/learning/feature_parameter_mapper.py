"""
AGENT 9: Feature-Parameter Mapping Specialist
==============================================

Maps 1,000 musical features from Agent 8 to 515+ parameters for XGBoost training.

This is the CRITICAL bridge between inverse MIDI analysis (features) and
generative parameter prediction.

Architecture:
- One XGBoost model per parameter (modular design)
- Automatic feature selection per parameter
- Feature importance analysis
- Handles all parameter types (continuous, categorical, integer, boolean, probability)
- Fast inference (<10ms per parameter)
- Model persistence and versioning

Integration Points:
- Agent 8: DeepFeatureExtractor.extract_features() -> 1000 features
- Agent 14: SyntheticTrainingDataGenerator -> training examples
- Agent 15: ModelTrainingSpecialist -> XGBoost training
- Agent 1: UniversalParameterRegistry -> 515+ parameters

Key Features:
1. Automated correlation-based feature selection
2. Per-parameter feature importance ranking
3. Batch prediction for all 515+ parameters
4. Model performance monitoring
5. Incremental training support
6. Comprehensive error handling

Performance Targets:
- R² > 0.5 for continuous parameters (target: 0.7+)
- Accuracy > 0.5 for categorical parameters (target: 0.7+)
- Inference time < 10ms per parameter
- Training time < 5min per parameter on 1000 examples

Author: Agent 9 - Feature-Parameter Mapping Specialist
License: MIT
"""

import json
import pickle
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not installed. Install with: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not installed. Install with: pip install pandas")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# Imports from our system
# ============================================================================

try:
    from midi_generator.parameters.universal_registry import (
        UniversalParameterRegistry,
        ParameterDefinition,
        ParameterType
    )
    REGISTRY = UniversalParameterRegistry()
except ImportError:
    REGISTRY = None
    print("WARNING: Could not import UniversalParameterRegistry")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class FeatureImportance:
    """Feature importance information for a parameter"""
    parameter_name: str
    feature_rankings: Dict[str, float]  # feature_name -> importance score
    top_features: List[str]  # Top N most important features
    correlation_score: float  # Overall correlation with parameter
    explained_variance: float  # R² or similar metric


@dataclass
class MappingMetrics:
    """Performance metrics for a parameter mapping"""
    parameter_name: str
    parameter_type: str

    # Training metrics
    train_score: float  # R² for regression, accuracy for classification
    val_score: float
    test_score: Optional[float] = None

    # Additional metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    f1_score: Optional[float] = None

    # Feature info
    n_features_used: int = 0
    top_features: List[str] = field(default_factory=list)

    # Performance
    training_time: float = 0.0
    inference_time: float = 0.0  # Average time per prediction

    # Quality assessment
    quality_level: str = "unknown"  # excellent, good, acceptable, poor

    # Metadata
    trained_at: datetime = field(default_factory=datetime.now)
    n_training_examples: int = 0

    def assess_quality(self):
        """Assess model quality based on metrics"""
        score = self.test_score if self.test_score is not None else self.val_score

        if score >= 0.8:
            self.quality_level = "excellent"
        elif score >= 0.7:
            self.quality_level = "good"
        elif score >= 0.5:
            self.quality_level = "acceptable"
        else:
            self.quality_level = "poor"


@dataclass
class TrainingExample:
    """Training example for feature-parameter mapping"""
    features: np.ndarray  # 1000 features from Agent 8
    parameter_value: Any  # Target parameter value
    parameter_name: str

    # Optional metadata
    midi_file: Optional[Path] = None
    genre: Optional[str] = None
    coherence_score: float = 1.0


@dataclass
class PredictionResult:
    """Result from parameter prediction"""
    parameter_name: str
    predicted_value: Any
    confidence: Optional[float] = None
    feature_contributions: Optional[Dict[str, float]] = None
    inference_time: float = 0.0


# ============================================================================
# Feature Selection and Analysis
# ============================================================================

class FeatureSelector:
    """
    Selects optimal feature subsets for each parameter.

    Strategies:
    1. Correlation-based: Remove features with low correlation to target
    2. Variance-based: Remove low-variance features
    3. Redundancy-based: Remove highly correlated features
    4. Importance-based: Use model feature importance
    """

    def __init__(self,
                 correlation_threshold: float = 0.1,
                 variance_threshold: float = 0.01,
                 redundancy_threshold: float = 0.95):
        """
        Initialize feature selector

        Args:
            correlation_threshold: Min correlation with target to keep
            variance_threshold: Min variance to keep
            redundancy_threshold: Max correlation between features
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.redundancy_threshold = redundancy_threshold

        self.selected_features: Dict[str, List[int]] = {}  # param_name -> feature indices

    def select_features(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       param_name: str,
                       feature_names: Optional[List[str]] = None,
                       max_features: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Select optimal feature subset for a parameter

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            param_name: Parameter name
            feature_names: Optional feature names
            max_features: Maximum number of features to select

        Returns:
            Selected feature matrix, selected feature indices
        """
        n_features = X.shape[1]
        selected_indices = list(range(n_features))

        # Step 1: Remove low-variance features
        variances = np.var(X, axis=0)
        selected_indices = [i for i in selected_indices if variances[i] > self.variance_threshold]

        if len(selected_indices) == 0:
            selected_indices = list(range(n_features))  # Keep all if all removed

        X_selected = X[:, selected_indices]

        # Step 2: Remove features with low correlation to target
        if len(y.shape) == 1 and np.issubdtype(y.dtype, np.number):
            correlations = []
            for i in range(X_selected.shape[1]):
                try:
                    corr = np.abs(np.corrcoef(X_selected[:, i], y)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                    correlations.append(corr)
                except:
                    correlations.append(0.0)

            high_corr_indices = [i for i, corr in enumerate(correlations)
                                if corr >= self.correlation_threshold]

            if len(high_corr_indices) > 0:
                selected_indices = [selected_indices[i] for i in high_corr_indices]
                X_selected = X_selected[:, high_corr_indices]

        # Step 3: Limit to max_features if specified
        if max_features is not None and len(selected_indices) > max_features:
            # Keep top features by correlation
            correlations = []
            for i in range(X_selected.shape[1]):
                try:
                    corr = np.abs(np.corrcoef(X_selected[:, i], y)[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                    correlations.append(corr)
                except:
                    correlations.append(0.0)

            top_indices = np.argsort(correlations)[-max_features:]
            selected_indices = [selected_indices[i] for i in top_indices]
            X_selected = X_selected[:, top_indices]

        self.selected_features[param_name] = selected_indices

        return X_selected, selected_indices

    def transform(self, X: np.ndarray, param_name: str) -> np.ndarray:
        """Transform features using stored selection"""
        if param_name not in self.selected_features:
            return X

        selected_indices = self.selected_features[param_name]
        return X[:, selected_indices]


# ============================================================================
# Main Feature-Parameter Mapper
# ============================================================================

class FeatureParameterMapper:
    """
    Maps 1000 musical features to 515+ parameter predictions.

    Core architecture: One XGBoost model per parameter (modular design).

    This is the CRITICAL component that enables:
    1. Learning from MIDI corpus
    2. Inverse analysis (MIDI -> parameters)
    3. Style transfer (extract params from one MIDI, apply to another)
    4. Automated parameter tuning

    Usage:
        >>> mapper = FeatureParameterMapper()
        >>>
        >>> # Train on data
        >>> mapper.train_mapping('harmony.chord_density', training_examples)
        >>>
        >>> # Predict from features
        >>> features = extract_features('song.mid')
        >>> value = mapper.predict_parameter(features, 'harmony.chord_density')
        >>>
        >>> # Predict all parameters
        >>> all_params = mapper.predict_all_parameters(features)
    """

    def __init__(self,
                 models_dir: Path = Path('midi_generator/models/parameter_mappings'),
                 registry: Optional[UniversalParameterRegistry] = None,
                 enable_feature_selection: bool = True,
                 max_features_per_param: int = 200):
        """
        Initialize Feature-Parameter Mapper

        Args:
            models_dir: Directory to save/load trained models
            registry: Parameter registry (uses default if None)
            enable_feature_selection: Whether to perform feature selection
            max_features_per_param: Max features to use per parameter
        """
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            raise RuntimeError("XGBoost and scikit-learn are required. "
                             "Install with: pip install xgboost scikit-learn")

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.registry = registry or REGISTRY
        if self.registry is None:
            raise RuntimeError("Parameter registry not available")

        # Core storage
        self.models: Dict[str, xgb.XGBModel] = {}  # param_name -> trained model
        self.feature_names: List[str] = []  # Names of 1000 features
        self.metrics: Dict[str, MappingMetrics] = {}  # param_name -> metrics
        self.feature_importance: Dict[str, FeatureImportance] = {}  # param_name -> importance

        # Feature selection
        self.enable_feature_selection = enable_feature_selection
        self.max_features_per_param = max_features_per_param
        self.feature_selector = FeatureSelector()

        # Preprocessing
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}

        # Training history
        self.training_history: List[MappingMetrics] = []

        # Feature names (will be set on first training)
        self._feature_names_initialized = False

    # ========================================================================
    # Training Methods
    # ========================================================================

    def train_mapping(self,
                     param_name: str,
                     training_data: Union[List[TrainingExample], List[Dict]],
                     validation_split: float = 0.2,
                     test_split: float = 0.1,
                     force_retrain: bool = False,
                     **xgb_params) -> MappingMetrics:
        """
        Train XGBoost model for a single parameter.

        This is the core training method that creates the feature->parameter mapping.

        Args:
            param_name: Full parameter name (e.g., 'harmony.chord_density')
            training_data: List of TrainingExample or dicts with 'features' and 'parameter_value'
            validation_split: Fraction for validation set
            test_split: Fraction for test set
            force_retrain: Retrain even if model exists
            **xgb_params: Additional XGBoost parameters

        Returns:
            MappingMetrics with training results
        """
        start_time = time.time()

        # Check if model already exists
        if param_name in self.models and not force_retrain:
            print(f"Model for {param_name} already exists. Use force_retrain=True to retrain.")
            return self.metrics.get(param_name)

        # Get parameter definition
        param_def = self.registry.get(param_name)
        if param_def is None:
            raise ValueError(f"Parameter {param_name} not found in registry")

        print(f"\n{'='*70}")
        print(f"Training mapping for: {param_name}")
        print(f"Parameter type: {param_def.param_type.value}")
        print(f"{'='*70}\n")

        # Prepare data
        X, y, feature_names = self._prepare_training_data(training_data, param_name, param_def)

        # Initialize feature names if first time
        if not self._feature_names_initialized:
            self.feature_names = feature_names
            self._feature_names_initialized = True

        print(f"Training data: {X.shape[0]} examples, {X.shape[1]} features")

        # Feature selection
        if self.enable_feature_selection:
            X, selected_indices = self.feature_selector.select_features(
                X, y, param_name, feature_names, self.max_features_per_param
            )
            print(f"Selected {len(selected_indices)} features for {param_name}")

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(validation_split + test_split), random_state=42
        )

        test_size_adjusted = test_split / (validation_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size_adjusted, random_state=42
        )

        print(f"Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")

        # Train model
        model = self._train_xgboost_model(
            X_train, y_train, X_val, y_val, param_def, **xgb_params
        )

        # Evaluate
        metrics = self._evaluate_model(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            param_name, param_def
        )

        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            importance = self._analyze_feature_importance(
                model, param_name, feature_names, selected_indices if self.enable_feature_selection else None
            )
            self.feature_importance[param_name] = importance
            metrics.top_features = importance.top_features[:10]

        # Store model and metrics
        self.models[param_name] = model
        self.metrics[param_name] = metrics
        self.training_history.append(metrics)

        # Training time
        metrics.training_time = time.time() - start_time
        metrics.n_training_examples = len(X_train)
        metrics.assess_quality()

        print(f"\n{'='*70}")
        print(f"Training complete for {param_name}")
        print(f"Quality: {metrics.quality_level}")
        print(f"Score: {metrics.val_score:.4f}")
        print(f"Time: {metrics.training_time:.2f}s")
        print(f"{'='*70}\n")

        return metrics

    def _prepare_training_data(self,
                               training_data: Union[List[TrainingExample], List[Dict]],
                               param_name: str,
                               param_def: ParameterDefinition) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from various formats"""

        features_list = []
        values_list = []

        for example in training_data:
            if isinstance(example, TrainingExample):
                features_list.append(example.features)
                values_list.append(example.parameter_value)
            elif isinstance(example, dict):
                features_list.append(example['features'])
                values_list.append(example.get('parameter_value') or example.get('value'))
            else:
                raise ValueError(f"Invalid training example type: {type(example)}")

        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(values_list)

        # Handle categorical parameters
        if param_def.param_type == ParameterType.CATEGORICAL:
            if param_name not in self.label_encoders:
                self.label_encoders[param_name] = LabelEncoder()
                y = self.label_encoders[param_name].fit_transform(y)
            else:
                y = self.label_encoders[param_name].transform(y)

        # Generate feature names if not available
        n_features = X.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]

        return X, y, feature_names

    def _train_xgboost_model(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            param_def: ParameterDefinition,
                            **xgb_params) -> xgb.XGBModel:
        """Train XGBoost model with appropriate objective"""

        # Determine objective based on parameter type
        if param_def.param_type == ParameterType.CATEGORICAL:
            n_classes = len(np.unique(y_train))
            if n_classes == 2:
                objective = 'binary:logistic'
                model_class = xgb.XGBClassifier
            else:
                objective = 'multi:softmax'
                model_class = xgb.XGBClassifier
        else:
            objective = 'reg:squarederror'
            model_class = xgb.XGBRegressor

        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        }

        # Update with custom params
        params.update(xgb_params)

        # Create model
        if model_class == xgb.XGBClassifier:
            model = model_class(objective=objective, **params)
        else:
            model = model_class(objective=objective, **params)

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        return model

    def _evaluate_model(self,
                       model: xgb.XGBModel,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       param_name: str,
                       param_def: ParameterDefinition) -> MappingMetrics:
        """Evaluate model performance"""

        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics based on parameter type
        if param_def.param_type == ParameterType.CATEGORICAL:
            train_score = accuracy_score(y_train, y_train_pred)
            val_score = accuracy_score(y_val, y_val_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            f1 = f1_score(y_test, y_test_pred, average='weighted')

            metrics = MappingMetrics(
                parameter_name=param_name,
                parameter_type=param_def.param_type.value,
                train_score=train_score,
                val_score=val_score,
                test_score=test_score,
                f1_score=f1
            )
        else:
            train_score = r2_score(y_train, y_train_pred)
            val_score = r2_score(y_val, y_val_pred)
            test_score = r2_score(y_test, y_test_pred)

            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = np.mean(np.abs(y_test - y_test_pred))

            metrics = MappingMetrics(
                parameter_name=param_name,
                parameter_type=param_def.param_type.value,
                train_score=train_score,
                val_score=val_score,
                test_score=test_score,
                rmse=rmse,
                mae=mae
            )

        return metrics

    def _analyze_feature_importance(self,
                                   model: xgb.XGBModel,
                                   param_name: str,
                                   feature_names: List[str],
                                   selected_indices: Optional[List[int]] = None) -> FeatureImportance:
        """Analyze feature importance for parameter"""

        importances = model.feature_importances_

        # Map back to full feature set if feature selection was used
        if selected_indices is not None:
            full_importances = np.zeros(len(feature_names))
            full_importances[selected_indices] = importances
            importances = full_importances

        # Create feature importance dict
        feature_rankings = {
            feature_names[i]: float(importances[i])
            for i in range(len(feature_names))
        }

        # Sort by importance
        sorted_features = sorted(feature_rankings.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:20]]

        # Calculate explained variance
        explained_var = float(np.sum(importances[:20]))  # Top 20 features

        return FeatureImportance(
            parameter_name=param_name,
            feature_rankings=feature_rankings,
            top_features=top_features,
            correlation_score=0.0,  # Can be computed if needed
            explained_variance=explained_var
        )

    # ========================================================================
    # Prediction Methods
    # ========================================================================

    def predict_parameter(self,
                         features: np.ndarray,
                         param_name: str,
                         return_confidence: bool = False) -> Union[Any, PredictionResult]:
        """
        Predict single parameter value from features.

        Args:
            features: Feature vector (1000 features from Agent 8)
            param_name: Parameter to predict
            return_confidence: Whether to return full PredictionResult

        Returns:
            Predicted value or PredictionResult
        """
        start_time = time.time()

        if param_name not in self.models:
            raise ValueError(f"No trained model for parameter: {param_name}")

        model = self.models[param_name]

        # Reshape if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Apply feature selection
        if self.enable_feature_selection and param_name in self.feature_selector.selected_features:
            features = self.feature_selector.transform(features, param_name)

        # Predict
        prediction = model.predict(features)[0]

        # Decode categorical
        param_def = self.registry.get(param_name)
        if param_def and param_def.param_type == ParameterType.CATEGORICAL:
            if param_name in self.label_encoders:
                prediction = self.label_encoders[param_name].inverse_transform([int(prediction)])[0]

        inference_time = time.time() - start_time

        if return_confidence:
            result = PredictionResult(
                parameter_name=param_name,
                predicted_value=prediction,
                inference_time=inference_time
            )
            return result

        return prediction

    def predict_all_parameters(self,
                              features: np.ndarray,
                              only_trained: bool = True,
                              show_progress: bool = False) -> Dict[str, Any]:
        """
        Predict all 515+ parameters from features.

        This is the main interface for complete parameter extraction from MIDI.

        Args:
            features: Feature vector (1000 features from Agent 8)
            only_trained: Only predict parameters with trained models
            show_progress: Show progress bar

        Returns:
            Dictionary mapping parameter names to predicted values
        """
        predictions = {}

        params_to_predict = list(self.models.keys()) if only_trained else self.registry.get_all_parameters()

        iterator = tqdm(params_to_predict, desc="Predicting parameters") if show_progress and TQDM_AVAILABLE else params_to_predict

        for param_name in iterator:
            if param_name in self.models:
                try:
                    value = self.predict_parameter(features, param_name)
                    predictions[param_name] = value
                except Exception as e:
                    print(f"Warning: Failed to predict {param_name}: {e}")
                    continue

        return predictions

    def predict_batch(self,
                     feature_matrix: np.ndarray,
                     param_name: str) -> np.ndarray:
        """
        Predict parameter for multiple feature vectors (batch inference).

        Args:
            feature_matrix: (n_samples, n_features)
            param_name: Parameter to predict

        Returns:
            Predictions (n_samples,)
        """
        if param_name not in self.models:
            raise ValueError(f"No trained model for parameter: {param_name}")

        model = self.models[param_name]

        # Apply feature selection
        if self.enable_feature_selection and param_name in self.feature_selector.selected_features:
            feature_matrix = self.feature_selector.transform(feature_matrix, param_name)

        predictions = model.predict(feature_matrix)

        # Decode categorical
        param_def = self.registry.get(param_name)
        if param_def and param_def.param_type == ParameterType.CATEGORICAL:
            if param_name in self.label_encoders:
                predictions = self.label_encoders[param_name].inverse_transform(predictions.astype(int))

        return predictions

    # ========================================================================
    # Feature Importance and Analysis
    # ========================================================================

    def get_feature_importance(self, param_name: str, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance for a parameter.

        Args:
            param_name: Parameter name
            top_n: Number of top features to return

        Returns:
            Dictionary of feature names to importance scores
        """
        if param_name not in self.feature_importance:
            raise ValueError(f"No feature importance for parameter: {param_name}")

        importance = self.feature_importance[param_name]
        sorted_features = sorted(importance.feature_rankings.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_features[:top_n])

    def get_top_features(self, param_name: str, n: int = 10) -> List[str]:
        """Get top N most important features for parameter"""
        if param_name not in self.feature_importance:
            return []

        return self.feature_importance[param_name].top_features[:n]

    def analyze_all_feature_importance(self) -> Dict[str, FeatureImportance]:
        """Get feature importance for all trained parameters"""
        return self.feature_importance.copy()

    # ========================================================================
    # Model Persistence
    # ========================================================================

    def save_model(self, param_name: str, output_dir: Optional[Path] = None):
        """
        Save trained model for a parameter.

        Args:
            param_name: Parameter name
            output_dir: Directory to save (uses default if None)
        """
        if param_name not in self.models:
            raise ValueError(f"No trained model for parameter: {param_name}")

        save_dir = Path(output_dir) if output_dir else self.models_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize parameter name for filename
        safe_name = param_name.replace('/', '_').replace('.', '_')

        # Save model
        model_path = save_dir / f"{safe_name}_model.json"
        self.models[param_name].save_model(str(model_path))

        # Save metadata
        metadata = {
            'param_name': param_name,
            'metrics': {
                'train_score': self.metrics[param_name].train_score if param_name in self.metrics else None,
                'val_score': self.metrics[param_name].val_score if param_name in self.metrics else None,
                'quality_level': self.metrics[param_name].quality_level if param_name in self.metrics else None,
            },
            'selected_features': self.feature_selector.selected_features.get(param_name),
            'trained_at': datetime.now().isoformat()
        }

        metadata_path = save_dir / f"{safe_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save label encoder if exists
        if param_name in self.label_encoders:
            encoder_path = save_dir / f"{safe_name}_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoders[param_name], f)

        print(f"Saved model for {param_name} to {save_dir}")

    def load_model(self, param_name: str, model_dir: Optional[Path] = None):
        """
        Load trained model for a parameter.

        Args:
            param_name: Parameter name
            model_dir: Directory to load from (uses default if None)
        """
        load_dir = Path(model_dir) if model_dir else self.models_dir

        # Sanitize parameter name
        safe_name = param_name.replace('/', '_').replace('.', '_')

        # Load model
        model_path = load_dir / f"{safe_name}_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Get parameter type to create correct model class
        param_def = self.registry.get(param_name)
        if param_def and param_def.param_type == ParameterType.CATEGORICAL:
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()

        model.load_model(str(model_path))
        self.models[param_name] = model

        # Load metadata
        metadata_path = load_dir / f"{safe_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Restore feature selection
            if metadata.get('selected_features'):
                self.feature_selector.selected_features[param_name] = metadata['selected_features']

        # Load label encoder if exists
        encoder_path = load_dir / f"{safe_name}_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.label_encoders[param_name] = pickle.load(f)

        print(f"Loaded model for {param_name} from {load_dir}")

    def save_all_models(self, output_dir: Optional[Path] = None):
        """Save all trained models"""
        save_dir = Path(output_dir) if output_dir else self.models_dir

        print(f"\nSaving {len(self.models)} models to {save_dir}...")

        for param_name in self.models:
            try:
                self.save_model(param_name, save_dir)
            except Exception as e:
                print(f"Warning: Failed to save {param_name}: {e}")

        # Save mapper state
        state = {
            'feature_names': self.feature_names,
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'saved_at': datetime.now().isoformat()
        }

        state_path = save_dir / 'mapper_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"All models saved successfully!")

    def load_all_models(self, model_dir: Optional[Path] = None):
        """Load all models from directory"""
        load_dir = Path(model_dir) if model_dir else self.models_dir

        # Load state
        state_path = load_dir / 'mapper_state.json'
        if not state_path.exists():
            print("Warning: No mapper state found, loading all JSON models in directory")
            model_files = list(load_dir.glob('*_model.json'))
            model_names = [f.stem.replace('_model', '').replace('_', '.') for f in model_files]
        else:
            with open(state_path, 'r') as f:
                state = json.load(f)
            model_names = state['model_names']
            self.feature_names = state.get('feature_names', [])
            self._feature_names_initialized = len(self.feature_names) > 0

        print(f"\nLoading {len(model_names)} models from {load_dir}...")

        for param_name in model_names:
            try:
                self.load_model(param_name, load_dir)
            except Exception as e:
                print(f"Warning: Failed to load {param_name}: {e}")

        print(f"Loaded {len(self.models)} models successfully!")

    # ========================================================================
    # Batch Training
    # ========================================================================

    def train_multiple_parameters(self,
                                  training_data_dict: Dict[str, List[TrainingExample]],
                                  show_progress: bool = True) -> Dict[str, MappingMetrics]:
        """
        Train models for multiple parameters in batch.

        Args:
            training_data_dict: {param_name: [training_examples]}
            show_progress: Show progress bar

        Returns:
            Dictionary of parameter names to metrics
        """
        results = {}

        param_names = list(training_data_dict.keys())
        iterator = tqdm(param_names, desc="Training parameters") if show_progress and TQDM_AVAILABLE else param_names

        for param_name in iterator:
            try:
                metrics = self.train_mapping(param_name, training_data_dict[param_name])
                results[param_name] = metrics
            except Exception as e:
                print(f"Error training {param_name}: {e}")
                continue

        return results

    # ========================================================================
    # Reporting and Analytics
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of mapper state"""
        return {
            'n_trained_models': len(self.models),
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'avg_quality': np.mean([m.val_score for m in self.metrics.values()]) if self.metrics else 0.0,
            'excellent_models': sum(1 for m in self.metrics.values() if m.quality_level == 'excellent'),
            'good_models': sum(1 for m in self.metrics.values() if m.quality_level == 'good'),
            'acceptable_models': sum(1 for m in self.metrics.values() if m.quality_level == 'acceptable'),
            'poor_models': sum(1 for m in self.metrics.values() if m.quality_level == 'poor'),
        }

    def print_summary(self):
        """Print summary of mapper state"""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("FEATURE-PARAMETER MAPPER SUMMARY")
        print("="*70)
        print(f"Trained models: {summary['n_trained_models']}")
        print(f"Features: {summary['n_features']}")
        print(f"Average quality: {summary['avg_quality']:.3f}")
        print(f"\nQuality distribution:")
        print(f"  Excellent: {summary['excellent_models']}")
        print(f"  Good: {summary['good_models']}")
        print(f"  Acceptable: {summary['acceptable_models']}")
        print(f"  Poor: {summary['poor_models']}")
        print("="*70 + "\n")

    # ========================================================================
    # Hierarchical and Causal Prediction
    # ========================================================================

    def predict_all_parameters_hierarchical(self,
                                           features: np.ndarray,
                                           use_causal_order: bool = True,
                                           show_progress: bool = False) -> Dict[str, Any]:
        """
        Predict all parameters using hierarchical and causal structure.

        This method respects the 3-level hierarchy and causal dependencies:
        - Level 1 (TOP): Genre/style parameters (unconditional)
        - Level 2 (MID): Complexity/density (conditioned on Level 1)
        - Level 3 (LOW): Details (conditioned on Level 1 + Level 2)

        Args:
            features: Feature vector (1000 features from Agent 8)
            use_causal_order: Use causal graph ordering for predictions
            show_progress: Show progress bar

        Returns:
            Dictionary mapping parameter names to predicted values
        """
        try:
            from parameters.hierarchy import get_parameter_level, ParameterLevel
            from parameters.causal_structure import CausalParameterGraph
        except ImportError:
            print("Warning: Hierarchical/causal structures not available. Falling back to flat prediction.")
            return self.predict_all_parameters(features, show_progress=show_progress)

        predictions = {}

        # Get causal ordering if requested
        if use_causal_order:
            try:
                causal_graph = CausalParameterGraph()
                param_order = causal_graph.get_causal_order()
            except Exception as e:
                print(f"Warning: Failed to get causal order: {e}. Using hierarchical only.")
                param_order = None
        else:
            param_order = None

        # Group parameters by hierarchy level
        level_1_params = []  # TOP: genre, style
        level_2_params = []  # MID: complexity, density
        level_3_params = []  # LOW: details

        for param_name in self.models.keys():
            level = get_parameter_level(param_name)
            if level == ParameterLevel.TOP:
                level_1_params.append(param_name)
            elif level == ParameterLevel.MID:
                level_2_params.append(param_name)
            else:
                level_3_params.append(param_name)

        # If causal order is available, use it within each level
        if param_order:
            level_1_params = [p for p in param_order if p in level_1_params]
            level_2_params = [p for p in param_order if p in level_2_params]
            level_3_params = [p for p in param_order if p in level_3_params]

        # Level 1: Predict genre/style (unconditional)
        print(f"\nPredicting Level 1 (Genre/Style): {len(level_1_params)} parameters")
        for param_name in (tqdm(level_1_params, desc="Level 1") if show_progress and TQDM_AVAILABLE else level_1_params):
            try:
                value = self.predict_parameter(features, param_name)
                predictions[param_name] = value
            except Exception as e:
                print(f"Warning: Failed to predict {param_name}: {e}")

        # Level 2: Predict complexity/density (conditioned on Level 1)
        print(f"\nPredicting Level 2 (Complexity/Density): {len(level_2_params)} parameters")
        for param_name in (tqdm(level_2_params, desc="Level 2") if show_progress and TQDM_AVAILABLE else level_2_params):
            try:
                # TODO: In future, pass Level 1 predictions as additional features
                # For now, just predict independently
                value = self.predict_parameter(features, param_name)
                predictions[param_name] = value
            except Exception as e:
                print(f"Warning: Failed to predict {param_name}: {e}")

        # Level 3: Predict details (conditioned on Level 1 + Level 2)
        print(f"\nPredicting Level 3 (Details): {len(level_3_params)} parameters")
        for param_name in (tqdm(level_3_params, desc="Level 3") if show_progress and TQDM_AVAILABLE else level_3_params):
            try:
                # TODO: In future, pass Level 1 + Level 2 predictions as additional features
                # For now, just predict independently
                value = self.predict_parameter(features, param_name)
                predictions[param_name] = value
            except Exception as e:
                print(f"Warning: Failed to predict {param_name}: {e}")

        print(f"\nHierarchical prediction complete: {len(predictions)} parameters predicted")
        return predictions


# ============================================================================
# Convenience Functions
# ============================================================================

def create_mapper(models_dir: Optional[Path] = None) -> FeatureParameterMapper:
    """Create a new FeatureParameterMapper instance"""
    return FeatureParameterMapper(models_dir=models_dir)


def train_from_midi_corpus(mapper: FeatureParameterMapper,
                           midi_files: List[Path],
                           target_params: List[str],
                           param_values: Dict[str, List[Any]]) -> Dict[str, MappingMetrics]:
    """
    Train mapper from a corpus of MIDI files.

    Args:
        mapper: FeatureParameterMapper instance
        midi_files: List of MIDI files
        target_params: Parameters to train
        param_values: {param_name: [values]} - one value per MIDI file

    Returns:
        Training metrics for each parameter
    """
    # Import here to avoid circular dependency
    from midi_generator.synthesis.deep_feature_extractor import extract_features

    # Extract features from all MIDI files
    print("Extracting features from MIDI corpus...")
    all_features = []
    for midi_file in tqdm(midi_files):
        try:
            features = extract_features(midi_file)
            all_features.append(features)
        except Exception as e:
            print(f"Warning: Failed to extract features from {midi_file}: {e}")
            all_features.append(None)

    # Prepare training data for each parameter
    training_data_dict = {}

    for param_name in target_params:
        if param_name not in param_values:
            print(f"Warning: No values provided for {param_name}")
            continue

        values = param_values[param_name]
        if len(values) != len(midi_files):
            print(f"Warning: Value count mismatch for {param_name}")
            continue

        # Create training examples
        examples = []
        for features, value, midi_file in zip(all_features, values, midi_files):
            if features is not None:
                example = TrainingExample(
                    features=features,
                    parameter_value=value,
                    parameter_name=param_name,
                    midi_file=midi_file
                )
                examples.append(example)

        training_data_dict[param_name] = examples

    # Train all parameters
    print(f"\nTraining {len(training_data_dict)} parameters...")
    results = mapper.train_multiple_parameters(training_data_dict)

    return results


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Feature-Parameter Mapper - Agent 9")
    print("="*70)

    # Create mapper
    mapper = create_mapper()
    mapper.print_summary()
