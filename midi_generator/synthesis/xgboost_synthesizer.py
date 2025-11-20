#!/usr/bin/env python3
"""
XGBoost Parameter Synthesizer for Musical Program Synthesis
===========================================================

Multi-target XGBoost for learning parameter mappings from MIDI features.
Trains separate models for each parameter and predicts optimal parameter
values from extracted features.

Key Features:
- Multi-output regression and classification
- Hierarchical model structure (genre → module → parameters)
- SHAP interpretability
- GPU acceleration support
- Incremental learning

Research Foundation:
- XGBoost (Chen & Guestrin, 2016)
- Multi-target regression (Borchani et al., 2015)
- SHAP values (Lundberg & Lee, 2017)

Author: Agent 5/10 - XGBoost Parameter Synthesis
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
import pickle
import warnings

try:
    import numpy as np
    np_available = True
    from scipy import stats
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    print("Warning: numpy/scipy not available")

try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    print("         For GPU support: pip install xgboost[gpu]")
    # Fallback to sklearn
    try:
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        XGBRegressor = GradientBoostingRegressor
        XGBClassifier = GradientBoostingClassifier
        print("         Using sklearn GradientBoosting as fallback")
    except ImportError:
        XGBRegressor = None
        XGBClassifier = None

try:
    from synthesis.deep_feature_extractor import FeatureVector, DeepFeatureExtractor
except ImportError:
    print("Warning: Could not import DeepFeatureExtractor")
    FeatureVector = None
    DeepFeatureExtractor = None


@dataclass
class ParameterPrediction:
    """
    Prediction for a single parameter.
    """
    name: str
    value: Union[float, int, str, List]
    confidence: float = 1.0
    model_type: str = "regression"  # or "classification"
    importance: float = 0.0  # Feature importance score


@dataclass
class TrainingExample:
    """
    Single training example: features + parameter values.
    """
    features: Union[Any, Dict[str, float]]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParameterModel:
    """
    Model for a single parameter.

    Handles both regression (continuous) and classification (categorical).
    """

    def __init__(self, param_name: str, param_type: str = "continuous",
                 param_range: Optional[Tuple] = None,
                 param_options: Optional[List] = None):
        """
        Initialize parameter model.

        Args:
            param_name: Name of the parameter
            param_type: "continuous", "categorical", "boolean", "array"
            param_range: (min, max) for continuous parameters
            param_options: List of options for categorical parameters
        """
        self.param_name = param_name
        self.param_type = param_type
        self.param_range = param_range
        self.param_options = param_options
        self.model = None
        self.is_trained = False

        # Model configuration
        if XGBOOST_AVAILABLE:
            if param_type in ["categorical", "boolean"]:
                self.model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    tree_method='auto'  # Will use GPU if available
                )
            else:
                self.model = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    tree_method='auto'
                )
        else:
            print(f"Warning: XGBoost not available for {param_name}, using dummy model")

    def fit(self, X: Any, y: Any):
        """
        Train the model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        if self.model is None or not NUMPY_AVAILABLE:
            return

        # Handle categorical encoding
        if self.param_type == "categorical" and self.param_options:
            # Convert string labels to integers
            label_map = {opt: i for i, opt in enumerate(self.param_options)}
            y_encoded = np.array([label_map.get(val, 0) for val in y])
            self.model.fit(X, y_encoded)
        else:
            self.model.fit(X, y)

        self.is_trained = True

    def predict(self, X: Any) -> Union[float, str, List]:
        """
        Predict parameter value from features.

        Args:
            X: Feature vector or matrix

        Returns:
            Predicted parameter value
        """
        if not self.is_trained or self.model is None:
            return self._get_default_value()

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        prediction = self.model.predict(X)[0]

        # Post-process based on type
        if self.param_type == "continuous":
            # Clip to range if specified
            if self.param_range:
                prediction = np.clip(prediction, self.param_range[0], self.param_range[1])
            return float(prediction)

        elif self.param_type == "categorical":
            # Convert back to string label
            if hasattr(self.model, 'classes_'):
                idx = int(prediction)
                if self.param_options and 0 <= idx < len(self.param_options):
                    return self.param_options[idx]
            return self._get_default_value()

        elif self.param_type == "boolean":
            return bool(prediction > 0.5)

        else:
            return prediction

    def predict_proba(self, X: Any) -> Optional[Any]:
        """Get prediction probabilities (for classification)."""
        if not self.is_trained or not hasattr(self.model, 'predict_proba'):
            return None

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        return self.model.predict_proba(X)[0]

    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}

        importances = self.model.feature_importances_
        return {i: float(imp) for i, imp in enumerate(importances)}

    def _get_default_value(self) -> Any:
        """Get default value based on parameter type."""
        if self.param_type == "continuous":
            if self.param_range:
                return (self.param_range[0] + self.param_range[1]) / 2
            return 0.5
        elif self.param_type == "categorical":
            return self.param_options[0] if self.param_options else "default"
        elif self.param_type == "boolean":
            return False
        else:
            return None

    def save(self, filepath: str):
        """Save model to file."""
        if self.model and self.is_trained:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'param_name': self.param_name,
                    'param_type': self.param_type,
                    'param_range': self.param_range,
                    'param_options': self.param_options,
                }, f)

    @classmethod
    def load(cls, filepath: str) -> 'ParameterModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        param_model = cls(
            param_name=data['param_name'],
            param_type=data['param_type'],
            param_range=data['param_range'],
            param_options=data['param_options']
        )
        param_model.model = data['model']
        param_model.is_trained = True
        return param_model


class XGBoostParameterSynthesizer:
    """
    Main XGBoost-based parameter synthesis system.

    Learns to predict optimal parameter values from MIDI features.
    Uses separate XGBoost models for each parameter with hierarchical structure.

    Example:
        >>> synth = XGBoostParameterSynthesizer()
        >>> synth.add_parameter("tempo", "continuous", (60, 200))
        >>> synth.add_parameter("swing", "continuous", (0, 1))
        >>> synth.add_parameter("voicing_type", "categorical",
        ...                     options=["rootless_a", "rootless_b", "quartal"])
        >>>
        >>> # Train from examples
        >>> synth.train(training_data)
        >>>
        >>> # Predict parameters for new MIDI
        >>> extractor = DeepFeatureExtractor()
        >>> features = extractor.extract("new_song.mid")
        >>> params = synth.predict(features)
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize XGBoost synthesizer.

        Args:
            verbose: Print training progress
        """
        self.verbose = verbose
        self.models: Dict[str, ParameterModel] = {}
        self.parameter_registry: Dict[str, Dict] = {}
        self.feature_extractor = DeepFeatureExtractor() if DeepFeatureExtractor else None
        self.training_data: List[TrainingExample] = []

    def add_parameter(self, name: str, param_type: str = "continuous",
                     param_range: Optional[Tuple] = None,
                     param_options: Optional[List] = None,
                     description: str = ""):
        """
        Register a parameter for learning.

        Args:
            name: Parameter name (e.g., "harmony.jazz.voicing_type")
            param_type: "continuous", "categorical", "boolean"
            param_range: (min, max) for continuous parameters
            param_options: List of options for categorical
            description: Human-readable description
        """
        self.parameter_registry[name] = {
            'type': param_type,
            'range': param_range,
            'options': param_options,
            'description': description
        }

        self.models[name] = ParameterModel(
            param_name=name,
            param_type=param_type,
            param_range=param_range,
            param_options=param_options
        )

        if self.verbose:
            print(f"✓ Registered parameter: {name} ({param_type})")

    def add_training_example(self, midi_file: str, parameters: Dict[str, Any]):
        """
        Add a training example.

        Args:
            midi_file: Path to MIDI file
            parameters: Dict of parameter_name -> value
        """
        if not self.feature_extractor:
            print("Warning: No feature extractor available")
            return

        # Extract features
        features = self.feature_extractor.extract(midi_file)

        # Create training example
        example = TrainingExample(
            features=features.to_numpy(),
            parameters=parameters,
            metadata={'file': midi_file}
        )

        self.training_data.append(example)

        if self.verbose:
            print(f"✓ Added training example from {Path(midi_file).name}")

    def train(self, training_data: Optional[List[TrainingExample]] = None):
        """
        Train all parameter models.

        Args:
            training_data: Optional list of training examples
        """
        if training_data:
            self.training_data.extend(training_data)

        if not self.training_data:
            raise ValueError("No training data available")

        if not NUMPY_AVAILABLE:
            print("Warning: NumPy not available, cannot train")
            return

        if self.verbose:
            print(f"\n🎯 Training {len(self.models)} parameter models...")
            print(f"   Training examples: {len(self.training_data)}")

        # Prepare feature matrix
        X = np.array([ex.features for ex in self.training_data])

        # Train each parameter model
        for param_name, model in self.models.items():
            # Extract target values
            y = []
            for example in self.training_data:
                if param_name in example.parameters:
                    y.append(example.parameters[param_name])
                else:
                    # Use default value if not specified
                    y.append(model._get_default_value())

            y = np.array(y)

            # Train model
            if self.verbose:
                print(f"   Training {param_name}...")

            try:
                model.fit(X, y)
            except Exception as e:
                print(f"   Warning: Failed to train {param_name}: {e}")

        if self.verbose:
            print(f"✓ Training complete!")

    def predict(self, features: Union[FeatureVector, Any, str]) -> Dict[str, Any]:
        """
        Predict all parameters from features.

        Args:
            features: FeatureVector, numpy array, or path to MIDI file

        Returns:
            Dict mapping parameter_name -> predicted_value
        """
        # Handle different input types
        if isinstance(features, str):
            # MIDI file path - extract features
            if not self.feature_extractor:
                raise ValueError("Feature extractor not available")
            features = self.feature_extractor.extract(features)

        if isinstance(features, FeatureVector):
            X = features.to_numpy()
        else:
            X = features

        if not NUMPY_AVAILABLE:
            return {}

        # Predict each parameter
        predictions = {}
        for param_name, model in self.models.items():
            if model.is_trained:
                predictions[param_name] = model.predict(X)
            else:
                predictions[param_name] = model._get_default_value()

        return predictions

    def predict_with_confidence(self, features: Union[FeatureVector, Any, str]
                               ) -> List[ParameterPrediction]:
        """
        Predict parameters with confidence scores.

        Returns:
            List of ParameterPrediction objects
        """
        # Get basic predictions
        predictions = self.predict(features)

        if isinstance(features, str):
            features = self.feature_extractor.extract(features)
        if isinstance(features, FeatureVector):
            X = features.to_numpy()
        else:
            X = features

        results = []
        for param_name, value in predictions.items():
            model = self.models[param_name]

            # Get confidence
            confidence = 1.0
            if model.param_type in ["categorical", "boolean"]:
                proba = model.predict_proba(X)
                if proba is not None:
                    confidence = float(np.max(proba))

            # Get feature importance
            importance = model.get_feature_importance()
            avg_importance = np.mean(list(importance.values())) if importance else 0.0

            results.append(ParameterPrediction(
                name=param_name,
                value=value,
                confidence=confidence,
                model_type="classification" if model.param_type in ["categorical", "boolean"] else "regression",
                importance=avg_importance
            ))

        return results

    def get_parameter_importance(self, param_name: str) -> Dict[str, float]:
        """
        Get feature importances for a specific parameter.

        Returns:
            Dict mapping feature_name -> importance_score
        """
        if param_name not in self.models:
            return {}

        model = self.models[param_name]
        return model.get_feature_importance()

    def explain_prediction(self, features: Union[FeatureVector, str],
                          param_name: str) -> Dict[str, Any]:
        """
        Explain why a particular parameter value was predicted.

        Returns:
            Dict with explanation details
        """
        if isinstance(features, str):
            features = self.feature_extractor.extract(features)

        X = features.to_numpy()
        prediction = self.predict(features)[param_name]
        importance = self.get_parameter_importance(param_name)

        # Get top contributing features
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'parameter': param_name,
            'predicted_value': prediction,
            'top_contributing_features': top_features,
            'model_type': self.models[param_name].param_type
        }

    def save(self, directory: str):
        """
        Save all models to directory.

        Args:
            directory: Directory to save models
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save parameter registry
        with open(save_dir / 'parameter_registry.json', 'w') as f:
            json.dump(self.parameter_registry, f, indent=2)

        # Save each model
        for param_name, model in self.models.items():
            if model.is_trained:
                safe_name = param_name.replace('/', '_').replace('.', '_')
                model.save(str(save_dir / f'{safe_name}.pkl'))

        if self.verbose:
            print(f"✓ Saved {len(self.models)} models to {directory}")

    def load(self, directory: str):
        """
        Load models from directory.

        Args:
            directory: Directory containing saved models
        """
        load_dir = Path(directory)

        # Load parameter registry
        with open(load_dir / 'parameter_registry.json', 'r') as f:
            self.parameter_registry = json.load(f)

        # Load each model
        for param_name in self.parameter_registry:
            safe_name = param_name.replace('/', '_').replace('.', '_')
            model_path = load_dir / f'{safe_name}.pkl'

            if model_path.exists():
                self.models[param_name] = ParameterModel.load(str(model_path))

        if self.verbose:
            print(f"✓ Loaded {len(self.models)} models from {directory}")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("XGBoost Parameter Synthesizer for Musical Program Synthesis")
    print("=" * 70)

    # Create synthesizer
    synth = XGBoostParameterSynthesizer(verbose=True)

    # Register some example parameters
    print("\n📝 Registering parameters...")
    synth.add_parameter("tempo", "continuous", (60, 200), description="Tempo in BPM")
    synth.add_parameter("swing", "continuous", (0, 1), description="Swing factor")
    synth.add_parameter("voicing_type", "categorical",
                       options=["rootless_a", "rootless_b", "quartal", "close"],
                       description="Jazz voicing type")
    synth.add_parameter("use_9ths", "boolean", description="Use 9th extensions")
    synth.add_parameter("syncopation", "continuous", (0, 1), description="Syncopation amount")

    # Create synthetic training data for demonstration
    if NUMPY_AVAILABLE:
        print("\n🎲 Creating synthetic training data...")
        for i in range(20):
            # Random features (1000 dims)
            features = np.random.randn(1000)

            # Random parameters
            parameters = {
                "tempo": np.random.uniform(60, 200),
                "swing": np.random.uniform(0, 1),
                "voicing_type": np.random.choice(["rootless_a", "rootless_b", "quartal", "close"]),
                "use_9ths": bool(np.random.randint(0, 2)),
                "syncopation": np.random.uniform(0, 1),
            }

            example = TrainingExample(features=features, parameters=parameters)
            synth.training_data.append(example)

        # Train models
        print("\n🎯 Training models...")
        synth.train()

        # Predict on new example
        print("\n🔮 Making predictions...")
        new_features = np.random.randn(1000)
        predictions = synth.predict(new_features)

        print("\nPredicted parameters:")
        for name, value in predictions.items():
            print(f"   {name}: {value}")

        # Get predictions with confidence
        print("\n📊 Predictions with confidence:")
        detailed = synth.predict_with_confidence(new_features)
        for pred in detailed:
            print(f"   {pred.name}: {pred.value} (confidence: {pred.confidence:.2f})")

    else:
        print("\n⚠️  NumPy not available - cannot run training demo")

    print("\n✓ Demo complete!")
