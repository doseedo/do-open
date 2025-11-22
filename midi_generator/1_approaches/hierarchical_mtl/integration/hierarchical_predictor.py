"""
AGENT 05: Hierarchical MTL Inference Pipeline
==============================================

Production-ready inference pipeline for parameter prediction from MIDI files.

Features:
    - Load trained model
    - Predict parameters from MIDI files
    - Batch prediction
    - Caching for performance
    - Export predictions to JSON/YAML
    - Integration with HarmonyModule API

Author: Agent 05 - Hierarchical MTL Architect
License: MIT
Date: November 20, 2025
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("WARNING: NumPy not installed")

# Import our modules
try:
    from midi_generator.learning.hierarchical_mtl import (
        HierarchicalMTLModel,
        MTLConfig,
        create_model,
        ALL_PARAMETERS,
    )
except ImportError:
    print("WARNING: Could not import hierarchical_mtl module")


# ============================================================================
# Predictor
# ============================================================================

class HierarchicalParameterPredictor:
    """
    Production inference pipeline for hierarchical parameter prediction.

    Usage:
        >>> predictor = HierarchicalParameterPredictor.from_checkpoint('model.pt')
        >>> features = extract_features('song.mid')  # 200-dimensional vector
        >>> params = predictor.predict(features)
        >>> print(params['genre.primary'])  # 'jazz'
        >>> print(params['tempo.bpm'])      # 180.5
    """

    def __init__(self,
                 model: HierarchicalMTLModel,
                 device: str = 'cpu',
                 use_cache: bool = True):
        """
        Initialize predictor.

        Args:
            model: Trained Hierarchical MTL model
            device: Device to run inference on
            use_cache: Enable prediction caching
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for inference")

        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.use_cache = use_cache
        self._cache = {} if use_cache else None

        # Parameter definitions
        self.param_defs = {p.name: p for p in ALL_PARAMETERS}

    @classmethod
    def from_checkpoint(cls,
                       checkpoint_path: Union[str, Path],
                       device: str = 'auto',
                       config: Optional[MTLConfig] = None) -> 'HierarchicalParameterPredictor':
        """
        Load predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
            config: Model configuration (loads from checkpoint if None)

        Returns:
            Initialized predictor
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model
        if config is None:
            # Try to load config from checkpoint
            if 'config' in checkpoint:
                config = MTLConfig(**checkpoint['config'].get('model_config', {}))
            else:
                config = MTLConfig()  # Use default

        model = create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Loaded model from {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")

        return cls(model, device=device)

    def predict(self,
               features: Union[np.ndarray, torch.Tensor],
               return_raw: bool = False) -> Dict[str, Any]:
        """
        Predict all 50 hierarchical parameters from features.

        Args:
            features: Feature vector (200 dimensions) or batch (N, 200)
            return_raw: Return raw model outputs instead of interpreted values

        Returns:
            Dictionary of parameter predictions
        """
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()

        # Move to device
        features = features.to(self.device)

        # Handle single sample
        single_sample = False
        if features.dim() == 1:
            features = features.unsqueeze(0)
            single_sample = True

        # Check cache (for single samples)
        if single_sample and self.use_cache:
            cache_key = hash(features.cpu().numpy().tobytes())
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Predict
        with torch.no_grad():
            predictions = self.model.predict(features)

        # Extract single sample if needed
        if single_sample:
            # Cache result
            if self.use_cache:
                self._cache[cache_key] = predictions

        return predictions

    def predict_batch(self,
                     features_batch: Union[np.ndarray, torch.Tensor],
                     batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Predict parameters for multiple feature vectors.

        Args:
            features_batch: (N, 200) feature matrix
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        # Convert to tensor
        if isinstance(features_batch, np.ndarray):
            features_batch = torch.from_numpy(features_batch).float()

        n_samples = features_batch.size(0)
        all_predictions = []

        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch = features_batch[i:i+batch_size]
            batch_preds = self.predict(batch)

            # If batch output, split into individual predictions
            if isinstance(batch_preds, dict) and any(isinstance(v, (list, torch.Tensor)) for v in batch_preds.values()):
                # Split batch predictions
                batch_size_actual = batch.size(0)
                for j in range(batch_size_actual):
                    sample_pred = {k: v[j] if isinstance(v, (list, torch.Tensor)) else v
                                  for k, v in batch_preds.items()}
                    all_predictions.append(sample_pred)
            else:
                all_predictions.append(batch_preds)

        return all_predictions

    def predict_from_midi(self,
                         midi_path: Union[str, Path],
                         feature_extractor: Optional[Any] = None) -> Dict[str, Any]:
        """
        Predict parameters directly from MIDI file.

        Args:
            midi_path: Path to MIDI file
            feature_extractor: Feature extractor instance (uses default if None)

        Returns:
            Parameter predictions
        """
        # Import feature extractor
        if feature_extractor is None:
            try:
                from midi_generator.learning.feature_extractor import extract_features
                feature_extractor = extract_features
            except ImportError:
                raise ImportError("Feature extractor not available. "
                                "Please provide a feature_extractor function.")

        # Extract features
        features = feature_extractor(midi_path)

        # Ensure 200 dimensions
        if len(features) != 200:
            raise ValueError(f"Expected 200 features, got {len(features)}")

        # Predict
        return self.predict(features)

    def predict_level1_only(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Predict only Level 1 (global context) parameters.

        Useful for quick genre/style detection.

        Args:
            features: Feature vector (200 dimensions)

        Returns:
            Level 1 parameter predictions only
        """
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()

        features = features.to(self.device)

        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            # Get encoded features
            encoded = self.model.encoder(features)

            # Predict Level 1 only
            predictions = {}
            for param_name, head in self.model.level1_heads.items():
                output = head(encoded)

                # Convert to interpretable value
                param_def = self.model.level1_params[param_name]
                if param_def.param_type == 'categorical':
                    pred_idx = torch.argmax(output, dim=-1).item()
                    predictions[param_name] = param_def.categories[pred_idx]
                elif param_def.param_type == 'integer':
                    predictions[param_name] = int(torch.round(output).item())
                else:
                    predictions[param_name] = float(output.item())

        return predictions

    def explain_prediction(self,
                          features: Union[np.ndarray, torch.Tensor],
                          top_k: int = 10) -> Dict[str, Any]:
        """
        Explain prediction with feature attributions (requires additional setup).

        Args:
            features: Feature vector
            top_k: Number of top features to return

        Returns:
            Predictions with feature attributions
        """
        # This is a placeholder for future feature attribution methods
        # Could use integrated gradients, SHAP, etc.
        predictions = self.predict(features)

        return {
            'predictions': predictions,
            'explanation': 'Feature attribution not yet implemented'
        }

    def save_predictions(self,
                        predictions: Dict[str, Any],
                        output_path: Union[str, Path],
                        format: str = 'json'):
        """
        Save predictions to file.

        Args:
            predictions: Prediction dictionary
            output_path: Output file path
            format: Output format ('json' or 'yaml')
        """
        output_path = Path(output_path)

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
        elif format == 'yaml':
            try:
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(predictions, f, default_flow_style=False)
            except ImportError:
                print("WARNING: PyYAML not installed. Saving as JSON instead.")
                with open(output_path, 'w') as f:
                    json.dump(predictions, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Saved predictions to {output_path}")

    def get_genre_specific_params(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the genre-specific parameters relevant to predicted genre.

        Args:
            predictions: Full predictions dictionary

        Returns:
            Filtered predictions with only relevant genre-specific params
        """
        genre = predictions.get('genre.primary')

        if genre is None:
            return {}

        relevant_params = {}

        for param_name, value in predictions.items():
            param_def = self.param_defs.get(param_name)

            if param_def and param_def.genre_specific:
                # Check if this parameter is relevant to the predicted genre
                if param_def.genres and genre in param_def.genres:
                    relevant_params[param_name] = value

        return relevant_params

    def clear_cache(self):
        """Clear prediction cache"""
        if self._cache is not None:
            self._cache.clear()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            'total_parameters': total_params,
            'device': str(self.device),
            'input_dim': self.model.input_dim,
            'encoder_output_dim': self.model.encoder.output_dim,
            'num_level1_params': len(self.model.level1_heads),
            'num_level2_params': len(self.model.level2_heads),
            'num_level3_params': len(self.model.level3_heads),
            'cache_enabled': self.use_cache,
        }


# ============================================================================
# Batch Processing Utilities
# ============================================================================

class BatchPredictor:
    """
    Utility for batch processing multiple MIDI files.
    """

    def __init__(self, predictor: HierarchicalParameterPredictor):
        self.predictor = predictor

    def process_directory(self,
                         midi_dir: Path,
                         output_dir: Path,
                         feature_extractor: Optional[Any] = None,
                         pattern: str = '*.mid') -> List[Dict[str, Any]]:
        """
        Process all MIDI files in a directory.

        Args:
            midi_dir: Directory containing MIDI files
            output_dir: Directory to save predictions
            feature_extractor: Feature extraction function
            pattern: Glob pattern for MIDI files

        Returns:
            List of prediction results
        """
        midi_dir = Path(midi_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        midi_files = list(midi_dir.glob(pattern))

        print(f"Processing {len(midi_files)} MIDI files...")

        results = []

        for midi_file in midi_files:
            try:
                # Predict
                predictions = self.predictor.predict_from_midi(midi_file, feature_extractor)

                # Save predictions
                output_file = output_dir / f"{midi_file.stem}_predictions.json"
                self.predictor.save_predictions(predictions, output_file)

                results.append({
                    'file': str(midi_file),
                    'status': 'success',
                    'predictions': predictions
                })

            except Exception as e:
                print(f"ERROR processing {midi_file}: {e}")
                results.append({
                    'file': str(midi_file),
                    'status': 'error',
                    'error': str(e)
                })

        # Save summary
        summary_file = output_dir / 'batch_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'total_files': len(midi_files),
                'successful': sum(1 for r in results if r['status'] == 'success'),
                'failed': sum(1 for r in results if r['status'] == 'error'),
                'results': results
            }, f, indent=2)

        print(f"\nBatch processing complete! Summary saved to {summary_file}")

        return results


# ============================================================================
# Integration with HarmonyModule API
# ============================================================================

class HarmonyModuleIntegration:
    """
    Integration layer for using hierarchical MTL predictions with HarmonyModule.
    """

    def __init__(self, predictor: HierarchicalParameterPredictor):
        self.predictor = predictor

    def predict_and_generate(self,
                            midi_path: Path,
                            harmony_module_api: Any,
                            feature_extractor: Optional[Any] = None) -> Any:
        """
        Predict parameters from MIDI and generate new music with HarmonyModule.

        Args:
            midi_path: Input MIDI file
            harmony_module_api: HarmonyModule API instance
            feature_extractor: Feature extractor

        Returns:
            Generated MIDI
        """
        # Predict parameters
        predictions = self.predictor.predict_from_midi(midi_path, feature_extractor)

        # Convert predictions to HarmonyModule format
        harmony_params = self._convert_to_harmony_params(predictions)

        # Generate music
        generated_midi = harmony_module_api.generate(**harmony_params)

        return generated_midi

    def _convert_to_harmony_params(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert hierarchical predictions to HarmonyModule parameter format.

        Args:
            predictions: Hierarchical parameter predictions

        Returns:
            Parameters in HarmonyModule format
        """
        # This is a mapping function that converts our 50 hierarchical params
        # to the format expected by HarmonyModule API

        harmony_params = {}

        # Map Level 1 parameters
        if 'genre.primary' in predictions:
            harmony_params['genre'] = predictions['genre.primary']
        if 'tempo.bpm' in predictions:
            harmony_params['tempo'] = predictions['tempo.bpm']
        if 'key.tonic' in predictions:
            harmony_params['key'] = predictions['key.tonic']
        if 'key.mode' in predictions:
            harmony_params['mode'] = predictions['key.mode']

        # Map Level 2 harmony parameters
        if 'harmony.chord_density' in predictions:
            harmony_params['chord_density'] = predictions['harmony.chord_density']
        if 'harmony.complexity' in predictions:
            harmony_params['harmony_complexity'] = predictions['harmony.complexity']

        # Map Level 2 melody parameters
        if 'melody.note_density' in predictions:
            harmony_params['melody_density'] = predictions['melody.note_density']
        if 'melody.range_semitones' in predictions:
            harmony_params['melody_range'] = predictions['melody.range_semitones']

        # Map Level 2 rhythm parameters
        if 'rhythm.syncopation' in predictions:
            harmony_params['syncopation'] = predictions['rhythm.syncopation']
        if 'rhythm.swing_amount' in predictions:
            harmony_params['swing'] = predictions['rhythm.swing_amount']

        # Map dynamics
        if 'dynamics.overall_level' in predictions:
            harmony_params['dynamics'] = predictions['dynamics.overall_level']

        # Add more mappings as needed...

        return harmony_params


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_predict(checkpoint_path: Union[str, Path],
                 features: np.ndarray,
                 device: str = 'auto') -> Dict[str, Any]:
    """
    Quick one-line prediction.

    Args:
        checkpoint_path: Path to model checkpoint
        features: Feature vector (200 dims)
        device: Device to use

    Returns:
        Predictions
    """
    predictor = HierarchicalParameterPredictor.from_checkpoint(checkpoint_path, device)
    return predictor.predict(features)


def predict_from_midi(checkpoint_path: Union[str, Path],
                     midi_path: Union[str, Path],
                     device: str = 'auto') -> Dict[str, Any]:
    """
    Predict parameters directly from MIDI file.

    Args:
        checkpoint_path: Path to model checkpoint
        midi_path: Path to MIDI file
        device: Device to use

    Returns:
        Predictions
    """
    predictor = HierarchicalParameterPredictor.from_checkpoint(checkpoint_path, device)
    return predictor.predict_from_midi(midi_path)


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available")
        exit(1)

    print("Hierarchical MTL Predictor - Agent 05")
    print("="*70)

    # Example usage
    checkpoint_path = Path("midi_generator/models/hierarchical_mtl/checkpoints/best_model.pt")

    if checkpoint_path.exists():
        print(f"\nLoading model from: {checkpoint_path}")
        predictor = HierarchicalParameterPredictor.from_checkpoint(checkpoint_path)

        print("\nModel info:")
        info = predictor.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\nPredictor ready for inference!")
    else:
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please train the model first using hierarchical_trainer.py")

    print("="*70)
