# Agent 7: Integration Pipeline - Integration Guide

**Author**: Agent 7 - Integration Pipeline
**Date**: November 21, 2025
**Version**: 1.0.0

---

## Overview

This document provides detailed guidance for Agents 1-6 on how to integrate their components with the **SemanticDiscoveryPipeline**.

The pipeline coordinates all components to discover semantic musical parameters from MIDI corpora through a 6-stage process.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SEMANTIC DISCOVERY PIPELINE               │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐        ┌─────────┐
   │ Stage 1 │         │ Stage 2 │        │ Stage 3 │
   │  Gaps   │────────>│Training │───────>│Interpret│
   └─────────┘         └─────────┘        └─────────┘
        │                                       │
        │                                       ▼
        │                                  ┌─────────┐
        │                                  │ Stage 4 │
        │                                  │Validate │
        │                                  └─────────┘
        │                                       │
        │                                       ▼
        │                                  ┌─────────┐
        │                                  │ Stage 5 │
        │                                  │Register │
        │                                  └─────────┘
        │                                       │
        └───────────────────────────────────────┼──>
                                                ▼
                                           ┌─────────┐
                                           │ Stage 6 │
                                           │Evaluate │
                                           └─────────┘
```

---

## For Agent 1: Musical Locality Functions

**File**: `midi_generator/learning/musical_locality.py`

### Interface to Implement

```python
class MusicalLocalityFunctionsInterface:
    """Agent 1 must implement this interface"""

    def apply_transformation(self, midi_data: np.ndarray, transform_type: str) -> np.ndarray:
        """
        Apply musical transformation to MIDI representation.

        Args:
            midi_data: MIDI representation (200D or 50D features)
            transform_type: One of: 'transpose', 'invert', 'augment', 'retrograde', etc.

        Returns:
            Transformed MIDI representation
        """
        # Your implementation here

    def get_available_transformations(self) -> List[str]:
        """Get list of available transformation types"""
        return ['transpose', 'invert', 'augment', 'retrograde', 'time_shift', ...]
```

### How Pipeline Uses This

**Stage 2: Feature Training**
- Pipeline imports your module: `from midi_generator.learning.musical_locality import MusicalLocalityFunctions`
- Creates instances for locality constraints during training
- Applies transformations to validate that semantic features are locality-invariant

### Integration Points

1. **Import Path**: `midi_generator.learning.musical_locality`
2. **Class Name**: `MusicalLocalityFunctions`
3. **Used By**: Agent 5 (Trainer), Agent 3 (Encoder)

### Testing Your Integration

```python
from midi_generator.learning.musical_locality import MusicalLocalityFunctions
from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline

# Create instance
locality = MusicalLocalityFunctions()

# Test transformations
sample_features = np.random.randn(200)
transposed = locality.apply_transformation(sample_features, 'transpose')

assert transposed.shape == sample_features.shape
```

---

## For Agent 2: Semantic Feature Representations

**File**: `midi_generator/learning/semantic_features.py`

### Classes to Implement

```python
@dataclass
class SemanticFeature:
    """Representation of a semantic feature"""
    feature_id: int
    name: str
    modality: FeatureModality  # harmony|melody|rhythm|dynamics|texture
    activation_pattern: np.ndarray

    # Methods
    def generate_variants(self, locality_functions) -> List['SemanticFeature']:
        """Generate locality variants"""

    def matches_pattern(self, midi_data: np.ndarray) -> float:
        """Compute activation strength"""

class SemanticFeatureBank:
    """Collection of semantic features"""
    def __init__(self):
        self.features: Dict[int, SemanticFeature] = {}

    def get_activations(self, midi_data: np.ndarray) -> np.ndarray:
        """Get activation vector for MIDI data"""

    def save(self, path: str):
        """Save to disk"""

    @classmethod
    def load(cls, path: str) -> 'SemanticFeatureBank':
        """Load from disk"""
```

### How Pipeline Uses This

**Stage 3: Feature Interpretation**
- Pipeline creates `SemanticFeatureBank` from trained encoder
- Stores interpretations and metadata
- Saves bank for later use

**Stage 5: Parameter Registration**
- Loads feature bank to register parameters

### Integration Points

1. **Import Path**: `midi_generator.learning.semantic_features`
2. **Classes**: `SemanticFeature`, `SemanticFeatureBank`
3. **Used By**: Agent 3 (Encoder), Agent 6 (Interpreter)

### Example Usage

```python
from midi_generator.learning.semantic_features import SemanticFeatureBank

# Load discovered features
bank = SemanticFeatureBank.load("output/discovery/feature_bank.pkl")

# Get activations for new MIDI
features_200d = extractor.extract("song.mid")
activations = bank.get_activations(features_200d)

# activations is now a vector of semantic feature strengths
```

---

## For Agent 3: Neural Architecture (Encoder)

**File**: `midi_generator/learning/semantic_encoder.py`

### Interface to Implement

```python
class SemanticFeatureEncoder(nn.Module):
    """Neural encoder for semantic features"""

    def __init__(self, input_dim: int, num_features: int, hidden_dim: int):
        """
        Args:
            input_dim: 200 (from OptimizedFeatureExtractor)
            num_features: 20-30 (target semantic features)
            hidden_dim: 512 (hidden layer size)
        """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input features [batch, 200]

        Returns:
            semantic_features: [batch, num_features]
            reconstructed_200d: [batch, 200]
            locality_predictions: [batch, num_transformations]
        """

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute training loss

        Returns:
            {
                'reconstruction_loss': float,
                'sparsity_loss': float,
                'locality_loss': float,
                'total_loss': float
            }
        """

    def extract_semantic_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract semantic features (encoder only, no decoder)"""
```

### How Pipeline Uses This

**Stage 2: Feature Training**
- Pipeline creates encoder instance with config parameters
- Passes to Agent 5 (Trainer)
- After training, encoder is used for interpretation

**Stage 3: Feature Interpretation**
- Pipeline uses trained encoder to analyze learned features
- Extracts semantic features for test data

### Integration Points

1. **Import Path**: `midi_generator.learning.semantic_encoder`
2. **Class**: `SemanticFeatureEncoder`
3. **Used By**: Agent 5 (Trainer), Agent 6 (Interpreter), Pipeline

### Configuration from Pipeline

```python
# Pipeline creates encoder with these settings from PipelineConfig:
encoder = SemanticFeatureEncoder(
    input_dim=200,
    num_features=config.num_semantic_features,  # 20-30
    hidden_dim=config.hidden_dim,                # 512
)
```

---

## For Agent 4: Gap Dataset Creation

**File**: `midi_generator/learning/gap_dataset.py`

### Interface to Implement

```python
class GapDataset(torch.utils.data.Dataset):
    """PyTorch dataset for gap-based training"""

    def __init__(
        self,
        midi_files: List[Path],
        feature_extractor_200d: OptimizedFeatureExtractor,
        parameter_extractor_50d: HierarchicalParameterExtractor,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize dataset

        Args:
            midi_files: List of MIDI file paths
            feature_extractor_200d: 200D feature extractor
            parameter_extractor_50d: 50D parameter extractor
            cache_dir: Cache directory for precomputed gaps
        """

    def __len__(self) -> int:
        """Number of samples"""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get dataset sample

        Returns:
            features_200d: [200] tensor
            parameters_50d: [50] tensor
            gap_vector: [200] tensor (what 50D params can't reconstruct)
        """
```

### How Pipeline Uses This

**Stage 1: Gap Computation**
- Pipeline finds MIDI files in corpus
- Passes files + extractors to create dataset
- Dataset computes and caches gaps

**Stage 2: Feature Training**
- Dataset is passed to Agent 5 (Trainer)
- Trainer uses dataset for training loop

### Integration Points

1. **Import Path**: `midi_generator.learning.gap_dataset`
2. **Class**: `GapDataset`
3. **Used By**: Pipeline (Stage 1), Agent 5 (Trainer)

### Gap Computation Logic

The dataset should:

1. Extract 200D features from MIDI
2. Extract 50D parameters from MIDI
3. **Compute gap**: What can't be reconstructed from 50D → 200D?
   - Train a baseline model: 50D → 200D
   - Gap = Actual 200D - Predicted 200D
4. Return (200D, 50D, gap) tuples

---

## For Agent 5: Training Infrastructure

**File**: `midi_generator/learning/gap_discovery_trainer.py`

### Interface to Implement

```python
class GapDiscoveryTrainer:
    """Trainer for semantic feature discovery"""

    def __init__(
        self,
        encoder: SemanticFeatureEncoder,
        dataset: GapDataset,
        config: PipelineConfig
    ):
        """
        Initialize trainer

        Args:
            encoder: Semantic feature encoder (Agent 3)
            dataset: Gap dataset (Agent 4)
            config: Pipeline configuration
        """

    def train(self) -> Dict[str, Any]:
        """
        Run complete training loop

        Returns:
            {
                'final_loss': float,
                'training_losses': List[float],
                'validation_losses': List[float],
                'best_checkpoint': str,
                'epochs_trained': int
            }
        """

    def train_epoch(self) -> float:
        """Train one epoch, return loss"""

    def validate(self) -> float:
        """Run validation, return loss"""

    def get_trained_model(self) -> SemanticFeatureEncoder:
        """Get trained encoder"""
```

### How Pipeline Uses This

**Stage 2: Feature Training**
- Pipeline creates trainer with encoder, dataset, config
- Calls `trainer.train()` to run full training
- Retrieves trained model for interpretation

### Integration Points

1. **Import Path**: `midi_generator.learning.gap_discovery_trainer`
2. **Class**: `GapDiscoveryTrainer`
3. **Used By**: Pipeline (Stage 2)

### Configuration from Pipeline

```python
# Training settings from config:
- config.max_epochs
- config.batch_size
- config.learning_rate
- config.early_stopping_patience
- config.sparsity_weight
- config.locality_weight
- config.checkpoint_frequency
- config.device ('cuda' or 'cpu')
```

### Expected Training Flow

1. Load dataset (from Agent 4)
2. Create dataloaders (train/val/test)
3. Initialize optimizer
4. Training loop:
   - For each epoch:
     - Train on batches
     - Compute loss (reconstruction + sparsity + locality)
     - Update weights
     - Validate
     - Save checkpoint if best
     - Early stopping if no improvement
5. Return trained model + metrics

---

## For Agent 6: Feature Interpretation

**File**: `midi_generator/learning/feature_interpreter.py`

### Interface to Implement

```python
class FeatureInterpreter:
    """Interprets learned semantic features as musical parameters"""

    def __init__(
        self,
        threshold: float = 0.6,
        concept_matching_threshold: float = 0.7
    ):
        """
        Initialize interpreter

        Args:
            threshold: Minimum confidence for interpretation
            concept_matching_threshold: Threshold for concept matching
        """

    def interpret_feature(
        self,
        feature_idx: int,
        encoder: SemanticFeatureEncoder
    ) -> Dict[str, Any]:
        """
        Interpret a single semantic feature

        Args:
            feature_idx: Index of feature to interpret
            encoder: Trained encoder

        Returns:
            {
                'name': str,  # e.g., "harmonic_tension_level"
                'modality': str,  # 'harmony'|'melody'|'rhythm'|'dynamics'|'texture'
                'confidence': float,  # 0.0-1.0
                'concept_match': str,  # Matched musical concept
                'extraction_function': callable,  # Function to extract from MIDI
                'description': str,  # Human-readable description
            }
        """

    def interpret_all_features(
        self,
        encoder: SemanticFeatureEncoder
    ) -> Dict[int, Dict[str, Any]]:
        """
        Interpret all semantic features

        Returns:
            Dictionary mapping feature_idx -> interpretation
        """
```

### How Pipeline Uses This

**Stage 3: Feature Interpretation**
- Pipeline creates interpreter with config thresholds
- Calls `interpret_all_features(encoder)` on trained encoder
- Stores interpretations for validation and registration

**Stage 5: Parameter Registration**
- Uses extraction functions from interpretations
- Registers parameters in UniversalParameterRegistry

### Integration Points

1. **Import Path**: `midi_generator.learning.feature_interpreter`
2. **Class**: `FeatureInterpreter`
3. **Used By**: Pipeline (Stage 3, Stage 5)

### Interpretation Process

Your interpreter should:

1. **Analyze feature activations** across test corpus
2. **Test with musical patterns** (scales, arpeggios, chord progressions, etc.)
3. **Classify modality** (harmony vs melody vs rhythm, etc.)
4. **Match to concepts** (e.g., "this activates strongly for swing rhythms")
5. **Generate extraction function** that can extract this feature from new MIDI
6. **Name the parameter** based on what it represents

### Example Interpretation

```python
# Example output for feature #5:
{
    'name': 'swing_intensity',
    'modality': 'rhythm',
    'confidence': 0.85,
    'concept_match': 'swing_feel',
    'extraction_function': lambda midi_path: extract_swing(midi_path),
    'description': 'Measures the intensity of swing feel (0.0=straight, 1.0=hard swing)'
}
```

---

## For Agent 8: Constraint Validation

**File**: `midi_generator/learning/semantic_constraints.py`

### Interface to Implement

```python
class SemanticFeatureValidator:
    """Validates semantic features for musical validity"""

    def __init__(
        self,
        redundancy_threshold: float = 0.9,
        strict_musical_validity: bool = True
    ):
        """
        Initialize validator

        Args:
            redundancy_threshold: Correlation threshold for redundancy
            strict_musical_validity: Whether to enforce strict musical rules
        """

    def validate_feature(
        self,
        feature_interpretation: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate a semantic feature

        Args:
            feature_interpretation: Output from FeatureInterpreter

        Returns:
            (is_valid, reason)

        Checks:
        - Musical validity (does it make musical sense?)
        - Locality consistency (invariant to musically-equivalent transformations?)
        - Redundancy (correlation with existing parameters < threshold?)
        """
```

### How Pipeline Uses This

**Stage 4: Feature Validation**
- Pipeline creates validator with config thresholds
- Validates each interpreted feature
- Filters out invalid features before registration

### Integration Points

1. **Import Path**: `midi_generator.learning.semantic_constraints`
2. **Class**: `SemanticFeatureValidator`
3. **Used By**: Pipeline (Stage 4)

### Validation Checks

1. **Musical Validity**
   - Does the feature represent a valid musical concept?
   - Is the range appropriate?
   - Are constraints satisfied?

2. **Locality Consistency**
   - Is feature invariant to transpose?
   - Is it invariant to time shift?
   - Does it respect musical equivalences?

3. **Redundancy**
   - Compute correlation with existing parameters
   - Reject if too similar to existing

---

## For Agent 9: Evaluation & Metrics

**File**: `midi_generator/evaluation/semantic_evaluation.py`

### Interface to Implement

```python
class SemanticFeatureEvaluator:
    """Evaluates discovered semantic features"""

    def evaluate_reconstruction(
        self,
        discovered_features: SemanticFeatureBank,
        test_corpus: List[Path]
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction improvement

        Returns:
            {
                'baseline_error': float,  # Error with only 50D params
                'with_semantic_error': float,  # Error with 50D + semantic
                'improvement': float,  # Percentage improvement
                'coverage': float  # Fraction of gap explained
            }
        """

    def evaluate_interpretability(
        self,
        interpretations: Dict[int, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate interpretability

        Returns:
            {
                'interpreted_fraction': float,  # Fraction successfully interpreted
                'average_confidence': float,
                'modality_distribution': Dict[str, int]
            }
        """
```

### How Pipeline Uses This

**Stage 6: Evaluation**
- Pipeline creates evaluator
- Runs all evaluation metrics
- Compiles comprehensive report

### Integration Points

1. **Import Path**: `midi_generator.evaluation.semantic_evaluation`
2. **Class**: `SemanticFeatureEvaluator`
3. **Used By**: Pipeline (Stage 6)

---

## Pipeline Configuration Reference

Here are all the configuration options agents might need:

```python
@dataclass
class PipelineConfig:
    # Paths
    midi_corpus_dir: Path
    output_dir: Path
    cache_dir: Optional[Path] = None

    # Corpus
    max_files: Optional[int] = None
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Features
    use_200d_features: bool = True
    use_50d_parameters: bool = True

    # Neural training
    num_semantic_features: int = 30
    hidden_dim: int = 512
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Sparsity
    sparsity_weight: float = 0.01
    target_sparsity: float = 0.1

    # Locality
    locality_weight: float = 0.1
    locality_transformations: List[str] = ['transpose', 'invert', ...]

    # Interpretation
    interpretation_threshold: float = 0.6
    concept_matching_threshold: float = 0.7

    # Validation
    redundancy_threshold: float = 0.9
    musical_validity_strict: bool = True

    # Computational
    device: str = "cuda"
    num_workers: int = 4
    use_mixed_precision: bool = True

    # Checkpointing
    checkpoint_frequency: int = 5
    resume_from_checkpoint: Optional[str] = None

    # Logging
    verbose: bool = True
    log_frequency: int = 10
```

---

## Testing Your Integration

### Unit Tests

Create tests in `midi_generator/tests/test_<your_module>.py`:

```python
import unittest
from midi_generator.learning.your_module import YourClass

class TestYourIntegration(unittest.TestCase):
    def test_basic_functionality(self):
        # Test your component works standalone
        pass

    def test_pipeline_integration(self):
        # Test integration with pipeline
        from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline
        # Test your component works in pipeline
        pass
```

### Integration Test

Run the full pipeline with your component:

```bash
python examples/run_semantic_discovery.py \
    --corpus data/midi \
    --output output/test \
    --features 5 \
    --max-files 10 \
    --epochs 10
```

---

## Communication & Coordination

### Questions?

If you need clarification on:
- **Interface requirements**: Check this document's interface for your agent
- **Data formats**: See the type hints in interfaces
- **Pipeline flow**: See the architecture diagram above
- **Configuration**: See Pipeline Configuration Reference

### Reporting Progress

When your component is ready:

1. Ensure it implements the required interface
2. Add unit tests
3. Test with pipeline
4. Document any deviations from interface
5. Update this guide if you discover improvements

---

## Example: Plugging In Your Component

Here's how Agent 4 would integrate `GapDataset`:

### 1. Implement the interface

```python
# midi_generator/learning/gap_dataset.py
import torch
from torch.utils.data import Dataset

class GapDataset(Dataset):
    def __init__(self, midi_files, feature_extractor_200d, parameter_extractor_50d, cache_dir=None):
        self.midi_files = midi_files
        self.fe_200d = feature_extractor_200d
        self.pe_50d = parameter_extractor_50d
        # ... initialization

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        # Extract features
        features_200d = self.fe_200d.extract(self.midi_files[idx])
        params_50d = self.pe_50d.extract(self.midi_files[idx])

        # Compute gap
        gap = self._compute_gap(features_200d, params_50d)

        return torch.tensor(features_200d), torch.tensor(params_50d), torch.tensor(gap)
```

### 2. Test standalone

```python
# Test
from midi_generator.learning.gap_dataset import GapDataset

dataset = GapDataset(midi_files, fe, pe)
features, params, gap = dataset[0]

assert features.shape == (200,)
assert params.shape == (50,)
assert gap.shape == (200,)
```

### 3. Test with pipeline

```python
# Pipeline will automatically import and use it
from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline

pipeline = SemanticDiscoveryPipeline(config)
results = pipeline.run()  # Should use GapDataset in Stage 1
```

---

## Summary

- **Agent 1**: Implement `MusicalLocalityFunctions` in `musical_locality.py`
- **Agent 2**: Implement `SemanticFeature`, `SemanticFeatureBank` in `semantic_features.py`
- **Agent 3**: Implement `SemanticFeatureEncoder` in `semantic_encoder.py`
- **Agent 4**: Implement `GapDataset` in `gap_dataset.py`
- **Agent 5**: Implement `GapDiscoveryTrainer` in `gap_discovery_trainer.py`
- **Agent 6**: Implement `FeatureInterpreter` in `feature_interpreter.py`
- **Agent 8**: Implement `SemanticFeatureValidator` in `semantic_constraints.py`
- **Agent 9**: Implement `SemanticFeatureEvaluator` in `semantic_evaluation.py`

The pipeline will automatically import and coordinate all components when they're available.

---

**Questions?** Refer to the interface definitions in `semantic_discovery_pipeline.py` or this guide.

**Good luck with your implementation!** 🚀
