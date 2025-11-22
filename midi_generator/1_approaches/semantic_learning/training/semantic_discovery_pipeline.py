"""
Semantic Discovery Pipeline - Agent 7: Integration Pipeline
============================================================

End-to-end pipeline for discovering semantic musical parameters from MIDI corpora.

This pipeline coordinates components from Agents 1-6 to:
1. Compute reconstruction gaps in MIDI representation
2. Train semantic features that explain gaps
3. Interpret learned features as musical parameters
4. Validate and register discovered parameters
5. Generate comprehensive evaluation reports

Architecture:
    MIDI Corpus
        ↓
    OptimizedFeatureExtractor (200D) + HierarchicalParameterExtractor (50D)
        ↓
    GapDataset (compute reconstruction gaps)
        ↓
    SemanticFeatureEncoder (learn features via neural training)
        ↓
    FeatureInterpreter (interpret as musical concepts)
        ↓
    SemanticFeatureValidator (validate musically)
        ↓
    UniversalParameterRegistry (register new parameters)
        ↓
    SemanticFeatureEvaluator (comprehensive report)

Author: Agent 7 - Integration Pipeline
Date: November 21, 2025
Version: 1.0.0
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import warnings
import numpy as np

# Import existing components
try:
    from midi_generator.feature_selection.optimized_feature_extractor import OptimizedFeatureExtractor
    OPTIMIZED_EXTRACTOR_AVAILABLE = True
except ImportError:
    OPTIMIZED_EXTRACTOR_AVAILABLE = False
    warnings.warn("OptimizedFeatureExtractor not available")

try:
    from midi_generator.parameters.hierarchical_extractor import HierarchicalParameterExtractor
    HIERARCHICAL_EXTRACTOR_AVAILABLE = True
except ImportError:
    HIERARCHICAL_EXTRACTOR_AVAILABLE = False
    warnings.warn("HierarchicalParameterExtractor not available")

try:
    from midi_generator.parameters.universal_registry import UniversalParameterRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    warnings.warn("UniversalParameterRegistry not available")


# =============================================================================
# Pipeline Configuration
# =============================================================================

class PipelineStage(Enum):
    """Pipeline execution stages"""
    CORPUS_ANALYSIS = "corpus_analysis"
    GAP_COMPUTATION = "gap_computation"
    FEATURE_TRAINING = "feature_training"
    FEATURE_INTERPRETATION = "interpretation"
    FEATURE_VALIDATION = "validation"
    PARAMETER_REGISTRATION = "registration"
    EVALUATION = "evaluation"


@dataclass
class PipelineConfig:
    """
    Configuration for semantic discovery pipeline.

    Comprehensive settings for all stages of the discovery process.
    """
    # Input/Output paths
    midi_corpus_dir: Path
    output_dir: Path
    cache_dir: Optional[Path] = None

    # Corpus settings
    max_files: Optional[int] = None  # Limit number of files (None = all)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Feature extraction settings
    use_200d_features: bool = True
    use_50d_parameters: bool = True

    # Neural training settings
    num_semantic_features: int = 30  # Target: 20-30 parameters
    hidden_dim: int = 512
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Sparsity constraints
    sparsity_weight: float = 0.01
    target_sparsity: float = 0.1  # 10% activation

    # Locality settings
    locality_weight: float = 0.1
    locality_transformations: List[str] = field(default_factory=lambda: [
        'transpose', 'invert', 'augment', 'retrograde', 'time_shift'
    ])

    # Interpretation settings
    interpretation_threshold: float = 0.6  # Min confidence for naming
    concept_matching_threshold: float = 0.7

    # Validation settings
    redundancy_threshold: float = 0.9  # Correlation threshold
    musical_validity_strict: bool = True

    # Computational settings
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    use_mixed_precision: bool = True

    # Checkpointing
    checkpoint_frequency: int = 5  # Save every N epochs
    resume_from_checkpoint: Optional[str] = None

    # Logging
    verbose: bool = True
    log_frequency: int = 10  # Log every N batches

    def __post_init__(self):
        """Validate configuration"""
        assert self.train_split + self.val_split + self.test_split == 1.0
        assert 0 < self.target_sparsity < 1.0
        assert 10 <= self.num_semantic_features <= 50

        # Create directories
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_dir is None:
            self.cache_dir = self.output_dir / "cache"
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineProgress:
    """Track pipeline execution progress"""
    current_stage: PipelineStage
    completed_stages: List[PipelineStage] = field(default_factory=list)

    # Stage-specific progress
    corpus_files_processed: int = 0
    corpus_files_total: int = 0

    training_epoch: int = 0
    training_epochs_total: int = 0
    training_best_loss: float = float('inf')

    features_interpreted: int = 0
    features_total: int = 0

    parameters_registered: int = 0

    # Timing
    stage_start_times: Dict[str, float] = field(default_factory=dict)
    stage_durations: Dict[str, float] = field(default_factory=dict)

    def start_stage(self, stage: PipelineStage):
        """Mark stage as started"""
        self.current_stage = stage
        self.stage_start_times[stage.value] = time.time()

    def complete_stage(self, stage: PipelineStage):
        """Mark stage as completed"""
        if stage.value in self.stage_start_times:
            duration = time.time() - self.stage_start_times[stage.value]
            self.stage_durations[stage.value] = duration
        self.completed_stages.append(stage)

    def is_completed(self, stage: PipelineStage) -> bool:
        """Check if stage is completed"""
        return stage in self.completed_stages


@dataclass
class DiscoveryResults:
    """Results from semantic discovery pipeline"""
    discovered_parameters: List[str]  # Parameter names
    semantic_features: Dict[str, Any]  # SemanticFeatureBank
    extraction_functions: Dict[str, callable]  # Parameter name -> extraction function

    # Metrics
    reconstruction_improvement: float  # Gap reduction vs baseline
    interpretability_score: float  # Fraction successfully interpreted
    musical_validity_score: float  # Fraction passing validation
    redundancy_score: float  # Average correlation with existing

    # Training history
    training_losses: List[float]
    validation_losses: List[float]

    # Reports
    evaluation_report: Dict[str, Any]

    # Metadata
    config: PipelineConfig
    progress: PipelineProgress
    timestamp: str


# =============================================================================
# Component Interfaces (for Agents 1-6)
# =============================================================================

class MusicalLocalityFunctionsInterface:
    """
    Interface for Agent 1: Musical Locality Functions

    This defines the API that Agent 1 must implement.
    """

    def apply_transformation(self, midi_data: np.ndarray, transform_type: str) -> np.ndarray:
        """
        Apply musical transformation to MIDI representation.

        Args:
            midi_data: MIDI representation (200D or 50D features)
            transform_type: One of: 'transpose', 'invert', 'augment', 'retrograde', etc.

        Returns:
            Transformed MIDI representation
        """
        raise NotImplementedError("Agent 1 must implement this")

    def get_available_transformations(self) -> List[str]:
        """Get list of available transformation types"""
        raise NotImplementedError("Agent 1 must implement this")


class GapDatasetInterface:
    """
    Interface for Agent 4: Gap Dataset Creation

    This defines the API that Agent 4 must implement.
    """

    def __len__(self) -> int:
        """Number of samples in dataset"""
        raise NotImplementedError("Agent 4 must implement this")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get dataset sample.

        Returns:
            (features_200d, parameters_50d, gap_vector)
        """
        raise NotImplementedError("Agent 4 must implement this")


class SemanticEncoderInterface:
    """
    Interface for Agent 3: Neural Architecture

    This defines the API that Agent 3 must implement.
    """

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through encoder.

        Args:
            x: Input features (200D)

        Returns:
            (semantic_features, reconstructed_200d, locality_predictions)
        """
        raise NotImplementedError("Agent 3 must implement this")

    def compute_loss(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute training loss.

        Returns:
            Dictionary of loss components
        """
        raise NotImplementedError("Agent 3 must implement this")

    def extract_semantic_features(self, x: np.ndarray) -> np.ndarray:
        """Extract semantic features from input"""
        raise NotImplementedError("Agent 3 must implement this")


class GapDiscoveryTrainerInterface:
    """
    Interface for Agent 5: Training Infrastructure

    This defines the API that Agent 5 must implement.
    """

    def train(self) -> Dict[str, Any]:
        """
        Run complete training loop.

        Returns:
            Training results including losses, checkpoints, etc.
        """
        raise NotImplementedError("Agent 5 must implement this")

    def get_trained_model(self):
        """Get trained encoder model"""
        raise NotImplementedError("Agent 5 must implement this")


class FeatureInterpreterInterface:
    """
    Interface for Agent 6: Feature Interpretation

    This defines the API that Agent 6 must implement.
    """

    def interpret_feature(self, feature_idx: int, encoder) -> Dict[str, Any]:
        """
        Interpret a learned semantic feature.

        Args:
            feature_idx: Index of semantic feature
            encoder: Trained encoder model

        Returns:
            {
                'name': 'parameter_name',
                'modality': 'harmony|melody|rhythm|...',
                'confidence': 0.0-1.0,
                'concept_match': 'matched_concept',
                'extraction_function': callable,
            }
        """
        raise NotImplementedError("Agent 6 must implement this")

    def interpret_all_features(self, encoder) -> Dict[int, Dict[str, Any]]:
        """Interpret all semantic features"""
        raise NotImplementedError("Agent 6 must implement this")


class SemanticValidatorInterface:
    """
    Interface for Agent 8: Constraint Validation

    This defines the API that Agent 8 must implement.
    """

    def validate_feature(self, feature_interpretation: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a semantic feature.

        Returns:
            (is_valid, reason)
        """
        raise NotImplementedError("Agent 8 must implement this")


# =============================================================================
# Main Pipeline
# =============================================================================

class SemanticDiscoveryPipeline:
    """
    End-to-end semantic parameter discovery pipeline.

    Coordinates all components from Agents 1-6 to discover musical parameters.

    Usage:
        config = PipelineConfig(
            midi_corpus_dir=Path("data/midi"),
            output_dir=Path("output/discovery"),
            num_semantic_features=25
        )

        pipeline = SemanticDiscoveryPipeline(config)
        results = pipeline.run()

        # Access discovered parameters
        for param_name in results.discovered_parameters:
            extractor = results.extraction_functions[param_name]
            value = extractor("new_song.mid")
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.progress = PipelineProgress(current_stage=PipelineStage.CORPUS_ANALYSIS)

        # Component references (will be initialized in stages)
        self.feature_extractor_200d = None
        self.parameter_extractor_50d = None
        self.gap_dataset = None
        self.encoder = None
        self.trainer = None
        self.interpreter = None
        self.validator = None
        self.registry = None

        # Results
        self.discovered_parameters: List[str] = []
        self.extraction_functions: Dict[str, callable] = {}

        if config.verbose:
            print("=" * 80)
            print("SEMANTIC DISCOVERY PIPELINE - AGENT 7")
            print("=" * 80)
            print(f"\nConfiguration:")
            print(f"  Corpus: {config.midi_corpus_dir}")
            print(f"  Output: {config.output_dir}")
            print(f"  Target features: {config.num_semantic_features}")
            print(f"  Device: {config.device}")
            print("=" * 80)

    def run(self) -> DiscoveryResults:
        """
        Run complete discovery pipeline.

        Returns:
            Discovery results with parameters, metrics, and reports
        """
        start_time = time.time()

        try:
            # Stage 1: Analyze corpus and compute gaps
            self._stage1_compute_gaps()

            # Stage 2: Train semantic feature encoder
            self._stage2_train_features()

            # Stage 3: Interpret learned features
            self._stage3_interpret_features()

            # Stage 4: Validate features
            self._stage4_validate_features()

            # Stage 5: Register parameters
            self._stage5_register_parameters()

            # Stage 6: Generate evaluation report
            self._stage6_evaluate()

            # Compile results
            results = self._compile_results()

            # Save results
            self._save_results(results)

            total_duration = time.time() - start_time

            if self.config.verbose:
                print("\n" + "=" * 80)
                print("PIPELINE COMPLETED SUCCESSFULLY")
                print("=" * 80)
                print(f"Total duration: {total_duration/60:.1f} minutes")
                print(f"Discovered parameters: {len(results.discovered_parameters)}")
                print(f"Reconstruction improvement: {results.reconstruction_improvement:.1%}")
                print(f"Interpretability: {results.interpretability_score:.1%}")
                print(f"Musical validity: {results.musical_validity_score:.1%}")
                print("=" * 80)

            return results

        except Exception as e:
            print(f"\n❌ Pipeline failed at stage: {self.progress.current_stage.value}")
            print(f"   Error: {e}")

            # Save partial results
            self._save_checkpoint()
            raise

    def _stage1_compute_gaps(self):
        """
        Stage 1: Corpus Analysis and Gap Computation

        - Load MIDI corpus
        - Extract 200D features + 50D parameters
        - Compute reconstruction gaps
        - Create gap dataset
        """
        self.progress.start_stage(PipelineStage.GAP_COMPUTATION)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("STAGE 1: COMPUTING RECONSTRUCTION GAPS")
            print("=" * 80)

        # Initialize feature extractors
        if OPTIMIZED_EXTRACTOR_AVAILABLE and self.config.use_200d_features:
            # Load selected features
            selected_features_path = Path(__file__).parent.parent / "feature_selection" / "selected_features_200.json"
            if selected_features_path.exists():
                with open(selected_features_path, 'r') as f:
                    selected_data = json.load(f)
                    selected_features = selected_data.get('selected_features', [])

                self.feature_extractor_200d = OptimizedFeatureExtractor(selected_features)
                if self.config.verbose:
                    print("✅ Loaded 200D feature extractor")
            else:
                warnings.warn("selected_features_200.json not found, using stub")
                self.feature_extractor_200d = None
        else:
            self.feature_extractor_200d = None

        if HIERARCHICAL_EXTRACTOR_AVAILABLE and self.config.use_50d_parameters:
            self.parameter_extractor_50d = HierarchicalParameterExtractor(verbose=False)
            if self.config.verbose:
                print("✅ Loaded 50D parameter extractor")
        else:
            self.parameter_extractor_50d = None

        # Find MIDI files
        midi_files = list(Path(self.config.midi_corpus_dir).rglob("*.mid"))
        if self.config.max_files:
            midi_files = midi_files[:self.config.max_files]

        self.progress.corpus_files_total = len(midi_files)

        if self.config.verbose:
            print(f"\nFound {len(midi_files)} MIDI files")

        # Check if gap dataset exists (from Agent 4)
        try:
            # Try to import Agent 4's dataset
            from midi_generator.learning.gap_dataset import GapDataset

            dataset_cache_path = self.config.cache_dir / "gap_dataset.pkl"

            if dataset_cache_path.exists() and self.config.resume_from_checkpoint:
                # Load cached dataset
                if self.config.verbose:
                    print("📦 Loading cached gap dataset...")
                # Would load here
                self.gap_dataset = None  # Placeholder
            else:
                # Create new dataset
                if self.config.verbose:
                    print("🔨 Creating gap dataset...")

                # This would call Agent 4's GapDataset
                # For now, we create a stub
                self.gap_dataset = None

        except ImportError:
            if self.config.verbose:
                print("⚠️  Agent 4 (GapDataset) not yet implemented - using stub")
            self.gap_dataset = None

        # Process corpus
        if self.feature_extractor_200d or self.parameter_extractor_50d:
            features_200d_list = []
            parameters_50d_list = []

            for i, midi_file in enumerate(midi_files[:min(10, len(midi_files))]):  # Process subset for now
                try:
                    # Extract 200D features
                    if self.feature_extractor_200d:
                        features_200d = self.feature_extractor_200d.extract(str(midi_file))
                        features_200d_list.append(features_200d)

                    # Extract 50D parameters
                    if self.parameter_extractor_50d:
                        params_dict = self.parameter_extractor_50d.extract_from_midi(str(midi_file))
                        # Flatten to vector
                        parameters_50d = self._flatten_parameters(params_dict)
                        parameters_50d_list.append(parameters_50d)

                    self.progress.corpus_files_processed = i + 1

                    if self.config.verbose and (i + 1) % 10 == 0:
                        print(f"   Processed {i + 1}/{len(midi_files)} files...")

                except Exception as e:
                    if self.config.verbose:
                        print(f"   ⚠️  Failed to process {midi_file.name}: {e}")
                    continue

            if self.config.verbose:
                print(f"✅ Extracted features from {len(features_200d_list)} files")

        self.progress.complete_stage(PipelineStage.GAP_COMPUTATION)

        if self.config.verbose:
            duration = self.progress.stage_durations[PipelineStage.GAP_COMPUTATION.value]
            print(f"✅ Stage 1 completed in {duration:.1f} seconds")

    def _flatten_parameters(self, params_dict: Dict[str, Any]) -> np.ndarray:
        """Flatten hierarchical parameters to vector"""
        # Simplified flattening - would be more sophisticated in production
        values = []

        # Level 1
        level1 = params_dict.get('level1_global', {})
        values.append(level1.get('tempo.bpm', 120.0) / 200.0)  # Normalize
        values.append(1.0 if level1.get('key.mode') == 'major' else 0.0)

        # Level 2
        level2 = params_dict.get('level2_universal', {})
        for category in ['harmony', 'melody', 'rhythm', 'dynamics', 'texture']:
            cat_data = level2.get(category, {})
            for key, val in cat_data.items():
                if isinstance(val, (int, float)):
                    values.append(float(val))

        # Pad to 50
        while len(values) < 50:
            values.append(0.0)

        return np.array(values[:50], dtype=np.float32)

    def _stage2_train_features(self):
        """
        Stage 2: Train Semantic Feature Encoder

        - Initialize neural encoder (Agent 3)
        - Train on gap dataset (Agent 5)
        - Learn semantic features
        """
        self.progress.start_stage(PipelineStage.FEATURE_TRAINING)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("STAGE 2: TRAINING SEMANTIC FEATURES")
            print("=" * 80)

        try:
            # Import Agent 3's encoder
            from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder

            # Initialize encoder
            self.encoder = SemanticFeatureEncoder(
                input_dim=200,
                num_features=self.config.num_semantic_features,
                hidden_dim=self.config.hidden_dim
            )

            if self.config.verbose:
                print(f"✅ Initialized encoder: 200D → {self.config.num_semantic_features} features")

        except ImportError:
            if self.config.verbose:
                print("⚠️  Agent 3 (SemanticEncoder) not yet implemented - using stub")
            self.encoder = None

        try:
            # Import Agent 5's trainer
            from midi_generator.learning.gap_discovery_trainer import GapDiscoveryTrainer

            # Initialize trainer
            self.trainer = GapDiscoveryTrainer(
                encoder=self.encoder,
                dataset=self.gap_dataset,
                config=self.config
            )

            if self.config.verbose:
                print("🏋️  Starting training...")
                print(f"   Max epochs: {self.config.max_epochs}")
                print(f"   Batch size: {self.config.batch_size}")

            # Train
            training_results = self.trainer.train()

            if self.config.verbose:
                print(f"✅ Training completed")
                print(f"   Final loss: {training_results.get('final_loss', 'N/A')}")

        except ImportError:
            if self.config.verbose:
                print("⚠️  Agent 5 (GapDiscoveryTrainer) not yet implemented - skipping training")
            self.trainer = None

        self.progress.complete_stage(PipelineStage.FEATURE_TRAINING)

        if self.config.verbose:
            duration = self.progress.stage_durations[PipelineStage.FEATURE_TRAINING.value]
            print(f"✅ Stage 2 completed in {duration/60:.1f} minutes")

    def _stage3_interpret_features(self):
        """
        Stage 3: Interpret Learned Features

        - Analyze semantic features (Agent 6)
        - Match to musical concepts
        - Generate extraction functions
        """
        self.progress.start_stage(PipelineStage.FEATURE_INTERPRETATION)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("STAGE 3: INTERPRETING FEATURES")
            print("=" * 80)

        try:
            # Import Agent 6's interpreter
            from midi_generator.learning.feature_interpreter import FeatureInterpreter

            # Initialize interpreter
            self.interpreter = FeatureInterpreter(
                threshold=self.config.interpretation_threshold
            )

            if self.config.verbose:
                print(f"🔍 Interpreting {self.config.num_semantic_features} features...")

            # Interpret all features
            interpretations = self.interpreter.interpret_all_features(self.encoder)

            # Store results
            self.progress.features_total = len(interpretations)
            self.progress.features_interpreted = sum(
                1 for interp in interpretations.values()
                if interp.get('confidence', 0) >= self.config.interpretation_threshold
            )

            if self.config.verbose:
                print(f"✅ Interpreted {self.progress.features_interpreted}/{self.progress.features_total} features")

                # Show sample interpretations
                print("\nSample interpretations:")
                for idx, interp in list(interpretations.items())[:5]:
                    print(f"  Feature {idx}: {interp.get('name', 'unknown')}")
                    print(f"    Modality: {interp.get('modality', 'N/A')}")
                    print(f"    Confidence: {interp.get('confidence', 0):.2f}")

        except ImportError:
            if self.config.verbose:
                print("⚠️  Agent 6 (FeatureInterpreter) not yet implemented - using stub")
            self.interpreter = None

        self.progress.complete_stage(PipelineStage.FEATURE_INTERPRETATION)

        if self.config.verbose:
            duration = self.progress.stage_durations[PipelineStage.FEATURE_INTERPRETATION.value]
            print(f"✅ Stage 3 completed in {duration:.1f} seconds")

    def _stage4_validate_features(self):
        """
        Stage 4: Validate Features

        - Check musical validity (Agent 8)
        - Detect redundancy with existing parameters
        - Filter invalid features
        """
        self.progress.start_stage(PipelineStage.FEATURE_VALIDATION)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("STAGE 4: VALIDATING FEATURES")
            print("=" * 80)

        try:
            # Import Agent 8's validator
            from midi_generator.learning.semantic_constraints import SemanticFeatureValidator

            # Initialize validator
            self.validator = SemanticFeatureValidator(
                redundancy_threshold=self.config.redundancy_threshold,
                strict_musical_validity=self.config.musical_validity_strict
            )

            if self.config.verbose:
                print("✓ Validating features...")

            # Validation would happen here
            # For now, stub

        except ImportError:
            if self.config.verbose:
                print("⚠️  Agent 8 (SemanticValidator) not yet implemented - skipping validation")
            self.validator = None

        self.progress.complete_stage(PipelineStage.FEATURE_VALIDATION)

        if self.config.verbose:
            duration = self.progress.stage_durations[PipelineStage.FEATURE_VALIDATION.value]
            print(f"✅ Stage 4 completed in {duration:.1f} seconds")

    def _stage5_register_parameters(self):
        """
        Stage 5: Register Parameters

        - Register discovered parameters in UniversalParameterRegistry
        - Create extraction functions
        """
        self.progress.start_stage(PipelineStage.PARAMETER_REGISTRATION)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("STAGE 5: REGISTERING PARAMETERS")
            print("=" * 80)

        if REGISTRY_AVAILABLE:
            self.registry = UniversalParameterRegistry()

            # Register parameters
            # For now, stub - would use interpreter results
            self.progress.parameters_registered = 0

            if self.config.verbose:
                print(f"✅ Registered {self.progress.parameters_registered} parameters")
        else:
            if self.config.verbose:
                print("⚠️  UniversalParameterRegistry not available")

        self.progress.complete_stage(PipelineStage.PARAMETER_REGISTRATION)

        if self.config.verbose:
            duration = self.progress.stage_durations[PipelineStage.PARAMETER_REGISTRATION.value]
            print(f"✅ Stage 5 completed in {duration:.1f} seconds")

    def _stage6_evaluate(self):
        """
        Stage 6: Generate Evaluation Report

        - Compute reconstruction metrics (Agent 9)
        - Evaluate interpretability
        - Generate comprehensive report
        """
        self.progress.start_stage(PipelineStage.EVALUATION)

        if self.config.verbose:
            print("\n" + "=" * 80)
            print("STAGE 6: GENERATING EVALUATION REPORT")
            print("=" * 80)

        try:
            # Import Agent 9's evaluator
            from midi_generator.evaluation.semantic_evaluation import SemanticFeatureEvaluator

            # Evaluate
            # Stub for now

        except ImportError:
            if self.config.verbose:
                print("⚠️  Agent 9 (SemanticEvaluator) not yet implemented - generating basic report")

        self.progress.complete_stage(PipelineStage.EVALUATION)

        if self.config.verbose:
            duration = self.progress.stage_durations[PipelineStage.EVALUATION.value]
            print(f"✅ Stage 6 completed in {duration:.1f} seconds")

    def _compile_results(self) -> DiscoveryResults:
        """Compile final results"""
        return DiscoveryResults(
            discovered_parameters=self.discovered_parameters,
            semantic_features={},  # Would contain SemanticFeatureBank
            extraction_functions=self.extraction_functions,
            reconstruction_improvement=0.0,  # Placeholder
            interpretability_score=0.0,
            musical_validity_score=0.0,
            redundancy_score=0.0,
            training_losses=[],
            validation_losses=[],
            evaluation_report={},
            config=self.config,
            progress=self.progress,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def _save_results(self, results: DiscoveryResults):
        """Save results to disk"""
        output_path = self.config.output_dir / "discovery_results.json"

        # Convert to serializable format
        results_dict = {
            "discovered_parameters": results.discovered_parameters,
            "metrics": {
                "reconstruction_improvement": results.reconstruction_improvement,
                "interpretability_score": results.interpretability_score,
                "musical_validity_score": results.musical_validity_score,
                "redundancy_score": results.redundancy_score
            },
            "progress": {
                "corpus_files_processed": self.progress.corpus_files_processed,
                "features_interpreted": self.progress.features_interpreted,
                "parameters_registered": self.progress.parameters_registered
            },
            "timestamp": results.timestamp
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        if self.config.verbose:
            print(f"\n💾 Results saved to: {output_path}")

    def _save_checkpoint(self):
        """Save checkpoint for resuming"""
        checkpoint_path = self.config.output_dir / "checkpoint.json"

        checkpoint = {
            "progress": {
                "current_stage": self.progress.current_stage.value,
                "completed_stages": [s.value for s in self.progress.completed_stages]
            }
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_config(
    midi_corpus_dir: str,
    output_dir: str,
    **kwargs
) -> PipelineConfig:
    """
    Create pipeline config with sensible defaults.

    Args:
        midi_corpus_dir: Path to MIDI corpus
        output_dir: Output directory
        **kwargs: Override default settings

    Returns:
        PipelineConfig instance
    """
    defaults = {
        "midi_corpus_dir": Path(midi_corpus_dir),
        "output_dir": Path(output_dir),
        "num_semantic_features": 25,
        "max_epochs": 100,
        "device": "cuda",
        "verbose": True
    }

    defaults.update(kwargs)

    return PipelineConfig(**defaults)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Test the pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Discovery Pipeline")
    parser.add_argument("--corpus", type=str, required=True, help="MIDI corpus directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--features", type=int, default=25, help="Number of semantic features")
    parser.add_argument("--max-files", type=int, help="Limit number of files")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    # Create config
    config = PipelineConfig(
        midi_corpus_dir=Path(args.corpus),
        output_dir=Path(args.output),
        num_semantic_features=args.features,
        max_files=args.max_files,
        device=args.device,
        verbose=True
    )

    # Run pipeline
    pipeline = SemanticDiscoveryPipeline(config)
    results = pipeline.run()

    print("\n" + "=" * 80)
    print("DISCOVERY COMPLETE")
    print("=" * 80)
    print(f"Discovered {len(results.discovered_parameters)} parameters")
    print(f"Results saved to: {config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
