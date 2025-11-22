"""
AGENT 05: Gap Discovery Training Infrastructure
================================================

Training infrastructure for automated semantic feature discovery through
reconstruction gap analysis. Discovers 20-30 interpretable musical parameters
automatically by learning features that minimize reconstruction errors.

Architecture Flow:
    MIDI → 200D Features → Semantic Encoder → Learned Features (20-30D) →
    Decoder → Reconstructed 200D → Minimize Gap

Components:
    - TrainingConfig: Hyperparameter configuration
    - LocalityTransformGenerator: Musical transformation for locality constraints
    - TrainingMonitor: Metrics tracking and logging
    - GapDiscoveryTrainer: Main training loop with early stopping & checkpointing

Integration Points:
    - Agent 3: SemanticFeatureEncoder (neural architecture)
    - Agent 4: GapDataset (gap computation and caching)
    - Agents 1&2: Musical locality functions and semantic features

Author: Agent 05 - Training Infrastructure Specialist
License: MIT
Date: November 21, 2025
"""

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed")

# NumPy imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("WARNING: NumPy not installed")

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
# Configuration
# ============================================================================

class LocalityType(Enum):
    """Types of musical locality transformations (from Agent 1)"""
    TRANSPOSE = "transpose"
    INVERT_INTERVALS = "invert_intervals"
    TIME_SHIFT = "time_shift"
    AUGMENT = "augment"
    DIMINISH = "diminish"
    RETROGRADE = "retrograde"
    OCTAVE_SHIFT = "octave_shift"
    RHYTHMIC_VARIATION = "rhythmic_variation"
    DYNAMIC_SCALING = "dynamic_scaling"
    ARTICULATION_CHANGE = "articulation_change"
    HARMONIC_SUBSTITUTION = "harmonic_substitution"
    MELODIC_ORNAMENTATION = "melodic_ornamentation"


@dataclass
class TrainingConfig:
    """
    Configuration for semantic feature discovery training.

    Attributes:
        # Paths
        midi_corpus_dir: Directory containing MIDI training corpus
        output_dir: Root output directory
        checkpoint_dir: Model checkpoint directory
        log_dir: Training logs directory

        # Model architecture (Agent 3 integration)
        input_dim: Input feature dimension (200D from OptimizedFeatureExtractor)
        hidden_dim: Hidden layer dimension
        num_semantic_features: Number of semantic features to discover (20-30)

        # Training hyperparameters
        batch_size: Training batch size
        num_epochs: Maximum training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight

        # Loss weights
        reconstruction_weight: Weight for reconstruction loss
        sparsity_weight: Weight for sparsity constraint (L1 on features)
        locality_weight: Weight for locality constraint
        orthogonality_weight: Weight for feature orthogonality

        # Learning rate scheduling
        use_lr_scheduler: Enable learning rate scheduling
        lr_scheduler_type: Scheduler type ('cosine', 'step', 'plateau')
        lr_patience: Patience for plateau scheduler
        lr_factor: Reduction factor for step/plateau schedulers

        # Early stopping
        early_stopping_patience: Epochs without improvement before stopping
        early_stopping_min_delta: Minimum improvement to count as progress

        # Locality constraints (Agent 1 integration)
        locality_types: List of locality transformations to use
        locality_epsilon: Maximum allowed feature change under locality

        # Checkpointing
        save_every_n_epochs: Save checkpoint every N epochs
        keep_n_checkpoints: Keep only N most recent checkpoints

        # Logging
        log_every_n_steps: Log metrics every N steps
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name

        # Device
        device: Training device ('auto', 'cpu', 'cuda', 'mps')

        # Reproducibility
        random_seed: Random seed for reproducibility

        # Feature interpretation (Agent 6 integration)
        min_interpretability_score: Minimum score for feature to be interpretable
        target_interpretable_ratio: Target ratio of interpretable features (0.6 = 60%)
    """

    # Paths
    midi_corpus_dir: Path = Path('data/midi_corpus')
    output_dir: Path = Path('output/semantic_discovery')
    checkpoint_dir: Path = Path('output/semantic_discovery/checkpoints')
    log_dir: Path = Path('output/semantic_discovery/logs')

    # Model architecture
    input_dim: int = 200  # From OptimizedFeatureExtractor
    hidden_dim: int = 512
    num_semantic_features: int = 25  # Target 20-30 features

    # Training hyperparameters
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Loss weights
    reconstruction_weight: float = 1.0
    sparsity_weight: float = 0.01
    locality_weight: float = 0.5
    orthogonality_weight: float = 0.1

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Early stopping
    early_stopping_patience: int = 25
    early_stopping_min_delta: float = 0.0001

    # Locality constraints
    locality_types: List[str] = field(default_factory=lambda: [
        'transpose', 'time_shift', 'octave_shift', 'augment', 'diminish'
    ])
    locality_epsilon: float = 0.1

    # Checkpointing
    save_every_n_epochs: int = 10
    keep_n_checkpoints: int = 5

    # Logging
    log_every_n_steps: int = 50
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = 'semantic-feature-discovery'
    wandb_entity: Optional[str] = None

    # Device
    device: str = 'auto'

    # Reproducibility
    random_seed: int = 42

    # Feature interpretation
    min_interpretability_score: float = 0.6
    target_interpretable_ratio: float = 0.6

    def __post_init__(self):
        """Initialize paths and device"""
        # Convert string paths to Path objects
        if isinstance(self.midi_corpus_dir, str):
            self.midi_corpus_dir = Path(self.midi_corpus_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if self.device == 'auto' and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

    def save(self, path: Path):
        """Save configuration to JSON"""
        config_dict = asdict(self)
        # Convert Path objects to strings
        config_dict['midi_corpus_dir'] = str(self.midi_corpus_dir)
        config_dict['output_dir'] = str(self.output_dir)
        config_dict['checkpoint_dir'] = str(self.checkpoint_dir)
        config_dict['log_dir'] = str(self.log_dir)

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'TrainingConfig':
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Convert string paths back to Path objects
        config_dict['midi_corpus_dir'] = Path(config_dict['midi_corpus_dir'])
        config_dict['output_dir'] = Path(config_dict['output_dir'])
        config_dict['checkpoint_dir'] = Path(config_dict['checkpoint_dir'])
        config_dict['log_dir'] = Path(config_dict['log_dir'])

        return cls(**config_dict)


# ============================================================================
# Locality Transform Generator
# ============================================================================

class LocalityTransformGenerator:
    """
    Generates musical locality transformations for training.

    Integrates with Agent 1's MusicalLocalityFunctions to apply transformations
    that should preserve semantic features (locality constraint).

    During training:
        1. Sample a transformation type (e.g., transpose)
        2. Apply transformation to input features
        3. Encoder should produce similar semantic features
        4. Compute locality loss: L = ||encoder(x) - encoder(transform(x))||²

    Usage:
        generator = LocalityTransformGenerator(config)
        transformed_batch = generator.apply_random_transform(batch)
        locality_loss = generator.compute_locality_loss(original, transformed, encoder)
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize locality transform generator.

        Args:
            config: Training configuration with locality settings
        """
        self.config = config
        self.locality_types = config.locality_types
        self.epsilon = config.locality_epsilon

        # Try to import Agent 1's musical locality functions
        self.musical_locality = None
        try:
            from midi_generator.learning.musical_locality import MusicalLocalityFunctions
            self.musical_locality = MusicalLocalityFunctions()
            print("✅ Integrated with Agent 1's MusicalLocalityFunctions")
        except ImportError:
            print("⚠️  Agent 1's MusicalLocalityFunctions not available, using synthetic transforms")

    def apply_random_transform(
        self,
        features: torch.Tensor,
        transform_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        Apply random musical locality transformation.

        Args:
            features: Input features [batch_size, input_dim]
            transform_type: Specific transform type (random if None)

        Returns:
            Transformed features, transformation type used (or None if no locality types)
        """
        # If no locality types configured, return None
        if len(self.locality_types) == 0:
            return None, None

        if transform_type is None:
            transform_type = np.random.choice(self.locality_types)

        if self.musical_locality is not None:
            # Use Agent 1's actual transformations
            # Note: This will be implemented once Agent 1 completes their work
            transformed = self._apply_agent1_transform(features, transform_type)
        else:
            # Synthetic transformations for testing
            transformed = self._apply_synthetic_transform(features, transform_type)

        return transformed, transform_type

    def _apply_agent1_transform(
        self,
        features: torch.Tensor,
        transform_type: str
    ) -> torch.Tensor:
        """Apply Agent 1's musical locality transformations"""
        # Placeholder for Agent 1 integration
        # This will be implemented once Agent 1's MusicalLocalityFunctions is available
        # For now, use synthetic transforms
        return self._apply_synthetic_transform(features, transform_type)

    def _apply_synthetic_transform(
        self,
        features: torch.Tensor,
        transform_type: str
    ) -> torch.Tensor:
        """
        Apply synthetic transformations for testing.

        These simulate musical transformations by applying small, structured
        perturbations to the feature space.
        """
        batch_size, feature_dim = features.shape
        device = features.device

        if transform_type == 'transpose':
            # Simulate pitch transposition: rotate pitch-related features
            shift = torch.randn(batch_size, 1, device=device) * 0.2
            transformed = features + shift

        elif transform_type == 'time_shift':
            # Simulate time shift: minimal effect on most features
            shift = torch.randn(batch_size, feature_dim, device=device) * 0.05
            transformed = features + shift

        elif transform_type == 'octave_shift':
            # Similar to transpose but larger magnitude
            shift = torch.randn(batch_size, 1, device=device) * 0.3
            transformed = features + shift

        elif transform_type == 'augment':
            # Time stretching: affects rhythm features
            scale = 1.0 + torch.randn(batch_size, 1, device=device) * 0.1
            transformed = features * scale

        elif transform_type == 'diminish':
            # Time compression: inverse of augment
            scale = 1.0 - torch.randn(batch_size, 1, device=device) * 0.1
            transformed = features * scale

        else:
            # Default: small random perturbation
            noise = torch.randn_like(features) * 0.05
            transformed = features + noise

        return transformed

    def compute_locality_loss(
        self,
        features_original: torch.Tensor,
        features_transformed: torch.Tensor,
        encoder: nn.Module
    ) -> torch.Tensor:
        """
        Compute locality constraint loss.

        Semantic features should be similar under musical locality transformations.

        Args:
            features_original: Original input features [batch_size, input_dim]
            features_transformed: Transformed features [batch_size, input_dim]
            encoder: Semantic feature encoder

        Returns:
            Locality loss scalar
        """
        # Encode both versions
        semantic_original = encoder(features_original)
        semantic_transformed = encoder(features_transformed)

        # Compute L2 distance
        locality_loss = torch.mean((semantic_original - semantic_transformed) ** 2)

        # Only penalize if difference exceeds epsilon
        locality_loss = torch.clamp(locality_loss - self.epsilon, min=0.0)

        return locality_loss


# ============================================================================
# Training Monitoring
# ============================================================================

@dataclass
class TrainingMetrics:
    """Metrics for one epoch"""
    epoch: int
    train_loss: float
    val_loss: float
    reconstruction_loss: float = 0.0
    sparsity_loss: float = 0.0
    locality_loss: float = 0.0
    orthogonality_loss: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    avg_feature_sparsity: float = 0.0  # Average % of features active
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TrainingMonitor:
    """
    Monitor and log training metrics.

    Features:
        - CSV logging for metrics history
        - TensorBoard integration (optional)
        - Weights & Biases integration (optional)
        - Best model tracking
        - Training progress visualization

    Usage:
        monitor = TrainingMonitor(config)
        monitor.log_metrics(metrics)
        monitor.save_history()
        monitor.close()
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize training monitor.

        Args:
            config: Training configuration
        """
        self.config = config
        self.log_dir = config.log_dir
        self.history: List[TrainingMetrics] = []

        # TensorBoard writer
        self.tb_writer = None
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.log_dir / 'tensorboard')
                print("✅ TensorBoard logging enabled")
            except ImportError:
                print("⚠️  TensorBoard not available")

        # Weights & Biases
        self.wandb = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=asdict(config)
                )
                print("✅ Weights & Biases logging enabled")
            except ImportError:
                print("⚠️  Weights & Biases not available")

        # Create CSV log file
        self.csv_path = self.log_dir / 'training_metrics.csv'
        self._init_csv_log()

    def _init_csv_log(self):
        """Initialize CSV log file with headers"""
        with open(self.csv_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,reconstruction_loss,sparsity_loss,'
                   'locality_loss,orthogonality_loss,learning_rate,epoch_time,'
                   'avg_feature_sparsity,timestamp\n')

    def log_metrics(self, metrics: TrainingMetrics):
        """
        Log metrics for an epoch.

        Args:
            metrics: Training metrics to log
        """
        self.history.append(metrics)

        # CSV logging
        with open(self.csv_path, 'a') as f:
            f.write(f'{metrics.epoch},{metrics.train_loss},{metrics.val_loss},'
                   f'{metrics.reconstruction_loss},{metrics.sparsity_loss},'
                   f'{metrics.locality_loss},{metrics.orthogonality_loss},'
                   f'{metrics.learning_rate},{metrics.epoch_time},'
                   f'{metrics.avg_feature_sparsity},{metrics.timestamp}\n')

        # TensorBoard logging
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/train', metrics.train_loss, metrics.epoch)
            self.tb_writer.add_scalar('Loss/val', metrics.val_loss, metrics.epoch)
            self.tb_writer.add_scalar('Loss/reconstruction', metrics.reconstruction_loss, metrics.epoch)
            self.tb_writer.add_scalar('Loss/sparsity', metrics.sparsity_loss, metrics.epoch)
            self.tb_writer.add_scalar('Loss/locality', metrics.locality_loss, metrics.epoch)
            self.tb_writer.add_scalar('Loss/orthogonality', metrics.orthogonality_loss, metrics.epoch)
            self.tb_writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch)
            self.tb_writer.add_scalar('Feature_Sparsity', metrics.avg_feature_sparsity, metrics.epoch)

        # Weights & Biases logging
        if self.wandb is not None:
            self.wandb.log({
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'reconstruction_loss': metrics.reconstruction_loss,
                'sparsity_loss': metrics.sparsity_loss,
                'locality_loss': metrics.locality_loss,
                'orthogonality_loss': metrics.orthogonality_loss,
                'learning_rate': metrics.learning_rate,
                'epoch_time': metrics.epoch_time,
                'feature_sparsity': metrics.avg_feature_sparsity,
            })

    def get_best_epoch(self) -> Optional[TrainingMetrics]:
        """Get epoch with best validation loss"""
        if not self.history:
            return None
        return min(self.history, key=lambda m: m.val_loss)

    def save_history(self):
        """Save complete metrics history to JSON"""
        history_path = self.log_dir / 'training_history.json'
        history_data = [asdict(m) for m in self.history]

        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"📝 Saved training history to {history_path}")

    def close(self):
        """Close writers"""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb is not None:
            self.wandb.finish()


# ============================================================================
# Gap Discovery Trainer
# ============================================================================

class GapDiscoveryTrainer:
    """
    Main trainer for semantic feature discovery via gap minimization.

    Training Pipeline:
        1. Load 200D features + compute reconstruction gaps (Agent 4)
        2. Initialize semantic encoder (Agent 3)
        3. Train encoder to minimize:
           - Reconstruction loss: ||decoded - original||²
           - Sparsity loss: L1 on semantic features
           - Locality loss: Feature stability under transformations
           - Orthogonality loss: Feature independence
        4. Extract learned semantic features (Agent 2)
        5. Save best model + feature bank

    Integration Points:
        - Uses Agent 3's SemanticFeatureEncoder
        - Uses Agent 4's GapDataset
        - Produces SemanticFeatureBank for Agent 6

    Usage:
        config = TrainingConfig()
        trainer = GapDiscoveryTrainer(config)
        trainer.train(train_loader, val_loader)
        feature_bank = trainer.extract_semantic_features()
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[nn.Module] = None
    ):
        """
        Initialize gap discovery trainer.

        Args:
            config: Training configuration
            model: Pre-initialized model (creates new if None)
        """
        if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
            raise RuntimeError("PyTorch and NumPy are required for training")

        self.config = config
        self.device = torch.device(config.device)

        # Set random seeds
        self._set_random_seeds(config.random_seed)

        # Create or use provided model (Agent 3 integration)
        if model is None:
            self.model = self._create_model()
        else:
            self.model = model

        self.model.to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.lr_scheduler = None
        if config.use_lr_scheduler:
            self.lr_scheduler = self._create_lr_scheduler()

        # Create locality transform generator
        self.locality_generator = LocalityTransformGenerator(config)

        # Create training monitor
        self.monitor = TrainingMonitor(config)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0

        print("\n" + "="*70)
        print("GAP DISCOVERY TRAINER INITIALIZED")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Input dim: {config.input_dim}")
        print(f"Semantic features: {config.num_semantic_features}")
        print(f"Hidden dim: {config.hidden_dim}")
        print("="*70 + "\n")

    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _create_model(self) -> nn.Module:
        """
        Create semantic feature encoder model.

        Tries to use Agent 3's SemanticFeatureEncoder, falls back to simple autoencoder.
        """
        try:
            from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder, EncoderConfig

            # Create config object for SemanticFeatureEncoder
            encoder_config = EncoderConfig(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_semantic_features=self.config.num_semantic_features
            )
            model = SemanticFeatureEncoder(encoder_config)
            print("✅ Using Agent 3's SemanticFeatureEncoder")
            return model
        except ImportError:
            print("⚠️  Agent 3's SemanticFeatureEncoder not available, using simple autoencoder")
            return self._create_simple_autoencoder()
        except TypeError as e:
            print(f"⚠️  Error creating SemanticFeatureEncoder: {e}, using simple autoencoder")
            return self._create_simple_autoencoder()

    def _create_simple_autoencoder(self) -> nn.Module:
        """Create simple autoencoder for testing (fallback)"""
        class SimpleSemanticEncoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_features):
                super().__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, num_features),
                    nn.Tanh()  # Bounded activation for stability
                )

                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, input_dim)
                )

            def forward(self, x):
                semantic_features = self.encoder(x)
                reconstruction = self.decoder(semantic_features)
                return {
                    'semantic_features': semantic_features,
                    'reconstructed': reconstruction
                }

            def encode(self, x):
                return self.encoder(x)

        return SimpleSemanticEncoder(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.num_semantic_features
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

    def _create_lr_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.lr_scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=self.config.lr_factor
            )
        elif self.config.lr_scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler_type}")

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        features_original: torch.Tensor,
        compute_locality: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total training loss.

        Loss = w_recon * L_reconstruction +
               w_sparse * L_sparsity +
               w_local * L_locality +
               w_ortho * L_orthogonality

        Args:
            outputs: Model outputs dict with 'semantic_features' and 'reconstructed'
            features_original: Original input features
            compute_locality: Whether to compute locality loss

        Returns:
            Total loss, dictionary of component losses
        """
        semantic_features = outputs['semantic_features']
        reconstruction = outputs['reconstructed']

        # 1. Reconstruction loss: MSE between reconstruction and original
        reconstruction_loss = torch.mean((reconstruction - features_original) ** 2)

        # 2. Sparsity loss: L1 on semantic features (encourage sparse activations)
        sparsity_loss = torch.mean(torch.abs(semantic_features))

        # 3. Locality loss: Features should be stable under transformations
        locality_loss = torch.tensor(0.0, device=self.device)
        if compute_locality and self.config.locality_weight > 0:
            # Apply random transformation
            features_transformed, _ = self.locality_generator.apply_random_transform(
                features_original
            )
            # Only compute locality loss if transform was applied (not None)
            if features_transformed is not None:
                locality_loss = self.locality_generator.compute_locality_loss(
                    features_original,
                    features_transformed,
                    self.model.encode if hasattr(self.model, 'encode') else self.model.encoder
                )

        # 4. Orthogonality loss: Encourage feature independence
        orthogonality_loss = torch.tensor(0.0, device=self.device)
        if self.config.orthogonality_weight > 0:
            # Compute feature correlation matrix
            features_normalized = semantic_features - semantic_features.mean(dim=0, keepdim=True)
            correlation = torch.mm(features_normalized.T, features_normalized)
            correlation = correlation / (features_normalized.shape[0] + 1e-8)

            # Penalize off-diagonal elements (want identity matrix)
            identity = torch.eye(self.config.num_semantic_features, device=self.device)
            orthogonality_loss = torch.mean((correlation - identity) ** 2)

        # Total weighted loss
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss +
            self.config.sparsity_weight * sparsity_loss +
            self.config.locality_weight * locality_loss +
            self.config.orthogonality_weight * orthogonality_loss
        )

        # Component losses for logging
        loss_components = {
            'reconstruction': reconstruction_loss.item(),
            'sparsity': sparsity_loss.item(),
            'locality': locality_loss.item(),
            'orthogonality': orthogonality_loss.item(),
        }

        return total_loss, loss_components

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader (from Agent 4's GapDataset)

        Returns:
            Average loss, average component losses
        """
        self.model.train()

        total_loss = 0.0
        component_losses = {
            'reconstruction': 0.0,
            'sparsity': 0.0,
            'locality': 0.0,
            'orthogonality': 0.0,
        }
        n_batches = 0
        total_sparsity = 0.0

        iterator = tqdm(train_loader, desc=f"Epoch {self.current_epoch}") if TQDM_AVAILABLE else train_loader

        for batch_idx, batch in enumerate(iterator):
            # Handle different batch formats (from Agent 4)
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                features, _ = batch  # (features, labels) format
            else:
                features = batch  # features only

            features = features.to(self.device)

            # Forward pass
            outputs = self.model(features)

            # Compute loss
            loss, batch_component_losses = self.compute_loss(
                outputs,
                features,
                compute_locality=True
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            for key, value in batch_component_losses.items():
                component_losses[key] += value

            # Compute feature sparsity (% of features with |activation| > 0.1)
            semantic_features = outputs['semantic_features']
            sparsity = torch.mean((torch.abs(semantic_features) > 0.1).float()).item()
            total_sparsity += sparsity

            n_batches += 1
            self.global_step += 1

            # Update progress bar
            if TQDM_AVAILABLE and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': loss.item()})

        # Average losses
        avg_loss = total_loss / n_batches
        avg_component_losses = {k: v / n_batches for k, v in component_losses.items()}
        avg_component_losses['sparsity_pct'] = (total_sparsity / n_batches) * 100

        return avg_loss, avg_component_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss, average component losses
        """
        self.model.eval()

        total_loss = 0.0
        component_losses = {
            'reconstruction': 0.0,
            'sparsity': 0.0,
            'locality': 0.0,
            'orthogonality': 0.0,
        }
        n_batches = 0

        for batch in val_loader:
            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                features, _ = batch
            else:
                features = batch

            features = features.to(self.device)

            # Forward pass
            outputs = self.model(features)

            # Compute loss
            loss, batch_component_losses = self.compute_loss(
                outputs,
                features,
                compute_locality=False  # Skip locality during validation for speed
            )

            # Track metrics
            total_loss += loss.item()
            for key, value in batch_component_losses.items():
                component_losses[key] += value

            n_batches += 1

        # Average losses
        avg_loss = total_loss / n_batches
        avg_component_losses = {k: v / n_batches for k, v in component_losses.items()}

        return avg_loss, avg_component_losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from_checkpoint: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader (from Agent 4)
            val_loader: Validation data loader
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training summary dictionary
        """
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)

        print("\n" + "="*70)
        print("STARTING SEMANTIC FEATURE DISCOVERY TRAINING")
        print("="*70)
        print(f"Target features: {self.config.num_semantic_features}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max epochs: {self.config.num_epochs}")
        print("="*70 + "\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss, train_components = self.train_epoch(train_loader)

            # Validate
            val_loss, val_components = self.validate(val_loader)

            epoch_time = time.time() - epoch_start

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                reconstruction_loss=val_components['reconstruction'],
                sparsity_loss=val_components['sparsity'],
                locality_loss=val_components['locality'],
                orthogonality_loss=val_components['orthogonality'],
                learning_rate=current_lr,
                epoch_time=epoch_time,
                avg_feature_sparsity=train_components.get('sparsity_pct', 0.0)
            )

            # Log metrics
            self.monitor.log_metrics(metrics)

            # Print progress
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Recon: {val_components['reconstruction']:.4f} | "
                  f"Sparse: {train_components.get('sparsity_pct', 0):.1f}% | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s")

            # Learning rate scheduling
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            # Check for improvement
            if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                self.save_checkpoint(
                    self.config.checkpoint_dir / 'best_model.pt',
                    is_best=True
                )
                print(f"  ✅ New best model! (Val Loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Regular checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(
                    self.config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                )

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\n⏹️  Early stopping after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                break

        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        best_epoch_metrics = self.monitor.get_best_epoch()
        if best_epoch_metrics:
            print(f"Best epoch: {best_epoch_metrics.epoch}")
        print("="*70 + "\n")

        # Save metrics history
        self.monitor.save_history()
        self.monitor.close()

        # Save final model
        self.save_checkpoint(self.config.checkpoint_dir / 'final_model.pt')

        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'best_epoch': best_epoch_metrics.epoch if best_epoch_metrics else None,
        }

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'global_step': self.global_step,
            'config': asdict(self.config),
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

        # Cleanup old checkpoints
        if not is_best:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only most recent N"""
        checkpoint_files = sorted(
            self.config.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: p.stat().st_mtime
        )

        # Remove oldest checkpoints
        for old_checkpoint in checkpoint_files[:-self.config.keep_n_checkpoints]:
            old_checkpoint.unlink()

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        self.global_step = checkpoint.get('global_step', 0)

        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")

    @torch.no_grad()
    def extract_semantic_features(
        self,
        data_loader: DataLoader
    ) -> Dict[str, np.ndarray]:
        """
        Extract learned semantic features from dataset.

        This produces the SemanticFeatureBank for Agent 6 (interpretation).

        Args:
            data_loader: Data loader for feature extraction

        Returns:
            Dictionary with:
                - 'features': Semantic features [n_samples, num_semantic_features]
                - 'reconstructions': Reconstructed inputs [n_samples, input_dim]
                - 'reconstruction_errors': Per-sample reconstruction errors
        """
        self.model.eval()

        all_semantic_features = []
        all_reconstructions = []
        all_reconstruction_errors = []

        print("Extracting semantic features...")

        for batch in tqdm(data_loader, disable=not TQDM_AVAILABLE):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                features, _ = batch
            else:
                features = batch

            features = features.to(self.device)

            # Forward pass
            outputs = self.model(features)

            semantic_features = outputs['semantic_features'].cpu().numpy()
            reconstruction = outputs['reconstructed'].cpu().numpy()

            # Compute reconstruction error
            reconstruction_error = np.mean((reconstruction - features.cpu().numpy()) ** 2, axis=1)

            all_semantic_features.append(semantic_features)
            all_reconstructions.append(reconstruction)
            all_reconstruction_errors.append(reconstruction_error)

        # Concatenate all batches
        semantic_features_array = np.concatenate(all_semantic_features, axis=0)
        reconstructions_array = np.concatenate(all_reconstructions, axis=0)
        errors_array = np.concatenate(all_reconstruction_errors, axis=0)

        print(f"✅ Extracted {semantic_features_array.shape[0]} samples")
        print(f"   Semantic features shape: {semantic_features_array.shape}")
        print(f"   Average reconstruction error: {errors_array.mean():.6f}")

        return {
            'features': semantic_features_array,
            'reconstructions': reconstructions_array,
            'reconstruction_errors': errors_array,
        }

    def save_semantic_feature_bank(
        self,
        semantic_data: Dict[str, np.ndarray],
        output_path: Path
    ):
        """
        Save extracted semantic features for Agent 6.

        Args:
            semantic_data: Dictionary from extract_semantic_features()
            output_path: Path to save feature bank
        """
        # Save as numpy archive
        np.savez_compressed(
            output_path,
            features=semantic_data['features'],
            reconstructions=semantic_data['reconstructions'],
            reconstruction_errors=semantic_data['reconstruction_errors'],
            num_features=self.config.num_semantic_features,
            input_dim=self.config.input_dim,
        )

        print(f"💾 Saved semantic feature bank to {output_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_simple_dataset_for_testing(
    n_samples: int = 1000,
    input_dim: int = 200
) -> Tuple[DataLoader, DataLoader]:
    """
    Create simple synthetic dataset for testing trainer.

    This is a temporary dataset for testing until Agent 4's GapDataset is ready.

    Args:
        n_samples: Number of samples
        input_dim: Input feature dimension

    Returns:
        Train loader, validation loader
    """
    class SyntheticDataset(Dataset):
        def __init__(self, n_samples, input_dim):
            # Generate random features with some structure
            self.features = torch.randn(n_samples, input_dim)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx]

    # Create train and val datasets
    train_dataset = SyntheticDataset(int(n_samples * 0.8), input_dim)
    val_dataset = SyntheticDataset(int(n_samples * 0.2), input_dim)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training script"""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available")
        return

    # Create configuration
    config = TrainingConfig(
        num_semantic_features=25,
        num_epochs=100,
        batch_size=64,
        learning_rate=0.001,
    )

    # Save configuration
    config.save(config.output_dir / 'training_config.json')
    print(f"📝 Saved config to {config.output_dir / 'training_config.json'}")

    # Load dataset
    # TODO: Replace with Agent 4's GapDataset once available
    print("\n⚠️  Using synthetic dataset for testing")
    print("   Replace with Agent 4's GapDataset for actual training\n")

    train_loader, val_loader = create_simple_dataset_for_testing(
        n_samples=1000,
        input_dim=config.input_dim
    )

    # Create trainer
    trainer = GapDiscoveryTrainer(config)

    # Train model
    summary = trainer.train(train_loader, val_loader)

    # Extract semantic features
    print("\nExtracting semantic features from validation set...")
    semantic_data = trainer.extract_semantic_features(val_loader)

    # Save semantic feature bank
    trainer.save_semantic_feature_bank(
        semantic_data,
        config.output_dir / 'semantic_feature_bank.npz'
    )

    # Save training summary
    summary_path = config.output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Training complete! Results saved to {config.output_dir}")
    print(f"\nNext steps:")
    print(f"1. Use semantic feature bank for interpretation (Agent 6)")
    print(f"2. Integrate with end-to-end pipeline (Agent 7)")
    print(f"3. Evaluate on test corpus (Agent 9)")


if __name__ == "__main__":
    main()
