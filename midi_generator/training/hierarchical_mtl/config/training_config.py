"""
Training Configuration for Hierarchical Multi-Task Learning.

This module defines all configuration classes for the hierarchical MTL training pipeline.

Author: Agent 06
Date: November 20, 2025
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Supported learning rate scheduler types."""
    COSINE = "cosine"
    STEP = "step"
    PLATEAU = "plateau"
    EXPONENTIAL = "exponential"
    NONE = "none"


class LossType(Enum):
    """Supported loss function types."""
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    CROSS_ENTROPY = "cross_entropy"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Adam/AdamW specific
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # SGD specific
    momentum: float = 0.9
    nesterov: bool = True

    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    clip_grad_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'optimizer_type': self.optimizer_type.value,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'eps': self.eps,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'clip_grad_norm': self.clip_grad_norm,
            'clip_grad_value': self.clip_grad_value,
        }


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    scheduler_type: SchedulerType = SchedulerType.COSINE

    # Cosine annealing
    T_max: int = 100
    eta_min: float = 1e-6

    # Step LR
    step_size: int = 30
    gamma: float = 0.1

    # ReduceLROnPlateau
    mode: str = "min"
    factor: float = 0.5
    patience: int = 10
    threshold: float = 1e-4
    cooldown: int = 0
    min_lr: float = 1e-7

    # Exponential LR
    decay_rate: float = 0.95

    # Warmup
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scheduler_type': self.scheduler_type.value,
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'mode': self.mode,
            'factor': self.factor,
            'patience': self.patience,
            'threshold': self.threshold,
            'cooldown': self.cooldown,
            'min_lr': self.min_lr,
            'decay_rate': self.decay_rate,
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
        }


@dataclass
class LossConfig:
    """Configuration for loss functions."""

    # Level-specific loss weights
    level1_weight: float = 1.0
    level2_weight: float = 1.0
    level3_weight: float = 1.0

    # Parameter-specific loss types (can be overridden)
    default_regression_loss: LossType = LossType.MSE
    default_classification_loss: LossType = LossType.CROSS_ENTROPY

    # Huber loss delta
    huber_delta: float = 1.0

    # Auxiliary losses
    use_regularization: bool = True
    l1_reg_weight: float = 0.0
    l2_reg_weight: float = 1e-4

    # Hierarchical consistency loss (optional)
    use_consistency_loss: bool = False
    consistency_weight: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level1_weight': self.level1_weight,
            'level2_weight': self.level2_weight,
            'level3_weight': self.level3_weight,
            'default_regression_loss': self.default_regression_loss.value,
            'default_classification_loss': self.default_classification_loss.value,
            'huber_delta': self.huber_delta,
            'use_regularization': self.use_regularization,
            'l1_reg_weight': self.l1_reg_weight,
            'l2_reg_weight': self.l2_reg_weight,
            'use_consistency_loss': self.use_consistency_loss,
            'consistency_weight': self.consistency_weight,
        }


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Paths
    corpus_dir: Path = Path("midi_corpus")
    labeled_dataset_path: Path = Path("labeled_dataset.json")
    features_dir: Optional[Path] = None

    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    stratify_by_genre: bool = True

    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.3
    noise_std: float = 0.01

    # Feature normalization
    normalize_features: bool = True
    normalization_method: str = "standardize"  # "standardize" or "minmax"

    # Genre handling
    supported_genres: List[str] = field(default_factory=lambda: [
        'jazz', 'classical', 'rock', 'electronic', 'pop', 'hip-hop', 'latin'
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'corpus_dir': str(self.corpus_dir),
            'labeled_dataset_path': str(self.labeled_dataset_path),
            'features_dir': str(self.features_dir) if self.features_dir else None,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'stratify_by_genre': self.stratify_by_genre,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'prefetch_factor': self.prefetch_factor,
            'use_augmentation': self.use_augmentation,
            'augmentation_prob': self.augmentation_prob,
            'noise_std': self.noise_std,
            'normalize_features': self.normalize_features,
            'normalization_method': self.normalization_method,
            'supported_genres': self.supported_genres,
        }


@dataclass
class HierarchicalMTLConfig:
    """Main configuration for hierarchical MTL training."""

    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Training settings
    num_epochs: int = 100
    early_stopping_patience: int = 15
    min_delta: float = 1e-4

    # Model architecture (will be used by Agent 05)
    shared_encoder_dim: int = 512
    level1_hidden_dim: int = 256
    level2_hidden_dim: int = 256
    level3_hidden_dim: int = 128
    dropout_rate: float = 0.1

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every_n_epochs: int = 5
    save_best_only: bool = True
    keep_n_checkpoints: int = 3

    # Logging
    log_dir: Path = Path("logs")
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1

    # Experiment tracking
    use_wandb: bool = False
    use_mlflow: bool = False
    experiment_name: str = "hierarchical_mtl"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Mixed precision training
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0
    backend: str = "nccl"  # "nccl" or "gloo"

    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True

    # Device
    device: str = "cuda"  # "cuda", "cpu", or "mps"

    # Validation
    validate_on_start: bool = True
    compute_train_metrics: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary."""
        return {
            'optimizer': self.optimizer.to_dict(),
            'scheduler': self.scheduler.to_dict(),
            'loss': self.loss.to_dict(),
            'data': self.data.to_dict(),
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'min_delta': self.min_delta,
            'shared_encoder_dim': self.shared_encoder_dim,
            'level1_hidden_dim': self.level1_hidden_dim,
            'level2_hidden_dim': self.level2_hidden_dim,
            'level3_hidden_dim': self.level3_hidden_dim,
            'dropout_rate': self.dropout_rate,
            'checkpoint_dir': str(self.checkpoint_dir),
            'save_every_n_epochs': self.save_every_n_epochs,
            'save_best_only': self.save_best_only,
            'keep_n_checkpoints': self.keep_n_checkpoints,
            'log_dir': str(self.log_dir),
            'log_every_n_steps': self.log_every_n_steps,
            'eval_every_n_epochs': self.eval_every_n_epochs,
            'use_wandb': self.use_wandb,
            'use_mlflow': self.use_mlflow,
            'experiment_name': self.experiment_name,
            'run_name': self.run_name,
            'tags': self.tags,
            'use_amp': self.use_amp,
            'amp_dtype': self.amp_dtype,
            'distributed': self.distributed,
            'world_size': self.world_size,
            'local_rank': self.local_rank,
            'backend': self.backend,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'benchmark': self.benchmark,
            'device': self.device,
            'validate_on_start': self.validate_on_start,
            'compute_train_metrics': self.compute_train_metrics,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HierarchicalMTLConfig':
        """Create config from dictionary."""
        # Extract sub-configs
        optimizer_dict = config_dict.pop('optimizer', {})
        scheduler_dict = config_dict.pop('scheduler', {})
        loss_dict = config_dict.pop('loss', {})
        data_dict = config_dict.pop('data', {})

        # Create sub-configs
        optimizer = OptimizerConfig(**optimizer_dict) if optimizer_dict else OptimizerConfig()
        scheduler = SchedulerConfig(**scheduler_dict) if scheduler_dict else SchedulerConfig()
        loss = LossConfig(**loss_dict) if loss_dict else LossConfig()
        data = DataConfig(**data_dict) if data_dict else DataConfig()

        # Create main config
        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            data=data,
            **config_dict
        )

    def save(self, path: Path):
        """Save config to JSON file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'HierarchicalMTLConfig':
        """Load config from JSON or YAML file."""
        import json
        path = Path(path)

        # Determine file type
        if path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            # Default to JSON
            with open(path, 'r') as f:
                config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def save_yaml(self, path: Path):
        """Save config to YAML file."""
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @classmethod
    def load_yaml(cls, path: Path) -> 'HierarchicalMTLConfig':
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


# Preset configurations for common scenarios
def get_default_config() -> HierarchicalMTLConfig:
    """Get default configuration for hierarchical MTL training."""
    return HierarchicalMTLConfig()


def get_fast_config() -> HierarchicalMTLConfig:
    """Get configuration for fast experimentation (smaller model, less epochs)."""
    config = HierarchicalMTLConfig()
    config.num_epochs = 30
    config.shared_encoder_dim = 256
    config.level1_hidden_dim = 128
    config.level2_hidden_dim = 128
    config.level3_hidden_dim = 64
    config.data.batch_size = 64
    return config


def get_production_config() -> HierarchicalMTLConfig:
    """Get configuration for production training (larger model, more epochs)."""
    config = HierarchicalMTLConfig()
    config.num_epochs = 200
    config.early_stopping_patience = 25
    config.shared_encoder_dim = 1024
    config.level1_hidden_dim = 512
    config.level2_hidden_dim = 512
    config.level3_hidden_dim = 256
    config.optimizer.learning_rate = 5e-4
    config.data.batch_size = 16
    config.use_amp = True
    return config
