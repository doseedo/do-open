"""
AGENT 4: Model Configuration Management
=========================================

Configuration classes for ScaledHierarchicalMTL model and training.

Provides:
    1. ModelConfig: Architecture configuration
    2. LossConfig: Loss function configuration
    3. TrainingConfig: Training hyperparameters
    4. Config serialization/deserialization (JSON/YAML)
    5. Config validation
    6. Default configurations

Author: Agent 4 - Model Architecture Engineer
Date: November 21, 2025
Version: 1.0.0
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for ScaledHierarchicalMTL model architecture.
    """
    # Input/Output dimensions
    input_dim: int = 600
    shared_dim: int = 768
    output_hierarchical: int = 50  # 8 + 20 + 22
    output_modular: int = 120  # 30 + 20 + 15 + 25 + 20 + 10
    output_rich: int = 130  # 80 + 40 + 10

    # Encoder configuration
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [1024, 1024, 768])
    use_attention: bool = True
    num_attention_heads: int = 8
    dropout: float = 0.3

    # Conditioning
    conditioning_dim: int = 32
    num_genres: int = 7

    # Head configurations
    level_head_hidden_dim: int = 128
    modular_head_hidden_dim: int = 256
    rich_head_hidden_dim: int = 256

    # Per-track and temporal settings
    num_tracks: int = 8
    params_per_track: int = 10
    num_sections: int = 4
    params_per_section: int = 10

    def validate(self) -> bool:
        """Validate configuration"""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.shared_dim > 0, "shared_dim must be positive"
        assert len(self.encoder_hidden_dims) >= 2, "Need at least 2 hidden layers"
        assert self.encoder_hidden_dims[-1] == self.shared_dim, \
            f"Last encoder dim ({self.encoder_hidden_dims[-1]}) must equal shared_dim ({self.shared_dim})"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary"""
        return cls(**config_dict)

    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModelConfig':
        """Load configuration from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class LossConfig:
    """
    Configuration for loss function.
    """
    # Hierarchical level weights
    level1_weight: float = 3.0
    level2_weight: float = 2.0
    level3_weight: float = 1.5

    # Category weights
    hierarchical_weight: float = 2.0
    modular_weight: float = 1.5
    rich_weight: float = 1.0

    # Loss strategy
    use_uncertainty_weighting: bool = True
    use_gradient_balancing: bool = False
    mse_reduction: str = 'mean'

    def validate(self) -> bool:
        """Validate configuration"""
        assert self.level1_weight >= 0, "level1_weight must be non-negative"
        assert self.level2_weight >= 0, "level2_weight must be non-negative"
        assert self.level3_weight >= 0, "level3_weight must be non-negative"
        assert self.mse_reduction in ['mean', 'sum'], "mse_reduction must be 'mean' or 'sum'"
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LossConfig':
        """Create from dictionary"""
        return cls(**config_dict)

    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LossConfig':
        """Load configuration from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer.
    """
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam/AdamW
    eps: float = 1e-8

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau', 'warmup_cosine'
    warmup_steps: int = 500
    warmup_start_lr: float = 1e-6

    # For step scheduler
    step_size: int = 10
    gamma: float = 0.1

    # For plateau scheduler
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6

    def validate(self) -> bool:
        """Validate configuration"""
        assert self.optimizer_type in ['adam', 'adamw', 'sgd'], \
            "optimizer_type must be 'adam', 'adamw', or 'sgd'"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.scheduler_type in ['cosine', 'step', 'plateau', 'warmup_cosine'], \
            "Invalid scheduler_type"
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        # Convert tuple to list for JSON serialization
        d['betas'] = list(d['betas'])
        return d

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizerConfig':
        """Create from dictionary"""
        # Convert list back to tuple
        if 'betas' in config_dict and isinstance(config_dict['betas'], list):
            config_dict['betas'] = tuple(config_dict['betas'])
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """
    Configuration for training process.
    """
    # Data
    data_dir: str = "data/labeled_dataset_comprehensive.json"
    output_dir: str = "models/scaled_hierarchical_mtl"

    # Training
    batch_size: int = 128
    num_epochs: int = 200
    gradient_accumulation_steps: int = 4  # Effective batch = 512

    # Optimization
    gradient_clip_value: float = 1.0

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Checkpointing
    save_frequency: int = 5  # Save every N epochs
    keep_best_n: int = 3  # Keep best N checkpoints

    # Logging
    log_frequency: int = 10  # Log every N batches
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "midi-generator-scaled-mtl"
    wandb_entity: Optional[str] = None

    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True

    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision

    # Data split
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Random seed
    seed: int = 42

    def validate(self) -> bool:
        """Validate configuration"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.train_split + self.val_split + self.test_split == 1.0, \
            "Data splits must sum to 1.0"
        assert self.device in ['auto', 'cpu'] or self.device.startswith('cuda'), \
            "Invalid device"
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**config_dict)

    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class FullConfig:
    """
    Complete configuration combining all sub-configs.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> bool:
        """Validate all sub-configurations"""
        return (
            self.model.validate() and
            self.loss.validate() and
            self.optimizer.validate() and
            self.training.validate()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model': self.model.to_dict(),
            'loss': self.loss.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'training': self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FullConfig':
        """Create from dictionary"""
        return cls(
            model=ModelConfig.from_dict(config_dict.get('model', {})),
            loss=LossConfig.from_dict(config_dict.get('loss', {})),
            optimizer=OptimizerConfig.from_dict(config_dict.get('optimizer', {})),
            training=TrainingConfig.from_dict(config_dict.get('training', {})),
        )

    def save(self, path: Union[str, Path]):
        """Save complete configuration to JSON file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FullConfig':
        """Load complete configuration from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)

        print("\nMODEL:")
        print(f"  Input dim: {self.model.input_dim}")
        print(f"  Shared dim: {self.model.shared_dim}")
        print(f"  Encoder: {' → '.join(map(str, self.model.encoder_hidden_dims))}")
        print(f"  Attention: {self.model.use_attention} ({self.model.num_attention_heads} heads)")
        print(f"  Dropout: {self.model.dropout}")

        print("\nOUTPUT PARAMETERS:")
        print(f"  Hierarchical: {self.model.output_hierarchical}")
        print(f"  Modular: {self.model.output_modular}")
        print(f"  Rich: {self.model.output_rich}")
        print(f"  Total: {self.model.output_hierarchical + self.model.output_modular + self.model.output_rich}")

        print("\nLOSS:")
        print(f"  Hierarchical weight: {self.loss.hierarchical_weight}")
        print(f"  Modular weight: {self.loss.modular_weight}")
        print(f"  Rich weight: {self.loss.rich_weight}")
        print(f"  Uncertainty weighting: {self.loss.use_uncertainty_weighting}")
        print(f"  Gradient balancing: {self.loss.use_gradient_balancing}")

        print("\nOPTIMIZER:")
        print(f"  Type: {self.optimizer.optimizer_type}")
        print(f"  Learning rate: {self.optimizer.learning_rate}")
        print(f"  Weight decay: {self.optimizer.weight_decay}")
        print(f"  Scheduler: {self.optimizer.scheduler_type if self.optimizer.use_scheduler else 'None'}")

        print("\nTRAINING:")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Gradient accumulation: {self.training.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.training.batch_size * self.training.gradient_accumulation_steps}")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Early stopping: {self.training.early_stopping} (patience: {self.training.early_stopping_patience})")
        print(f"  Mixed precision: {self.training.use_amp}")
        print(f"  Device: {self.training.device}")

        print("="*70 + "\n")


# ============================================================================
# Preset Configurations
# ============================================================================

def get_default_config() -> FullConfig:
    """Get default configuration"""
    return FullConfig()


def get_fast_config() -> FullConfig:
    """Get configuration for fast training (smaller model, fewer epochs)"""
    config = FullConfig()

    # Smaller model
    config.model.encoder_hidden_dims = [512, 512, 384]
    config.model.shared_dim = 384
    config.model.dropout = 0.2

    # Fewer epochs
    config.training.num_epochs = 50
    config.training.batch_size = 64

    # Less frequent checkpointing
    config.training.save_frequency = 10

    return config


def get_large_config() -> FullConfig:
    """Get configuration for large-scale training"""
    config = FullConfig()

    # Larger model
    config.model.encoder_hidden_dims = [1536, 1536, 1024]
    config.model.shared_dim = 1024
    config.model.num_attention_heads = 16
    config.model.dropout = 0.4

    # More training
    config.training.num_epochs = 300
    config.training.batch_size = 256
    config.training.gradient_accumulation_steps = 8

    # Slower learning rate
    config.optimizer.learning_rate = 5e-4

    return config


def get_gpu_optimized_config() -> FullConfig:
    """Get configuration optimized for multi-GPU training"""
    config = FullConfig()

    # Standard model size
    config.model.encoder_hidden_dims = [1024, 1024, 768]

    # Large batch for multi-GPU
    config.training.batch_size = 256
    config.training.gradient_accumulation_steps = 2
    config.training.num_workers = 8
    config.training.pin_memory = True
    config.training.use_amp = True

    return config


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Model Configuration - Agent 4")
    print("="*70)

    # Create default config
    print("\n1. Creating default configuration...")
    config = get_default_config()
    config.validate()
    print("   ✅ Configuration created and validated")

    # Print summary
    config.print_summary()

    # Save configuration
    print("2. Testing save/load...")
    save_path = Path("/tmp/test_config.json")
    config.save(save_path)
    print(f"   ✅ Saved to {save_path}")

    # Load configuration
    loaded_config = FullConfig.load(save_path)
    loaded_config.validate()
    print(f"   ✅ Loaded from {save_path}")

    # Test other presets
    print("\n3. Testing preset configurations...")
    fast_config = get_fast_config()
    fast_config.validate()
    print("   ✅ Fast config validated")

    large_config = get_large_config()
    large_config.validate()
    print("   ✅ Large config validated")

    gpu_config = get_gpu_optimized_config()
    gpu_config.validate()
    print("   ✅ GPU-optimized config validated")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
