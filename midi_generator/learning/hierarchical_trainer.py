"""
AGENT 05: Hierarchical MTL Training Pipeline
=============================================

End-to-end training pipeline for the Hierarchical Multi-Task Learning model.

Features:
    - Data loading and preprocessing
    - Training loop with early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Metrics tracking and visualization
    - TensorBoard/Wandb integration
    - Distributed training support (optional)

Author: Agent 05 - Hierarchical MTL Architect
License: MIT
Date: November 20, 2025
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
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

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

# Import our MTL modules
try:
    from midi_generator.learning.hierarchical_mtl import (
        HierarchicalMTLModel,
        HierarchicalMTLLoss,
        MIDIParameterDataset,
        MTLConfig,
        create_model,
        create_loss_function,
        ALL_PARAMETERS,
        LEVEL_1_PARAMETERS,
        LEVEL_2_PARAMETERS,
        LEVEL_3_PARAMETERS,
    )
except ImportError:
    print("WARNING: Could not import hierarchical_mtl module")


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training"""

    # Paths
    data_dir: Path = Path('midi_corpus/labeled_dataset')
    output_dir: Path = Path('midi_generator/models/hierarchical_mtl')
    checkpoint_dir: Path = Path('midi_generator/models/hierarchical_mtl/checkpoints')
    log_dir: Path = Path('midi_generator/models/hierarchical_mtl/logs')

    # Model configuration
    model_config: MTLConfig = field(default_factory=MTLConfig)

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Learning rate schedule
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    lr_patience: int = 5  # for plateau scheduler
    lr_factor: float = 0.5  # for step/plateau schedulers

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Optimization
    optimizer_type: str = 'adam'  # 'adam', 'adamw', 'sgd'
    gradient_clip_value: Optional[float] = 1.0

    # Weight sparsity for superposition reduction (OFF by default)
    enable_weight_sparsity: bool = False  # Enable weight sparsity during training
    sparsity_ratio: float = 0.001  # Target sparsity ratio (0.001 = 0.1% of weights kept)
    initial_sparsity: float = 0.5  # Initial sparsity ratio at start of training
    sparsity_warmup_epochs: int = 50  # Number of epochs to gradually increase sparsity

    # Checkpointing
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3

    # Logging
    log_every_n_steps: int = 10
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: str = 'midi-generator-mtl'
    wandb_entity: Optional[str] = None

    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'

    # Reproducibility
    random_seed: int = 42

    def __post_init__(self):
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if self.device == 'auto':
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
            else:
                self.device = 'cpu'

    def save(self, path: Path):
        """Save configuration to JSON"""
        # Convert to dict
        config_dict = {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'checkpoint_dir': str(self.checkpoint_dir),
            'log_dir': str(self.log_dir),
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'use_lr_scheduler': self.use_lr_scheduler,
            'lr_scheduler_type': self.lr_scheduler_type,
            'lr_patience': self.lr_patience,
            'lr_factor': self.lr_factor,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'optimizer_type': self.optimizer_type,
            'gradient_clip_value': self.gradient_clip_value,
            'save_every_n_epochs': self.save_every_n_epochs,
            'keep_n_checkpoints': self.keep_n_checkpoints,
            'log_every_n_steps': self.log_every_n_steps,
            'use_tensorboard': self.use_tensorboard,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'device': self.device,
            'random_seed': self.random_seed,
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Convert paths back
        config_dict['data_dir'] = Path(config_dict['data_dir'])
        config_dict['output_dir'] = Path(config_dict['output_dir'])
        config_dict['checkpoint_dir'] = Path(config_dict['checkpoint_dir'])
        config_dict['log_dir'] = Path(config_dict['log_dir'])

        return cls(**config_dict)


# ============================================================================
# Metrics Tracking
# ============================================================================

@dataclass
class TrainingMetrics:
    """Metrics for one epoch"""
    epoch: int
    train_loss: float
    val_loss: float
    train_losses_per_param: Dict[str, float] = field(default_factory=dict)
    val_losses_per_param: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    weight_sparsity_ratio: float = 1.0  # Actual sparsity ratio achieved (1.0 = all weights non-zero)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsTracker:
    """Track and visualize training metrics"""

    def __init__(self, log_dir: Path, use_tensorboard: bool = False, use_wandb: bool = False):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        self.history: List[TrainingMetrics] = []

        # TensorBoard writer
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir / 'tensorboard')
            except ImportError:
                print("WARNING: TensorBoard not available")
                self.use_tensorboard = False

        # Wandb initialization
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("WARNING: Wandb not available")
                self.use_wandb = False

    def log_metrics(self, metrics: TrainingMetrics):
        """Log metrics for an epoch"""
        self.history.append(metrics)

        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalar('Loss/train', metrics.train_loss, metrics.epoch)
            self.tb_writer.add_scalar('Loss/val', metrics.val_loss, metrics.epoch)
            self.tb_writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch)
            self.tb_writer.add_scalar('Sparsity/weight_sparsity_ratio', metrics.weight_sparsity_ratio, metrics.epoch)

            # Per-parameter losses
            for param_name, loss in metrics.train_losses_per_param.items():
                self.tb_writer.add_scalar(f'Train_Loss/{param_name}', loss, metrics.epoch)
            for param_name, loss in metrics.val_losses_per_param.items():
                self.tb_writer.add_scalar(f'Val_Loss/{param_name}', loss, metrics.epoch)

        # Log to Wandb
        if self.use_wandb:
            log_dict = {
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'learning_rate': metrics.learning_rate,
                'epoch_time': metrics.epoch_time,
                'weight_sparsity_ratio': metrics.weight_sparsity_ratio,
            }
            log_dict.update({f'train_{k}': v for k, v in metrics.train_losses_per_param.items()})
            log_dict.update({f'val_{k}': v for k, v in metrics.val_losses_per_param.items()})
            self.wandb.log(log_dict)

    def save_history(self, path: Path):
        """Save metrics history to JSON"""
        history_data = []
        for metrics in self.history:
            history_data.append({
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'learning_rate': metrics.learning_rate,
                'epoch_time': metrics.epoch_time,
                'timestamp': metrics.timestamp,
            })

        with open(path, 'w') as f:
            json.dump(history_data, f, indent=2)

    def get_best_epoch(self) -> Optional[TrainingMetrics]:
        """Get epoch with best validation loss"""
        if not self.history:
            return None
        return min(self.history, key=lambda m: m.val_loss)

    def close(self):
        """Close writers"""
        if self.use_tensorboard and hasattr(self, 'tb_writer'):
            self.tb_writer.close()


# ============================================================================
# Trainer
# ============================================================================

class HierarchicalMTLTrainer:
    """
    Trainer for Hierarchical Multi-Task Learning model.

    Handles:
        - Data loading and preprocessing
        - Training loop with validation
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        - Metrics tracking
    """

    def __init__(self,
                 config: TrainingConfig,
                 model: Optional[HierarchicalMTLModel] = None,
                 loss_fn: Optional[HierarchicalMTLLoss] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            model: Pre-initialized model (creates new if None)
            loss_fn: Loss function (creates new if None)
        """
        if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
            raise RuntimeError("PyTorch and NumPy are required for training")

        self.config = config
        self.device = torch.device(config.device)

        # Set random seed
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Create or use provided model
        if model is None:
            self.model = create_model(config.model_config)
        else:
            self.model = model

        self.model.to(self.device)

        # Create or use provided loss function
        if loss_fn is None:
            self.loss_fn = create_loss_function(config.model_config)
        else:
            self.loss_fn = loss_fn

        self.loss_fn.to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler() if config.use_lr_scheduler else None

        # Metrics tracker
        self.metrics_tracker = MetricsTracker(
            config.log_dir,
            use_tensorboard=config.use_tensorboard,
            use_wandb=config.use_wandb
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Parameter definitions
        self.param_defs = {p.name: p for p in ALL_PARAMETERS}

        # Weight sparsity for superposition reduction
        self.sparsity_scheduler = None
        if config.enable_weight_sparsity:
            try:
                from midi_generator.learning.semantic_encoder import SparsityScheduler
                self.sparsity_scheduler = SparsityScheduler(
                    initial_sparsity=config.initial_sparsity,
                    target_sparsity=config.sparsity_ratio,
                    warmup_epochs=config.sparsity_warmup_epochs,
                    schedule_type='linear'
                )
                print(f"✅ Weight sparsity enabled: {config.sparsity_ratio*100:.2f}% of weights kept (target)")
            except ImportError:
                print("⚠️  Warning: SparsityScheduler not available, weight sparsity disabled")
                config.enable_weight_sparsity = False

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config.optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

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
                step_size=10,
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
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")

    def _apply_weight_sparsity(self, sparsity_ratio: float):
        """
        Apply Top-K weight sparsity to all Linear layers in the model.

        Args:
            sparsity_ratio: Proportion of weights to keep (e.g., 0.001 = keep 0.1% of weights)
        """
        if not self.config.enable_weight_sparsity:
            return

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # Get weight magnitudes
                    weight_abs = torch.abs(module.weight.data)

                    # Calculate number of weights to keep
                    total_params = weight_abs.numel()
                    k = max(1, int(total_params * sparsity_ratio))  # Keep at least 1 weight

                    # Find threshold (k-th largest magnitude)
                    threshold = torch.topk(weight_abs.flatten(), k).values[-1]

                    # Create and apply mask
                    mask = weight_abs >= threshold
                    module.weight.data *= mask.float()

    def _compute_weight_sparsity_ratio(self) -> float:
        """
        Compute the actual sparsity ratio of model weights.

        Returns:
            Proportion of non-zero weights (0.0 = all zeros, 1.0 = all non-zero)
        """
        total_params = 0
        nonzero_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    weight_abs = torch.abs(module.weight.data)
                    total_params += weight_abs.numel()
                    # Count as non-zero if magnitude > 1e-8
                    nonzero_params += (weight_abs > 1e-8).sum().item()

        if total_params == 0:
            return 1.0

        return nonzero_params / total_params

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss, per-parameter losses
        """
        self.model.train()

        total_loss = 0.0
        param_losses = {p.name: 0.0 for p in ALL_PARAMETERS}
        n_batches = 0

        iterator = tqdm(train_loader, desc=f"Epoch {self.current_epoch}") if TQDM_AVAILABLE else train_loader

        for batch_idx, (features, labels) in enumerate(iterator):
            # Move to device
            features = features.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            predictions = self.model(features)

            # Compute loss
            loss, losses_dict = self.loss_fn(predictions, labels, self.param_defs)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            for param_name, param_loss in losses_dict.items():
                param_losses[param_name] += param_loss

            n_batches += 1

            # Update progress bar
            if TQDM_AVAILABLE and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': loss.item()})

        # Average losses
        avg_loss = total_loss / n_batches
        avg_param_losses = {k: v / n_batches for k, v in param_losses.items()}

        return avg_loss, avg_param_losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss, per-parameter losses
        """
        self.model.eval()

        total_loss = 0.0
        param_losses = {p.name: 0.0 for p in ALL_PARAMETERS}
        n_batches = 0

        for features, labels in val_loader:
            # Move to device
            features = features.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass
            predictions = self.model(features)

            # Compute loss
            loss, losses_dict = self.loss_fn(predictions, labels, self.param_defs)

            # Track metrics
            total_loss += loss.item()
            for param_name, param_loss in losses_dict.items():
                param_losses[param_name] += param_loss

            n_batches += 1

        # Average losses
        avg_loss = total_loss / n_batches
        avg_param_losses = {k: v / n_batches for k, v in param_losses.items()}

        return avg_loss, avg_param_losses

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              resume_from_checkpoint: Optional[Path] = None) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training summary
        """
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)

        print("\n" + "="*70)
        print("STARTING HIERARCHICAL MTL TRAINING")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Optimizer: {self.config.optimizer_type}")
        print("="*70 + "\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Apply weight sparsity (if enabled)
            if self.config.enable_weight_sparsity and self.sparsity_scheduler is not None:
                current_sparsity = self.sparsity_scheduler.get_sparsity(epoch)
                self._apply_weight_sparsity(current_sparsity)

            # Train
            train_loss, train_param_losses = self.train_epoch(train_loader)

            # Validate
            val_loss, val_param_losses = self.validate(val_loader)

            epoch_time = time.time() - epoch_start

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Compute weight sparsity ratio (if enabled)
            weight_sparsity_ratio = 1.0  # Default: all weights non-zero
            if self.config.enable_weight_sparsity:
                weight_sparsity_ratio = self._compute_weight_sparsity_ratio()

            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_losses_per_param=train_param_losses,
                val_losses_per_param=val_param_losses,
                learning_rate=current_lr,
                epoch_time=epoch_time,
                weight_sparsity_ratio=weight_sparsity_ratio
            )

            # Log metrics
            self.metrics_tracker.log_metrics(metrics)

            # Print progress
            sparsity_str = f" | Sparsity: {weight_sparsity_ratio*100:.2f}%" if self.config.enable_weight_sparsity else ""
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}{sparsity_str} | "
                  f"Time: {epoch_time:.2f}s")

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
                print(f"  → New best model! (Val Loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Regular checkpointing
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(
                    self.config.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                )

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                break

        total_time = time.time() - start_time

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best epoch: {self.metrics_tracker.get_best_epoch().epoch}")
        print("="*70 + "\n")

        # Save metrics history
        self.metrics_tracker.save_history(self.config.log_dir / 'training_history.json')
        self.metrics_tracker.close()

        # Save final model
        self.save_checkpoint(self.config.checkpoint_dir / 'final_model.pt')

        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'best_epoch': self.metrics_tracker.get_best_epoch().epoch,
        }

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'config': self.config.__dict__,
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"  → Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']

        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_dataset(data_dir: Path, config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and split dataset into train/val/test loaders.

    Args:
        data_dir: Directory containing labeled dataset
        config: Training configuration

    Returns:
        Train, validation, and test data loaders
    """
    # Load features and labels
    # This assumes data is stored in numpy format
    features_path = data_dir / 'features.npy'
    labels_path = data_dir / 'labels.json'

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Dataset not found in {data_dir}")

    # Load data
    features = np.load(features_path)

    with open(labels_path, 'r') as f:
        labels_data = json.load(f)

    # Create param_defs dict
    param_defs = {p.name: p for p in ALL_PARAMETERS}

    # Convert labels to numpy arrays
    labels = {}
    for param_name in labels_data:
        if param_name in param_defs:
            labels[param_name] = np.array(labels_data[param_name])

    # Create dataset
    dataset = MIDIParameterDataset(features, labels, param_defs)

    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * config.train_split)
    n_val = int(n_total * config.val_split)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.random_seed)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to > 0 for multiprocessing
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Features: {features.shape[1]}")
    print(f"  Parameters: {len(labels)}\n")

    return train_loader, val_loader, test_loader


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training script"""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available")
        return

    # Create configuration
    config = TrainingConfig()

    # Save configuration
    config.save(config.output_dir / 'training_config.json')

    # Load dataset
    try:
        train_loader, val_loader, test_loader = load_dataset(config.data_dir, config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please prepare the dataset first using Agent 03")
        return

    # Create trainer
    trainer = HierarchicalMTLTrainer(config)

    # Train model
    summary = trainer.train(train_loader, val_loader)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_param_losses = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # Save test results
    with open(config.log_dir / 'test_results.json', 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'test_param_losses': test_param_losses,
            'training_summary': summary,
        }, f, indent=2)

    print("\nTraining complete! Model saved to:", config.checkpoint_dir)


if __name__ == "__main__":
    main()
