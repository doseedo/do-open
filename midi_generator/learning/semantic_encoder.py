"""
Semantic Feature Encoder - Agent 3
===================================

Neural architecture for discovering semantic musical parameters through reconstruction gaps.

This encoder learns to:
1. Compress 200D feature vectors into semantic features
2. Reconstruct the original features from semantic features
3. Predict locality transformations (from Agent 1)
4. Discover interpretable musical parameters

Architecture:
    Input: 200D feature vector (from OptimizedFeatureExtractor)
    Encoder: [200] → [512] → [num_features]
    Decoder: [num_features] → [512] → [200]
    Locality Predictor: [num_features * 2] → [512] → [num_locality_types]

Loss Components:
    - Reconstruction loss: MSE between input and reconstructed features
    - Locality loss: Cross-entropy for predicting locality transformations
    - Sparsity loss: L1 regularization on semantic features

Author: Agent 3 - Neural Architecture
Date: November 21, 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import warnings

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Neural network functionality will be disabled.")

# NumPy is required
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not installed. Some functionality will be disabled.")


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class EncoderConfig:
    """Configuration for SemanticFeatureEncoder"""

    # Architecture dimensions
    input_dim: int = 200  # Input feature dimension
    hidden_dim: int = 512  # Hidden layer dimension
    num_semantic_features: int = 30  # Number of semantic features to discover
    num_locality_types: int = 12  # Number of musical locality transformations

    # Loss weights
    reconstruction_weight: float = 1.0  # Weight for reconstruction loss
    locality_weight: float = 0.5  # Weight for locality prediction loss
    sparsity_weight: float = 0.01  # Weight for L1 sparsity regularization

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    dropout: float = 0.1

    # Regularization
    weight_decay: float = 1e-5

    # Feature constraints
    feature_activation: str = "relu"  # Activation for semantic features: relu, sigmoid, tanh
    normalize_features: bool = True  # Normalize semantic features

    # Advanced options
    use_batch_norm: bool = True
    residual_connections: bool = False

    # Weight sparsity for superposition reduction (OFF by default)
    enable_weight_sparsity: bool = False  # Enable weight sparsity during training
    sparsity_ratio: float = 0.001  # Target sparsity ratio (0.001 = 0.1% of weights kept)
    initial_sparsity: float = 0.5  # Initial sparsity ratio at start of training
    target_sparsity: float = 0.001  # Target sparsity ratio after warmup
    sparsity_warmup_epochs: int = 50  # Number of epochs to gradually increase sparsity
    sparsity_method: str = "topk"  # Sparsity method: 'topk' (magnitude-based) or 'threshold'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_semantic_features': self.num_semantic_features,
            'num_locality_types': self.num_locality_types,
            'reconstruction_weight': self.reconstruction_weight,
            'locality_weight': self.locality_weight,
            'sparsity_weight': self.sparsity_weight,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
            'feature_activation': self.feature_activation,
            'normalize_features': self.normalize_features,
            'use_batch_norm': self.use_batch_norm,
            'residual_connections': self.residual_connections,
            'enable_weight_sparsity': self.enable_weight_sparsity,
            'sparsity_ratio': self.sparsity_ratio,
            'initial_sparsity': self.initial_sparsity,
            'target_sparsity': self.target_sparsity,
            'sparsity_warmup_epochs': self.sparsity_warmup_epochs,
            'sparsity_method': self.sparsity_method
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EncoderConfig':
        """Create from dictionary"""
        return cls(**config_dict)

    def save(self, path: Path):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'EncoderConfig':
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    epoch: int
    total_loss: float
    reconstruction_loss: float
    locality_loss: float
    sparsity_loss: float
    locality_accuracy: float = 0.0
    weight_sparsity_ratio: float = 0.0  # Actual sparsity ratio achieved

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'epoch': self.epoch,
            'total_loss': self.total_loss,
            'reconstruction_loss': self.reconstruction_loss,
            'locality_loss': self.locality_loss,
            'sparsity_loss': self.sparsity_loss,
            'locality_accuracy': self.locality_accuracy,
            'weight_sparsity_ratio': self.weight_sparsity_ratio
        }


class SparsityScheduler:
    """
    Gradual sparsity annealing scheduler for weight sparsity.

    Gradually increases sparsity (reduces the ratio of kept weights) during training
    to help the model learn monosemantic features and reduce superposition.

    Args:
        initial_sparsity: Starting sparsity ratio (e.g., 0.5 = keep 50% of weights)
        target_sparsity: Target sparsity ratio (e.g., 0.001 = keep 0.1% of weights)
        warmup_epochs: Number of epochs to gradually transition from initial to target
        schedule_type: Type of annealing schedule ('linear', 'cosine', 'exponential')

    Example:
        scheduler = SparsityScheduler(initial_sparsity=0.5, target_sparsity=0.001, warmup_epochs=50)
        for epoch in range(100):
            current_sparsity = scheduler.get_sparsity(epoch)
            # Apply sparsity to model...
    """

    def __init__(
        self,
        initial_sparsity: float = 0.5,
        target_sparsity: float = 0.001,
        warmup_epochs: int = 50,
        schedule_type: str = 'linear'
    ):
        self.initial = initial_sparsity
        self.target = target_sparsity
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type

        # Validate inputs
        if not (0 < initial_sparsity <= 1.0):
            raise ValueError(f"initial_sparsity must be in (0, 1], got {initial_sparsity}")
        if not (0 < target_sparsity <= initial_sparsity):
            raise ValueError(f"target_sparsity must be in (0, initial_sparsity], got {target_sparsity}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")

    def get_sparsity(self, epoch: int) -> float:
        """
        Get current sparsity ratio for the given epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            Current sparsity ratio (proportion of weights to keep)
        """
        # After warmup, use target sparsity
        if epoch >= self.warmup_epochs:
            return self.target

        # During warmup, interpolate based on schedule type
        progress = epoch / self.warmup_epochs  # 0.0 to 1.0

        if self.schedule_type == 'linear':
            # Linear interpolation
            current_sparsity = self.initial * (1 - progress) + self.target * progress

        elif self.schedule_type == 'cosine':
            # Cosine annealing (smooth transition)
            import math
            cosine_progress = (1 - math.cos(progress * math.pi)) / 2
            current_sparsity = self.initial * (1 - cosine_progress) + self.target * cosine_progress

        elif self.schedule_type == 'exponential':
            # Exponential decay (faster at start, slower at end)
            import math
            # Solve for decay rate: initial * exp(-rate * warmup) = target
            decay_rate = -math.log(self.target / self.initial) / self.warmup_epochs
            current_sparsity = self.initial * math.exp(-decay_rate * epoch)

        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        # Ensure we don't go below target
        return max(current_sparsity, self.target)

    def __repr__(self) -> str:
        return (f"SparsityScheduler(initial={self.initial}, target={self.target}, "
                f"warmup_epochs={self.warmup_epochs}, schedule='{self.schedule_type}')")


# ============================================================================
# Neural Network Components
# ============================================================================

if not TORCH_AVAILABLE:
    # Create dummy base class when PyTorch is not available
    class nn:
        class Module:
            def __init__(self):
                pass


class EncoderNetwork(nn.Module):
    """
    Encoder network: [200D] → [512D] → [num_features]

    Compresses raw musical features into semantic feature space.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # First layer: 200 → 512
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.bn1 = nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(config.dropout)

        # Second layer: 512 → num_semantic_features
        self.fc2 = nn.Linear(config.hidden_dim, config.num_semantic_features)
        self.bn2 = nn.BatchNorm1d(config.num_semantic_features) if config.use_batch_norm else nn.Identity()

        # Activation function for semantic features
        if config.feature_activation == "relu":
            self.feature_activation = nn.ReLU()
        elif config.feature_activation == "sigmoid":
            self.feature_activation = nn.Sigmoid()
        elif config.feature_activation == "tanh":
            self.feature_activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {config.feature_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input features [batch_size, 200]

        Returns:
            Semantic features [batch_size, num_semantic_features]
        """
        # Layer 1
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout1(h)

        # Layer 2
        z = self.fc2(h)
        z = self.bn2(z)
        z = self.feature_activation(z)

        # Optional normalization
        if self.config.normalize_features:
            z = F.normalize(z, p=2, dim=1)

        return z


class DecoderNetwork(nn.Module):
    """
    Decoder network: [num_features] → [512D] → [200D]

    Reconstructs original features from semantic features.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # First layer: num_semantic_features → 512
        self.fc1 = nn.Linear(config.num_semantic_features, config.hidden_dim)
        self.bn1 = nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(config.dropout)

        # Second layer: 512 → 200
        self.fc2 = nn.Linear(config.hidden_dim, config.input_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Semantic features [batch_size, num_semantic_features]

        Returns:
            Reconstructed features [batch_size, 200]
        """
        # Layer 1
        h = self.fc1(z)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout1(h)

        # Layer 2
        x_recon = self.fc2(h)

        return x_recon


class LocalityPredictor(nn.Module):
    """
    Locality predictor: [num_features * 2] → [512] → [num_locality_types]

    Predicts which musical locality transformation was applied between two pieces.
    This enforces that semantic features capture musically meaningful dimensions.

    Input: Concatenation of two semantic feature vectors
    Output: Probability distribution over locality types
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Input is concatenation of two feature vectors
        input_dim = config.num_semantic_features * 2

        # First layer
        self.fc1 = nn.Linear(input_dim, config.hidden_dim)
        self.bn1 = nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(config.dropout)

        # Second layer
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim // 2) if config.use_batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(config.dropout)

        # Output layer
        self.fc3 = nn.Linear(config.hidden_dim // 2, config.num_locality_types)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Predict locality transformation between two pieces.

        Args:
            z1: Semantic features of original [batch_size, num_semantic_features]
            z2: Semantic features of transformed [batch_size, num_semantic_features]

        Returns:
            Logits for locality types [batch_size, num_locality_types]
        """
        # Concatenate features
        z_concat = torch.cat([z1, z2], dim=1)

        # Layer 1
        h = self.fc1(z_concat)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout1(h)

        # Layer 2
        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout2(h)

        # Output layer
        logits = self.fc3(h)

        return logits


# ============================================================================
# Main Encoder Model
# ============================================================================

class SemanticFeatureEncoder(nn.Module):
    """
    Complete semantic feature encoder for musical parameter discovery.

    This model learns to:
    1. Encode 200D features into semantic features
    2. Decode semantic features back to 200D
    3. Predict musical locality transformations

    The training process discovers interpretable musical parameters by:
    - Minimizing reconstruction error
    - Maximizing locality prediction accuracy
    - Enforcing sparsity on semantic features

    Usage:
        # Initialize
        config = EncoderConfig(num_semantic_features=30)
        encoder = SemanticFeatureEncoder(config)

        # Forward pass
        features = torch.randn(32, 200)  # Batch of 32 feature vectors
        output = encoder(features)

        # Compute loss
        transformed_features = torch.randn(32, 200)
        locality_labels = torch.randint(0, 12, (32,))
        loss_dict = encoder.compute_loss(
            features,
            transformed_features,
            locality_labels
        )

        # Extract semantic features
        semantic_features = encoder.extract_semantic_features(features)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Sub-networks
        self.encoder = EncoderNetwork(config)
        self.decoder = DecoderNetwork(config)
        self.locality_predictor = LocalityPredictor(config)

        # Track training progress
        self.training_step = 0

        # Weight sparsity for superposition reduction
        self.sparsity_scheduler = None
        self.weight_masks = {}  # Store masks for each layer
        if config.enable_weight_sparsity:
            self.sparsity_scheduler = SparsityScheduler(
                initial_sparsity=config.initial_sparsity,
                target_sparsity=config.target_sparsity,
                warmup_epochs=config.sparsity_warmup_epochs,
                schedule_type='linear'
            )

    def forward(
        self,
        x: torch.Tensor,
        x_transformed: Optional[torch.Tensor] = None,
        compute_locality: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.

        Args:
            x: Input features [batch_size, 200]
            x_transformed: Transformed features [batch_size, 200] (optional, for locality prediction)
            compute_locality: Whether to compute locality predictions

        Returns:
            Dictionary with:
                - 'semantic_features': Encoded features [batch_size, num_semantic_features]
                - 'reconstructed': Reconstructed features [batch_size, 200]
                - 'locality_logits': Locality predictions [batch_size, num_locality_types] (if x_transformed provided)
        """
        # Encode
        z = self.encoder(x)

        # Decode
        x_recon = self.decoder(z)

        output = {
            'semantic_features': z,
            'reconstructed': x_recon
        }

        # Locality prediction (if transformed features provided)
        if x_transformed is not None and compute_locality:
            z_transformed = self.encoder(x_transformed)
            locality_logits = self.locality_predictor(z, z_transformed)
            output['locality_logits'] = locality_logits
            output['semantic_features_transformed'] = z_transformed

        return output

    def compute_loss(
        self,
        x: torch.Tensor,
        x_transformed: Optional[torch.Tensor] = None,
        locality_labels: Optional[torch.Tensor] = None,
        return_components: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and its components.

        Args:
            x: Input features [batch_size, 200]
            x_transformed: Transformed features [batch_size, 200] (optional)
            locality_labels: Ground truth locality types [batch_size] (optional)
            return_components: Whether to return individual loss components

        Returns:
            Dictionary with:
                - 'total_loss': Total weighted loss
                - 'reconstruction_loss': MSE reconstruction loss
                - 'locality_loss': Cross-entropy locality loss (if labels provided)
                - 'sparsity_loss': L1 sparsity loss
                - 'locality_accuracy': Accuracy of locality prediction (if labels provided)
        """
        # Forward pass
        output = self.forward(x, x_transformed, compute_locality=(locality_labels is not None))

        z = output['semantic_features']
        x_recon = output['reconstructed']

        # 1. Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(x_recon, x)

        # 2. Sparsity loss (L1 on semantic features)
        sparsity_loss = torch.mean(torch.abs(z))

        # 3. Locality loss (if labels provided)
        locality_loss = torch.tensor(0.0, device=x.device)
        locality_accuracy = torch.tensor(0.0, device=x.device)

        if locality_labels is not None and 'locality_logits' in output:
            locality_logits = output['locality_logits']
            locality_loss = F.cross_entropy(locality_logits, locality_labels)

            # Compute accuracy
            predictions = torch.argmax(locality_logits, dim=1)
            locality_accuracy = (predictions == locality_labels).float().mean()

        # 4. Total loss (weighted sum)
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss +
            self.config.locality_weight * locality_loss +
            self.config.sparsity_weight * sparsity_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'locality_loss': locality_loss,
            'sparsity_loss': sparsity_loss,
            'locality_accuracy': locality_accuracy
        }

        return loss_dict

    def extract_semantic_features(
        self,
        x: torch.Tensor,
        as_numpy: bool = False
    ) -> torch.Tensor:
        """
        Extract semantic features from input.

        This is the main inference method for extracting learned parameters.

        Args:
            x: Input features [batch_size, 200] or [200]
            as_numpy: Return as numpy array instead of tensor

        Returns:
            Semantic features [batch_size, num_semantic_features] or [num_semantic_features]
        """
        # Handle single sample
        single_sample = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True

        # Extract features
        with torch.no_grad():
            z = self.encoder(x)

        # Remove batch dimension if single sample
        if single_sample:
            z = z.squeeze(0)

        # Convert to numpy if requested
        if as_numpy:
            z = z.cpu().numpy()

        return z

    def get_feature_importance(self) -> np.ndarray:
        """
        Compute importance scores for each semantic feature.

        Importance is measured by the L2 norm of decoder weights for each feature.
        Higher values indicate features that have more influence on reconstruction.

        Returns:
            Importance scores [num_semantic_features]
        """
        # Get decoder weights (first layer)
        decoder_weights = self.decoder.fc1.weight.data  # [hidden_dim, num_semantic_features]

        # Compute L2 norm for each feature
        importance = torch.norm(decoder_weights, p=2, dim=0)  # [num_semantic_features]

        return importance.cpu().numpy()

    def apply_weight_sparsity(self, sparsity_ratio: float):
        """
        Apply Top-K weight sparsity to all Linear layers.

        This zeroes out all but the top-k weights (by magnitude) in each layer,
        encouraging monosemantic features and reducing superposition.

        Args:
            sparsity_ratio: Proportion of weights to keep (e.g., 0.001 = keep 0.1% of weights)

        Note:
            This modifies weights in-place. Call during training after gradient updates.
        """
        if not self.config.enable_weight_sparsity:
            return

        # Apply sparsity to all Linear layers in all sub-networks
        for name, module in self.named_modules():
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

                    # Store mask for analysis
                    self.weight_masks[name] = mask

    def compute_weight_sparsity_ratio(self) -> float:
        """
        Compute the actual sparsity ratio of model weights.

        Returns:
            Proportion of non-zero weights (0.0 = all zeros, 1.0 = all non-zero)
        """
        total_params = 0
        nonzero_params = 0

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    weight_abs = torch.abs(module.weight.data)
                    total_params += weight_abs.numel()
                    # Count as non-zero if magnitude > 1e-8
                    nonzero_params += (weight_abs > 1e-8).sum().item()

        if total_params == 0:
            return 1.0

        return nonzero_params / total_params

    def update_sparsity(self, epoch: int):
        """
        Update weight sparsity based on current epoch.

        This should be called at the start or end of each training epoch.

        Args:
            epoch: Current training epoch (0-indexed)
        """
        if not self.config.enable_weight_sparsity or self.sparsity_scheduler is None:
            return

        # Get current sparsity ratio from scheduler
        current_sparsity = self.sparsity_scheduler.get_sparsity(epoch)

        # Apply sparsity
        self.apply_weight_sparsity(current_sparsity)

    def save(self, path: Path, include_config: bool = True):
        """
        Save model checkpoint.

        Args:
            path: Path to save model
            include_config: Whether to save config alongside model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'training_step': self.training_step,
            'weight_masks': self.weight_masks  # Save sparsity masks
        }

        torch.save(checkpoint, path)

        # Save config separately
        if include_config:
            config_path = path.with_suffix('.json')
            self.config.save(config_path)

        print(f"✅ Model saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'SemanticFeatureEncoder':
        """
        Load model checkpoint.

        Args:
            path: Path to model checkpoint
            device: Device to load model on ('cpu', 'cuda', etc.)

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)

        # Create config
        config = EncoderConfig.from_dict(checkpoint['config'])

        # Create model
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_step = checkpoint.get('training_step', 0)
        model.weight_masks = checkpoint.get('weight_masks', {})

        model.to(device)
        model.eval()

        print(f"✅ Model loaded from {path}")
        print(f"   Training step: {model.training_step}")
        print(f"   Semantic features: {config.num_semantic_features}")
        if config.enable_weight_sparsity:
            sparsity_ratio = model.compute_weight_sparsity_ratio()
            print(f"   Weight sparsity: {sparsity_ratio:.4f} (enabled)")

        return model


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_encoder(
    num_semantic_features: int = 30,
    device: str = 'cpu'
) -> SemanticFeatureEncoder:
    """
    Create encoder with default configuration.

    Args:
        num_semantic_features: Number of semantic features to discover
        device: Device to create model on

    Returns:
        Initialized encoder
    """
    config = EncoderConfig(num_semantic_features=num_semantic_features)
    encoder = SemanticFeatureEncoder(config)
    encoder.to(device)
    return encoder


def compute_reconstruction_quality(
    encoder: SemanticFeatureEncoder,
    features: torch.Tensor
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.

    Args:
        encoder: Trained encoder
        features: Input features [batch_size, 200]

    Returns:
        Dictionary with quality metrics:
            - 'mse': Mean squared error
            - 'mae': Mean absolute error
            - 'r2': R-squared score
            - 'correlation': Average feature correlation
    """
    encoder.eval()

    with torch.no_grad():
        output = encoder(features)
        reconstructed = output['reconstructed']

    # MSE
    mse = F.mse_loss(reconstructed, features).item()

    # MAE
    mae = F.l1_loss(reconstructed, features).item()

    # R-squared
    ss_res = torch.sum((features - reconstructed) ** 2)
    ss_tot = torch.sum((features - torch.mean(features)) ** 2)
    r2 = (1 - ss_res / ss_tot).item()

    # Correlation
    features_np = features.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()

    correlations = []
    for i in range(features_np.shape[1]):
        corr = np.corrcoef(features_np[:, i], reconstructed_np[:, i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    avg_correlation = np.mean(correlations) if correlations else 0.0

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'correlation': avg_correlation
    }


def analyze_semantic_features(
    encoder: SemanticFeatureEncoder,
    features_dataset: torch.Tensor,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Analyze learned semantic features.

    Args:
        encoder: Trained encoder
        features_dataset: Dataset of features [num_samples, 200]
        top_k: Number of top features to analyze

    Returns:
        Analysis dictionary with:
            - 'feature_importance': Importance scores for all features
            - 'top_features': Indices of most important features
            - 'activation_statistics': Mean/std of activations per feature
            - 'sparsity': Sparsity level (% of near-zero activations)
    """
    encoder.eval()

    # Extract semantic features for dataset
    with torch.no_grad():
        semantic_features = encoder.extract_semantic_features(features_dataset, as_numpy=True)

    # Feature importance
    importance = encoder.get_feature_importance()
    top_features = np.argsort(importance)[::-1][:top_k]

    # Activation statistics
    activation_stats = {
        'mean': np.mean(semantic_features, axis=0),
        'std': np.std(semantic_features, axis=0),
        'min': np.min(semantic_features, axis=0),
        'max': np.max(semantic_features, axis=0)
    }

    # Sparsity (% of activations < 0.01)
    sparsity = np.mean(np.abs(semantic_features) < 0.01)

    return {
        'feature_importance': importance,
        'top_features': top_features,
        'activation_statistics': activation_stats,
        'sparsity': sparsity
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Semantic Feature Encoder - Agent 3")
    print("="*70)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch is not installed.")
        print("   Install PyTorch to use this module:")
        print("   pip install torch")
        exit(1)

    # Create encoder
    print("\n1. Creating encoder...")
    config = EncoderConfig(
        num_semantic_features=30,
        num_locality_types=12,
        hidden_dim=512
    )
    encoder = create_default_encoder(num_semantic_features=30)
    print(f"   ✅ Created encoder with {config.num_semantic_features} semantic features")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 16
    x = torch.randn(batch_size, 200)
    x_transformed = torch.randn(batch_size, 200)
    locality_labels = torch.randint(0, 12, (batch_size,))

    output = encoder(x, x_transformed)
    print(f"   ✅ Forward pass successful")
    print(f"      Semantic features shape: {output['semantic_features'].shape}")
    print(f"      Reconstructed shape: {output['reconstructed'].shape}")
    print(f"      Locality logits shape: {output['locality_logits'].shape}")

    # Test loss computation
    print("\n3. Testing loss computation...")
    loss_dict = encoder.compute_loss(x, x_transformed, locality_labels)
    print(f"   ✅ Loss computation successful")
    print(f"      Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"      Reconstruction: {loss_dict['reconstruction_loss'].item():.4f}")
    print(f"      Locality: {loss_dict['locality_loss'].item():.4f}")
    print(f"      Sparsity: {loss_dict['sparsity_loss'].item():.4f}")
    print(f"      Locality accuracy: {loss_dict['locality_accuracy'].item():.2%}")

    # Test semantic feature extraction
    print("\n4. Testing semantic feature extraction...")
    semantic_features = encoder.extract_semantic_features(x)
    print(f"   ✅ Extraction successful")
    print(f"      Features shape: {semantic_features.shape}")
    print(f"      Features range: [{semantic_features.min():.3f}, {semantic_features.max():.3f}]")

    # Test save/load
    print("\n5. Testing save/load...")
    save_path = Path("/tmp/test_encoder.pt")
    encoder.save(save_path)
    encoder_loaded = SemanticFeatureEncoder.load(save_path)
    print(f"   ✅ Save/load successful")

    # Test reconstruction quality
    print("\n6. Testing reconstruction quality...")
    quality = compute_reconstruction_quality(encoder, x)
    print(f"   ✅ Quality metrics computed")
    print(f"      MSE: {quality['mse']:.4f}")
    print(f"      MAE: {quality['mae']:.4f}")
    print(f"      R²: {quality['r2']:.4f}")
    print(f"      Correlation: {quality['correlation']:.4f}")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
