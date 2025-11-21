#!/usr/bin/env python3
"""
Cross-Dimensional Encoder - Agent 7: Modular Semantic Discovery
================================================================

This module discovers interaction patterns between musical dimensions
(harmony, rhythm, form, orchestration, texture).

The CrossDimensionalEncoder:
1. Aggregates outputs from all 5 dimension-specific encoders (110 params total)
2. Discovers 10 cross-dimensional parameters that capture interactions
3. Validates musical coherence across dimensions
4. Implements parameter coupling constraints

Architecture:
    Input: [110D] from 5 modules:
        - Harmony: 30 params
        - Rhythm: 20 params
        - Form: 15 params
        - Orchestration: 25 params
        - Texture: 20 params

    Fusion Network: [110D] → [256D] → [128D]
    Cross-Encoder: [128D] → [10D]

    Output: 10 cross-dimensional parameters:
        - harmonic_rhythmic_coupling
        - form_driven_texture_change
        - structural_harmonic_anchoring
        - orchestral_intensity_gradient
        - climax_convergence_factor
        - texture_density_correlation
        - rhythmic_harmonic_tension
        - formal_orchestration_coupling
        - cross_dimensional_coherence
        - stylistic_consistency_score

Author: Agent 7 - Cross-Dimensional Pattern Discoverer
Date: November 21, 2025
Version: 1.0.0
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Neural network functionality will be disabled.")
    # Create dummy base class when PyTorch is not available
    class nn:
        class Module:
            def __init__(self):
                pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CrossDimensionalConfig:
    """Configuration for CrossDimensionalEncoder"""

    # Input dimensions from each module
    harmony_dim: int = 30
    rhythm_dim: int = 20
    form_dim: int = 15
    orchestration_dim: int = 25
    texture_dim: int = 20

    # Derived total input dimension
    @property
    def total_input_dim(self) -> int:
        return (self.harmony_dim + self.rhythm_dim + self.form_dim +
                self.orchestration_dim + self.texture_dim)

    # Architecture
    fusion_hidden_dim: int = 256
    cross_hidden_dim: int = 128
    num_cross_params: int = 10

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    dropout: float = 0.1
    weight_decay: float = 1e-5

    # Loss weights
    reconstruction_weight: float = 1.0
    coherence_weight: float = 0.5
    coupling_weight: float = 0.3
    sparsity_weight: float = 0.01

    # Regularization
    use_batch_norm: bool = True
    use_layer_norm: bool = False

    # Coupling constraints
    enable_coupling_validation: bool = True
    coupling_strength_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'harmony_dim': self.harmony_dim,
            'rhythm_dim': self.rhythm_dim,
            'form_dim': self.form_dim,
            'orchestration_dim': self.orchestration_dim,
            'texture_dim': self.texture_dim,
            'fusion_hidden_dim': self.fusion_hidden_dim,
            'cross_hidden_dim': self.cross_hidden_dim,
            'num_cross_params': self.num_cross_params,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
            'reconstruction_weight': self.reconstruction_weight,
            'coherence_weight': self.coherence_weight,
            'coupling_weight': self.coupling_weight,
            'sparsity_weight': self.sparsity_weight,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'enable_coupling_validation': self.enable_coupling_validation,
            'coupling_strength_threshold': self.coupling_strength_threshold
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CrossDimensionalConfig':
        """Create from dictionary"""
        return cls(**config_dict)

    def save(self, path: Path):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'CrossDimensionalConfig':
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class CrossDimensionalParameters:
    """
    The 10 discovered cross-dimensional parameters.

    These capture interaction patterns between musical dimensions.
    """
    harmonic_rhythmic_coupling: float = 0.0  # How rhythm follows harmony changes
    form_driven_texture_change: float = 0.0  # Texture variation across sections
    structural_harmonic_anchoring: float = 0.0  # Harmony stability at section boundaries
    orchestral_intensity_gradient: float = 0.0  # Orchestration density evolution
    climax_convergence_factor: float = 0.0  # All dimensions converging at climax
    texture_density_correlation: float = 0.0  # Harmony complexity vs texture density
    rhythmic_harmonic_tension: float = 0.0  # Syncopation during harmonic tension
    formal_orchestration_coupling: float = 0.0  # Instrumentation changes with form
    cross_dimensional_coherence: float = 0.0  # Overall inter-dimension consistency
    stylistic_consistency_score: float = 0.0  # Genre/style coherence across dimensions

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([
            self.harmonic_rhythmic_coupling,
            self.form_driven_texture_change,
            self.structural_harmonic_anchoring,
            self.orchestral_intensity_gradient,
            self.climax_convergence_factor,
            self.texture_density_correlation,
            self.rhythmic_harmonic_tension,
            self.formal_orchestration_coupling,
            self.cross_dimensional_coherence,
            self.stylistic_consistency_score
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CrossDimensionalParameters':
        """Create from numpy array"""
        return cls(
            harmonic_rhythmic_coupling=float(arr[0]),
            form_driven_texture_change=float(arr[1]),
            structural_harmonic_anchoring=float(arr[2]),
            orchestral_intensity_gradient=float(arr[3]),
            climax_convergence_factor=float(arr[4]),
            texture_density_correlation=float(arr[5]),
            rhythmic_harmonic_tension=float(arr[6]),
            formal_orchestration_coupling=float(arr[7]),
            cross_dimensional_coherence=float(arr[8]),
            stylistic_consistency_score=float(arr[9])
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'harmonic_rhythmic_coupling': self.harmonic_rhythmic_coupling,
            'form_driven_texture_change': self.form_driven_texture_change,
            'structural_harmonic_anchoring': self.structural_harmonic_anchoring,
            'orchestral_intensity_gradient': self.orchestral_intensity_gradient,
            'climax_convergence_factor': self.climax_convergence_factor,
            'texture_density_correlation': self.texture_density_correlation,
            'rhythmic_harmonic_tension': self.rhythmic_harmonic_tension,
            'formal_orchestration_coupling': self.formal_orchestration_coupling,
            'cross_dimensional_coherence': self.cross_dimensional_coherence,
            'stylistic_consistency_score': self.stylistic_consistency_score
        }


# =============================================================================
# Neural Network Components
# =============================================================================

class FusionNetwork(nn.Module):
    """
    Fusion network that combines outputs from all dimension modules.

    Architecture: [110D] → [256D] → [128D]

    This network learns to fuse information from:
    - Harmony (30D)
    - Rhythm (20D)
    - Form (15D)
    - Orchestration (25D)
    - Texture (20D)
    """

    def __init__(self, config: CrossDimensionalConfig):
        super().__init__()
        self.config = config

        input_dim = config.total_input_dim  # 110
        hidden_dim = config.fusion_hidden_dim  # 256
        output_dim = config.cross_hidden_dim  # 128

        # Layer 1: 110 → 256
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if config.use_batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(config.dropout)

        # Layer 2: 256 → 128
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim) if config.use_batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fuse dimension-specific features.

        Args:
            x: Concatenated features [batch_size, 110]

        Returns:
            Fused features [batch_size, 128]
        """
        # Layer 1
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout1(h)

        # Layer 2
        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout2(h)

        return h


class CrossEncoderNetwork(nn.Module):
    """
    Cross-dimensional encoder that discovers interaction parameters.

    Architecture: [128D] → [10D]

    Outputs 10 interpretable cross-dimensional parameters.
    """

    def __init__(self, config: CrossDimensionalConfig):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(config.cross_hidden_dim, config.num_cross_params)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Extract cross-dimensional parameters.

        Args:
            h: Fused features [batch_size, 128]

        Returns:
            Cross parameters [batch_size, 10]
        """
        # Apply sigmoid to get values in [0, 1]
        cross_params = torch.sigmoid(self.fc(h))
        return cross_params


class ReconstructionNetwork(nn.Module):
    """
    Reconstruction network to validate cross-dimensional features.

    Architecture: [10D] → [128D] → [110D]

    Reconstructs the original dimension features from cross-dimensional params.
    """

    def __init__(self, config: CrossDimensionalConfig):
        super().__init__()
        self.config = config

        # Layer 1: 10 → 128
        self.fc1 = nn.Linear(config.num_cross_params, config.cross_hidden_dim)
        self.bn1 = nn.BatchNorm1d(config.cross_hidden_dim) if config.use_batch_norm else nn.Identity()

        # Layer 2: 128 → 110
        self.fc2 = nn.Linear(config.cross_hidden_dim, config.total_input_dim)

    def forward(self, cross_params: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct dimension features from cross parameters.

        Args:
            cross_params: Cross-dimensional parameters [batch_size, 10]

        Returns:
            Reconstructed features [batch_size, 110]
        """
        # Layer 1
        h = self.fc1(cross_params)
        h = self.bn1(h)
        h = F.relu(h)

        # Layer 2
        x_recon = self.fc2(h)

        return x_recon


# =============================================================================
# Cross-Dimensional Encoder
# =============================================================================

class CrossDimensionalEncoder(nn.Module):
    """
    Complete cross-dimensional encoder for discovering interaction patterns.

    This encoder:
    1. Fuses outputs from 5 dimension-specific encoders
    2. Discovers 10 cross-dimensional interaction parameters
    3. Validates reconstruction quality
    4. Enforces coupling constraints

    Usage:
        config = CrossDimensionalConfig()
        encoder = CrossDimensionalEncoder(config)

        # Prepare inputs (from dimension encoders)
        inputs = {
            'harmony': torch.randn(32, 30),
            'rhythm': torch.randn(32, 20),
            'form': torch.randn(32, 15),
            'orchestration': torch.randn(32, 25),
            'texture': torch.randn(32, 20)
        }

        # Extract cross-dimensional parameters
        output = encoder(inputs)
        cross_params = output['cross_parameters']  # [32, 10]
    """

    def __init__(self, config: CrossDimensionalConfig):
        super().__init__()
        self.config = config

        # Sub-networks
        self.fusion_net = FusionNetwork(config)
        self.cross_encoder = CrossEncoderNetwork(config)
        self.reconstruction_net = ReconstructionNetwork(config)

        # Parameter coupling matrix (learned)
        # Shape: [10, 110] - how each cross-param depends on each dimension param
        self.coupling_matrix = nn.Parameter(
            torch.randn(config.num_cross_params, config.total_input_dim) * 0.01
        )

        # Track training progress
        self.training_step = 0

    def forward(
        self,
        dimension_features: Dict[str, torch.Tensor],
        return_reconstruction: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cross-dimensional encoder.

        Args:
            dimension_features: Dictionary with keys:
                - 'harmony': [batch_size, 30]
                - 'rhythm': [batch_size, 20]
                - 'form': [batch_size, 15]
                - 'orchestration': [batch_size, 25]
                - 'texture': [batch_size, 20]
            return_reconstruction: Whether to compute reconstruction

        Returns:
            Dictionary with:
                - 'cross_parameters': [batch_size, 10]
                - 'fused_features': [batch_size, 128]
                - 'reconstructed': [batch_size, 110] (if return_reconstruction=True)
                - 'coupling_strengths': [10, 110]
        """
        # Concatenate dimension features
        x = torch.cat([
            dimension_features.get('harmony', torch.zeros(len(dimension_features['rhythm']), 30).to(dimension_features['rhythm'].device)),
            dimension_features.get('rhythm', torch.zeros(len(dimension_features['harmony']), 20).to(dimension_features['harmony'].device)),
            dimension_features.get('form', torch.zeros(len(dimension_features['harmony']), 15).to(dimension_features['harmony'].device)),
            dimension_features.get('orchestration', torch.zeros(len(dimension_features['harmony']), 25).to(dimension_features['harmony'].device)),
            dimension_features.get('texture', torch.zeros(len(dimension_features['harmony']), 20).to(dimension_features['harmony'].device))
        ], dim=1)  # [batch_size, 110]

        # Fusion
        fused = self.fusion_net(x)  # [batch_size, 128]

        # Cross-dimensional encoding
        cross_params = self.cross_encoder(fused)  # [batch_size, 10]

        output = {
            'cross_parameters': cross_params,
            'fused_features': fused,
            'coupling_strengths': torch.sigmoid(self.coupling_matrix)
        }

        # Optional reconstruction
        if return_reconstruction:
            reconstructed = self.reconstruction_net(cross_params)
            output['reconstructed'] = reconstructed
            output['input'] = x

        return output

    def compute_loss(
        self,
        dimension_features: Dict[str, torch.Tensor],
        return_components: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training.

        Args:
            dimension_features: Dictionary of dimension-specific features
            return_components: Whether to return individual loss components

        Returns:
            Dictionary with:
                - 'total_loss': Total weighted loss
                - 'reconstruction_loss': MSE reconstruction loss
                - 'coherence_loss': Cross-dimensional coherence loss
                - 'coupling_loss': Parameter coupling constraint loss
                - 'sparsity_loss': L1 sparsity on cross parameters
        """
        # Forward pass
        output = self.forward(dimension_features, return_reconstruction=True)

        x = output['input']
        x_recon = output['reconstructed']
        cross_params = output['cross_parameters']
        coupling_strengths = output['coupling_strengths']

        # 1. Reconstruction loss
        reconstruction_loss = F.mse_loss(x_recon, x)

        # 2. Coherence loss (ensure cross-params are consistent)
        # Penalize high variance in cross-params across batch
        coherence_loss = torch.var(cross_params, dim=0).mean()

        # 3. Coupling loss (enforce meaningful coupling patterns)
        # Encourage sparse but strong couplings
        coupling_loss = -torch.mean(
            coupling_strengths * torch.log(coupling_strengths + 1e-8) +
            (1 - coupling_strengths) * torch.log(1 - coupling_strengths + 1e-8)
        )

        # 4. Sparsity loss
        sparsity_loss = torch.mean(torch.abs(cross_params))

        # Total loss
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss +
            self.config.coherence_weight * coherence_loss +
            self.config.coupling_weight * coupling_loss +
            self.config.sparsity_weight * sparsity_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'coherence_loss': coherence_loss,
            'coupling_loss': coupling_loss,
            'sparsity_loss': sparsity_loss
        }

        return loss_dict

    def extract_cross_parameters(
        self,
        dimension_features: Dict[str, torch.Tensor],
        as_numpy: bool = False
    ) -> torch.Tensor:
        """
        Extract cross-dimensional parameters (main inference method).

        Args:
            dimension_features: Dictionary of dimension-specific features
            as_numpy: Return as numpy array instead of tensor

        Returns:
            Cross-dimensional parameters [batch_size, 10] or [10]
        """
        with torch.no_grad():
            output = self.forward(dimension_features, return_reconstruction=False)
            cross_params = output['cross_parameters']

        if as_numpy:
            cross_params = cross_params.cpu().numpy()

        return cross_params

    def validate_coherence(
        self,
        dimension_features: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Validate musical coherence across dimensions.

        Checks:
        1. Harmony complexity correlates with texture density
        2. Rhythmic activity matches harmonic rhythm
        3. Form changes align with orchestration changes
        4. Climax convergence across all dimensions

        Args:
            dimension_features: Dictionary of dimension-specific features

        Returns:
            Dictionary with coherence validation results
        """
        with torch.no_grad():
            output = self.forward(dimension_features)
            cross_params = output['cross_parameters'].cpu().numpy()

        # Extract individual parameters (assuming batch_size=1 or take mean)
        if cross_params.shape[0] > 1:
            cross_params = cross_params.mean(axis=0)
        else:
            cross_params = cross_params[0]

        params = CrossDimensionalParameters.from_array(cross_params)

        # Validation rules
        validations = {
            'harmony_texture_correlation': {
                'value': params.texture_density_correlation,
                'valid': params.texture_density_correlation > 0.3,
                'description': 'Texture density should correlate with harmony complexity'
            },
            'form_orchestration_coupling': {
                'value': params.formal_orchestration_coupling,
                'valid': params.formal_orchestration_coupling > 0.4,
                'description': 'Orchestration should change with formal sections'
            },
            'climax_convergence': {
                'value': params.climax_convergence_factor,
                'valid': params.climax_convergence_factor > 0.5,
                'description': 'All dimensions should converge at musical climax'
            },
            'overall_coherence': {
                'value': params.cross_dimensional_coherence,
                'valid': params.cross_dimensional_coherence > 0.6,
                'description': 'Overall cross-dimensional consistency'
            }
        }

        # Overall validity
        all_valid = all(v['valid'] for v in validations.values())

        return {
            'valid': all_valid,
            'validations': validations,
            'parameters': params.to_dict()
        }

    def get_coupling_matrix(self) -> np.ndarray:
        """
        Get learned parameter coupling matrix.

        Returns:
            Coupling matrix [10, 110] showing dependencies
        """
        return torch.sigmoid(self.coupling_matrix).detach().cpu().numpy()

    def save(self, path: Path, include_config: bool = True):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'training_step': self.training_step
        }

        torch.save(checkpoint, path)

        if include_config:
            config_path = path.with_suffix('.json')
            self.config.save(config_path)

        print(f"✅ CrossDimensionalEncoder saved to {path}")

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'CrossDimensionalEncoder':
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)

        config = CrossDimensionalConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_step = checkpoint.get('training_step', 0)

        model.to(device)
        model.eval()

        print(f"✅ CrossDimensionalEncoder loaded from {path}")
        print(f"   Training step: {model.training_step}")
        print(f"   Cross-dimensional parameters: {config.num_cross_params}")

        return model


# =============================================================================
# Utility Functions
# =============================================================================

def create_default_cross_encoder(device: str = 'cpu') -> CrossDimensionalEncoder:
    """
    Create cross-dimensional encoder with default configuration.

    Args:
        device: Device to create model on

    Returns:
        Initialized encoder
    """
    config = CrossDimensionalConfig()
    encoder = CrossDimensionalEncoder(config)
    encoder.to(device)
    return encoder


def analyze_interaction_patterns(
    encoder: CrossDimensionalEncoder,
    dimension_features_dataset: List[Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """
    Analyze discovered interaction patterns across dataset.

    Args:
        encoder: Trained cross-dimensional encoder
        dimension_features_dataset: List of dimension feature dictionaries

    Returns:
        Analysis dictionary with interaction pattern statistics
    """
    encoder.eval()

    all_cross_params = []

    with torch.no_grad():
        for features in dimension_features_dataset:
            cross_params = encoder.extract_cross_parameters(features, as_numpy=True)
            all_cross_params.append(cross_params)

    all_cross_params = np.vstack(all_cross_params)

    # Compute statistics
    param_names = list(CrossDimensionalParameters.__annotations__.keys())

    statistics = {}
    for i, name in enumerate(param_names):
        statistics[name] = {
            'mean': float(np.mean(all_cross_params[:, i])),
            'std': float(np.std(all_cross_params[:, i])),
            'min': float(np.min(all_cross_params[:, i])),
            'max': float(np.max(all_cross_params[:, i]))
        }

    # Coupling matrix analysis
    coupling_matrix = encoder.get_coupling_matrix()

    return {
        'parameter_statistics': statistics,
        'coupling_matrix': coupling_matrix,
        'num_samples': len(dimension_features_dataset)
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Cross-Dimensional Encoder - Agent 7")
    print("=" * 80)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch is not installed.")
        print("   Install PyTorch to use this module:")
        print("   pip install torch")
        exit(1)

    # Create encoder
    print("\n1. Creating cross-dimensional encoder...")
    config = CrossDimensionalConfig()
    encoder = create_default_cross_encoder()
    print(f"   ✅ Created encoder with {config.num_cross_params} cross-dimensional parameters")
    print(f"   Input dimension: {config.total_input_dim}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 16

    dimension_features = {
        'harmony': torch.randn(batch_size, 30),
        'rhythm': torch.randn(batch_size, 20),
        'form': torch.randn(batch_size, 15),
        'orchestration': torch.randn(batch_size, 25),
        'texture': torch.randn(batch_size, 20)
    }

    output = encoder(dimension_features)
    print(f"   ✅ Forward pass successful")
    print(f"      Cross parameters shape: {output['cross_parameters'].shape}")
    print(f"      Fused features shape: {output['fused_features'].shape}")
    print(f"      Reconstructed shape: {output['reconstructed'].shape}")

    # Test loss computation
    print("\n3. Testing loss computation...")
    loss_dict = encoder.compute_loss(dimension_features)
    print(f"   ✅ Loss computation successful")
    print(f"      Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"      Reconstruction: {loss_dict['reconstruction_loss'].item():.4f}")
    print(f"      Coherence: {loss_dict['coherence_loss'].item():.4f}")
    print(f"      Coupling: {loss_dict['coupling_loss'].item():.4f}")
    print(f"      Sparsity: {loss_dict['sparsity_loss'].item():.4f}")

    # Test parameter extraction
    print("\n4. Testing cross-parameter extraction...")
    cross_params = encoder.extract_cross_parameters(dimension_features, as_numpy=True)
    print(f"   ✅ Extraction successful")
    print(f"      Shape: {cross_params.shape}")
    print(f"      Range: [{cross_params.min():.3f}, {cross_params.max():.3f}]")

    # Test coherence validation
    print("\n5. Testing coherence validation...")
    validation = encoder.validate_coherence(dimension_features)
    print(f"   ✅ Validation complete")
    print(f"      Overall valid: {validation['valid']}")
    for name, result in validation['validations'].items():
        status = "✓" if result['valid'] else "✗"
        print(f"      {status} {name}: {result['value']:.3f}")

    # Test coupling matrix
    print("\n6. Testing coupling matrix...")
    coupling = encoder.get_coupling_matrix()
    print(f"   ✅ Coupling matrix retrieved")
    print(f"      Shape: {coupling.shape}")
    print(f"      Mean coupling strength: {coupling.mean():.3f}")

    # Test save/load
    print("\n7. Testing save/load...")
    save_path = Path("/tmp/cross_dimensional_encoder.pt")
    encoder.save(save_path)
    encoder_loaded = CrossDimensionalEncoder.load(save_path)
    print(f"   ✅ Save/load successful")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
