"""
Training modules for Inverse Audio Effects System.
"""

from .losses import (
    MultiResolutionSTFTLoss,
    ContrastiveLoss,
    ChainLoss,
    InverseAFxLoss,
)
from .train_system import InverseAFxSystem
from .train_unified import (
    UnifiedInverterSystem,
    UnifiedTrainingConfig,
    UnifiedDataset,
    UnifiedDataModule,
)
from .train_latent_flow import (
    LatentFlowInverterSystem,
    LatentFlowTrainingConfig,
    LatentFlowDataModule,
    create_trainer,
)

__all__ = [
    # Losses
    "MultiResolutionSTFTLoss",
    "ContrastiveLoss",
    "ChainLoss",
    "InverseAFxLoss",
    # Original training system
    "InverseAFxSystem",
    # Unified training system
    "UnifiedInverterSystem",
    "UnifiedTrainingConfig",
    "UnifiedDataset",
    "UnifiedDataModule",
    # Latent flow training system
    "LatentFlowInverterSystem",
    "LatentFlowTrainingConfig",
    "LatentFlowDataModule",
    "create_trainer",
]
