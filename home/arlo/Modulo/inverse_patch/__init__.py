"""
Inverse Audio Effects System.

A deep learning system for recovering dry audio signals and identifying
effect chains from processed (wet) audio.

Main components:
- EffectEncoder: Encodes wet audio into effect-aware embeddings
- IterativeChainEstimator: Estimates effect chain by iterative inversion
- DifferentiableFXChain: Applies effects differentiably for verification
- InverseAFxSystem: Full training system combining all components

Usage:
    from inverse_afx.training import InverseAFxSystem
    from inverse_afx.export import chain_to_daw_preset

    # Load trained model
    model = InverseAFxSystem.load_from_checkpoint("path/to/checkpoint.ckpt")

    # Process audio
    dry_estimate, chain = model(wet_audio)

    # Export to DAW format
    chain_json = chain_to_daw_preset(chain, format='json')
"""

__version__ = "0.1.0"
__author__ = "Inverse AFx Team"

from .models import (
    EffectEncoder,
    EffectEncoderConfig,
    IterativeChainEstimator,
    ChainEstimatorConfig,
    DifferentiableFXChain,
)

from .training import (
    InverseAFxSystem,
    InverseAFxLoss,
    MultiResolutionSTFTLoss,
)

from .data import (
    EffectChainGenerator,
    InverseAFxDataset,
    InverseAFxDataModule,
)

from .export import (
    chain_to_daw_preset,
    export_json,
    map_to_plugin,
)

from .evaluation import (
    evaluate_system,
    evaluate_dry_recovery,
    evaluate_chain_estimation,
)

__all__ = [
    # Models
    "EffectEncoder",
    "EffectEncoderConfig",
    "IterativeChainEstimator",
    "ChainEstimatorConfig",
    "DifferentiableFXChain",
    # Training
    "InverseAFxSystem",
    "InverseAFxLoss",
    "MultiResolutionSTFTLoss",
    # Data
    "EffectChainGenerator",
    "InverseAFxDataset",
    "InverseAFxDataModule",
    # Export
    "chain_to_daw_preset",
    "export_json",
    "map_to_plugin",
    # Evaluation
    "evaluate_system",
    "evaluate_dry_recovery",
    "evaluate_chain_estimation",
]
