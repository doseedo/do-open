"""
Models for Inverse Audio Effects System.

Includes:
- Effect encoder and chain estimator (original approach)
- Temporal encoders for universal temporal pattern detection
- Flow matching inverter for unified effect inversion
"""

from .effect_encoder import EffectEncoder, EffectEncoderConfig
from .chain_estimator import IterativeChainEstimator, ChainEstimatorConfig
from .differentiable_chain import DifferentiableFXChain

# Temporal pattern encoders (work for ALL temporal effects)
from .temporal_encoder import (
    TemporalCorrelationEncoder,
    MultiScaleTemporalEncoder,
    TemporalDifferenceEncoder,
    UnifiedTemporalEncoder,
)

# Flow matching inverter (unified approach for all effects)
from .flow_inverter import (
    FlowMatchingInverter,
    LightweightFlowInverter,
    create_flow_inverter,
)

# Latent flow matching inverter (operates in DAC latent space)
from .latent_flow_inverter import (
    LatentFlowMatchingInverter,
    LightweightLatentFlowInverter,
    create_latent_flow_inverter,
    FiLM,
)

__all__ = [
    # Original components
    "EffectEncoder",
    "EffectEncoderConfig",
    "IterativeChainEstimator",
    "ChainEstimatorConfig",
    "DifferentiableFXChain",
    # Temporal encoders
    "TemporalCorrelationEncoder",
    "MultiScaleTemporalEncoder",
    "TemporalDifferenceEncoder",
    "UnifiedTemporalEncoder",
    # Flow inverter
    "FlowMatchingInverter",
    "LightweightFlowInverter",
    "create_flow_inverter",
    # Latent flow inverter
    "LatentFlowMatchingInverter",
    "LightweightLatentFlowInverter",
    "create_latent_flow_inverter",
    "FiLM",
]
