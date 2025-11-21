"""
Models Package
==============

Model registry and neural network architectures for the Musical Program
Synthesis system.

Modules:
- registry_manager: Complete model lifecycle management, version control,
  performance tracking, and validation (Agent 33)

- scaled_hierarchical_mtl: Scaled Hierarchical Multi-Task Learning model
  for 600D → 300+ parameter prediction (Agent 4)

- loss_functions: Advanced multi-task loss functions with uncertainty weighting
  and gradient balancing (Agent 4)

- model_config: Configuration management for model architecture, loss functions,
  and training (Agent 4)
"""

from .registry_manager import (
    ModelRegistryManager,
    ModelMetadata,
    ModelStatus,
    ModelType,
    PerformanceMetric,
    ModelPerformanceSnapshot,
    ModelComparison,
    ModelRegistry,
    initialize_registry_from_parameters,
)

# Neural network models (Agent 4) - conditionally import if PyTorch available
try:
    from .scaled_hierarchical_mtl import (
        ScaledHierarchicalMTL,
        create_scaled_model,
        print_model_summary,
    )
    from .loss_functions import (
        ScaledHierarchicalMTLLoss,
        UncertaintyWeightedLoss,
        GradientBalancedLoss,
        create_loss_function,
    )
    from .model_config import (
        ModelConfig,
        LossConfig,
        OptimizerConfig,
        TrainingConfig,
        FullConfig,
        get_default_config,
        get_fast_config,
        get_large_config,
        get_gpu_optimized_config,
    )

    NEURAL_MODELS_AVAILABLE = True

    __all__ = [
        # Registry (Agent 33)
        'ModelRegistryManager',
        'ModelMetadata',
        'ModelStatus',
        'ModelType',
        'PerformanceMetric',
        'ModelPerformanceSnapshot',
        'ModelComparison',
        'ModelRegistry',
        'initialize_registry_from_parameters',

        # Neural Models (Agent 4)
        'ScaledHierarchicalMTL',
        'create_scaled_model',
        'print_model_summary',
        'ScaledHierarchicalMTLLoss',
        'UncertaintyWeightedLoss',
        'GradientBalancedLoss',
        'create_loss_function',
        'ModelConfig',
        'LossConfig',
        'OptimizerConfig',
        'TrainingConfig',
        'FullConfig',
        'get_default_config',
        'get_fast_config',
        'get_large_config',
        'get_gpu_optimized_config',
    ]

except ImportError:
    NEURAL_MODELS_AVAILABLE = False

    __all__ = [
        'ModelRegistryManager',
        'ModelMetadata',
        'ModelStatus',
        'ModelType',
        'PerformanceMetric',
        'ModelPerformanceSnapshot',
        'ModelComparison',
        'ModelRegistry',
        'initialize_registry_from_parameters',
    ]
