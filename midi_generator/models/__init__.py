"""
Models Package - Agent 33
==========================

Model registry and management for 800+ XGBoost models in the Musical Program
Synthesis system.

Modules:
- registry_manager: Complete model lifecycle management, version control,
  performance tracking, and validation
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
