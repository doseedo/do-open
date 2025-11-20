"""
Musical Validation Module
==========================

This module contains validators for ensuring musical correctness of parameters and code.

Modules:
- musical_validator: Agent 13 - Musical Validator (parameter proposals)
- validation_pipeline: Agent 08 - Validation Pipeline Framework (model validation)
- validation_utils: Agent 08 - Validation Utilities
- musical_quality: Agent 08 - Musical Quality Validators

Author: Musical Program Synthesis Team
License: MIT
"""

# Agent 13: Parameter validation
from .musical_validator import (
    MusicalValidator,
    ValidationSeverity,
    ValidationCheck,
    ParameterValidationResult as Agent13ParameterValidationResult,
    CodeValidationResult,
    MusicTheoryRules
)

# Agent 08: Model validation framework
from .validation_pipeline import (
    ValidationPipeline,
    BaseValidator,
    ParameterValidator,
    MusicalQualityValidator,
    GenreValidator,
    ParameterValidationResult,
    MusicalValidationResult,
    GenreValidationResult,
    ValidationReport,
    ValidationStatus,
    ParameterLevel,
    ValidationCategory
)

# Agent 08: Musical quality validators
from .musical_quality import (
    IntervalValidator,
    HarmonyValidator,
    RhythmValidator,
    VoiceRangeValidator
)

__all__ = [
    # Agent 13 (Parameter Validation)
    'MusicalValidator',
    'ValidationSeverity',
    'ValidationCheck',
    'Agent13ParameterValidationResult',
    'CodeValidationResult',
    'MusicTheoryRules',

    # Agent 08 (Model Validation Framework)
    'ValidationPipeline',
    'BaseValidator',
    'ParameterValidator',
    'MusicalQualityValidator',
    'GenreValidator',
    'ParameterValidationResult',
    'MusicalValidationResult',
    'GenreValidationResult',
    'ValidationReport',
    'ValidationStatus',
    'ParameterLevel',
    'ValidationCategory',

    # Agent 08 (Musical Quality Validators)
    'IntervalValidator',
    'HarmonyValidator',
    'RhythmValidator',
    'VoiceRangeValidator',
]
