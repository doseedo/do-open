#!/usr/bin/env python3
"""
Musical Constraint Validation System
=====================================

Part of the Musical Program Synthesis System (Agent 8)

This package provides comprehensive music theory constraint validation
and automatic correction for generated musical parameters.

Modules:
--------
- musical_validator: Core constraint validation and correction engine
- voice_leading_rules: Classical voice leading constraints
- instrument_ranges: Physical instrument limitations
- harmonic_rules: Tonal harmony and resolution rules
- counterpoint_rules: Multi-voice interaction constraints
- orchestration_rules: Ensemble and arrangement constraints

Author: Agent 8 - Constraint Validator
Phase: 2 (Learning System Implementation)
"""

from .musical_validator import (
    MusicalConstraintValidator,
    ValidationResult,
    ValidationSeverity,
    ConstraintViolation
)

__all__ = [
    'MusicalConstraintValidator',
    'ValidationResult',
    'ValidationSeverity',
    'ConstraintViolation'
]
