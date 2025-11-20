#!/usr/bin/env python3
"""
Foundation Parameter System
============================

Unified parameter registry for the MIDI Generator library.
Part of the Focused Parameter Refactoring project.

This module will eventually contain ~500-800 parameters across all domains:
- harmony.* (150 params) - Agent 3
- melody.* (100 params) - Agent 4
- rhythm.* (100 params) - Agent 5
- structure.* (50 params) - Agent 6
- instrumentation.* (50 params) - Agent 7 ✅
- dynamics.* (50 params) - Agent 8
- transformation.* (remaining) - Agent 8

Current Status:
- Agent 7 (Instrumentation & Orchestration): ✅ In Progress

Author: Focused Parameter Refactoring - Agents 1-10
"""

from .instrumentation_params import (
    INSTRUMENTATION_PARAMETERS,
    get_instrumentation_parameter,
    get_all_instrumentation_parameters
)

__all__ = [
    'INSTRUMENTATION_PARAMETERS',
    'get_instrumentation_parameter',
    'get_all_instrumentation_parameters'
]
