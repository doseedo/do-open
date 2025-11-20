"""
Expansion Orchestration Module
===============================

This module contains the master orchestrator for automated system expansion.

Modules:
- expansion_orchestrator: Agent 16 - Expansion Orchestrator

Author: Musical Program Synthesis Team
License: MIT
"""

from .expansion_orchestrator import (
    ExpansionOrchestrator,
    ExpansionStage,
    ExpansionStatus,
    ExpansionCheckpoint,
    ParameterExpansion,
    ExpansionWorkflowResult,
    SafetyMonitor
)

__all__ = [
    'ExpansionOrchestrator',
    'ExpansionStage',
    'ExpansionStatus',
    'ExpansionCheckpoint',
    'ParameterExpansion',
    'ExpansionWorkflowResult',
    'SafetyMonitor'
]
