"""
Tracking Module - Agent 30
==========================

System tracking and analytics for the self-expanding inverse music generation system.

Modules:
- expansion_history: Comprehensive tracking of parameter expansions, analytics, and evolution

Main Classes:
- ExpansionHistoryTracker: Central tracking system
- ExpansionEvent: Log of parameter additions
- GapDetectionRecord: Detected system capability gaps
- LLMProposalRecord: LLM-proposed expansions
- SystemSnapshot: Point-in-time system state

Usage:
    from tracking import ExpansionHistoryTracker, ExpansionEvent, ExpansionTrigger

    tracker = ExpansionHistoryTracker()

    event = ExpansionEvent(
        parameters_added=["harmony.jazz.voicing_type"],
        trigger=ExpansionTrigger.RECONSTRUCTION_FAILURE,
        phase=ExpansionPhase.PHASE_1
    )

    tracker.log_expansion_event(event)
    analytics = tracker.generate_analytics()
"""

from .expansion_history import (
    # Main tracker
    ExpansionHistoryTracker,

    # Enums
    ExpansionTrigger,
    ExpansionStatus,
    ParameterImpact,
    ExpansionPhase,

    # Data structures
    ExpansionEvent,
    ParameterHistoryEntry,
    GapDetectionRecord,
    LLMProposalRecord,
    SystemSnapshot,
    ExpansionAnalytics,
)

__all__ = [
    # Main tracker
    'ExpansionHistoryTracker',

    # Enums
    'ExpansionTrigger',
    'ExpansionStatus',
    'ParameterImpact',
    'ExpansionPhase',

    # Data structures
    'ExpansionEvent',
    'ParameterHistoryEntry',
    'GapDetectionRecord',
    'LLMProposalRecord',
    'SystemSnapshot',
    'ExpansionAnalytics',
]

__version__ = '1.0.0'
__author__ = 'Agent 30 - Expansion History Tracker'
