"""
Human-in-Loop Interface for Self-Expanding Music Generation System
====================================================================

Agent 29: Human Oversight Dashboard

This module provides a comprehensive web-based interface for human oversight
of the self-expanding inverse music generation system.

Components:
    - HumanOversightEngine: Core engine for managing expansions
    - Flask web application: Real-time dashboard and API
    - Database models: Proposal tracking and metrics storage
    - WebSocket support: Real-time notifications

Usage:
    >>> from interface.human_oversight import HumanOversightEngine, run_server
    >>>
    >>> # Programmatic usage
    >>> engine = HumanOversightEngine()
    >>> proposal_id = engine.create_proposal(
    ...     parameter_name="voicing_complexity",
    ...     parameter_path="harmony.jazz.voicing_complexity",
    ...     parameter_type="continuous",
    ...     description="Controls voicing complexity",
    ...     source=ExpansionSource.LLM_PROPOSAL
    ... )
    >>>
    >>> # Web server
    >>> run_server(host='0.0.0.0', port=5000)

Command Line:
    # Start web server
    python -m interface.human_oversight --server

    # Custom host/port
    python -m interface.human_oversight --server --host localhost --port 8080

    # Debug mode
    python -m interface.human_oversight --server --debug

Features:
    - Real-time monitoring of parameter expansions
    - Approval workflow for LLM-proposed parameters
    - Quality metrics visualization
    - Audit logging
    - Batch operations
    - WebSocket for live updates
    - REST API for integration

Author: Agent 29
Version: 1.0.0
License: MIT
"""

from .human_oversight import (
    # Core classes
    HumanOversightEngine,

    # Enums
    ExpansionStatus,
    ExpansionSource,
    QualityMetric,

    # Constants
    QUALITY_THRESHOLDS,
)

# Only import Flask app if available
try:
    from .human_oversight import app, socketio, run_server
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Agent 29 - Human-in-Loop Interface"
__all__ = [
    "HumanOversightEngine",
    "ExpansionStatus",
    "ExpansionSource",
    "QualityMetric",
    "QUALITY_THRESHOLDS",
]

if FLASK_AVAILABLE:
    __all__.extend(["app", "socketio", "run_server"])
