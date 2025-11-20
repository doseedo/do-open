"""
Expansion History Tracker - Agent 30
====================================

Comprehensive tracking system for all parameter expansions in the
Musical Program Synthesis self-expanding inverse music generation system.

This module provides:
1. Complete expansion history (what, when, why, results)
2. Analytics on expansion effectiveness
3. Parameter evolution tracking over time
4. Performance metrics for XGBoost models
5. Gap detection and closure monitoring
6. LLM-guided expansion audit trail
7. Synthetic training data tracking
8. Model retraining avoidance verification

Key Features:
- Track every parameter addition with full context
- Measure reconstruction accuracy improvements
- Monitor parameter utilization across genres
- Detect parameter redundancy and conflicts
- Generate expansion effectiveness reports
- Track LLM proposal acceptance/rejection rates
- Monitor system growth toward 800+ parameter goal

Author: Agent 30 - Expansion History Tracker
License: MIT
"""

import json
import sqlite3
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from collections import defaultdict, Counter
import statistics
import pickle


# =============================================================================
# Core Data Structures
# =============================================================================

class ExpansionTrigger(Enum):
    """What triggered the expansion"""
    RECONSTRUCTION_FAILURE = "reconstruction_failure"  # MIDI → params → MIDI failed
    GAP_DETECTION = "gap_detection"  # Feature extractor found missing capability
    LLM_PROPOSAL = "llm_proposal"  # LLM suggested new parameter
    MANUAL_ADDITION = "manual_addition"  # Developer added parameter
    GENRE_REQUIREMENT = "genre_requirement"  # New genre needs new parameters
    AGENT_INITIATIVE = "agent_initiative"  # Agent proactively added parameter
    USER_REQUEST = "user_request"  # User requested new capability


class ExpansionStatus(Enum):
    """Status of an expansion"""
    PROPOSED = "proposed"  # Parameter proposed but not yet implemented
    IMPLEMENTING = "implementing"  # Currently being implemented
    TESTING = "testing"  # Implementation complete, testing in progress
    DEPLOYED = "deployed"  # Successfully deployed and active
    REJECTED = "rejected"  # Proposal rejected (with reason)
    DEPRECATED = "deprecated"  # Parameter replaced or no longer used
    FAILED = "failed"  # Implementation failed


class ParameterImpact(Enum):
    """Impact assessment of parameter"""
    TRANSFORMATIVE = "transformative"  # Fundamentally improved system capability
    HIGH_VALUE = "high_value"  # Significant improvement
    MODERATE = "moderate"  # Useful addition
    LOW_VALUE = "low_value"  # Minimal impact
    REDUNDANT = "redundant"  # Overlaps with existing parameters
    HARMFUL = "harmful"  # Decreased system performance


class ExpansionPhase(Enum):
    """System expansion phases"""
    FOUNDATION = "foundation"  # Initial 165 parameters
    PHASE_1 = "phase_1"  # Target: 515 parameters
    PHASE_2 = "phase_2"  # Target: 800+ parameters
    MATURE = "mature"  # System fully expanded


@dataclass
class ExpansionEvent:
    """
    Single expansion event - adding one or more parameters to the system
    """
    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # What expanded
    parameters_added: List[str] = field(default_factory=list)  # Full parameter paths
    parameter_count: int = 0

    # Why expanded
    trigger: ExpansionTrigger = ExpansionTrigger.MANUAL_ADDITION
    trigger_details: Dict[str, Any] = field(default_factory=dict)

    # Context
    reconstruction_accuracy_before: Optional[float] = None
    reconstruction_accuracy_after: Optional[float] = None
    gap_description: Optional[str] = None
    llm_proposal: Optional[str] = None

    # Implementation
    code_files_modified: List[str] = field(default_factory=list)
    code_lines_added: int = 0
    generator_enhanced: bool = False  # Did we update HarmonyModule generator?
    feature_extractor_enhanced: bool = False  # Did we update deep_feature_extractor?

    # Results
    status: ExpansionStatus = ExpansionStatus.PROPOSED
    impact_assessment: Optional[ParameterImpact] = None

    # Training data
    synthetic_samples_generated: int = 0
    training_data_size_before: int = 0
    training_data_size_after: int = 0

    # Model metrics
    xgboost_models_trained: int = 0
    model_training_time_seconds: float = 0.0
    model_accuracy_metrics: Dict[str, float] = field(default_factory=dict)

    # Validation
    test_midi_files_passed: int = 0
    test_midi_files_failed: int = 0
    musical_validation_passed: bool = False

    # Metadata
    phase: ExpansionPhase = ExpansionPhase.FOUNDATION
    agent_responsible: Optional[str] = None
    notes: str = ""
    error_log: List[str] = field(default_factory=list)

    # Dependencies
    depends_on_events: List[str] = field(default_factory=list)  # event_ids
    enables_events: List[str] = field(default_factory=list)  # event_ids


@dataclass
class ParameterHistoryEntry:
    """
    Complete history of a single parameter
    """
    parameter_path: str
    parameter_name: str

    # Creation
    created_timestamp: datetime
    creation_event_id: str
    creation_trigger: ExpansionTrigger

    # Definition
    parameter_type: str
    default_value: Any
    category: str

    # Evolution
    modification_events: List[str] = field(default_factory=list)  # event_ids
    deprecation_event_id: Optional[str] = None
    replacement_parameter: Optional[str] = None

    # Usage statistics
    times_used_in_generation: int = 0
    times_used_in_reconstruction: int = 0
    genres_utilized: Set[str] = field(default_factory=set)

    # Performance
    average_prediction_accuracy: float = 0.0
    feature_importance_score: float = 0.0
    correlation_with_parameters: Dict[str, float] = field(default_factory=dict)

    # Impact
    reconstruction_improvements: List[float] = field(default_factory=list)
    musical_impact_score: float = 0.0


@dataclass
class GapDetectionRecord:
    """
    Record of a detected gap in system capabilities
    """
    # Gap details (required)
    gap_type: str  # "missing_feature", "reconstruction_failure", "genre_unsupported"
    description: str
    severity: str  # "critical", "high", "medium", "low"

    # Identity (with defaults)
    gap_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected_timestamp: datetime = field(default_factory=datetime.now)

    # Context
    midi_file_triggering: Optional[str] = None
    genre_context: Optional[str] = None
    musical_characteristics: Dict[str, Any] = field(default_factory=dict)

    # Resolution
    resolution_status: str = "open"  # "open", "proposed", "resolved", "wontfix"
    proposed_parameters: List[str] = field(default_factory=list)
    resolution_event_id: Optional[str] = None
    resolution_timestamp: Optional[datetime] = None

    # Metrics
    frequency_detected: int = 1
    affected_midi_files: List[str] = field(default_factory=list)


@dataclass
class LLMProposalRecord:
    """
    Record of an LLM-proposed expansion
    """
    # Proposal (required)
    llm_model: str  # "claude-sonnet-4-5", etc.
    prompt_hash: str
    rationale: str

    # Identity (with defaults)
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Proposal details
    proposed_parameters: List[Dict[str, Any]] = field(default_factory=list)

    # Context
    triggered_by_gap: Optional[str] = None  # gap_id
    reconstruction_failure_context: Optional[Dict[str, Any]] = None

    # Review
    reviewed: bool = False
    review_timestamp: Optional[datetime] = None
    review_decision: Optional[str] = None  # "accepted", "rejected", "modified"
    review_notes: str = ""

    # Implementation
    accepted_parameters: List[str] = field(default_factory=list)
    expansion_event_id: Optional[str] = None

    # Outcome
    outcome_assessment: Optional[ParameterImpact] = None
    effectiveness_score: float = 0.0


@dataclass
class SystemSnapshot:
    """
    Snapshot of system state at a point in time
    """
    # Size metrics (required)
    total_parameters: int
    total_code_lines: int
    generator_lines: int
    feature_extractor_features: int

    # Performance metrics (required)
    average_reconstruction_accuracy: float

    # Model metrics (required)
    xgboost_models_count: int
    average_model_accuracy: float
    training_data_size: int

    # System health (required)
    active_parameters: int
    deprecated_parameters: int
    redundant_parameters: int

    # Progress (required)
    phase: ExpansionPhase
    progress_to_phase_goal: float  # Percentage

    # Identity (with defaults)
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional metrics
    parameters_by_category: Dict[str, int] = field(default_factory=dict)
    parameters_by_type: Dict[str, int] = field(default_factory=dict)
    reconstruction_accuracy_by_genre: Dict[str, float] = field(default_factory=dict)
    estimated_completion: Optional[datetime] = None


@dataclass
class ExpansionAnalytics:
    """
    Analytics results for expansion effectiveness
    """
    # Time range (required)
    analysis_start: datetime
    analysis_end: datetime

    # Expansion metrics (required)
    total_expansions: int
    parameters_added: int
    parameters_deprecated: int
    net_parameter_growth: int
    successful_expansions: int
    failed_expansions: int

    # Optional expansion metrics
    expansions_by_trigger: Dict[str, int] = field(default_factory=dict)
    expansions_by_phase: Dict[str, int] = field(default_factory=dict)
    growth_rate_per_day: float = 0.0
    success_rate: float = 0.0

    # Impact metrics
    average_accuracy_improvement: float = 0.0
    transformative_expansions: int = 0
    high_value_expansions: int = 0
    low_value_expansions: int = 0
    redundant_expansions: int = 0

    # LLM effectiveness
    llm_proposals_total: int = 0
    llm_proposals_accepted: int = 0
    llm_acceptance_rate: float = 0.0
    llm_average_effectiveness: float = 0.0

    # Gap closure
    gaps_detected: int = 0
    gaps_resolved: int = 0
    gap_resolution_rate: float = 0.0
    average_gap_resolution_time_days: float = 0.0

    # Efficiency metrics
    average_parameters_per_expansion: float = 0.0
    average_code_lines_per_parameter: float = 0.0
    average_training_time_per_model: float = 0.0


# =============================================================================
# Main Expansion History Tracker
# =============================================================================

class ExpansionHistoryTracker:
    """
    Central tracking system for all parameter expansions

    Responsibilities:
    1. Log every expansion event with full context
    2. Track parameter evolution over time
    3. Monitor gap detection and closure
    4. Track LLM proposal effectiveness
    5. Generate analytics and reports
    6. Maintain SQLite database for persistence
    7. Provide query interface for historical data
    """

    def __init__(self, db_path: str = None, json_backup_path: str = None):
        """
        Initialize the tracker

        Args:
            db_path: Path to SQLite database (default: tracking/expansion_history.db)
            json_backup_path: Path for JSON backup (default: tracking/expansion_history.json)
        """
        if db_path is None:
            db_path = str(Path(__file__).parent / "expansion_history.db")
        if json_backup_path is None:
            json_backup_path = str(Path(__file__).parent / "expansion_history.json")

        self.db_path = db_path
        self.json_backup_path = json_backup_path

        # In-memory caches
        self.events: Dict[str, ExpansionEvent] = {}
        self.parameter_history: Dict[str, ParameterHistoryEntry] = {}
        self.gap_records: Dict[str, GapDetectionRecord] = {}
        self.llm_proposals: Dict[str, LLMProposalRecord] = {}
        self.snapshots: List[SystemSnapshot] = []

        # For in-memory databases, maintain a single connection
        self._conn = None
        if db_path == ':memory:':
            self._conn = sqlite3.connect(db_path)

        # Initialize database
        self._initialize_database()

        # Load existing data (skip for in-memory databases)
        if self.db_path != ':memory:':
            self._load_from_database()

    def _get_connection(self):
        """Get database connection (reuse for in-memory databases)"""
        if self._conn:
            return self._conn, False  # Don't close
        return sqlite3.connect(self.db_path), True  # Should close

    def _initialize_database(self):
        """Initialize SQLite database schema"""
        conn, should_close = self._get_connection()
        cursor = conn.cursor()

        # Expansion events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expansion_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                parameters_added TEXT,
                parameter_count INTEGER,
                trigger TEXT,
                trigger_details TEXT,
                reconstruction_accuracy_before REAL,
                reconstruction_accuracy_after REAL,
                gap_description TEXT,
                llm_proposal TEXT,
                code_files_modified TEXT,
                code_lines_added INTEGER,
                generator_enhanced BOOLEAN,
                feature_extractor_enhanced BOOLEAN,
                status TEXT,
                impact_assessment TEXT,
                synthetic_samples_generated INTEGER,
                training_data_size_before INTEGER,
                training_data_size_after INTEGER,
                xgboost_models_trained INTEGER,
                model_training_time_seconds REAL,
                model_accuracy_metrics TEXT,
                test_midi_files_passed INTEGER,
                test_midi_files_failed INTEGER,
                musical_validation_passed BOOLEAN,
                phase TEXT,
                agent_responsible TEXT,
                notes TEXT,
                error_log TEXT,
                depends_on_events TEXT,
                enables_events TEXT
            )
        """)

        # Parameter history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameter_history (
                parameter_path TEXT PRIMARY KEY,
                parameter_name TEXT,
                created_timestamp TEXT,
                creation_event_id TEXT,
                creation_trigger TEXT,
                parameter_type TEXT,
                default_value TEXT,
                category TEXT,
                modification_events TEXT,
                deprecation_event_id TEXT,
                replacement_parameter TEXT,
                times_used_in_generation INTEGER,
                times_used_in_reconstruction INTEGER,
                genres_utilized TEXT,
                average_prediction_accuracy REAL,
                feature_importance_score REAL,
                correlation_with_parameters TEXT,
                reconstruction_improvements TEXT,
                musical_impact_score REAL
            )
        """)

        # Gap detection table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gap_detection (
                gap_id TEXT PRIMARY KEY,
                detected_timestamp TEXT,
                gap_type TEXT,
                description TEXT,
                severity TEXT,
                midi_file_triggering TEXT,
                genre_context TEXT,
                musical_characteristics TEXT,
                resolution_status TEXT,
                proposed_parameters TEXT,
                resolution_event_id TEXT,
                resolution_timestamp TEXT,
                frequency_detected INTEGER,
                affected_midi_files TEXT
            )
        """)

        # LLM proposals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_proposals (
                proposal_id TEXT PRIMARY KEY,
                timestamp TEXT,
                llm_model TEXT,
                prompt_hash TEXT,
                proposed_parameters TEXT,
                rationale TEXT,
                triggered_by_gap TEXT,
                reconstruction_failure_context TEXT,
                reviewed BOOLEAN,
                review_timestamp TEXT,
                review_decision TEXT,
                review_notes TEXT,
                accepted_parameters TEXT,
                expansion_event_id TEXT,
                outcome_assessment TEXT,
                effectiveness_score REAL
            )
        """)

        # System snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                timestamp TEXT,
                total_parameters INTEGER,
                parameters_by_category TEXT,
                parameters_by_type TEXT,
                total_code_lines INTEGER,
                generator_lines INTEGER,
                feature_extractor_features INTEGER,
                average_reconstruction_accuracy REAL,
                reconstruction_accuracy_by_genre TEXT,
                xgboost_models_count INTEGER,
                average_model_accuracy REAL,
                training_data_size INTEGER,
                active_parameters INTEGER,
                deprecated_parameters INTEGER,
                redundant_parameters INTEGER,
                phase TEXT,
                progress_to_phase_goal REAL,
                estimated_completion TEXT
            )
        """)

        # Create indices for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON expansion_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_phase ON expansion_events(phase)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON expansion_events(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_gaps_status ON gap_detection(resolution_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_proposals_reviewed ON llm_proposals(reviewed)")

        conn.commit()
        if should_close:
            conn.close()

    def _load_from_database(self):
        """Load existing data from database into memory"""
        conn, should_close = self._get_connection()
        cursor = conn.cursor()

        # Load events
        cursor.execute("SELECT * FROM expansion_events")
        for row in cursor.fetchall():
            event = self._row_to_expansion_event(row)
            self.events[event.event_id] = event

        # Load parameter history
        cursor.execute("SELECT * FROM parameter_history")
        for row in cursor.fetchall():
            param = self._row_to_parameter_history(row)
            self.parameter_history[param.parameter_path] = param

        # Load gaps
        cursor.execute("SELECT * FROM gap_detection")
        for row in cursor.fetchall():
            gap = self._row_to_gap_record(row)
            self.gap_records[gap.gap_id] = gap

        # Load LLM proposals
        cursor.execute("SELECT * FROM llm_proposals")
        for row in cursor.fetchall():
            proposal = self._row_to_llm_proposal(row)
            self.llm_proposals[proposal.proposal_id] = proposal

        # Load snapshots
        cursor.execute("SELECT * FROM system_snapshots ORDER BY timestamp")
        for row in cursor.fetchall():
            snapshot = self._row_to_system_snapshot(row)
            self.snapshots.append(snapshot)

        if should_close:
            conn.close()

    def _row_to_expansion_event(self, row) -> ExpansionEvent:
        """Convert database row to ExpansionEvent"""
        return ExpansionEvent(
            event_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            parameters_added=json.loads(row[2]) if row[2] else [],
            parameter_count=row[3] or 0,
            trigger=ExpansionTrigger(row[4]) if row[4] else ExpansionTrigger.MANUAL_ADDITION,
            trigger_details=json.loads(row[5]) if row[5] else {},
            reconstruction_accuracy_before=row[6],
            reconstruction_accuracy_after=row[7],
            gap_description=row[8],
            llm_proposal=row[9],
            code_files_modified=json.loads(row[10]) if row[10] else [],
            code_lines_added=row[11] or 0,
            generator_enhanced=bool(row[12]) if row[12] is not None else False,
            feature_extractor_enhanced=bool(row[13]) if row[13] is not None else False,
            status=ExpansionStatus(row[14]) if row[14] else ExpansionStatus.PROPOSED,
            impact_assessment=ParameterImpact(row[15]) if row[15] else None,
            synthetic_samples_generated=row[16] or 0,
            training_data_size_before=row[17] or 0,
            training_data_size_after=row[18] or 0,
            xgboost_models_trained=row[19] or 0,
            model_training_time_seconds=row[20] or 0.0,
            model_accuracy_metrics=json.loads(row[21]) if row[21] else {},
            test_midi_files_passed=row[22] or 0,
            test_midi_files_failed=row[23] or 0,
            musical_validation_passed=bool(row[24]) if row[24] is not None else False,
            phase=ExpansionPhase(row[25]) if row[25] else ExpansionPhase.FOUNDATION,
            agent_responsible=row[26],
            notes=row[27] or "",
            error_log=json.loads(row[28]) if row[28] else [],
            depends_on_events=json.loads(row[29]) if row[29] else [],
            enables_events=json.loads(row[30]) if row[30] else []
        )

    def _row_to_parameter_history(self, row) -> ParameterHistoryEntry:
        """Convert database row to ParameterHistoryEntry"""
        return ParameterHistoryEntry(
            parameter_path=row[0],
            parameter_name=row[1],
            created_timestamp=datetime.fromisoformat(row[2]),
            creation_event_id=row[3],
            creation_trigger=ExpansionTrigger(row[4]) if row[4] else ExpansionTrigger.MANUAL_ADDITION,
            parameter_type=row[5] or "unknown",
            default_value=json.loads(row[6]) if row[6] else None,
            category=row[7] or "other",
            modification_events=json.loads(row[8]) if row[8] else [],
            deprecation_event_id=row[9],
            replacement_parameter=row[10],
            times_used_in_generation=row[11] or 0,
            times_used_in_reconstruction=row[12] or 0,
            genres_utilized=set(json.loads(row[13])) if row[13] else set(),
            average_prediction_accuracy=row[14] or 0.0,
            feature_importance_score=row[15] or 0.0,
            correlation_with_parameters=json.loads(row[16]) if row[16] else {},
            reconstruction_improvements=json.loads(row[17]) if row[17] else [],
            musical_impact_score=row[18] or 0.0
        )

    def _row_to_gap_record(self, row) -> GapDetectionRecord:
        """Convert database row to GapDetectionRecord"""
        return GapDetectionRecord(
            gap_id=row[0],
            detected_timestamp=datetime.fromisoformat(row[1]),
            gap_type=row[2],
            description=row[3],
            severity=row[4],
            midi_file_triggering=row[5],
            genre_context=row[6],
            musical_characteristics=json.loads(row[7]) if row[7] else {},
            resolution_status=row[8],
            proposed_parameters=json.loads(row[9]) if row[9] else [],
            resolution_event_id=row[10],
            resolution_timestamp=datetime.fromisoformat(row[11]) if row[11] else None,
            frequency_detected=row[12] or 1,
            affected_midi_files=json.loads(row[13]) if row[13] else []
        )

    def _row_to_llm_proposal(self, row) -> LLMProposalRecord:
        """Convert database row to LLMProposalRecord"""
        return LLMProposalRecord(
            proposal_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            llm_model=row[2],
            prompt_hash=row[3],
            proposed_parameters=json.loads(row[4]) if row[4] else [],
            rationale=row[5],
            triggered_by_gap=row[6],
            reconstruction_failure_context=json.loads(row[7]) if row[7] else None,
            reviewed=bool(row[8]) if row[8] is not None else False,
            review_timestamp=datetime.fromisoformat(row[9]) if row[9] else None,
            review_decision=row[10],
            review_notes=row[11] or "",
            accepted_parameters=json.loads(row[12]) if row[12] else [],
            expansion_event_id=row[13],
            outcome_assessment=ParameterImpact(row[14]) if row[14] else None,
            effectiveness_score=row[15] or 0.0
        )

    def _row_to_system_snapshot(self, row) -> SystemSnapshot:
        """Convert database row to SystemSnapshot"""
        return SystemSnapshot(
            snapshot_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            total_parameters=row[2],
            parameters_by_category=json.loads(row[3]) if row[3] else {},
            parameters_by_type=json.loads(row[4]) if row[4] else {},
            total_code_lines=row[5],
            generator_lines=row[6],
            feature_extractor_features=row[7],
            average_reconstruction_accuracy=row[8],
            reconstruction_accuracy_by_genre=json.loads(row[9]) if row[9] else {},
            xgboost_models_count=row[10],
            average_model_accuracy=row[11],
            training_data_size=row[12],
            active_parameters=row[13],
            deprecated_parameters=row[14],
            redundant_parameters=row[15],
            phase=ExpansionPhase(row[16]) if row[16] else ExpansionPhase.FOUNDATION,
            progress_to_phase_goal=row[17] or 0.0,
            estimated_completion=datetime.fromisoformat(row[18]) if row[18] else None
        )

    # =========================================================================
    # Core Logging Methods
    # =========================================================================

    def log_expansion_event(self, event: ExpansionEvent) -> str:
        """
        Log a new expansion event

        Args:
            event: ExpansionEvent to log

        Returns:
            event_id of the logged event
        """
        # Add to memory cache
        self.events[event.event_id] = event

        # Update parameter history
        for param_path in event.parameters_added:
            if param_path not in self.parameter_history:
                self.parameter_history[param_path] = ParameterHistoryEntry(
                    parameter_path=param_path,
                    parameter_name=param_path.split('.')[-1],
                    created_timestamp=event.timestamp,
                    creation_event_id=event.event_id,
                    creation_trigger=event.trigger,
                    parameter_type="unknown",  # Will be updated later
                    default_value=None,
                    category="unknown"
                )
            else:
                # Parameter modified
                self.parameter_history[param_path].modification_events.append(event.event_id)

        # Save to database
        self._save_event_to_db(event)

        # Backup to JSON
        self._backup_to_json()

        return event.event_id

    def log_gap_detection(self, gap: GapDetectionRecord) -> str:
        """
        Log a detected gap in system capabilities

        Args:
            gap: GapDetectionRecord to log

        Returns:
            gap_id of the logged gap
        """
        self.gap_records[gap.gap_id] = gap
        self._save_gap_to_db(gap)
        self._backup_to_json()
        return gap.gap_id

    def log_llm_proposal(self, proposal: LLMProposalRecord) -> str:
        """
        Log an LLM proposal for expansion

        Args:
            proposal: LLMProposalRecord to log

        Returns:
            proposal_id of the logged proposal
        """
        self.llm_proposals[proposal.proposal_id] = proposal
        self._save_llm_proposal_to_db(proposal)
        self._backup_to_json()
        return proposal.proposal_id

    def capture_system_snapshot(self, snapshot: SystemSnapshot) -> str:
        """
        Capture a snapshot of current system state

        Args:
            snapshot: SystemSnapshot to log

        Returns:
            snapshot_id of the logged snapshot
        """
        self.snapshots.append(snapshot)
        self._save_snapshot_to_db(snapshot)
        self._backup_to_json()
        return snapshot.snapshot_id

    # =========================================================================
    # Update Methods
    # =========================================================================

    def update_event_status(self, event_id: str, status: ExpansionStatus, notes: str = ""):
        """Update the status of an expansion event"""
        if event_id in self.events:
            self.events[event_id].status = status
            if notes:
                self.events[event_id].notes += f"\n[{datetime.now()}] {notes}"
            self._save_event_to_db(self.events[event_id])

    def update_event_impact(self, event_id: str, impact: ParameterImpact,
                           accuracy_improvement: float = None):
        """Update the impact assessment of an expansion event"""
        if event_id in self.events:
            self.events[event_id].impact_assessment = impact
            if accuracy_improvement is not None:
                self.events[event_id].reconstruction_accuracy_after = \
                    (self.events[event_id].reconstruction_accuracy_before or 0.0) + accuracy_improvement
            self._save_event_to_db(self.events[event_id])

    def update_gap_status(self, gap_id: str, status: str, resolution_event_id: str = None):
        """Update the resolution status of a gap"""
        if gap_id in self.gap_records:
            self.gap_records[gap_id].resolution_status = status
            if resolution_event_id:
                self.gap_records[gap_id].resolution_event_id = resolution_event_id
                self.gap_records[gap_id].resolution_timestamp = datetime.now()
            self._save_gap_to_db(self.gap_records[gap_id])

    def review_llm_proposal(self, proposal_id: str, decision: str, notes: str = "",
                           accepted_params: List[str] = None):
        """Review and decide on an LLM proposal"""
        if proposal_id in self.llm_proposals:
            proposal = self.llm_proposals[proposal_id]
            proposal.reviewed = True
            proposal.review_timestamp = datetime.now()
            proposal.review_decision = decision
            proposal.review_notes = notes
            if accepted_params:
                proposal.accepted_parameters = accepted_params
            self._save_llm_proposal_to_db(proposal)

    def update_parameter_usage(self, parameter_path: str,
                              used_in_generation: bool = False,
                              used_in_reconstruction: bool = False,
                              genre: str = None):
        """Update usage statistics for a parameter"""
        if parameter_path in self.parameter_history:
            param = self.parameter_history[parameter_path]
            if used_in_generation:
                param.times_used_in_generation += 1
            if used_in_reconstruction:
                param.times_used_in_reconstruction += 1
            if genre:
                param.genres_utilized.add(genre)
            self._save_parameter_history_to_db(param)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_events_by_phase(self, phase: ExpansionPhase) -> List[ExpansionEvent]:
        """Get all expansion events for a specific phase"""
        return [e for e in self.events.values() if e.phase == phase]

    def get_events_by_trigger(self, trigger: ExpansionTrigger) -> List[ExpansionEvent]:
        """Get all expansion events triggered by a specific cause"""
        return [e for e in self.events.values() if e.trigger == trigger]

    def get_events_by_status(self, status: ExpansionStatus) -> List[ExpansionEvent]:
        """Get all expansion events with a specific status"""
        return [e for e in self.events.values() if e.status == status]

    def get_events_in_date_range(self, start: datetime, end: datetime) -> List[ExpansionEvent]:
        """Get all expansion events within a date range"""
        return [e for e in self.events.values() if start <= e.timestamp <= end]

    def get_parameter_history(self, parameter_path: str) -> Optional[ParameterHistoryEntry]:
        """Get complete history of a parameter"""
        return self.parameter_history.get(parameter_path)

    def get_parameters_by_trigger(self, trigger: ExpansionTrigger) -> List[ParameterHistoryEntry]:
        """Get all parameters created by a specific trigger"""
        return [p for p in self.parameter_history.values() if p.creation_trigger == trigger]

    def get_open_gaps(self) -> List[GapDetectionRecord]:
        """Get all unresolved gaps"""
        return [g for g in self.gap_records.values() if g.resolution_status == "open"]

    def get_unreviewed_proposals(self) -> List[LLMProposalRecord]:
        """Get all LLM proposals awaiting review"""
        return [p for p in self.llm_proposals.values() if not p.reviewed]

    def get_latest_snapshot(self) -> Optional[SystemSnapshot]:
        """Get the most recent system snapshot"""
        return self.snapshots[-1] if self.snapshots else None

    # =========================================================================
    # Analytics Methods
    # =========================================================================

    def generate_analytics(self, start_date: datetime = None,
                          end_date: datetime = None) -> ExpansionAnalytics:
        """
        Generate comprehensive analytics for expansion effectiveness

        Args:
            start_date: Start of analysis period (default: all time)
            end_date: End of analysis period (default: now)

        Returns:
            ExpansionAnalytics object with comprehensive metrics
        """
        if start_date is None:
            start_date = min((e.timestamp for e in self.events.values()),
                           default=datetime.now())
        if end_date is None:
            end_date = datetime.now()

        # Filter events in date range
        events = self.get_events_in_date_range(start_date, end_date)

        analytics = ExpansionAnalytics(
            analysis_start=start_date,
            analysis_end=end_date,
            total_expansions=len(events)
        )

        # Expansion metrics
        for event in events:
            trigger_name = event.trigger.value
            analytics.expansions_by_trigger[trigger_name] = \
                analytics.expansions_by_trigger.get(trigger_name, 0) + 1

            phase_name = event.phase.value
            analytics.expansions_by_phase[phase_name] = \
                analytics.expansions_by_phase.get(phase_name, 0) + 1

        # Parameter growth
        analytics.parameters_added = sum(e.parameter_count for e in events)
        analytics.parameters_deprecated = len([
            p for p in self.parameter_history.values()
            if p.deprecation_event_id and
            start_date <= self.events[p.deprecation_event_id].timestamp <= end_date
        ])
        analytics.net_parameter_growth = analytics.parameters_added - analytics.parameters_deprecated

        days = (end_date - start_date).days or 1
        analytics.growth_rate_per_day = analytics.net_parameter_growth / days

        # Success metrics
        analytics.successful_expansions = len([
            e for e in events if e.status == ExpansionStatus.DEPLOYED
        ])
        analytics.failed_expansions = len([
            e for e in events if e.status == ExpansionStatus.FAILED
        ])
        analytics.success_rate = (analytics.successful_expansions / len(events)
                                 if events else 0.0)

        # Impact metrics
        accuracy_improvements = [
            e.reconstruction_accuracy_after - e.reconstruction_accuracy_before
            for e in events
            if e.reconstruction_accuracy_before is not None
            and e.reconstruction_accuracy_after is not None
        ]
        analytics.average_accuracy_improvement = (
            statistics.mean(accuracy_improvements) if accuracy_improvements else 0.0
        )

        for event in events:
            if event.impact_assessment == ParameterImpact.TRANSFORMATIVE:
                analytics.transformative_expansions += 1
            elif event.impact_assessment == ParameterImpact.HIGH_VALUE:
                analytics.high_value_expansions += 1
            elif event.impact_assessment == ParameterImpact.LOW_VALUE:
                analytics.low_value_expansions += 1
            elif event.impact_assessment == ParameterImpact.REDUNDANT:
                analytics.redundant_expansions += 1

        # LLM effectiveness
        proposals = [p for p in self.llm_proposals.values()
                    if start_date <= p.timestamp <= end_date]
        analytics.llm_proposals_total = len(proposals)
        analytics.llm_proposals_accepted = len([
            p for p in proposals if p.review_decision == "accepted"
        ])
        analytics.llm_acceptance_rate = (
            analytics.llm_proposals_accepted / analytics.llm_proposals_total
            if analytics.llm_proposals_total > 0 else 0.0
        )

        effectiveness_scores = [p.effectiveness_score for p in proposals
                               if p.effectiveness_score > 0]
        analytics.llm_average_effectiveness = (
            statistics.mean(effectiveness_scores) if effectiveness_scores else 0.0
        )

        # Gap closure
        gaps = [g for g in self.gap_records.values()
               if start_date <= g.detected_timestamp <= end_date]
        analytics.gaps_detected = len(gaps)
        analytics.gaps_resolved = len([g for g in gaps
                                      if g.resolution_status == "resolved"])
        analytics.gap_resolution_rate = (
            analytics.gaps_resolved / analytics.gaps_detected
            if analytics.gaps_detected > 0 else 0.0
        )

        resolution_times = [
            (g.resolution_timestamp - g.detected_timestamp).days
            for g in gaps
            if g.resolution_timestamp
        ]
        analytics.average_gap_resolution_time_days = (
            statistics.mean(resolution_times) if resolution_times else 0.0
        )

        # Efficiency metrics
        analytics.average_parameters_per_expansion = (
            analytics.parameters_added / len(events) if events else 0.0
        )

        code_lines = [e.code_lines_added for e in events if e.code_lines_added > 0]
        params = [e.parameter_count for e in events if e.parameter_count > 0]
        if code_lines and params:
            analytics.average_code_lines_per_parameter = sum(code_lines) / sum(params)

        training_times = [e.model_training_time_seconds for e in events
                         if e.model_training_time_seconds > 0]
        analytics.average_training_time_per_model = (
            statistics.mean(training_times) if training_times else 0.0
        )

        return analytics

    def generate_parameter_evolution_report(self, parameter_path: str) -> Dict[str, Any]:
        """
        Generate a detailed evolution report for a specific parameter

        Args:
            parameter_path: Full path of parameter (e.g., "harmony.jazz.voicing_type")

        Returns:
            Dictionary with complete parameter evolution data
        """
        if parameter_path not in self.parameter_history:
            return {"error": f"Parameter {parameter_path} not found"}

        param = self.parameter_history[parameter_path]
        creation_event = self.events.get(param.creation_event_id)

        report = {
            "parameter_path": parameter_path,
            "parameter_name": param.parameter_name,
            "created": param.created_timestamp.isoformat(),
            "age_days": (datetime.now() - param.created_timestamp).days,
            "creation_trigger": param.creation_trigger.value,
            "creation_event": {
                "event_id": param.creation_event_id,
                "trigger_details": creation_event.trigger_details if creation_event else {},
                "agent_responsible": creation_event.agent_responsible if creation_event else None
            },
            "type": param.parameter_type,
            "default_value": param.default_value,
            "category": param.category,
            "usage": {
                "times_used_in_generation": param.times_used_in_generation,
                "times_used_in_reconstruction": param.times_used_in_reconstruction,
                "genres_utilized": list(param.genres_utilized),
                "total_usage": param.times_used_in_generation + param.times_used_in_reconstruction
            },
            "performance": {
                "average_prediction_accuracy": param.average_prediction_accuracy,
                "feature_importance_score": param.feature_importance_score,
                "musical_impact_score": param.musical_impact_score
            },
            "modifications": [
                {
                    "event_id": event_id,
                    "timestamp": self.events[event_id].timestamp.isoformat(),
                    "notes": self.events[event_id].notes
                }
                for event_id in param.modification_events
                if event_id in self.events
            ],
            "reconstruction_improvements": param.reconstruction_improvements,
            "correlations": param.correlation_with_parameters,
            "deprecated": param.deprecation_event_id is not None,
            "replacement": param.replacement_parameter
        }

        return report

    def generate_expansion_effectiveness_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report on expansion effectiveness

        Returns:
            Dictionary with detailed effectiveness metrics and recommendations
        """
        analytics = self.generate_analytics()

        # Calculate phase progress
        latest_snapshot = self.get_latest_snapshot()
        phase_targets = {
            ExpansionPhase.FOUNDATION: 165,
            ExpansionPhase.PHASE_1: 515,
            ExpansionPhase.PHASE_2: 800
        }

        current_params = latest_snapshot.total_parameters if latest_snapshot else 0
        current_phase = latest_snapshot.phase if latest_snapshot else ExpansionPhase.FOUNDATION
        target = phase_targets.get(current_phase, 800)
        progress = (current_params / target) * 100 if target > 0 else 0

        # Identify most effective expansion triggers
        trigger_effectiveness = {}
        for trigger in ExpansionTrigger:
            events = self.get_events_by_trigger(trigger)
            if events:
                avg_impact = statistics.mean([
                    self._impact_to_score(e.impact_assessment)
                    for e in events if e.impact_assessment
                ]) if any(e.impact_assessment for e in events) else 0
                trigger_effectiveness[trigger.value] = {
                    "count": len(events),
                    "average_impact": avg_impact,
                    "success_rate": len([e for e in events
                                       if e.status == ExpansionStatus.DEPLOYED]) / len(events)
                }

        # Identify underutilized parameters
        underutilized = sorted(
            [p for p in self.parameter_history.values()
             if (p.times_used_in_generation + p.times_used_in_reconstruction) < 10
             and (datetime.now() - p.created_timestamp).days > 30],
            key=lambda p: p.times_used_in_generation + p.times_used_in_reconstruction
        )[:20]

        # Identify high-impact parameters
        high_impact = sorted(
            self.parameter_history.values(),
            key=lambda p: p.musical_impact_score,
            reverse=True
        )[:20]

        report = {
            "generated_at": datetime.now().isoformat(),
            "system_overview": {
                "current_parameters": current_params,
                "current_phase": current_phase.value,
                "target_for_phase": target,
                "progress_percentage": round(progress, 2),
                "parameters_to_goal": target - current_params
            },
            "expansion_summary": {
                "total_expansions": analytics.total_expansions,
                "success_rate": round(analytics.success_rate * 100, 2),
                "average_accuracy_improvement": round(analytics.average_accuracy_improvement * 100, 2),
                "parameters_added": analytics.parameters_added,
                "net_growth": analytics.net_parameter_growth,
                "growth_rate_per_day": round(analytics.growth_rate_per_day, 2)
            },
            "effectiveness_by_trigger": trigger_effectiveness,
            "impact_distribution": {
                "transformative": analytics.transformative_expansions,
                "high_value": analytics.high_value_expansions,
                "low_value": analytics.low_value_expansions,
                "redundant": analytics.redundant_expansions
            },
            "llm_effectiveness": {
                "total_proposals": analytics.llm_proposals_total,
                "acceptance_rate": round(analytics.llm_acceptance_rate * 100, 2),
                "average_effectiveness": round(analytics.llm_average_effectiveness, 2)
            },
            "gap_closure": {
                "gaps_detected": analytics.gaps_detected,
                "gaps_resolved": analytics.gaps_resolved,
                "resolution_rate": round(analytics.gap_resolution_rate * 100, 2),
                "average_resolution_time_days": round(analytics.average_gap_resolution_time_days, 2),
                "open_gaps": len(self.get_open_gaps())
            },
            "efficiency_metrics": {
                "average_parameters_per_expansion": round(analytics.average_parameters_per_expansion, 2),
                "average_code_lines_per_parameter": round(analytics.average_code_lines_per_parameter, 2),
                "average_training_time_per_model_seconds": round(analytics.average_training_time_per_model, 2)
            },
            "underutilized_parameters": [
                {
                    "path": p.parameter_path,
                    "age_days": (datetime.now() - p.created_timestamp).days,
                    "usage_count": p.times_used_in_generation + p.times_used_in_reconstruction,
                    "creation_trigger": p.creation_trigger.value
                }
                for p in underutilized[:10]
            ],
            "high_impact_parameters": [
                {
                    "path": p.parameter_path,
                    "impact_score": round(p.musical_impact_score, 2),
                    "usage_count": p.times_used_in_generation + p.times_used_in_reconstruction,
                    "accuracy": round(p.average_prediction_accuracy, 2)
                }
                for p in high_impact[:10]
            ],
            "recommendations": self._generate_recommendations(analytics, underutilized, high_impact)
        }

        return report

    def _impact_to_score(self, impact: Optional[ParameterImpact]) -> float:
        """Convert impact assessment to numerical score"""
        if impact is None:
            return 0.0
        scores = {
            ParameterImpact.TRANSFORMATIVE: 5.0,
            ParameterImpact.HIGH_VALUE: 4.0,
            ParameterImpact.MODERATE: 3.0,
            ParameterImpact.LOW_VALUE: 2.0,
            ParameterImpact.REDUNDANT: 1.0,
            ParameterImpact.HARMFUL: 0.0
        }
        return scores.get(impact, 0.0)

    def _generate_recommendations(self, analytics: ExpansionAnalytics,
                                 underutilized: List[ParameterHistoryEntry],
                                 high_impact: List[ParameterHistoryEntry]) -> List[str]:
        """Generate actionable recommendations based on analytics"""
        recommendations = []

        # Success rate recommendations
        if analytics.success_rate < 0.7:
            recommendations.append(
                f"⚠️ Expansion success rate is {analytics.success_rate*100:.1f}%. "
                "Consider more thorough testing before deployment."
            )

        # Growth rate recommendations
        if analytics.growth_rate_per_day < 1.0:
            recommendations.append(
                f"📈 Current growth rate is {analytics.growth_rate_per_day:.2f} params/day. "
                "Consider accelerating expansion to meet phase goals."
            )
        elif analytics.growth_rate_per_day > 10.0:
            recommendations.append(
                f"⚡ Rapid expansion detected ({analytics.growth_rate_per_day:.2f} params/day). "
                "Ensure adequate testing and validation."
            )

        # LLM effectiveness recommendations
        if analytics.llm_acceptance_rate < 0.5:
            recommendations.append(
                f"🤖 LLM proposal acceptance rate is low ({analytics.llm_acceptance_rate*100:.1f}%). "
                "Consider refining prompts or providing more context."
            )
        elif analytics.llm_acceptance_rate > 0.8:
            recommendations.append(
                f"✨ LLM proposals are highly effective ({analytics.llm_acceptance_rate*100:.1f}% accepted). "
                "Consider increasing LLM-guided expansion."
            )

        # Gap closure recommendations
        if analytics.gap_resolution_rate < 0.6:
            recommendations.append(
                f"🔍 Gap resolution rate is {analytics.gap_resolution_rate*100:.1f}%. "
                "Prioritize resolving open gaps to improve system coverage."
            )

        # Underutilized parameters
        if len(underutilized) > 20:
            recommendations.append(
                f"📊 {len(underutilized)} parameters are underutilized. "
                "Consider deprecating or better integrating these parameters."
            )

        # Redundancy check
        if analytics.redundant_expansions > analytics.high_value_expansions:
            recommendations.append(
                "⚠️ More redundant than high-value expansions detected. "
                "Improve gap analysis to avoid duplicate capabilities."
            )

        # Efficiency recommendations
        if analytics.average_code_lines_per_parameter > 100:
            recommendations.append(
                f"💻 Average {analytics.average_code_lines_per_parameter:.0f} lines per parameter. "
                "Consider refactoring for more efficient parameter integration."
            )

        return recommendations

    # =========================================================================
    # Persistence Methods
    # =========================================================================

    def _save_event_to_db(self, event: ExpansionEvent):
        """Save expansion event to database"""
        conn, should_close = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO expansion_events VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            event.event_id,
            event.timestamp.isoformat(),
            json.dumps(event.parameters_added),
            event.parameter_count,
            event.trigger.value,
            json.dumps(event.trigger_details),
            event.reconstruction_accuracy_before,
            event.reconstruction_accuracy_after,
            event.gap_description,
            event.llm_proposal,
            json.dumps(event.code_files_modified),
            event.code_lines_added,
            event.generator_enhanced,
            event.feature_extractor_enhanced,
            event.status.value,
            event.impact_assessment.value if event.impact_assessment else None,
            event.synthetic_samples_generated,
            event.training_data_size_before,
            event.training_data_size_after,
            event.xgboost_models_trained,
            event.model_training_time_seconds,
            json.dumps(event.model_accuracy_metrics),
            event.test_midi_files_passed,
            event.test_midi_files_failed,
            event.musical_validation_passed,
            event.phase.value,
            event.agent_responsible,
            event.notes,
            json.dumps(event.error_log),
            json.dumps(event.depends_on_events),
            json.dumps(event.enables_events)
        ))

        conn.commit()
        if should_close:
            conn.close()

    def _save_parameter_history_to_db(self, param: ParameterHistoryEntry):
        """Save parameter history to database"""
        conn, should_close = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO parameter_history VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            param.parameter_path,
            param.parameter_name,
            param.created_timestamp.isoformat(),
            param.creation_event_id,
            param.creation_trigger.value,
            param.parameter_type,
            json.dumps(param.default_value),
            param.category,
            json.dumps(param.modification_events),
            param.deprecation_event_id,
            param.replacement_parameter,
            param.times_used_in_generation,
            param.times_used_in_reconstruction,
            json.dumps(list(param.genres_utilized)),
            param.average_prediction_accuracy,
            param.feature_importance_score,
            json.dumps(param.correlation_with_parameters),
            json.dumps(param.reconstruction_improvements),
            param.musical_impact_score
        ))

        conn.commit()
        if should_close:
            conn.close()

    def _save_gap_to_db(self, gap: GapDetectionRecord):
        """Save gap detection record to database"""
        conn, should_close = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO gap_detection VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            gap.gap_id,
            gap.detected_timestamp.isoformat(),
            gap.gap_type,
            gap.description,
            gap.severity,
            gap.midi_file_triggering,
            gap.genre_context,
            json.dumps(gap.musical_characteristics),
            gap.resolution_status,
            json.dumps(gap.proposed_parameters),
            gap.resolution_event_id,
            gap.resolution_timestamp.isoformat() if gap.resolution_timestamp else None,
            gap.frequency_detected,
            json.dumps(gap.affected_midi_files)
        ))

        conn.commit()
        if should_close:
            conn.close()

    def _save_llm_proposal_to_db(self, proposal: LLMProposalRecord):
        """Save LLM proposal to database"""
        conn, should_close = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO llm_proposals VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            proposal.proposal_id,
            proposal.timestamp.isoformat(),
            proposal.llm_model,
            proposal.prompt_hash,
            json.dumps(proposal.proposed_parameters),
            proposal.rationale,
            proposal.triggered_by_gap,
            json.dumps(proposal.reconstruction_failure_context),
            proposal.reviewed,
            proposal.review_timestamp.isoformat() if proposal.review_timestamp else None,
            proposal.review_decision,
            proposal.review_notes,
            json.dumps(proposal.accepted_parameters),
            proposal.expansion_event_id,
            proposal.outcome_assessment.value if proposal.outcome_assessment else None,
            proposal.effectiveness_score
        ))

        conn.commit()
        if should_close:
            conn.close()

    def _save_snapshot_to_db(self, snapshot: SystemSnapshot):
        """Save system snapshot to database"""
        conn, should_close = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO system_snapshots VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            snapshot.snapshot_id,
            snapshot.timestamp.isoformat(),
            snapshot.total_parameters,
            json.dumps(snapshot.parameters_by_category),
            json.dumps(snapshot.parameters_by_type),
            snapshot.total_code_lines,
            snapshot.generator_lines,
            snapshot.feature_extractor_features,
            snapshot.average_reconstruction_accuracy,
            json.dumps(snapshot.reconstruction_accuracy_by_genre),
            snapshot.xgboost_models_count,
            snapshot.average_model_accuracy,
            snapshot.training_data_size,
            snapshot.active_parameters,
            snapshot.deprecated_parameters,
            snapshot.redundant_parameters,
            snapshot.phase.value,
            snapshot.progress_to_phase_goal,
            snapshot.estimated_completion.isoformat() if snapshot.estimated_completion else None
        ))

        conn.commit()
        if should_close:
            conn.close()

    def _backup_to_json(self):
        """Backup all data to JSON file"""
        backup_data = {
            "events": {eid: self._event_to_dict(e) for eid, e in self.events.items()},
            "parameter_history": {path: self._param_history_to_dict(p)
                                 for path, p in self.parameter_history.items()},
            "gaps": {gid: self._gap_to_dict(g) for gid, g in self.gap_records.items()},
            "llm_proposals": {pid: self._proposal_to_dict(p)
                            for pid, p in self.llm_proposals.items()},
            "snapshots": [self._snapshot_to_dict(s) for s in self.snapshots]
        }

        with open(self.json_backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)

    def _event_to_dict(self, event: ExpansionEvent) -> dict:
        """Convert ExpansionEvent to JSON-serializable dict"""
        d = asdict(event)
        d['timestamp'] = event.timestamp.isoformat()
        d['trigger'] = event.trigger.value
        d['status'] = event.status.value
        d['impact_assessment'] = event.impact_assessment.value if event.impact_assessment else None
        d['phase'] = event.phase.value
        return d

    def _param_history_to_dict(self, param: ParameterHistoryEntry) -> dict:
        """Convert ParameterHistoryEntry to JSON-serializable dict"""
        d = asdict(param)
        d['created_timestamp'] = param.created_timestamp.isoformat()
        d['creation_trigger'] = param.creation_trigger.value
        d['genres_utilized'] = list(param.genres_utilized)
        return d

    def _gap_to_dict(self, gap: GapDetectionRecord) -> dict:
        """Convert GapDetectionRecord to JSON-serializable dict"""
        d = asdict(gap)
        d['detected_timestamp'] = gap.detected_timestamp.isoformat()
        d['resolution_timestamp'] = (gap.resolution_timestamp.isoformat()
                                    if gap.resolution_timestamp else None)
        return d

    def _proposal_to_dict(self, proposal: LLMProposalRecord) -> dict:
        """Convert LLMProposalRecord to JSON-serializable dict"""
        d = asdict(proposal)
        d['timestamp'] = proposal.timestamp.isoformat()
        d['review_timestamp'] = (proposal.review_timestamp.isoformat()
                                if proposal.review_timestamp else None)
        d['outcome_assessment'] = (proposal.outcome_assessment.value
                                  if proposal.outcome_assessment else None)
        return d

    def _snapshot_to_dict(self, snapshot: SystemSnapshot) -> dict:
        """Convert SystemSnapshot to JSON-serializable dict"""
        d = asdict(snapshot)
        d['timestamp'] = snapshot.timestamp.isoformat()
        d['phase'] = snapshot.phase.value
        d['estimated_completion'] = (snapshot.estimated_completion.isoformat()
                                     if snapshot.estimated_completion else None)
        return d


# =============================================================================
# Example Usage and Testing
# =============================================================================

def example_usage():
    """
    Example demonstrating how to use the ExpansionHistoryTracker
    """
    # Initialize tracker
    tracker = ExpansionHistoryTracker()

    # Example 1: Log a reconstruction failure-triggered expansion
    event = ExpansionEvent(
        parameters_added=[
            "harmony.jazz.voicing_cluster_density",
            "harmony.jazz.voicing_spread_factor"
        ],
        parameter_count=2,
        trigger=ExpansionTrigger.RECONSTRUCTION_FAILURE,
        trigger_details={
            "midi_file": "examples/bill_evans_waltz.mid",
            "failed_feature": "close_position_clusters",
            "reconstruction_error": 0.45
        },
        reconstruction_accuracy_before=0.72,
        reconstruction_accuracy_after=0.89,
        code_files_modified=["generators/advanced_harmony_generator.py"],
        code_lines_added=85,
        generator_enhanced=True,
        xgboost_models_trained=2,
        model_training_time_seconds=45.3,
        musical_validation_passed=True,
        phase=ExpansionPhase.PHASE_1,
        agent_responsible="Agent 5 - Reconstruction Engine",
        notes="Added parameters to handle Bill Evans-style close position voicings"
    )

    event_id = tracker.log_expansion_event(event)
    tracker.update_event_status(event_id, ExpansionStatus.DEPLOYED)
    tracker.update_event_impact(event_id, ParameterImpact.HIGH_VALUE, 0.17)

    # Example 2: Log a gap detection
    gap = GapDetectionRecord(
        gap_type="missing_feature",
        description="Cannot reconstruct polyrhythmic patterns with odd subdivisions",
        severity="high",
        midi_file_triggering="examples/take_five.mid",
        genre_context="jazz",
        musical_characteristics={
            "time_signature": "5/4",
            "subdivision": "quintuplets",
            "pattern_type": "polyrhythmic"
        }
    )

    gap_id = tracker.log_gap_detection(gap)

    # Example 3: Log an LLM proposal
    proposal = LLMProposalRecord(
        llm_model="claude-sonnet-4-5",
        prompt_hash=hashlib.sha256(b"analyze gap and propose parameters").hexdigest(),
        proposed_parameters=[
            {
                "name": "rhythm.polyrhythm.odd_subdivision_support",
                "type": "boolean",
                "default": True
            },
            {
                "name": "rhythm.polyrhythm.subdivision_ratio",
                "type": "array_int",
                "default": [3, 5, 7]
            }
        ],
        rationale="To support odd-time polyrhythmic patterns like those in Take Five",
        triggered_by_gap=gap_id
    )

    proposal_id = tracker.log_llm_proposal(proposal)
    tracker.review_llm_proposal(proposal_id, "accepted",
                               "Excellent proposal, addresses gap directly",
                               ["rhythm.polyrhythm.odd_subdivision_support"])

    # Example 4: Capture system snapshot
    snapshot = SystemSnapshot(
        total_parameters=167,
        parameters_by_category={"harmony": 45, "melody": 32, "rhythm": 28, "bass": 15},
        parameters_by_type={"continuous": 78, "categorical": 42, "boolean": 35},
        total_code_lines=108500,
        generator_lines=86200,
        feature_extractor_features=138,
        average_reconstruction_accuracy=0.85,
        xgboost_models_count=167,
        average_model_accuracy=0.82,
        training_data_size=1500,
        active_parameters=165,
        deprecated_parameters=2,
        redundant_parameters=0,
        phase=ExpansionPhase.PHASE_1,
        progress_to_phase_goal=32.4
    )

    tracker.capture_system_snapshot(snapshot)

    # Example 5: Generate analytics
    analytics = tracker.generate_analytics()
    print("\n" + "="*80)
    print("EXPANSION ANALYTICS")
    print("="*80)
    print(f"Total Expansions: {analytics.total_expansions}")
    print(f"Success Rate: {analytics.success_rate*100:.1f}%")
    print(f"Average Accuracy Improvement: {analytics.average_accuracy_improvement*100:.2f}%")
    print(f"Growth Rate: {analytics.growth_rate_per_day:.2f} parameters/day")

    # Example 6: Generate effectiveness report
    report = tracker.generate_expansion_effectiveness_report()
    print("\n" + "="*80)
    print("EXPANSION EFFECTIVENESS REPORT")
    print("="*80)
    print(json.dumps(report, indent=2, default=str))

    return tracker


if __name__ == "__main__":
    # Run example usage
    tracker = example_usage()
    print("\n✅ Expansion History Tracker demonstration complete!")
    print(f"📊 Database: {tracker.db_path}")
    print(f"💾 JSON Backup: {tracker.json_backup_path}")
