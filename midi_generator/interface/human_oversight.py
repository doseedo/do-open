#!/usr/bin/env python3
"""
Human-in-Loop Interface - Agent 29
===================================

Web dashboard for monitoring self-expanding inverse music generation system.
Provides real-time monitoring, approval workflow, and quality visualization.

Features:
- Real-time dashboard for expansion monitoring
- Approval workflow for LLM-proposed parameters
- Quality metrics visualization
- Parameter registry integration
- Synthetic training data management
- Model performance tracking
- Gap detection visualization
- Batch approval/rejection
- Export/import capabilities
- WebSocket for real-time updates

Architecture:
- Flask web application with SQLAlchemy ORM
- SQLite database for persistence
- WebSocket via Flask-SocketIO for real-time updates
- REST API for programmatic access
- Bootstrap 5 frontend
- Chart.js for visualizations
- Integration with universal_registry.py

Author: Agent 29 - Human-in-Loop Interface
License: MIT
"""

import os
import sys
import json
import logging
import sqlite3
import hashlib
import datetime
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import time

# Flask and extensions
try:
    from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
    from flask_socketio import SocketIO, emit, join_room, leave_room
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    print("WARNING: Flask not installed. Run: pip install flask flask-socketio flask-cors")
    FLASK_AVAILABLE = False

# Database
try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, scoped_session
    from sqlalchemy.pool import StaticPool
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    print("WARNING: SQLAlchemy not installed. Run: pip install sqlalchemy")
    SQLALCHEMY_AVAILABLE = False
    Base = None

# Scientific computing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("WARNING: NumPy not installed. Run: pip install numpy")
    NUMPY_AVAILABLE = False

# Import from parameter registry
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from parameters.universal_registry import (
        REGISTRY, ParameterDefinition, ParameterType,
        ParameterCategory, MusicalImpact
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    print("WARNING: Universal registry not available")
    REGISTRY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ExpansionStatus(Enum):
    """Status of parameter expansion proposals"""
    PENDING = "pending"              # Awaiting human review
    APPROVED = "approved"            # Approved by human
    REJECTED = "rejected"            # Rejected by human
    IMPLEMENTED = "implemented"      # Code generated and integrated
    TRAINING = "training"            # Synthetic training in progress
    MODEL_READY = "model_ready"      # XGBoost model trained
    ACTIVE = "active"                # Parameter live in system
    FAILED = "failed"                # Implementation or training failed
    ROLLBACK = "rollback"            # Rolled back due to issues


class ExpansionSource(Enum):
    """Source of expansion proposal"""
    GAP_DETECTION = "gap_detection"          # From reconstruction failure
    LLM_PROPOSAL = "llm_proposal"            # Direct LLM suggestion
    HUMAN_REQUEST = "human_request"          # User-requested parameter
    GENRE_ANALYSIS = "genre_analysis"        # From genre corpus analysis
    ERROR_ANALYSIS = "error_analysis"        # From error pattern analysis
    PEER_SYSTEM = "peer_system"              # From other music systems


class QualityMetric(Enum):
    """Types of quality metrics tracked"""
    RECONSTRUCTION_ERROR = "reconstruction_error"
    MUSICAL_COHERENCE = "musical_coherence"
    GENRE_ACCURACY = "genre_accuracy"
    FEATURE_COVERAGE = "feature_coverage"
    MODEL_CONFIDENCE = "model_confidence"
    PARAMETER_IMPORTANCE = "parameter_importance"
    SYNTHETIC_QUALITY = "synthetic_quality"
    VALIDATION_SCORE = "validation_score"


# Quality thresholds
QUALITY_THRESHOLDS = {
    QualityMetric.RECONSTRUCTION_ERROR: 0.15,      # Max error threshold
    QualityMetric.MUSICAL_COHERENCE: 0.70,         # Min coherence threshold
    QualityMetric.GENRE_ACCURACY: 0.80,            # Min accuracy threshold
    QualityMetric.FEATURE_COVERAGE: 0.60,          # Min coverage threshold
    QualityMetric.MODEL_CONFIDENCE: 0.75,          # Min confidence threshold
    QualityMetric.SYNTHETIC_QUALITY: 0.70,         # Min synthetic quality
    QualityMetric.VALIDATION_SCORE: 0.80,          # Min validation score
}


# ============================================================================
# DATABASE MODELS
# ============================================================================

if SQLALCHEMY_AVAILABLE:

    class ExpansionProposal(Base):
        """Database model for parameter expansion proposals"""
        __tablename__ = 'expansion_proposals'

        id = Column(Integer, primary_key=True)
        proposal_id = Column(String(64), unique=True, nullable=False)

        # Parameter details
        parameter_name = Column(String(256), nullable=False)
        parameter_path = Column(String(512), nullable=False)
        parameter_type = Column(String(64), nullable=False)
        description = Column(Text, nullable=False)

        # Proposal metadata
        source = Column(String(64), nullable=False)
        status = Column(String(64), default=ExpansionStatus.PENDING.value)
        priority = Column(Integer, default=50)  # 0-100

        # Context
        gap_context = Column(JSON, nullable=True)  # MIDI that failed reconstruction
        llm_reasoning = Column(Text, nullable=True)
        expected_impact = Column(Text, nullable=True)

        # Timestamps
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        reviewed_at = Column(DateTime, nullable=True)
        implemented_at = Column(DateTime, nullable=True)
        activated_at = Column(DateTime, nullable=True)

        # Review
        reviewer_notes = Column(Text, nullable=True)
        rejection_reason = Column(Text, nullable=True)

        # Implementation
        code_generated = Column(Boolean, default=False)
        code_path = Column(String(512), nullable=True)
        model_trained = Column(Boolean, default=False)
        model_path = Column(String(512), nullable=True)

        # Relationships
        metrics = relationship("MetricRecord", back_populates="proposal", cascade="all, delete-orphan")
        training_data = relationship("TrainingDataRecord", back_populates="proposal", cascade="all, delete-orphan")

        def to_dict(self):
            """Convert to dictionary"""
            return {
                'id': self.id,
                'proposal_id': self.proposal_id,
                'parameter_name': self.parameter_name,
                'parameter_path': self.parameter_path,
                'parameter_type': self.parameter_type,
                'description': self.description,
                'source': self.source,
                'status': self.status,
                'priority': self.priority,
                'gap_context': self.gap_context,
                'llm_reasoning': self.llm_reasoning,
                'expected_impact': self.expected_impact,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
                'implemented_at': self.implemented_at.isoformat() if self.implemented_at else None,
                'activated_at': self.activated_at.isoformat() if self.activated_at else None,
                'reviewer_notes': self.reviewer_notes,
                'rejection_reason': self.rejection_reason,
                'code_generated': self.code_generated,
                'code_path': self.code_path,
                'model_trained': self.model_trained,
                'model_path': self.model_path,
            }


    class MetricRecord(Base):
        """Database model for quality metrics"""
        __tablename__ = 'metric_records'

        id = Column(Integer, primary_key=True)
        proposal_id = Column(Integer, ForeignKey('expansion_proposals.id'), nullable=False)

        metric_type = Column(String(64), nullable=False)
        metric_value = Column(Float, nullable=False)
        threshold = Column(Float, nullable=True)
        passed = Column(Boolean, nullable=False)

        recorded_at = Column(DateTime, default=datetime.datetime.utcnow)
        context = Column(JSON, nullable=True)

        # Relationship
        proposal = relationship("ExpansionProposal", back_populates="metrics")

        def to_dict(self):
            return {
                'id': self.id,
                'proposal_id': self.proposal_id,
                'metric_type': self.metric_type,
                'metric_value': self.metric_value,
                'threshold': self.threshold,
                'passed': self.passed,
                'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None,
                'context': self.context
            }


    class TrainingDataRecord(Base):
        """Database model for synthetic training data"""
        __tablename__ = 'training_data_records'

        id = Column(Integer, primary_key=True)
        proposal_id = Column(Integer, ForeignKey('expansion_proposals.id'), nullable=False)

        midi_file_path = Column(String(512), nullable=False)
        parameter_value = Column(JSON, nullable=False)

        # Quality
        synthetic_score = Column(Float, nullable=True)
        validation_passed = Column(Boolean, default=False)

        # Metadata
        generated_at = Column(DateTime, default=datetime.datetime.utcnow)
        generation_time = Column(Float, nullable=True)  # Seconds

        # Relationship
        proposal = relationship("ExpansionProposal", back_populates="training_data")

        def to_dict(self):
            return {
                'id': self.id,
                'proposal_id': self.proposal_id,
                'midi_file_path': self.midi_file_path,
                'parameter_value': self.parameter_value,
                'synthetic_score': self.synthetic_score,
                'validation_passed': self.validation_passed,
                'generated_at': self.generated_at.isoformat() if self.generated_at else None,
                'generation_time': self.generation_time
            }


    class SystemMetric(Base):
        """Database model for system-wide metrics"""
        __tablename__ = 'system_metrics'

        id = Column(Integer, primary_key=True)

        # Metric
        metric_name = Column(String(128), nullable=False)
        metric_value = Column(Float, nullable=False)

        # Context
        category = Column(String(64), nullable=True)
        tags = Column(JSON, nullable=True)

        # Timestamp
        recorded_at = Column(DateTime, default=datetime.datetime.utcnow)

        def to_dict(self):
            return {
                'id': self.id,
                'metric_name': self.metric_name,
                'metric_value': self.metric_value,
                'category': self.category,
                'tags': self.tags,
                'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None
            }


    class AuditLog(Base):
        """Database model for audit logging"""
        __tablename__ = 'audit_logs'

        id = Column(Integer, primary_key=True)

        # Action
        action_type = Column(String(64), nullable=False)
        action_description = Column(Text, nullable=False)

        # User
        user_id = Column(String(128), nullable=True)
        user_name = Column(String(256), nullable=True)

        # Context
        proposal_id = Column(String(64), nullable=True)
        details = Column(JSON, nullable=True)

        # Timestamp
        timestamp = Column(DateTime, default=datetime.datetime.utcnow)

        def to_dict(self):
            return {
                'id': self.id,
                'action_type': self.action_type,
                'action_description': self.action_description,
                'user_id': self.user_id,
                'user_name': self.user_name,
                'proposal_id': self.proposal_id,
                'details': self.details,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None
            }


# ============================================================================
# CORE HUMAN OVERSIGHT ENGINE
# ============================================================================

class HumanOversightEngine:
    """
    Core engine for human-in-loop oversight of system expansion.

    Manages:
    - Expansion proposal tracking
    - Quality metric monitoring
    - Approval workflow
    - Real-time notifications
    - Audit logging
    """

    def __init__(self, db_path: str = "human_oversight.db"):
        """Initialize oversight engine"""
        self.db_path = db_path
        self.engine = None
        self.Session = None

        # Real-time notification queue
        self.notification_queue = queue.Queue()

        # Statistics cache
        self.stats_cache = {}
        self.stats_cache_time = 0
        self.stats_cache_ttl = 5  # seconds

        # Initialize database
        self._init_database()

        logger.info(f"HumanOversightEngine initialized with database: {db_path}")

    def _init_database(self):
        """Initialize database connection and create tables"""
        if not SQLALCHEMY_AVAILABLE:
            logger.error("SQLAlchemy not available, cannot initialize database")
            return

        # Create engine
        self.engine = create_engine(
            f'sqlite:///{self.db_path}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool,
            echo=False
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))

        logger.info("Database initialized successfully")

    def create_proposal(
        self,
        parameter_name: str,
        parameter_path: str,
        parameter_type: str,
        description: str,
        source: ExpansionSource,
        gap_context: Optional[Dict] = None,
        llm_reasoning: Optional[str] = None,
        expected_impact: Optional[str] = None,
        priority: int = 50
    ) -> str:
        """
        Create new expansion proposal.

        Args:
            parameter_name: Name of parameter
            parameter_path: Full path (e.g., "harmony.jazz.voicing_complexity")
            parameter_type: Type of parameter
            description: Description of parameter
            source: Source of proposal
            gap_context: Context from gap detection (MIDI that failed)
            llm_reasoning: LLM's reasoning for proposal
            expected_impact: Expected impact description
            priority: Priority 0-100 (higher = more important)

        Returns:
            proposal_id: Unique ID of created proposal
        """
        session = self.Session()
        try:
            # Generate unique ID
            proposal_id = self._generate_proposal_id(parameter_path)

            # Create proposal
            proposal = ExpansionProposal(
                proposal_id=proposal_id,
                parameter_name=parameter_name,
                parameter_path=parameter_path,
                parameter_type=parameter_type,
                description=description,
                source=source.value,
                gap_context=gap_context,
                llm_reasoning=llm_reasoning,
                expected_impact=expected_impact,
                priority=priority,
                status=ExpansionStatus.PENDING.value
            )

            session.add(proposal)
            session.commit()

            # Log audit
            self._log_audit(
                session=session,
                action_type="CREATE_PROPOSAL",
                action_description=f"Created expansion proposal: {parameter_name}",
                proposal_id=proposal_id,
                details={'source': source.value, 'priority': priority}
            )

            # Notify
            self._notify("new_proposal", proposal.to_dict())

            logger.info(f"Created proposal {proposal_id}: {parameter_name}")
            return proposal_id

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create proposal: {e}")
            raise
        finally:
            session.close()

    def approve_proposal(
        self,
        proposal_id: str,
        reviewer_notes: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Approve expansion proposal.

        Args:
            proposal_id: Proposal ID
            reviewer_notes: Optional notes from reviewer
            user_id: ID of user approving
            user_name: Name of user approving

        Returns:
            success: True if approved successfully
        """
        session = self.Session()
        try:
            proposal = session.query(ExpansionProposal).filter_by(
                proposal_id=proposal_id
            ).first()

            if not proposal:
                logger.error(f"Proposal {proposal_id} not found")
                return False

            if proposal.status != ExpansionStatus.PENDING.value:
                logger.error(f"Proposal {proposal_id} is not pending (status: {proposal.status})")
                return False

            # Update proposal
            proposal.status = ExpansionStatus.APPROVED.value
            proposal.reviewed_at = datetime.datetime.utcnow()
            proposal.reviewer_notes = reviewer_notes

            session.commit()

            # Log audit
            self._log_audit(
                session=session,
                action_type="APPROVE_PROPOSAL",
                action_description=f"Approved proposal: {proposal.parameter_name}",
                proposal_id=proposal_id,
                user_id=user_id,
                user_name=user_name,
                details={'notes': reviewer_notes}
            )

            # Notify
            self._notify("proposal_approved", proposal.to_dict())

            logger.info(f"Approved proposal {proposal_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to approve proposal: {e}")
            return False
        finally:
            session.close()

    def reject_proposal(
        self,
        proposal_id: str,
        rejection_reason: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Reject expansion proposal.

        Args:
            proposal_id: Proposal ID
            rejection_reason: Reason for rejection
            user_id: ID of user rejecting
            user_name: Name of user rejecting

        Returns:
            success: True if rejected successfully
        """
        session = self.Session()
        try:
            proposal = session.query(ExpansionProposal).filter_by(
                proposal_id=proposal_id
            ).first()

            if not proposal:
                logger.error(f"Proposal {proposal_id} not found")
                return False

            if proposal.status != ExpansionStatus.PENDING.value:
                logger.error(f"Proposal {proposal_id} is not pending (status: {proposal.status})")
                return False

            # Update proposal
            proposal.status = ExpansionStatus.REJECTED.value
            proposal.reviewed_at = datetime.datetime.utcnow()
            proposal.rejection_reason = rejection_reason

            session.commit()

            # Log audit
            self._log_audit(
                session=session,
                action_type="REJECT_PROPOSAL",
                action_description=f"Rejected proposal: {proposal.parameter_name}",
                proposal_id=proposal_id,
                user_id=user_id,
                user_name=user_name,
                details={'reason': rejection_reason}
            )

            # Notify
            self._notify("proposal_rejected", proposal.to_dict())

            logger.info(f"Rejected proposal {proposal_id}: {rejection_reason}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to reject proposal: {e}")
            return False
        finally:
            session.close()

    def batch_approve(
        self,
        proposal_ids: List[str],
        reviewer_notes: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Batch approve multiple proposals.

        Returns:
            (success_count, failure_count)
        """
        success_count = 0
        failure_count = 0

        for proposal_id in proposal_ids:
            if self.approve_proposal(proposal_id, reviewer_notes, user_id, user_name):
                success_count += 1
            else:
                failure_count += 1

        logger.info(f"Batch approved {success_count}/{len(proposal_ids)} proposals")
        return success_count, failure_count

    def batch_reject(
        self,
        proposal_ids: List[str],
        rejection_reason: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Batch reject multiple proposals.

        Returns:
            (success_count, failure_count)
        """
        success_count = 0
        failure_count = 0

        for proposal_id in proposal_ids:
            if self.reject_proposal(proposal_id, rejection_reason, user_id, user_name):
                success_count += 1
            else:
                failure_count += 1

        logger.info(f"Batch rejected {success_count}/{len(proposal_ids)} proposals")
        return success_count, failure_count

    def update_proposal_status(
        self,
        proposal_id: str,
        new_status: ExpansionStatus,
        details: Optional[Dict] = None
    ) -> bool:
        """Update proposal status (for system use)"""
        session = self.Session()
        try:
            proposal = session.query(ExpansionProposal).filter_by(
                proposal_id=proposal_id
            ).first()

            if not proposal:
                return False

            old_status = proposal.status
            proposal.status = new_status.value

            # Update timestamps
            if new_status == ExpansionStatus.IMPLEMENTED:
                proposal.implemented_at = datetime.datetime.utcnow()
            elif new_status == ExpansionStatus.ACTIVE:
                proposal.activated_at = datetime.datetime.utcnow()

            # Update flags
            if details:
                if 'code_generated' in details:
                    proposal.code_generated = details['code_generated']
                if 'code_path' in details:
                    proposal.code_path = details['code_path']
                if 'model_trained' in details:
                    proposal.model_trained = details['model_trained']
                if 'model_path' in details:
                    proposal.model_path = details['model_path']

            session.commit()

            # Log audit
            self._log_audit(
                session=session,
                action_type="UPDATE_STATUS",
                action_description=f"Status changed: {old_status} -> {new_status.value}",
                proposal_id=proposal_id,
                details=details
            )

            # Notify
            self._notify("status_updated", proposal.to_dict())

            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update status: {e}")
            return False
        finally:
            session.close()

    def record_metric(
        self,
        proposal_id: str,
        metric_type: QualityMetric,
        metric_value: float,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Record quality metric for proposal.

        Args:
            proposal_id: Proposal ID
            metric_type: Type of metric
            metric_value: Metric value
            context: Additional context

        Returns:
            success: True if recorded successfully
        """
        session = self.Session()
        try:
            proposal = session.query(ExpansionProposal).filter_by(
                proposal_id=proposal_id
            ).first()

            if not proposal:
                logger.error(f"Proposal {proposal_id} not found")
                return False

            # Get threshold
            threshold = QUALITY_THRESHOLDS.get(metric_type)

            # Determine if passed
            if threshold is not None:
                if metric_type == QualityMetric.RECONSTRUCTION_ERROR:
                    # Lower is better
                    passed = metric_value <= threshold
                else:
                    # Higher is better
                    passed = metric_value >= threshold
            else:
                passed = True

            # Create metric record
            metric = MetricRecord(
                proposal_id=proposal.id,
                metric_type=metric_type.value,
                metric_value=metric_value,
                threshold=threshold,
                passed=passed,
                context=context
            )

            session.add(metric)
            session.commit()

            # Notify
            self._notify("metric_recorded", {
                'proposal_id': proposal_id,
                'metric_type': metric_type.value,
                'metric_value': metric_value,
                'passed': passed
            })

            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record metric: {e}")
            return False
        finally:
            session.close()

    def record_training_data(
        self,
        proposal_id: str,
        midi_file_path: str,
        parameter_value: Any,
        synthetic_score: Optional[float] = None,
        validation_passed: bool = False,
        generation_time: Optional[float] = None
    ) -> bool:
        """Record synthetic training data for proposal"""
        session = self.Session()
        try:
            proposal = session.query(ExpansionProposal).filter_by(
                proposal_id=proposal_id
            ).first()

            if not proposal:
                return False

            # Create training data record
            training_data = TrainingDataRecord(
                proposal_id=proposal.id,
                midi_file_path=midi_file_path,
                parameter_value=parameter_value,
                synthetic_score=synthetic_score,
                validation_passed=validation_passed,
                generation_time=generation_time
            )

            session.add(training_data)
            session.commit()

            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record training data: {e}")
            return False
        finally:
            session.close()

    def get_proposal(self, proposal_id: str) -> Optional[Dict]:
        """Get proposal by ID"""
        session = self.Session()
        try:
            proposal = session.query(ExpansionProposal).filter_by(
                proposal_id=proposal_id
            ).first()

            if not proposal:
                return None

            return proposal.to_dict()
        finally:
            session.close()

    def get_proposals(
        self,
        status: Optional[ExpansionStatus] = None,
        source: Optional[ExpansionSource] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get proposals with filtering"""
        session = self.Session()
        try:
            query = session.query(ExpansionProposal)

            if status:
                query = query.filter_by(status=status.value)
            if source:
                query = query.filter_by(source=source.value)

            query = query.order_by(ExpansionProposal.created_at.desc())
            query = query.limit(limit).offset(offset)

            proposals = query.all()
            return [p.to_dict() for p in proposals]
        finally:
            session.close()

    def get_proposal_metrics(self, proposal_id: str) -> List[Dict]:
        """Get all metrics for a proposal"""
        session = self.Session()
        try:
            proposal = session.query(ExpansionProposal).filter_by(
                proposal_id=proposal_id
            ).first()

            if not proposal:
                return []

            metrics = session.query(MetricRecord).filter_by(
                proposal_id=proposal.id
            ).order_by(MetricRecord.recorded_at.desc()).all()

            return [m.to_dict() for m in metrics]
        finally:
            session.close()

    def get_proposal_training_data(self, proposal_id: str) -> List[Dict]:
        """Get all training data for a proposal"""
        session = self.Session()
        try:
            proposal = session.query(ExpansionProposal).filter_by(
                proposal_id=proposal_id
            ).first()

            if not proposal:
                return []

            training_data = session.query(TrainingDataRecord).filter_by(
                proposal_id=proposal.id
            ).order_by(TrainingDataRecord.generated_at.desc()).all()

            return [td.to_dict() for td in training_data]
        finally:
            session.close()

    def get_statistics(self) -> Dict:
        """
        Get system statistics.

        Returns comprehensive statistics about:
        - Proposal counts by status
        - Average approval time
        - Quality metric summaries
        - Training data statistics
        - System health indicators
        """
        # Check cache
        now = time.time()
        if now - self.stats_cache_time < self.stats_cache_ttl:
            return self.stats_cache

        session = self.Session()
        try:
            stats = {}

            # Proposal counts by status
            stats['proposals_by_status'] = {}
            for status in ExpansionStatus:
                count = session.query(ExpansionProposal).filter_by(
                    status=status.value
                ).count()
                stats['proposals_by_status'][status.value] = count

            # Proposal counts by source
            stats['proposals_by_source'] = {}
            for source in ExpansionSource:
                count = session.query(ExpansionProposal).filter_by(
                    source=source.value
                ).count()
                stats['proposals_by_source'][source.value] = count

            # Total proposals
            stats['total_proposals'] = session.query(ExpansionProposal).count()

            # Parameter count
            if REGISTRY_AVAILABLE:
                stats['current_parameters'] = len(REGISTRY.get_all_parameters())
            else:
                stats['current_parameters'] = 165  # Default

            # Average review time (for approved/rejected)
            reviewed_proposals = session.query(ExpansionProposal).filter(
                ExpansionProposal.reviewed_at.isnot(None)
            ).all()

            if reviewed_proposals:
                review_times = [
                    (p.reviewed_at - p.created_at).total_seconds()
                    for p in reviewed_proposals
                ]
                stats['avg_review_time_seconds'] = sum(review_times) / len(review_times)
            else:
                stats['avg_review_time_seconds'] = 0

            # Quality metrics summary
            stats['quality_metrics'] = {}
            for metric_type in QualityMetric:
                metrics = session.query(MetricRecord).filter_by(
                    metric_type=metric_type.value
                ).all()

                if metrics:
                    values = [m.metric_value for m in metrics]
                    stats['quality_metrics'][metric_type.value] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values),
                        'passed': sum(1 for m in metrics if m.passed),
                        'failed': sum(1 for m in metrics if not m.passed)
                    }

            # Training data statistics
            total_training_data = session.query(TrainingDataRecord).count()
            validated_training_data = session.query(TrainingDataRecord).filter_by(
                validation_passed=True
            ).count()

            stats['training_data'] = {
                'total': total_training_data,
                'validated': validated_training_data,
                'validation_rate': validated_training_data / total_training_data if total_training_data > 0 else 0
            }

            # System health
            pending_proposals = session.query(ExpansionProposal).filter_by(
                status=ExpansionStatus.PENDING.value
            ).count()

            failed_proposals = session.query(ExpansionProposal).filter_by(
                status=ExpansionStatus.FAILED.value
            ).count()

            stats['system_health'] = {
                'pending_queue_size': pending_proposals,
                'failed_count': failed_proposals,
                'health_score': self._calculate_health_score(stats)
            }

            # Recent activity
            recent_proposals = session.query(ExpansionProposal).order_by(
                ExpansionProposal.created_at.desc()
            ).limit(10).all()

            stats['recent_activity'] = [p.to_dict() for p in recent_proposals]

            # Cache stats
            self.stats_cache = stats
            self.stats_cache_time = now

            return stats

        finally:
            session.close()

    def get_audit_logs(
        self,
        action_type: Optional[str] = None,
        proposal_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get audit logs with filtering"""
        session = self.Session()
        try:
            query = session.query(AuditLog)

            if action_type:
                query = query.filter_by(action_type=action_type)
            if proposal_id:
                query = query.filter_by(proposal_id=proposal_id)

            query = query.order_by(AuditLog.timestamp.desc()).limit(limit)

            logs = query.all()
            return [log.to_dict() for log in logs]
        finally:
            session.close()

    def record_system_metric(
        self,
        metric_name: str,
        metric_value: float,
        category: Optional[str] = None,
        tags: Optional[Dict] = None
    ):
        """Record system-wide metric"""
        session = self.Session()
        try:
            metric = SystemMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                category=category,
                tags=tags
            )
            session.add(metric)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record system metric: {e}")
        finally:
            session.close()

    def export_data(self, output_path: str, format: str = "json") -> bool:
        """
        Export all data to file.

        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')

        Returns:
            success: True if exported successfully
        """
        try:
            session = self.Session()

            # Get all data
            proposals = session.query(ExpansionProposal).all()
            metrics = session.query(MetricRecord).all()
            training_data = session.query(TrainingDataRecord).all()
            audit_logs = session.query(AuditLog).all()

            data = {
                'proposals': [p.to_dict() for p in proposals],
                'metrics': [m.to_dict() for m in metrics],
                'training_data': [td.to_dict() for td in training_data],
                'audit_logs': [log.to_dict() for log in audit_logs],
                'export_time': datetime.datetime.utcnow().isoformat(),
                'version': '1.0'
            }

            session.close()

            # Write to file
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Exported data to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return False

    def import_data(self, input_path: str) -> bool:
        """Import data from file"""
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)

            # Note: This is a simplified import
            # In production, would need proper validation and conflict resolution

            logger.info(f"Imported data from {input_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return False

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _generate_proposal_id(self, parameter_path: str) -> str:
        """Generate unique proposal ID"""
        timestamp = datetime.datetime.utcnow().isoformat()
        data = f"{parameter_path}_{timestamp}"
        hash_obj = hashlib.sha256(data.encode())
        return hash_obj.hexdigest()[:16]

    def _log_audit(
        self,
        session,
        action_type: str,
        action_description: str,
        proposal_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """Log audit event"""
        try:
            log = AuditLog(
                action_type=action_type,
                action_description=action_description,
                proposal_id=proposal_id,
                user_id=user_id,
                user_name=user_name,
                details=details
            )
            session.add(log)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log audit: {e}")

    def _notify(self, event_type: str, data: Dict):
        """Add notification to queue"""
        self.notification_queue.put({
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.datetime.utcnow().isoformat()
        })

    def _calculate_health_score(self, stats: Dict) -> float:
        """Calculate system health score 0-100"""
        score = 100.0

        # Penalize high pending queue
        pending = stats.get('system_health', {}).get('pending_queue_size', 0)
        if pending > 50:
            score -= min(30, (pending - 50) * 0.5)

        # Penalize failures
        failed = stats.get('system_health', {}).get('failed_count', 0)
        if failed > 10:
            score -= min(20, (failed - 10) * 0.5)

        # Reward high quality metrics
        quality_metrics = stats.get('quality_metrics', {})
        if quality_metrics:
            avg_pass_rate = sum(
                m.get('passed', 0) / max(m.get('count', 1), 1)
                for m in quality_metrics.values()
            ) / len(quality_metrics)
            score += avg_pass_rate * 10

        return max(0.0, min(100.0, score))


# ============================================================================
# FLASK WEB APPLICATION
# ============================================================================

if FLASK_AVAILABLE:

    # Initialize Flask app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['JSON_SORT_KEYS'] = False

    # Initialize SocketIO for real-time updates
    socketio = SocketIO(app, cors_allowed_origins="*")

    # Enable CORS
    CORS(app)

    # Initialize oversight engine
    oversight_engine = HumanOversightEngine()

    # ========================================================================
    # WEB ROUTES
    # ========================================================================

    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('dashboard.html')

    @app.route('/dashboard')
    def dashboard():
        """Dashboard page"""
        return render_template('dashboard.html')

    @app.route('/proposals')
    def proposals_page():
        """Proposals management page"""
        return render_template('proposals.html')

    @app.route('/metrics')
    def metrics_page():
        """Metrics visualization page"""
        return render_template('metrics.html')

    @app.route('/audit')
    def audit_page():
        """Audit log page"""
        return render_template('audit.html')

    # ========================================================================
    # API ENDPOINTS
    # ========================================================================

    @app.route('/api/health', methods=['GET'])
    def api_health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'version': '1.0'
        })

    @app.route('/api/statistics', methods=['GET'])
    def api_statistics():
        """Get system statistics"""
        try:
            stats = oversight_engine.get_statistics()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals', methods=['GET'])
    def api_get_proposals():
        """Get proposals with filtering"""
        try:
            status = request.args.get('status')
            source = request.args.get('source')
            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))

            status_enum = ExpansionStatus(status) if status else None
            source_enum = ExpansionSource(source) if source else None

            proposals = oversight_engine.get_proposals(
                status=status_enum,
                source=source_enum,
                limit=limit,
                offset=offset
            )

            return jsonify(proposals)
        except Exception as e:
            logger.error(f"Error getting proposals: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals/<proposal_id>', methods=['GET'])
    def api_get_proposal(proposal_id):
        """Get single proposal"""
        try:
            proposal = oversight_engine.get_proposal(proposal_id)
            if not proposal:
                return jsonify({'error': 'Proposal not found'}), 404
            return jsonify(proposal)
        except Exception as e:
            logger.error(f"Error getting proposal: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals', methods=['POST'])
    def api_create_proposal():
        """Create new proposal"""
        try:
            data = request.json

            proposal_id = oversight_engine.create_proposal(
                parameter_name=data['parameter_name'],
                parameter_path=data['parameter_path'],
                parameter_type=data['parameter_type'],
                description=data['description'],
                source=ExpansionSource(data['source']),
                gap_context=data.get('gap_context'),
                llm_reasoning=data.get('llm_reasoning'),
                expected_impact=data.get('expected_impact'),
                priority=data.get('priority', 50)
            )

            return jsonify({'proposal_id': proposal_id}), 201
        except Exception as e:
            logger.error(f"Error creating proposal: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals/<proposal_id>/approve', methods=['POST'])
    def api_approve_proposal(proposal_id):
        """Approve proposal"""
        try:
            data = request.json or {}

            success = oversight_engine.approve_proposal(
                proposal_id=proposal_id,
                reviewer_notes=data.get('reviewer_notes'),
                user_id=data.get('user_id'),
                user_name=data.get('user_name')
            )

            if success:
                return jsonify({'status': 'approved'})
            else:
                return jsonify({'error': 'Failed to approve proposal'}), 400
        except Exception as e:
            logger.error(f"Error approving proposal: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals/<proposal_id>/reject', methods=['POST'])
    def api_reject_proposal(proposal_id):
        """Reject proposal"""
        try:
            data = request.json

            success = oversight_engine.reject_proposal(
                proposal_id=proposal_id,
                rejection_reason=data['rejection_reason'],
                user_id=data.get('user_id'),
                user_name=data.get('user_name')
            )

            if success:
                return jsonify({'status': 'rejected'})
            else:
                return jsonify({'error': 'Failed to reject proposal'}), 400
        except Exception as e:
            logger.error(f"Error rejecting proposal: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals/batch/approve', methods=['POST'])
    def api_batch_approve():
        """Batch approve proposals"""
        try:
            data = request.json
            proposal_ids = data['proposal_ids']
            reviewer_notes = data.get('reviewer_notes')
            user_id = data.get('user_id')
            user_name = data.get('user_name')

            success_count, failure_count = oversight_engine.batch_approve(
                proposal_ids=proposal_ids,
                reviewer_notes=reviewer_notes,
                user_id=user_id,
                user_name=user_name
            )

            return jsonify({
                'success_count': success_count,
                'failure_count': failure_count
            })
        except Exception as e:
            logger.error(f"Error batch approving: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals/batch/reject', methods=['POST'])
    def api_batch_reject():
        """Batch reject proposals"""
        try:
            data = request.json
            proposal_ids = data['proposal_ids']
            rejection_reason = data['rejection_reason']
            user_id = data.get('user_id')
            user_name = data.get('user_name')

            success_count, failure_count = oversight_engine.batch_reject(
                proposal_ids=proposal_ids,
                rejection_reason=rejection_reason,
                user_id=user_id,
                user_name=user_name
            )

            return jsonify({
                'success_count': success_count,
                'failure_count': failure_count
            })
        except Exception as e:
            logger.error(f"Error batch rejecting: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals/<proposal_id>/metrics', methods=['GET'])
    def api_get_proposal_metrics(proposal_id):
        """Get proposal metrics"""
        try:
            metrics = oversight_engine.get_proposal_metrics(proposal_id)
            return jsonify(metrics)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/proposals/<proposal_id>/metrics', methods=['POST'])
    def api_record_metric(proposal_id):
        """Record metric for proposal"""
        try:
            data = request.json

            success = oversight_engine.record_metric(
                proposal_id=proposal_id,
                metric_type=QualityMetric(data['metric_type']),
                metric_value=data['metric_value'],
                context=data.get('context')
            )

            if success:
                return jsonify({'status': 'recorded'})
            else:
                return jsonify({'error': 'Failed to record metric'}), 400
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/audit', methods=['GET'])
    def api_get_audit_logs():
        """Get audit logs"""
        try:
            action_type = request.args.get('action_type')
            proposal_id = request.args.get('proposal_id')
            limit = int(request.args.get('limit', 100))

            logs = oversight_engine.get_audit_logs(
                action_type=action_type,
                proposal_id=proposal_id,
                limit=limit
            )

            return jsonify(logs)
        except Exception as e:
            logger.error(f"Error getting audit logs: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/export', methods=['POST'])
    def api_export_data():
        """Export data"""
        try:
            data = request.json
            output_path = data['output_path']
            format = data.get('format', 'json')

            success = oversight_engine.export_data(output_path, format)

            if success:
                return jsonify({'status': 'exported', 'path': output_path})
            else:
                return jsonify({'error': 'Export failed'}), 500
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return jsonify({'error': str(e)}), 500

    # ========================================================================
    # WEBSOCKET EVENTS
    # ========================================================================

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'status': 'connected'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Subscribe to updates"""
        room = data.get('room', 'general')
        join_room(room)
        logger.info(f"Client {request.sid} subscribed to {room}")

    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Unsubscribe from updates"""
        room = data.get('room', 'general')
        leave_room(room)
        logger.info(f"Client {request.sid} unsubscribed from {room}")

    # Background task to broadcast notifications
    def notification_broadcaster():
        """Broadcast notifications from queue"""
        while True:
            try:
                notification = oversight_engine.notification_queue.get(timeout=1)
                socketio.emit('notification', notification, room='general')
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error broadcasting notification: {e}")

    # Start broadcaster thread
    broadcaster_thread = threading.Thread(target=notification_broadcaster, daemon=True)
    broadcaster_thread.start()

    # ========================================================================
    # MAIN
    # ========================================================================

    def run_server(host='0.0.0.0', port=5000, debug=False):
        """Run Flask server"""
        logger.info(f"Starting Human Oversight Dashboard on {host}:{port}")
        socketio.run(app, host=host, port=port, debug=debug)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for CLI"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Human-in-Loop Oversight for Self-Expanding Music Generation'
    )

    parser.add_argument(
        '--server',
        action='store_true',
        help='Run web server'
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Server host (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Server port (default: 5000)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    parser.add_argument(
        '--db',
        default='human_oversight.db',
        help='Database path (default: human_oversight.db)'
    )

    args = parser.parse_args()

    if args.server:
        if not FLASK_AVAILABLE:
            print("ERROR: Flask not installed. Install with: pip install flask flask-socketio flask-cors")
            sys.exit(1)

        run_server(host=args.host, port=args.port, debug=args.debug)
    else:
        # Interactive CLI mode
        print("Human-in-Loop Oversight System")
        print("=" * 50)
        print("\nUse --server to run web interface")
        print("\nExample usage:")
        print("  python human_oversight.py --server")
        print("  python human_oversight.py --server --host localhost --port 8080")


    # Background task to monitor system health
    def system_health_monitor():
        """Background task to monitor system health"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                stats = oversight_engine.get_statistics()
                health_score = stats.get('system_health', {}).get('health_score', 100)

                # Record system metric
                oversight_engine.record_system_metric(
                    metric_name='system_health_score',
                    metric_value=health_score,
                    category='health'
                )

                # Alert if health is low
                if health_score < 50:
                    logger.warning(f"System health is low: {health_score:.1f}/100")

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    # Start health monitor thread
    health_monitor_thread = threading.Thread(target=system_health_monitor, daemon=True)
    health_monitor_thread.start()


# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

class ProposalValidator:
    """Validator for expansion proposals"""

    @staticmethod
    def validate_parameter_name(name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate parameter name.

        Returns:
            (valid, error_message)
        """
        if not name:
            return False, "Parameter name is required"

        if len(name) > 256:
            return False, "Parameter name too long (max 256 characters)"

        # Check format (alphanumeric, underscores only)
        import re
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
            return False, "Parameter name must start with letter and contain only letters, numbers, and underscores"

        return True, None

    @staticmethod
    def validate_parameter_path(path: str) -> Tuple[bool, Optional[str]]:
        """Validate parameter path"""
        if not path:
            return False, "Parameter path is required"

        if len(path) > 512:
            return False, "Parameter path too long (max 512 characters)"

        # Check format (dot-separated components)
        import re
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*(\.[a-zA-Z][a-zA-Z0-9_]*)*$', path):
            return False, "Invalid parameter path format (use dot notation like 'harmony.jazz.voicing')"

        # Check depth (max 5 levels)
        if path.count('.') > 4:
            return False, "Parameter path too deep (max 5 levels)"

        return True, None

    @staticmethod
    def validate_parameter_type(param_type: str) -> Tuple[bool, Optional[str]]:
        """Validate parameter type"""
        valid_types = [
            'continuous', 'integer', 'categorical', 'boolean',
            'array_int', 'array_float', 'probability', 'midi_note',
            'velocity', 'duration'
        ]

        if param_type not in valid_types:
            return False, f"Invalid parameter type. Must be one of: {', '.join(valid_types)}"

        return True, None

    @staticmethod
    def validate_priority(priority: int) -> Tuple[bool, Optional[str]]:
        """Validate priority value"""
        if not isinstance(priority, int):
            return False, "Priority must be an integer"

        if priority < 0 or priority > 100:
            return False, "Priority must be between 0 and 100"

        return True, None

    @classmethod
    def validate_proposal(cls, proposal_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate complete proposal data.

        Returns:
            (valid, list_of_errors)
        """
        errors = []

        # Validate name
        valid, error = cls.validate_parameter_name(proposal_data.get('parameter_name', ''))
        if not valid:
            errors.append(error)

        # Validate path
        valid, error = cls.validate_parameter_path(proposal_data.get('parameter_path', ''))
        if not valid:
            errors.append(error)

        # Validate type
        valid, error = cls.validate_parameter_type(proposal_data.get('parameter_type', ''))
        if not valid:
            errors.append(error)

        # Validate priority
        priority = proposal_data.get('priority', 50)
        valid, error = cls.validate_priority(priority)
        if not valid:
            errors.append(error)

        # Validate description
        description = proposal_data.get('description', '')
        if not description or len(description.strip()) < 10:
            errors.append("Description must be at least 10 characters")

        return len(errors) == 0, errors


class MetricAnalyzer:
    """Analyzer for quality metrics"""

    @staticmethod
    def calculate_metric_trend(metrics: List[Dict], window_size: int = 10) -> Dict:
        """
        Calculate metric trends.

        Args:
            metrics: List of metric records
            window_size: Size of moving average window

        Returns:
            Dictionary with trend information
        """
        if not metrics or len(metrics) < 2:
            return {
                'trend': 'insufficient_data',
                'direction': 'unknown',
                'magnitude': 0.0
            }

        # Sort by time
        sorted_metrics = sorted(metrics, key=lambda m: m.get('recorded_at', ''))

        # Get values
        values = [m.get('metric_value', 0) for m in sorted_metrics]

        # Calculate moving average
        if len(values) >= window_size:
            recent_avg = sum(values[-window_size:]) / window_size
            older_avg = sum(values[:window_size]) / window_size

            magnitude = abs(recent_avg - older_avg)
            direction = 'improving' if recent_avg > older_avg else 'declining'

            return {
                'trend': 'stable' if magnitude < 0.05 else 'changing',
                'direction': direction,
                'magnitude': magnitude,
                'recent_avg': recent_avg,
                'older_avg': older_avg
            }

        return {
            'trend': 'insufficient_data',
            'direction': 'unknown',
            'magnitude': 0.0
        }

    @staticmethod
    def detect_metric_anomalies(metrics: List[Dict], std_threshold: float = 2.0) -> List[Dict]:
        """
        Detect anomalies in metrics using standard deviation.

        Args:
            metrics: List of metric records
            std_threshold: Number of standard deviations for anomaly

        Returns:
            List of anomalous metrics
        """
        if not NUMPY_AVAILABLE or len(metrics) < 5:
            return []

        values = np.array([m.get('metric_value', 0) for m in metrics])
        mean = np.mean(values)
        std = np.std(values)

        anomalies = []
        for i, metric in enumerate(metrics):
            value = metric.get('metric_value', 0)
            z_score = abs((value - mean) / std) if std > 0 else 0

            if z_score > std_threshold:
                anomalies.append({
                    **metric,
                    'z_score': z_score,
                    'anomaly_type': 'outlier'
                })

        return anomalies

    @staticmethod
    def calculate_metric_correlation(
        metrics1: List[float],
        metrics2: List[float]
    ) -> Optional[float]:
        """
        Calculate correlation between two metric series.

        Returns:
            Correlation coefficient or None if insufficient data
        """
        if not NUMPY_AVAILABLE or len(metrics1) != len(metrics2) or len(metrics1) < 3:
            return None

        try:
            return float(np.corrcoef(metrics1, metrics2)[0, 1])
        except Exception:
            return None


class ReportGenerator:
    """Generate reports from oversight data"""

    @staticmethod
    def generate_proposal_summary(engine: HumanOversightEngine) -> Dict:
        """Generate comprehensive proposal summary"""
        stats = engine.get_statistics()

        summary = {
            'total_proposals': stats.get('total_proposals', 0),
            'by_status': stats.get('proposals_by_status', {}),
            'by_source': stats.get('proposals_by_source', {}),
            'avg_review_time_hours': stats.get('avg_review_time_seconds', 0) / 3600,
            'current_parameters': stats.get('current_parameters', 165),
            'parameter_growth_rate': 0.0,  # Would need historical data
            'system_health': stats.get('system_health', {})
        }

        # Calculate acceptance rate
        approved = stats.get('proposals_by_status', {}).get('approved', 0)
        rejected = stats.get('proposals_by_status', {}).get('rejected', 0)
        total_reviewed = approved + rejected

        if total_reviewed > 0:
            summary['acceptance_rate'] = approved / total_reviewed
        else:
            summary['acceptance_rate'] = 0.0

        return summary

    @staticmethod
    def generate_quality_report(engine: HumanOversightEngine) -> Dict:
        """Generate quality metrics report"""
        stats = engine.get_statistics()
        quality_metrics = stats.get('quality_metrics', {})

        report = {
            'metrics_summary': {},
            'overall_quality_score': 0.0,
            'passing_metrics': 0,
            'total_metrics': len(quality_metrics)
        }

        total_pass_rate = 0.0
        for metric_name, metric_data in quality_metrics.items():
            count = metric_data.get('count', 0)
            if count > 0:
                pass_rate = metric_data.get('passed', 0) / count
                total_pass_rate += pass_rate

                if pass_rate >= 0.8:
                    report['passing_metrics'] += 1

            report['metrics_summary'][metric_name] = {
                'average': metric_data.get('avg', 0),
                'min': metric_data.get('min', 0),
                'max': metric_data.get('max', 0),
                'pass_rate': pass_rate if count > 0 else 0.0,
                'count': count
            }

        if len(quality_metrics) > 0:
            report['overall_quality_score'] = total_pass_rate / len(quality_metrics)

        return report

    @staticmethod
    def generate_markdown_report(engine: HumanOversightEngine) -> str:
        """Generate markdown-formatted report"""
        proposal_summary = ReportGenerator.generate_proposal_summary(engine)
        quality_report = ReportGenerator.generate_quality_report(engine)

        report = f"""# Human Oversight System Report

Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

## System Overview

- **Current Parameters**: {proposal_summary['current_parameters']}
- **Total Proposals**: {proposal_summary['total_proposals']}
- **System Health**: {proposal_summary['system_health'].get('health_score', 0):.1f}/100

## Proposal Statistics

### By Status
"""
        for status, count in proposal_summary['by_status'].items():
            report += f"- **{status.title()}**: {count}\n"

        report += f"""
### By Source
"""
        for source, count in proposal_summary['by_source'].items():
            report += f"- **{source.replace('_', ' ').title()}**: {count}\n"

        report += f"""
### Review Metrics
- **Average Review Time**: {proposal_summary['avg_review_time_hours']:.2f} hours
- **Acceptance Rate**: {proposal_summary['acceptance_rate']*100:.1f}%

## Quality Metrics

- **Overall Quality Score**: {quality_report['overall_quality_score']*100:.1f}%
- **Passing Metrics**: {quality_report['passing_metrics']}/{quality_report['total_metrics']}

### Detailed Metrics
"""
        for metric_name, data in quality_report['metrics_summary'].items():
            report += f"""
#### {metric_name.replace('_', ' ').title()}
- Average: {data['average']:.3f}
- Range: {data['min']:.3f} - {data['max']:.3f}
- Pass Rate: {data['pass_rate']*100:.1f}%
- Sample Count: {data['count']}
"""

        return report


class ConfigManager:
    """Configuration management for oversight system"""

    DEFAULT_CONFIG = {
        'database': {
            'path': 'human_oversight.db',
            'echo': False
        },
        'server': {
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False
        },
        'cache': {
            'stats_ttl': 5,  # seconds
            'max_notifications': 50
        },
        'monitoring': {
            'health_check_interval': 60,  # seconds
            'low_health_threshold': 50
        },
        'quality': {
            'thresholds': QUALITY_THRESHOLDS
        },
        'limits': {
            'max_proposals_per_query': 1000,
            'max_audit_logs': 10000
        }
    }

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults"""
        config = ConfigManager.DEFAULT_CONFIG.copy()

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)

                # Deep merge
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config:
                        config[key].update(value)
                    else:
                        config[key] = value

                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

        return config

    @staticmethod
    def save_config(config: Dict, config_path: str) -> bool:
        """Save configuration to file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False


class DataExporter:
    """Enhanced data export functionality"""

    @staticmethod
    def export_to_json(engine: HumanOversightEngine, output_path: str) -> bool:
        """Export all data to JSON"""
        return engine.export_data(output_path, format='json')

    @staticmethod
    def export_proposals_to_csv(engine: HumanOversightEngine, output_path: str) -> bool:
        """Export proposals to CSV format"""
        try:
            import csv

            proposals = engine.get_proposals(limit=10000)

            with open(output_path, 'w', newline='') as csvfile:
                if not proposals:
                    return True

                fieldnames = proposals[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for proposal in proposals:
                    # Convert complex types to strings
                    row = {k: json.dumps(v) if isinstance(v, (dict, list)) else v
                           for k, v in proposal.items()}
                    writer.writerow(row)

            logger.info(f"Exported {len(proposals)} proposals to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export proposals to CSV: {e}")
            return False

    @staticmethod
    def export_metrics_to_csv(engine: HumanOversightEngine, output_path: str) -> bool:
        """Export metrics to CSV format"""
        try:
            import csv

            stats = engine.get_statistics()
            quality_metrics = stats.get('quality_metrics', {})

            with open(output_path, 'w', newline='') as csvfile:
                fieldnames = ['metric_type', 'average', 'min', 'max', 'count', 'passed', 'failed', 'pass_rate']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for metric_type, data in quality_metrics.items():
                    count = data.get('count', 0)
                    pass_rate = data.get('passed', 0) / count if count > 0 else 0

                    writer.writerow({
                        'metric_type': metric_type,
                        'average': data.get('avg', 0),
                        'min': data.get('min', 0),
                        'max': data.get('max', 0),
                        'count': count,
                        'passed': data.get('passed', 0),
                        'failed': data.get('failed', 0),
                        'pass_rate': pass_rate
                    })

            logger.info(f"Exported metrics to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics to CSV: {e}")
            return False


class SystemMonitor:
    """System monitoring and alerting"""

    def __init__(self, engine: HumanOversightEngine):
        self.engine = engine
        self.alerts = deque(maxlen=100)

    def check_system_health(self) -> Dict:
        """Perform comprehensive system health check"""
        stats = self.engine.get_statistics()

        health_report = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'overall_health': 'healthy',
            'issues': []
        }

        # Check pending queue size
        pending = stats.get('proposals_by_status', {}).get('pending', 0)
        if pending > 50:
            health_report['issues'].append({
                'severity': 'warning',
                'category': 'queue',
                'message': f'Large pending queue: {pending} proposals'
            })

        # Check failure rate
        failed = stats.get('proposals_by_status', {}).get('failed', 0)
        total = stats.get('total_proposals', 0)
        if total > 0 and (failed / total) > 0.2:
            health_report['issues'].append({
                'severity': 'error',
                'category': 'failures',
                'message': f'High failure rate: {(failed/total)*100:.1f}%'
            })

        # Check quality metrics
        quality_metrics = stats.get('quality_metrics', {})
        for metric_name, data in quality_metrics.items():
            count = data.get('count', 0)
            if count > 0:
                pass_rate = data.get('passed', 0) / count
                if pass_rate < 0.6:
                    health_report['issues'].append({
                        'severity': 'warning',
                        'category': 'quality',
                        'message': f'Low pass rate for {metric_name}: {pass_rate*100:.1f}%'
                    })

        # Determine overall health
        if any(issue['severity'] == 'error' for issue in health_report['issues']):
            health_report['overall_health'] = 'unhealthy'
        elif any(issue['severity'] == 'warning' for issue in health_report['issues']):
            health_report['overall_health'] = 'degraded'

        return health_report

    def add_alert(self, severity: str, category: str, message: str):
        """Add alert to alert queue"""
        alert = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'severity': severity,
            'category': category,
            'message': message
        }
        self.alerts.append(alert)
        logger.warning(f"Alert [{severity}] {category}: {message}")

    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts"""
        return list(self.alerts)[-limit:]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()
