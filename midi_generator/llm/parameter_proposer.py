"""
LLM Parameter Proposal Agent - Agent 11
========================================

Uses Claude API to propose well-designed parameter definitions based on gap analysis
from MIDI reconstruction failures.

Core Innovation:
When the gap detector identifies missing musical capabilities, this agent uses Claude
to propose comprehensive, musically-informed parameter definitions that can be
immediately integrated into the system.

Architecture:
- Gap Analysis → LLM Prompt → Claude API → JSON Response → Validation → Parameter Definition
- Musical knowledge embedded in system prompts
- Automatic parameter naming, typing, and range selection
- Test case generation for validation
- Conflict detection with existing parameters
- Genre-specific example value generation

Author: Agent 11 - Parameter Proposal Agent
License: MIT
"""

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
import os

# Optional anthropic import - graceful degradation if not available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Import from existing system
import sys
sys.path.append(str(Path(__file__).parent.parent))

from parameters.universal_registry import (
    UniversalParameterRegistry,
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
    MusicalImpact,
    REGISTRY
)

# ============================================================================
# Configuration
# ============================================================================

# API Configuration
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4000
TEMPERATURE = 0.3  # Low temperature for consistency

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class ProposalStatus(Enum):
    """Status of a parameter proposal"""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    DEPRECATED = "deprecated"


@dataclass
class TestCase:
    """Test case for parameter validation"""
    value: Any
    expected_description: str
    test_features: Dict[str, Any]
    test_midi_exists: bool = False
    test_midi_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'value': self.value,
            'expected': self.expected_description,
            'test_features': self.test_features,
            'test_midi_exists': self.test_midi_exists,
            'test_midi_path': self.test_midi_path
        }


@dataclass
class ParameterProposal:
    """
    Complete parameter proposal from LLM

    This represents a fully-specified parameter definition that can be
    integrated into the system after validation.
    """
    # Core definition
    name: str
    param_type: str  # Will be converted to ParameterType
    range: Union[List[Any], Tuple[Any, Any]]
    default: Any
    description: str

    # Musical context
    musical_context: str
    implementation_strategy: str

    # Integration
    affected_features: List[str]
    generator_integration_points: List[str]

    # Testing
    test_cases: List[TestCase]

    # Examples
    example_values: Dict[str, Any]

    # Relationships
    related_parameters: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Metadata
    proposal_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    status: ProposalStatus = ProposalStatus.PENDING
    confidence_score: float = 0.0
    gap_analysis_id: Optional[str] = None

    # Validation results
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'type': self.param_type,
            'range': list(self.range) if isinstance(self.range, tuple) else self.range,
            'default': self.default,
            'description': self.description,
            'musical_context': self.musical_context,
            'implementation_strategy': self.implementation_strategy,
            'affected_features': self.affected_features,
            'generator_integration_points': self.generator_integration_points,
            'test_cases': [tc.to_dict() for tc in self.test_cases],
            'example_values': self.example_values,
            'related_parameters': self.related_parameters,
            'conflicts': self.conflicts,
            'dependencies': self.dependencies,
            'proposal_id': self.proposal_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'status': self.status.value,
            'confidence_score': self.confidence_score,
            'gap_analysis_id': self.gap_analysis_id,
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings
        }

    def to_parameter_definition(self) -> ParameterDefinition:
        """Convert to ParameterDefinition for registry integration"""

        # Parse parameter type
        try:
            param_type = ParameterType(self.param_type.lower())
        except ValueError:
            # Try to map common variations
            type_mapping = {
                'continuous': ParameterType.CONTINUOUS,
                'categorical': ParameterType.CATEGORICAL,
                'boolean': ParameterType.BOOLEAN,
                'array_int': ParameterType.ARRAY_INT,
                'array_float': ParameterType.ARRAY_FLOAT,
                'probability': ParameterType.PROBABILITY,
                'integer': ParameterType.INTEGER,
                'midi_note': ParameterType.MIDI_NOTE,
                'velocity': ParameterType.VELOCITY,
                'duration': ParameterType.DURATION
            }
            param_type = type_mapping.get(self.param_type.lower(), ParameterType.CONTINUOUS)

        # Parse category from name
        domain = self.name.split('.')[0]
        category_mapping = {
            'harmony': ParameterCategory.HARMONY,
            'melody': ParameterCategory.MELODY,
            'rhythm': ParameterCategory.RHYTHM,
            'bass': ParameterCategory.BASS,
            'voice': ParameterCategory.VOICE,
            'drums': ParameterCategory.DRUMS,
            'timbre': ParameterCategory.TIMBRE,
            'dynamics': ParameterCategory.DYNAMICS,
            'articulation': ParameterCategory.ARTICULATION,
            'structure': ParameterCategory.STRUCTURE,
            'genre': ParameterCategory.GENRE,
            'style': ParameterCategory.STYLE
        }
        category = category_mapping.get(domain, None)

        # Extract min/max or options
        min_value = None
        max_value = None
        options = None

        if param_type in [ParameterType.CONTINUOUS, ParameterType.INTEGER,
                         ParameterType.PROBABILITY, ParameterType.DURATION]:
            if len(self.range) == 2:
                min_value = float(self.range[0])
                max_value = float(self.range[1])
        elif param_type == ParameterType.CATEGORICAL:
            options = list(self.range)

        # Extract genre relevance from example_values
        genre_relevance = list(self.example_values.keys())

        return ParameterDefinition(
            name=self.name.split('.')[-1],
            full_path=self.name,
            description=self.description,
            param_type=param_type,
            default_value=self.default,
            min_value=min_value,
            max_value=max_value,
            options=options,
            category=category,
            musical_impact=MusicalImpact.HIGH,  # Conservative default
            genre_relevance=genre_relevance,
            depends_on=self.dependencies,
            mutually_exclusive_with=self.conflicts,
            learnable=True
        )


@dataclass
class GapAnalysis:
    """
    Gap analysis input from gap detector

    This represents a detected gap in the system's capabilities based on
    reconstruction failures.
    """
    suggested_parameter: str
    affected_features: List[str]
    avg_error: float
    impact_score: float
    rationale: str
    confidence: float
    priority: str
    parameter_info: Dict[str, Any]

    # Optional metadata
    gap_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    sample_midi_files: List[str] = field(default_factory=list)
    feature_statistics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Validation System
# ============================================================================

class ProposalValidator:
    """
    Validates LLM-generated parameter proposals

    Ensures proposals meet all requirements:
    - Naming conventions
    - Type consistency
    - Range validity
    - No duplicates
    - Musical coherence
    - Implementation clarity
    """

    def __init__(self, registry: UniversalParameterRegistry):
        self.registry = registry
        self.existing_params = set(registry.get_all_parameters())

        # Validation rules
        self.valid_types = {
            'CONTINUOUS', 'INTEGER', 'CATEGORICAL', 'BOOLEAN',
            'ARRAY_INT', 'ARRAY_FLOAT', 'PROBABILITY',
            'MIDI_NOTE', 'VELOCITY', 'DURATION'
        }

        self.valid_domains = {
            'harmony', 'melody', 'rhythm', 'bass', 'voice', 'drums',
            'timbre', 'dynamics', 'articulation', 'structure',
            'genre', 'style', 'instrumentation', 'expression', 'texture'
        }

        self.naming_pattern = re.compile(r'^[a-z_]+\.[a-z_]+\.[a-z_]+$')

    def validate(self, proposal: ParameterProposal) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a parameter proposal

        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # 1. Name validation
        if not self._validate_name(proposal.name, errors, warnings):
            return False, errors, warnings

        # 2. Type validation
        if not self._validate_type(proposal, errors, warnings):
            return False, errors, warnings

        # 3. Range validation
        if not self._validate_range(proposal, errors, warnings):
            return False, errors, warnings

        # 4. Default validation
        if not self._validate_default(proposal, errors, warnings):
            return False, errors, warnings

        # 5. Test cases validation
        if not self._validate_test_cases(proposal, errors, warnings):
            return False, errors, warnings

        # 6. Musical coherence validation
        self._validate_musical_coherence(proposal, errors, warnings)

        # 7. Implementation clarity validation
        self._validate_implementation(proposal, errors, warnings)

        # 8. Relationship validation
        self._validate_relationships(proposal, errors, warnings)

        # Success if no errors (warnings are acceptable)
        is_valid = len(errors) == 0

        return is_valid, errors, warnings

    def _validate_name(self, name: str, errors: List[str], warnings: List[str]) -> bool:
        """Validate parameter name"""

        # Check format: domain.module.parameter
        if not self.naming_pattern.match(name):
            errors.append(
                f"Invalid name format '{name}'. "
                f"Expected: domain.module.parameter (lowercase with underscores)"
            )
            return False

        # Check if already exists
        if name in self.existing_params:
            errors.append(f"Parameter '{name}' already exists in registry")
            return False

        # Check domain validity
        domain = name.split('.')[0]
        if domain not in self.valid_domains:
            warnings.append(
                f"Domain '{domain}' is not standard. "
                f"Valid domains: {', '.join(sorted(self.valid_domains))}"
            )

        # Check name length
        if len(name) > 100:
            warnings.append(f"Parameter name is very long ({len(name)} chars)")

        return True

    def _validate_type(self, proposal: ParameterProposal,
                       errors: List[str], warnings: List[str]) -> bool:
        """Validate parameter type"""

        param_type = proposal.param_type.upper()

        if param_type not in self.valid_types:
            errors.append(
                f"Invalid type '{param_type}'. "
                f"Valid types: {', '.join(sorted(self.valid_types))}"
            )
            return False

        # Normalize type
        proposal.param_type = param_type

        return True

    def _validate_range(self, proposal: ParameterProposal,
                        errors: List[str], warnings: List[str]) -> bool:
        """Validate parameter range"""

        param_type = proposal.param_type
        param_range = proposal.range

        # Continuous/Integer/Probability/Duration: [min, max]
        if param_type in ['CONTINUOUS', 'INTEGER', 'PROBABILITY', 'DURATION']:
            if not isinstance(param_range, (list, tuple)) or len(param_range) != 2:
                errors.append(
                    f"Type {param_type} requires [min, max] range, got {param_range}"
                )
                return False

            min_val, max_val = param_range[0], param_range[1]

            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                errors.append(f"Range values must be numeric, got {param_range}")
                return False

            if min_val >= max_val:
                errors.append(f"Min ({min_val}) must be < max ({max_val})")
                return False

            # Special validation for probability
            if param_type == 'PROBABILITY':
                if min_val < 0.0 or max_val > 1.0:
                    errors.append(f"Probability range must be within [0.0, 1.0], got {param_range}")
                    return False

            # Special validation for MIDI values
            if param_type in ['MIDI_NOTE', 'VELOCITY']:
                if min_val < 0 or max_val > 127:
                    errors.append(f"{param_type} range must be within [0, 127], got {param_range}")
                    return False

        # Categorical: list of options
        elif param_type == 'CATEGORICAL':
            if not isinstance(param_range, (list, tuple)):
                errors.append(f"Categorical type requires list of options, got {type(param_range)}")
                return False

            if len(param_range) < 2:
                warnings.append(
                    f"Categorical parameter has only {len(param_range)} option(s). "
                    f"Consider using BOOLEAN or rethinking design."
                )

            # Check for duplicates
            if len(set(param_range)) != len(param_range):
                errors.append(f"Categorical options contain duplicates: {param_range}")
                return False

        # Boolean: no range needed
        elif param_type == 'BOOLEAN':
            if param_range is not None and param_range != [True, False]:
                warnings.append(f"Boolean type doesn't need explicit range")

        # Array types: optional range for element values
        elif param_type in ['ARRAY_INT', 'ARRAY_FLOAT']:
            # Range is optional for arrays
            pass

        return True

    def _validate_default(self, proposal: ParameterProposal,
                          errors: List[str], warnings: List[str]) -> bool:
        """Validate default value"""

        param_type = proposal.param_type
        default = proposal.default
        param_range = proposal.range

        # Check type-specific defaults
        if param_type in ['CONTINUOUS', 'INTEGER', 'PROBABILITY', 'DURATION']:
            if not isinstance(default, (int, float)):
                errors.append(f"Default must be numeric for {param_type}, got {type(default)}")
                return False

            min_val, max_val = param_range[0], param_range[1]
            if not (min_val <= default <= max_val):
                errors.append(
                    f"Default {default} outside valid range [{min_val}, {max_val}]"
                )
                return False

            # Warn if default is at extremes
            if default == min_val or default == max_val:
                warnings.append(
                    f"Default value {default} is at range boundary. "
                    f"Consider using middle of range for better generalization."
                )

        elif param_type == 'CATEGORICAL':
            if default not in param_range:
                errors.append(
                    f"Default '{default}' not in categorical options {param_range}"
                )
                return False

        elif param_type == 'BOOLEAN':
            if not isinstance(default, bool):
                errors.append(f"Default must be boolean for BOOLEAN type, got {type(default)}")
                return False

        elif param_type in ['ARRAY_INT', 'ARRAY_FLOAT']:
            if not isinstance(default, (list, tuple)):
                errors.append(f"Default must be array for {param_type}, got {type(default)}")
                return False

        return True

    def _validate_test_cases(self, proposal: ParameterProposal,
                            errors: List[str], warnings: List[str]) -> bool:
        """Validate test cases"""

        if len(proposal.test_cases) < 2:
            errors.append(
                f"Must provide at least 2 test cases, got {len(proposal.test_cases)}"
            )
            return False

        # Check test case coverage
        param_type = proposal.param_type
        param_range = proposal.range

        if param_type in ['CONTINUOUS', 'INTEGER', 'PROBABILITY', 'DURATION']:
            # Should test min, max, and middle values
            test_values = [tc.value for tc in proposal.test_cases]
            min_val, max_val = param_range[0], param_range[1]

            has_min = any(abs(v - min_val) < 0.001 for v in test_values)
            has_max = any(abs(v - max_val) < 0.001 for v in test_values)

            if not has_min or not has_max:
                warnings.append(
                    f"Test cases should cover full range [{min_val}, {max_val}]. "
                    f"Currently testing: {test_values}"
                )

        elif param_type == 'CATEGORICAL':
            # Should test each option
            test_values = [tc.value for tc in proposal.test_cases]
            untested_options = set(param_range) - set(test_values)

            if untested_options:
                warnings.append(
                    f"Not all categorical options tested. "
                    f"Untested: {untested_options}"
                )

        # Validate test case structure
        for i, tc in enumerate(proposal.test_cases):
            if not tc.expected_description:
                errors.append(f"Test case {i}: missing expected description")
                return False

            if not tc.test_features:
                warnings.append(f"Test case {i}: no test features specified")

        return True

    def _validate_musical_coherence(self, proposal: ParameterProposal,
                                    errors: List[str], warnings: List[str]):
        """Validate musical coherence"""

        # Check description quality
        if len(proposal.description) < 20:
            warnings.append(
                f"Description is very short ({len(proposal.description)} chars). "
                f"Provide more detail."
            )

        # Check musical context
        if len(proposal.musical_context) < 50:
            warnings.append(
                f"Musical context is thin ({len(proposal.musical_context)} chars). "
                f"Add more music theory background, composer references, or genre context."
            )

        # Check for musical terminology
        musical_terms = [
            'chord', 'voicing', 'harmony', 'melody', 'rhythm', 'scale',
            'interval', 'progression', 'cadence', 'resolution', 'tension',
            'jazz', 'classical', 'blues', 'rock', 'bebop', 'swing'
        ]

        combined_text = (
            proposal.description.lower() + ' ' +
            proposal.musical_context.lower()
        )

        has_musical_terms = any(term in combined_text for term in musical_terms)
        if not has_musical_terms:
            warnings.append(
                "No musical terminology found in description/context. "
                "Ensure parameter has clear musical meaning."
            )

    def _validate_implementation(self, proposal: ParameterProposal,
                                errors: List[str], warnings: List[str]):
        """Validate implementation strategy"""

        # Check implementation strategy detail
        if len(proposal.implementation_strategy) < 50:
            warnings.append(
                f"Implementation strategy is vague ({len(proposal.implementation_strategy)} chars). "
                f"Provide specific algorithm/logic details."
            )

        # Check for integration points
        if not proposal.generator_integration_points:
            warnings.append(
                "No generator integration points specified. "
                "Where will this parameter be used?"
            )
        else:
            # Validate integration point format
            for point in proposal.generator_integration_points:
                if '::' not in point:
                    warnings.append(
                        f"Integration point '{point}' should use format 'file.py::function()'"
                    )

    def _validate_relationships(self, proposal: ParameterProposal,
                               errors: List[str], warnings: List[str]):
        """Validate parameter relationships"""

        # Check conflicts and dependencies for validity
        for conflict in proposal.conflicts:
            if conflict not in self.existing_params:
                warnings.append(
                    f"Conflict parameter '{conflict}' not found in registry. "
                    f"Will need to be added later."
                )

        for dependency in proposal.dependencies:
            if dependency not in self.existing_params:
                warnings.append(
                    f"Dependency parameter '{dependency}' not found in registry. "
                    f"Will need to be added later."
                )

        # Check for circular dependencies
        if proposal.name in proposal.dependencies:
            errors.append("Parameter cannot depend on itself")

        # Check for self-conflicts
        if proposal.name in proposal.conflicts:
            errors.append("Parameter cannot conflict with itself")


# ============================================================================
# History Tracking
# ============================================================================

class ProposalHistory:
    """
    Tracks history of all parameter proposals

    Maintains:
    - All proposals (accepted and rejected)
    - Timestamps and metadata
    - Validation results
    - Implementation status
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent / 'proposal_history.json'

        self.storage_path = storage_path
        self.proposals: Dict[str, ParameterProposal] = {}

        # Load existing history
        if self.storage_path.exists():
            self.load()

    def add_proposal(self, proposal: ParameterProposal):
        """Add a proposal to history"""

        # Generate ID if not present
        if proposal.proposal_id is None:
            proposal.proposal_id = self._generate_id()

        # Set timestamp if not present
        if proposal.timestamp is None:
            proposal.timestamp = datetime.now()

        self.proposals[proposal.proposal_id] = proposal
        self.save()

    def get_proposal(self, proposal_id: str) -> Optional[ParameterProposal]:
        """Get a proposal by ID"""
        return self.proposals.get(proposal_id)

    def get_by_status(self, status: ProposalStatus) -> List[ParameterProposal]:
        """Get all proposals with a given status"""
        return [p for p in self.proposals.values() if p.status == status]

    def get_by_parameter_name(self, name: str) -> List[ParameterProposal]:
        """Get all proposals for a parameter name"""
        return [p for p in self.proposals.values() if p.name == name]

    def update_status(self, proposal_id: str, new_status: ProposalStatus):
        """Update proposal status"""
        if proposal_id in self.proposals:
            self.proposals[proposal_id].status = new_status
            self.save()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about proposals"""

        total = len(self.proposals)
        by_status = {}

        for status in ProposalStatus:
            count = len(self.get_by_status(status))
            by_status[status.value] = count

        return {
            'total_proposals': total,
            'by_status': by_status,
            'acceptance_rate': by_status.get('implemented', 0) / max(total, 1),
            'pending_count': by_status.get('pending', 0) + by_status.get('validated', 0)
        }

    def save(self):
        """Save history to disk"""

        # Convert to JSON-serializable format
        data = {
            'proposals': {
                pid: self._proposal_to_json(p)
                for pid, p in self.proposals.items()
            },
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'total_proposals': len(self.proposals)
            }
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load history from disk"""

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            for pid, p_data in data['proposals'].items():
                self.proposals[pid] = self._json_to_proposal(p_data)

            logger.info(f"Loaded {len(self.proposals)} proposals from history")

        except Exception as e:
            logger.error(f"Failed to load proposal history: {e}")

    def _generate_id(self) -> str:
        """Generate unique proposal ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        counter = len(self.proposals)
        return f"PROP_{timestamp}_{counter:04d}"

    def _proposal_to_json(self, proposal: ParameterProposal) -> Dict[str, Any]:
        """Convert proposal to JSON-serializable dict"""
        data = proposal.to_dict()
        return data

    def _json_to_proposal(self, data: Dict[str, Any]) -> ParameterProposal:
        """Convert JSON dict to ParameterProposal"""

        # Convert test cases
        test_cases = [
            TestCase(
                value=tc['value'],
                expected_description=tc['expected'],
                test_features=tc['test_features'],
                test_midi_exists=tc.get('test_midi_exists', False),
                test_midi_path=tc.get('test_midi_path')
            )
            for tc in data['test_cases']
        ]

        # Convert timestamp
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data['timestamp'])

        # Convert status
        status = ProposalStatus(data.get('status', 'pending'))

        return ParameterProposal(
            name=data['name'],
            param_type=data['type'],
            range=data['range'],
            default=data['default'],
            description=data['description'],
            musical_context=data['musical_context'],
            implementation_strategy=data['implementation_strategy'],
            affected_features=data['affected_features'],
            generator_integration_points=data['generator_integration_points'],
            test_cases=test_cases,
            example_values=data['example_values'],
            related_parameters=data.get('related_parameters', []),
            conflicts=data.get('conflicts', []),
            dependencies=data.get('dependencies', []),
            proposal_id=data.get('proposal_id'),
            timestamp=timestamp,
            status=status,
            confidence_score=data.get('confidence_score', 0.0),
            gap_analysis_id=data.get('gap_analysis_id'),
            validation_errors=data.get('validation_errors', []),
            validation_warnings=data.get('validation_warnings', [])
        )


# ============================================================================
# Parameter Formatting Utilities
# ============================================================================

class ParameterFormatter:
    """
    Utilities for formatting parameter information for LLM prompts
    """

    def __init__(self, registry: UniversalParameterRegistry):
        self.registry = registry

    def format_existing_parameters(self, max_per_domain: int = 15) -> str:
        """
        Format existing parameters for LLM context

        Groups by domain and shows representative examples.
        """

        params_by_domain = {}
        for name in self.registry.get_all_parameters():
            domain = name.split('.')[0]
            if domain not in params_by_domain:
                params_by_domain[domain] = []
            params_by_domain[domain].append(name)

        lines = []
        for domain, params in sorted(params_by_domain.items()):
            lines.append(f"\n{domain.upper()} ({len(params)} parameters):")

            # Show first N as examples
            for param in sorted(params)[:max_per_domain]:
                lines.append(f"  - {param}")

            if len(params) > max_per_domain:
                lines.append(f"  ... and {len(params) - max_per_domain} more")

        return '\n'.join(lines)

    def format_parameter_examples(self) -> str:
        """
        Format detailed parameter examples for LLM reference
        """

        # Select diverse examples
        example_params = [
            'harmony.voicing.spread',
            'harmony.voicing.type',
            'rhythm.swing.amount',
            'melody.intervals.stepwise_probability'
        ]

        examples = []
        for param_name in example_params:
            param = self.registry.get(param_name)
            if param:
                examples.append(self._format_single_parameter(param))

        return '\n\n'.join(examples)

    def _format_single_parameter(self, param: ParameterDefinition) -> str:
        """Format a single parameter definition"""

        lines = [
            f"Parameter: {param.full_path}",
            f"Type: {param.param_type.value}",
            f"Default: {param.default_value}",
        ]

        if param.min_value is not None and param.max_value is not None:
            lines.append(f"Range: [{param.min_value}, {param.max_value}]")

        if param.options:
            lines.append(f"Options: {param.options}")

        lines.append(f"Description: {param.description}")

        if param.genre_relevance:
            lines.append(f"Genres: {', '.join(param.genre_relevance)}")

        return '\n'.join(lines)

    def format_gap_analysis(self, gap: GapAnalysis) -> str:
        """Format gap analysis for LLM prompt"""

        lines = [
            f"Suggested Parameter: {gap.suggested_parameter}",
            "",
            "Affected Features:",
            *[f"  - {f}" for f in gap.affected_features],
            "",
            f"Average Reconstruction Error: {gap.avg_error:.2f} (0.0=perfect, 1.0=total failure)",
            f"Impact Score: {gap.impact_score:.2f}",
            f"Confidence: {gap.confidence:.2f}",
            f"Priority: {gap.priority}",
            "",
            f"Rationale: {gap.rationale}",
        ]

        if gap.parameter_info:
            lines.append("")
            lines.append("Parameter Type Hint: " + gap.parameter_info.get('type', 'UNKNOWN'))

            if 'musical_rationale' in gap.parameter_info:
                lines.append("")
                lines.append("Musical Context: " + gap.parameter_info['musical_rationale'])

            if 'typical_usage' in gap.parameter_info:
                lines.append("")
                lines.append("Typical Usage Examples:")
                lines.append(json.dumps(gap.parameter_info['typical_usage'], indent=2))

        return '\n'.join(lines)


# ============================================================================
# Main LLM Parameter Proposal Agent
# ============================================================================

class LLMParameterProposalAgent:
    """
    Main agent for LLM-powered parameter proposal

    Uses Claude API to generate comprehensive parameter definitions from gap analysis.

    Workflow:
    1. Receive gap analysis from gap detector
    2. Build LLM prompt with musical context
    3. Call Claude API
    4. Parse JSON response
    5. Validate proposal
    6. Track in history
    7. Return validated proposal (or errors)
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 registry: Optional[UniversalParameterRegistry] = None,
                 model: str = CLAUDE_MODEL):
        """
        Initialize the agent

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            registry: Parameter registry (or use global REGISTRY)
            model: Claude model to use
        """

        # API setup
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')

        if not ANTHROPIC_AVAILABLE:
            logger.warning(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
            self.llm = None
        elif api_key is None:
            logger.warning(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
            self.llm = None
        else:
            self.llm = anthropic.Anthropic(api_key=api_key)

        self.model = model

        # Registry
        self.registry = registry if registry else REGISTRY

        # Components
        self.validator = ProposalValidator(self.registry)
        self.formatter = ParameterFormatter(self.registry)
        self.history = ProposalHistory()

        # Metrics
        self.metrics = {
            'total_proposals': 0,
            'successful_proposals': 0,
            'failed_proposals': 0,
            'api_calls': 0,
            'api_errors': 0
        }

        logger.info(f"Initialized LLMParameterProposalAgent with model {model}")

    def propose_parameter(self, gap_analysis: Union[GapAnalysis, Dict[str, Any]]) -> ParameterProposal:
        """
        Propose a parameter definition based on gap analysis

        Args:
            gap_analysis: Gap analysis from gap detector

        Returns:
            ParameterProposal with validation results

        Raises:
            ValueError: If gap analysis is invalid
            RuntimeError: If LLM API fails
        """

        # Convert dict to GapAnalysis if needed
        if isinstance(gap_analysis, dict):
            gap_analysis = self._dict_to_gap_analysis(gap_analysis)

        logger.info(f"Proposing parameter for: {gap_analysis.suggested_parameter}")

        # Build prompt
        prompt = self._build_proposal_prompt(gap_analysis)
        system_prompt = self._get_system_prompt()

        # Call Claude
        try:
            response = self._call_claude(system_prompt, prompt)
            self.metrics['api_calls'] += 1
            self.metrics['total_proposals'] += 1

        except Exception as e:
            self.metrics['api_errors'] += 1
            logger.error(f"Claude API call failed: {e}")
            raise RuntimeError(f"LLM API call failed: {e}")

        # Parse response
        try:
            proposal = self._parse_llm_response(response, gap_analysis)
        except Exception as e:
            self.metrics['failed_proposals'] += 1
            logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError(f"Failed to parse LLM response: {e}")

        # Validate
        is_valid, errors, warnings = self.validator.validate(proposal)

        proposal.validation_errors = errors
        proposal.validation_warnings = warnings

        if is_valid:
            proposal.status = ProposalStatus.VALIDATED
            self.metrics['successful_proposals'] += 1
            logger.info(f"✅ Successfully proposed parameter: {proposal.name}")
        else:
            proposal.status = ProposalStatus.REJECTED
            self.metrics['failed_proposals'] += 1
            logger.warning(f"❌ Proposal rejected for: {proposal.name}")
            logger.warning(f"Errors: {errors}")

        if warnings:
            logger.info(f"⚠️  Warnings: {warnings}")

        # Track in history
        self.history.add_proposal(proposal)

        return proposal

    def _get_system_prompt(self) -> str:
        """Build comprehensive system prompt for Claude"""

        existing_params = self.formatter.format_existing_parameters()
        param_examples = self.formatter.format_parameter_examples()

        return f"""
You are an expert music theorist and software engineer designing parameters for a generative music system.

Your task is to create well-designed, musically-informed parameter definitions based on gap analysis from MIDI reconstruction failures.

PARAMETER DESIGN PRINCIPLES:

1. **Musical Validity**: Parameter must have clear musical meaning
   - Based on established music theory concepts
   - Used by real composers/musicians
   - Has clear audible effect

2. **Implementation Clarity**: How generator uses parameter must be obvious
   - Specific algorithm/logic described
   - Integration points identified
   - Edge cases handled

3. **Appropriate Granularity**:
   - One parameter = one musical concept
   - Not too broad (e.g., "jazz_style")
   - Not too narrow (e.g., "C_major_triad_in_bar_5")

4. **Type & Range Selection**:
   - CONTINUOUS: Probabilities (0-1), densities (0-1), amounts (0-1)
   - INTEGER: Counts, MIDI values, discrete amounts
   - CATEGORICAL: Distinct musical choices (chord types, scales, patterns)
   - BOOLEAN: On/off features (walking_bass, brushes, modal_harmony)
   - PROBABILITY: Special continuous in [0.0, 1.0]
   - ARRAY_INT/ARRAY_FLOAT: Patterns, rhythms, custom sequences

5. **Default Values**:
   - Should produce reasonable music without user adjustment
   - Middle of range for creative parameters
   - Genre-appropriate for stylistic parameters

6. **Test Cases**:
   - Cover full parameter range
   - Define expected behavior clearly
   - Include measurable feature outcomes

NAMING CONVENTION:
Format: domain.module.parameter
- domain: {{harmony, melody, rhythm, dynamics, texture, structure, instrumentation, articulation, expression, bass, drums, genre, style}}
- module: specific subcategory
- parameter: descriptive name

Examples:
- harmony.voicing.quartal_probability
- melody.interval.bebop_chromatic_prob
- instrumentation.drums.brush_technique
- rhythm.pattern.clave_pattern
- bass.style.walking_probability

EXISTING PARAMETERS (avoid duplicates):
{existing_params}

PARAMETER EXAMPLES (for reference):
{param_examples}

OUTPUT FORMAT (strict JSON):
{{
  "name": "domain.module.parameter",
  "type": "CONTINUOUS|INTEGER|CATEGORICAL|BOOLEAN|PROBABILITY|ARRAY_INT|ARRAY_FLOAT",
  "range": [min, max] or [value1, value2, ...],
  "default": <appropriate_default>,
  "description": "Clear one-sentence description",
  "musical_context": "When/why this is used in real music, reference composers/styles",
  "implementation_strategy": "Detailed explanation of how generator should use this parameter",
  "affected_features": ["feature1", "feature2", ...],
  "generator_integration_points": ["file.py::function()", ...],
  "test_cases": [
    {{
      "value": <test_value>,
      "expected": "Clear description of expected musical output",
      "test_features": {{"feature": expected_value}}
    }}
  ],
  "example_values": {{
    "genre1": value,
    "genre2": value
  }},
  "related_parameters": ["param1", "param2"],
  "conflicts": ["param_that_contradicts"],
  "dependencies": ["param_that_should_be_set_together"]
}}

CRITICAL: Output ONLY valid JSON, no additional text or markdown formatting.
"""

    def _build_proposal_prompt(self, gap: GapAnalysis) -> str:
        """Build specific prompt for this gap"""

        gap_formatted = self.formatter.format_gap_analysis(gap)

        return f"""
GAP ANALYSIS:

{gap_formatted}

YOUR TASK:
Design a complete, implementable parameter definition for this missing musical capability.

Consider:
1. Exact parameter name (follow naming convention)
2. Appropriate type and range
3. Reasonable default value
4. Clear implementation strategy
5. Comprehensive test cases (minimum 3, covering full range)
6. Genre-specific example values
7. Related parameters and potential conflicts

Output ONLY valid JSON following the format specified in your system prompt.
No markdown code blocks, no additional commentary.
"""

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude API"""

        if self.llm is None:
            raise RuntimeError("No API key configured. Cannot call Claude API.")

        logger.debug(f"Calling Claude API with model {self.model}")

        response = self.llm.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract text content
        text_content = response.content[0].text

        logger.debug(f"Received {len(text_content)} chars from Claude")

        return text_content

    def _parse_llm_response(self, response_text: str, gap: GapAnalysis) -> ParameterProposal:
        """Parse Claude's JSON response into ParameterProposal"""

        # Claude sometimes wraps JSON in markdown code blocks - remove them
        cleaned_text = response_text.strip()

        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json")[1].split("```")[0]
        elif "```" in cleaned_text:
            # Try to find JSON in any code block
            cleaned_text = cleaned_text.split("```")[1].split("```")[0]

        cleaned_text = cleaned_text.strip()

        # Parse JSON
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response text: {cleaned_text[:500]}...")
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

        # Convert to ParameterProposal
        try:
            # Parse test cases
            test_cases = []
            for tc_data in data.get('test_cases', []):
                test_cases.append(TestCase(
                    value=tc_data['value'],
                    expected_description=tc_data['expected'],
                    test_features=tc_data.get('test_features', {})
                ))

            proposal = ParameterProposal(
                name=data['name'],
                param_type=data['type'],
                range=data['range'],
                default=data['default'],
                description=data['description'],
                musical_context=data['musical_context'],
                implementation_strategy=data['implementation_strategy'],
                affected_features=data['affected_features'],
                generator_integration_points=data['generator_integration_points'],
                test_cases=test_cases,
                example_values=data['example_values'],
                related_parameters=data.get('related_parameters', []),
                conflicts=data.get('conflicts', []),
                dependencies=data.get('dependencies', []),
                gap_analysis_id=gap.gap_id,
                confidence_score=gap.confidence
            )

            return proposal

        except KeyError as e:
            raise ValueError(f"Missing required field in LLM response: {e}")
        except Exception as e:
            raise ValueError(f"Error constructing ParameterProposal: {e}")

    def _dict_to_gap_analysis(self, data: Dict[str, Any]) -> GapAnalysis:
        """Convert dictionary to GapAnalysis object"""

        return GapAnalysis(
            suggested_parameter=data['suggested_parameter'],
            affected_features=data['affected_features'],
            avg_error=data['avg_error'],
            impact_score=data['impact_score'],
            rationale=data['rationale'],
            confidence=data['confidence'],
            priority=data['priority'],
            parameter_info=data.get('parameter_info', {}),
            gap_id=data.get('gap_id'),
            sample_midi_files=data.get('sample_midi_files', []),
            feature_statistics=data.get('feature_statistics', {})
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""

        metrics = self.metrics.copy()
        metrics['history_stats'] = self.history.get_statistics()

        if metrics['total_proposals'] > 0:
            metrics['success_rate'] = metrics['successful_proposals'] / metrics['total_proposals']
        else:
            metrics['success_rate'] = 0.0

        return metrics

    def export_proposal(self, proposal: ParameterProposal, output_path: Path):
        """Export proposal to JSON file"""

        with open(output_path, 'w') as f:
            json.dump(proposal.to_dict(), f, indent=2)

        logger.info(f"Exported proposal to {output_path}")

    def integrate_proposal(self, proposal: ParameterProposal) -> ParameterDefinition:
        """
        Convert validated proposal to ParameterDefinition and register it

        Args:
            proposal: Validated ParameterProposal

        Returns:
            ParameterDefinition registered in system

        Raises:
            ValueError: If proposal is not validated
        """

        if proposal.status != ProposalStatus.VALIDATED:
            raise ValueError(
                f"Cannot integrate proposal with status {proposal.status}. "
                f"Must be VALIDATED."
            )

        # Convert to ParameterDefinition
        param_def = proposal.to_parameter_definition()

        # Register
        self.registry.register(param_def)

        # Update proposal status
        proposal.status = ProposalStatus.IMPLEMENTED
        self.history.update_status(proposal.proposal_id, ProposalStatus.IMPLEMENTED)

        logger.info(f"✅ Integrated parameter: {param_def.full_path}")

        return param_def


# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_gap_analysis() -> GapAnalysis:
    """Create a sample gap analysis for testing"""

    return GapAnalysis(
        suggested_parameter='harmony.voicing.quartal_probability',
        affected_features=[
            'quartal_voicing_count',
            'fourth_interval_ratio',
            'open_voicing_ratio'
        ],
        avg_error=0.75,
        impact_score=0.88,
        rationale='Input MIDI has extensive quartal harmony in McCoy Tyner style, system cannot reproduce',
        confidence=0.92,
        priority='HIGH',
        parameter_info={
            'type': 'PROBABILITY',
            'musical_rationale': (
                'Quartal voicings use stacked fourths instead of thirds, '
                'common in modal jazz (McCoy Tyner, Herbie Hancock), '
                'modern classical (Hindemith, Bartók), and impressionist music'
            ),
            'typical_usage': {
                'modal_jazz': 0.7,
                'bebop': 0.1,
                'impressionist': 0.6,
                'swing': 0.0
            }
        },
        gap_id='GAP_20250120_001',
        sample_midi_files=['tyner_passion_dance.mid', 'hancock_maiden_voyage.mid']
    )


# ============================================================================
# Main - Testing and Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LLM PARAMETER PROPOSAL AGENT - Agent 11")
    print("=" * 80)

    # Create agent
    agent = LLMParameterProposalAgent()

    # Create sample gap analysis
    gap = create_sample_gap_analysis()

    print("\n📋 Gap Analysis:")
    print(f"   Parameter: {gap.suggested_parameter}")
    print(f"   Error: {gap.avg_error:.2f}")
    print(f"   Impact: {gap.impact_score:.2f}")
    print(f"   Confidence: {gap.confidence:.2f}")
    print(f"   Priority: {gap.priority}")

    # Check if API key is available
    if os.getenv('ANTHROPIC_API_KEY'):
        print("\n🤖 Calling Claude API to generate proposal...")

        try:
            # Generate proposal
            proposal = agent.propose_parameter(gap)

            print(f"\n{'='*80}")
            print("PROPOSAL RESULTS")
            print(f"{'='*80}")

            print(f"\n✨ Proposed Parameter: {proposal.name}")
            print(f"   Type: {proposal.param_type}")
            print(f"   Range: {proposal.range}")
            print(f"   Default: {proposal.default}")
            print(f"\n   Description: {proposal.description}")
            print(f"\n   Musical Context: {proposal.musical_context[:200]}...")
            print(f"\n   Implementation: {proposal.implementation_strategy[:200]}...")

            print(f"\n📊 Validation:")
            print(f"   Status: {proposal.status.value}")

            if proposal.validation_errors:
                print(f"\n   ❌ Errors:")
                for err in proposal.validation_errors:
                    print(f"      - {err}")

            if proposal.validation_warnings:
                print(f"\n   ⚠️  Warnings:")
                for warn in proposal.validation_warnings:
                    print(f"      - {warn}")

            if proposal.status == ProposalStatus.VALIDATED:
                print(f"\n   ✅ Proposal is valid and ready for integration!")

            print(f"\n🧪 Test Cases:")
            for i, tc in enumerate(proposal.test_cases, 1):
                print(f"   {i}. Value: {tc.value}")
                print(f"      Expected: {tc.expected_description}")

            print(f"\n🎵 Example Values:")
            for genre, value in proposal.example_values.items():
                print(f"   {genre}: {value}")

            # Export proposal
            export_path = Path(__file__).parent / f"proposal_{proposal.proposal_id}.json"
            agent.export_proposal(proposal, export_path)
            print(f"\n💾 Exported to: {export_path}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\n⚠️  No ANTHROPIC_API_KEY found. Skipping API call.")
        print("   Set environment variable to test Claude integration.")

    # Show metrics
    metrics = agent.get_metrics()
    print(f"\n📈 Agent Metrics:")
    print(f"   Total proposals: {metrics['total_proposals']}")
    print(f"   Successful: {metrics['successful_proposals']}")
    print(f"   Failed: {metrics['failed_proposals']}")
    print(f"   Success rate: {metrics['success_rate']:.1%}")
    print(f"   API calls: {metrics['api_calls']}")
    print(f"   API errors: {metrics['api_errors']}")

    # Show history stats
    history_stats = metrics['history_stats']
    print(f"\n📚 Proposal History:")
    print(f"   Total: {history_stats['total_proposals']}")
    print(f"   Acceptance rate: {history_stats['acceptance_rate']:.1%}")
    print(f"   Pending: {history_stats['pending_count']}")

    print("\n" + "=" * 80)
    print("✅ Agent 11 - LLM Parameter Proposal Agent Ready!")
    print("=" * 80)
