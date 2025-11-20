"""
AGENT 13: Musical Validator
============================

Validates proposed parameters and generated code for musical correctness.

This agent ensures that:
1. Parameters represent real musical concepts
2. Parameter names follow conventions
3. No duplicate or conflicting parameters
4. Ranges are musically appropriate
5. Implementation is viable
6. Test coverage is adequate
7. Music theory is consistent
8. Generated code is clean and safe

Author: Agent 13 - Musical Validation Specialist
License: MIT
"""

import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import os

try:
    import anthropic
except ImportError:
    anthropic = None


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    ERROR = "error"  # Must fix - prevents deployment
    WARNING = "warning"  # Should fix - may cause issues
    INFO = "info"  # FYI - no action needed


@dataclass
class ValidationCheck:
    """Result of a single validation check"""
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[dict] = None
    score: float = 1.0  # 0.0-1.0, where 1.0 is perfect


@dataclass
class ParameterValidationResult:
    """Complete validation result for a parameter proposal"""
    parameter_name: str
    valid: bool
    overall_score: float  # 0.0-1.0
    checks: Dict[str, ValidationCheck]
    warnings: List[str]
    errors: List[str]
    suggestions: List[str]
    llm_validations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeValidationResult:
    """Validation result for generated code"""
    valid: bool
    checks: Dict[str, ValidationCheck]
    syntax_errors: List[str] = field(default_factory=list)
    pattern_violations: List[str] = field(default_factory=list)
    compatibility_issues: List[str] = field(default_factory=list)


class MusicTheoryRules:
    """Music theory rules and constraints"""

    # Contradictory musical concepts
    CONTRADICTIONS = {
        'quartal': ['tertian', 'triad'],
        'atonal': ['functional', 'tonal', 'key'],
        'straight': ['swing', 'shuffle'],
        'legato': ['staccato'],
        'monophonic': ['polyphonic', 'chord', 'harmony'],
        'rubato': ['strict_tempo'],
        'chromatic': ['diatonic'],
        'lydian': ['locrian'],  # Modes with contradictory characteristics
    }

    # Compatible musical concepts (often used together)
    COMPATIBLE = {
        'swing': ['bebop', 'blues', 'jazz'],
        'modal': ['lydian', 'dorian', 'mixolydian', 'phrygian'],
        'bebop': ['chromatic', 'altered', 'diminished'],
        'blues': ['pentatonic', 'shuffle'],
        'bossa': ['brazilian', 'latin', 'syncopation'],
    }

    # Valid parameter ranges by type
    TYPICAL_RANGES = {
        'probability': (0.0, 1.0),
        'density': (0.0, 1.0),
        'intensity': (0.0, 1.0),
        'complexity': (0.0, 1.0),
        'velocity': (0, 127),
        'midi_note': (0, 127),
        'tempo': (40, 240),
        'swing_amount': (0.0, 1.0),
    }

    # Domains and their sub-modules
    VALID_DOMAINS = {
        'harmony',
        'melody',
        'rhythm',
        'bass',
        'drums',
        'voice',
        'voicing',
        'dynamics',
        'articulation',
        'expression',
        'texture',
        'structure',
        'instrumentation',
        'timbre',
        'form',
        'counterpoint',
        'orchestration'
    }


class MusicalValidator:
    """
    Main validation agent for parameter proposals and generated code

    Performs comprehensive validation including:
    - Naming conventions
    - Musical validity (LLM-powered)
    - Duplicate detection
    - Range appropriateness
    - Implementation viability
    - Test coverage
    - Music theory consistency
    """

    def __init__(self, api_key: Optional[str] = None, mock_mode: bool = False):
        """
        Initialize musical validator

        Args:
            api_key: Anthropic API key for LLM validation
            mock_mode: If True, skip LLM calls and use heuristics only
        """
        self.mock_mode = mock_mode or anthropic is None

        if not self.mock_mode:
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.llm = anthropic.Anthropic(api_key=api_key)
            else:
                print("WARNING: No API key found, using mock mode")
                self.mock_mode = True

        self.existing_params = self._load_existing_parameters()
        self.music_theory = MusicTheoryRules()
        self.validation_history: List[ParameterValidationResult] = []

    def _load_existing_parameters(self) -> Set[str]:
        """Load existing parameter names from registry"""
        existing = set()

        try:
            # Load from universal_registry.py
            registry_file = Path('midi_generator/parameters/universal_registry.py')
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    content = f.read()

                # Extract parameter names
                # Look for patterns like "name.module.param"
                pattern = r'"([a-z_]+\.[a-z_]+\.[a-z_]+)"'
                matches = re.findall(pattern, content)
                existing.update(matches)

            # Also load from registry.json if exists
            json_file = Path('midi_generator/parameters/registry.json')
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        existing.update(data.keys())
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'name' in item:
                                existing.add(item['name'])

        except Exception as e:
            print(f"Warning: Could not load existing parameters: {e}")

        return existing

    def validate_parameter(self, proposal: dict) -> ParameterValidationResult:
        """
        Comprehensive validation of parameter proposal

        Args:
            proposal: Parameter proposal dict with keys:
                - name: Full parameter name
                - type: Parameter type
                - range: Valid range
                - default: Default value
                - description: What it does
                - musical_context: When/why used
                - implementation_strategy: How to implement
                - test_cases: Test scenarios

        Returns:
            ParameterValidationResult with all check results
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING PARAMETER: {proposal.get('name', 'UNKNOWN')}")
        print(f"{'='*80}\n")

        checks = {}
        warnings = []
        errors = []
        suggestions = []

        # Run all validation checks
        print("Running validation checks...")

        # 1. Naming Convention
        print("  [1/7] Checking naming convention...")
        naming_check = self._check_naming_convention(proposal)
        checks['naming_convention'] = naming_check
        self._process_check_result(naming_check, errors, warnings)

        # 2. Musical Validity (LLM-powered)
        print("  [2/7] Checking musical validity...")
        if self.mock_mode:
            musical_check = self._check_musical_validity_heuristic(proposal)
        else:
            musical_check = self._check_musical_validity_llm(proposal)
        checks['musical_validity'] = musical_check
        self._process_check_result(musical_check, errors, warnings)

        # 3. Duplicate Detection
        print("  [3/7] Checking for duplicates...")
        duplicate_check = self._check_duplicates(proposal)
        checks['no_duplicates'] = duplicate_check
        self._process_check_result(duplicate_check, errors, warnings)

        # 4. Range Appropriateness
        print("  [4/7] Checking range appropriateness...")
        range_check = self._check_range_appropriate(proposal)
        checks['range_appropriate'] = range_check
        self._process_check_result(range_check, errors, warnings)

        # 5. Implementation Viability
        print("  [5/7] Checking implementation viability...")
        impl_check = self._check_implementation_viable(proposal)
        checks['implementation_viable'] = impl_check
        self._process_check_result(impl_check, errors, warnings)

        # 6. Test Coverage
        print("  [6/7] Checking test coverage...")
        test_check = self._check_test_coverage(proposal)
        checks['test_coverage'] = test_check
        self._process_check_result(test_check, errors, warnings)

        # 7. Music Theory Consistency
        print("  [7/7] Checking music theory consistency...")
        theory_check = self._check_theory_consistency(proposal)
        checks['theory_consistency'] = theory_check
        self._process_check_result(theory_check, errors, warnings)

        # Calculate overall score
        passed_checks = sum(1 for check in checks.values() if check.passed)
        overall_score = passed_checks / len(checks) if checks else 0.0

        # Weight by check scores
        weighted_score = sum(check.score for check in checks.values()) / len(checks) if checks else 0.0

        # Determine validity (no errors = valid)
        valid = len(errors) == 0

        # Generate suggestions
        suggestions = self._generate_suggestions(proposal, checks)

        # Create result
        result = ParameterValidationResult(
            parameter_name=proposal.get('name', 'UNKNOWN'),
            valid=valid,
            overall_score=weighted_score,
            checks=checks,
            warnings=warnings,
            errors=errors,
            suggestions=suggestions
        )

        # Record in history
        self.validation_history.append(result)

        # Print summary
        self._print_validation_summary(result)

        return result

    def _process_check_result(self, check: ValidationCheck,
                             errors: List[str], warnings: List[str]):
        """Process check result and add to errors/warnings lists"""
        if not check.passed:
            if check.severity == ValidationSeverity.ERROR:
                errors.append(check.message)
            elif check.severity == ValidationSeverity.WARNING:
                warnings.append(check.message)

    def _check_naming_convention(self, proposal: dict) -> ValidationCheck:
        """Validate parameter naming follows convention"""

        name = proposal.get('name', '')

        # Must be domain.module.parameter
        pattern = r'^[a-z_]+\.[a-z_]+\.[a-z_]+$'

        if not re.match(pattern, name):
            return ValidationCheck(
                check_name='naming_convention',
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Parameter name '{name}' does not follow convention: domain.module.parameter",
                score=0.0
            )

        # Check domain is valid
        parts = name.split('.')
        domain = parts[0]

        if domain not in self.music_theory.VALID_DOMAINS:
            return ValidationCheck(
                check_name='naming_convention',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Domain '{domain}' not in standard domains. Valid: {self.music_theory.VALID_DOMAINS}",
                score=0.5
            )

        # Check for reasonable length
        if len(name) > 60:
            return ValidationCheck(
                check_name='naming_convention',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Parameter name too long (>{60} chars): {name}",
                score=0.7
            )

        # Check for descriptive naming
        if any(len(part) < 2 for part in parts):
            return ValidationCheck(
                check_name='naming_convention',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Parameter name parts should be descriptive (>=2 chars): {name}",
                score=0.8
            )

        return ValidationCheck(
            check_name='naming_convention',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='Naming convention OK',
            score=1.0
        )

    def _check_musical_validity_llm(self, proposal: dict) -> ValidationCheck:
        """Use LLM to validate musical sense of parameter"""

        prompt = f"""
Analyze this proposed music generation parameter for musical validity:

Parameter: {proposal.get('name', '')}
Type: {proposal.get('type', '')}
Range: {proposal.get('range', '')}
Description: {proposal.get('description', '')}
Musical Context: {proposal.get('musical_context', '')}
Implementation: {proposal.get('implementation_strategy', '')}

Evaluate:
1. Does this parameter represent a real musical concept?
2. Is it clearly defined and unambiguous?
3. Would it produce audible, meaningful differences in generated music?
4. Is it used by real composers/musicians?
5. Does the implementation strategy make musical sense?
6. Are there any music theory contradictions?

Respond ONLY with JSON:
{{
  "valid": true/false,
  "score": 0.0-1.0,
  "issues": ["list of issues if any"],
  "rationale": "Brief explanation"
}}
"""

        try:
            response = self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            result = json.loads(response.content[0].text)

            if not result.get('valid', False):
                return ValidationCheck(
                    check_name='musical_validity',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Musical validity issues: {', '.join(result.get('issues', []))}",
                    details={'llm_response': result},
                    score=result.get('score', 0.0)
                )

            if result.get('score', 0.0) < 0.7:
                return ValidationCheck(
                    check_name='musical_validity',
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Musical validity score low ({result.get('score', 0):.2f}): {result.get('rationale', '')}",
                    details={'llm_response': result},
                    score=result.get('score', 0.0)
                )

            return ValidationCheck(
                check_name='musical_validity',
                passed=True,
                severity=ValidationSeverity.INFO,
                message=f"Musical validity confirmed (score: {result.get('score', 0):.2f})",
                details={'llm_response': result},
                score=result.get('score', 1.0)
            )

        except Exception as e:
            print(f"  WARNING: LLM validation failed: {e}, using heuristic")
            return self._check_musical_validity_heuristic(proposal)

    def _check_musical_validity_heuristic(self, proposal: dict) -> ValidationCheck:
        """Heuristic-based musical validity check (fallback)"""

        name = proposal.get('name', '')
        description = proposal.get('description', '')
        musical_context = proposal.get('musical_context', '')

        # Check if description is detailed enough
        if len(description) < 20:
            return ValidationCheck(
                check_name='musical_validity',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message='Description too brief - should explain musical concept clearly',
                score=0.5
            )

        # Check if musical context is provided
        if len(musical_context) < 20:
            return ValidationCheck(
                check_name='musical_validity',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message='Musical context missing or too brief',
                score=0.6
            )

        # Check for common musical terms
        musical_terms = [
            'chord', 'note', 'scale', 'melody', 'harmony', 'rhythm',
            'tempo', 'beat', 'bass', 'voice', 'voicing', 'jazz',
            'swing', 'interval', 'mode', 'key', 'progression',
            'dynamics', 'articulation', 'phrase', 'motif'
        ]

        text = (name + ' ' + description + ' ' + musical_context).lower()
        term_count = sum(1 for term in musical_terms if term in text)

        if term_count == 0:
            return ValidationCheck(
                check_name='musical_validity',
                passed=False,
                severity=ValidationSeverity.ERROR,
                message='No recognizable musical terms found - may not be a valid musical parameter',
                score=0.0
            )

        # Passed heuristic checks
        return ValidationCheck(
            check_name='musical_validity',
            passed=True,
            severity=ValidationSeverity.INFO,
            message=f'Musical validity OK (heuristic check - {term_count} musical terms found)',
            score=min(1.0, 0.7 + (term_count * 0.05))
        )

    def _check_duplicates(self, proposal: dict) -> ValidationCheck:
        """Check if parameter already exists or is too similar"""

        name = proposal.get('name', '')

        # Exact duplicate
        if name in self.existing_params:
            return ValidationCheck(
                check_name='no_duplicates',
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Parameter already exists: {name}",
                score=0.0
            )

        # Similar names (might be duplicate concept)
        similar = []
        name_parts = set(name.lower().replace('_', ' ').replace('.', ' ').split())

        for existing_name in self.existing_params:
            existing_parts = set(existing_name.lower().replace('_', ' ').replace('.', ' ').split())

            # Check for high overlap
            overlap = len(name_parts & existing_parts)
            if overlap >= 2:  # At least 2 words in common
                similar.append(existing_name)

        if similar:
            return ValidationCheck(
                check_name='no_duplicates',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message=f"Similar parameters exist: {', '.join(similar[:3])}. Verify this is not a duplicate.",
                details={'similar_parameters': similar},
                score=0.6
            )

        return ValidationCheck(
            check_name='no_duplicates',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='No duplicates found',
            score=1.0
        )

    def _check_range_appropriate(self, proposal: dict) -> ValidationCheck:
        """Validate parameter range makes musical sense"""

        param_type = proposal.get('type', 'CONTINUOUS')
        param_range = proposal.get('range', (0.0, 1.0))
        default = proposal.get('default', 0.5)
        name = proposal.get('name', '')

        if param_type == 'CONTINUOUS' or param_type == 'PROBABILITY':
            # Check range is [min, max] tuple
            if not (isinstance(param_range, (list, tuple)) and len(param_range) == 2):
                return ValidationCheck(
                    check_name='range_appropriate',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"CONTINUOUS parameter must have [min, max] range, got: {param_range}",
                    score=0.0
                )

            min_val, max_val = param_range

            # Min < Max
            if min_val >= max_val:
                return ValidationCheck(
                    check_name='range_appropriate',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid range: min ({min_val}) must be < max ({max_val})",
                    score=0.0
                )

            # Default in range
            if not (min_val <= default <= max_val):
                return ValidationCheck(
                    check_name='range_appropriate',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Default ({default}) outside range [{min_val}, {max_val}]",
                    score=0.0
                )

            # Check for appropriate ranges based on name
            if 'prob' in name or 'probability' in name:
                if not (min_val == 0.0 and max_val == 1.0):
                    return ValidationCheck(
                        check_name='range_appropriate',
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Probability parameters should use [0.0, 1.0] range, got [{min_val}, {max_val}]",
                        score=0.7
                    )

            if 'density' in name or 'intensity' in name:
                if not (min_val >= 0.0 and max_val <= 1.0):
                    return ValidationCheck(
                        check_name='range_appropriate',
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Density/intensity typically use [0.0, 1.0] range",
                        score=0.7
                    )

            if 'velocity' in name:
                if not (0 <= min_val and max_val <= 127):
                    return ValidationCheck(
                        check_name='range_appropriate',
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Velocity should be in MIDI range [0, 127]",
                        score=0.6
                    )

        elif param_type == 'CATEGORICAL':
            # Check has at least 2 options
            if not (isinstance(param_range, list) and len(param_range) >= 2):
                return ValidationCheck(
                    check_name='range_appropriate',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"CATEGORICAL parameter must have at least 2 options, got: {param_range}",
                    score=0.0
                )

            # Default in options
            if default not in param_range:
                return ValidationCheck(
                    check_name='range_appropriate',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Default '{default}' not in categorical options: {param_range}",
                    score=0.0
                )

        elif param_type == 'BOOLEAN':
            # Default must be bool
            if not isinstance(default, bool):
                return ValidationCheck(
                    check_name='range_appropriate',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"BOOLEAN parameter must have boolean default, got: {type(default)}",
                    score=0.0
                )

        return ValidationCheck(
            check_name='range_appropriate',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='Range appropriate',
            score=1.0
        )

    def _check_implementation_viable(self, proposal: dict) -> ValidationCheck:
        """Check if implementation strategy is clear and viable"""

        impl_strategy = proposal.get('implementation_strategy', '')

        # Must have implementation strategy
        if not impl_strategy or len(impl_strategy) < 30:
            return ValidationCheck(
                check_name='implementation_viable',
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Implementation strategy missing or too brief (need detailed explanation)",
                score=0.0
            )

        # Should mention how it will be used
        key_terms = [
            'generator', 'check', 'if', 'probability', 'create', 'build',
            'calculate', 'algorithm', 'when', 'apply', 'use', 'modify'
        ]
        if not any(term in impl_strategy.lower() for term in key_terms):
            return ValidationCheck(
                check_name='implementation_viable',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Implementation strategy should describe HOW generator will use parameter",
                score=0.5
            )

        # Should not be vague
        vague_phrases = ['somehow', 'maybe', 'possibly', 'unclear', 'TBD', 'TODO']
        if any(phrase in impl_strategy.lower() for phrase in vague_phrases):
            return ValidationCheck(
                check_name='implementation_viable',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Implementation strategy contains vague language - be more specific",
                score=0.6
            )

        return ValidationCheck(
            check_name='implementation_viable',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='Implementation strategy viable',
            score=1.0
        )

    def _check_test_coverage(self, proposal: dict) -> ValidationCheck:
        """Validate test cases are comprehensive"""

        test_cases = proposal.get('test_cases', [])

        # Must have test cases
        if not test_cases or len(test_cases) < 2:
            return ValidationCheck(
                check_name='test_coverage',
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Need at least 2 test cases (min and max values)",
                score=0.4
            )

        param_type = proposal.get('type', 'CONTINUOUS')

        if param_type == 'CONTINUOUS' or param_type == 'PROBABILITY':
            # Should test extremes
            values = [tc.get('value') for tc in test_cases if 'value' in tc]
            param_range = proposal.get('range', (0.0, 1.0))

            if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                min_val, max_val = param_range

                if min_val not in values:
                    return ValidationCheck(
                        check_name='test_coverage',
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Should test minimum value ({min_val})",
                        score=0.6
                    )

                if max_val not in values:
                    return ValidationCheck(
                        check_name='test_coverage',
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Should test maximum value ({max_val})",
                        score=0.6
                    )

        elif param_type == 'CATEGORICAL':
            # Should test all options
            values = [tc.get('value') for tc in test_cases if 'value' in tc]
            options = proposal.get('range', [])

            if isinstance(options, list):
                untested = set(options) - set(values)
                if untested:
                    return ValidationCheck(
                        check_name='test_coverage',
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Untested categorical options: {untested}",
                        score=0.7
                    )

        # Test cases should have expected results
        for tc in test_cases:
            if 'expected' not in tc or not tc['expected']:
                return ValidationCheck(
                    check_name='test_coverage',
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message="All test cases should have 'expected' field describing outcome",
                    score=0.7
                )

        return ValidationCheck(
            check_name='test_coverage',
            passed=True,
            severity=ValidationSeverity.INFO,
            message=f'Test coverage adequate ({len(test_cases)} cases)',
            score=1.0
        )

    def _check_theory_consistency(self, proposal: dict) -> ValidationCheck:
        """Check for music theory contradictions"""

        name = proposal.get('name', '')
        description = proposal.get('description', '').lower()
        musical_context = proposal.get('musical_context', '').lower()

        # Check for known contradictions
        for term, conflicting_terms in self.music_theory.CONTRADICTIONS.items():
            if term in name.lower() or term in description:
                for conflict in conflicting_terms:
                    if conflict in description or conflict in musical_context:
                        return ValidationCheck(
                            check_name='theory_consistency',
                            passed=False,
                            severity=ValidationSeverity.WARNING,
                            message=f"Potential contradiction: '{term}' and '{conflict}' are typically opposed concepts",
                            score=0.5
                        )

        # Domain-specific checks
        domain = name.split('.')[0] if '.' in name else ''

        if domain == 'harmony':
            # Voicing checks
            if 'voicing' in name:
                if 'omit' in name and 'double' in name:
                    return ValidationCheck(
                        check_name='theory_consistency',
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message="Cannot simultaneously omit and double a chord tone",
                        score=0.0
                    )

        elif domain == 'rhythm':
            if 'polyrhythm' in name:
                # Should have ratio specification
                if 'ratio' not in name and 'ratio' not in str(proposal.get('range', [])):
                    return ValidationCheck(
                        check_name='theory_consistency',
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message="Polyrhythm parameters should specify ratio (e.g., 3:2, 4:3)",
                        score=0.7
                    )

        return ValidationCheck(
            check_name='theory_consistency',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='Music theory consistent',
            score=1.0
        )

    def _generate_suggestions(self, proposal: dict, checks: Dict[str, ValidationCheck]) -> List[str]:
        """Generate improvement suggestions"""

        suggestions = []

        # Suggest related parameters
        domain = proposal.get('name', '').split('.')[0] if '.' in proposal.get('name', '') else ''
        related_count = sum(1 for p in self.existing_params if p.startswith(domain + '.'))

        if related_count > 5:
            suggestions.append(
                f"Consider how this parameter interacts with {related_count} existing {domain} parameters"
            )

        # Suggest example values
        example_values = proposal.get('example_values', {})
        if not example_values or len(example_values) < 3:
            suggestions.append("Add more genre-specific example values (aim for 5+ genres)")

        # Suggest feature correlation
        affected_features = proposal.get('affected_features', [])
        if isinstance(affected_features, list) and len(affected_features) <= 1:
            suggestions.append("Consider whether this parameter affects other features beyond those listed")

        # Suggest integration points
        if 'generator_integration_points' not in proposal or not proposal.get('generator_integration_points'):
            suggestions.append("Specify exact generator integration points (file::class::method)")

        return suggestions

    def validate_code(self, code: dict, proposal: dict) -> CodeValidationResult:
        """
        Validate generated code

        Args:
            code: Generated code dict with sections
            proposal: Original parameter proposal

        Returns:
            CodeValidationResult
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING GENERATED CODE")
        print(f"{'='*80}\n")

        checks = {}
        syntax_errors = []
        pattern_violations = []
        compatibility_issues = []

        # 1. Syntax validity
        print("  [1/4] Checking syntax...")
        syntax_check = self._check_code_syntax(code, syntax_errors)
        checks['syntax_valid'] = syntax_check

        # 2. Integration cleanliness
        print("  [2/4] Checking integration...")
        integration_check = self._check_clean_integration(code, proposal, pattern_violations)
        checks['integrates_cleanly'] = integration_check

        # 3. Edge case handling
        print("  [3/4] Checking edge cases...")
        edge_case_check = self._check_edge_cases(code, proposal, pattern_violations)
        checks['handles_edge_cases'] = edge_case_check

        # 4. Backward compatibility
        print("  [4/4] Checking backward compatibility...")
        compat_check = self._check_backward_compatible(code, compatibility_issues)
        checks['backward_compatible'] = compat_check

        all_passed = all(check.passed for check in checks.values())

        result = CodeValidationResult(
            valid=all_passed,
            checks=checks,
            syntax_errors=syntax_errors,
            pattern_violations=pattern_violations,
            compatibility_issues=compatibility_issues
        )

        self._print_code_validation_summary(result)

        return result

    def _check_code_syntax(self, code: dict, errors: List[str]) -> ValidationCheck:
        """Check Python syntax validity"""

        for file_path, file_code in code.get('generator_modifications', {}).items():
            if not file_code or not file_code.strip():
                continue

            try:
                compile(file_code, file_path, 'exec')
            except SyntaxError as e:
                error_msg = f"Syntax error in {file_path}: {e}"
                errors.append(error_msg)
                return ValidationCheck(
                    check_name='syntax_valid',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=error_msg,
                    score=0.0
                )

        # Check test code
        test_code = code.get('test_code', '')
        if test_code and test_code.strip():
            try:
                compile(test_code, 'test_file.py', 'exec')
            except SyntaxError as e:
                error_msg = f"Syntax error in test code: {e}"
                errors.append(error_msg)
                return ValidationCheck(
                    check_name='syntax_valid',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=error_msg,
                    score=0.0
                )

        return ValidationCheck(
            check_name='syntax_valid',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='Syntax valid',
            score=1.0
        )

    def _check_clean_integration(self, code: dict, proposal: dict, violations: List[str]) -> ValidationCheck:
        """Check code integrates cleanly"""

        param_name = proposal.get('name', '')

        for file_path, file_code in code.get('generator_modifications', {}).items():
            if not file_code:
                continue

            # Must use .get() for parameter access
            if param_name in file_code:
                if f".get('{param_name}'" not in file_code and f'.get("{param_name}"' not in file_code:
                    violation = f"Must use params.get('{param_name}', default) for backward compatibility"
                    violations.append(violation)
                    return ValidationCheck(
                        check_name='integrates_cleanly',
                        passed=False,
                        severity=ValidationSeverity.ERROR,
                        message=violation,
                        score=0.0
                    )

            # Should have docstring for new methods
            if 'def ' in file_code and ('"""' not in file_code and "'''" not in file_code):
                violation = f"New/modified methods in {file_path} should have docstrings"
                violations.append(violation)
                return ValidationCheck(
                    check_name='integrates_cleanly',
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=violation,
                    score=0.7
                )

        return ValidationCheck(
            check_name='integrates_cleanly',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='Integration clean',
            score=1.0
        )

    def _check_edge_cases(self, code: dict, proposal: dict, violations: List[str]) -> ValidationCheck:
        """Check edge case handling"""

        param_type = proposal.get('type', 'CONTINUOUS')

        for file_code in code.get('generator_modifications', {}).values():
            if not file_code:
                continue

            if param_type == 'CONTINUOUS' or param_type == 'PROBABILITY':
                # Should validate/clamp values
                has_validation = any(pattern in file_code for pattern in [
                    'max(', 'min(', 'clip', 'clamp', '< 0', '> 1', '<= 0', '>= 1'
                ])

                if not has_validation:
                    violation = "CONTINUOUS parameters should validate range (use max/min or np.clip)"
                    violations.append(violation)
                    return ValidationCheck(
                        check_name='handles_edge_cases',
                        passed=False,
                        severity=ValidationSeverity.WARNING,
                        message=violation,
                        score=0.6
                    )

        return ValidationCheck(
            check_name='handles_edge_cases',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='Edge cases handled',
            score=1.0
        )

    def _check_backward_compatible(self, code: dict, issues: List[str]) -> ValidationCheck:
        """Ensure backward compatibility"""

        for file_code in code.get('generator_modifications', {}).values():
            if not file_code:
                continue

            # Must use .get() not direct access
            if 'params[' in file_code and 'params.get(' not in file_code:
                issue = "Use params.get() not params[] for backward compatibility"
                issues.append(issue)
                return ValidationCheck(
                    check_name='backward_compatible',
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=issue,
                    score=0.0
                )

        return ValidationCheck(
            check_name='backward_compatible',
            passed=True,
            severity=ValidationSeverity.INFO,
            message='Backward compatible',
            score=1.0
        )

    def _print_validation_summary(self, result: ParameterValidationResult):
        """Print formatted validation summary"""

        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY: {result.parameter_name}")
        print(f"{'='*80}")
        print(f"\nOverall: {'✅ VALID' if result.valid else '❌ INVALID'}")
        print(f"Score: {result.overall_score:.2f}/1.00")

        print(f"\nChecks ({len(result.checks)}):")
        for check_name, check in result.checks.items():
            status = '✅' if check.passed else ('⚠️' if check.severity == ValidationSeverity.WARNING else '❌')
            print(f"  {status} {check_name}: {check.message} (score: {check.score:.2f})")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  ❌ {error}")

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  ⚠️ {warning}")

        if result.suggestions:
            print(f"\nSuggestions ({len(result.suggestions)}):")
            for suggestion in result.suggestions:
                print(f"  💡 {suggestion}")

        print(f"\n{'='*80}\n")

    def _print_code_validation_summary(self, result: CodeValidationResult):
        """Print code validation summary"""

        print(f"\n{'='*80}")
        print(f"CODE VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"\nOverall: {'✅ VALID' if result.valid else '❌ INVALID'}")

        print(f"\nChecks ({len(result.checks)}):")
        for check_name, check in result.checks.items():
            status = '✅' if check.passed else ('⚠️' if check.severity == ValidationSeverity.WARNING else '❌')
            print(f"  {status} {check_name}: {check.message}")

        if result.syntax_errors:
            print(f"\nSyntax Errors ({len(result.syntax_errors)}):")
            for error in result.syntax_errors:
                print(f"  ❌ {error}")

        if result.pattern_violations:
            print(f"\nPattern Violations ({len(result.pattern_violations)}):")
            for violation in result.pattern_violations:
                print(f"  ⚠️ {violation}")

        if result.compatibility_issues:
            print(f"\nCompatibility Issues ({len(result.compatibility_issues)}):")
            for issue in result.compatibility_issues:
                print(f"  ❌ {issue}")

        print(f"\n{'='*80}\n")


# Example usage
if __name__ == '__main__':
    # Example parameter proposal
    example_proposal = {
        'name': 'harmony.voicing.quartal_probability',
        'type': 'CONTINUOUS',
        'range': (0.0, 1.0),
        'default': 0.3,
        'description': 'Probability of using quartal (fourth-based) voicings instead of tertian (third-based)',
        'musical_context': 'Quartal harmony is common in modal jazz and modern jazz. Higher values create more open, modern sounds.',
        'implementation_strategy': 'In voicing generation, check this probability before creating chord voicings. If triggered, build voicings in fourths instead of thirds.',
        'test_cases': [
            {'value': 0.0, 'expected': 'No quartal voicings'},
            {'value': 0.5, 'expected': 'Mix of quartal and tertian'},
            {'value': 1.0, 'expected': 'All quartal voicings'}
        ],
        'example_values': {
            'modal_jazz': 0.7,
            'bebop': 0.1,
            'fusion': 0.6
        }
    }

    # Create validator in mock mode
    validator = MusicalValidator(mock_mode=True)

    # Validate parameter
    result = validator.validate_parameter(example_proposal)

    # Check result
    if result.valid:
        print("\n✅ Parameter proposal is VALID and ready for implementation!")
    else:
        print(f"\n❌ Parameter proposal has {len(result.errors)} errors that must be fixed")
