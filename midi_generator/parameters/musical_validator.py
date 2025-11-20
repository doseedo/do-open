#!/usr/bin/env python3
"""
AGENT 13: Musical Validator
============================

Comprehensive musical validation system for parameter and code proposals in
the Musical Program Synthesis self-expanding inverse music generation system.

This module ensures all expansions to the Musical Program Synthesis system
are musically sound, theoretically consistent, and correctly implemented.

OVERVIEW
========

The Musical Validator provides a comprehensive 7-stage validation pipeline
for parameter proposals, plus advanced code validation capabilities. It combines
rule-based checks with optional LLM-powered musical validity assessment to ensure
that all new parameters added to the system meet high standards for:

- Naming conventions and clarity
- Musical validity and real-world applicability
- Non-duplication with existing parameters
- Appropriate value ranges and defaults
- Clear implementation strategies
- Comprehensive test coverage
- Music theory consistency

VALIDATION PIPELINE
===================

1. NAMING CONVENTION CHECK
   - Validates hierarchical parameter naming (domain.module.parameter)
   - Checks domain validity against known musical domains
   - Ensures descriptive, meaningful names
   - Validates name length and character usage

2. MUSICAL VALIDITY CHECK (LLM-POWERED)
   - Verifies parameter represents real musical concept
   - Checks for clear, unambiguous definition
   - Validates audible, meaningful impact on music
   - Confirms usage by real composers/musicians
   - Assesses implementation strategy musical sense
   - Falls back to heuristic checks if LLM unavailable

3. DUPLICATE DETECTION
   - Prevents exact duplicate parameters
   - Detects similar existing parameters via name/description analysis
   - Uses multi-metric similarity scoring
   - Warns about potential conceptual overlap

4. RANGE APPROPRIATENESS
   - Validates range format matches parameter type
   - Checks min < max for continuous parameters
   - Verifies default value within range
   - Validates categorical options completeness
   - Checks standard ranges (probability: [0,1], velocity: [0,127], etc.)

5. IMPLEMENTATION VIABILITY
   - Ensures implementation strategy is detailed and clear
   - Validates generator integration points specified
   - Checks for algorithmic clarity
   - Verifies practical implementability

6. TEST COVERAGE
   - Validates comprehensive test cases provided
   - Checks boundary value testing (min/max)
   - Ensures all categorical options tested
   - Validates expected outcome descriptions

7. MUSIC THEORY CONSISTENCY
   - Detects contradictory musical terminology
   - Domain-specific validation (harmony, rhythm, melody)
   - Genre compatibility checking
   - Voice leading rule validation
   - Harmonic function analysis

CODE VALIDATION
===============

- Syntax validation using AST parsing
- Clean integration checking (proper .get() usage)
- Edge case handling verification
- Backward compatibility assurance
- Function length and complexity analysis
- Docstring quality assessment

ADVANCED FEATURES
=================

- Parameter relationship analysis
- Dependency graph generation
- Batch validation with progress tracking
- Multiple report formats (Markdown, HTML, JSON)
- Validation history tracking with trends
- Genre-specific validation rules
- Validation profiles (strict/standard/lenient/research/production)
- Parameter template library
- Example parameter library for learning

USAGE
=====

Basic usage:

    from parameters.musical_validator import create_validator, validate_parameter_proposal

    # Create validator
    validator = create_validator(enable_llm=True)

    # Validate a parameter
    proposal = {
        'name': 'harmony.extensions.ninth_probability',
        'type': 'PROBABILITY',
        'range': [0.0, 1.0],
        'default': 0.7,
        'description': '...',
        # ... more fields
    }

    result = validator.validate_parameter(proposal)

    if result.valid:
        print("✅ Parameter is valid!")
        print(f"Score: {result.score:.0%}")
    else:
        print("❌ Parameter has issues:")
        for error in result.errors:
            print(f"  - {error}")

Advanced usage with profiles:

    from parameters.musical_validator import MusicalValidator, ValidationProfileManager

    # Use strict validation for production
    profile_mgr = ValidationProfileManager()
    profile = profile_mgr.get_profile('production')

    validator = MusicalValidator()
    result = validator.validate_parameter(proposal)

    if result.score < profile.get_min_score():
        print("Failed to meet production standards")

INTEGRATION WITH SYSTEM
========================

The Musical Validator integrates with the broader Musical Program Synthesis
system to ensure quality and consistency as the system self-expands from 165
parameters to 515+ parameters. It serves as a critical quality gate, preventing
the introduction of invalid, redundant, or musically nonsensical parameters.

Key integration points:
- Reads from UniversalParameterRegistry for duplicate detection
- Uses MusicTheoryKnowledgeBase for theory validation
- Provides ValidationHistory for monitoring expansion trends
- Generates reports for human review and approval

ARCHITECTURE
============

The module follows a layered architecture:

1. Core Layer: MusicalValidator, ValidationCheck, ParameterValidationResult
2. Knowledge Layer: MusicTheoryKnowledgeBase with comprehensive music theory
3. Analysis Layer: ParameterRelationshipAnalyzer, GenreSpecificValidator
4. Reporting Layer: ValidationReportGenerator, ValidationHistory
5. Utilities Layer: ParameterTemplateLibrary, ExampleParameterLibrary
6. Profile Layer: ValidationProfile, ValidationProfileManager

DEPENDENCIES
============

- anthropic: Optional, for LLM-powered validation
- parameters.universal_registry: For accessing existing parameters
- Standard library: ast, json, re, logging, typing

EXTENSION POINTS
================

The validator is designed to be extensible:

- Add new validation checks by extending MusicalValidator
- Create custom validation profiles via ValidationProfileManager
- Extend MusicTheoryKnowledgeBase with domain-specific rules
- Add new parameter templates to ParameterTemplateLibrary
- Implement custom report formats in ValidationReportGenerator

Author: Agent 13 - Musical Validator
Date: 2025
Version: 1.0.0
License: MIT
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import ast
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# Try to import Anthropic SDK
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic SDK not available. LLM validation will be disabled.")

# Import from existing modules
from parameters.universal_registry import (
    UniversalParameterRegistry,
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
    MusicalImpact,
    REGISTRY
)


# ============================================================================
# CONFIGURATION
# ============================================================================

VALIDATION_CONFIG = {
    # Naming conventions
    'max_parameter_name_length': 60,
    'min_name_part_length': 3,
    'valid_domains': [
        'harmony', 'melody', 'rhythm', 'dynamics', 'texture',
        'structure', 'instrumentation', 'articulation', 'expression',
        'voicing', 'bass', 'drums', 'genre'
    ],

    # Musical validity
    'min_musical_validity_score': 0.7,
    'llm_model': 'claude-sonnet-4-20250514',
    'llm_temperature': 0.3,

    # Test coverage
    'min_test_cases': 2,

    # Code validation
    'max_line_length': 120,
    'required_docstrings': True,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ValidationCheck:
    """Result of a single validation check"""
    pass_status: bool
    message: str
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pass': self.pass_status,
            'message': self.message,
            'severity': self.severity,
            'details': self.details
        }


@dataclass
class ParameterValidationResult:
    """Complete validation result for a parameter proposal"""
    valid: bool
    score: float  # 0.0-1.0 overall quality score
    checks: Dict[str, ValidationCheck]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'valid': self.valid,
            'score': self.score,
            'checks': {k: v.to_dict() for k, v in self.checks.items()},
            'warnings': self.warnings,
            'errors': self.errors,
            'suggestions': self.suggestions
        }

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("PARAMETER VALIDATION RESULT")
        lines.append("=" * 80)
        lines.append(f"Overall Status: {'✅ VALID' if self.valid else '❌ INVALID'}")
        lines.append(f"Quality Score: {self.score:.2%}")
        lines.append("")

        if self.errors:
            lines.append("🚫 ERRORS:")
            for error in self.errors:
                lines.append(f"  - {error}")
            lines.append("")

        if self.warnings:
            lines.append("⚠️  WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        if self.suggestions:
            lines.append("💡 SUGGESTIONS:")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")
            lines.append("")

        lines.append("VALIDATION CHECKS:")
        for check_name, check_result in self.checks.items():
            status_icon = "✅" if check_result.pass_status else "❌"
            lines.append(f"  {status_icon} {check_name}: {check_result.message}")

        lines.append("=" * 80)
        return "\n".join(lines)


@dataclass
class CodeValidationResult:
    """Result of code validation"""
    valid: bool
    checks: Dict[str, ValidationCheck]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'valid': self.valid,
            'checks': {k: v.to_dict() for k, v in self.checks.items()},
            'errors': self.errors,
            'warnings': self.warnings
        }


# ============================================================================
# MUSIC THEORY KNOWLEDGE BASE
# ============================================================================

class MusicTheoryKnowledgeBase:
    """
    Comprehensive music theory rules and constraints

    This knowledge base contains:
    - Contradictory musical concepts
    - Complementary concepts that work together
    - Standard parameter ranges for different musical elements  - Genre-specific constraints
    - Interval relationships
    - Chord theory rules
    - Voice leading principles
    - Rhythmic patterns and relationships
    """

    def __init__(self):
        # Known contradictory concepts
        self.contradictions = {
            'quartal': ['tertian', 'triad', 'third'],
            'tertian': ['quartal', 'quintal'],
            'quintal': ['tertian', 'quartal'],
            'atonal': ['functional', 'tonal', 'key', 'diatonic'],
            'tonal': ['atonal', 'non-tonal'],
            'chromatic': ['diatonic', 'pure_diatonic'],
            'diatonic': ['chromatic'],
            'straight': ['swing', 'shuffle', 'swung'],
            'swing': ['straight'],
            'shuffle': ['straight'],
            'legato': ['staccato', 'detached', 'marcato'],
            'staccato': ['legato', 'sustained', 'tenuto'],
            'marcato': ['legato'],
            'monophonic': ['polyphonic', 'chord', 'harmony'],
            'polyphonic': ['monophonic'],
            'homophonic': ['polyphonic', 'contrapuntal'],
            'major': ['minor', 'diminished'],
            'minor': ['major', 'augmented'],
            'ascending': ['descending'],
            'descending': ['ascending'],
            'consonant': ['dissonant'],
            'dissonant': ['consonant'],
            'simple': ['compound'],
            'compound': ['simple'],
            'duple': ['triple'],
            'triple': ['duple'],
            'binary': ['ternary'],
            'ternary': ['binary'],
            'forte': ['piano'],
            'piano': ['forte'],
            'crescendo': ['diminuendo', 'decrescendo'],
            'diminuendo': ['crescendo'],
            'accelerando': ['ritardando'],
            'ritardando': ['accelerando'],
        }

        # Extended interval knowledge
        self.interval_properties = {
            'perfect_unison': {'semitones': 0, 'quality': 'perfect', 'consonance': 'perfect'},
            'minor_second': {'semitones': 1, 'quality': 'minor', 'consonance': 'dissonant'},
            'major_second': {'semitones': 2, 'quality': 'major', 'consonance': 'dissonant'},
            'minor_third': {'semitones': 3, 'quality': 'minor', 'consonance': 'imperfect_consonant'},
            'major_third': {'semitones': 4, 'quality': 'major', 'consonance': 'imperfect_consonant'},
            'perfect_fourth': {'semitones': 5, 'quality': 'perfect', 'consonance': 'perfect'},
            'tritone': {'semitones': 6, 'quality': 'augmented', 'consonance': 'dissonant'},
            'perfect_fifth': {'semitones': 7, 'quality': 'perfect', 'consonance': 'perfect'},
            'minor_sixth': {'semitones': 8, 'quality': 'minor', 'consonance': 'imperfect_consonant'},
            'major_sixth': {'semitones': 9, 'quality': 'major', 'consonance': 'imperfect_consonant'},
            'minor_seventh': {'semitones': 10, 'quality': 'minor', 'consonance': 'dissonant'},
            'major_seventh': {'semitones': 11, 'quality': 'major', 'consonance': 'dissonant'},
            'perfect_octave': {'semitones': 12, 'quality': 'perfect', 'consonance': 'perfect'},
        }

        # Chord theory knowledge
        self.chord_types = {
            'major': {'intervals': [0, 4, 7], 'quality': 'consonant'},
            'minor': {'intervals': [0, 3, 7], 'quality': 'consonant'},
            'diminished': {'intervals': [0, 3, 6], 'quality': 'dissonant'},
            'augmented': {'intervals': [0, 4, 8], 'quality': 'dissonant'},
            'major_seventh': {'intervals': [0, 4, 7, 11], 'quality': 'dissonant'},
            'minor_seventh': {'intervals': [0, 3, 7, 10], 'quality': 'dissonant'},
            'dominant_seventh': {'intervals': [0, 4, 7, 10], 'quality': 'dissonant'},
            'half_diminished': {'intervals': [0, 3, 6, 10], 'quality': 'dissonant'},
            'diminished_seventh': {'intervals': [0, 3, 6, 9], 'quality': 'dissonant'},
        }

        # Voice leading rules
        self.voice_leading_rules = {
            'parallel_fifths': {'forbidden': True, 'reason': 'Weakens independence of voices'},
            'parallel_octaves': {'forbidden': True, 'reason': 'Weakens independence of voices'},
            'parallel_unisons': {'forbidden': True, 'reason': 'Reduces texture'},
            'hidden_fifths': {'forbidden': False, 'caution': 'Avoid in outer voices'},
            'hidden_octaves': {'forbidden': False, 'caution': 'Avoid in outer voices'},
            'voice_crossing': {'forbidden': False, 'caution': 'Use sparingly'},
            'voice_overlap': {'forbidden': False, 'caution': 'Generally avoided'},
            'direct_motion_to_perfect': {'forbidden': False, 'caution': 'One voice should move by step'},
        }

        # Scale theory
        self.scale_types = {
            'major': {'pattern': [2, 2, 1, 2, 2, 2, 1], 'modes': 7},
            'natural_minor': {'pattern': [2, 1, 2, 2, 1, 2, 2], 'modes': 7},
            'harmonic_minor': {'pattern': [2, 1, 2, 2, 1, 3, 1], 'modes': 7},
            'melodic_minor': {'pattern': [2, 1, 2, 2, 2, 2, 1], 'modes': 7},
            'pentatonic_major': {'pattern': [2, 2, 3, 2, 3], 'modes': 5},
            'pentatonic_minor': {'pattern': [3, 2, 2, 3, 2], 'modes': 5},
            'blues': {'pattern': [3, 2, 1, 1, 3, 2], 'modes': 1},
            'whole_tone': {'pattern': [2, 2, 2, 2, 2, 2], 'modes': 1},
            'chromatic': {'pattern': [1] * 12, 'modes': 1},
        }

        # Complementary concepts (should often appear together)
        self.complementary = {
            'tritone': ['substitution', 'dominant'],
            'modal': ['interchange', 'mixture'],
            'voice_leading': ['smoothness', 'stepwise', 'contrary'],
            'bebop': ['chromatic', 'enclosure', 'approach'],
            'swing': ['eighth', 'triplet', 'ratio'],
            'syncopation': ['offbeat', 'anticipation'],
            'suspension': ['resolution', 'preparation'],
            'pedal': ['drone', 'sustained'],
            'counterpoint': ['independence', 'imitation', 'canon'],
            'fugue': ['subject', 'answer', 'countersubject'],
            'cadence': ['resolution', 'closure', 'phrase'],
            'modulation': ['key_change', 'transition'],
            'sequence': ['repetition', 'transposition'],
            'ostinato': ['repetition', 'pattern'],
        }

        # Rhythmic patterns
        self.rhythmic_patterns = {
            'common_time': {'numerator': 4, 'denominator': 4, 'feel': 'duple'},
            'waltz': {'numerator': 3, 'denominator': 4, 'feel': 'triple'},
            'march': {'numerator': 2, 'denominator': 4, 'feel': 'duple'},
            'compound_duple': {'numerator': 6, 'denominator': 8, 'feel': 'compound'},
            'compound_triple': {'numerator': 9, 'denominator': 8, 'feel': 'compound'},
            'five_four': {'numerator': 5, 'denominator': 4, 'feel': 'asymmetric'},
            'seven_eight': {'numerator': 7, 'denominator': 8, 'feel': 'asymmetric'},
        }

        # Form structures
        self.form_types = {
            'binary': {'sections': ['A', 'B'], 'common_in': ['baroque', 'dance']},
            'ternary': {'sections': ['A', 'B', 'A'], 'common_in': ['classical', 'da_capo_aria']},
            'rondo': {'sections': ['A', 'B', 'A', 'C', 'A'], 'common_in': ['classical', 'romantic']},
            'sonata': {'sections': ['exposition', 'development', 'recapitulation'], 'common_in': ['classical']},
            'verse_chorus': {'sections': ['verse', 'chorus'], 'common_in': ['pop', 'rock']},
            'twelve_bar_blues': {'sections': ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'I'], 'common_in': ['blues', 'jazz']},
            'aaba': {'sections': ['A', 'A', 'B', 'A'], 'common_in': ['jazz_standards', 'tin_pan_alley']},
        }

        # Articulation types
        self.articulation_types = {
            'legato': {'description': 'Smooth, connected', 'symbol': '—'},
            'staccato': {'description': 'Short, detached', 'symbol': '•'},
            'marcato': {'description': 'Accented, emphasized', 'symbol': '^'},
            'tenuto': {'description': 'Full value, slightly emphasized', 'symbol': '—'},
            'accent': {'description': 'Emphasized', 'symbol': '>'},
            'sforzando': {'description': 'Sudden emphasis', 'symbol': 'sfz'},
            'portato': {'description': 'Semi-detached', 'symbol': '—•'},
        }

        # Dynamic markings
        self.dynamic_markings = {
            'ppp': {'name': 'pianississimo', 'level': 1, 'velocity_range': (10, 20)},
            'pp': {'name': 'pianissimo', 'level': 2, 'velocity_range': (20, 35)},
            'p': {'name': 'piano', 'level': 3, 'velocity_range': (35, 50)},
            'mp': {'name': 'mezzo-piano', 'level': 4, 'velocity_range': (50, 65)},
            'mf': {'name': 'mezzo-forte', 'level': 5, 'velocity_range': (65, 80)},
            'f': {'name': 'forte', 'level': 6, 'velocity_range': (80, 95)},
            'ff': {'name': 'fortissimo', 'level': 7, 'velocity_range': (95, 110)},
            'fff': {'name': 'fortississimo', 'level': 8, 'velocity_range': (110, 127)},
        }

        # Tempo categories
        self.tempo_markings = {
            'larghissimo': {'bpm_range': (0, 24), 'feel': 'extremely_slow'},
            'grave': {'bpm_range': (25, 45), 'feel': 'very_slow'},
            'largo': {'bpm_range': (40, 60), 'feel': 'slow'},
            'lento': {'bpm_range': (45, 60), 'feel': 'slow'},
            'adagio': {'bpm_range': (55, 75), 'feel': 'slow_leisurely'},
            'andante': {'bpm_range': (76, 108), 'feel': 'walking_pace'},
            'moderato': {'bpm_range': (108, 120), 'feel': 'moderate'},
            'allegretto': {'bpm_range': (112, 120), 'feel': 'moderately_fast'},
            'allegro': {'bpm_range': (120, 156), 'feel': 'fast'},
            'vivace': {'bpm_range': (156, 176), 'feel': 'lively_fast'},
            'presto': {'bpm_range': (168, 200), 'feel': 'very_fast'},
            'prestissimo': {'bpm_range': (200, 300), 'feel': 'extremely_fast'},
        }

        # Valid ranges for common parameter types
        self.standard_ranges = {
            'probability': (0.0, 1.0),
            'density': (0.0, 1.0),
            'velocity': (0, 127),
            'midi_note': (0, 127),
            'semitones': (-24, 24),
            'octaves': (0, 8),
            'ratio': (0.0, 2.0),
            'percentage': (0.0, 100.0),
        }

        # Parameter naming patterns
        self.naming_patterns = {
            'probability': r'(prob|probability|chance|likelihood)',
            'density': r'(density|sparseness|thickness)',
            'range': r'(min|max|range|low|high)',
            'velocity': r'(velocity|dynamics|volume|loudness)',
            'duration': r'(duration|length|time)',
            'interval': r'(interval|distance|step|leap)',
        }

        # Genre-specific constraints
        self.genre_constraints = {
            'jazz': {
                'required_features': ['swing', 'syncopation', 'extensions'],
                'typical_ranges': {
                    'swing_ratio': (0.55, 0.75),
                    'chord_extensions': (7, 13),
                }
            },
            'classical': {
                'required_features': ['voice_leading', 'counterpoint'],
                'forbidden_features': ['power_chord', 'distortion'],
            },
            'rock': {
                'typical_ranges': {
                    'distortion': (0.0, 1.0),
                }
            },
            'blues': {
                'required_features': ['blue_notes', 'pentatonic'],
            }
        }

    def get_contradictions(self, term: str) -> List[str]:
        """Get list of contradictory terms"""
        term_lower = term.lower()
        for key, conflicts in self.contradictions.items():
            if key in term_lower:
                return conflicts
        return []

    def is_contradiction(self, term1: str, term2: str) -> bool:
        """Check if two terms are contradictory"""
        term1_lower = term1.lower()
        term2_lower = term2.lower()

        for key, conflicts in self.contradictions.items():
            if key in term1_lower and any(c in term2_lower for c in conflicts):
                return True
            if key in term2_lower and any(c in term1_lower for c in conflicts):
                return True

        return False

    def get_standard_range(self, param_name: str) -> Optional[Tuple[float, float]]:
        """Get standard range for parameter based on name"""
        param_lower = param_name.lower()

        for range_type, (min_val, max_val) in self.standard_ranges.items():
            if range_type in param_lower:
                return (min_val, max_val)

        return None

    def get_interval_name(self, semitones: int) -> Optional[str]:
        """Get interval name from semitone distance"""
        semitones = semitones % 12  # Normalize to octave
        for name, props in self.interval_properties.items():
            if props['semitones'] == semitones:
                return name
        return None

    def is_consonant_interval(self, semitones: int) -> bool:
        """Check if interval is consonant"""
        interval_name = self.get_interval_name(semitones)
        if not interval_name:
            return False

        props = self.interval_properties.get(interval_name, {})
        consonance = props.get('consonance', 'dissonant')
        return consonance in ['perfect', 'imperfect_consonant']

    def get_chord_quality(self, intervals: List[int]) -> Optional[str]:
        """Get chord quality from interval pattern"""
        for chord_type, props in self.chord_types.items():
            if props['intervals'] == intervals[:len(props['intervals'])]:
                return chord_type
        return None

    def is_valid_voice_leading(self, motion_type: str) -> bool:
        """Check if voice leading motion is allowed"""
        rule = self.voice_leading_rules.get(motion_type)
        if not rule:
            return True  # Unknown rule, assume valid
        return not rule.get('forbidden', False)

    def get_scale_intervals(self, scale_type: str) -> Optional[List[int]]:
        """Get interval pattern for scale type"""
        scale = self.scale_types.get(scale_type)
        return scale['pattern'] if scale else None

    def classify_tempo(self, bpm: float) -> Optional[str]:
        """Classify BPM into tempo marking"""
        for marking, props in self.tempo_markings.items():
            min_bpm, max_bpm = props['bpm_range']
            if min_bpm <= bpm <= max_bpm:
                return marking
        return None

    def get_dynamic_range(self, marking: str) -> Optional[Tuple[int, int]]:
        """Get MIDI velocity range for dynamic marking"""
        dynamic = self.dynamic_markings.get(marking)
        return dynamic['velocity_range'] if dynamic else None

    def validate_meter(self, numerator: int, denominator: int) -> bool:
        """Validate time signature"""
        # Denominator must be power of 2
        if denominator not in [1, 2, 4, 8, 16, 32]:
            return False

        # Numerator must be positive
        if numerator <= 0:
            return False

        return True

    def suggest_related_parameters(self, concept: str) -> List[str]:
        """Suggest parameters related to a musical concept"""
        suggestions = []

        # Check complementary concepts
        for key, complements in self.complementary.items():
            if key in concept.lower():
                suggestions.extend(complements)
            elif any(comp in concept.lower() for comp in complements):
                suggestions.append(key)

        return list(set(suggestions))

    def detect_style_period(self, keywords: List[str]) -> Optional[str]:
        """Detect musical style period from keywords"""
        periods = {
            'baroque': ['figured_bass', 'ornament', 'harpsichord', 'fugue', 'toccata'],
            'classical': ['sonata', 'symphony', 'alberti', 'homophonic'],
            'romantic': ['chromaticism', 'rubato', 'expression', 'program_music'],
            'modern': ['atonal', 'serial', 'aleatory', 'minimalist'],
            'jazz': ['swing', 'bebop', 'modal', 'cool', 'fusion'],
            'blues': ['twelve_bar', 'blue_note', 'shuffle', 'call_response'],
            'rock': ['power_chord', 'distortion', 'backbeat'],
            'pop': ['verse_chorus', 'hook', 'catchy'],
        }

        keyword_lower = [k.lower() for k in keywords]

        for period, indicators in periods.items():
            if any(ind in ' '.join(keyword_lower) for ind in indicators):
                return period

        return None

    def analyze_harmonic_function(self, chord_root: int, key_root: int) -> str:
        """Analyze harmonic function of chord in key"""
        # Calculate scale degree (1-7)
        degree = ((chord_root - key_root) % 12) + 1

        functions = {
            0: 'tonic',        # I
            2: 'supertonic',   # ii
            4: 'mediant',      # iii
            5: 'subdominant',  # IV
            7: 'dominant',     # V
            9: 'submediant',   # vi
            11: 'leading_tone' # vii°
        }

        return functions.get(degree % 12, 'chromatic')

    def get_common_progressions(self, genre: str) -> List[List[str]]:
        """Get common chord progressions for genre"""
        progressions = {
            'pop': [
                ['I', 'V', 'vi', 'IV'],  # Axis progression
                ['I', 'IV', 'V', 'I'],   # Standard
                ['vi', 'IV', 'I', 'V'],  # Sensitive female
            ],
            'jazz': [
                ['I', 'vi', 'ii', 'V'],      # I-vi-ii-V
                ['IIm7', 'V7', 'Imaj7'],     # ii-V-I
                ['Im7', 'IV7', 'VIIm7b5'],   # Minor ii-V-i
            ],
            'blues': [
                ['I7', 'I7', 'I7', 'I7', 'IV7', 'IV7', 'I7', 'I7', 'V7', 'IV7', 'I7', 'V7'],
            ],
            'classical': [
                ['I', 'IV', 'V', 'I'],       # Plagal
                ['I', 'V', 'I'],             # Authentic
                ['I', 'ii', 'V', 'I'],       # Standard
            ]
        }

        return progressions.get(genre, [])

    def validate_harmonic_sequence(self, chords: List[str]) -> Dict[str, Any]:
        """Validate a sequence of chords"""
        issues = []
        warnings = []

        # Check for valid chord symbols
        for i, chord in enumerate(chords):
            if not self._is_valid_chord_symbol(chord):
                issues.append(f"Invalid chord symbol at position {i}: {chord}")

        # Check for direct V-IV motion (unusual)
        for i in range(len(chords) - 1):
            if 'V' in chords[i] and 'IV' in chords[i+1]:
                warnings.append(f"Unusual V-IV progression at position {i}-{i+1}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def _is_valid_chord_symbol(self, symbol: str) -> bool:
        """Check if chord symbol is valid"""
        # Basic validation - could be expanded
        valid_roots = ['C', 'D', 'E', 'F', 'G', 'A', 'B',
                      'Cb', 'Db', 'Eb', 'Fb', 'Gb', 'Ab', 'Bb',
                      'C#', 'D#', 'E#', 'F#', 'G#', 'A#', 'B#']

        valid_qualities = ['', 'm', 'maj', 'min', 'dim', 'aug', '7', 'M7', 'm7',
                          'maj7', 'min7', 'dim7', 'sus', 'sus2', 'sus4']

        # Roman numerals
        valid_romans = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII',
                       'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']

        # Check if it's a Roman numeral
        base_roman = symbol.split('m')[0].split('7')[0].split('b')[0].split('#')[0]
        if base_roman in valid_romans:
            return True

        # Check if it's a letter name chord
        if any(symbol.startswith(root) for root in valid_roots):
            return True

        return False


# ============================================================================
# MUSICAL VALIDATOR - Main Class
# ============================================================================

class MusicalValidator:
    """
    Comprehensive musical validation system for parameter and code proposals.

    Performs 7 validation checks on parameter proposals:
    1. Naming Convention
    2. Musical Validity (LLM-powered)
    3. Duplicate Detection
    4. Range Appropriateness
    5. Implementation Viability
    6. Test Coverage
    7. Music Theory Consistency

    Also validates generated code for:
    - Syntax correctness
    - Clean integration
    - Edge case handling
    - Backward compatibility
    """

    def __init__(self,
                 registry: Optional[UniversalParameterRegistry] = None,
                 anthropic_api_key: Optional[str] = None,
                 enable_llm: bool = True):
        """
        Initialize the Musical Validator

        Args:
            registry: Parameter registry (uses global REGISTRY if not provided)
            anthropic_api_key: API key for Anthropic (uses env var if not provided)
            enable_llm: Enable LLM-powered validation (requires API key)
        """
        self.registry = registry or REGISTRY
        self.theory_kb = MusicTheoryKnowledgeBase()

        # Initialize LLM client if available
        self.llm_client = None
        self.enable_llm = enable_llm and ANTHROPIC_AVAILABLE

        if self.enable_llm:
            api_key = anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                self.llm_client = anthropic.Anthropic(api_key=api_key)
            else:
                logging.warning("No Anthropic API key found. LLM validation disabled.")
                self.enable_llm = False

        # Cache for existing parameters
        self.existing_params = self._load_existing_parameters()

        # Statistics
        self.validation_count = 0
        self.pass_count = 0
        self.fail_count = 0

    def _load_existing_parameters(self) -> Dict[str, ParameterDefinition]:
        """Load existing parameters from registry"""
        return {
            path: param
            for path, param in self.registry.parameters.items()
        }

    # ========================================================================
    # MAIN VALIDATION METHOD
    # ========================================================================

    def validate_parameter(self, proposal: Dict[str, Any]) -> ParameterValidationResult:
        """
        Comprehensive validation of parameter proposal.

        Args:
            proposal: Dictionary containing parameter proposal with fields:
                - name: str (parameter name)
                - type: str (CONTINUOUS, CATEGORICAL, BOOLEAN, etc.)
                - range: Union[Tuple, List] (valid range/options)
                - default: Any (default value)
                - description: str
                - musical_context: str
                - implementation_strategy: str
                - generator_integration_points: List[str]
                - test_cases: List[Dict]
                - example_values: Dict[str, Any] (genre examples)
                - affected_features: List[str] (optional)

        Returns:
            ParameterValidationResult with detailed validation results
        """
        self.validation_count += 1

        checks = {}
        warnings = []
        errors = []
        suggestions = []

        # 1. Naming Convention Check
        naming_check = self._check_naming_convention(proposal)
        checks['naming_convention'] = naming_check
        if not naming_check.pass_status:
            if naming_check.severity == 'ERROR':
                errors.append(naming_check.message)
            else:
                warnings.append(naming_check.message)

        # 2. Musical Validity Check (LLM-powered)
        musical_check = self._check_musical_validity(proposal)
        checks['musical_validity'] = musical_check
        if not musical_check.pass_status:
            if musical_check.severity == 'ERROR':
                errors.append(musical_check.message)
            else:
                warnings.append(musical_check.message)

        # 3. Duplicate Check
        duplicate_check = self._check_duplicates(proposal)
        checks['no_duplicates'] = duplicate_check
        if not duplicate_check.pass_status:
            if duplicate_check.severity == 'ERROR':
                errors.append(duplicate_check.message)
            else:
                warnings.append(duplicate_check.message)

        # 4. Range Appropriateness
        range_check = self._check_range_appropriate(proposal)
        checks['range_appropriate'] = range_check
        if not range_check.pass_status:
            if range_check.severity == 'ERROR':
                errors.append(range_check.message)
            else:
                warnings.append(range_check.message)

        # 5. Implementation Viability
        impl_check = self._check_implementation_viable(proposal)
        checks['implementation_viable'] = impl_check
        if not impl_check.pass_status:
            if impl_check.severity == 'ERROR':
                errors.append(impl_check.message)
            else:
                warnings.append(impl_check.message)

        # 6. Test Coverage
        test_check = self._check_test_coverage(proposal)
        checks['test_coverage'] = test_check
        if not test_check.pass_status:
            if test_check.severity == 'ERROR':
                errors.append(test_check.message)
            else:
                warnings.append(test_check.message)

        # 7. Music Theory Consistency
        theory_check = self._check_theory_consistency(proposal)
        checks['theory_consistency'] = theory_check
        if not theory_check.pass_status:
            if theory_check.severity == 'ERROR':
                errors.append(theory_check.message)
            else:
                warnings.append(theory_check.message)

        # Calculate overall score
        pass_count = sum(1 for check in checks.values() if check.pass_status)
        score = pass_count / len(checks)

        # Determine validity (no errors = valid)
        valid = len(errors) == 0

        if valid:
            self.pass_count += 1
        else:
            self.fail_count += 1

        # Generate suggestions
        suggestions = self._generate_suggestions(proposal, checks)

        return ParameterValidationResult(
            valid=valid,
            score=score,
            checks=checks,
            warnings=warnings,
            errors=errors,
            suggestions=suggestions
        )

    # ========================================================================
    # VALIDATION CHECK 1: Naming Convention
    # ========================================================================

    def _check_naming_convention(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """
        Validate parameter naming follows convention.

        Convention: domain.module.parameter
        Example: harmony.voicing.spread
        """
        name = proposal.get('name', '')

        # Must be domain.module.parameter
        pattern = r'^([a-z_]+)\.([a-z_]+)\.([a-z_]+)$'
        match = re.match(pattern, name)

        if not match:
            return ValidationCheck(
                pass_status=False,
                message=f"Parameter name '{name}' does not follow convention: domain.module.parameter",
                severity='ERROR',
                details={'pattern': 'domain.module.parameter'}
            )

        domain, module, param = match.groups()

        # Check domain is valid
        if domain not in VALIDATION_CONFIG['valid_domains']:
            return ValidationCheck(
                pass_status=False,
                message=f"Domain '{domain}' not in valid domains: {VALIDATION_CONFIG['valid_domains']}",
                severity='ERROR',
                details={'valid_domains': VALIDATION_CONFIG['valid_domains']}
            )

        # Check for reasonable length
        if len(name) > VALIDATION_CONFIG['max_parameter_name_length']:
            return ValidationCheck(
                pass_status=False,
                message=f"Parameter name too long (>{VALIDATION_CONFIG['max_parameter_name_length']} chars): {name}",
                severity='WARNING',
                details={'length': len(name)}
            )

        # Check for descriptive naming
        min_length = VALIDATION_CONFIG['min_name_part_length']
        parts = [domain, module, param]
        short_parts = [p for p in parts if len(p) < min_length]

        if short_parts:
            return ValidationCheck(
                pass_status=False,
                message=f"Parameter name parts should be descriptive (>={min_length} chars): {short_parts}",
                severity='WARNING',
                details={'short_parts': short_parts}
            )

        # Check for numbers in inappropriate places
        if any(char.isdigit() for char in domain):
            return ValidationCheck(
                pass_status=False,
                message=f"Domain should not contain numbers: {domain}",
                severity='WARNING'
            )

        return ValidationCheck(
            pass_status=True,
            message='Naming convention OK',
            severity='INFO',
            details={'domain': domain, 'module': module, 'parameter': param}
        )

    # ========================================================================
    # VALIDATION CHECK 2: Musical Validity (LLM-powered)
    # ========================================================================

    def _check_musical_validity(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """
        Use LLM to validate musical sense of parameter.

        This check ensures the parameter represents a real, meaningful
        musical concept that would be used by actual composers/musicians.
        """
        if not self.enable_llm or not self.llm_client:
            # Fallback to heuristic check
            return self._check_musical_validity_heuristic(proposal)

        try:
            prompt = self._build_musical_validity_prompt(proposal)

            response = self.llm_client.messages.create(
                model=VALIDATION_CONFIG['llm_model'],
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                temperature=VALIDATION_CONFIG['llm_temperature']
            )

            # Parse LLM response
            result = self._parse_llm_response(response.content[0].text)

            if not result:
                return ValidationCheck(
                    pass_status=False,
                    message="Failed to parse LLM validation response",
                    severity='ERROR'
                )

            if not result['valid']:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Musical validity issues: {', '.join(result['issues'])}. {result['rationale']}",
                    severity='ERROR',
                    details={'llm_score': result['score'], 'issues': result['issues']}
                )

            min_score = VALIDATION_CONFIG['min_musical_validity_score']
            if result['score'] < min_score:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Musical validity score too low ({result['score']:.2f} < {min_score}): {result['rationale']}",
                    severity='WARNING',
                    details={'llm_score': result['score']}
                )

            return ValidationCheck(
                pass_status=True,
                message=f"Musical validity confirmed (score: {result['score']:.2f})",
                severity='INFO',
                details={'llm_score': result['score'], 'rationale': result['rationale']}
            )

        except Exception as e:
            logging.error(f"LLM validation error: {e}")
            # Fallback to heuristic
            return self._check_musical_validity_heuristic(proposal)

    def _build_musical_validity_prompt(self, proposal: Dict[str, Any]) -> str:
        """Build prompt for LLM musical validity check"""
        return f"""Analyze this proposed music generation parameter for musical validity:

Parameter: {proposal.get('name', 'N/A')}
Type: {proposal.get('type', 'N/A')}
Range: {proposal.get('range', 'N/A')}
Default: {proposal.get('default', 'N/A')}
Description: {proposal.get('description', 'N/A')}
Musical Context: {proposal.get('musical_context', 'N/A')}
Implementation: {proposal.get('implementation_strategy', 'N/A')}

Evaluate:
1. Does this parameter represent a real musical concept?
2. Is it clearly defined and unambiguous?
3. Would it produce audible, meaningful differences in generated music?
4. Is it used by real composers/musicians?
5. Does the implementation strategy make musical sense?
6. Are there any music theory contradictions?
7. Is the range/default appropriate for this musical concept?

Respond ONLY with JSON in this exact format:
{{
  "valid": true/false,
  "score": 0.0-1.0,
  "issues": ["list of specific issues if any"],
  "rationale": "Brief explanation (1-2 sentences)"
}}"""

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM"""
        try:
            # Try to extract JSON from response
            # Look for JSON block
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

                # Validate structure
                required_keys = ['valid', 'score', 'issues', 'rationale']
                if all(key in result for key in required_keys):
                    return result

            return None

        except json.JSONDecodeError:
            return None

    def _check_musical_validity_heuristic(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """
        Fallback heuristic check for musical validity.

        Uses rule-based checks when LLM is not available.
        """
        name = proposal.get('name', '')
        description = proposal.get('description', '')
        musical_context = proposal.get('musical_context', '')

        issues = []

        # Check description length (should be substantial)
        if len(description) < 20:
            issues.append("Description too brief (should explain musical concept)")

        # Check for musical context
        if not musical_context or len(musical_context) < 30:
            issues.append("Musical context missing or too brief")

        # Check for common music terms in description/context
        music_terms = [
            'chord', 'note', 'scale', 'melody', 'harmony', 'rhythm',
            'voice', 'bass', 'drum', 'tempo', 'beat', 'measure',
            'interval', 'dynamic', 'articulation', 'phrase', 'progression'
        ]

        text = (description + ' ' + musical_context).lower()
        if not any(term in text for term in music_terms):
            issues.append("No common musical terms found in description/context")

        # Check implementation strategy exists
        if not proposal.get('implementation_strategy'):
            issues.append("Implementation strategy missing")

        if issues:
            return ValidationCheck(
                pass_status=False,
                message=f"Musical validity concerns: {'; '.join(issues)}",
                severity='WARNING',
                details={'issues': issues}
            )

        return ValidationCheck(
            pass_status=True,
            message='Musical validity heuristic check passed',
            severity='INFO',
            details={'method': 'heuristic'}
        )

    # ========================================================================
    # VALIDATION CHECK 3: Duplicate Detection
    # ========================================================================

    def _check_duplicates(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """
        Check if parameter already exists or is too similar.

        Prevents redundant parameters and conceptual overlap.
        """
        name = proposal.get('name', '')

        # Exact duplicate
        if name in self.existing_params:
            return ValidationCheck(
                pass_status=False,
                message=f"Parameter already exists: {name}",
                severity='ERROR',
                details={'existing': True}
            )

        # Similar names (might be duplicate concept)
        similar = self._find_similar_parameters(name, proposal)

        if similar:
            similar_names = [s['name'] for s in similar[:3]]
            similarity_scores = [s['score'] for s in similar[:3]]

            # High similarity is an error
            if similarity_scores[0] > 0.8:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Very similar parameter exists (similarity={similarity_scores[0]:.2f}): {similar_names[0]}",
                    severity='ERROR',
                    details={'similar_parameters': similar}
                )

            # Medium similarity is a warning
            if similarity_scores[0] > 0.6:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Similar parameters exist, verify not duplicate: {', '.join(similar_names)}",
                    severity='WARNING',
                    details={'similar_parameters': similar}
                )

        return ValidationCheck(
            pass_status=True,
            message='No duplicates found',
            severity='INFO'
        )

    def _find_similar_parameters(self, name: str, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find parameters similar to the proposed one.

        Uses multiple similarity metrics:
        - Name token overlap
        - Description similarity
        - Same domain/module
        """
        similar = []

        name_parts = set(name.lower().replace('_', ' ').replace('.', ' ').split())
        description = proposal.get('description', '').lower()
        desc_words = set(description.split())

        for existing_name, existing_param in self.existing_params.items():
            # Name similarity
            existing_parts = set(existing_name.lower().replace('_', ' ').replace('.', ' ').split())
            name_overlap = len(name_parts & existing_parts)
            name_union = len(name_parts | existing_parts)
            name_similarity = name_overlap / name_union if name_union > 0 else 0.0

            # Description similarity
            existing_desc = existing_param.description.lower()
            existing_desc_words = set(existing_desc.split())
            desc_overlap = len(desc_words & existing_desc_words)
            desc_union = len(desc_words | existing_desc_words)
            desc_similarity = desc_overlap / desc_union if desc_union > 0 else 0.0

            # Combined similarity
            similarity = 0.6 * name_similarity + 0.4 * desc_similarity

            if similarity > 0.3:  # Threshold for "similar"
                similar.append({
                    'name': existing_name,
                    'score': similarity,
                    'name_similarity': name_similarity,
                    'desc_similarity': desc_similarity
                })

        # Sort by similarity score
        similar.sort(key=lambda x: x['score'], reverse=True)

        return similar

    # ========================================================================
    # VALIDATION CHECK 4: Range Appropriateness
    # ========================================================================

    def _check_range_appropriate(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """
        Validate parameter range makes musical sense.

        Checks:
        - Range format matches type
        - Min < Max for continuous
        - Default in range
        - Range appropriate for parameter type
        """
        param_type = proposal.get('type', '')
        param_range = proposal.get('range')
        default = proposal.get('default')
        name = proposal.get('name', '')

        # Convert type string to ParameterType if needed
        if isinstance(param_type, str):
            try:
                param_type_enum = ParameterType(param_type.lower())
            except ValueError:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Invalid parameter type: {param_type}",
                    severity='ERROR'
                )

        # CONTINUOUS type validation
        if param_type in ['CONTINUOUS', 'continuous', ParameterType.CONTINUOUS]:
            return self._validate_continuous_range(name, param_range, default)

        # CATEGORICAL type validation
        elif param_type in ['CATEGORICAL', 'categorical', ParameterType.CATEGORICAL]:
            return self._validate_categorical_range(name, param_range, default)

        # BOOLEAN type validation
        elif param_type in ['BOOLEAN', 'boolean', ParameterType.BOOLEAN]:
            return self._validate_boolean_range(name, default)

        # INTEGER type validation
        elif param_type in ['INTEGER', 'integer', ParameterType.INTEGER]:
            return self._validate_integer_range(name, param_range, default)

        # PROBABILITY type validation
        elif param_type in ['PROBABILITY', 'probability', ParameterType.PROBABILITY]:
            return self._validate_probability_range(name, param_range, default)

        # Default: assume valid
        return ValidationCheck(
            pass_status=True,
            message=f'Range validation not implemented for type {param_type}',
            severity='INFO'
        )

    def _validate_continuous_range(self, name: str, param_range: Any, default: Any) -> ValidationCheck:
        """Validate CONTINUOUS parameter range"""

        # Check range is [min, max] tuple/list
        if not (isinstance(param_range, (list, tuple)) and len(param_range) == 2):
            return ValidationCheck(
                pass_status=False,
                message=f"CONTINUOUS parameter must have [min, max] range, got: {param_range}",
                severity='ERROR',
                details={'range': param_range}
            )

        min_val, max_val = param_range

        # Check types
        if not (isinstance(min_val, (int, float)) and isinstance(max_val, (int, float))):
            return ValidationCheck(
                pass_status=False,
                message=f"Range values must be numeric, got: [{type(min_val)}, {type(max_val)}]",
                severity='ERROR'
            )

        # Min < Max
        if min_val >= max_val:
            return ValidationCheck(
                pass_status=False,
                message=f"Invalid range: min ({min_val}) must be < max ({max_val})",
                severity='ERROR',
                details={'min': min_val, 'max': max_val}
            )

        # Default in range
        if not isinstance(default, (int, float)):
            return ValidationCheck(
                pass_status=False,
                message=f"Default must be numeric for CONTINUOUS parameter, got: {type(default)}",
                severity='ERROR'
            )

        if not (min_val <= default <= max_val):
            return ValidationCheck(
                pass_status=False,
                message=f"Default ({default}) outside range [{min_val}, {max_val}]",
                severity='ERROR',
                details={'default': default, 'range': [min_val, max_val]}
            )

        # Check for standard ranges
        standard_range = self.theory_kb.get_standard_range(name)
        if standard_range:
            std_min, std_max = standard_range
            if not (min_val >= std_min and max_val <= std_max):
                return ValidationCheck(
                    pass_status=False,
                    message=f"Range [{min_val}, {max_val}] outside standard range [{std_min}, {std_max}] for this parameter type",
                    severity='WARNING',
                    details={'standard_range': standard_range}
                )

        # Common ranges validation
        if 'prob' in name or 'probability' in name:
            if not (min_val == 0.0 and max_val == 1.0):
                return ValidationCheck(
                    pass_status=False,
                    message=f"Probability parameters should use [0.0, 1.0] range, got [{min_val}, {max_val}]",
                    severity='WARNING'
                )

        if 'density' in name:
            if not (min_val >= 0.0 and max_val <= 1.0):
                return ValidationCheck(
                    pass_status=False,
                    message=f"Density parameters typically use [0.0, 1.0] range",
                    severity='WARNING'
                )

        return ValidationCheck(
            pass_status=True,
            message='Range appropriate',
            severity='INFO',
            details={'range': [min_val, max_val], 'default': default}
        )

    def _validate_categorical_range(self, name: str, param_range: Any, default: Any) -> ValidationCheck:
        """Validate CATEGORICAL parameter range"""

        # Check has at least 2 options
        if not (isinstance(param_range, list) and len(param_range) >= 2):
            return ValidationCheck(
                pass_status=False,
                message=f"CATEGORICAL parameter must have at least 2 options, got: {param_range}",
                severity='ERROR',
                details={'range': param_range}
            )

        # Default in options
        if default not in param_range:
            return ValidationCheck(
                pass_status=False,
                message=f"Default '{default}' not in categorical options: {param_range}",
                severity='ERROR',
                details={'default': default, 'options': param_range}
            )

        # Check for duplicates in options
        if len(param_range) != len(set(param_range)):
            return ValidationCheck(
                pass_status=False,
                message=f"Categorical options contain duplicates: {param_range}",
                severity='ERROR'
            )

        # Check option names are descriptive
        if any(len(str(opt)) < 2 for opt in param_range):
            return ValidationCheck(
                pass_status=False,
                message=f"Categorical options should be descriptive (>1 char)",
                severity='WARNING'
            )

        return ValidationCheck(
            pass_status=True,
            message=f'Categorical range appropriate ({len(param_range)} options)',
            severity='INFO',
            details={'options': param_range, 'default': default}
        )

    def _validate_boolean_range(self, name: str, default: Any) -> ValidationCheck:
        """Validate BOOLEAN parameter"""

        # Default must be bool
        if not isinstance(default, bool):
            return ValidationCheck(
                pass_status=False,
                message=f"BOOLEAN parameter must have boolean default, got: {type(default)}",
                severity='ERROR',
                details={'default': default, 'default_type': type(default).__name__}
            )

        return ValidationCheck(
            pass_status=True,
            message='Boolean parameter OK',
            severity='INFO',
            details={'default': default}
        )

    def _validate_integer_range(self, name: str, param_range: Any, default: Any) -> ValidationCheck:
        """Validate INTEGER parameter range"""

        # Check range is [min, max] tuple/list
        if not (isinstance(param_range, (list, tuple)) and len(param_range) == 2):
            return ValidationCheck(
                pass_status=False,
                message=f"INTEGER parameter must have [min, max] range, got: {param_range}",
                severity='ERROR'
            )

        min_val, max_val = param_range

        # Check types
        if not (isinstance(min_val, int) and isinstance(max_val, int)):
            return ValidationCheck(
                pass_status=False,
                message=f"INTEGER range values must be integers, got: [{type(min_val)}, {type(max_val)}]",
                severity='ERROR'
            )

        # Min < Max
        if min_val >= max_val:
            return ValidationCheck(
                pass_status=False,
                message=f"Invalid range: min ({min_val}) must be < max ({max_val})",
                severity='ERROR'
            )

        # Default in range
        if not isinstance(default, int):
            return ValidationCheck(
                pass_status=False,
                message=f"Default must be integer, got: {type(default)}",
                severity='ERROR'
            )

        if not (min_val <= default <= max_val):
            return ValidationCheck(
                pass_status=False,
                message=f"Default ({default}) outside range [{min_val}, {max_val}]",
                severity='ERROR'
            )

        return ValidationCheck(
            pass_status=True,
            message='Integer range appropriate',
            severity='INFO',
            details={'range': [min_val, max_val], 'default': default}
        )

    def _validate_probability_range(self, name: str, param_range: Any, default: Any) -> ValidationCheck:
        """Validate PROBABILITY parameter"""

        # Should be [0.0, 1.0]
        if param_range != [0.0, 1.0] and param_range != (0.0, 1.0):
            return ValidationCheck(
                pass_status=False,
                message=f"PROBABILITY parameters must have [0.0, 1.0] range, got: {param_range}",
                severity='ERROR',
                details={'range': param_range}
            )

        # Default must be in [0.0, 1.0]
        if not isinstance(default, (int, float)):
            return ValidationCheck(
                pass_status=False,
                message=f"PROBABILITY default must be numeric, got: {type(default)}",
                severity='ERROR'
            )

        if not (0.0 <= default <= 1.0):
            return ValidationCheck(
                pass_status=False,
                message=f"PROBABILITY default ({default}) must be in [0.0, 1.0]",
                severity='ERROR',
                details={'default': default}
            )

        return ValidationCheck(
            pass_status=True,
            message='Probability range appropriate',
            severity='INFO',
            details={'default': default}
        )

    # ========================================================================
    # VALIDATION CHECK 5: Implementation Viability
    # ========================================================================

    def _check_implementation_viable(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """
        Check if implementation strategy is clear and viable.

        Implementation strategy should explain:
        - HOW the generator will use this parameter
        - WHAT code changes are needed
        - WHERE in the generator it will be integrated
        """
        impl_strategy = proposal.get('implementation_strategy', '')

        # Must have implementation strategy
        if not impl_strategy or len(impl_strategy) < 50:
            return ValidationCheck(
                pass_status=False,
                message="Implementation strategy missing or too brief (need detailed explanation, min 50 chars)",
                severity='ERROR',
                details={'length': len(impl_strategy)}
            )

        # Should mention key implementation concepts
        key_terms = [
            'generator', 'check', 'if', 'probability', 'create', 'build',
            'calculate', 'algorithm', 'function', 'method', 'use', 'apply'
        ]

        impl_lower = impl_strategy.lower()
        found_terms = [term for term in key_terms if term in impl_lower]

        if len(found_terms) < 2:
            return ValidationCheck(
                pass_status=False,
                message="Implementation strategy should describe HOW generator will use parameter (mention algorithm/function/method)",
                severity='WARNING',
                details={'found_terms': found_terms}
            )

        # Should have integration points
        integration_points = proposal.get('generator_integration_points', [])
        if not integration_points:
            return ValidationCheck(
                pass_status=False,
                message="No generator integration points specified - must indicate WHERE in generator code this will be used",
                severity='ERROR'
            )

        if len(integration_points) < 1:
            return ValidationCheck(
                pass_status=False,
                message="Need at least one generator integration point",
                severity='ERROR'
            )

        # Check integration points are specific
        vague_points = ['generator', 'somewhere', 'TBD', 'TODO', 'unknown']
        if any(vague in str(integration_points).lower() for vague in vague_points):
            return ValidationCheck(
                pass_status=False,
                message="Integration points too vague - need specific file/function names",
                severity='WARNING',
                details={'integration_points': integration_points}
            )

        return ValidationCheck(
            pass_status=True,
            message=f'Implementation strategy viable ({len(integration_points)} integration points)',
            severity='INFO',
            details={
                'strategy_length': len(impl_strategy),
                'integration_points': integration_points,
                'found_terms': found_terms
            }
        )

    # ========================================================================
    # VALIDATION CHECK 6: Test Coverage
    # ========================================================================

    def _check_test_coverage(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """
        Validate test cases are comprehensive.

        Test cases should cover:
        - Boundary values (min/max)
        - Default value
        - Representative middle values
        - All categorical options
        """
        test_cases = proposal.get('test_cases', [])
        param_type = proposal.get('type', '')

        # Must have test cases
        min_tests = VALIDATION_CONFIG['min_test_cases']
        if not test_cases or len(test_cases) < min_tests:
            return ValidationCheck(
                pass_status=False,
                message=f"Need at least {min_tests} test cases (min and max values), got {len(test_cases)}",
                severity='WARNING',
                details={'test_count': len(test_cases)}
            )

        # Type-specific validation
        if param_type in ['CONTINUOUS', 'continuous', ParameterType.CONTINUOUS]:
            return self._validate_continuous_tests(proposal, test_cases)

        elif param_type in ['CATEGORICAL', 'categorical', ParameterType.CATEGORICAL]:
            return self._validate_categorical_tests(proposal, test_cases)

        elif param_type in ['INTEGER', 'integer', ParameterType.INTEGER]:
            return self._validate_integer_tests(proposal, test_cases)

        # Generic validation
        for i, tc in enumerate(test_cases):
            if 'value' not in tc:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Test case {i} missing 'value' field",
                    severity='ERROR'
                )

            if 'expected' not in tc or not tc['expected']:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Test case {i} should have 'expected' field describing outcome",
                    severity='WARNING'
                )

        return ValidationCheck(
            pass_status=True,
            message=f'Test coverage adequate ({len(test_cases)} cases)',
            severity='INFO',
            details={'test_count': len(test_cases)}
        )

    def _validate_continuous_tests(self, proposal: Dict[str, Any], test_cases: List[Dict]) -> ValidationCheck:
        """Validate test cases for CONTINUOUS parameters"""

        param_range = proposal.get('range', [])
        if len(param_range) != 2:
            return ValidationCheck(
                pass_status=True,
                message='Cannot validate test coverage without valid range',
                severity='INFO'
            )

        min_val, max_val = param_range
        values = [tc.get('value') for tc in test_cases if 'value' in tc]

        # Should test both extremes
        if min_val not in values:
            return ValidationCheck(
                pass_status=False,
                message=f"Should test minimum value ({min_val})",
                severity='WARNING',
                details={'missing': 'min', 'min_val': min_val}
            )

        if max_val not in values:
            return ValidationCheck(
                pass_status=False,
                message=f"Should test maximum value ({max_val})",
                severity='WARNING',
                details={'missing': 'max', 'max_val': max_val}
            )

        # Should test middle value
        mid_val = (min_val + max_val) / 2
        if not any(abs(v - mid_val) < (max_val - min_val) * 0.2 for v in values if v is not None):
            return ValidationCheck(
                pass_status=False,
                message=f"Should test middle value (around {mid_val:.2f})",
                severity='INFO'
            )

        return ValidationCheck(
            pass_status=True,
            message=f'CONTINUOUS test coverage good ({len(test_cases)} cases including boundaries)',
            severity='INFO',
            details={'tested_values': values}
        )

    def _validate_categorical_tests(self, proposal: Dict[str, Any], test_cases: List[Dict]) -> ValidationCheck:
        """Validate test cases for CATEGORICAL parameters"""

        options = proposal.get('range', [])
        if not options:
            return ValidationCheck(
                pass_status=True,
                message='Cannot validate test coverage without options',
                severity='INFO'
            )

        values = [tc.get('value') for tc in test_cases if 'value' in tc]

        # Should test all options
        untested = set(options) - set(values)
        if untested:
            return ValidationCheck(
                pass_status=False,
                message=f"Untested categorical options: {untested}",
                severity='WARNING',
                details={'untested_options': list(untested)}
            )

        return ValidationCheck(
            pass_status=True,
            message=f'CATEGORICAL test coverage complete (all {len(options)} options tested)',
            severity='INFO',
            details={'tested_values': values}
        )

    def _validate_integer_tests(self, proposal: Dict[str, Any], test_cases: List[Dict]) -> ValidationCheck:
        """Validate test cases for INTEGER parameters"""

        param_range = proposal.get('range', [])
        if len(param_range) != 2:
            return ValidationCheck(
                pass_status=True,
                message='Cannot validate test coverage without valid range',
                severity='INFO'
            )

        min_val, max_val = param_range
        values = [tc.get('value') for tc in test_cases if 'value' in tc]

        # Should test both extremes
        if min_val not in values:
            return ValidationCheck(
                pass_status=False,
                message=f"Should test minimum value ({min_val})",
                severity='WARNING',
                details={'missing': 'min'}
            )

        if max_val not in values:
            return ValidationCheck(
                pass_status=False,
                message=f"Should test maximum value ({max_val})",
                severity='WARNING',
                details={'missing': 'max'}
            )

        return ValidationCheck(
            pass_status=True,
            message=f'INTEGER test coverage good ({len(test_cases)} cases)',
            severity='INFO',
            details={'tested_values': values}
        )

    # ========================================================================
    # VALIDATION CHECK 7: Music Theory Consistency
    # ========================================================================

    def _check_theory_consistency(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """
        Check for music theory contradictions.

        Validates:
        - No contradictory terms in name/description
        - Domain-specific rules (harmony, rhythm, etc.)
        - Genre compatibility
        """
        name = proposal.get('name', '')
        description = proposal.get('description', '').lower()
        musical_context = proposal.get('musical_context', '').lower()

        # Extract domain from name
        domain = name.split('.')[0] if '.' in name else ''

        # Check for contradictions in terminology
        contradiction_check = self._check_contradictory_terms(name, description, musical_context)
        if not contradiction_check.pass_status:
            return contradiction_check

        # Domain-specific validation
        if domain == 'harmony':
            return self._validate_harmony_theory(proposal)
        elif domain == 'rhythm':
            return self._validate_rhythm_theory(proposal)
        elif domain == 'melody':
            return self._validate_melody_theory(proposal)
        elif domain == 'voicing':
            return self._validate_voicing_theory(proposal)

        # Default: no theory issues found
        return ValidationCheck(
            pass_status=True,
            message='Music theory consistent',
            severity='INFO'
        )

    def _check_contradictory_terms(self, name: str, description: str, musical_context: str) -> ValidationCheck:
        """Check for contradictory musical terms"""

        text = f"{name} {description} {musical_context}".lower()

        # Check all known contradictions
        for term, conflicting_terms in self.theory_kb.contradictions.items():
            if term in text:
                for conflict in conflicting_terms:
                    if conflict in text:
                        return ValidationCheck(
                            pass_status=False,
                            message=f"Potential music theory contradiction: '{term}' and '{conflict}' are typically opposed concepts",
                            severity='WARNING',
                            details={'term1': term, 'term2': conflict}
                        )

        return ValidationCheck(
            pass_status=True,
            message='No contradictory terms found',
            severity='INFO'
        )

    def _validate_harmony_theory(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """Validate harmony-specific theory rules"""

        name = proposal.get('name', '')
        description = proposal.get('description', '').lower()

        # Voicing checks
        if 'voicing' in name:
            # Cannot simultaneously omit and double a chord tone
            if 'omit' in name and 'doubled' in name:
                return ValidationCheck(
                    pass_status=False,
                    message="Cannot simultaneously omit and double a chord tone",
                    severity='ERROR'
                )

            # Voicing should specify which voices
            if 'voice' not in description and 'chord' not in description:
                return ValidationCheck(
                    pass_status=False,
                    message="Voicing parameters should reference voices or chords in description",
                    severity='WARNING'
                )

        # Extension checks
        if 'extension' in name or 'tension' in name:
            # Should reference scale degrees or intervals
            if not any(term in description for term in ['9th', '11th', '13th', 'seventh', 'ninth', 'eleventh']):
                return ValidationCheck(
                    pass_status=False,
                    message="Extension parameters should specify which extensions (9th, 11th, 13th)",
                    severity='WARNING'
                )

        # Substitution checks
        if 'substitution' in name or 'sub' in name:
            # Should explain what is being substituted
            if 'replace' not in description and 'instead' not in description:
                return ValidationCheck(
                    pass_status=False,
                    message="Substitution parameters should explain what is being replaced",
                    severity='WARNING'
                )

        return ValidationCheck(
            pass_status=True,
            message='Harmony theory consistent',
            severity='INFO'
        )

    def _validate_rhythm_theory(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """Validate rhythm-specific theory rules"""

        name = proposal.get('name', '')
        description = proposal.get('description', '').lower()
        param_range = proposal.get('range')

        # Polyrhythm checks
        if 'polyrhythm' in name:
            # Should have ratio specification
            if 'ratio' not in name and 'ratio' not in description and 'ratio' not in str(param_range):
                return ValidationCheck(
                    pass_status=False,
                    message="Polyrhythm parameters should specify ratio (e.g., 3:2, 4:3)",
                    severity='WARNING'
                )

        # Swing checks
        if 'swing' in name:
            # Swing ratio typically 0.5-0.75
            if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                min_val, max_val = param_range
                if not (0.5 <= min_val <= 0.75 and 0.5 <= max_val <= 0.75):
                    return ValidationCheck(
                        pass_status=False,
                        message="Swing ratio typically ranges from 0.5 (straight) to 0.75 (hard swing)",
                        severity='INFO',
                        details={'suggested_range': [0.5, 0.75]}
                    )

        # Tuplet checks
        if 'tuplet' in name or 'triplet' in name:
            # Should specify ratio
            if not any(term in description for term in ['3:2', '5:4', '6:4', '7:4', 'ratio']):
                return ValidationCheck(
                    pass_status=False,
                    message="Tuplet parameters should specify ratio (e.g., 3:2 for triplets)",
                    severity='WARNING'
                )

        return ValidationCheck(
            pass_status=True,
            message='Rhythm theory consistent',
            severity='INFO'
        )

    def _validate_melody_theory(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """Validate melody-specific theory rules"""

        name = proposal.get('name', '')
        description = proposal.get('description', '').lower()

        # Interval checks
        if 'interval' in name or 'leap' in name:
            # Should specify size or direction
            if not any(term in description for term in ['semitone', 'step', 'octave', 'ascending', 'descending']):
                return ValidationCheck(
                    pass_status=False,
                    message="Interval parameters should specify size or direction",
                    severity='WARNING'
                )

        # Contour checks
        if 'contour' in name:
            # Should mention shape
            if not any(term in description for term in ['arch', 'ascending', 'descending', 'wave', 'shape']):
                return ValidationCheck(
                    pass_status=False,
                    message="Contour parameters should describe melodic shape",
                    severity='WARNING'
                )

        return ValidationCheck(
            pass_status=True,
            message='Melody theory consistent',
            severity='INFO'
        )

    def _validate_voicing_theory(self, proposal: Dict[str, Any]) -> ValidationCheck:
        """Validate voicing-specific theory rules"""

        name = proposal.get('name', '')
        description = proposal.get('description', '').lower()

        # Should reference chord structure
        if not any(term in description for term in ['chord', 'note', 'voice', 'interval', 'spacing']):
            return ValidationCheck(
                pass_status=False,
                message="Voicing parameters should reference chord structure or spacing",
                severity='WARNING'
            )

        return ValidationCheck(
            pass_status=True,
            message='Voicing theory consistent',
            severity='INFO'
        )

    # ========================================================================
    # SUGGESTION GENERATION
    # ========================================================================

    def _generate_suggestions(self, proposal: Dict[str, Any], checks: Dict[str, ValidationCheck]) -> List[str]:
        """Generate improvement suggestions based on validation results"""

        suggestions = []
        name = proposal.get('name', '')

        # Suggest related parameters
        domain = name.split('.')[0] if '.' in name else ''
        if domain:
            related = [p for p in self.existing_params.keys() if p.startswith(domain)]
            if len(related) > 5:
                suggestions.append(
                    f"Consider how this parameter interacts with {len(related)} existing {domain} parameters"
                )

        # Suggest example values
        example_values = proposal.get('example_values', {})
        if not example_values or len(example_values) < 3:
            suggestions.append(
                "Add more genre-specific example values (aim for 5+ genres)"
            )

        # Suggest feature correlation
        affected_features = proposal.get('affected_features', [])
        if len(affected_features) == 1:
            suggestions.append(
                "Consider whether this parameter affects other features beyond the one listed"
            )

        # Suggest documentation
        if len(proposal.get('description', '')) < 100:
            suggestions.append(
                "Expand description with more detail about musical use cases and examples"
            )

        # Suggest integration testing
        integration_points = proposal.get('generator_integration_points', [])
        if len(integration_points) == 1:
            suggestions.append(
                "Consider whether this parameter should be integrated at multiple points in the generator"
            )

        # Suggest complementary parameters
        for term, complements in self.theory_kb.complementary.items():
            if term in name.lower():
                missing = [c for c in complements if c not in proposal.get('description', '').lower()]
                if missing:
                    suggestions.append(
                        f"Consider mentioning complementary concepts: {', '.join(missing[:2])}"
                    )

        return suggestions

    # ========================================================================
    # CODE VALIDATION
    # ========================================================================

    def validate_code(self, code: Dict[str, Any], proposal: Dict[str, Any]) -> CodeValidationResult:
        """
        Validate generated code for a parameter implementation.

        Args:
            code: Dictionary containing:
                - generator_modifications: Dict[str, str] (file_path -> code)
                - test_code: str (test implementation)
                - registry_entry: str (registry addition code)
            proposal: Original parameter proposal

        Returns:
            CodeValidationResult
        """
        checks = {}
        errors = []
        warnings = []

        # 1. Syntax validity
        syntax_check = self._check_code_syntax(code)
        checks['syntax_valid'] = syntax_check
        if not syntax_check.pass_status:
            if syntax_check.severity == 'ERROR':
                errors.append(syntax_check.message)
            else:
                warnings.append(syntax_check.message)

        # 2. Integration cleanliness
        integration_check = self._check_clean_integration(code, proposal)
        checks['integrates_cleanly'] = integration_check
        if not integration_check.pass_status:
            if integration_check.severity == 'ERROR':
                errors.append(integration_check.message)
            else:
                warnings.append(integration_check.message)

        # 3. Edge case handling
        edge_case_check = self._check_edge_cases(code, proposal)
        checks['handles_edge_cases'] = edge_case_check
        if not edge_case_check.pass_status:
            if edge_case_check.severity == 'ERROR':
                errors.append(edge_case_check.message)
            else:
                warnings.append(edge_case_check.message)

        # 4. Backward compatibility
        compat_check = self._check_backward_compatible(code)
        checks['backward_compatible'] = compat_check
        if not compat_check.pass_status:
            if compat_check.severity == 'ERROR':
                errors.append(compat_check.message)
            else:
                warnings.append(compat_check.message)

        # Determine validity
        valid = len(errors) == 0

        return CodeValidationResult(
            valid=valid,
            checks=checks,
            errors=errors,
            warnings=warnings
        )

    def _check_code_syntax(self, code: Dict[str, Any]) -> ValidationCheck:
        """Check Python syntax validity"""

        generator_mods = code.get('generator_modifications', {})

        for file_path, file_code in generator_mods.items():
            try:
                ast.parse(file_code)
            except SyntaxError as e:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Syntax error in {file_path}: {e}",
                    severity='ERROR',
                    details={'file': file_path, 'error': str(e)}
                )

        # Check test code
        test_code = code.get('test_code')
        if test_code:
            try:
                ast.parse(test_code)
            except SyntaxError as e:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Syntax error in test code: {e}",
                    severity='ERROR',
                    details={'error': str(e)}
                )

        # Check registry entry
        registry_entry = code.get('registry_entry')
        if registry_entry:
            try:
                ast.parse(registry_entry)
            except SyntaxError as e:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Syntax error in registry entry: {e}",
                    severity='ERROR',
                    details={'error': str(e)}
                )

        return ValidationCheck(
            pass_status=True,
            message='Syntax valid',
            severity='INFO'
        )

    def _check_clean_integration(self, code: Dict[str, Any], proposal: Dict[str, Any]) -> ValidationCheck:
        """Check code integrates cleanly"""

        param_name = proposal.get('name', '')
        generator_mods = code.get('generator_modifications', {})

        for file_path, file_code in generator_mods.items():
            # Must use .get() for parameter access (backward compatibility)
            if param_name in file_code:
                # Check for direct dictionary access
                if f"params['{param_name}']" in file_code or f'params["{param_name}"]' in file_code:
                    return ValidationCheck(
                        pass_status=False,
                        message=f"Must use params.get('{param_name}', default) for backward compatibility, not params['{param_name}']",
                        severity='ERROR',
                        details={'file': file_path}
                    )

            # Should have docstrings for new functions
            if 'def ' in file_code:
                # Parse AST to check for docstrings
                try:
                    tree = ast.parse(file_code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if not ast.get_docstring(node):
                                return ValidationCheck(
                                    pass_status=False,
                                    message=f"Function '{node.name}' in {file_path} should have docstring",
                                    severity='WARNING',
                                    details={'file': file_path, 'function': node.name}
                                )
                except:
                    pass  # Already checked syntax

        return ValidationCheck(
            pass_status=True,
            message='Integration clean',
            severity='INFO'
        )

    def _check_edge_cases(self, code: Dict[str, Any], proposal: Dict[str, Any]) -> ValidationCheck:
        """Check edge case handling"""

        param_type = proposal.get('type', '')
        generator_mods = code.get('generator_modifications', {})

        for file_path, file_code in generator_mods.items():
            # Continuous/probability parameters should validate range
            if param_type in ['CONTINUOUS', 'PROBABILITY', 'continuous', 'probability']:
                # Should clamp or validate values
                has_validation = any(
                    term in file_code
                    for term in ['max(', 'min(', 'clip', 'clamp', 'np.clip', 'if ', '>=', '<=']
                )

                if not has_validation:
                    return ValidationCheck(
                        pass_status=False,
                        message=f"{param_type} parameters should validate range (use max/min, np.clip, or conditionals)",
                        severity='WARNING',
                        details={'file': file_path}
                    )

            # Should handle None/missing values
            if '.get(' in file_code:
                # Good - using .get() which handles missing keys
                pass
            else:
                # Check for None handling
                if 'is not None' not in file_code and 'is None' not in file_code:
                    return ValidationCheck(
                        pass_status=False,
                        message="Should check for None values or use .get() with defaults",
                        severity='WARNING',
                        details={'file': file_path}
                    )

        return ValidationCheck(
            pass_status=True,
            message='Edge cases handled',
            severity='INFO'
        )

    def _check_backward_compatible(self, code: Dict[str, Any]) -> ValidationCheck:
        """Ensure backward compatibility"""

        generator_mods = code.get('generator_modifications', {})

        for file_path, file_code in generator_mods.items():
            # Must use .get() with defaults, not direct access
            if 'params[' in file_code and 'params[i]' not in file_code:
                # Check if it's array indexing or dict access
                # Look for string keys
                if "params['" in file_code or 'params["' in file_code:
                    return ValidationCheck(
                        pass_status=False,
                        message="Use params.get() not params[] for backward compatibility",
                        severity='ERROR',
                        details={'file': file_path}
                    )

            # Should not remove or modify existing parameters
            # This is hard to check without original code, so we just warn
            if 'del params' in file_code or 'params.pop' in file_code:
                return ValidationCheck(
                    pass_status=False,
                    message="Should not delete existing parameters - breaks backward compatibility",
                    severity='ERROR',
                    details={'file': file_path}
                )

        return ValidationCheck(
            pass_status=True,
            message='Backward compatible',
            severity='INFO'
        )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            'total_validations': self.validation_count,
            'passed': self.pass_count,
            'failed': self.fail_count,
            'pass_rate': self.pass_count / self.validation_count if self.validation_count > 0 else 0.0,
            'llm_enabled': self.enable_llm,
            'existing_parameters': len(self.existing_params)
        }

    def export_validation_report(self, result: ParameterValidationResult, filepath: str):
        """Export validation result to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def batch_validate(self, proposals: List[Dict[str, Any]]) -> List[ParameterValidationResult]:
        """Validate multiple parameter proposals"""
        results = []
        for proposal in proposals:
            result = self.validate_parameter(proposal)
            results.append(result)
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_validator(anthropic_api_key: Optional[str] = None, enable_llm: bool = True) -> MusicalValidator:
    """
    Create a MusicalValidator instance.

    Args:
        anthropic_api_key: API key for Anthropic (uses env var if not provided)
        enable_llm: Enable LLM-powered validation

    Returns:
        Configured MusicalValidator
    """
    return MusicalValidator(
        registry=REGISTRY,
        anthropic_api_key=anthropic_api_key,
        enable_llm=enable_llm
    )


def validate_parameter_proposal(proposal: Dict[str, Any],
                                 anthropic_api_key: Optional[str] = None) -> ParameterValidationResult:
    """
    Convenience function to validate a single parameter proposal.

    Args:
        proposal: Parameter proposal dictionary
        anthropic_api_key: API key for Anthropic (optional)

    Returns:
        ParameterValidationResult
    """
    validator = create_validator(anthropic_api_key=anthropic_api_key)
    return validator.validate_parameter(proposal)


def validate_code_implementation(code: Dict[str, Any],
                                  proposal: Dict[str, Any]) -> CodeValidationResult:
    """
    Convenience function to validate code implementation.

    Args:
        code: Code implementation dictionary
        proposal: Original parameter proposal

    Returns:
        CodeValidationResult
    """
    validator = create_validator(enable_llm=False)  # LLM not needed for code validation
    return validator.validate_code(code, proposal)


# ============================================================================
# PARAMETER TEMPLATES AND EXAMPLES
# ============================================================================

class ParameterTemplateLibrary:
    """
    Library of parameter templates for common musical concepts.

    Provides validated templates that can be used as starting points
    for new parameter proposals.
    """

    @staticmethod
    def get_probability_template(
        domain: str,
        module: str,
        param_name: str,
        description: str,
        musical_context: str
    ) -> Dict[str, Any]:
        """Generate a probability parameter template"""
        return {
            'name': f'{domain}.{module}.{param_name}',
            'type': 'PROBABILITY',
            'range': [0.0, 1.0],
            'default': 0.5,
            'description': description,
            'musical_context': musical_context,
            'implementation_strategy': f'Check this probability when {param_name}. If random() < value, apply the effect.',
            'generator_integration_points': [f'{domain.capitalize()}Module.method_name'],
            'test_cases': [
                {'value': 0.0, 'expected': f'No {param_name} applied'},
                {'value': 0.5, 'expected': f'{param_name} applied ~50% of the time'},
                {'value': 1.0, 'expected': f'{param_name} always applied'}
            ],
            'example_values': {
                'jazz': 0.7,
                'classical': 0.3,
                'pop': 0.5,
                'rock': 0.4
            }
        }

    @staticmethod
    def get_continuous_template(
        domain: str,
        module: str,
        param_name: str,
        description: str,
        musical_context: str,
        min_val: float,
        max_val: float,
        default_val: float
    ) -> Dict[str, Any]:
        """Generate a continuous parameter template"""
        return {
            'name': f'{domain}.{module}.{param_name}',
            'type': 'CONTINUOUS',
            'range': [min_val, max_val],
            'default': default_val,
            'description': description,
            'musical_context': musical_context,
            'implementation_strategy': f'Use this value to control {param_name} amount. Scale the effect proportionally within the range.',
            'generator_integration_points': [f'{domain.capitalize()}Module.method_name'],
            'test_cases': [
                {'value': min_val, 'expected': f'Minimum {param_name}'},
                {'value': default_val, 'expected': f'Default {param_name}'},
                {'value': max_val, 'expected': f'Maximum {param_name}'}
            ],
            'example_values': {
                'jazz': (min_val + max_val) * 0.7,
                'classical': (min_val + max_val) * 0.3,
                'pop': default_val,
                'rock': (min_val + max_val) * 0.6
            }
        }

    @staticmethod
    def get_categorical_template(
        domain: str,
        module: str,
        param_name: str,
        description: str,
        musical_context: str,
        options: List[str],
        default_option: str
    ) -> Dict[str, Any]:
        """Generate a categorical parameter template"""
        return {
            'name': f'{domain}.{module}.{param_name}',
            'type': 'CATEGORICAL',
            'range': options,
            'default': default_option,
            'description': description,
            'musical_context': musical_context,
            'implementation_strategy': f'Switch on the selected option to determine behavior. Each option should produce distinct musical output.',
            'generator_integration_points': [f'{domain.capitalize()}Module.method_name'],
            'test_cases': [
                {'value': opt, 'expected': f'{param_name} using {opt} behavior'}
                for opt in options
            ],
            'example_values': {
                genre: options[i % len(options)]
                for i, genre in enumerate(['jazz', 'classical', 'pop', 'rock', 'blues'])
            }
        }

    @staticmethod
    def get_harmony_parameter_template(
        module: str,
        param_name: str,
        description: str
    ) -> Dict[str, Any]:
        """Get harmony-specific parameter template"""
        return ParameterTemplateLibrary.get_probability_template(
            domain='harmony',
            module=module,
            param_name=param_name,
            description=description,
            musical_context=f'Harmonic parameter controlling {param_name} in chord progressions and voicings.'
        )

    @staticmethod
    def get_rhythm_parameter_template(
        module: str,
        param_name: str,
        description: str
    ) -> Dict[str, Any]:
        """Get rhythm-specific parameter template"""
        return ParameterTemplateLibrary.get_probability_template(
            domain='rhythm',
            module=module,
            param_name=param_name,
            description=description,
            musical_context=f'Rhythmic parameter controlling {param_name} timing and patterns.'
        )

    @staticmethod
    def get_melody_parameter_template(
        module: str,
        param_name: str,
        description: str
    ) -> Dict[str, Any]:
        """Get melody-specific parameter template"""
        return ParameterTemplateLibrary.get_probability_template(
            domain='melody',
            module=module,
            param_name=param_name,
            description=description,
            musical_context=f'Melodic parameter controlling {param_name} in note selection and contour.'
        )


class ValidationProfile:
    """
    Validation profile for different validation strictness levels.

    Allows customizing validation behavior based on context:
    - strict: All checks must pass, no warnings allowed
    - standard: All error checks must pass, warnings OK
    - lenient: Only critical errors fail validation
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    def should_fail_on_warning(self) -> bool:
        """Check if warnings should fail validation"""
        return self.config.get('fail_on_warning', False)

    def get_min_score(self) -> float:
        """Get minimum acceptable score"""
        return self.config.get('min_score', 0.5)

    def is_check_required(self, check_name: str) -> bool:
        """Check if a validation check is required"""
        required_checks = self.config.get('required_checks', [])
        if not required_checks:
            return True  # All checks required by default
        return check_name in required_checks

    def get_severity_threshold(self) -> str:
        """Get minimum severity level to report"""
        return self.config.get('severity_threshold', 'WARNING')


class ValidationProfileManager:
    """
    Manages validation profiles for different use cases.
    """

    def __init__(self):
        self.profiles = self._load_default_profiles()

    def _load_default_profiles(self) -> Dict[str, ValidationProfile]:
        """Load default validation profiles"""
        return {
            'strict': ValidationProfile('strict', {
                'fail_on_warning': True,
                'min_score': 1.0,
                'required_checks': [
                    'naming_convention',
                    'musical_validity',
                    'no_duplicates',
                    'range_appropriate',
                    'implementation_viable',
                    'test_coverage',
                    'theory_consistency'
                ],
                'severity_threshold': 'INFO'
            }),
            'standard': ValidationProfile('standard', {
                'fail_on_warning': False,
                'min_score': 0.7,
                'required_checks': [
                    'naming_convention',
                    'musical_validity',
                    'no_duplicates',
                    'range_appropriate',
                    'implementation_viable',
                    'theory_consistency'
                ],
                'severity_threshold': 'WARNING'
            }),
            'lenient': ValidationProfile('lenient', {
                'fail_on_warning': False,
                'min_score': 0.5,
                'required_checks': [
                    'naming_convention',
                    'no_duplicates',
                    'range_appropriate'
                ],
                'severity_threshold': 'ERROR'
            }),
            'research': ValidationProfile('research', {
                'fail_on_warning': False,
                'min_score': 0.3,
                'required_checks': [
                    'naming_convention',
                    'range_appropriate'
                ],
                'severity_threshold': 'ERROR',
                'description': 'Minimal validation for experimental parameters'
            }),
            'production': ValidationProfile('production', {
                'fail_on_warning': True,
                'min_score': 0.9,
                'required_checks': [
                    'naming_convention',
                    'musical_validity',
                    'no_duplicates',
                    'range_appropriate',
                    'implementation_viable',
                    'test_coverage',
                    'theory_consistency'
                ],
                'severity_threshold': 'INFO',
                'description': 'Maximum validation for production deployment'
            })
        }

    def get_profile(self, name: str) -> Optional[ValidationProfile]:
        """Get validation profile by name"""
        return self.profiles.get(name)

    def add_profile(self, name: str, config: Dict[str, Any]):
        """Add custom validation profile"""
        self.profiles[name] = ValidationProfile(name, config)

    def list_profiles(self) -> List[str]:
        """List available profile names"""
        return list(self.profiles.keys())


class ExampleParameterLibrary:
    """
    Library of example parameter proposals for testing and learning.
    """

    @staticmethod
    def get_example_valid_parameter() -> Dict[str, Any]:
        """Get a fully valid example parameter"""
        return {
            'name': 'harmony.extensions.ninth_probability',
            'type': 'PROBABILITY',
            'range': [0.0, 1.0],
            'default': 0.7,
            'description': 'Probability of adding 9th extensions to seventh chords for richer harmonic color and sophistication',
            'musical_context': 'In jazz and contemporary music, 9th extensions are commonly added to dominant and major seventh chords to create sophisticated harmonies. This parameter controls how frequently these extensions appear in generated harmony.',
            'implementation_strategy': 'When building a chord voicing in HarmonyModule, check this probability value. If a random value is less than this probability, add the 9th (root + 14 semitones) to the chord tones. Ensure the 9th does not create undesirable dissonance with the melody note.',
            'generator_integration_points': [
                'HarmonyModule._build_chord_voicing',
                'HarmonyModule._add_extensions',
                'ChordEngine.generate_voicing'
            ],
            'test_cases': [
                {
                    'value': 0.0,
                    'expected': 'No 9th extensions added to any seventh chords'
                },
                {
                    'value': 0.5,
                    'expected': '9th extensions added to approximately 50% of eligible seventh chords'
                },
                {
                    'value': 1.0,
                    'expected': '9th extensions added to all eligible seventh chords'
                }
            ],
            'example_values': {
                'jazz': 0.9,
                'bebop': 0.95,
                'fusion': 0.85,
                'classical': 0.3,
                'pop': 0.5,
                'rock': 0.2,
                'blues': 0.4,
                'electronic': 0.6
            },
            'affected_features': [
                'chord_complexity',
                'harmonic_richness',
                'extension_count',
                'dissonance_level'
            ]
        }

    @staticmethod
    def get_example_invalid_naming() -> Dict[str, Any]:
        """Get example with invalid naming"""
        param = ExampleParameterLibrary.get_example_valid_parameter()
        param['name'] = 'BadParameterName'  # Invalid format
        return param

    @staticmethod
    def get_example_invalid_range() -> Dict[str, Any]:
        """Get example with invalid range"""
        param = ExampleParameterLibrary.get_example_valid_parameter()
        param['range'] = [1.0, 0.0]  # Wrong order
        return param

    @staticmethod
    def get_example_missing_tests() -> Dict[str, Any]:
        """Get example with insufficient test coverage"""
        param = ExampleParameterLibrary.get_example_valid_parameter()
        param['test_cases'] = []  # No tests
        return param

    @staticmethod
    def get_example_categorical() -> Dict[str, Any]:
        """Get valid categorical parameter example"""
        return {
            'name': 'harmony.voicing.type',
            'type': 'CATEGORICAL',
            'range': ['close', 'spread', 'drop2', 'drop3', 'quartal', 'rootless'],
            'default': 'close',
            'description': 'Type of chord voicing to use, determining how chord tones are distributed across the range',
            'musical_context': 'Different voicing types create distinct sonic textures. Close voicings have tones within an octave, spread voicings expand beyond an octave, drop voicings strategically lower specific tones, and quartal voicings stack fourths instead of thirds.',
            'implementation_strategy': 'Use a switch/case statement on the voicing type to determine the algorithm for arranging chord tones. Each voicing type has specific rules for tone spacing and distribution.',
            'generator_integration_points': [
                'HarmonyModule._voice_chord',
                'VoicingEngine.apply_voicing_type'
            ],
            'test_cases': [
                {'value': 'close', 'expected': 'Tones within one octave'},
                {'value': 'spread', 'expected': 'Tones spread across multiple octaves'},
                {'value': 'drop2', 'expected': 'Second-highest tone dropped one octave'},
                {'value': 'drop3', 'expected': 'Third-highest tone dropped one octave'},
                {'value': 'quartal', 'expected': 'Tones stacked in fourths'},
                {'value': 'rootless', 'expected': 'Root tone omitted from voicing'}
            ],
            'example_values': {
                'jazz': 'drop2',
                'classical': 'close',
                'pop': 'spread',
                'rock': 'close',
                'fusion': 'quartal'
            },
            'affected_features': [
                'voicing_density',
                'chord_spread',
                'harmonic_texture'
            ]
        }

    @staticmethod
    def get_all_examples() -> Dict[str, Dict[str, Any]]:
        """Get all example parameters"""
        return {
            'valid_parameter': ExampleParameterLibrary.get_example_valid_parameter(),
            'invalid_naming': ExampleParameterLibrary.get_example_invalid_naming(),
            'invalid_range': ExampleParameterLibrary.get_example_invalid_range(),
            'missing_tests': ExampleParameterLibrary.get_example_missing_tests(),
            'categorical': ExampleParameterLibrary.get_example_categorical()
        }


# ============================================================================
# ADVANCED VALIDATION FEATURES
# ============================================================================

class ParameterRelationshipAnalyzer:
    """
    Analyzes relationships between parameters to detect conflicts,
    dependencies, and optimization opportunities.
    """

    def __init__(self, registry: UniversalParameterRegistry):
        self.registry = registry
        self.theory_kb = MusicTheoryKnowledgeBase()

    def analyze_dependencies(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze potential dependencies between proposed parameter
        and existing parameters.
        """
        name = proposal.get('name', '')
        domain = name.split('.')[0] if '.' in name else ''

        dependencies = {
            'required': [],  # Parameters this depends on
            'conflicts': [],  # Parameters that conflict
            'related': [],   # Related parameters to consider
            'suggestions': []
        }

        # Find parameters in same domain
        for param_name, param_def in self.registry.parameters.items():
            if param_name.startswith(domain):
                # Check for potential dependencies
                if self._might_depend_on(proposal, param_def):
                    dependencies['required'].append(param_name)

                # Check for conflicts
                if self._might_conflict_with(proposal, param_def):
                    dependencies['conflicts'].append(param_name)

                # Related parameters
                if self._is_related_to(proposal, param_def):
                    dependencies['related'].append(param_name)

        return dependencies

    def _might_depend_on(self, proposal: Dict[str, Any], param: ParameterDefinition) -> bool:
        """Check if proposal might depend on existing parameter"""
        proposal_name = proposal.get('name', '').lower()
        param_name = param.name.lower()

        # Check for hierarchical dependencies
        # e.g., harmony.extensions.ninth depends on harmony.chord.type
        proposal_parts = proposal_name.split('.')
        param_parts = param_name.split('.')

        # Same module, different specificity
        if len(proposal_parts) > 2 and len(param_parts) > 2:
            if proposal_parts[:2] == param_parts[:2]:
                # More specific parameter depends on general one
                if len(proposal_parts) > len(param_parts):
                    return True

        return False

    def _might_conflict_with(self, proposal: Dict[str, Any], param: ParameterDefinition) -> bool:
        """Check if proposal might conflict with existing parameter"""
        proposal_desc = proposal.get('description', '').lower()
        param_desc = param.description.lower()

        # Check for contradictory terms
        for term1, conflicts in self.theory_kb.contradictions.items():
            if term1 in proposal_desc:
                for term2 in conflicts:
                    if term2 in param_desc:
                        return True

        return False

    def _is_related_to(self, proposal: Dict[str, Any], param: ParameterDefinition) -> bool:
        """Check if parameters are related"""
        proposal_name = proposal.get('name', '').lower()
        param_name = param.name.lower()

        # Same domain and module
        proposal_parts = proposal_name.split('.')
        param_parts = param_name.split('.')

        if len(proposal_parts) >= 2 and len(param_parts) >= 2:
            if proposal_parts[:2] == param_parts[:2]:
                return True

        return False

    def generate_dependency_graph(self, params: List[str]) -> Dict[str, List[str]]:
        """Generate dependency graph for list of parameters"""
        graph = defaultdict(list)

        for param_name in params:
            param = self.registry.get(param_name)
            if param and param.depends_on:
                graph[param_name] = param.depends_on

        return dict(graph)


class ValidationReportGenerator:
    """
    Generates comprehensive validation reports in multiple formats.
    """

    def __init__(self):
        pass

    def generate_markdown_report(self, results: List[ParameterValidationResult],
                                  title: str = "Parameter Validation Report") -> str:
        """Generate markdown validation report"""
        lines = []

        # Header
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"Generated: {self._get_timestamp()}")
        lines.append("")

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.valid)
        failed = total - passed
        avg_score = sum(r.score for r in results) / total if total > 0 else 0.0

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Parameters**: {total}")
        lines.append(f"- **Passed**: {passed} ({passed/total*100:.1f}%)")
        lines.append(f"- **Failed**: {failed} ({failed/total*100:.1f}%)")
        lines.append(f"- **Average Score**: {avg_score:.2%}")
        lines.append("")

        # Detailed results
        lines.append("## Detailed Results")
        lines.append("")

        for i, result in enumerate(results, 1):
            status = "✅ PASS" if result.valid else "❌ FAIL"
            lines.append(f"### {i}. Validation Result - {status}")
            lines.append("")
            lines.append(f"**Score**: {result.score:.2%}")
            lines.append("")

            if result.errors:
                lines.append("**Errors**:")
                for error in result.errors:
                    lines.append(f"- {error}")
                lines.append("")

            if result.warnings:
                lines.append("**Warnings**:")
                for warning in result.warnings:
                    lines.append(f"- {warning}")
                lines.append("")

            if result.suggestions:
                lines.append("**Suggestions**:")
                for suggestion in result.suggestions:
                    lines.append(f"- {suggestion}")
                lines.append("")

            # Check details
            lines.append("**Validation Checks**:")
            lines.append("")
            for check_name, check in result.checks.items():
                status_icon = "✅" if check.pass_status else "❌"
                lines.append(f"- {status_icon} **{check_name}**: {check.message}")
            lines.append("")

        return "\n".join(lines)

    def generate_html_report(self, results: List[ParameterValidationResult],
                            title: str = "Parameter Validation Report") -> str:
        """Generate HTML validation report"""
        total = len(results)
        passed = sum(1 for r in results if r.valid)
        failed = total - passed
        avg_score = sum(r.score for r in results) / total if total > 0 else 0.0

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .summary {{ background: #e8f5e9; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .summary-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .result {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }}
        .result.pass {{ border-left: 5px solid #4CAF50; }}
        .result.fail {{ border-left: 5px solid #f44336; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .errors {{ color: #f44336; }}
        .warnings {{ color: #ff9800; }}
        .suggestions {{ color: #2196F3; }}
        .check {{ padding: 5px 0; }}
        .check.pass {{ color: #4CAF50; }}
        .check.fail {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p><em>Generated: {self._get_timestamp()}</em></p>

        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-stats">
                <div class="stat">
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{passed}</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{failed}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{avg_score:.0%}</div>
                    <div class="stat-label">Avg Score</div>
                </div>
            </div>
        </div>

        <h2>Detailed Results</h2>
"""

        for i, result in enumerate(results, 1):
            status_class = "pass" if result.valid else "fail"
            status_text = "✅ PASS" if result.valid else "❌ FAIL"

            html += f"""
        <div class="result {status_class}">
            <h3>Result {i} - {status_text}</h3>
            <div class="score">Score: {result.score:.0%}</div>
"""

            if result.errors:
                html += '<div class="errors"><h4>Errors</h4><ul>'
                for error in result.errors:
                    html += f'<li>{error}</li>'
                html += '</ul></div>'

            if result.warnings:
                html += '<div class="warnings"><h4>Warnings</h4><ul>'
                for warning in result.warnings:
                    html += f'<li>{warning}</li>'
                html += '</ul></div>'

            if result.suggestions:
                html += '<div class="suggestions"><h4>Suggestions</h4><ul>'
                for suggestion in result.suggestions:
                    html += f'<li>{suggestion}</li>'
                html += '</ul></div>'

            html += '<h4>Validation Checks</h4>'
            for check_name, check in result.checks.items():
                check_class = "pass" if check.pass_status else "fail"
                icon = "✅" if check.pass_status else "❌"
                html += f'<div class="check {check_class}">{icon} <strong>{check_name}</strong>: {check.message}</div>'

            html += '</div>'

        html += """
    </div>
</body>
</html>
"""
        return html

    def generate_json_report(self, results: List[ParameterValidationResult]) -> str:
        """Generate JSON validation report"""
        report = {
            'timestamp': self._get_timestamp(),
            'summary': {
                'total': len(results),
                'passed': sum(1 for r in results if r.valid),
                'failed': sum(1 for r in results if not r.valid),
                'average_score': sum(r.score for r in results) / len(results) if results else 0.0
            },
            'results': [r.to_dict() for r in results]
        }

        return json.dumps(report, indent=2)

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ValidationHistory:
    """
    Track validation history for monitoring trends and improvements.
    """

    def __init__(self, history_file: Optional[str] = None):
        self.history_file = history_file or "/tmp/validation_history.json"
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load validation history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_history(self):
        """Save validation history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save validation history: {e}")

    def record_validation(self, result: ParameterValidationResult, param_name: str):
        """Record a validation result"""
        from datetime import datetime

        record = {
            'timestamp': datetime.now().isoformat(),
            'parameter_name': param_name,
            'valid': result.valid,
            'score': result.score,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings)
        }

        self.history.append(record)
        self._save_history()

    def get_statistics(self) -> Dict[str, Any]:
        """Get historical statistics"""
        if not self.history:
            return {
                'total_validations': 0,
                'success_rate': 0.0,
                'average_score': 0.0
            }

        total = len(self.history)
        passed = sum(1 for h in self.history if h['valid'])
        avg_score = sum(h['score'] for h in self.history) / total

        return {
            'total_validations': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total,
            'average_score': avg_score,
            'total_errors': sum(h['error_count'] for h in self.history),
            'total_warnings': sum(h['warning_count'] for h in self.history)
        }

    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """Get recent trend"""
        if len(self.history) < window:
            window = len(self.history)

        if window == 0:
            return {'trend': 'no_data'}

        recent = self.history[-window:]
        recent_passed = sum(1 for h in recent if h['valid'])
        recent_score = sum(h['score'] for h in recent) / window

        # Compare to previous window
        if len(self.history) >= window * 2:
            previous = self.history[-window*2:-window]
            previous_passed = sum(1 for h in previous if h['valid'])
            previous_score = sum(h['score'] for h in previous) / window

            trend = 'improving' if recent_score > previous_score else 'declining'
        else:
            trend = 'stable'

        return {
            'window_size': window,
            'recent_success_rate': recent_passed / window,
            'recent_average_score': recent_score,
            'trend': trend
        }


class GenreSpecificValidator:
    """
    Validates parameters against genre-specific constraints and conventions.
    """

    def __init__(self):
        self.genre_rules = self._load_genre_rules()

    def _load_genre_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load genre-specific validation rules"""
        return {
            'jazz': {
                'required_features': ['swing', 'chord_extensions', 'syncopation'],
                'forbidden_features': ['power_chord', 'heavy_distortion'],
                'typical_ranges': {
                    'swing_ratio': (0.6, 0.75),
                    'syncopation_prob': (0.3, 0.8),
                    'chord_extensions_prob': (0.7, 1.0)
                },
                'recommended_parameters': [
                    'harmony.extensions.use_9ths',
                    'rhythm.swing.amount',
                    'melody.chromaticism.amount'
                ]
            },
            'classical': {
                'required_features': ['voice_leading', 'counterpoint', 'dynamics'],
                'forbidden_features': ['swing', 'power_chord', 'distortion'],
                'typical_ranges': {
                    'voice_leading_smoothness': (0.8, 1.0),
                    'parallel_motion_tolerance': (0.0, 0.1)
                },
                'style_constraints': {
                    'baroque': ['ornaments', 'figured_bass'],
                    'romantic': ['chromaticism', 'rubato']
                }
            },
            'rock': {
                'required_features': ['power_chords', 'straight_rhythm'],
                'typical_ranges': {
                    'power_chord_prob': (0.5, 0.9),
                    'distortion': (0.3, 0.8)
                }
            },
            'electronic': {
                'required_features': ['quantization', 'synthesis'],
                'typical_ranges': {
                    'quantize_strength': (0.9, 1.0),
                    'swing_amount': (0.5, 0.6)
                }
            },
            'blues': {
                'required_features': ['blue_notes', 'shuffle', 'pentatonic'],
                'typical_ranges': {
                    'shuffle_amount': (0.6, 0.75),
                    'blue_note_prob': (0.3, 0.7)
                }
            }
        }

    def validate_genre_compatibility(self, proposal: Dict[str, Any],
                                     genre: str) -> ValidationCheck:
        """Validate parameter compatibility with genre"""
        if genre not in self.genre_rules:
            return ValidationCheck(
                pass_status=True,
                message=f'Genre {genre} not in validation rules',
                severity='INFO'
            )

        rules = self.genre_rules[genre]
        name = proposal.get('name', '').lower()
        description = proposal.get('description', '').lower()

        # Check forbidden features
        for forbidden in rules.get('forbidden_features', []):
            if forbidden in name or forbidden in description:
                return ValidationCheck(
                    pass_status=False,
                    message=f"Parameter uses '{forbidden}' which is not typical for {genre}",
                    severity='WARNING'
                )

        # Check typical ranges
        param_range = proposal.get('range')
        for param_pattern, (min_expected, max_expected) in rules.get('typical_ranges', {}).items():
            if param_pattern in name:
                if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                    min_val, max_val = param_range
                    if not (min_expected <= min_val and max_val <= max_expected):
                        return ValidationCheck(
                            pass_status=False,
                            message=f"Range [{min_val}, {max_val}] unusual for {genre} (typical: [{min_expected}, {max_expected}])",
                            severity='INFO',
                            details={'typical_range': [min_expected, max_expected]}
                        )

        return ValidationCheck(
            pass_status=True,
            message=f'Compatible with {genre} genre',
            severity='INFO'
        )


class AdvancedCodeQualityChecker:
    """
    Advanced code quality checks beyond basic syntax validation.
    """

    def __init__(self):
        self.quality_rules = self._load_quality_rules()

    def _load_quality_rules(self) -> Dict[str, Any]:
        """Load code quality rules"""
        return {
            'max_function_length': 50,
            'max_complexity': 10,
            'required_docstring_sections': ['Args', 'Returns'],
            'naming_conventions': {
                'function': r'^[a-z_][a-z0-9_]*$',
                'class': r'^[A-Z][a-zA-Z0-9]*$',
                'constant': r'^[A-Z_][A-Z0-9_]*$'
            }
        }

    def check_function_length(self, code: str) -> ValidationCheck:
        """Check function lengths"""
        try:
            tree = ast.parse(code)
            max_length = self.quality_rules['max_function_length']

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_length = node.end_lineno - node.lineno
                    if func_length > max_length:
                        return ValidationCheck(
                            pass_status=False,
                            message=f"Function '{node.name}' is too long ({func_length} lines > {max_length})",
                            severity='WARNING',
                            details={'function': node.name, 'length': func_length}
                        )

            return ValidationCheck(
                pass_status=True,
                message='Function lengths appropriate',
                severity='INFO'
            )
        except:
            return ValidationCheck(
                pass_status=True,
                message='Could not check function length',
                severity='INFO'
            )

    def check_complexity(self, code: str) -> ValidationCheck:
        """Check cyclomatic complexity"""
        try:
            tree = ast.parse(code)
            max_complexity = self.quality_rules['max_complexity']

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_complexity(node)
                    if complexity > max_complexity:
                        return ValidationCheck(
                            pass_status=False,
                            message=f"Function '{node.name}' too complex (complexity={complexity} > {max_complexity})",
                            severity='WARNING',
                            details={'function': node.name, 'complexity': complexity}
                        )

            return ValidationCheck(
                pass_status=True,
                message='Code complexity acceptable',
                severity='INFO'
            )
        except:
            return ValidationCheck(
                pass_status=True,
                message='Could not check complexity',
                severity='INFO'
            )

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def check_docstring_quality(self, code: str) -> ValidationCheck:
        """Check docstring quality"""
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if not docstring:
                        continue

                    # Check for required sections
                    for section in self.quality_rules['required_docstring_sections']:
                        if section not in docstring:
                            return ValidationCheck(
                                pass_status=False,
                                message=f"Function '{node.name}' docstring missing '{section}' section",
                                severity='INFO',
                                details={'function': node.name, 'missing_section': section}
                            )

            return ValidationCheck(
                pass_status=True,
                message='Docstring quality good',
                severity='INFO'
            )
        except:
            return ValidationCheck(
                pass_status=True,
                message='Could not check docstring quality',
                severity='INFO'
            )


class BatchValidator:
    """
    Batch validation with parallel processing and progress tracking.
    """

    def __init__(self, validator: MusicalValidator):
        self.validator = validator
        self.report_generator = ValidationReportGenerator()

    def validate_batch(self, proposals: List[Dict[str, Any]],
                      show_progress: bool = True) -> List[ParameterValidationResult]:
        """Validate multiple proposals"""
        results = []

        for i, proposal in enumerate(proposals):
            if show_progress:
                print(f"Validating {i+1}/{len(proposals)}: {proposal.get('name', 'unknown')}")

            result = self.validator.validate_parameter(proposal)
            results.append(result)

        return results

    def generate_batch_report(self, results: List[ParameterValidationResult],
                             format: str = 'markdown',
                             output_file: Optional[str] = None) -> str:
        """Generate batch validation report"""
        if format == 'markdown':
            report = self.report_generator.generate_markdown_report(results)
        elif format == 'html':
            report = self.report_generator.generate_html_report(results)
        elif format == 'json':
            report = self.report_generator.generate_json_report(results)
        else:
            raise ValueError(f"Unknown report format: {format}")

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)

        return report


# ============================================================================
# MAIN - TESTING & DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MUSICAL VALIDATOR - Agent 13")
    print("=" * 80)
    print()

    # Create validator
    validator = create_validator(enable_llm=False)  # Disable LLM for demo

    print(f"✓ Validator initialized")
    print(f"  - LLM enabled: {validator.enable_llm}")
    print(f"  - Existing parameters: {len(validator.existing_params)}")
    print()

    # Example 1: Valid parameter
    print("-" * 80)
    print("TEST 1: Valid Parameter")
    print("-" * 80)

    valid_proposal = {
        'name': 'harmony.extensions.use_9ths_probability',
        'type': 'PROBABILITY',
        'range': [0.0, 1.0],
        'default': 0.7,
        'description': 'Probability of adding 9th extensions to seventh chords, creating richer harmonic color',
        'musical_context': 'In jazz and contemporary music, 9th extensions are commonly added to dominant and major seventh chords to create color and sophistication. This parameter controls how frequently these extensions appear.',
        'implementation_strategy': 'When building a chord, check this probability. If random value < probability, add the 9th to available chord tones. Must ensure 9th does not conflict with melody note.',
        'generator_integration_points': ['HarmonyModule._build_chord_voicing', 'HarmonyModule._add_extensions'],
        'test_cases': [
            {'value': 0.0, 'expected': 'No 9ths added'},
            {'value': 1.0, 'expected': '9ths added to all eligible chords'},
            {'value': 0.5, 'expected': '9ths added to ~50% of chords'}
        ],
        'example_values': {
            'jazz': 0.9,
            'classical': 0.3,
            'pop': 0.5,
            'rock': 0.2
        }
    }

    result1 = validator.validate_parameter(valid_proposal)
    print(result1.summary())

    # Example 2: Invalid parameter (naming issue)
    print("-" * 80)
    print("TEST 2: Invalid Parameter (Naming Issue)")
    print("-" * 80)

    invalid_proposal = {
        'name': 'BadName',  # Missing domain.module.parameter structure
        'type': 'CONTINUOUS',
        'range': [0.0, 1.0],
        'default': 0.5,
        'description': 'A bad parameter',
        'musical_context': 'Test',
        'implementation_strategy': 'Use it somehow',
        'generator_integration_points': [],
        'test_cases': []
    }

    result2 = validator.validate_parameter(invalid_proposal)
    print(result2.summary())

    # Example 3: Code validation
    print("-" * 80)
    print("TEST 3: Code Validation")
    print("-" * 80)

    good_code = {
        'generator_modifications': {
            'harmony_module.py': '''
def apply_extensions(self, chord, params):
    """Apply chord extensions based on parameters"""
    use_9ths = params.get('harmony.extensions.use_9ths_probability', 0.7)

    if random.random() < use_9ths:
        # Add 9th if appropriate
        ninth = chord.root + 14
        if 0 <= ninth <= 127:
            chord.add_note(ninth)

    return chord
'''
        },
        'test_code': '''
def test_9ths_extension():
    """Test 9th extension parameter"""
    params = {'harmony.extensions.use_9ths_probability': 1.0}
    # Test implementation
    assert True
'''
    }

    code_result = validator.validate_code(good_code, valid_proposal)
    print(f"Code Validation: {'✅ VALID' if code_result.valid else '❌ INVALID'}")
    for check_name, check in code_result.checks.items():
        status = "✅" if check.pass_status else "❌"
        print(f"  {status} {check_name}: {check.message}")
    print()

    # Statistics
    print("-" * 80)
    print("VALIDATION STATISTICS")
    print("-" * 80)
    stats = validator.get_statistics()
    print(f"Total validations: {stats['total_validations']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Pass rate: {stats['pass_rate']:.1%}")
    print()

    print("=" * 80)
    print("✓ Musical Validator demonstration complete")
    print("=" * 80)
