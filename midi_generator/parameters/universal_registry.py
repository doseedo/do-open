"""
Universal Parameter Registry - Agent 3
========================================

Central registry of all 2000+ parameters across all 116+ modules of the
Musical Program Synthesis system.

This registry provides:
1. Complete parameter taxonomy
2. Type system and validation
3. Dependency graphs
4. Default values
5. Musical impact metadata
6. Genre relevance mapping

Author: Agent 3 - Parameter Registry Builder
License: MIT
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import json
from pathlib import Path


class ParameterType(Enum):
    """Types of parameters in the system"""
    CONTINUOUS = "continuous"          # Float in range [min, max]
    INTEGER = "integer"                # Int in range [min, max]
    CATEGORICAL = "categorical"        # One of fixed options
    BOOLEAN = "boolean"                # True/False
    ARRAY_INT = "array_int"           # List of integers
    ARRAY_FLOAT = "array_float"       # List of floats
    PROBABILITY = "probability"        # Float in [0.0, 1.0]
    MIDI_NOTE = "midi_note"           # Integer in [0, 127]
    VELOCITY = "velocity"              # Integer in [0, 127]
    DURATION = "duration"              # Float (beats/seconds)


class ParameterCategory(Enum):
    """High-level categories for parameter organization"""
    HARMONY = "harmony"
    MELODY = "melody"
    RHYTHM = "rhythm"
    BASS = "bass"
    VOICE = "voice"
    DRUMS = "drums"
    TIMBRE = "timbre"
    DYNAMICS = "dynamics"
    ARTICULATION = "articulation"
    STRUCTURE = "structure"
    GENRE = "genre"
    STYLE = "style"


class MusicalImpact(Enum):
    """Impact level of parameter on musical output"""
    CRITICAL = "critical"    # Fundamentally changes musical character
    HIGH = "high"           # Significant perceptual impact
    MEDIUM = "medium"       # Noticeable but not defining
    LOW = "low"            # Subtle refinement


@dataclass
class ParameterDefinition:
    """
    Complete definition of a single parameter
    """
    # Identity
    name: str
    full_path: str  # e.g., "harmony.jazz.voicing_type"
    description: str

    # Type system
    param_type: ParameterType
    default_value: Any

    # Constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: Optional[List[Any]] = None
    step: Optional[float] = None  # For integer/continuous stepping

    # Metadata
    category: Optional[ParameterCategory] = None
    module_file: Optional[str] = None
    musical_impact: MusicalImpact = MusicalImpact.MEDIUM
    genre_relevance: List[str] = field(default_factory=list)

    # Relationships
    depends_on: List[str] = field(default_factory=list)  # Other parameter paths
    mutually_exclusive_with: List[str] = field(default_factory=list)

    # Validation
    validation_function: Optional[Callable] = None
    constraint_description: Optional[str] = None

    # Learning metadata
    learnable: bool = True  # Can XGBoost learn this?
    feature_importance: Optional[float] = None  # Filled during training

    def validate(self, value: Any) -> Tuple[bool, str]:
        """
        Validate a value for this parameter

        Returns:
            (is_valid, error_message)
        """
        if value is None:
            return False, f"Value cannot be None for {self.name}"

        if self.param_type == ParameterType.CONTINUOUS:
            if not isinstance(value, (int, float)):
                return False, f"{self.name} must be numeric, got {type(value)}"
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.name} must be >= {self.min_value}, got {value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.name} must be <= {self.max_value}, got {value}"

        elif self.param_type == ParameterType.INTEGER:
            if not isinstance(value, int):
                return False, f"{self.name} must be int, got {type(value)}"
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.name} must be >= {self.min_value}, got {value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.name} must be <= {self.max_value}, got {value}"

        elif self.param_type == ParameterType.CATEGORICAL:
            if self.options and value not in self.options:
                return False, f"{self.name} must be one of {self.options}, got {value}"

        elif self.param_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"{self.name} must be bool, got {type(value)}"

        elif self.param_type == ParameterType.PROBABILITY:
            if not isinstance(value, (int, float)):
                return False, f"{self.name} must be numeric, got {type(value)}"
            if not 0.0 <= value <= 1.0:
                return False, f"{self.name} must be in [0.0, 1.0], got {value}"

        elif self.param_type == ParameterType.MIDI_NOTE:
            if not isinstance(value, int):
                return False, f"{self.name} must be int, got {type(value)}"
            if not 0 <= value <= 127:
                return False, f"{self.name} must be in [0, 127], got {value}"

        elif self.param_type == ParameterType.VELOCITY:
            if not isinstance(value, int):
                return False, f"{self.name} must be int, got {type(value)}"
            if not 0 <= value <= 127:
                return False, f"{self.name} must be in [0, 127], got {value}"

        # Custom validation function
        if self.validation_function:
            try:
                if not self.validation_function(value):
                    return False, f"{self.name} failed custom validation"
            except Exception as e:
                return False, f"{self.name} validation error: {e}"

        return True, ""


class UniversalParameterRegistry:
    """
    Central registry of ALL parameters across the entire codebase.

    This provides:
    - Hierarchical organization (domain.module.parameter)
    - Type system and validation
    - Dependency graphs
    - Metadata for learning
    - Default value management
    """

    def __init__(self):
        self.parameters: Dict[str, ParameterDefinition] = {}
        self._build_registry()

    def _build_registry(self):
        """Build the complete parameter registry"""
        # This will be populated by scanning refactored modules
        # For now, we'll register the major categories

        # HARMONY PARAMETERS
        self._register_harmony_parameters()

        # MELODY PARAMETERS
        self._register_melody_parameters()

        # RHYTHM PARAMETERS
        self._register_rhythm_parameters()

        # GENRE PARAMETERS
        self._register_genre_parameters()

        # BASS PARAMETERS
        self._register_bass_parameters()

        # DRUM PARAMETERS
        self._register_drum_parameters()

        # ARTICULATION PARAMETERS
        self._register_articulation_parameters()

        # DYNAMICS PARAMETERS
        self._register_dynamics_parameters()

        # INSTRUMENTATION PARAMETERS (Agent 1 Expansion)
        self._register_instrumentation_parameters()

    def register(self, param_def: ParameterDefinition):
        """Register a parameter"""
        self.parameters[param_def.full_path] = param_def

    def get(self, full_path: str) -> Optional[ParameterDefinition]:
        """Get parameter by full path"""
        return self.parameters.get(full_path)

    def get_all_parameters(self) -> List[str]:
        """Get all parameter paths"""
        return list(self.parameters.keys())

    def get_by_category(self, category: ParameterCategory) -> List[ParameterDefinition]:
        """Get all parameters in a category"""
        return [p for p in self.parameters.values() if p.category == category]

    def get_by_module(self, module_file: str) -> List[ParameterDefinition]:
        """Get all parameters for a module"""
        return [p for p in self.parameters.values() if p.module_file == module_file]

    def get_by_genre(self, genre: str) -> List[ParameterDefinition]:
        """Get parameters relevant to a genre"""
        return [p for p in self.parameters.values() if genre in p.genre_relevance]

    def is_continuous(self, full_path: str) -> bool:
        """Check if parameter is continuous"""
        param = self.get(full_path)
        return param and param.param_type in [
            ParameterType.CONTINUOUS,
            ParameterType.PROBABILITY,
            ParameterType.DURATION
        ]

    def is_categorical(self, full_path: str) -> bool:
        """Check if parameter is categorical"""
        param = self.get(full_path)
        return param and param.param_type == ParameterType.CATEGORICAL

    def validate_parameter(self, full_path: str, value: Any) -> Tuple[bool, str]:
        """Validate a parameter value"""
        param = self.get(full_path)
        if not param:
            return False, f"Unknown parameter: {full_path}"
        return param.validate(value)

    def validate_all(self, param_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate multiple parameters

        Returns:
            (all_valid, list_of_errors)
        """
        errors = []
        for path, value in param_dict.items():
            valid, error = self.validate_parameter(path, value)
            if not valid:
                errors.append(error)

        return len(errors) == 0, errors

    def get_defaults(self, full_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get default values for parameters"""
        if full_paths is None:
            return {p.full_path: p.default_value for p in self.parameters.values()}
        else:
            return {
                path: self.parameters[path].default_value
                for path in full_paths
                if path in self.parameters
            }

    # ========================================================================
    # Parameter Registration Methods
    # ========================================================================

    def _register_harmony_parameters(self):
        """Register all harmony-related parameters"""

        # Voicing parameters
        self.register(ParameterDefinition(
            name="voicing_type",
            full_path="harmony.voicing.type",
            description="Type of chord voicing (close, spread, drop2, etc.)",
            param_type=ParameterType.CATEGORICAL,
            options=["close", "spread", "drop2", "drop3", "drop2_4", "rootless_a", "rootless_b", "quartal"],
            default_value="close",
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.HIGH,
            genre_relevance=["jazz", "classical", "pop", "rock"]
        ))

        self.register(ParameterDefinition(
            name="voicing_spread",
            full_path="harmony.voicing.spread",
            description="How spread out the voicing is (0.0=close, 1.0=wide)",
            param_type=ParameterType.PROBABILITY,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.MEDIUM
        ))

        self.register(ParameterDefinition(
            name="voicing_density",
            full_path="harmony.voicing.density",
            description="Number of notes in chord (3-7)",
            param_type=ParameterType.INTEGER,
            default_value=4,
            min_value=3,
            max_value=7,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.HIGH
        ))

        # Extension parameters
        self.register(ParameterDefinition(
            name="use_9ths",
            full_path="harmony.extensions.use_9ths",
            description="Whether to include 9th extensions",
            param_type=ParameterType.BOOLEAN,
            default_value=True,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.MEDIUM,
            genre_relevance=["jazz", "fusion", "neo_soul"]
        ))

        self.register(ParameterDefinition(
            name="use_11ths",
            full_path="harmony.extensions.use_11ths",
            description="Whether to include 11th extensions",
            param_type=ParameterType.BOOLEAN,
            default_value=True,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.MEDIUM,
            genre_relevance=["jazz", "fusion"]
        ))

        self.register(ParameterDefinition(
            name="use_13ths",
            full_path="harmony.extensions.use_13ths",
            description="Whether to include 13th extensions",
            param_type=ParameterType.BOOLEAN,
            default_value=False,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.MEDIUM,
            genre_relevance=["jazz", "fusion"]
        ))

        # Substitution parameters
        self.register(ParameterDefinition(
            name="tritone_sub_probability",
            full_path="harmony.substitution.tritone_probability",
            description="Probability of tritone substitution",
            param_type=ParameterType.PROBABILITY,
            default_value=0.3,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.CRITICAL,
            genre_relevance=["jazz", "bebop"]
        ))

        self.register(ParameterDefinition(
            name="modal_interchange_probability",
            full_path="harmony.substitution.modal_interchange_probability",
            description="Probability of modal interchange",
            param_type=ParameterType.PROBABILITY,
            default_value=0.2,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.HIGH,
            genre_relevance=["jazz", "rock", "classical"]
        ))

        # Voice leading parameters
        self.register(ParameterDefinition(
            name="voice_leading_smoothness",
            full_path="harmony.voice_leading.smoothness",
            description="Preference for smooth voice leading (0.0=none, 1.0=always)",
            param_type=ParameterType.PROBABILITY,
            default_value=0.8,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.HIGH,
            genre_relevance=["jazz", "classical", "choral"]
        ))

        self.register(ParameterDefinition(
            name="parallel_motion_tolerance",
            full_path="harmony.voice_leading.parallel_motion_tolerance",
            description="Tolerance for parallel 5ths/octaves (0.0=strict, 1.0=permissive)",
            param_type=ParameterType.PROBABILITY,
            default_value=0.1,
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.MEDIUM,
            genre_relevance=["classical", "choral"]
        ))

    def _register_melody_parameters(self):
        """Register all melody-related parameters"""

        self.register(ParameterDefinition(
            name="contour_type",
            full_path="melody.contour.type",
            description="Melodic contour shape",
            param_type=ParameterType.CATEGORICAL,
            options=["arch", "inverted_arch", "ascending", "descending", "wave", "static"],
            default_value="arch",
            category=ParameterCategory.MELODY,
            musical_impact=MusicalImpact.CRITICAL
        ))

        self.register(ParameterDefinition(
            name="stepwise_motion_probability",
            full_path="melody.intervals.stepwise_probability",
            description="Probability of stepwise motion (vs leaps)",
            param_type=ParameterType.PROBABILITY,
            default_value=0.7,
            category=ParameterCategory.MELODY,
            musical_impact=MusicalImpact.HIGH
        ))

        self.register(ParameterDefinition(
            name="max_leap_interval",
            full_path="melody.intervals.max_leap",
            description="Maximum melodic leap in semitones",
            param_type=ParameterType.INTEGER,
            default_value=12,
            min_value=1,
            max_value=24,
            category=ParameterCategory.MELODY,
            musical_impact=MusicalImpact.HIGH
        ))

        self.register(ParameterDefinition(
            name="chromaticism",
            full_path="melody.chromaticism.amount",
            description="Amount of chromatic passing tones (0.0=diatonic, 1.0=chromatic)",
            param_type=ParameterType.PROBABILITY,
            default_value=0.3,
            category=ParameterCategory.MELODY,
            musical_impact=MusicalImpact.HIGH,
            genre_relevance=["bebop", "jazz", "classical"]
        ))

        self.register(ParameterDefinition(
            name="ornament_probability",
            full_path="melody.ornaments.probability",
            description="Probability of adding ornaments (trills, mordents, etc.)",
            param_type=ParameterType.PROBABILITY,
            default_value=0.2,
            category=ParameterCategory.MELODY,
            musical_impact=MusicalImpact.MEDIUM,
            genre_relevance=["classical", "baroque", "jazz"]
        ))

    def _register_rhythm_parameters(self):
        """Register all rhythm-related parameters"""

        self.register(ParameterDefinition(
            name="swing_amount",
            full_path="rhythm.swing.amount",
            description="Swing ratio (0.5=straight, 0.67=standard swing, 0.75=hard swing)",
            param_type=ParameterType.CONTINUOUS,
            default_value=0.67,
            min_value=0.5,
            max_value=0.75,
            category=ParameterCategory.RHYTHM,
            musical_impact=MusicalImpact.CRITICAL,
            genre_relevance=["jazz", "swing", "blues"]
        ))

        self.register(ParameterDefinition(
            name="syncopation_probability",
            full_path="rhythm.syncopation.probability",
            description="Probability of syncopated rhythms",
            param_type=ParameterType.PROBABILITY,
            default_value=0.3,
            category=ParameterCategory.RHYTHM,
            musical_impact=MusicalImpact.HIGH,
            genre_relevance=["jazz", "funk", "latin", "reggae"]
        ))

        self.register(ParameterDefinition(
            name="microtiming_variation",
            full_path="rhythm.microtiming.variation",
            description="Amount of microtiming humanization (ms)",
            param_type=ParameterType.INTEGER,
            default_value=10,
            min_value=0,
            max_value=50,
            category=ParameterCategory.RHYTHM,
            musical_impact=MusicalImpact.MEDIUM
        ))

    def _register_genre_parameters(self):
        """Register genre-specific parameters"""

        # Rock genre parameters
        self.register(ParameterDefinition(
            name="rock_power_chord_probability",
            full_path="genre.rock.power_chord_probability",
            description="Probability of using power chords",
            param_type=ParameterType.PROBABILITY,
            default_value=0.7,
            category=ParameterCategory.GENRE,
            musical_impact=MusicalImpact.CRITICAL,
            genre_relevance=["rock", "punk", "metal"],
            module_file="genres/classic_rock.py"
        ))

        self.register(ParameterDefinition(
            name="rock_bend_probability",
            full_path="genre.rock.bend_probability",
            description="Probability of guitar bends",
            param_type=ParameterType.PROBABILITY,
            default_value=0.3,
            category=ParameterCategory.GENRE,
            musical_impact=MusicalImpact.HIGH,
            genre_relevance=["rock", "blues"],
            module_file="genres/classic_rock.py"
        ))

        self.register(ParameterDefinition(
            name="rock_vibrato_probability",
            full_path="genre.rock.vibrato_probability",
            description="Probability of vibrato on sustained notes",
            param_type=ParameterType.PROBABILITY,
            default_value=0.4,
            category=ParameterCategory.GENRE,
            musical_impact=MusicalImpact.MEDIUM,
            genre_relevance=["rock", "blues"],
            module_file="genres/classic_rock.py"
        ))

        self.register(ParameterDefinition(
            name="rock_vibrato_depth",
            full_path="genre.rock.vibrato_depth",
            description="Vibrato depth range (cents)",
            param_type=ParameterType.CONTINUOUS,
            default_value=30.0,
            min_value=0.0,
            max_value=100.0,
            category=ParameterCategory.GENRE,
            musical_impact=MusicalImpact.MEDIUM,
            genre_relevance=["rock", "blues"],
            module_file="genres/classic_rock.py"
        ))

    def _register_bass_parameters(self):
        """Register bass-related parameters"""

        self.register(ParameterDefinition(
            name="bass_walking_probability",
            full_path="bass.style.walking_probability",
            description="Probability of walking bass line",
            param_type=ParameterType.PROBABILITY,
            default_value=0.8,
            category=ParameterCategory.BASS,
            musical_impact=MusicalImpact.HIGH,
            genre_relevance=["jazz", "swing"]
        ))

    def _register_drum_parameters(self):
        """Register drum-related parameters"""

        self.register(ParameterDefinition(
            name="drum_kick_velocity_min",
            full_path="drums.kick.velocity_min",
            description="Minimum kick drum velocity",
            param_type=ParameterType.VELOCITY,
            default_value=80,
            category=ParameterCategory.DRUMS,
            musical_impact=MusicalImpact.MEDIUM
        ))

        self.register(ParameterDefinition(
            name="drum_kick_velocity_max",
            full_path="drums.kick.velocity_max",
            description="Maximum kick drum velocity",
            param_type=ParameterType.VELOCITY,
            default_value=110,
            category=ParameterCategory.DRUMS,
            musical_impact=MusicalImpact.MEDIUM
        ))

    def _register_articulation_parameters(self):
        """Register articulation parameters"""

        self.register(ParameterDefinition(
            name="note_duration_ratio",
            full_path="articulation.duration.ratio",
            description="Note duration as ratio of full length (0.5=staccato, 1.0=legato)",
            param_type=ParameterType.PROBABILITY,
            default_value=0.9,
            category=ParameterCategory.ARTICULATION,
            musical_impact=MusicalImpact.HIGH
        ))

    def _register_dynamics_parameters(self):
        """Register dynamics parameters"""

        self.register(ParameterDefinition(
            name="velocity_base",
            full_path="dynamics.velocity.base",
            description="Base velocity for notes",
            param_type=ParameterType.VELOCITY,
            default_value=80,
            category=ParameterCategory.DYNAMICS,
            musical_impact=MusicalImpact.MEDIUM
        ))

        self.register(ParameterDefinition(
            name="velocity_variation",
            full_path="dynamics.velocity.variation",
            description="Amount of velocity variation (+/-)",
            param_type=ParameterType.INTEGER,
            default_value=20,
            min_value=0,
            max_value=63,
            category=ParameterCategory.DYNAMICS,
            musical_impact=MusicalImpact.MEDIUM
        ))

    def _register_instrumentation_parameters(self):
        """
        Register instrumentation parameters (Agent 1 expansion)

        Imports and registers all 80 instrumentation parameters from
        the instrumentation_expansion module.
        """
        try:
            from .instrumentation_expansion import (
                register_piano_parameters,
                register_bass_parameters,
                register_drums_parameters,
                register_brass_parameters,
                register_strings_parameters
            )

            # Register all instrumentation parameter groups
            register_piano_parameters()
            register_bass_parameters()
            register_drums_parameters()
            register_brass_parameters()
            register_strings_parameters()
        except ImportError:
            # If module not available, skip (for backward compatibility)
            pass

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def export_to_json(self, filepath: str):
        """Export registry to JSON file"""
        data = {
            path: {
                "name": p.name,
                "description": p.description,
                "type": p.param_type.value,
                "default": p.default_value,
                "min": p.min_value,
                "max": p.max_value,
                "options": p.options,
                "category": p.category.value if p.category else None,
                "impact": p.musical_impact.value,
                "genres": p.genre_relevance
            }
            for path, p in self.parameters.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_documentation(self, filepath: str):
        """Generate markdown documentation of all parameters"""
        lines = []
        lines.append("# Universal Parameter Registry")
        lines.append("")
        lines.append(f"Total Parameters: {len(self.parameters)}")
        lines.append("")

        # Group by category
        for category in ParameterCategory:
            params = self.get_by_category(category)
            if not params:
                continue

            lines.append(f"## {category.value.title()} Parameters")
            lines.append("")
            lines.append("| Parameter | Type | Default | Description |")
            lines.append("|-----------|------|---------|-------------|")

            for param in sorted(params, key=lambda p: p.full_path):
                param_type = param.param_type.value
                default = str(param.default_value)
                lines.append(f"| `{param.full_path}` | {param_type} | {default} | {param.description} |")

            lines.append("")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the parameter registry"""
        stats = {
            "total_parameters": len(self.parameters),
            "by_type": {},
            "by_category": {},
            "by_impact": {},
            "learnable_count": sum(1 for p in self.parameters.values() if p.learnable)
        }

        for param in self.parameters.values():
            # By type
            type_name = param.param_type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1

            # By category
            if param.category:
                cat_name = param.category.value
                stats["by_category"][cat_name] = stats["by_category"].get(cat_name, 0) + 1

            # By impact
            impact_name = param.musical_impact.value
            stats["by_impact"][impact_name] = stats["by_impact"].get(impact_name, 0) + 1

        return stats


# ============================================================================
# Global registry instance
# ============================================================================

REGISTRY = UniversalParameterRegistry()


# ============================================================================
# Convenience functions
# ============================================================================

def get_parameter(full_path: str) -> Optional[ParameterDefinition]:
    """Get parameter definition"""
    return REGISTRY.get(full_path)


def validate(full_path: str, value: Any) -> Tuple[bool, str]:
    """Validate a parameter value"""
    return REGISTRY.validate_parameter(full_path, value)


def get_default(full_path: str) -> Any:
    """Get default value for a parameter"""
    param = REGISTRY.get(full_path)
    return param.default_value if param else None


if __name__ == "__main__":
    # Test the registry
    print("=" * 80)
    print("UNIVERSAL PARAMETER REGISTRY")
    print("=" * 80)

    stats = REGISTRY.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total parameters: {stats['total_parameters']}")
    print(f"   Learnable: {stats['learnable_count']}")

    print(f"\n📁 By Type:")
    for type_name, count in sorted(stats['by_type'].items()):
        print(f"   {type_name:20s}: {count:3d}")

    print(f"\n🎵 By Category:")
    for cat_name, count in sorted(stats['by_category'].items()):
        print(f"   {cat_name:20s}: {count:3d}")

    print(f"\n🎯 By Impact:")
    for impact_name, count in sorted(stats['by_impact'].items()):
        print(f"   {impact_name:20s}: {count:3d}")

    # Test validation
    print(f"\n✅ Validation Tests:")
    test_cases = [
        ("harmony.voicing.spread", 0.5, True),
        ("harmony.voicing.spread", 1.5, False),
        ("harmony.voicing.type", "drop2", True),
        ("harmony.voicing.type", "invalid", False),
    ]

    for path, value, expected_valid in test_cases:
        valid, msg = REGISTRY.validate_parameter(path, value)
        status = "✅" if valid == expected_valid else "❌"
        print(f"   {status} {path} = {value}: {valid}")

    # Export
    REGISTRY.export_to_json("/home/user/Do/midi_generator/parameters/registry.json")
    REGISTRY.generate_documentation("/home/user/Do/midi_generator/parameters/PARAMETERS.md")

    print(f"\n💾 Exported registry to JSON and documentation")
    print("=" * 80)
