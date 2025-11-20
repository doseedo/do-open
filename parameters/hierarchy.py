"""
Hierarchical Parameter Structure

Reduces effective parameter space from 800 to ~50 high-level + conditional lower levels.

Hierarchy:
    Level 1 (Top): Genre (18 categories) → predicts ~100 params
    Level 2 (Mid): Complexity (3-5 levels) → predicts ~50 params per genre
    Level 3 (Low): Detailed parameters (remaining) → conditional on levels 1-2

Author: Musical Program Synthesis Team
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum


class ParameterLevel(Enum):
    """Hierarchy level for parameters."""
    TOP = 1      # Genre, style, form (high-level)
    MID = 2      # Complexity, density, tempo (mid-level)
    LOW = 3      # Specific voicings, ornaments (detailed)


@dataclass
class HierarchicalParameter:
    """Parameter with hierarchical metadata."""
    name: str
    level: ParameterLevel
    parent_parameters: List[str]  # Parameters that condition this one
    child_parameters: List[str]   # Parameters conditioned on this one
    importance: float             # 0-1, higher = more important


# Hierarchy Definition
PARAMETER_HIERARCHY = {
    # ========================================
    # LEVEL 1: TOP-LEVEL (Genre & Style)
    # ========================================
    'genre': {
        'level': ParameterLevel.TOP,
        'parameters': [
            'style.genre',                    # jazz, classical, electronic, etc.
            'style.sub_genre',                # bebop, modal, fusion, etc.
            'style.era',                      # 1920s, 1940s, modern, etc.
            'structure.form_type',            # AABA, verse-chorus, sonata, etc.
            'instrumentation.ensemble_type'   # big_band, string_quartet, solo, etc.
        ],
        'predicts': [
            'harmony.*',          # Genre → harmony style
            'melody.*',           # Genre → melodic patterns
            'rhythm.*',           # Genre → rhythmic patterns
            'instrumentation.*'   # Genre → instrument choices
        ],
        'description': 'High-level style decisions that constrain all other parameters'
    },

    # ========================================
    # LEVEL 2: MID-LEVEL (Complexity & Density)
    # ========================================
    'complexity': {
        'level': ParameterLevel.MID,
        'parameters': [
            'harmony.complexity',             # 0-1 scale
            'melody.complexity',
            'rhythm.complexity',
            'texture.complexity',
            'structure.complexity'
        ],
        'conditioned_on': ['style.genre'],
        'predicts': [
            'harmony.chord_density',
            'harmony.chromaticism',
            'melody.note_density',
            'melody.interval_complexity',
            'rhythm.syncopation'
        ],
        'description': 'Complexity level within the chosen genre'
    },

    'density': {
        'level': ParameterLevel.MID,
        'parameters': [
            'harmony.chord_density',          # chords per measure
            'melody.note_density',            # notes per beat
            'rhythm.event_density',           # events per bar
            'texture.voice_count'             # number of simultaneous voices
        ],
        'conditioned_on': ['style.genre', 'harmony.complexity'],
        'predicts': [
            'harmony.voicing.spread',
            'melody.articulation',
            'dynamics.variation'
        ],
        'description': 'Information density within complexity level'
    },

    'tempo': {
        'level': ParameterLevel.MID,
        'parameters': [
            'tempo.bpm',
            'tempo.variation',
            'tempo.feel'                      # laid_back, pushed, strict
        ],
        'conditioned_on': ['style.genre', 'style.sub_genre'],
        'predicts': [
            'rhythm.subdivision',
            'rhythm.swing.amount',
            'melody.articulation'
        ],
        'description': 'Tempo and rhythmic feel'
    },

    # ========================================
    # LEVEL 3: LOW-LEVEL (Detailed Parameters)
    # ========================================
    'harmony_details': {
        'level': ParameterLevel.LOW,
        'parameters': [
            'harmony.voicing.type',           # drop_2, spread, cluster
            'harmony.voicing.spread',
            'harmony.extension.usage',        # 9ths, 11ths, 13ths
            'harmony.substitution.type',
            'harmony.modal_mixture'
        ],
        'conditioned_on': [
            'style.genre',
            'harmony.complexity',
            'harmony.chord_density'
        ],
        'description': 'Specific harmonic choices within established style and complexity'
    },

    'melody_details': {
        'level': ParameterLevel.LOW,
        'parameters': [
            'melody.contour.shape',           # arch, wave, ascending
            'melody.ornamentation.type',      # trill, turn, mordent
            'melody.motif_development',
            'melody.sequence.type'
        ],
        'conditioned_on': [
            'style.genre',
            'melody.complexity',
            'melody.note_density',
            'harmony.complexity'              # Melody follows harmony
        ],
        'description': 'Specific melodic choices within established style'
    },

    'rhythm_details': {
        'level': ParameterLevel.LOW,
        'parameters': [
            'rhythm.swing.amount',            # 0-100%
            'rhythm.syncopation',
            'rhythm.polyrhythm.ratio',
            'rhythm.clave_type'               # son, rumba, bossa
        ],
        'conditioned_on': [
            'style.genre',
            'rhythm.complexity',
            'tempo.bpm',
            'tempo.feel'
        ],
        'description': 'Specific rhythmic choices within tempo and style'
    },

    'dynamics_details': {
        'level': ParameterLevel.LOW,
        'parameters': [
            'dynamics.overall_level',         # pp, p, mp, mf, f, ff
            'dynamics.variation',             # how much dynamic range
            'dynamics.articulation_coupling',
            'dynamics.adsr.attack',
            'dynamics.adsr.release'
        ],
        'conditioned_on': [
            'style.genre',
            'instrumentation.ensemble_type',
            'tempo.bpm'
        ],
        'description': 'Dynamic shaping within style and ensemble'
    },

    'instrumentation_details': {
        'level': ParameterLevel.LOW,
        'parameters': [
            'instrumentation.voicing.balance',
            'instrumentation.doubling',
            'instrumentation.register_distribution',
            'instrumentation.section_emphasis'
        ],
        'conditioned_on': [
            'instrumentation.ensemble_type',
            'style.genre',
            'texture.complexity'
        ],
        'description': 'Specific orchestration choices within ensemble type'
    }
}


class HierarchicalParameterModel:
    """
    Hierarchical parameter prediction model.

    Predicts parameters level-by-level:
    1. Level 1 (genre) from features
    2. Level 2 (complexity) from features + Level 1
    3. Level 3 (details) from features + Level 1 + Level 2
    """

    def __init__(self):
        self.level_1_models = {}  # genre classifiers
        self.level_2_models = {}  # complexity predictors (conditioned on L1)
        self.level_3_models = {}  # detail predictors (conditioned on L1+L2)

    def train_level_1(self, training_data):
        """
        Train top-level (genre) classifiers.

        Input: 1000 features
        Output: ~5 genre parameters
        """
        # Train genre classifier
        # style.genre: 18 classes (jazz, classical, electronic, etc.)
        # This is the most important classification!
        pass

    def train_level_2(self, training_data, genre_conditioning):
        """
        Train mid-level (complexity) predictors.

        Input: 1000 features + genre
        Output: ~50 complexity/density parameters
        """
        # Train per-genre complexity models
        # Example: jazz complexity model, classical complexity model
        pass

    def train_level_3(self, training_data, genre_conditioning, complexity_conditioning):
        """
        Train low-level (detail) predictors.

        Input: 1000 features + genre + complexity
        Output: ~745 detailed parameters
        """
        # Train conditional detail models
        # Example: jazz_complex_voicing_model, jazz_simple_voicing_model
        pass

    def predict_hierarchical(self, features):
        """
        Predict all parameters hierarchically.

        Args:
            features: 1000-dimensional feature vector

        Returns:
            dict with all 800 parameters predicted hierarchically
        """
        results = {}

        # Level 1: Predict genre
        level_1 = self._predict_level_1(features)
        results.update(level_1)

        # Level 2: Predict complexity (conditioned on genre)
        level_2 = self._predict_level_2(features, level_1)
        results.update(level_2)

        # Level 3: Predict details (conditioned on genre + complexity)
        level_3 = self._predict_level_3(features, level_1, level_2)
        results.update(level_3)

        return results

    def _predict_level_1(self, features):
        """Predict top-level parameters."""
        # Predict genre (most important!)
        genre = self.level_1_models['genre'].predict(features)

        return {
            'style.genre': genre,
            # ... other level 1 parameters
        }

    def _predict_level_2(self, features, level_1):
        """Predict mid-level parameters conditioned on level 1."""
        genre = level_1['style.genre']

        # Use genre-specific complexity model
        model = self.level_2_models[f'complexity_{genre}']
        complexity = model.predict(features)

        return {
            'harmony.complexity': complexity['harmony'],
            'melody.complexity': complexity['melody'],
            # ... other level 2 parameters
        }

    def _predict_level_3(self, features, level_1, level_2):
        """Predict low-level parameters conditioned on level 1 + 2."""
        genre = level_1['style.genre']
        complexity = level_2.get('harmony.complexity', 0.5)

        # Use genre + complexity specific model
        model_key = f'{genre}_complexity_{int(complexity * 10)}'
        model = self.level_3_models.get(model_key)

        if model:
            return model.predict(features)
        else:
            # Fallback to unconditional model
            return self.level_3_models['default'].predict(features)


def get_parameter_level(param_name: str) -> ParameterLevel:
    """Determine hierarchy level for a parameter."""

    # Level 1: Genre and high-level style
    level_1_keywords = ['genre', 'style', 'era', 'form_type', 'ensemble_type']
    if any(kw in param_name for kw in level_1_keywords):
        return ParameterLevel.TOP

    # Level 2: Complexity and density
    level_2_keywords = ['complexity', 'density', 'tempo', 'bpm']
    if any(kw in param_name for kw in level_2_keywords):
        return ParameterLevel.MID

    # Level 3: Everything else (detailed parameters)
    return ParameterLevel.LOW


def get_parent_parameters(param_name: str) -> List[str]:
    """Get parent parameters that condition this parameter."""

    # Search hierarchy for this parameter
    for group_name, group_data in PARAMETER_HIERARCHY.items():
        if param_name in group_data.get('parameters', []):
            return group_data.get('conditioned_on', [])

    return []


def get_child_parameters(param_name: str) -> List[str]:
    """Get child parameters conditioned on this parameter."""

    children = []

    # Search hierarchy for parameters conditioned on this one
    for group_name, group_data in PARAMETER_HIERARCHY.items():
        conditioned_on = group_data.get('conditioned_on', [])

        # Check if param_name is in conditioned_on list
        if param_name in conditioned_on:
            children.extend(group_data.get('parameters', []))

        # Check if param_name predicts these parameters
        if param_name in group_data.get('predicts', []):
            children.extend(group_data.get('parameters', []))

    return children


def reduce_parameter_space(genre: str, complexity: float) -> Set[str]:
    """
    Reduce effective parameter space based on genre and complexity.

    Example:
        - Jazz + complex → 500 active parameters
        - Classical + simple → 300 active parameters

    Args:
        genre: Musical genre (jazz, classical, etc.)
        complexity: Complexity level 0-1

    Returns:
        Set of active parameter names
    """
    active_params = set()

    # Always include level 1 and level 2 parameters
    for group_name in ['genre', 'complexity', 'density', 'tempo']:
        group = PARAMETER_HIERARCHY.get(group_name, {})
        active_params.update(group.get('parameters', []))

    # Conditionally include level 3 based on genre and complexity
    # ... (genre-specific logic here)

    return active_params


# Example: Get hierarchy info
def print_hierarchy_info():
    """Print hierarchical structure."""
    print("PARAMETER HIERARCHY")
    print("=" * 60)

    for level in [ParameterLevel.TOP, ParameterLevel.MID, ParameterLevel.LOW]:
        print(f"\n{level.name} LEVEL:")
        print("-" * 60)

        for group_name, group_data in PARAMETER_HIERARCHY.items():
            if group_data.get('level') == level:
                print(f"\n  Group: {group_name}")
                print(f"  Parameters: {len(group_data.get('parameters', []))}")
                print(f"  Description: {group_data.get('description', 'N/A')}")

                if 'conditioned_on' in group_data:
                    print(f"  Conditioned on: {', '.join(group_data['conditioned_on'])}")


if __name__ == "__main__":
    print_hierarchy_info()
