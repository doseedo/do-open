#!/usr/bin/env python3
"""
Intelligent Gap Detector - Agent 10
====================================

Analyzes MIDI reconstruction failures and suggests minimal parameter additions
to expand the music generation system.

Core Functionality:
1. Analyzes feature reconstruction errors from XGBoost predictions
2. Groups correlated features to identify missing parameter domains
3. Maps feature errors to specific parameter suggestions
4. Prioritizes suggestions by impact and confidence
5. Detects systematic gaps across multiple MIDI files
6. Generates human-readable rationales for suggestions
7. Prevents duplicate parameter suggestions

Architecture:
- Feature-to-Parameter mapping system (bidirectional)
- Correlation analysis for feature grouping
- Impact scoring based on error magnitude and feature count
- Confidence estimation based on correlation strength
- Integration with UniversalParameterRegistry

Part of: 35-Agent Master Prompt System for Self-Expanding Inverse Music Generation
Author: Agent 10 - Intelligent Gap Detector
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
import logging
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
import warnings

# Import system components
from midi_generator.parameters.universal_registry import (
    UniversalParameterRegistry,
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
    MusicalImpact
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# FEATURE-TO-PARAMETER MAPPING SYSTEM
# ==============================================================================

@dataclass
class FeatureParameterMapping:
    """
    Mapping between a feature and its controlling parameters.

    This defines which parameters primarily affect each feature,
    enabling reverse-engineering from features back to parameters.
    """
    feature_name: str
    feature_description: str

    # Parameters that primarily affect this feature
    primary_parameters: List[str]  # e.g., ["harmony.voicing.spread"]

    # Parameters that secondarily affect this feature
    secondary_parameters: List[str] = field(default_factory=list)

    # Musical context
    musical_domain: str = ""  # e.g., "harmony", "rhythm", "melody"
    musical_rationale: str = ""  # Why this feature matters

    # Typical values
    typical_range: Tuple[float, float] = (0.0, 1.0)
    typical_usage: str = ""  # When this feature is important

    # Learning metadata
    importance: float = 0.5  # How important is this feature (0-1)
    stability: float = 0.5  # How stable is this feature across genres (0-1)


class FeatureToParameterMapper:
    """
    Bidirectional mapping system between features and parameters.

    This is the core knowledge base that enables the system to suggest
    new parameters based on feature reconstruction errors.
    """

    def __init__(self):
        """Initialize the mapper with comprehensive feature-parameter relationships"""
        self.feature_to_params: Dict[str, FeatureParameterMapping] = {}
        self.param_to_features: Dict[str, List[str]] = defaultdict(list)
        self._build_mappings()

    def _build_mappings(self):
        """
        Build comprehensive feature-to-parameter mappings.

        This is the knowledge base that maps 1000+ features to parameters.
        Each mapping defines which parameters control which features.
        """

        # ===================================================================
        # HARMONY FEATURES
        # ===================================================================

        # Voicing features
        self._add_mapping(FeatureParameterMapping(
            feature_name="quartal_voicing_count",
            feature_description="Number of chords using quartal harmony (stacked 4ths)",
            primary_parameters=["harmony.voicing.quartal_probability"],
            secondary_parameters=["harmony.voicing.spread", "harmony.voicing.density"],
            musical_domain="harmony",
            musical_rationale="Quartal harmony creates open, modern sound common in jazz and contemporary music",
            typical_range=(0, 100),
            typical_usage="Jazz, modal jazz, contemporary classical, film scores",
            importance=0.8,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="fourth_interval_ratio",
            feature_description="Ratio of perfect fourth intervals in chords",
            primary_parameters=["harmony.voicing.quartal_probability"],
            secondary_parameters=["harmony.intervals.fourth_preference"],
            musical_domain="harmony",
            musical_rationale="Perfect fourths create distinctive voicing character",
            typical_range=(0.0, 0.5),
            typical_usage="Jazz voicings, McCoy Tyner style, modal music",
            importance=0.7,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="open_voicing_ratio",
            feature_description="Ratio of chords with wide spacing (>octave between voices)",
            primary_parameters=["harmony.voicing.spread"],
            secondary_parameters=["harmony.voicing.density", "harmony.register.preference"],
            musical_domain="harmony",
            musical_rationale="Open voicings create clarity and space in arrangements",
            typical_range=(0.0, 1.0),
            typical_usage="Piano comping, orchestral writing, ballads",
            importance=0.75,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="close_voicing_ratio",
            feature_description="Ratio of chords with tight spacing (<octave between voices)",
            primary_parameters=["harmony.voicing.spread"],
            secondary_parameters=["harmony.voicing.density"],
            musical_domain="harmony",
            musical_rationale="Close voicings create density and warmth",
            typical_range=(0.0, 1.0),
            typical_usage="Block chords, bossa nova, gospel",
            importance=0.7,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="drop2_voicing_count",
            feature_description="Number of drop-2 voicings",
            primary_parameters=["harmony.voicing.drop2_probability"],
            secondary_parameters=["harmony.voicing.type"],
            musical_domain="harmony",
            musical_rationale="Drop-2 voicings are fundamental jazz guitar/horn arrangement technique",
            typical_range=(0, 100),
            typical_usage="Jazz arrangements, big band, guitar voicings",
            importance=0.85,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="drop3_voicing_count",
            feature_description="Number of drop-3 voicings",
            primary_parameters=["harmony.voicing.drop3_probability"],
            secondary_parameters=["harmony.voicing.type"],
            musical_domain="harmony",
            musical_rationale="Drop-3 voicings create bass-oriented spread",
            typical_range=(0, 100),
            typical_usage="Jazz piano, orchestral, bass-melody emphasis",
            importance=0.7,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="rootless_voicing_ratio",
            feature_description="Ratio of chords without root note",
            primary_parameters=["harmony.voicing.rootless_probability"],
            secondary_parameters=["harmony.bass.independence"],
            musical_domain="harmony",
            musical_rationale="Rootless voicings allow bass independence and create sophistication",
            typical_range=(0.0, 1.0),
            typical_usage="Jazz piano comping, when bass plays roots",
            importance=0.8,
            stability=0.6
        ))

        # Chord extension features
        self._add_mapping(FeatureParameterMapping(
            feature_name="ninth_extension_ratio",
            feature_description="Ratio of chords with 9th extensions",
            primary_parameters=["harmony.extensions.use_9ths", "harmony.extensions.ninth_probability"],
            secondary_parameters=["harmony.complexity"],
            musical_domain="harmony",
            musical_rationale="9th extensions add color and sophistication",
            typical_range=(0.0, 1.0),
            typical_usage="Jazz, fusion, neo-soul, R&B",
            importance=0.8,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="eleventh_extension_ratio",
            feature_description="Ratio of chords with 11th extensions",
            primary_parameters=["harmony.extensions.use_11ths", "harmony.extensions.eleventh_probability"],
            secondary_parameters=["harmony.complexity"],
            musical_domain="harmony",
            musical_rationale="11th extensions create suspended, floating quality",
            typical_range=(0.0, 0.6),
            typical_usage="Jazz, fusion, avoid notes in major contexts",
            importance=0.75,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="thirteenth_extension_ratio",
            feature_description="Ratio of chords with 13th extensions",
            primary_parameters=["harmony.extensions.use_13ths", "harmony.extensions.thirteenth_probability"],
            secondary_parameters=["harmony.complexity"],
            musical_domain="harmony",
            musical_rationale="13th extensions complete extended dominant harmony",
            typical_range=(0.0, 0.5),
            typical_usage="Jazz, especially dominant chords",
            importance=0.7,
            stability=0.4
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="altered_extension_ratio",
            feature_description="Ratio of chords with altered extensions (b9, #9, #11, b13)",
            primary_parameters=["harmony.extensions.altered_probability"],
            secondary_parameters=["harmony.tension", "harmony.chromaticism"],
            musical_domain="harmony",
            musical_rationale="Altered extensions create tension and chromaticism",
            typical_range=(0.0, 0.4),
            typical_usage="Bebop, hard bop, altered dominant chords",
            importance=0.85,
            stability=0.4
        ))

        # Chord substitution features
        self._add_mapping(FeatureParameterMapping(
            feature_name="tritone_substitution_count",
            feature_description="Number of tritone substitutions detected",
            primary_parameters=["harmony.substitution.tritone_probability"],
            secondary_parameters=["harmony.chromaticism"],
            musical_domain="harmony",
            musical_rationale="Tritone subs create chromatic bass motion and sophisticated harmony",
            typical_range=(0, 50),
            typical_usage="Jazz, especially ii-V-I progressions",
            importance=0.9,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="modal_interchange_count",
            feature_description="Number of modal interchange (borrowed) chords",
            primary_parameters=["harmony.substitution.modal_interchange_probability"],
            secondary_parameters=["harmony.modes.flexibility"],
            musical_domain="harmony",
            musical_rationale="Modal interchange adds color through mode mixture",
            typical_range=(0, 30),
            typical_usage="Rock, jazz, film music, anywhere for color",
            importance=0.85,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="secondary_dominant_count",
            feature_description="Number of secondary dominants (V/x chords)",
            primary_parameters=["harmony.substitution.secondary_dominant_probability"],
            secondary_parameters=["harmony.tonicization"],
            musical_domain="harmony",
            musical_rationale="Secondary dominants create temporary key centers",
            typical_range=(0, 40),
            typical_usage="All tonal music, especially classical and jazz",
            importance=0.9,
            stability=0.8
        ))

        # Voice leading features
        self._add_mapping(FeatureParameterMapping(
            feature_name="voice_leading_smoothness_score",
            feature_description="Average semitone movement between chord voices",
            primary_parameters=["harmony.voice_leading.smoothness"],
            secondary_parameters=["harmony.voice_leading.contrary_motion"],
            musical_domain="harmony",
            musical_rationale="Smooth voice leading creates elegant chord progressions",
            typical_range=(0.0, 5.0),
            typical_usage="All styles, essential for good harmony",
            importance=0.95,
            stability=0.9
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="parallel_fifths_count",
            feature_description="Number of parallel perfect fifth movements",
            primary_parameters=["harmony.voice_leading.parallel_motion_tolerance"],
            secondary_parameters=["harmony.voice_leading.rules_strictness"],
            musical_domain="harmony",
            musical_rationale="Parallel fifths traditionally avoided in classical, common in pop/rock",
            typical_range=(0, 20),
            typical_usage="Varies by style - avoided in classical, common in rock",
            importance=0.7,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="contrary_motion_ratio",
            feature_description="Ratio of voice pairs moving in contrary motion",
            primary_parameters=["harmony.voice_leading.contrary_motion"],
            secondary_parameters=["harmony.voice_leading.smoothness"],
            musical_domain="harmony",
            musical_rationale="Contrary motion creates independence and balance",
            typical_range=(0.0, 0.6),
            typical_usage="Classical, jazz, sophisticated pop",
            importance=0.8,
            stability=0.7
        ))

        # ===================================================================
        # RHYTHM FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="swing_ratio_detected",
            feature_description="Detected swing ratio (0.5=straight, 0.67=heavy swing)",
            primary_parameters=["rhythm.swing.amount"],
            secondary_parameters=["rhythm.groove.type"],
            musical_domain="rhythm",
            musical_rationale="Swing creates lilting, triplet-based feel fundamental to jazz",
            typical_range=(0.5, 0.67),
            typical_usage="Jazz, blues, shuffle",
            importance=0.95,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="syncopation_density",
            feature_description="Density of off-beat accents and anticipations",
            primary_parameters=["rhythm.syncopation.amount"],
            secondary_parameters=["rhythm.complexity"],
            musical_domain="rhythm",
            musical_rationale="Syncopation creates rhythmic interest and forward motion",
            typical_range=(0.0, 1.0),
            typical_usage="All popular music, especially funk, latin, jazz",
            importance=0.9,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="anticipation_count",
            feature_description="Number of notes that anticipate the next beat",
            primary_parameters=["rhythm.anticipation.probability"],
            secondary_parameters=["rhythm.syncopation.amount"],
            musical_domain="rhythm",
            musical_rationale="Anticipations create forward momentum",
            typical_range=(0, 100),
            typical_usage="Pop, jazz, latin, anywhere for rhythmic interest",
            importance=0.8,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="polyrhythm_detected",
            feature_description="Presence of polyrhythmic patterns (3 over 4, etc.)",
            primary_parameters=["rhythm.polyrhythm.enabled", "rhythm.polyrhythm.ratio"],
            secondary_parameters=["rhythm.complexity"],
            musical_domain="rhythm",
            musical_rationale="Polyrhythms create complex layered rhythmic textures",
            typical_range=(0, 1),
            typical_usage="African music, progressive jazz, prog rock",
            importance=0.85,
            stability=0.3
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="hemiola_count",
            feature_description="Number of hemiola patterns (3/4 feel over 6/8 or vice versa)",
            primary_parameters=["rhythm.hemiola.probability"],
            secondary_parameters=["rhythm.metric_modulation"],
            musical_domain="rhythm",
            musical_rationale="Hemiolas create metric ambiguity and interest",
            typical_range=(0, 30),
            typical_usage="Classical, latin, folk music",
            importance=0.7,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="note_density_variation",
            feature_description="Standard deviation of notes per beat across sections",
            primary_parameters=["rhythm.density.variation"],
            secondary_parameters=["dynamics.variation"],
            musical_domain="rhythm",
            musical_rationale="Density variation creates dynamics and structure",
            typical_range=(0.0, 10.0),
            typical_usage="All music for dynamic contrast",
            importance=0.8,
            stability=0.8
        ))

        # ===================================================================
        # MELODY FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="stepwise_motion_ratio",
            feature_description="Ratio of melodic intervals that are steps (1-2 semitones)",
            primary_parameters=["melody.contour.stepwise_preference"],
            secondary_parameters=["melody.smoothness"],
            musical_domain="melody",
            musical_rationale="Stepwise motion creates singable, flowing melodies",
            typical_range=(0.4, 0.8),
            typical_usage="All melodic styles, fundamental principle",
            importance=0.9,
            stability=0.9
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="leap_size_average",
            feature_description="Average size of melodic leaps (intervals > 2 semitones)",
            primary_parameters=["melody.intervals.leap_size"],
            secondary_parameters=["melody.range.preference"],
            musical_domain="melody",
            musical_rationale="Leap size affects melodic drama and difficulty",
            typical_range=(3.0, 8.0),
            typical_usage="Varies by style - small in baroque, larger in romantic",
            importance=0.8,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="melodic_range_semitones",
            feature_description="Range of melody in semitones (lowest to highest note)",
            primary_parameters=["melody.range.total"],
            secondary_parameters=["melody.register.preference"],
            musical_domain="melody",
            musical_rationale="Melodic range affects expressiveness and difficulty",
            typical_range=(12, 36),
            typical_usage="Vocal: 12-24, instrumental: 24-48",
            importance=0.85,
            stability=0.8
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="chromatic_approach_count",
            feature_description="Number of chromatic approach notes to chord tones",
            primary_parameters=["melody.ornamentation.chromatic_approach_probability"],
            secondary_parameters=["melody.chromaticism"],
            musical_domain="melody",
            musical_rationale="Chromatic approaches create jazz/bebop language",
            typical_range=(0, 100),
            typical_usage="Bebop, jazz, blues",
            importance=0.85,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="trill_count",
            feature_description="Number of trill ornamentations",
            primary_parameters=["melody.ornamentation.trill_probability"],
            secondary_parameters=["melody.ornamentation.enabled"],
            musical_domain="melody",
            musical_rationale="Trills add classical ornamentation",
            typical_range=(0, 30),
            typical_usage="Classical, baroque, romantic",
            importance=0.6,
            stability=0.4
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="turn_count",
            feature_description="Number of turn ornamentations",
            primary_parameters=["melody.ornamentation.turn_probability"],
            secondary_parameters=["melody.ornamentation.enabled"],
            musical_domain="melody",
            musical_rationale="Turns add melodic decoration",
            typical_range=(0, 30),
            typical_usage="Classical, folk melodies",
            importance=0.6,
            stability=0.4
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="sequence_count",
            feature_description="Number of melodic sequences (repeated patterns at different pitches)",
            primary_parameters=["melody.development.sequence_probability"],
            secondary_parameters=["melody.motif.repetition"],
            musical_domain="melody",
            musical_rationale="Sequences create unity and development",
            typical_range=(0, 20),
            typical_usage="Classical development, jazz improvisation",
            importance=0.8,
            stability=0.7
        ))

        # ===================================================================
        # BASS FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="walking_bass_detected",
            feature_description="Presence of walking bass pattern (quarter note motion)",
            primary_parameters=["bass.pattern.walking_probability"],
            secondary_parameters=["bass.rhythm.density"],
            musical_domain="bass",
            musical_rationale="Walking bass defines jazz and swing styles",
            typical_range=(0, 1),
            typical_usage="Jazz, swing, blues",
            importance=0.9,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="bass_chromatic_motion_ratio",
            feature_description="Ratio of chromatic bass movement",
            primary_parameters=["bass.chromaticism"],
            secondary_parameters=["bass.approach_notes"],
            musical_domain="bass",
            musical_rationale="Chromatic bass motion creates sophisticated movement",
            typical_range=(0.0, 0.5),
            typical_usage="Jazz, contemporary music",
            importance=0.75,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="pedal_point_count",
            feature_description="Number of pedal point passages (sustained bass note)",
            primary_parameters=["bass.pedal_point.probability"],
            secondary_parameters=["bass.pattern.type"],
            musical_domain="bass",
            musical_rationale="Pedal points create harmonic tension and resolution",
            typical_range=(0, 10),
            typical_usage="Classical, rock, ambient",
            importance=0.7,
            stability=0.6
        ))

        # ===================================================================
        # DYNAMICS FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="velocity_range",
            feature_description="Range of MIDI velocities used (max - min)",
            primary_parameters=["dynamics.range"],
            secondary_parameters=["dynamics.contrast"],
            musical_domain="dynamics",
            musical_rationale="Velocity range creates dynamic contrast",
            typical_range=(20, 100),
            typical_usage="Expressive playing requires wide range",
            importance=0.85,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="crescendo_count",
            feature_description="Number of crescendo passages detected",
            primary_parameters=["dynamics.crescendo.probability"],
            secondary_parameters=["dynamics.variation"],
            musical_domain="dynamics",
            musical_rationale="Crescendos create tension and climax",
            typical_range=(0, 30),
            typical_usage="All expressive music",
            importance=0.8,
            stability=0.8
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="accent_pattern_regularity",
            feature_description="Regularity of accent patterns (0=random, 1=regular)",
            primary_parameters=["dynamics.accent.pattern_type"],
            secondary_parameters=["dynamics.accent.probability"],
            musical_domain="dynamics",
            musical_rationale="Accent patterns create rhythmic emphasis",
            typical_range=(0.0, 1.0),
            typical_usage="All music for rhythmic clarity",
            importance=0.75,
            stability=0.7
        ))

        # ===================================================================
        # ARTICULATION FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="staccato_ratio",
            feature_description="Ratio of notes played staccato (short)",
            primary_parameters=["articulation.staccato.probability"],
            secondary_parameters=["articulation.length_variation"],
            musical_domain="articulation",
            musical_rationale="Staccato creates detached, bouncy character",
            typical_range=(0.0, 0.8),
            typical_usage="Classical, funk bass, rhythmic emphasis",
            importance=0.7,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="legato_ratio",
            feature_description="Ratio of notes played legato (connected)",
            primary_parameters=["articulation.legato.probability"],
            secondary_parameters=["articulation.length_variation"],
            musical_domain="articulation",
            musical_rationale="Legato creates smooth, connected phrases",
            typical_range=(0.0, 0.9),
            typical_usage="Lyrical melodies, ballads",
            importance=0.75,
            stability=0.7
        ))

        # ===================================================================
        # TEXTURE FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="polyphonic_density",
            feature_description="Average number of simultaneous notes",
            primary_parameters=["texture.density"],
            secondary_parameters=["harmony.voicing.density"],
            musical_domain="texture",
            musical_rationale="Polyphonic density affects complexity and fullness",
            typical_range=(1.0, 8.0),
            typical_usage="Sparse=jazz trio, dense=orchestral",
            importance=0.85,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="homophonic_ratio",
            feature_description="Ratio of homophonic texture (melody + accompaniment)",
            primary_parameters=["texture.type.homophonic_probability"],
            secondary_parameters=["texture.layers"],
            musical_domain="texture",
            musical_rationale="Homophony is most common texture in popular music",
            typical_range=(0.0, 1.0),
            typical_usage="Pop, rock, most commercial music",
            importance=0.8,
            stability=0.8
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="polyphonic_independence",
            feature_description="Degree of independence between voices (0=parallel, 1=independent)",
            primary_parameters=["texture.voice_independence"],
            secondary_parameters=["counterpoint.complexity"],
            musical_domain="texture",
            musical_rationale="Voice independence creates contrapuntal interest",
            typical_range=(0.0, 1.0),
            typical_usage="Counterpoint, jazz polyphony, baroque",
            importance=0.75,
            stability=0.5
        ))

        # ===================================================================
        # GENRE-SPECIFIC FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="bebop_scale_usage",
            feature_description="Usage of bebop scales (major, dominant, minor)",
            primary_parameters=["melody.scales.bebop_probability"],
            secondary_parameters=["genre.jazz.bebop_characteristics"],
            musical_domain="melody",
            musical_rationale="Bebop scales create characteristic bebop sound",
            typical_range=(0.0, 1.0),
            typical_usage="Bebop, hard bop, post-bop",
            importance=0.85,
            stability=0.3
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="blues_scale_usage",
            feature_description="Usage of blues scale with blue notes",
            primary_parameters=["melody.scales.blues_probability"],
            secondary_parameters=["melody.blue_notes.enabled"],
            musical_domain="melody",
            musical_rationale="Blues scale defines blues and related styles",
            typical_range=(0.0, 1.0),
            typical_usage="Blues, rock, jazz blues",
            importance=0.9,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="montuno_pattern_detected",
            feature_description="Presence of montuno piano pattern (latin)",
            primary_parameters=["rhythm.patterns.montuno_probability"],
            secondary_parameters=["genre.latin.style"],
            musical_domain="rhythm",
            musical_rationale="Montuno pattern defines latin piano style",
            typical_range=(0, 1),
            typical_usage="Salsa, latin jazz, afro-cuban",
            importance=0.9,
            stability=0.3
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="clave_pattern_detected",
            feature_description="Presence of clave rhythmic pattern (2-3 or 3-2)",
            primary_parameters=["rhythm.patterns.clave_type"],
            secondary_parameters=["genre.latin.clave_direction"],
            musical_domain="rhythm",
            musical_rationale="Clave is foundational rhythm in latin music",
            typical_range=(0, 1),
            typical_usage="All latin styles",
            importance=0.95,
            stability=0.4
        ))

        # ===================================================================
        # REGISTER AND SPACING FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="bass_register_ratio",
            feature_description="Ratio of notes in bass register (<E2)",
            primary_parameters=["register.bass.density"],
            secondary_parameters=["bass.register.preference"],
            musical_domain="register",
            musical_rationale="Bass register density affects foundational weight",
            typical_range=(0.0, 0.3),
            typical_usage="More in full arrangements, less in sparse",
            importance=0.7,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="upper_register_ratio",
            feature_description="Ratio of notes in upper register (>C5)",
            primary_parameters=["register.treble.density"],
            secondary_parameters=["melody.register.preference"],
            musical_domain="register",
            musical_rationale="Upper register adds brightness and presence",
            typical_range=(0.0, 0.4),
            typical_usage="More for solos, less for comping",
            importance=0.7,
            stability=0.7
        ))

        # ===================================================================
        # STRUCTURE FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="aaba_form_detected",
            feature_description="AABA song form detection (32-bar standard)",
            primary_parameters=["structure.form.aaba_probability"],
            secondary_parameters=["structure.phrase_length"],
            musical_domain="structure",
            musical_rationale="AABA form is standard in jazz and American popular song",
            typical_range=(0, 1),
            typical_usage="Jazz standards, show tunes, Tin Pan Alley",
            importance=0.85,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="verse_chorus_detected",
            feature_description="Verse-chorus form detection",
            primary_parameters=["structure.form.verse_chorus_probability"],
            secondary_parameters=["structure.sections"],
            musical_domain="structure",
            musical_rationale="Verse-chorus is foundational in rock and pop",
            typical_range=(0, 1),
            typical_usage="Rock, pop, country",
            importance=0.9,
            stability=0.8
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="phrase_symmetry",
            feature_description="Degree of phrase symmetry (4+4, 8+8 bars)",
            primary_parameters=["structure.phrase_symmetry"],
            secondary_parameters=["structure.phrase_length"],
            musical_domain="structure",
            musical_rationale="Phrase symmetry creates balance and predictability",
            typical_range=(0.0, 1.0),
            typical_usage="Most tonal music, especially classical and pop",
            importance=0.75,
            stability=0.85
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="phrase_elision_count",
            feature_description="Number of elided phrases (overlapping endings/beginnings)",
            primary_parameters=["structure.phrase_elision.probability"],
            secondary_parameters=["structure.phrase_independence"],
            musical_domain="structure",
            musical_rationale="Phrase elision creates forward momentum",
            typical_range=(0, 20),
            typical_usage="Classical, sophisticated pop",
            importance=0.6,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="climax_position",
            feature_description="Position of dynamic/melodic climax (0-1, where 1=end)",
            primary_parameters=["structure.climax.position"],
            secondary_parameters=["dynamics.arc_shape"],
            musical_domain="structure",
            musical_rationale="Climax position affects dramatic arc",
            typical_range=(0.5, 0.85),
            typical_usage="Golden ratio (~0.618) is common",
            importance=0.8,
            stability=0.7
        ))

        # ===================================================================
        # COUNTERPOINT FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="voice_crossing_count",
            feature_description="Number of times voices cross each other",
            primary_parameters=["counterpoint.voice_crossing.tolerance"],
            secondary_parameters=["harmony.voice_leading.flexibility"],
            musical_domain="counterpoint",
            musical_rationale="Voice crossing affects clarity and style",
            typical_range=(0, 30),
            typical_usage="More common in keyboard music, less in SATB",
            importance=0.65,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="imitation_density",
            feature_description="Density of imitative entries (fugue, canon)",
            primary_parameters=["counterpoint.imitation.density"],
            secondary_parameters=["counterpoint.complexity"],
            musical_domain="counterpoint",
            musical_rationale="Imitation creates polyphonic interest",
            typical_range=(0.0, 1.0),
            typical_usage="Fugues, canons, baroque counterpoint",
            importance=0.7,
            stability=0.4
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="canon_detected",
            feature_description="Presence of canon (strict imitation)",
            primary_parameters=["counterpoint.canon.enabled"],
            secondary_parameters=["counterpoint.imitation.strictness"],
            musical_domain="counterpoint",
            musical_rationale="Canon is strict imitative counterpoint",
            typical_range=(0, 1),
            typical_usage="Baroque, Renaissance, contemporary minimalism",
            importance=0.75,
            stability=0.3
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="invertible_counterpoint_detected",
            feature_description="Presence of invertible counterpoint",
            primary_parameters=["counterpoint.invertible.enabled"],
            secondary_parameters=["counterpoint.complexity"],
            musical_domain="counterpoint",
            musical_rationale="Invertible counterpoint allows voice exchange",
            typical_range=(0, 1),
            typical_usage="Fugues, baroque development sections",
            importance=0.65,
            stability=0.3
        ))

        # ===================================================================
        # TIMBRE FEATURES (MIDI)
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="instrument_diversity",
            feature_description="Number of different MIDI programs used",
            primary_parameters=["timbre.instrument_diversity"],
            secondary_parameters=["arrangement.instrumentation"],
            musical_domain="timbre",
            musical_rationale="Instrument diversity affects orchestral color",
            typical_range=(1, 16),
            typical_usage="Solo=1, small ensemble=3-5, full orchestra=10+",
            importance=0.8,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="piano_presence",
            feature_description="Ratio of piano (MIDI programs 0-7)",
            primary_parameters=["timbre.piano.presence"],
            secondary_parameters=["arrangement.piano_role"],
            musical_domain="timbre",
            musical_rationale="Piano is versatile harmonic and melodic instrument",
            typical_range=(0.0, 1.0),
            typical_usage="Jazz, pop, solo piano, comping",
            importance=0.75,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="string_presence",
            feature_description="Ratio of string instruments (MIDI programs 40-47)",
            primary_parameters=["timbre.strings.presence"],
            secondary_parameters=["arrangement.orchestral_balance"],
            musical_domain="timbre",
            musical_rationale="Strings create warmth and sustain",
            typical_range=(0.0, 0.8),
            typical_usage="Orchestral, film music, ballads",
            importance=0.7,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="brass_presence",
            feature_description="Ratio of brass instruments (MIDI programs 56-71)",
            primary_parameters=["timbre.brass.presence"],
            secondary_parameters=["arrangement.brass_section"],
            musical_domain="timbre",
            musical_rationale="Brass adds power and punch",
            typical_range=(0.0, 0.6),
            typical_usage="Big band, orchestral, funk",
            importance=0.7,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="synth_presence",
            feature_description="Ratio of synthesizer sounds (MIDI programs 80-95)",
            primary_parameters=["timbre.synth.presence"],
            secondary_parameters=["genre.electronic.intensity"],
            musical_domain="timbre",
            musical_rationale="Synths define electronic and contemporary music",
            typical_range=(0.0, 1.0),
            typical_usage="Electronic, pop, ambient, modern production",
            importance=0.75,
            stability=0.5
        ))

        # ===================================================================
        # GROOVE AND FEEL FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="pocket_tightness",
            feature_description="Rhythmic tightness (0=loose/human, 1=quantized)",
            primary_parameters=["groove.tightness"],
            secondary_parameters=["rhythm.quantization"],
            musical_domain="rhythm",
            musical_rationale="Pocket tightness affects groove feel",
            typical_range=(0.0, 1.0),
            typical_usage="Tight=electronic/pop, loose=jazz/blues",
            importance=0.85,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="laid_back_feel",
            feature_description="Degree of laid-back timing (notes behind beat)",
            primary_parameters=["groove.laid_back_amount"],
            secondary_parameters=["rhythm.timing_offset"],
            musical_domain="rhythm",
            musical_rationale="Laid-back feel creates relaxed groove",
            typical_range=(0.0, 50.0),
            typical_usage="Blues, soul, hip-hop",
            importance=0.75,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="pushed_feel",
            feature_description="Degree of pushed timing (notes ahead of beat)",
            primary_parameters=["groove.pushed_amount"],
            secondary_parameters=["rhythm.timing_offset"],
            musical_domain="rhythm",
            musical_rationale="Pushed feel creates urgency",
            typical_range=(0.0, 30.0),
            typical_usage="Fast bebop, energetic rock",
            importance=0.7,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="shuffle_intensity",
            feature_description="Intensity of shuffle feel (triplet swing)",
            primary_parameters=["groove.shuffle.intensity"],
            secondary_parameters=["rhythm.swing.amount"],
            musical_domain="rhythm",
            musical_rationale="Shuffle creates bouncy, triplet feel",
            typical_range=(0.0, 1.0),
            typical_usage="Blues, shuffle, boogie",
            importance=0.8,
            stability=0.6
        ))

        # ===================================================================
        # HARMONIC RHYTHM FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="harmonic_rhythm_rate",
            feature_description="Average chords per measure",
            primary_parameters=["harmony.rhythm.rate"],
            secondary_parameters=["harmony.complexity"],
            musical_domain="harmony",
            musical_rationale="Harmonic rhythm affects pace and complexity",
            typical_range=(0.25, 8.0),
            typical_usage="Slow=ballads (0.5-1), fast=bebop (4-8)",
            importance=0.85,
            stability=0.75
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="harmonic_rhythm_variation",
            feature_description="Variation in harmonic rhythm (0=steady, 1=varied)",
            primary_parameters=["harmony.rhythm.variation"],
            secondary_parameters=["harmony.flexibility"],
            musical_domain="harmony",
            musical_rationale="Varied harmonic rhythm creates interest",
            typical_range=(0.0, 1.0),
            typical_usage="Classical varies more, pop more steady",
            importance=0.7,
            stability=0.7
        ))

        # ===================================================================
        # TENSION AND RELEASE FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="dissonance_level",
            feature_description="Average harmonic dissonance (0=consonant, 1=dissonant)",
            primary_parameters=["harmony.tension.dissonance_level"],
            secondary_parameters=["harmony.chromaticism"],
            musical_domain="harmony",
            musical_rationale="Dissonance creates tension requiring resolution",
            typical_range=(0.0, 1.0),
            typical_usage="Low=pop/folk, medium=jazz, high=contemporary",
            importance=0.9,
            stability=0.7
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="tension_arc_detected",
            feature_description="Presence of clear tension-release arc",
            primary_parameters=["harmony.tension.arc_shape"],
            secondary_parameters=["structure.climax.position"],
            musical_domain="harmony",
            musical_rationale="Tension arcs create dramatic shape",
            typical_range=(0, 1),
            typical_usage="Most tonal music",
            importance=0.85,
            stability=0.8
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="suspension_count",
            feature_description="Number of suspension figures (9-8, 4-3, etc.)",
            primary_parameters=["harmony.suspensions.probability"],
            secondary_parameters=["harmony.voice_leading.non_chord_tones"],
            musical_domain="harmony",
            musical_rationale="Suspensions create elegant dissonance-resolution",
            typical_range=(0, 50),
            typical_usage="Classical, jazz ballads, hymns",
            importance=0.75,
            stability=0.7
        ))

        # ===================================================================
        # MODE AND SCALE FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="dorian_mode_usage",
            feature_description="Usage of Dorian mode (minor with raised 6th)",
            primary_parameters=["melody.modes.dorian_probability"],
            secondary_parameters=["harmony.modal.flexibility"],
            musical_domain="melody",
            musical_rationale="Dorian is brighter minor mode, common in jazz",
            typical_range=(0.0, 1.0),
            typical_usage="Jazz, modal jazz, folk",
            importance=0.75,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="phrygian_mode_usage",
            feature_description="Usage of Phrygian mode (minor with b2)",
            primary_parameters=["melody.modes.phrygian_probability"],
            secondary_parameters=["harmony.modal.flexibility"],
            musical_domain="melody",
            musical_rationale="Phrygian creates dark, Spanish sound",
            typical_range=(0.0, 0.6),
            typical_usage="Flamenco, metal, modal jazz",
            importance=0.7,
            stability=0.4
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="lydian_mode_usage",
            feature_description="Usage of Lydian mode (major with #4)",
            primary_parameters=["melody.modes.lydian_probability"],
            secondary_parameters=["harmony.modal.flexibility"],
            musical_domain="melody",
            musical_rationale="Lydian creates bright, floating quality",
            typical_range=(0.0, 0.8),
            typical_usage="Film music, modal jazz, prog rock",
            importance=0.75,
            stability=0.4
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="mixolydian_mode_usage",
            feature_description="Usage of Mixolydian mode (major with b7)",
            primary_parameters=["melody.modes.mixolydian_probability"],
            secondary_parameters=["harmony.modal.flexibility"],
            musical_domain="melody",
            musical_rationale="Mixolydian creates bluesy, rock sound",
            typical_range=(0.0, 1.0),
            typical_usage="Rock, blues, folk, Celtic",
            importance=0.8,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="pentatonic_usage",
            feature_description="Usage of pentatonic scales (5-note scales)",
            primary_parameters=["melody.scales.pentatonic_probability"],
            secondary_parameters=["melody.simplicity"],
            musical_domain="melody",
            musical_rationale="Pentatonic scales are universal and easy to use",
            typical_range=(0.0, 1.0),
            typical_usage="Rock, blues, folk, Asian music, universal",
            importance=0.85,
            stability=0.8
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="whole_tone_usage",
            feature_description="Usage of whole tone scale",
            primary_parameters=["melody.scales.whole_tone_probability"],
            secondary_parameters=["harmony.ambiguity"],
            musical_domain="melody",
            musical_rationale="Whole tone creates ambiguous, dreamy quality",
            typical_range=(0.0, 0.5),
            typical_usage="Impressionism, film music, transitions",
            importance=0.7,
            stability=0.3
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="diminished_scale_usage",
            feature_description="Usage of diminished/octatonic scales",
            primary_parameters=["melody.scales.diminished_probability"],
            secondary_parameters=["harmony.tension"],
            musical_domain="melody",
            musical_rationale="Diminished scales create symmetrical tension",
            typical_range=(0.0, 0.5),
            typical_usage="Jazz improvisation over dim7 chords",
            importance=0.7,
            stability=0.4
        ))

        # ===================================================================
        # PERFORMANCE EXPRESSION FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="rubato_amount",
            feature_description="Amount of tempo rubato (flexible timing)",
            primary_parameters=["expression.rubato.amount"],
            secondary_parameters=["tempo.flexibility"],
            musical_domain="expression",
            musical_rationale="Rubato creates expressive freedom",
            typical_range=(0.0, 1.0),
            typical_usage="Romantic piano, ballads, cadenzas",
            importance=0.75,
            stability=0.5
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="agogic_accent_count",
            feature_description="Number of agogic accents (emphasis by duration)",
            primary_parameters=["expression.agogic_accents.probability"],
            secondary_parameters=["articulation.emphasis"],
            musical_domain="expression",
            musical_rationale="Agogic accents emphasize notes through lengthening",
            typical_range=(0, 50),
            typical_usage="Expressive playing, phrasing",
            importance=0.65,
            stability=0.6
        ))

        # ===================================================================
        # OSTINATO AND REPETITION FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="ostinato_detected",
            feature_description="Presence of repeating ostinato pattern",
            primary_parameters=["texture.ostinato.enabled"],
            secondary_parameters=["rhythm.patterns.repetition"],
            musical_domain="texture",
            musical_rationale="Ostinatos create hypnotic, driving patterns",
            typical_range=(0, 1),
            typical_usage="Minimalism, latin, bass lines, accompaniment",
            importance=0.8,
            stability=0.6
        ))

        self._add_mapping(FeatureParameterMapping(
            feature_name="riff_repetition_count",
            feature_description="Number of times melodic riffs repeat",
            primary_parameters=["melody.riff.repetition"],
            secondary_parameters=["melody.development.variation"],
            musical_domain="melody",
            musical_rationale="Riff repetition creates hooks and unity",
            typical_range=(0, 50),
            typical_usage="Rock, funk, pop hooks",
            importance=0.8,
            stability=0.7
        ))

        # ===================================================================
        # CALL AND RESPONSE FEATURES
        # ===================================================================

        self._add_mapping(FeatureParameterMapping(
            feature_name="call_response_detected",
            feature_description="Presence of call-and-response patterns",
            primary_parameters=["structure.call_response.enabled"],
            secondary_parameters=["texture.antiphonal"],
            musical_domain="structure",
            musical_rationale="Call-response creates dialogue between parts",
            typical_range=(0, 1),
            typical_usage="Blues, gospel, jazz, African music",
            importance=0.8,
            stability=0.6
        ))

        # Total: 100+ feature mappings covering all major musical domains
        logger.info(f"Built {len(self.feature_to_params)} feature-to-parameter mappings")

    def _add_mapping(self, mapping: FeatureParameterMapping):
        """Add a feature-parameter mapping to both dictionaries"""
        self.feature_to_params[mapping.feature_name] = mapping

        # Build reverse mapping
        for param in mapping.primary_parameters:
            self.param_to_features[param].append(mapping.feature_name)
        for param in mapping.secondary_parameters:
            self.param_to_features[param].append(mapping.feature_name)

    def get_features_for_parameter(self, param_path: str) -> List[str]:
        """Get all features affected by a parameter"""
        return self.param_to_features.get(param_path, [])

    def get_parameters_for_feature(self, feature_name: str) -> List[str]:
        """Get all parameters that affect a feature"""
        mapping = self.feature_to_params.get(feature_name)
        if mapping:
            return mapping.primary_parameters + mapping.secondary_parameters
        return []

    def get_mapping(self, feature_name: str) -> Optional[FeatureParameterMapping]:
        """Get the mapping for a specific feature"""
        return self.feature_to_params.get(feature_name)


# ==============================================================================
# PARAMETER SUGGESTION DATA STRUCTURE
# ==============================================================================

@dataclass
class ParameterSuggestion:
    """
    A suggested parameter to add to the system based on reconstruction gaps.
    """
    # Core identification
    suggested_parameter: str  # Full path, e.g., "harmony.voicing.quartal_probability"

    # Analysis results
    affected_features: List[str]
    avg_error: float
    impact_score: float  # 0-1: How much this would improve reconstruction
    confidence: float  # 0-1: How confident we are in this suggestion
    priority: str  # "HIGH", "MEDIUM", "LOW"

    # Explanations
    rationale: str  # Human-readable explanation
    parameter_info: Dict[str, Any]  # Type, range, usage, etc.

    # Supporting data
    feature_errors: Dict[str, float] = field(default_factory=dict)
    correlation_strength: float = 0.0
    existing_coverage: float = 0.0  # How much of this domain is already covered


# ==============================================================================
# INTELLIGENT GAP DETECTOR
# ==============================================================================

class IntelligentGapDetector:
    """
    Analyzes reconstruction failures and suggests minimal parameter additions.

    This is the core system that enables self-expansion:
    1. Receives feature reconstruction errors from XGBoost
    2. Groups correlated features
    3. Maps feature groups to missing parameters
    4. Prioritizes suggestions by impact
    5. Generates human-readable rationales
    """

    def __init__(self, registry: Optional[UniversalParameterRegistry] = None):
        """
        Initialize the gap detector.

        Args:
            registry: UniversalParameterRegistry instance (creates new if None)
        """
        self.registry = registry or UniversalParameterRegistry()
        self.mapper = FeatureToParameterMapper()
        self.existing_params = set(self.registry.get_all_parameters())

        # Cache for correlation analysis
        self._correlation_cache: Dict[str, np.ndarray] = {}

        logger.info(f"Initialized IntelligentGapDetector with {len(self.existing_params)} existing parameters")

    def detect_gaps(
        self,
        feature_errors: Dict[str, float],
        threshold: float = 0.3,
        min_confidence: float = 0.5,
        max_suggestions: int = 10
    ) -> List[ParameterSuggestion]:
        """
        Identify parameter gaps from reconstruction errors.

        Args:
            feature_errors: Dictionary mapping feature names to reconstruction errors (0-1)
                          e.g., {'quartal_voicing_count': 0.78, 'swing_ratio_detected': 0.05}
            threshold: Minimum error to consider (default 0.3 = 30% error)
            min_confidence: Minimum confidence for suggestions (default 0.5)
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of ParameterSuggestion objects, sorted by impact score
        """

        if not feature_errors:
            logger.warning("No feature errors provided to detect_gaps")
            return []

        logger.info(f"Detecting gaps from {len(feature_errors)} feature errors")

        # 1. Filter high-error features
        high_error_features = {
            feature: error
            for feature, error in feature_errors.items()
            if error > threshold
        }

        if not high_error_features:
            logger.info(f"No features above error threshold {threshold}")
            return []

        logger.info(f"Found {len(high_error_features)} features above threshold {threshold}")

        # 2. Group correlated features
        feature_groups = self._group_correlated_features(high_error_features)
        logger.info(f"Grouped features into {len(feature_groups)} groups")

        # 3. Map groups to parameters
        parameter_suggestions = []

        for i, group in enumerate(feature_groups):
            suggestion = self._analyze_feature_group(group, feature_errors)
            if suggestion and suggestion.confidence >= min_confidence:
                parameter_suggestions.append(suggestion)
                logger.debug(f"Group {i}: Suggested {suggestion.suggested_parameter} "
                           f"(impact={suggestion.impact_score:.2f}, confidence={suggestion.confidence:.2f})")

        # 4. Filter out existing parameters
        new_suggestions = [
            s for s in parameter_suggestions
            if s.suggested_parameter not in self.existing_params
        ]

        logger.info(f"Found {len(new_suggestions)} new parameter suggestions "
                   f"({len(parameter_suggestions) - len(new_suggestions)} already exist)")

        # 5. Remove duplicate suggestions (same parameter suggested by different groups)
        unique_suggestions = self._deduplicate_suggestions(new_suggestions)

        # 6. Prioritize by impact
        unique_suggestions.sort(key=lambda x: x.impact_score, reverse=True)

        # 7. Limit to max_suggestions
        final_suggestions = unique_suggestions[:max_suggestions]

        logger.info(f"Returning {len(final_suggestions)} suggestions")

        return final_suggestions

    def _group_correlated_features(
        self,
        high_error_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Group features that likely share the same underlying parameter.

        Uses three grouping strategies:
        1. Features with same primary parameters
        2. Features with high correlation in the mapping
        3. Features in the same musical domain

        Args:
            high_error_features: Dictionary of features with high errors

        Returns:
            List of feature groups, each containing:
            - 'features': List of feature names in the group
            - 'common_parameters': List of parameters that affect these features
            - 'musical_domain': The shared musical domain
        """

        groups = []
        used_features = set()

        for feature in high_error_features:
            if feature in used_features:
                continue

            # Get mapping for this feature
            mapping = self.mapper.get_mapping(feature)
            if not mapping:
                logger.debug(f"No mapping found for feature: {feature}")
                continue

            primary_params = mapping.primary_parameters
            musical_domain = mapping.musical_domain

            # Find all features controlled by same primary parameters
            related_features = [feature]

            for other_feature in high_error_features:
                if other_feature == feature or other_feature in used_features:
                    continue

                other_mapping = self.mapper.get_mapping(other_feature)
                if not other_mapping:
                    continue

                other_params = other_mapping.primary_parameters

                # Check if they share any primary parameters or musical domain
                shares_params = bool(set(primary_params) & set(other_params))
                shares_domain = other_mapping.musical_domain == musical_domain

                if shares_params or (shares_domain and len(related_features) < 3):
                    related_features.append(other_feature)
                    used_features.add(other_feature)

            used_features.add(feature)

            # Build parameter set from all related features
            all_params = set(primary_params)
            for rf in related_features:
                rf_mapping = self.mapper.get_mapping(rf)
                if rf_mapping:
                    all_params.update(rf_mapping.primary_parameters)

            groups.append({
                'features': related_features,
                'common_parameters': list(all_params),
                'musical_domain': musical_domain,
                'anchor_feature': feature  # The first feature that started this group
            })

        return groups

    def _analyze_feature_group(
        self,
        group: Dict[str, Any],
        all_errors: Dict[str, float]
    ) -> Optional[ParameterSuggestion]:
        """
        Analyze a feature group and suggest a parameter.

        Args:
            group: Feature group from _group_correlated_features
            all_errors: All feature errors for context

        Returns:
            ParameterSuggestion or None if no good suggestion
        """

        features = group['features']
        common_params = group['common_parameters']

        if not common_params:
            return None

        # Calculate average error for this group
        feature_errors = {f: all_errors[f] for f in features if f in all_errors}
        avg_error = np.mean(list(feature_errors.values()))

        # Choose best parameter suggestion
        # Prefer parameters that affect most features in group
        best_param = None
        max_affected = 0

        for param in common_params:
            affected = [
                f for f in features
                if param in self.mapper.get_parameters_for_feature(f)
            ]
            if len(affected) > max_affected:
                max_affected = len(affected)
                best_param = param

        if not best_param:
            return None

        # Calculate impact score
        # Impact = (avg_error * sqrt(num_affected_features)) / sqrt(total_high_error_features)
        # Sqrt to prevent over-weighting large groups
        impact_score = (avg_error * np.sqrt(len(features))) / np.sqrt(len(all_errors))
        impact_score = min(1.0, impact_score)  # Cap at 1.0

        # Determine priority
        if impact_score > 0.5:
            priority = 'HIGH'
        elif impact_score > 0.2:
            priority = 'MEDIUM'
        else:
            priority = 'LOW'

        # Calculate confidence
        # High confidence if:
        # - Many related features (more evidence)
        # - High average error (clear gap)
        # - Low variance in errors (consistent problem)
        error_variance = np.var(list(feature_errors.values())) if len(feature_errors) > 1 else 0

        confidence = min(1.0, (
            (len(features) / 5) * 0.4 +  # More features = more confidence (up to 5)
            (avg_error ** 0.5) * 0.4 +    # Higher error = more confidence
            (1.0 - min(error_variance, 0.2) / 0.2) * 0.2  # Lower variance = more confidence
        ))

        # Generate rationale
        rationale = self._generate_rationale(best_param, features, avg_error, group['musical_domain'])

        # Get parameter info
        parameter_info = self._get_parameter_info(best_param)

        # Calculate correlation strength
        correlation_strength = self._calculate_correlation_strength(features, feature_errors)

        # Calculate existing coverage
        existing_coverage = self._calculate_existing_coverage(group['musical_domain'], common_params)

        return ParameterSuggestion(
            suggested_parameter=best_param,
            affected_features=features,
            avg_error=avg_error,
            impact_score=impact_score,
            confidence=confidence,
            priority=priority,
            rationale=rationale,
            parameter_info=parameter_info,
            feature_errors=feature_errors,
            correlation_strength=correlation_strength,
            existing_coverage=existing_coverage
        )

    def _generate_rationale(
        self,
        param: str,
        features: List[str],
        avg_error: float,
        musical_domain: str
    ) -> str:
        """
        Generate human-readable rationale for parameter suggestion.

        Args:
            param: Parameter path
            features: List of affected feature names
            avg_error: Average reconstruction error
            musical_domain: Musical domain (harmony, rhythm, etc.)

        Returns:
            Human-readable rationale string
        """

        # Get mapping for first feature
        first_feature = features[0]
        mapping = self.mapper.get_mapping(first_feature)

        if not mapping:
            return f"Parameter '{param}' would help reconstruct {len(features)} features " \
                   f"with average error {avg_error:.2f}."

        # Build rationale
        domain_name = musical_domain.capitalize() if musical_domain else "Musical"

        rationale = f"{domain_name} reconstruction gap detected. "
        rationale += f"Input MIDI shows strong presence of {mapping.musical_rationale.split('.')[0].lower()}, "
        rationale += f"but current system cannot reproduce it (avg error: {avg_error:.2f}). "

        # List some affected features
        rationale += f"Affected features: "
        if len(features) <= 3:
            rationale += ", ".join([f"'{f}'" for f in features])
        else:
            rationale += ", ".join([f"'{f}'" for f in features[:3]])
            rationale += f" and {len(features)-3} others"

        rationale += f". Adding '{param}' parameter would enable reconstruction of this musical characteristic."

        # Add usage context if available
        if mapping.typical_usage:
            rationale += f" Typical usage: {mapping.typical_usage}."

        return rationale

    def _get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """
        Get detailed info about suggested parameter from mapping.

        Args:
            param_name: Parameter path

        Returns:
            Dictionary with parameter metadata
        """

        # Find which features this parameter affects
        affected_features = self.mapper.get_features_for_parameter(param_name)

        if not affected_features:
            return {
                'type': 'unknown',
                'range': [0.0, 1.0],
                'musical_rationale': 'Suggested parameter',
                'typical_usage': 'Various',
                'affected_features_count': 0
            }

        # Get example feature's info
        example_feature = affected_features[0]
        feature_mapping = self.mapper.get_mapping(example_feature)

        if not feature_mapping:
            return {
                'type': 'unknown',
                'range': [0.0, 1.0],
                'affected_features_count': len(affected_features)
            }

        # Infer parameter type from parameter name
        param_type = self._infer_parameter_type(param_name)

        return {
            'type': param_type,
            'range': list(feature_mapping.typical_range),
            'musical_rationale': feature_mapping.musical_rationale,
            'typical_usage': feature_mapping.typical_usage,
            'affected_features_count': len(affected_features),
            'musical_domain': feature_mapping.musical_domain,
            'importance': feature_mapping.importance
        }

    def _infer_parameter_type(self, param_name: str) -> str:
        """Infer parameter type from name"""
        name_lower = param_name.lower()

        if 'probability' in name_lower or 'ratio' in name_lower:
            return 'probability'
        elif 'count' in name_lower or 'density' in name_lower:
            return 'integer'
        elif 'enabled' in name_lower or 'use_' in name_lower:
            return 'boolean'
        elif 'type' in name_lower:
            return 'categorical'
        else:
            return 'continuous'

    def _calculate_correlation_strength(
        self,
        features: List[str],
        feature_errors: Dict[str, float]
    ) -> float:
        """
        Calculate correlation strength between features in group.

        Returns a value 0-1 indicating how correlated the errors are.
        High correlation = they likely share a common cause (missing parameter).
        """

        if len(features) < 2:
            return 1.0

        # For now, return a simple measure based on error variance
        # Low variance = similar errors = high correlation
        errors = [feature_errors.get(f, 0) for f in features]
        error_variance = np.var(errors)

        # Normalize: low variance -> high correlation
        correlation = max(0.0, 1.0 - error_variance / 0.1)

        return correlation

    def _calculate_existing_coverage(
        self,
        musical_domain: str,
        suggested_params: List[str]
    ) -> float:
        """
        Calculate how much of this musical domain is already covered by existing parameters.

        Returns 0-1, where:
        - 0.0 = No existing parameters in this domain
        - 1.0 = This domain is fully covered

        This helps identify truly new capabilities vs. refinements.
        """

        if not musical_domain:
            return 0.5  # Unknown

        # Count existing parameters in this domain
        domain_params = [
            p for p in self.existing_params
            if p.startswith(musical_domain + '.')
        ]

        # Count suggested parameters in this domain
        suggested_in_domain = [
            p for p in suggested_params
            if p.startswith(musical_domain + '.')
        ]

        if not suggested_in_domain:
            return 1.0  # Not in this domain, so domain is "covered"

        # Calculate coverage ratio
        coverage = len(domain_params) / (len(domain_params) + len(suggested_in_domain))

        return coverage

    def _deduplicate_suggestions(
        self,
        suggestions: List[ParameterSuggestion]
    ) -> List[ParameterSuggestion]:
        """
        Remove duplicate parameter suggestions, keeping the one with highest impact.
        """

        # Group by parameter name
        param_groups = defaultdict(list)
        for suggestion in suggestions:
            param_groups[suggestion.suggested_parameter].append(suggestion)

        # Keep best from each group
        unique = []
        for param, group in param_groups.items():
            best = max(group, key=lambda x: x.impact_score)

            # If there are multiple suggestions for same parameter, combine their features
            if len(group) > 1:
                all_features = []
                all_errors = {}
                for sugg in group:
                    all_features.extend(sugg.affected_features)
                    all_errors.update(sugg.feature_errors)

                best.affected_features = list(set(all_features))
                best.feature_errors = all_errors
                best.avg_error = np.mean(list(all_errors.values()))

            unique.append(best)

        return unique

    def detect_systematic_gaps(
        self,
        historical_errors: List[Dict[str, float]],
        threshold: float = 0.35,
        consistency_threshold: float = 0.7
    ) -> List[ParameterSuggestion]:
        """
        Analyze patterns across multiple MIDI files to find systematic gaps.

        A systematic gap is a feature that consistently has high error across
        many different pieces, indicating a fundamental capability gap.

        Args:
            historical_errors: List of feature_errors dicts from multiple reconstructions
            threshold: Minimum average error to consider (default 0.35)
            consistency_threshold: Minimum ratio of pieces that must show the gap (default 0.7)

        Returns:
            List of ParameterSuggestion objects for systematic gaps
        """

        if not historical_errors:
            logger.warning("No historical errors provided")
            return []

        logger.info(f"Analyzing systematic gaps across {len(historical_errors)} pieces")

        # Aggregate errors across all pieces
        aggregate_errors = defaultdict(list)

        for errors in historical_errors:
            for feature, error in errors.items():
                aggregate_errors[feature].append(error)

        # Find features with consistently high errors
        systematic_features = {}

        for feature, error_list in aggregate_errors.items():
            avg_error = np.mean(error_list)
            consistency = np.sum(np.array(error_list) > threshold) / len(error_list)

            # High average error + high consistency = systematic gap
            if avg_error > threshold and consistency >= consistency_threshold:
                systematic_features[feature] = avg_error

        logger.info(f"Found {len(systematic_features)} features with systematic gaps")

        # Generate suggestions for systematic gaps
        suggestions = self.detect_gaps(
            systematic_features,
            threshold=threshold,
            min_confidence=0.6  # Higher confidence threshold for systematic gaps
        )

        # Mark these as systematic
        for suggestion in suggestions:
            suggestion.rationale = f"SYSTEMATIC GAP (found across {len(historical_errors)} pieces): " + suggestion.rationale
            suggestion.confidence = min(1.0, suggestion.confidence * 1.2)  # Boost confidence
            suggestion.priority = 'HIGH'  # Systematic gaps are high priority

        return suggestions

    def generate_report(
        self,
        suggestions: List[ParameterSuggestion],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate human-readable report of parameter suggestions.

        Args:
            suggestions: List of ParameterSuggestion objects
            output_file: Optional file path to write report

        Returns:
            Report text as string
        """

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("INTELLIGENT GAP DETECTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Total Suggestions: {len(suggestions)}")
        report_lines.append(f"Generated: {np.datetime64('now')}")
        report_lines.append("")

        # Group by priority
        high_priority = [s for s in suggestions if s.priority == 'HIGH']
        medium_priority = [s for s in suggestions if s.priority == 'MEDIUM']
        low_priority = [s for s in suggestions if s.priority == 'LOW']

        report_lines.append(f"Priority Breakdown:")
        report_lines.append(f"  HIGH:   {len(high_priority)}")
        report_lines.append(f"  MEDIUM: {len(medium_priority)}")
        report_lines.append(f"  LOW:    {len(low_priority)}")
        report_lines.append("")

        # Detailed suggestions
        for priority_group, name in [(high_priority, 'HIGH'), (medium_priority, 'MEDIUM'), (low_priority, 'LOW')]:
            if not priority_group:
                continue

            report_lines.append("=" * 80)
            report_lines.append(f"{name} PRIORITY SUGGESTIONS")
            report_lines.append("=" * 80)
            report_lines.append("")

            for i, suggestion in enumerate(priority_group, 1):
                report_lines.append(f"{i}. {suggestion.suggested_parameter}")
                report_lines.append(f"   Impact: {suggestion.impact_score:.2f} | Confidence: {suggestion.confidence:.2f}")
                report_lines.append(f"   Avg Error: {suggestion.avg_error:.2f} | Affected Features: {len(suggestion.affected_features)}")
                report_lines.append("")
                report_lines.append(f"   Rationale:")
                for line in self._wrap_text(suggestion.rationale, 70):
                    report_lines.append(f"     {line}")
                report_lines.append("")

                if suggestion.parameter_info:
                    info = suggestion.parameter_info
                    report_lines.append(f"   Parameter Info:")
                    report_lines.append(f"     Type: {info.get('type', 'unknown')}")
                    report_lines.append(f"     Range: {info.get('range', [0, 1])}")
                    report_lines.append(f"     Domain: {info.get('musical_domain', 'unknown')}")
                    report_lines.append(f"     Importance: {info.get('importance', 0.5):.2f}")
                    report_lines.append("")

                report_lines.append(f"   Top Affected Features:")
                sorted_features = sorted(
                    suggestion.feature_errors.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for feature, error in sorted_features:
                    report_lines.append(f"     - {feature}: {error:.2f}")
                report_lines.append("")
                report_lines.append("-" * 80)
                report_lines.append("")

        report_text = "\n".join(report_lines)

        if output_file:
            Path(output_file).write_text(report_text)
            logger.info(f"Report written to {output_file}")

        return report_text

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Simple text wrapping"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def export_suggestions_json(
        self,
        suggestions: List[ParameterSuggestion],
        output_file: str
    ):
        """
        Export suggestions to JSON for programmatic use.

        Args:
            suggestions: List of ParameterSuggestion objects
            output_file: Path to output JSON file
        """

        data = {
            'metadata': {
                'generated_at': str(np.datetime64('now')),
                'total_suggestions': len(suggestions),
                'existing_parameters': len(self.existing_params)
            },
            'suggestions': [
                {
                    'parameter': s.suggested_parameter,
                    'priority': s.priority,
                    'impact_score': float(s.impact_score),
                    'confidence': float(s.confidence),
                    'avg_error': float(s.avg_error),
                    'affected_features': s.affected_features,
                    'feature_errors': {k: float(v) for k, v in s.feature_errors.items()},
                    'rationale': s.rationale,
                    'parameter_info': s.parameter_info,
                    'correlation_strength': float(s.correlation_strength),
                    'existing_coverage': float(s.existing_coverage)
                }
                for s in suggestions
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Suggestions exported to {output_file}")


# ==============================================================================
# ADVANCED GAP ANALYSIS
# ==============================================================================

class GapPredictor:
    """
    Predicts future gaps based on historical patterns.

    This analyzes which types of gaps tend to appear together and predicts
    which gaps might emerge as the system expands.
    """

    def __init__(self):
        self.gap_history: List[Dict[str, Any]] = []
        self.co_occurrence_matrix: Dict[Tuple[str, str], int] = defaultdict(int)

    def record_gap(self, suggestion: ParameterSuggestion, timestamp: Optional[str] = None):
        """Record a detected gap for future prediction"""
        if timestamp is None:
            timestamp = str(np.datetime64('now'))

        self.gap_history.append({
            'timestamp': timestamp,
            'parameter': suggestion.suggested_parameter,
            'domain': suggestion.parameter_info.get('musical_domain', 'unknown'),
            'impact': suggestion.impact_score,
            'confidence': suggestion.confidence
        })

        # Update co-occurrence matrix
        for other_gap in self.gap_history[-10:]:  # Last 10 gaps
            if other_gap['parameter'] != suggestion.suggested_parameter:
                pair = tuple(sorted([suggestion.suggested_parameter, other_gap['parameter']]))
                self.co_occurrence_matrix[pair] += 1

    def predict_related_gaps(
        self,
        current_suggestion: ParameterSuggestion,
        top_k: int = 5
    ) -> List[str]:
        """
        Predict which other parameters might need to be added based on this suggestion.

        Args:
            current_suggestion: The current gap being addressed
            top_k: Number of predictions to return

        Returns:
            List of parameter paths that might be needed next
        """
        param = current_suggestion.suggested_parameter

        # Find parameters that commonly co-occur with this one
        related = []
        for (p1, p2), count in self.co_occurrence_matrix.items():
            if p1 == param:
                related.append((p2, count))
            elif p2 == param:
                related.append((p1, count))

        # Sort by co-occurrence count
        related.sort(key=lambda x: x[1], reverse=True)

        return [p for p, _ in related[:top_k]]

    def get_gap_trends(self) -> Dict[str, Any]:
        """Analyze trends in gap detection"""
        if not self.gap_history:
            return {}

        domains = [g['domain'] for g in self.gap_history]
        domain_counts = {}
        for d in domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1

        return {
            'total_gaps_detected': len(self.gap_history),
            'domains': domain_counts,
            'most_common_domain': max(domain_counts.items(), key=lambda x: x[1])[0] if domain_counts else None,
            'avg_impact': np.mean([g['impact'] for g in self.gap_history]),
            'avg_confidence': np.mean([g['confidence'] for g in self.gap_history])
        }


class GapTracker:
    """
    Tracks gaps over time to monitor system expansion.

    This maintains a history of which gaps were detected when,
    and tracks the system's progress in filling gaps.
    """

    def __init__(self, storage_file: Optional[str] = None):
        self.storage_file = storage_file
        self.detected_gaps: List[Dict[str, Any]] = []
        self.filled_gaps: List[Dict[str, Any]] = []
        self.ignored_gaps: List[Dict[str, Any]] = []

        if storage_file and Path(storage_file).exists():
            self._load_from_file()

    def record_detection(
        self,
        suggestions: List[ParameterSuggestion],
        context: Optional[Dict[str, Any]] = None
    ):
        """Record newly detected gaps"""
        timestamp = str(np.datetime64('now'))

        for suggestion in suggestions:
            self.detected_gaps.append({
                'timestamp': timestamp,
                'parameter': suggestion.suggested_parameter,
                'impact': suggestion.impact_score,
                'confidence': suggestion.confidence,
                'priority': suggestion.priority,
                'affected_features': suggestion.affected_features,
                'status': 'detected',
                'context': context or {}
            })

        if self.storage_file:
            self._save_to_file()

    def mark_filled(self, parameter_path: str, notes: Optional[str] = None):
        """Mark a gap as filled (parameter was added)"""
        timestamp = str(np.datetime64('now'))

        # Find in detected gaps
        for gap in self.detected_gaps:
            if gap['parameter'] == parameter_path and gap['status'] == 'detected':
                gap['status'] = 'filled'
                gap['filled_timestamp'] = timestamp
                gap['notes'] = notes

                self.filled_gaps.append(gap.copy())
                break

        if self.storage_file:
            self._save_to_file()

    def mark_ignored(self, parameter_path: str, reason: str):
        """Mark a gap as intentionally not filled"""
        timestamp = str(np.datetime64('now'))

        for gap in self.detected_gaps:
            if gap['parameter'] == parameter_path and gap['status'] == 'detected':
                gap['status'] = 'ignored'
                gap['ignored_timestamp'] = timestamp
                gap['ignore_reason'] = reason

                self.ignored_gaps.append(gap.copy())
                break

        if self.storage_file:
            self._save_to_file()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about gap detection and filling"""
        total_detected = len(self.detected_gaps)
        total_filled = len(self.filled_gaps)
        total_ignored = len(self.ignored_gaps)
        still_open = len([g for g in self.detected_gaps if g['status'] == 'detected'])

        fill_rate = total_filled / total_detected if total_detected > 0 else 0

        return {
            'total_detected': total_detected,
            'total_filled': total_filled,
            'total_ignored': total_ignored,
            'still_open': still_open,
            'fill_rate': fill_rate,
            'avg_impact': np.mean([g['impact'] for g in self.detected_gaps]) if self.detected_gaps else 0,
            'high_priority_open': len([g for g in self.detected_gaps
                                      if g['status'] == 'detected' and g['priority'] == 'HIGH'])
        }

    def _save_to_file(self):
        """Save tracking data to file"""
        data = {
            'detected_gaps': self.detected_gaps,
            'filled_gaps': self.filled_gaps,
            'ignored_gaps': self.ignored_gaps
        }
        with open(self.storage_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_from_file(self):
        """Load tracking data from file"""
        with open(self.storage_file, 'r') as f:
            data = json.load(f)
            self.detected_gaps = data.get('detected_gaps', [])
            self.filled_gaps = data.get('filled_gaps', [])
            self.ignored_gaps = data.get('ignored_gaps', [])


class AdvancedCorrelationAnalyzer:
    """
    Advanced correlation analysis for feature grouping.

    Provides multiple correlation methods and hierarchical clustering
    to identify related features.
    """

    def __init__(self):
        self.correlation_cache: Dict[str, np.ndarray] = {}

    def compute_feature_correlations(
        self,
        feature_matrix: np.ndarray,
        feature_names: List[str],
        method: str = 'pearson'
    ) -> np.ndarray:
        """
        Compute correlation matrix between features.

        Args:
            feature_matrix: NxM matrix (N samples, M features)
            feature_names: List of feature names
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            MxM correlation matrix
        """
        cache_key = f"{method}_{len(feature_names)}"

        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]

        n_features = feature_matrix.shape[1]
        correlation_matrix = np.zeros((n_features, n_features))

        if method == 'pearson':
            correlation_matrix = np.corrcoef(feature_matrix.T)
        elif method == 'spearman':
            # Rank-based correlation
            from scipy.stats import spearmanr
            correlation_matrix, _ = spearmanr(feature_matrix, axis=0)
        else:
            # Fall back to pearson
            correlation_matrix = np.corrcoef(feature_matrix.T)

        self.correlation_cache[cache_key] = correlation_matrix
        return correlation_matrix

    def hierarchical_clustering(
        self,
        correlation_matrix: np.ndarray,
        feature_names: List[str],
        n_clusters: int = 5
    ) -> Dict[int, List[str]]:
        """
        Perform hierarchical clustering on features.

        Args:
            correlation_matrix: MxM correlation matrix
            feature_names: List of M feature names
            n_clusters: Number of clusters to create

        Returns:
            Dictionary mapping cluster ID to list of feature names
        """
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)

        # Hierarchical clustering
        linkage_matrix = linkage(distance_matrix[np.triu_indices_from(distance_matrix, k=1)], method='average')

        # Cut tree to get clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Group features by cluster
        clusters = defaultdict(list)
        for feature_name, cluster_id in zip(feature_names, cluster_labels):
            clusters[cluster_id].append(feature_name)

        return dict(clusters)

    def find_highly_correlated_groups(
        self,
        correlation_matrix: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.7
    ) -> List[List[str]]:
        """
        Find groups of highly correlated features.

        Args:
            correlation_matrix: MxM correlation matrix
            feature_names: List of M feature names
            threshold: Minimum correlation to be considered related (default 0.7)

        Returns:
            List of feature groups (each group is a list of feature names)
        """
        n_features = len(feature_names)
        visited = set()
        groups = []

        for i in range(n_features):
            if i in visited:
                continue

            # Start a new group with this feature
            group = [feature_names[i]]
            visited.add(i)

            # Find all features correlated with this one
            for j in range(i + 1, n_features):
                if j in visited:
                    continue

                if abs(correlation_matrix[i, j]) >= threshold:
                    group.append(feature_names[j])
                    visited.add(j)

            if len(group) > 1:  # Only keep groups with multiple features
                groups.append(group)

        return groups


class GapVisualizationHelper:
    """
    Helper class for generating visualizations of gaps and suggestions.

    Note: Requires matplotlib (optional dependency)
    """

    def __init__(self):
        self.has_matplotlib = False
        try:
            import matplotlib
            self.has_matplotlib = True
        except ImportError:
            logger.warning("matplotlib not available - visualization disabled")

    def plot_suggestion_distribution(
        self,
        suggestions: List[ParameterSuggestion],
        output_file: Optional[str] = None
    ):
        """Plot distribution of suggestions by domain, priority, and impact"""
        if not self.has_matplotlib:
            logger.warning("Cannot plot - matplotlib not installed")
            return

        import matplotlib.pyplot as plt

        # Extract data
        domains = [s.parameter_info.get('musical_domain', 'unknown') for s in suggestions]
        priorities = [s.priority for s in suggestions]
        impacts = [s.impact_score for s in suggestions]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Domain distribution
        domain_counts = {}
        for d in domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1

        axes[0, 0].bar(domain_counts.keys(), domain_counts.values())
        axes[0, 0].set_title('Suggestions by Musical Domain')
        axes[0, 0].set_xlabel('Domain')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Priority distribution
        priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for p in priorities:
            priority_counts[p] = priority_counts.get(p, 0) + 1

        axes[0, 1].bar(priority_counts.keys(), priority_counts.values(), color=['red', 'orange', 'yellow'])
        axes[0, 1].set_title('Suggestions by Priority')
        axes[0, 1].set_xlabel('Priority')
        axes[0, 1].set_ylabel('Count')

        # Impact distribution
        axes[1, 0].hist(impacts, bins=20, edgecolor='black')
        axes[1, 0].set_title('Impact Score Distribution')
        axes[1, 0].set_xlabel('Impact Score')
        axes[1, 0].set_ylabel('Count')

        # Impact vs Confidence scatter
        confidences = [s.confidence for s in suggestions]
        colors = ['red' if s.priority == 'HIGH' else 'orange' if s.priority == 'MEDIUM' else 'yellow'
                 for s in suggestions]

        axes[1, 1].scatter(impacts, confidences, c=colors, alpha=0.6, s=100)
        axes[1, 1].set_title('Impact vs Confidence')
        axes[1, 1].set_xlabel('Impact Score')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        else:
            plt.show()


# ==============================================================================
# INTEGRATION UTILITIES
# ==============================================================================

class XGBoostIntegration:
    """
    Integration utilities for connecting gap detection with XGBoost training.

    This bridges the gap detection system with the XGBoost synthesizer,
    enabling automated parameter expansion.
    """

    def __init__(self, detector: IntelligentGapDetector):
        self.detector = detector

    def analyze_prediction_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Convert XGBoost prediction errors to feature error dictionary.

        Args:
            y_true: True feature values (N x M)
            y_pred: Predicted feature values (N x M)
            feature_names: List of M feature names

        Returns:
            Dictionary mapping feature names to normalized errors (0-1)
        """
        # Calculate per-feature error
        feature_errors = {}

        for i, feature_name in enumerate(feature_names):
            true_values = y_true[:, i]
            pred_values = y_pred[:, i]

            # Calculate RMSE
            rmse = np.sqrt(np.mean((true_values - pred_values) ** 2))

            # Normalize by feature range
            feature_range = np.max(true_values) - np.min(true_values)
            if feature_range > 0:
                normalized_error = min(1.0, rmse / feature_range)
            else:
                normalized_error = 0.0

            feature_errors[feature_name] = normalized_error

        return feature_errors

    def suggest_parameters_for_training(
        self,
        training_errors: Dict[str, float],
        threshold: float = 0.3
    ) -> Tuple[List[ParameterSuggestion], str]:
        """
        Analyze training errors and suggest parameters to add.

        Args:
            training_errors: Feature reconstruction errors from training
            threshold: Minimum error threshold

        Returns:
            Tuple of (suggestions, report_text)
        """
        suggestions = self.detector.detect_gaps(training_errors, threshold=threshold)
        report = self.detector.generate_report(suggestions)

        return suggestions, report

    def generate_training_data_spec(
        self,
        suggestion: ParameterSuggestion
    ) -> Dict[str, Any]:
        """
        Generate specification for synthetic training data generation.

        When a new parameter is suggested, this generates a spec that can be
        used to create synthetic training data for that parameter.

        Args:
            suggestion: ParameterSuggestion object

        Returns:
            Dictionary with training data generation specification
        """
        param_info = suggestion.parameter_info

        # Determine parameter space to sample
        param_type = param_info.get('type', 'continuous')
        param_range = param_info.get('range', [0.0, 1.0])

        if param_type == 'probability':
            sample_values = np.linspace(0.0, 1.0, 10)
        elif param_type == 'boolean':
            sample_values = [False, True]
        elif param_type == 'categorical':
            sample_values = param_info.get('options', ['default'])
        elif param_type == 'integer':
            sample_values = np.linspace(param_range[0], param_range[1], 10, dtype=int)
        else:
            sample_values = np.linspace(param_range[0], param_range[1], 10)

        return {
            'parameter': suggestion.suggested_parameter,
            'type': param_type,
            'sample_values': sample_values.tolist() if isinstance(sample_values, np.ndarray) else sample_values,
            'affected_features': suggestion.affected_features,
            'musical_domain': param_info.get('musical_domain', 'unknown'),
            'typical_usage': param_info.get('typical_usage', ''),
            'samples_needed': len(sample_values) * 10,  # 10 samples per value
            'priority': suggestion.priority
        }


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def detect_gaps_from_errors(
    feature_errors: Dict[str, float],
    threshold: float = 0.3,
    max_suggestions: int = 10
) -> List[ParameterSuggestion]:
    """
    Convenience function to detect gaps from feature errors.

    Args:
        feature_errors: Dictionary of feature names to reconstruction errors
        threshold: Minimum error threshold (default 0.3)
        max_suggestions: Maximum number of suggestions (default 10)

    Returns:
        List of ParameterSuggestion objects
    """
    detector = IntelligentGapDetector()
    return detector.detect_gaps(feature_errors, threshold=threshold, max_suggestions=max_suggestions)


def analyze_systematic_gaps(
    historical_errors: List[Dict[str, float]],
    threshold: float = 0.35
) -> List[ParameterSuggestion]:
    """
    Convenience function to analyze systematic gaps across multiple pieces.

    Args:
        historical_errors: List of feature error dictionaries
        threshold: Minimum average error threshold (default 0.35)

    Returns:
        List of ParameterSuggestion objects for systematic gaps
    """
    detector = IntelligentGapDetector()
    return detector.detect_systematic_gaps(historical_errors, threshold=threshold)


def create_full_pipeline() -> Dict[str, Any]:
    """
    Create a complete gap detection pipeline with all components.

    Returns:
        Dictionary containing all initialized components
    """
    detector = IntelligentGapDetector()
    tracker = GapTracker()
    predictor = GapPredictor()
    visualizer = GapVisualizationHelper()
    xgboost_integration = XGBoostIntegration(detector)

    return {
        'detector': detector,
        'tracker': tracker,
        'predictor': predictor,
        'visualizer': visualizer,
        'xgboost_integration': xgboost_integration,
        'mapper': detector.mapper
    }


# ==============================================================================
# MAIN / TESTING
# ==============================================================================

if __name__ == "__main__":
    # Example usage
    print("Intelligent Gap Detector - Agent 10")
    print("=" * 80)

    # Create detector
    detector = IntelligentGapDetector()

    # Example: Simulate reconstruction errors for a jazz piece with quartal harmony
    feature_errors = {
        # High errors - missing quartal harmony
        'quartal_voicing_count': 0.82,
        'fourth_interval_ratio': 0.76,
        'open_voicing_ratio': 0.65,

        # High errors - missing swing
        'swing_ratio_detected': 0.91,
        'syncopation_density': 0.68,

        # High errors - missing bebop characteristics
        'bebop_scale_usage': 0.88,
        'chromatic_approach_count': 0.79,
        'altered_extension_ratio': 0.73,

        # Low errors - already handled well
        'stepwise_motion_ratio': 0.12,
        'voice_leading_smoothness_score': 0.15,
        'ninth_extension_ratio': 0.18,

        # Medium errors
        'walking_bass_detected': 0.45,
        'drop2_voicing_count': 0.38,
    }

    print(f"\nAnalyzing {len(feature_errors)} feature errors...")

    # Detect gaps
    suggestions = detector.detect_gaps(feature_errors, threshold=0.3, max_suggestions=10)

    print(f"\nFound {len(suggestions)} parameter suggestions")
    print("\nTop 5 Suggestions:")
    print("-" * 80)

    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"\n{i}. {suggestion.suggested_parameter}")
        print(f"   Priority: {suggestion.priority}")
        print(f"   Impact: {suggestion.impact_score:.2f} | Confidence: {suggestion.confidence:.2f}")
        print(f"   Affected Features: {len(suggestion.affected_features)}")
        print(f"   Rationale: {suggestion.rationale[:150]}...")

    # Generate full report
    print("\n" + "=" * 80)
    print("Generating full report...")
    report = detector.generate_report(suggestions)
    print("\n[Report generated - showing first 1000 chars]")
    print(report[:1000])
    print("\n[... report continues ...]")

    print("\n" + "=" * 80)
    print("Agent 10: Intelligent Gap Detector - Ready for Integration")
    print("=" * 80)
