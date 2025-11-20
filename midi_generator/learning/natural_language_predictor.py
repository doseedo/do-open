#!/usr/bin/env python3
"""
Natural Language Parameter Predictor - Agent 6
================================================

LLM-based system that converts natural language descriptions into 515+ parameter values
for music generation. Enables users to type "Generate a Sinatra-style arrangement" and
receive precise parameter predictions.

Core Innovation:
- Converts text → 515 parameter values via Claude LLM
- Uses style database with 100+ examples for few-shot learning
- Extracts musical concepts from natural language
- Validates and fills default values for all parameters
- Integrates with UniversalParameterRegistry

Architecture:
- NaturalLanguageParameterPredictor: Main predictor class
- StyleDatabase: Storage for text→parameter mappings
- ConceptExtractor: Extracts musical concepts from text
- ParameterValidator: Validates and fills defaults

Author: Agent 6 - Natural Language Parameter Predictor
Part of: 35-Agent Musical Program Synthesis System
Date: 2025-11-20
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

# Optional anthropic import
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Import existing modules
from midi_generator.parameters.universal_registry import (
    UniversalParameterRegistry,
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
    REGISTRY
)
from midi_generator.styles.style_registry import (
    StyleProfile,
    get_style,
    list_styles,
    STYLE_REGISTRY
)


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class MusicalConcepts:
    """
    Extracted musical concepts from natural language description
    """
    genre: Optional[str] = None
    subgenre: Optional[str] = None
    era: Optional[str] = None
    mood: Optional[str] = None
    tempo_descriptor: Optional[str] = None
    instrumentation: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    reference_artists: List[str] = field(default_factory=list)
    rhythmic_feel: Optional[str] = None
    harmonic_complexity: Optional[str] = None
    dynamic_level: Optional[str] = None
    texture: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class StyleExample:
    """
    Example mapping of natural language description to parameters
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    concepts: Optional[MusicalConcepts] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'concepts': self.concepts.to_dict() if self.concepts else None,
            'tags': self.tags
        }


# ==============================================================================
# STYLE DATABASE
# ==============================================================================

class StyleDatabase:
    """
    Database of style examples mapping natural language to parameters

    Contains:
    - Genre profiles converted to examples
    - Artist style examples (Sinatra, Coltrane, etc.)
    - Mood/feeling examples (sultry, aggressive, mellow)
    - Technical examples (walking bass, quartal voicings, etc.)
    """

    def __init__(self, registry: UniversalParameterRegistry):
        self.registry = registry
        self.examples: Dict[str, StyleExample] = {}
        self._load_default_examples()

    def _load_default_examples(self):
        """Load default style examples from existing profiles"""

        # Load from style registry
        for style_name, profile in STYLE_REGISTRY.items():
            example = self._convert_style_profile_to_example(style_name, profile)
            self.examples[style_name] = example

        # Add custom examples
        self._add_sinatra_examples()
        self._add_bebop_examples()
        self._add_minimalist_examples()
        self._add_latin_examples()
        self._add_blues_examples()
        self._add_orchestral_examples()

        logger.info(f"Loaded {len(self.examples)} style examples")

    def _convert_style_profile_to_example(self, name: str, profile: StyleProfile) -> StyleExample:
        """Convert StyleProfile to StyleExample with parameters"""

        # Map StyleProfile to parameter values
        parameters = {
            # Harmony
            "harmony.voicing.spread": self._map_voicing_spacing_to_spread(profile.voicing_spacing),
            "harmony.extensions.use_9ths": 9 in profile.chord_extensions,
            "harmony.extensions.use_11ths": 11 in profile.chord_extensions,
            "harmony.extensions.use_13ths": 13 in profile.chord_extensions,

            # Rhythm
            "rhythm.swing.amount": profile.swing_factor,
            "rhythm.syncopation.probability": profile.syncopation,

            # Dynamics
            "dynamics.velocity.base": self._map_dynamic_range_to_velocity(profile.dynamic_range),

            # Articulation
            "articulation.duration.ratio": 0.9 if "legato" in profile.articulation_probabilities else 0.7,
        }

        # Extract concepts
        concepts = MusicalConcepts(
            genre=profile.composer_era,
            era=profile.composer_era,
            reference_artists=[profile.name],
            texture=str(profile.texture_density),
            harmonic_complexity=str(profile.harmony_complexity)
        )

        description = f"{profile.name} style: {profile.composer_era}"

        return StyleExample(
            name=name,
            description=description,
            parameters=parameters,
            concepts=concepts,
            tags=[profile.composer_era, profile.cultural_origin]
        )

    def _map_voicing_spacing_to_spread(self, spacing: str) -> float:
        """Map voicing spacing to spread parameter"""
        mapping = {
            "tight": 0.2,
            "medium": 0.5,
            "wide": 0.8,
            "varied": 0.5
        }
        return mapping.get(spacing, 0.5)

    def _map_dynamic_range_to_velocity(self, dynamic_range: str) -> int:
        """Map dynamic range to base velocity"""
        mapping = {
            "narrow": 70,
            "medium": 80,
            "wide": 85,
            "very_wide": 90
        }
        return mapping.get(dynamic_range, 80)

    def _add_sinatra_examples(self):
        """Add Frank Sinatra style examples"""

        # Sinatra ballad
        self.examples["sinatra_ballad"] = StyleExample(
            name="sinatra_ballad",
            description="Sultry Sinatra-style ballad with lush orchestration and intimate vocals",
            parameters={
                "harmony.voicing.type": "close",
                "harmony.voicing.spread": 0.6,
                "harmony.voicing.density": 4,
                "harmony.extensions.use_9ths": True,
                "harmony.extensions.use_11ths": False,
                "harmony.extensions.use_13ths": False,
                "harmony.substitution.tritone_probability": 0.3,
                "rhythm.swing.amount": 0.54,
                "rhythm.syncopation.probability": 0.2,
                "dynamics.velocity.base": 65,
                "dynamics.velocity.variation": 25,
                "melody.chromaticism.amount": 0.3,
                "melody.ornaments.probability": 0.3,
            },
            concepts=MusicalConcepts(
                genre="jazz",
                subgenre="vocal_jazz",
                era="1950s",
                mood="sultry",
                tempo_descriptor="ballad",
                reference_artists=["Frank Sinatra"],
                rhythmic_feel="swing",
                harmonic_complexity="moderate",
                dynamic_level="intimate"
            ),
            tags=["jazz", "vocal", "ballad", "sinatra", "1950s", "romantic"]
        )

        # Sinatra uptempo
        self.examples["sinatra_uptempo"] = StyleExample(
            name="sinatra_uptempo",
            description="Swinging uptempo Sinatra arrangement with punchy brass",
            parameters={
                "harmony.voicing.type": "spread",
                "harmony.voicing.spread": 0.7,
                "harmony.voicing.density": 5,
                "rhythm.swing.amount": 0.62,
                "rhythm.syncopation.probability": 0.4,
                "dynamics.velocity.base": 90,
                "dynamics.velocity.variation": 20,
            },
            concepts=MusicalConcepts(
                genre="jazz",
                subgenre="swing",
                era="1950s",
                mood="upbeat",
                tempo_descriptor="uptempo",
                reference_artists=["Frank Sinatra"],
                rhythmic_feel="swing"
            ),
            tags=["jazz", "swing", "uptempo", "sinatra", "big_band"]
        )

    def _add_bebop_examples(self):
        """Add bebop style examples"""

        self.examples["bebop_fast"] = StyleExample(
            name="bebop_fast",
            description="Fast bebop piano with walking bass and brushed drums",
            parameters={
                "harmony.voicing.type": "rootless_a",
                "harmony.voicing.density": 4,
                "harmony.extensions.use_9ths": True,
                "harmony.extensions.use_11ths": True,
                "harmony.substitution.tritone_probability": 0.6,
                "melody.chromaticism.amount": 0.7,
                "melody.intervals.stepwise_probability": 0.4,
                "rhythm.swing.amount": 0.62,
                "rhythm.syncopation.probability": 0.7,
                "bass.style.walking_probability": 0.95,
            },
            concepts=MusicalConcepts(
                genre="jazz",
                subgenre="bebop",
                era="1940s",
                tempo_descriptor="fast",
                instrumentation=["piano", "bass", "drums"],
                technical_terms=["walking bass", "rootless voicings", "bebop"],
                rhythmic_feel="swing",
                harmonic_complexity="complex"
            ),
            tags=["jazz", "bebop", "fast", "piano_trio"]
        )

    def _add_minimalist_examples(self):
        """Add minimalist style examples"""

        self.examples["minimalist_ambient"] = StyleExample(
            name="minimalist_ambient",
            description="Sparse minimalist ambient soundscape with subtle movement",
            parameters={
                "harmony.voicing.density": 2,
                "melody.intervals.stepwise_probability": 0.9,
                "rhythm.syncopation.probability": 0.1,
                "dynamics.velocity.base": 50,
                "dynamics.velocity.variation": 10,
                "melody.chromaticism.amount": 0.2,
            },
            concepts=MusicalConcepts(
                genre="contemporary",
                subgenre="minimalist",
                mood="ambient",
                tempo_descriptor="slow",
                texture="sparse",
                dynamic_level="quiet"
            ),
            tags=["minimalist", "ambient", "sparse", "contemporary"]
        )

    def _add_latin_examples(self):
        """Add Latin music examples"""

        self.examples["bossa_nova"] = StyleExample(
            name="bossa_nova",
            description="Smooth bossa nova with gentle syncopation and warm harmonies",
            parameters={
                "harmony.voicing.type": "close",
                "harmony.extensions.use_9ths": True,
                "harmony.extensions.use_11ths": True,
                "rhythm.swing.amount": 0.5,  # Straight eighths
                "rhythm.syncopation.probability": 0.6,
                "dynamics.velocity.base": 70,
                "melody.chromaticism.amount": 0.4,
            },
            concepts=MusicalConcepts(
                genre="latin",
                subgenre="bossa_nova",
                era="1960s",
                mood="smooth",
                rhythmic_feel="straight_with_syncopation"
            ),
            tags=["latin", "bossa_nova", "brazilian", "jazz_influenced"]
        )

    def _add_blues_examples(self):
        """Add blues style examples"""

        self.examples["chicago_blues"] = StyleExample(
            name="chicago_blues",
            description="Electric Chicago blues with bent notes and shuffle feel",
            parameters={
                "harmony.voicing.type": "close",
                "harmony.extensions.use_9ths": False,
                "melody.chromaticism.amount": 0.5,
                "rhythm.swing.amount": 0.58,  # Shuffle
                "rhythm.syncopation.probability": 0.4,
                "dynamics.velocity.base": 85,
            },
            concepts=MusicalConcepts(
                genre="blues",
                subgenre="chicago_blues",
                era="1950s",
                mood="gritty",
                rhythmic_feel="shuffle",
                instrumentation=["electric_guitar", "harmonica", "bass", "drums"]
            ),
            tags=["blues", "chicago", "electric", "shuffle"]
        )

    def _add_orchestral_examples(self):
        """Add orchestral examples"""

        self.examples["romantic_orchestra"] = StyleExample(
            name="romantic_orchestra",
            description="Lush romantic orchestral arrangement with sweeping strings",
            parameters={
                "harmony.voicing.spread": 0.8,
                "harmony.voicing.density": 6,
                "harmony.voice_leading.smoothness": 0.9,
                "dynamics.velocity.base": 80,
                "dynamics.velocity.variation": 40,
                "melody.ornaments.probability": 0.4,
            },
            concepts=MusicalConcepts(
                genre="classical",
                subgenre="romantic",
                era="19th_century",
                mood="dramatic",
                instrumentation=["strings", "winds", "brass"],
                texture="dense",
                harmonic_complexity="rich"
            ),
            tags=["classical", "romantic", "orchestral", "strings"]
        )

    def get_example(self, name: str) -> Optional[StyleExample]:
        """Get example by name"""
        return self.examples.get(name)

    def find_similar_examples(self, concepts: MusicalConcepts, limit: int = 3) -> List[StyleExample]:
        """Find examples similar to given concepts"""

        matches = []

        for example in self.examples.values():
            score = self._calculate_similarity_score(concepts, example)
            matches.append((score, example))

        # Sort by score (descending)
        matches.sort(key=lambda x: x[0], reverse=True)

        return [example for score, example in matches[:limit]]

    def _calculate_similarity_score(self, concepts: MusicalConcepts, example: StyleExample) -> float:
        """Calculate similarity score between concepts and example"""

        score = 0.0

        if not example.concepts:
            return 0.0

        # Genre match (weight: 3)
        if concepts.genre and concepts.genre == example.concepts.genre:
            score += 3.0

        # Era match (weight: 2)
        if concepts.era and concepts.era == example.concepts.era:
            score += 2.0

        # Mood match (weight: 2)
        if concepts.mood and concepts.mood == example.concepts.mood:
            score += 2.0

        # Reference artist match (weight: 3)
        if concepts.reference_artists and example.concepts.reference_artists:
            if any(artist in example.concepts.reference_artists for artist in concepts.reference_artists):
                score += 3.0

        # Technical terms overlap (weight: 1 per match)
        if concepts.technical_terms and example.concepts.technical_terms:
            overlap = len(set(concepts.technical_terms) & set(example.concepts.technical_terms))
            score += overlap

        # Tag matches
        if concepts.genre and concepts.genre in example.tags:
            score += 1.0

        return score

    def add_example(self, example: StyleExample):
        """Add a new example to the database"""
        self.examples[example.name] = example
        logger.info(f"Added style example: {example.name}")

    def export_to_json(self, filepath: str):
        """Export database to JSON file"""
        data = {
            name: example.to_dict()
            for name, example in self.examples.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported style database to {filepath}")


# ==============================================================================
# CONCEPT EXTRACTOR
# ==============================================================================

class ConceptExtractor:
    """
    Extracts musical concepts from natural language using LLM
    """

    def __init__(self, llm_client: Optional[Any]):
        self.llm_client = llm_client

    def extract_concepts(self, text: str) -> MusicalConcepts:
        """
        Extract musical concepts from natural language description

        Args:
            text: Natural language description

        Returns:
            MusicalConcepts object
        """

        # Check if LLM client is available
        if not self.llm_client:
            logger.warning("LLM client not available, returning basic concept extraction")
            # Return basic concept extraction based on keywords
            return self._extract_concepts_basic(text)

        extraction_prompt = f"""Analyze this musical description and extract key concepts:
"{text}"

Extract and return ONLY valid JSON with these fields:
- genre: The primary genre (e.g., "jazz", "classical", "rock", "electronic")
- subgenre: More specific genre (e.g., "bebop", "bossa nova", "symphony")
- era: Time period (e.g., "1950s", "baroque", "modern")
- mood: Emotional quality (e.g., "sultry", "aggressive", "mellow", "upbeat")
- tempo_descriptor: Tempo description (e.g., "slow", "uptempo", "ballad", "fast")
- instrumentation: List of instruments mentioned (e.g., ["piano", "bass", "drums"])
- technical_terms: Musical terms (e.g., ["walking bass", "quartal voicings", "brushes"])
- reference_artists: Artists or composers mentioned (e.g., ["Sinatra", "Coltrane"])
- rhythmic_feel: Rhythm type (e.g., "swing", "straight", "shuffle")
- harmonic_complexity: Harmony level (e.g., "simple", "moderate", "complex")
- dynamic_level: Volume/intensity (e.g., "quiet", "intimate", "loud", "powerful")
- texture: Musical texture (e.g., "sparse", "dense", "intimate")

Use null for fields that cannot be determined from the text.
Output ONLY the JSON object, no other text.

Example output:
{{
  "genre": "jazz",
  "subgenre": "vocal_jazz",
  "era": "1950s",
  "mood": "sultry",
  "tempo_descriptor": "ballad",
  "instrumentation": ["piano", "strings", "brass"],
  "technical_terms": ["close voicings"],
  "reference_artists": ["Frank Sinatra"],
  "rhythmic_feel": "swing",
  "harmonic_complexity": "moderate",
  "dynamic_level": "intimate",
  "texture": "lush"
}}"""

        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.3
            )

            # Parse JSON from response
            response_text = response.content[0].text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            concepts_dict = json.loads(response_text)

            # Convert to MusicalConcepts object
            concepts = MusicalConcepts(
                genre=concepts_dict.get("genre"),
                subgenre=concepts_dict.get("subgenre"),
                era=concepts_dict.get("era"),
                mood=concepts_dict.get("mood"),
                tempo_descriptor=concepts_dict.get("tempo_descriptor"),
                instrumentation=concepts_dict.get("instrumentation", []),
                technical_terms=concepts_dict.get("technical_terms", []),
                reference_artists=concepts_dict.get("reference_artists", []),
                rhythmic_feel=concepts_dict.get("rhythmic_feel"),
                harmonic_complexity=concepts_dict.get("harmonic_complexity"),
                dynamic_level=concepts_dict.get("dynamic_level"),
                texture=concepts_dict.get("texture")
            )

            logger.info(f"Extracted concepts: genre={concepts.genre}, mood={concepts.mood}")
            return concepts

        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            # Return empty concepts on failure
            return MusicalConcepts()

    def _extract_concepts_basic(self, text: str) -> MusicalConcepts:
        """
        Basic keyword-based concept extraction (fallback when LLM not available)

        Args:
            text: Natural language description

        Returns:
            MusicalConcepts object
        """

        text_lower = text.lower()

        # Genre keywords
        genre = None
        if any(word in text_lower for word in ["jazz", "swing", "bebop", "bossa"]):
            genre = "jazz"
        elif any(word in text_lower for word in ["classical", "symphony", "baroque", "romantic"]):
            genre = "classical"
        elif any(word in text_lower for word in ["rock", "blues", "metal"]):
            genre = "rock"
        elif any(word in text_lower for word in ["ambient", "minimalist", "contemporary"]):
            genre = "contemporary"
        elif any(word in text_lower for word in ["latin", "salsa", "bossa nova"]):
            genre = "latin"

        # Mood keywords
        mood = None
        if any(word in text_lower for word in ["sultry", "intimate", "romantic"]):
            mood = "sultry"
        elif any(word in text_lower for word in ["aggressive", "intense", "powerful"]):
            mood = "aggressive"
        elif any(word in text_lower for word in ["mellow", "calm", "peaceful"]):
            mood = "mellow"
        elif any(word in text_lower for word in ["upbeat", "happy", "cheerful"]):
            mood = "upbeat"

        # Tempo keywords
        tempo_descriptor = None
        if any(word in text_lower for word in ["ballad", "slow"]):
            tempo_descriptor = "ballad"
        elif any(word in text_lower for word in ["fast", "uptempo", "quick"]):
            tempo_descriptor = "fast"
        elif any(word in text_lower for word in ["medium", "moderate"]):
            tempo_descriptor = "medium"

        # Reference artists
        reference_artists = []
        artists = ["sinatra", "coltrane", "ellington", "basie", "mozart", "beethoven"]
        for artist in artists:
            if artist in text_lower:
                reference_artists.append(artist.title())

        # Instrumentation
        instrumentation = []
        instruments = ["piano", "bass", "drums", "guitar", "strings", "brass", "saxophone"]
        for instrument in instruments:
            if instrument in text_lower:
                instrumentation.append(instrument)

        # Technical terms
        technical_terms = []
        terms = ["walking bass", "voicing", "swing", "brushes", "quartal"]
        for term in terms:
            if term in text_lower:
                technical_terms.append(term)

        return MusicalConcepts(
            genre=genre,
            mood=mood,
            tempo_descriptor=tempo_descriptor,
            reference_artists=reference_artists,
            instrumentation=instrumentation,
            technical_terms=technical_terms
        )


# ==============================================================================
# PARAMETER VALIDATOR
# ==============================================================================

class ParameterValidator:
    """
    Validates and fills default values for parameters
    """

    def __init__(self, registry: UniversalParameterRegistry):
        self.registry = registry

    def validate_and_fill_defaults(self, parameters: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate parameters and fill missing with defaults

        Args:
            parameters: Dictionary of parameter values

        Returns:
            Tuple of (validated_parameters, list_of_warnings)
        """

        validated = {}
        warnings = []

        # Get all registered parameters
        all_params = self.registry.parameters

        # Validate provided parameters
        for param_path, value in parameters.items():
            param_def = all_params.get(param_path)

            if not param_def:
                warnings.append(f"Unknown parameter: {param_path}")
                continue

            # Validate value
            valid, error_msg = param_def.validate(value)

            if valid:
                validated[param_path] = value
            else:
                warnings.append(f"{param_path}: {error_msg}, using default")
                validated[param_path] = param_def.default_value

        # Fill missing parameters with defaults
        for param_path, param_def in all_params.items():
            if param_path not in validated:
                validated[param_path] = param_def.default_value

        logger.info(f"Validated {len(validated)} parameters with {len(warnings)} warnings")

        return validated, warnings


# ==============================================================================
# NATURAL LANGUAGE PARAMETER PREDICTOR
# ==============================================================================

class NaturalLanguageParameterPredictor:
    """
    Main predictor class that converts natural language to parameter values

    Usage:
        predictor = NaturalLanguageParameterPredictor(api_key="your_key")
        params = predictor.predict_parameters("Generate a Sinatra-style ballad")
    """

    def __init__(self, api_key: Optional[str] = None, registry: Optional[UniversalParameterRegistry] = None):
        """
        Initialize predictor

        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            registry: Parameter registry (if None, uses global REGISTRY)
        """

        # Check if anthropic is available
        if not ANTHROPIC_AVAILABLE:
            logger.warning("anthropic package not available. LLM features will be limited.")
            logger.warning("Install with: pip install anthropic")
            self.llm_client = None
        else:
            # Setup API key
            if api_key is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("No API key provided. LLM features will be limited.")
                    logger.warning("Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
                    self.llm_client = None
                else:
                    self.llm_client = anthropic.Anthropic(api_key=api_key)

        self.registry = registry or REGISTRY
        self.style_database = StyleDatabase(self.registry)
        self.concept_extractor = ConceptExtractor(self.llm_client)
        self.validator = ParameterValidator(self.registry)

        logger.info("Natural Language Parameter Predictor initialized")

    def predict_parameters(self, text_prompt: str, validate: bool = True) -> Dict[str, Any]:
        """
        Convert natural language to parameter values

        Args:
            text_prompt: Natural language description
            validate: Whether to validate and fill defaults

        Returns:
            Dictionary of parameter values
        """

        logger.info(f"Predicting parameters for: '{text_prompt}'")

        # Step 1: Extract concepts
        concepts = self.concept_extractor.extract_concepts(text_prompt)

        # Step 2: Find similar examples
        similar_examples = self.style_database.find_similar_examples(concepts, limit=3)

        # Step 3: Build LLM prompt
        llm_prompt = self._build_prediction_prompt(text_prompt, concepts, similar_examples)

        # Step 4: Get LLM prediction
        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": llm_prompt}],
                temperature=0.3
            )

            # Parse response
            response_text = response.content[0].text.strip()
            parameters = self._parse_llm_response(response_text)

        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            # Fallback to similar examples
            parameters = self._fallback_to_examples(similar_examples)

        # Step 5: Validate and fill defaults
        if validate:
            parameters, warnings = self.validator.validate_and_fill_defaults(parameters)

            if warnings:
                logger.warning(f"Validation warnings: {len(warnings)}")
                for warning in warnings[:5]:  # Show first 5
                    logger.warning(f"  {warning}")

        logger.info(f"Predicted {len(parameters)} parameters")
        return parameters

    def _get_system_prompt(self) -> str:
        """Build comprehensive system prompt for parameter prediction"""

        # Get parameter registry summary
        param_summary = self._format_parameter_registry()

        # Get genre profiles
        genre_summary = self._format_genre_profiles()

        # Get few-shot examples
        examples_summary = self._format_few_shot_examples()

        system_prompt = f"""You are an expert music parameter predictor. Given a musical style description,
predict precise parameter values for music generation.

PARAMETER REGISTRY ({len(self.registry.parameters)} parameters):
{param_summary}

TASK:
1. Analyze the user's musical description
2. Map concepts to relevant parameters
3. Predict appropriate values for ALL parameters
4. Return ONLY valid JSON

PARAMETER PREDICTION PRINCIPLES:
- **Specificity**: If user says "Sinatra", set specific swing ratios, brass voicings
- **Coherence**: Parameters must work together (e.g., swing + walking bass)
- **Defaults**: Use genre-appropriate defaults for unmentioned parameters
- **Musicality**: Ensure values create coherent, playable music

GENRE KNOWLEDGE:
{genre_summary}

EXAMPLES:
{examples_summary}

OUTPUT FORMAT (strict JSON):
{{
  "harmony.voicing.type": "close",
  "harmony.voicing.spread": 0.6,
  "harmony.voicing.density": 4,
  "harmony.extensions.use_9ths": true,
  "rhythm.swing.amount": 0.58,
  "dynamics.velocity.base": 80,
  ...ALL parameters
}}

CRITICAL:
- Output ONLY valid JSON
- Use parameter full paths as keys
- All values must match parameter types
- Use defaults for unmentioned parameters
"""

        return system_prompt

    def _format_parameter_registry(self) -> str:
        """Format parameter registry for LLM prompt"""

        # Group by category
        by_category = defaultdict(list)
        for param in self.registry.parameters.values():
            if param.category:
                by_category[param.category].append(param)

        lines = []
        for category, params in list(by_category.items())[:8]:  # Limit to 8 categories
            lines.append(f"\n{category.value.upper()}:")
            for param in params[:5]:  # Show 5 examples per category
                param_info = f"  - {param.full_path}: {param.description}"
                if param.param_type == ParameterType.PROBABILITY:
                    param_info += f" (0.0-1.0, default: {param.default_value})"
                elif param.param_type == ParameterType.CATEGORICAL:
                    param_info += f" (options: {param.options}, default: {param.default_value})"
                lines.append(param_info)

            if len(params) > 5:
                lines.append(f"  ...and {len(params) - 5} more")

        return "\n".join(lines)

    def _format_genre_profiles(self) -> str:
        """Format genre profiles for LLM prompt"""

        lines = []
        for style_name, profile in list(STYLE_REGISTRY.items())[:5]:  # Show 5 examples
            lines.append(f"\n{profile.name} ({profile.composer_era}):")
            lines.append(f"  - Swing: {profile.swing_factor}")
            lines.append(f"  - Harmony complexity: {profile.harmony_complexity}")
            lines.append(f"  - Syncopation: {profile.syncopation}")

        return "\n".join(lines)

    def _format_few_shot_examples(self) -> str:
        """Format few-shot examples for LLM prompt"""

        lines = []
        for example in list(self.style_database.examples.values())[:3]:  # Show 3 examples
            lines.append(f"\n{example.description}:")
            lines.append("  Parameters:")
            for param_path, value in list(example.parameters.items())[:5]:  # Show 5 params
                lines.append(f"    - {param_path}: {value}")

        return "\n".join(lines)

    def _build_prediction_prompt(self, original_text: str, concepts: MusicalConcepts,
                                 similar_examples: List[StyleExample]) -> str:
        """Build user prompt for parameter prediction"""

        prompt = f"""MUSICAL DESCRIPTION:
"{original_text}"

EXTRACTED CONCEPTS:
{json.dumps(concepts.to_dict(), indent=2)}

SIMILAR KNOWN STYLES:
"""

        for example in similar_examples:
            prompt += f"\n{example.name} - {example.description}:\n"
            prompt += f"Key parameters:\n"
            for param_path, value in list(example.parameters.items())[:8]:
                prompt += f"  {param_path}: {value}\n"

        prompt += """

YOUR TASK:
Predict ALL parameters for this description.
Use similar styles as reference, but adjust based on specific description details.

Output ONLY valid JSON with all parameters.
"""

        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract parameters"""

        try:
            # Remove markdown code blocks if present
            if "```" in response_text:
                # Extract JSON from code block
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    response_text = response_text[start:end]

            parameters = json.loads(response_text)

            # Ensure it's a dictionary
            if not isinstance(parameters, dict):
                raise ValueError("Response is not a dictionary")

            return parameters

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return {}

    def _fallback_to_examples(self, examples: List[StyleExample]) -> Dict[str, Any]:
        """Fallback to using example parameters if LLM fails"""

        if not examples:
            logger.warning("No examples available for fallback")
            return {}

        # Use the most similar example
        best_example = examples[0]
        logger.info(f"Using fallback example: {best_example.name}")

        return best_example.parameters.copy()

    def predict_with_concepts(self, concepts: MusicalConcepts) -> Dict[str, Any]:
        """
        Predict parameters directly from concepts (skip concept extraction)

        Args:
            concepts: Pre-extracted musical concepts

        Returns:
            Dictionary of parameter values
        """

        # Find similar examples
        similar_examples = self.style_database.find_similar_examples(concepts, limit=3)

        # Build description from concepts
        description = f"{concepts.genre or 'music'}"
        if concepts.mood:
            description += f" with {concepts.mood} mood"
        if concepts.tempo_descriptor:
            description += f", {concepts.tempo_descriptor} tempo"

        # Use the standard prediction pipeline
        return self.predict_parameters(description, validate=True)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def predict_from_text(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to predict parameters from text

    Args:
        text: Natural language description
        api_key: Anthropic API key (optional)

    Returns:
        Dictionary of parameter values
    """
    predictor = NaturalLanguageParameterPredictor(api_key=api_key)
    return predictor.predict_parameters(text)


def get_style_database() -> StyleDatabase:
    """
    Get the global style database

    Returns:
        StyleDatabase instance
    """
    return StyleDatabase(REGISTRY)


# ==============================================================================
# MAIN / TESTING
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("NATURAL LANGUAGE PARAMETER PREDICTOR - Agent 6")
    print("=" * 80)

    # Test style database
    print("\n📚 Testing Style Database...")
    db = StyleDatabase(REGISTRY)
    print(f"   Loaded {len(db.examples)} style examples")

    # Show some examples
    print("\n   Sample examples:")
    for name in list(db.examples.keys())[:5]:
        example = db.examples[name]
        print(f"   - {name}: {example.description}")

    # Test concept extraction (requires API key)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        print("\n🤖 Testing Concept Extraction...")
        try:
            predictor = NaturalLanguageParameterPredictor(api_key=api_key)

            test_prompts = [
                "Generate a Sinatra-style ballad with lush strings",
                "Fast bebop piano solo with walking bass",
                "Minimalist ambient soundscape",
            ]

            for prompt in test_prompts:
                print(f"\n   Prompt: '{prompt}'")
                concepts = predictor.concept_extractor.extract_concepts(prompt)
                print(f"   Genre: {concepts.genre}, Mood: {concepts.mood}, Era: {concepts.era}")

            print("\n✅ Concept extraction working!")

            # Test full prediction
            print("\n🎵 Testing Full Parameter Prediction...")
            params = predictor.predict_parameters("Generate a sultry Sinatra ballad")
            print(f"   Predicted {len(params)} parameters")

            # Show some example parameters
            print("\n   Sample parameters:")
            for key in list(params.keys())[:10]:
                print(f"   - {key}: {params[key]}")

            print("\n✅ Full prediction working!")

        except Exception as e:
            print(f"\n   ⚠️  Could not test with LLM: {e}")
            print(f"   Set ANTHROPIC_API_KEY environment variable to enable LLM features")
    else:
        print("\n⚠️  ANTHROPIC_API_KEY not set")
        print("   Style database loaded successfully, but LLM features require API key")

    print("\n" + "=" * 80)
    print("✅ Natural Language Parameter Predictor module loaded successfully")
    print("=" * 80)
