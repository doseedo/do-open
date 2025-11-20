#!/usr/bin/env python3
"""
Unified API for HarmonyModule Modular Fusion System

This module provides a high-level, user-friendly API for all modular genre fusion
capabilities implemented across Agents 1-9:

- Agent 1: Genre Detection & Feature Extraction
- Agent 2: Component Abstraction Layer
- Agent 3: Context-Aware Generation
- Agent 4: Inpainting Engine
- Agent 5: Full Modular Fusion (N-Way Component Mixing)
- Agent 6: Tempo Conversion (Style-Appropriate)
- Agent 7: Meter Conversion
- Agent 8: Granular Control System
- Agent 9: Track-Level Genre Control

The API is designed to be:
- Intuitive: Simple methods for common tasks
- Powerful: Access to all advanced features
- Flexible: Multiple ways to achieve goals
- Production-ready: Error handling, validation, logging

Examples:
    # Quick genre fusion
    >>> api = HarmonyModuleAPI()
    >>> composition = api.quick_fusion(
    ...     harmony="jazz", rhythm="funk", instrumentation="edm",
    ...     tempo=120, key="Dm", measures=16
    ... )
    >>> composition.export("output.mid")

    # Detect genre and regenerate section
    >>> api.load_midi("input.mid")
    >>> detected = api.detect_genre()
    >>> api.inpaint_section(
    ...     tracks=[1, 2],
    ...     measures=(5, 8),
    ...     new_chords=["Dm7", "G7", "Cmaj7", "A7"]
    ... )
    >>> api.export("regenerated.mid")

    # Add bass to existing arrangement
    >>> api.load_midi("piano_drums.mid")
    >>> api.add_track(instrument=33, track_type="bass", genre="funk")
    >>> api.export("with_bass.mid")

Author: Agent 10 - Unified API & Integration
Date: 2025
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components (with graceful fallbacks for development)
try:
    from generators.style_fusion import GenreFeatures, GENRE_PROFILES
except ImportError:
    logger.warning("style_fusion not available, using placeholder")
    GenreFeatures = None
    GENRE_PROFILES = {}

try:
    from analysis.midi_analyzer import MIDIAnalyzer
except ImportError:
    logger.warning("midi_analyzer not available")
    MIDIAnalyzer = None

# Try to import Agent modules (with graceful degradation)
try:
    from analysis.genre_detector import GenreDetector
except ImportError:
    GenreDetector = None
    logger.debug("GenreDetector (Agent 1) not yet implemented")

try:
    from core.component_system import (
        CompositionBuilder, ComponentType, ComponentFactory,
        global_factory, GenerationContext
    )
except ImportError:
    CompositionBuilder = None
    ComponentType = None
    ComponentFactory = None
    global_factory = None
    GenerationContext = None
    logger.debug("Component System (Agent 2) not yet implemented")

try:
    from generators.context_aware_generator import (
        ContextAwareGenerator, TrackInpainter, SmartOrchestrator
    )
except ImportError:
    ContextAwareGenerator = None
    TrackInpainter = None
    SmartOrchestrator = None
    logger.debug("Context-Aware Generator (Agent 3) not yet implemented")

try:
    from transformation.inpainting_engine import (
        InpaintingEngine, ChordSubstitutionEngine, StyleTransitionBlender
    )
except ImportError:
    InpaintingEngine = None
    ChordSubstitutionEngine = None
    StyleTransitionBlender = None
    logger.debug("Inpainting Engine (Agent 4) not yet implemented")

try:
    from generators.style_fusion import (
        ModularFusion, ComponentReplacer, GenreCompatibilityAnalyzer,
        TrackLevelFusion, ProgressiveFusion
    )
except ImportError:
    ModularFusion = None
    ComponentReplacer = None
    GenreCompatibilityAnalyzer = None
    TrackLevelFusion = None
    ProgressiveFusion = None
    logger.debug("Modular Fusion extensions (Agent 5) not fully implemented")

try:
    from transformation.tempo_converter import TempoConverter, StyleTempoAdjuster
except ImportError:
    TempoConverter = None
    StyleTempoAdjuster = None
    logger.debug("Tempo Converter (Agent 6) not yet implemented")

try:
    from transformation.meter_converter import MeterConverter, MetricModulator
except ImportError:
    MeterConverter = None
    MetricModulator = None
    logger.debug("Meter Converter (Agent 7) not yet implemented")

try:
    from generators.granular_control import (
        GranularController, IdiomaticWriter, PatternApplicator
    )
except ImportError:
    GranularController = None
    IdiomaticWriter = None
    PatternApplicator = None
    logger.debug("Granular Control (Agent 8) not yet implemented")

try:
    from core.multi_genre_arranger import (
        MultiGenreArranger, TrackGenreManager
    )
except ImportError:
    MultiGenreArranger = None
    TrackGenreManager = None
    logger.debug("Multi-Genre Arranger (Agent 9) not yet implemented")


# ==============================================================================
# MAIN API CLASS
# ==============================================================================

class HarmonyModuleAPI:
    """
    Main unified API for all modular fusion capabilities

    This class provides a single entry point for all features from Agents 1-9.
    It manages state, handles file I/O, and provides both simple and advanced
    interfaces for music generation.

    Attributes:
        current_midi: Currently loaded MIDI file path
        composition: Current composition state
        detected_features: Detected genre features from analysis
        history: Operation history for undo capability
    """

    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the HarmonyModule API

        Args:
            output_dir: Directory for output files (created if doesn't exist)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_midi: Optional[str] = None
        self.composition: Optional[Any] = None
        self.detected_features: Optional[Dict] = None
        self.history: List[Dict] = []

        logger.info(f"HarmonyModule API initialized. Output: {self.output_dir}")

    # ==========================================================================
    # FILE OPERATIONS
    # ==========================================================================

    def load_midi(self, filepath: str) -> Dict[str, Any]:
        """
        Load MIDI file for analysis or transformation

        Args:
            filepath: Path to MIDI file

        Returns:
            Basic analysis of the loaded file

        Example:
            >>> api = HarmonyModuleAPI()
            >>> info = api.load_midi("song.mid")
            >>> print(f"Loaded {info['num_tracks']} tracks, {info['tempo']} BPM")
        """
        filepath = str(Path(filepath).resolve())
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MIDI file not found: {filepath}")

        self.current_midi = filepath
        logger.info(f"Loaded MIDI file: {filepath}")

        # Perform basic analysis
        if MIDIAnalyzer:
            analyzer = MIDIAnalyzer(filepath)
            info = {
                'filepath': filepath,
                'num_tracks': analyzer.num_tracks,
                'tempo': analyzer.tempo,
                'time_signature': analyzer.time_signature,
                'key': analyzer.key if hasattr(analyzer, 'key') else 'Unknown',
                'duration_seconds': analyzer.duration if hasattr(analyzer, 'duration') else None,
            }
        else:
            # Fallback basic info
            info = {'filepath': filepath, 'status': 'loaded (limited analysis)'}

        return info

    def export(self, filename: str, overwrite: bool = False) -> str:
        """
        Export current composition to MIDI file

        Args:
            filename: Output filename
            overwrite: Allow overwriting existing files

        Returns:
            Full path to exported file

        Example:
            >>> api.quick_fusion(harmony="jazz", rhythm="funk")
            >>> path = api.export("jazz_funk.mid")
        """
        output_path = self.output_dir / filename

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File exists: {output_path}. Use overwrite=True to replace."
            )

        if self.composition is None:
            raise ValueError("No composition to export. Generate or load first.")

        # Export composition
        if hasattr(self.composition, 'to_midi'):
            self.composition.to_midi(str(output_path))
        elif hasattr(self.composition, 'export'):
            self.composition.export(str(output_path))
        else:
            raise NotImplementedError("Composition export not available")

        logger.info(f"Exported to: {output_path}")
        return str(output_path)

    # ==========================================================================
    # GENRE DETECTION (Agent 1)
    # ==========================================================================

    def detect_genre(self, midi_file: Optional[str] = None,
                    top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Detect genre(s) from MIDI file

        Args:
            midi_file: Path to MIDI (None = use loaded file)
            top_n: Number of top genres to return

        Returns:
            List of (genre_name, confidence_score) tuples

        Example:
            >>> api.load_midi("mystery_song.mid")
            >>> genres = api.detect_genre()
            >>> print(genres)  # [('jazz', 0.85), ('blues', 0.62), ('funk', 0.58)]
        """
        midi_file = midi_file or self.current_midi
        if not midi_file:
            raise ValueError("No MIDI file loaded. Use load_midi() or provide path.")

        if not GenreDetector:
            raise NotImplementedError("GenreDetector (Agent 1) not yet implemented")

        detector = GenreDetector(midi_file)
        genres = detector.classify_genre(top_n=top_n)

        self.detected_features = {
            'genres': genres,
            'rhythmic': detector.extract_rhythmic_features(),
            'harmonic': detector.extract_harmonic_features(),
            'melodic': detector.extract_melodic_features(),
            'instrumentation': detector.extract_instrumentation_features(),
        }

        logger.info(f"Detected genres: {genres}")
        return genres

    def extract_features(self, midi_file: Optional[str] = None) -> GenreFeatures:
        """
        Extract full genre feature set from MIDI

        Args:
            midi_file: Path to MIDI (None = use loaded file)

        Returns:
            GenreFeatures object

        Example:
            >>> features = api.extract_features("song.mid")
            >>> print(f"Tempo: {features.tempo_range}")
            >>> print(f"Swing: {features.swing_factor}")
        """
        midi_file = midi_file or self.current_midi
        if not midi_file:
            raise ValueError("No MIDI file loaded")

        if not GenreDetector:
            raise NotImplementedError("GenreDetector not available")

        detector = GenreDetector(midi_file)
        features = detector.to_genre_features()

        return features

    # ==========================================================================
    # QUICK FUSION (Simple Interface)
    # ==========================================================================

    def quick_fusion(self,
                     harmony: str = "jazz",
                     rhythm: str = "funk",
                     melody: Optional[str] = None,
                     bass: Optional[str] = None,
                     drums: Optional[str] = None,
                     instrumentation: Optional[str] = None,
                     tempo: int = 120,
                     key: str = "C",
                     measures: int = 16,
                     time_signature: Tuple[int, int] = (4, 4)) -> Any:
        """
        Quick genre fusion - mix components from different genres

        This is the simplest way to create genre fusions. Specify which genre
        should provide each musical component.

        Args:
            harmony: Genre for chord progressions (jazz, blues, funk, etc.)
            rhythm: Genre for rhythmic feel
            melody: Genre for melodic style (None = use harmony genre)
            bass: Genre for bass patterns (None = use rhythm genre)
            drums: Genre for drum patterns (None = use rhythm genre)
            instrumentation: Genre for instrument selection (None = use harmony)
            tempo: Tempo in BPM
            key: Key signature
            measures: Length in measures
            time_signature: Time signature (numerator, denominator)

        Returns:
            Composition object

        Example:
            >>> # Jazz harmony + funk rhythm + EDM synths
            >>> comp = api.quick_fusion(
            ...     harmony="jazz",
            ...     rhythm="funk",
            ...     instrumentation="edm",
            ...     tempo=115,
            ...     key="Dm",
            ...     measures=32
            ... )
            >>> api.composition = comp
            >>> api.export("jazz_funk_edm.mid")
        """
        logger.info(f"Creating fusion: harmony={harmony}, rhythm={rhythm}")

        if not ModularFusion:
            raise NotImplementedError("ModularFusion (Agent 5) not yet implemented")

        fusion = ModularFusion()
        composition = fusion.fuse_components(
            rhythm_genre=rhythm,
            harmony_genre=harmony,
            melody_genre=melody,
            bass_genre=bass,
            drums_genre=drums,
            instrumentation_genre=instrumentation,
            tempo=tempo,
            key=key,
            length_measures=measures,
            numerator=time_signature[0],
            denominator=time_signature[1]
        )

        self.composition = composition
        self._add_to_history('quick_fusion', {
            'harmony': harmony, 'rhythm': rhythm, 'tempo': tempo, 'key': key
        })

        return composition

    # ==========================================================================
    # WEIGHTED BLENDING (Agent 5)
    # ==========================================================================

    def weighted_blend(self, blends: Dict[str, List[Tuple[str, float]]],
                      tempo: int = 120, key: str = "C",
                      measures: int = 16) -> Any:
        """
        Create weighted blends of multiple genres for each component

        Args:
            blends: Dict mapping component type to [(genre, weight), ...]
            tempo: Tempo in BPM
            key: Key signature
            measures: Length in measures

        Returns:
            Composition object

        Example:
            >>> # 60% jazz + 40% blues harmony, pure funk rhythm
            >>> comp = api.weighted_blend({
            ...     'harmony': [('jazz', 0.6), ('blues', 0.4)],
            ...     'rhythm': [('funk', 1.0)],
            ...     'bass': [('reggae', 0.7), ('funk', 0.3)]
            ... }, tempo=95, key="G")
        """
        if not ModularFusion:
            raise NotImplementedError("ModularFusion not available")

        if not ComponentType:
            raise NotImplementedError("ComponentType not available")

        # Convert string component types to enum
        component_specs = []
        type_mapping = {
            'harmony': ComponentType.HARMONY,
            'rhythm': ComponentType.RHYTHM,
            'melody': ComponentType.MELODY,
            'bass': ComponentType.BASS,
            'drums': ComponentType.DRUMS,
            'instrumentation': ComponentType.INSTRUMENTATION
        }

        for comp_name, genre_weights in blends.items():
            comp_type = type_mapping.get(comp_name.lower())
            if not comp_type:
                raise ValueError(f"Unknown component type: {comp_name}")

            for genre, weight in genre_weights:
                component_specs.append((comp_type, genre, weight))

        fusion = ModularFusion()
        composition = fusion.weighted_fusion(
            component_specs=component_specs,
            tempo=tempo,
            key=key,
            length_measures=measures
        )

        self.composition = composition
        return composition

    # ==========================================================================
    # CONTEXT-AWARE GENERATION (Agent 3)
    # ==========================================================================

    def add_track(self,
                  instrument: int,
                  track_type: str = "auto",
                  genre: Optional[str] = None,
                  midi_file: Optional[str] = None) -> List:
        """
        Add new track to existing arrangement (context-aware)

        The new track will analyze the existing arrangement and fit seamlessly
        with proper harmony, rhythm, and voice leading.

        Args:
            instrument: MIDI program number (0-127)
            track_type: 'bass', 'harmony', 'melody', 'percussion', 'auto'
            genre: Genre for new track (None = match existing)
            midi_file: MIDI file to add to (None = use loaded)

        Returns:
            List of notes for new track

        Example:
            >>> api.load_midi("piano_drums.mid")
            >>> # Add funk bass that fits the existing chords/rhythm
            >>> bass_notes = api.add_track(
            ...     instrument=33,  # Fingered bass
            ...     track_type="bass",
            ...     genre="funk"
            ... )
            >>> api.export("with_bass.mid")
        """
        midi_file = midi_file or self.current_midi
        if not midi_file:
            raise ValueError("No MIDI file loaded")

        if not ContextAwareGenerator:
            raise NotImplementedError("ContextAwareGenerator not available")

        generator = ContextAwareGenerator(midi_file)
        new_track = generator.add_track(
            instrument=instrument,
            genre=genre,
            track_type=track_type
        )

        logger.info(f"Added {track_type} track (instrument {instrument})")
        return new_track

    def suggest_tracks(self, midi_file: Optional[str] = None) -> List[Dict]:
        """
        Get intelligent suggestions for tracks to add

        Args:
            midi_file: MIDI file to analyze (None = use loaded)

        Returns:
            List of track suggestions with reasoning

        Example:
            >>> api.load_midi("sparse_arrangement.mid")
            >>> suggestions = api.suggest_tracks()
            >>> for sug in suggestions:
            ...     print(f"{sug['instrument']}: {sug['reason']}")
        """
        midi_file = midi_file or self.current_midi
        if not midi_file:
            raise ValueError("No MIDI file loaded")

        if not SmartOrchestrator:
            raise NotImplementedError("SmartOrchestrator not available")

        orchestrator = SmartOrchestrator(midi_file)
        suggestions = orchestrator.suggest_additions()

        return suggestions

    # ==========================================================================
    # INPAINTING (Agent 4)
    # ==========================================================================

    def inpaint_section(self,
                       tracks: List[int],
                       measures: Tuple[int, int],
                       new_chords: Optional[List[str]] = None,
                       new_genre: Optional[str] = None,
                       preserve_melody: bool = False,
                       midi_file: Optional[str] = None) -> Dict[int, List]:
        """
        Regenerate section of tracks (like Photoshop content-aware fill)

        Args:
            tracks: Track numbers to regenerate
            measures: (start_measure, end_measure) to regenerate
            new_chords: New chord progression (None = keep existing)
            new_genre: New genre (None = match existing)
            preserve_melody: Keep melody notes, change only harmony
            midi_file: MIDI file (None = use loaded)

        Returns:
            Dict mapping track_number to new notes

        Example:
            >>> api.load_midi("song.mid")
            >>> # Regenerate measures 9-16 with new chords
            >>> api.inpaint_section(
            ...     tracks=[1, 2, 3],
            ...     measures=(9, 16),
            ...     new_chords=["Dm7", "G7", "Cmaj7", "A7alt"] * 2
            ... )
            >>> api.export("reharmonized.mid")
        """
        midi_file = midi_file or self.current_midi
        if not midi_file:
            raise ValueError("No MIDI file loaded")

        if not InpaintingEngine:
            raise NotImplementedError("InpaintingEngine not available")

        engine = InpaintingEngine(midi_file)
        regenerated = engine.inpaint_measures(
            track_numbers=tracks,
            start_measure=measures[0],
            end_measure=measures[1],
            new_chords=new_chords,
            new_genre=new_genre,
            preserve_melody=preserve_melody
        )

        logger.info(f"Inpainted measures {measures[0]}-{measures[1]}")
        return regenerated

    def reharmonize(self,
                   measures: Tuple[int, int],
                   style: str = "jazz",
                   midi_file: Optional[str] = None) -> List[str]:
        """
        Generate new chord progression for section

        Args:
            measures: (start, end) measure range
            style: Reharmonization style ('jazz', 'romantic', 'modal', 'chromatic')
            midi_file: MIDI file (None = use loaded)

        Returns:
            New chord progression

        Example:
            >>> api.load_midi("simple_song.mid")
            >>> new_chords = api.reharmonize(
            ...     measures=(1, 8),
            ...     style="jazz"
            ... )
            >>> print(new_chords)  # ['Cmaj7', 'Dm7', 'G7alt', ...]
        """
        midi_file = midi_file or self.current_midi
        if not midi_file:
            raise ValueError("No MIDI file loaded")

        if not InpaintingEngine:
            raise NotImplementedError("InpaintingEngine not available")

        engine = InpaintingEngine(midi_file)
        new_chords = engine.reharmonize_section(
            start_measure=measures[0],
            end_measure=measures[1],
            reharmonization_style=style
        )

        return new_chords

    # ==========================================================================
    # TEMPO CONVERSION (Agent 6)
    # ==========================================================================

    def convert_tempo(self,
                     new_tempo: int,
                     style_adjust: bool = True,
                     midi_file: Optional[str] = None) -> Any:
        """
        Convert MIDI to different tempo (with style-appropriate adjustments)

        Args:
            new_tempo: Target tempo in BPM
            style_adjust: Adjust patterns for musicality (not just speed)
            midi_file: MIDI file (None = use loaded)

        Returns:
            Converted composition

        Example:
            >>> # Convert 90 BPM jazz to 140 BPM (creates double-time feel)
            >>> api.load_midi("slow_jazz.mid")
            >>> api.convert_tempo(140, style_adjust=True)
            >>> api.export("uptempo_jazz.mid")
        """
        midi_file = midi_file or self.current_midi
        if not midi_file:
            raise ValueError("No MIDI file loaded")

        if not TempoConverter:
            raise NotImplementedError("TempoConverter not available")

        converter = TempoConverter(midi_file)

        if style_adjust and StyleTempoAdjuster:
            # Intelligent tempo conversion
            result = StyleTempoAdjuster.convert_with_style(
                midi_file, new_tempo
            )
        else:
            # Simple tempo scaling
            result = converter.convert_tempo(new_tempo)

        self.composition = result
        logger.info(f"Converted to {new_tempo} BPM")
        return result

    # ==========================================================================
    # METER CONVERSION (Agent 7)
    # ==========================================================================

    def convert_meter(self,
                     new_time_signature: Tuple[int, int],
                     midi_file: Optional[str] = None) -> Any:
        """
        Convert MIDI to different time signature

        Args:
            new_time_signature: (numerator, denominator) e.g., (3, 4) or (7, 8)
            midi_file: MIDI file (None = use loaded)

        Returns:
            Converted composition

        Example:
            >>> # Convert 4/4 to 7/8
            >>> api.load_midi("four_four.mid")
            >>> api.convert_meter((7, 8))
            >>> api.export("seven_eight.mid")
        """
        midi_file = midi_file or self.current_midi
        if not midi_file:
            raise ValueError("No MIDI file loaded")

        if not MeterConverter:
            raise NotImplementedError("MeterConverter not available")

        converter = MeterConverter(midi_file)
        result = converter.convert_meter(
            numerator=new_time_signature[0],
            denominator=new_time_signature[1]
        )

        self.composition = result
        logger.info(f"Converted to {new_time_signature[0]}/{new_time_signature[1]}")
        return result

    # ==========================================================================
    # GRANULAR CONTROL (Agent 8)
    # ==========================================================================

    def apply_pattern(self,
                     rhythm_pattern: List[float],
                     chords: List[str],
                     instrument_section: str = "brass",
                     key: str = "C") -> List:
        """
        Apply custom rhythm pattern to chords with idiomatic instrument writing

        Args:
            rhythm_pattern: List of note positions (in beats)
            chords: Chord progression
            instrument_section: 'brass', 'strings', 'woodwinds', 'piano', etc.
            key: Key signature

        Returns:
            Generated notes

        Example:
            >>> # Brass hits on specific rhythm
            >>> rhythm = [1.0, 1.5, 3.0, 3.75]  # Syncopated hits
            >>> chords = ["Dm7", "G7", "Cmaj7", "A7"]
            >>> notes = api.apply_pattern(
            ...     rhythm_pattern=rhythm,
            ...     chords=chords,
            ...     instrument_section="brass"
            ... )
        """
        if not GranularController:
            raise NotImplementedError("GranularController not available")

        controller = GranularController()
        notes = controller.apply_pattern_to_chords(
            rhythm_pattern=rhythm_pattern,
            chords=chords,
            instrument_section=instrument_section,
            key=key
        )

        return notes

    # ==========================================================================
    # PROGRESSIVE FUSION (Agent 5)
    # ==========================================================================

    def progressive_morph(self,
                         from_genre: str,
                         to_genre: str,
                         measures: int = 32,
                         morph_type: str = "linear") -> Any:
        """
        Create composition that gradually morphs from one genre to another

        Args:
            from_genre: Starting genre
            to_genre: Ending genre
            measures: Total length in measures
            morph_type: 'linear', 'exponential', 's-curve'

        Returns:
            Composition that transitions between genres

        Example:
            >>> # Smooth 32-bar transition from jazz to EDM
            >>> comp = api.progressive_morph(
            ...     from_genre="jazz",
            ...     to_genre="edm",
            ...     measures=32,
            ...     morph_type="s-curve"
            ... )
            >>> api.composition = comp
            >>> api.export("jazz_to_edm.mid")
        """
        if not ProgressiveFusion:
            raise NotImplementedError("ProgressiveFusion not available")

        fusion = ProgressiveFusion(from_genre, to_genre, measures)
        composition = fusion.generate_progressive_fusion(morph_type=morph_type)

        self.composition = composition
        logger.info(f"Created progressive morph: {from_genre} → {to_genre}")
        return composition

    # ==========================================================================
    # GENRE COMPATIBILITY (Agent 5)
    # ==========================================================================

    def check_compatibility(self, genre_a: str, genre_b: str) -> Dict[str, float]:
        """
        Check how compatible two genres are for fusion

        Args:
            genre_a: First genre
            genre_b: Second genre

        Returns:
            Compatibility scores (overall, rhythmic, harmonic, timbral, cultural)

        Example:
            >>> compat = api.check_compatibility("jazz", "funk")
            >>> print(f"Overall: {compat['overall']:.2f}")
            >>> print(f"Rhythmic: {compat['rhythmic']:.2f}")
        """
        if not GenreCompatibilityAnalyzer:
            raise NotImplementedError("GenreCompatibilityAnalyzer not available")

        return GenreCompatibilityAnalyzer.analyze_compatibility(genre_a, genre_b)

    def suggest_fusion(self, genre_a: str, genre_b: str) -> Dict:
        """
        Get optimal blending parameters for two genres

        Args:
            genre_a: First genre
            genre_b: Second genre

        Returns:
            Recommended fusion parameters

        Example:
            >>> params = api.suggest_fusion("jazz", "edm")
            >>> print(params['recommended_weight_a'])  # 0.6
            >>> print(params['tempo'])  # 115
        """
        if not GenreCompatibilityAnalyzer:
            raise NotImplementedError("GenreCompatibilityAnalyzer not available")

        return GenreCompatibilityAnalyzer.suggest_fusion_parameters(genre_a, genre_b)

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

    def list_genres(self) -> List[str]:
        """Get list of available genres"""
        return list(GENRE_PROFILES.keys()) if GENRE_PROFILES else []

    def get_genre_info(self, genre: str) -> Optional[GenreFeatures]:
        """Get detailed information about a genre"""
        return GENRE_PROFILES.get(genre) if GENRE_PROFILES else None

    def _add_to_history(self, operation: str, params: Dict):
        """Add operation to history (for undo capability)"""
        self.history.append({
            'operation': operation,
            'params': params,
            'composition': self.composition
        })


# ==============================================================================
# CONVENIENCE FUNCTIONS (Functional API)
# ==============================================================================

class QuickFusion:
    """Quick functions for common fusion tasks (functional API)"""

    @staticmethod
    def jazz_funk(tempo: int = 115, key: str = "Dm", measures: int = 16) -> Any:
        """Create jazz-funk fusion"""
        api = HarmonyModuleAPI()
        return api.quick_fusion(harmony="jazz", rhythm="funk", tempo=tempo,
                               key=key, measures=measures)

    @staticmethod
    def electro_swing(tempo: int = 128, key: str = "G", measures: int = 16) -> Any:
        """Create electro-swing fusion"""
        api = HarmonyModuleAPI()
        return api.quick_fusion(harmony="jazz", rhythm="swing",
                               instrumentation="electronic",
                               tempo=tempo, key=key, measures=measures)

    @staticmethod
    def afro_cuban_jazz(tempo: int = 140, key: str = "Cm", measures: int = 16) -> Any:
        """Create Afro-Cuban jazz fusion"""
        api = HarmonyModuleAPI()
        return api.quick_fusion(harmony="jazz", rhythm="latin",
                               tempo=tempo, key=key, measures=measures)


class GenreBlend:
    """Genre blending utilities"""

    @staticmethod
    def blend_two(genre_a: str, genre_b: str,
                  weight_a: float = 0.5,
                  tempo: int = 120,
                  key: str = "C",
                  measures: int = 16) -> Any:
        """Blend two genres with specified weights"""
        api = HarmonyModuleAPI()
        weight_b = 1.0 - weight_a

        return api.weighted_blend(
            blends={
                'harmony': [(genre_a, weight_a), (genre_b, weight_b)],
                'rhythm': [(genre_a, weight_a), (genre_b, weight_b)]
            },
            tempo=tempo,
            key=key,
            measures=measures
        )


class ComponentMix:
    """Component-level mixing utilities"""

    @staticmethod
    def custom_mix(components: Dict[str, str], **kwargs) -> Any:
        """
        Create custom component mix

        Args:
            components: Dict of component_type: genre
            **kwargs: Additional parameters (tempo, key, measures)

        Example:
            >>> comp = ComponentMix.custom_mix({
            ...     'harmony': 'jazz',
            ...     'rhythm': 'funk',
            ...     'bass': 'reggae',
            ...     'drums': 'hiphop'
            ... }, tempo=110, key='Em')
        """
        api = HarmonyModuleAPI()
        return api.quick_fusion(
            harmony=components.get('harmony', 'jazz'),
            rhythm=components.get('rhythm', 'jazz'),
            melody=components.get('melody'),
            bass=components.get('bass'),
            drums=components.get('drums'),
            instrumentation=components.get('instrumentation'),
            **kwargs
        )


class ContextGeneration:
    """Context-aware generation utilities"""

    @staticmethod
    def add_bass_to_midi(midi_file: str, genre: str = "funk",
                         output: str = "with_bass.mid") -> str:
        """Add bass to existing MIDI"""
        api = HarmonyModuleAPI()
        api.load_midi(midi_file)
        api.add_track(instrument=33, track_type="bass", genre=genre)
        return api.export(output)

    @staticmethod
    def add_drums_to_midi(midi_file: str, genre: str = "jazz",
                          output: str = "with_drums.mid") -> str:
        """Add drums to existing MIDI"""
        api = HarmonyModuleAPI()
        api.load_midi(midi_file)
        api.add_track(instrument=0, track_type="drums", genre=genre)
        return api.export(output)


class InpaintSection:
    """Inpainting utilities"""

    @staticmethod
    def reharmonize_section(midi_file: str,
                           start: int, end: int,
                           new_chords: List[str],
                           output: str = "reharmonized.mid") -> str:
        """Reharmonize section of MIDI"""
        api = HarmonyModuleAPI()
        api.load_midi(midi_file)
        api.inpaint_section(
            tracks=[0, 1, 2],  # Common tracks
            measures=(start, end),
            new_chords=new_chords
        )
        return api.export(output)


class TransformTempo:
    """Tempo transformation utilities"""

    @staticmethod
    def double_time(midi_file: str, output: str = "double_time.mid") -> str:
        """Convert to double-time feel"""
        api = HarmonyModuleAPI()
        info = api.load_midi(midi_file)
        original_tempo = info.get('tempo', 120)
        api.convert_tempo(original_tempo * 2)
        return api.export(output)

    @staticmethod
    def half_time(midi_file: str, output: str = "half_time.mid") -> str:
        """Convert to half-time feel"""
        api = HarmonyModuleAPI()
        info = api.load_midi(midi_file)
        original_tempo = info.get('tempo', 120)
        api.convert_tempo(original_tempo // 2)
        return api.export(output)


class TransformMeter:
    """Meter transformation utilities"""

    @staticmethod
    def to_waltz(midi_file: str, output: str = "waltz.mid") -> str:
        """Convert to 3/4 (waltz)"""
        api = HarmonyModuleAPI()
        api.load_midi(midi_file)
        api.convert_meter((3, 4))
        return api.export(output)

    @staticmethod
    def to_odd_meter(midi_file: str, numerator: int = 7,
                     output: str = "odd_meter.mid") -> str:
        """Convert to odd meter (default 7/8)"""
        api = HarmonyModuleAPI()
        api.load_midi(midi_file)
        api.convert_meter((numerator, 8))
        return api.export(output)


class GranularControl:
    """Granular pattern control utilities"""

    @staticmethod
    def brass_hits(rhythm: List[float], chords: List[str],
                   key: str = "C") -> List:
        """Generate brass hits on custom rhythm"""
        api = HarmonyModuleAPI()
        return api.apply_pattern(rhythm, chords, "brass", key)

    @staticmethod
    def string_swell(rhythm: List[float], chords: List[str],
                     key: str = "C") -> List:
        """Generate string swells"""
        api = HarmonyModuleAPI()
        return api.apply_pattern(rhythm, chords, "strings", key)


# ==============================================================================
# VERSION INFO
# ==============================================================================

__version__ = "1.0.0"
__author__ = "Agent 10 - Unified API & Integration"
__date__ = "2025"

__all__ = [
    'HarmonyModuleAPI',
    'QuickFusion',
    'GenreBlend',
    'ComponentMix',
    'ContextGeneration',
    'InpaintSection',
    'TransformTempo',
    'TransformMeter',
    'GranularControl',
]
