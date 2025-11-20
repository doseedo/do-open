#!/usr/bin/env python3
"""
Generic Arranger - Base Class for All Arrangers

Provides a universal arranging pipeline that works for ANY ensemble (big band, orchestra,
chamber, world music, etc.) using the Template Method design pattern.

This base class defines the arranging process while allowing subclasses to customize
specific steps for their genre/ensemble.

Design Pattern: Template Method
- Defines algorithm structure in base class
- Allows subclasses to override specific steps
- Ensures consistent process across all arrangers

Author: Agent 19 - Genre Scalability Architect
Date: 2025-11-20
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import copy

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from analysis.midi_analyzer import NoteEvent, ChordEvent
except ImportError:
    # Fallback for testing
    @dataclass
    class NoteEvent:
        start_time: float
        duration: float
        start_tick: int
        duration_ticks: int
        pitch: int
        velocity: int
        channel: int
        track_idx: int

        @property
        def end_time(self):
            return self.start_time + self.duration

    @dataclass
    class ChordEvent:
        start_time: float
        duration: float
        root: int
        chord_type: str
        notes: List[int]

try:
    from core.ensemble_registry import EnsembleConfig, get_ensemble
    from styles.style_registry import StyleProfile, get_style
except ImportError:
    EnsembleConfig = None
    StyleProfile = None
    def get_ensemble(x): return None
    def get_style(x): return None

try:
    from generators.form_generator import MusicalForm
except ImportError:
    MusicalForm = None


class GenericArranger(ABC):
    """
    Base class for all arrangers (BigBand, Orchestra, Choir, etc.)

    Provides universal arranging pipeline while allowing genre-specific customization.

    Usage:
        class BigBandArranger(GenericArranger):
            def _arrange_melody(self, melody, form):
                # Big band specific melody arrangement
                pass

            # ... implement other abstract methods

        arranger = BigBandArranger("big_band", "basie")
        arrangement = arranger.arrange(melody, harmony)
    """

    def __init__(self,
                 ensemble_type: str,
                 style_name: Optional[str] = None):
        """
        Initialize arranger with ensemble configuration and optional style profile.

        Args:
            ensemble_type: Type of ensemble (e.g., "big_band", "string_quartet")
            style_name: Optional style profile name (e.g., "basie", "mozart")
        """
        # Load ensemble configuration
        if isinstance(ensemble_type, str):
            self.ensemble = get_ensemble(ensemble_type)
            if self.ensemble is None:
                raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        else:
            self.ensemble = ensemble_type

        # Load style profile
        self.style = get_style(style_name) if style_name else None

        # Initialize universal engines (would be imported from actual modules)
        # These are placeholders for the actual implementations
        self.voice_leading_optimizer = None  # VoiceLeadingOptimizer()
        self.dynamic_shaper = None  # DynamicShaping()
        self.humanizer = None  # HumanizationEngine()

    def arrange(self,
                melody: List[NoteEvent],
                harmony: List[ChordEvent],
                form: Optional[MusicalForm] = None) -> Dict[str, List[NoteEvent]]:
        """
        Main arranging pipeline (TEMPLATE METHOD PATTERN).

        This method defines the universal arranging process.
        Subclasses override specific steps via abstract methods.

        Args:
            melody: Lead melody as list of NoteEvents
            harmony: Chord progression as list of ChordEvents
            form: Optional musical form (AABA, sonata, etc.)

        Returns:
            Dictionary mapping section names to note lists
            Example: {
                'lead': [NoteEvent, ...],
                'harmony': [NoteEvent, ...],
                'bass': [NoteEvent, ...],
                'rhythm': [NoteEvent, ...]
            }
        """
        arrangement = {}

        # Step 1: Prepare form structure
        if form is None:
            form = self._default_form(melody, harmony)

        # Step 2: Create lead/melody line (GENRE-SPECIFIC)
        arrangement['lead'] = self._arrange_melody(melody, form)

        # Step 3: Create harmonic voicings (GENRE-SPECIFIC)
        arrangement['harmony'] = self._arrange_harmony(harmony, melody, form)

        # Step 4: Create bass line (GENRE-SPECIFIC)
        arrangement['bass'] = self._arrange_bass(harmony, form)

        # Step 5: Create rhythm section (GENRE-SPECIFIC)
        arrangement['rhythm'] = self._arrange_rhythm(harmony, form)

        # Step 6: Apply voice leading optimization (UNIVERSAL)
        if self.voice_leading_optimizer:
            arrangement = self._optimize_voice_leading(arrangement)

        # Step 7: Apply dynamics and phrasing (UNIVERSAL)
        if self.dynamic_shaper:
            arrangement = self._apply_dynamics(arrangement, form)

        # Step 8: Apply articulations (GENRE-SPECIFIC)
        arrangement = self._apply_articulations(arrangement)

        # Step 9: Apply humanization (UNIVERSAL)
        if self.humanizer:
            arrangement = self._apply_humanization(arrangement)

        # Step 10: Create intro/outro if needed
        if self.style and hasattr(self.style, 'intro_style'):
            arrangement = self._add_intro_outro(arrangement, harmony, form)

        return arrangement

    # ==========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # ==========================================================================

    @abstractmethod
    def _arrange_melody(self,
                       melody: List[NoteEvent],
                       form: Optional[MusicalForm]) -> List[NoteEvent]:
        """
        Arrange melody for this ensemble (genre-specific).

        Examples:
        - Big band: Assign to lead alto or trumpet
        - Orchestra: Assign to violin or flute
        - String quartet: First violin with ornamentation

        Args:
            melody: Original melody
            form: Musical form

        Returns:
            Arranged melody notes
        """
        pass

    @abstractmethod
    def _arrange_harmony(self,
                        harmony: List[ChordEvent],
                        melody: List[NoteEvent],
                        form: Optional[MusicalForm]) -> List[NoteEvent]:
        """
        Create harmonic voicings for this ensemble (genre-specific).

        Examples:
        - Big band: Sax soli with drop-2 voicing
        - Orchestra: String section divisi
        - Jazz combo: Piano rootless voicings
        - Indian classical: Tanpura drone

        Args:
            harmony: Chord progression
            melody: Melody (for harmonization)
            form: Musical form

        Returns:
            Harmony notes
        """
        pass

    @abstractmethod
    def _arrange_bass(self,
                     harmony: List[ChordEvent],
                     form: Optional[MusicalForm]) -> List[NoteEvent]:
        """
        Create bass line for this ensemble (genre-specific).

        Examples:
        - Big band: Walking bass line
        - Orchestra: Cello and double bass
        - Rock band: Electric bass with root-fifth pattern

        Args:
            harmony: Chord progression
            form: Musical form

        Returns:
            Bass notes
        """
        pass

    @abstractmethod
    def _arrange_rhythm(self,
                       harmony: List[ChordEvent],
                       form: Optional[MusicalForm]) -> List[NoteEvent]:
        """
        Create rhythm section/percussion for this ensemble (genre-specific).

        Examples:
        - Big band: Swing drums + piano comping + guitar
        - Orchestra: Timpani + percussion
        - Indian classical: Tabla with tala pattern

        Args:
            harmony: Chord progression (for comping)
            form: Musical form

        Returns:
            Rhythm/percussion notes
        """
        pass

    @abstractmethod
    def _apply_articulations(self,
                            arrangement: Dict[str, List[NoteEvent]]) -> Dict[str, List[NoteEvent]]:
        """
        Apply genre-specific articulations.

        Examples:
        - Big band: Falls, doits, rips on brass
        - Orchestra: Pizzicato, arco, tremolo on strings
        - Indian classical: Meend (glides), gamak (oscillations)

        Args:
            arrangement: Current arrangement

        Returns:
            Arrangement with articulations applied
        """
        pass

    # ==========================================================================
    # UNIVERSAL METHODS (Used by all arrangers)
    # ==========================================================================

    def _optimize_voice_leading(self,
                                arrangement: Dict[str, List[NoteEvent]]) -> Dict[str, List[NoteEvent]]:
        """
        Apply voice leading optimization (UNIVERSAL).

        Uses VoiceLeadingOptimizer to minimize voice movement between chords.
        This works for ANY harmony: jazz, classical, pop, etc.

        Args:
            arrangement: Current arrangement

        Returns:
            Arrangement with optimized voice leading
        """
        # Placeholder - would use actual VoiceLeadingOptimizer
        # if 'harmony' in arrangement:
        #     optimized = self.voice_leading_optimizer.optimize_chord_sequence(
        #         arrangement['harmony'],
        #         num_voices=self._get_num_harmony_voices(),
        #         voice_ranges=self._get_voice_ranges()
        #     )
        #     arrangement['harmony'] = optimized
        return arrangement

    def _apply_dynamics(self,
                       arrangement: Dict[str, List[NoteEvent]],
                       form: Optional[MusicalForm]) -> Dict[str, List[NoteEvent]]:
        """
        Apply dynamic shaping based on form (UNIVERSAL).

        Adds crescendo, diminuendo, phrase contours.
        Works for ANY music with phrasing.

        Args:
            arrangement: Current arrangement
            form: Musical form

        Returns:
            Arrangement with dynamics applied
        """
        # Placeholder - would use actual DynamicShaping
        # if self.dynamic_shaper and form:
        #     dynamic_map = self.dynamic_shaper.generate_dynamic_map_for_form(form)
        #
        #     for section_name, notes in arrangement.items():
        #         arrangement[section_name] = self.dynamic_shaper.apply_phrase_contour(
        #             notes,
        #             phrase_length_bars=4,
        #             contour="arch"
        #         )
        return arrangement

    def _apply_humanization(self,
                           arrangement: Dict[str, List[NoteEvent]]) -> Dict[str, List[NoteEvent]]:
        """
        Apply humanization (timing/velocity variation) (UNIVERSAL).

        Adds subtle timing and velocity variations to make it sound human.
        Works for ANY performed music.

        Args:
            arrangement: Current arrangement

        Returns:
            Arrangement with humanization applied
        """
        # Placeholder - would use actual HumanizationEngine
        # for section_name, notes in arrangement.items():
        #     arrangement[section_name] = self.humanizer.apply_timing_variation(
        #         notes,
        #         amount=0.02  # 2% variation
        #     )
        return arrangement

    def _add_intro_outro(self,
                        arrangement: Dict[str, List[NoteEvent]],
                        harmony: List[ChordEvent],
                        form: Optional[MusicalForm]) -> Dict[str, List[NoteEvent]]:
        """
        Add intro and outro sections based on style profile.

        Examples:
        - Basie: Vamp intro, button ending
        - Ellington: Rubato intro, fermata ending
        - Classical: Fanfare intro, authentic cadence

        Args:
            arrangement: Current arrangement
            harmony: Chord progression
            form: Musical form

        Returns:
            Arrangement with intro/outro
        """
        # Subclasses can override for specific intro/outro generation
        return arrangement

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _get_num_harmony_voices(self) -> int:
        """
        Get number of harmony voices from ensemble config.

        Returns:
            Number of voices available for harmony
        """
        if not self.ensemble:
            return 4  # Default

        count = 0
        for section in self.ensemble.sections.values():
            if 'harmony' in section.role.lower():
                count += len(section.instruments)
        return count if count > 0 else 4

    def _get_voice_ranges(self) -> List[Tuple[int, int]]:
        """
        Get voice ranges from ensemble config.

        Returns:
            List of (low, high) MIDI note ranges for each voice
        """
        if not self.ensemble:
            return [(48, 72)] * 4  # Default ranges

        ranges = []
        for section in self.ensemble.sections.values():
            for inst_type, range_tuple in section.ranges.items():
                ranges.append(range_tuple)
        return ranges if ranges else [(48, 72)] * 4

    def _default_form(self,
                     melody: List[NoteEvent],
                     harmony: List[ChordEvent]) -> Optional[MusicalForm]:
        """
        Create default form if none provided.

        Args:
            melody: Melody
            harmony: Harmony

        Returns:
            Default form (e.g., simple AABA)
        """
        # Subclasses can override to provide genre-specific default forms
        # For now, return None
        return None

    def _find_section_by_role(self, role: str):
        """
        Find ensemble section by role.

        Args:
            role: Role to search for (e.g., "melody", "harmony", "bass")

        Returns:
            SectionConfig if found, None otherwise
        """
        if not self.ensemble:
            return None

        for section in self.ensemble.sections.values():
            if role.lower() in section.role.lower():
                return section
        return None

    def _get_instruments_by_role(self, role: str) -> List[str]:
        """
        Get instrument names by role.

        Args:
            role: Role to search for

        Returns:
            List of instrument names
        """
        section = self._find_section_by_role(role)
        return section.instruments if section else []


# ==============================================================================
# EXAMPLE: Simple Arranger Implementation
# ==============================================================================

class SimpleArranger(GenericArranger):
    """
    Simple example arranger that provides basic implementations.

    This shows how to extend GenericArranger with minimal code.
    """

    def _arrange_melody(self, melody, form):
        """Just return melody as-is"""
        return melody

    def _arrange_harmony(self, harmony, melody, form):
        """Create simple harmony from chords"""
        harmony_notes = []
        for chord in harmony:
            # Simple close voicing
            for i, pitch in enumerate(chord.notes[:4]):  # Take first 4 notes
                note = NoteEvent(
                    start_time=chord.start_time,
                    duration=chord.duration,
                    start_tick=int(chord.start_time * 480),
                    duration_ticks=int(chord.duration * 480),
                    pitch=pitch,
                    velocity=70,
                    channel=i,
                    track_idx=i+1
                )
                harmony_notes.append(note)
        return harmony_notes

    def _arrange_bass(self, harmony, form):
        """Simple root notes on beat 1"""
        bass_notes = []
        for chord in harmony:
            note = NoteEvent(
                start_time=chord.start_time,
                duration=0.9,  # Quarter note
                start_tick=int(chord.start_time * 480),
                duration_ticks=int(0.9 * 480),
                pitch=chord.root + 24,  # Low octave
                velocity=90,
                channel=1,
                track_idx=10
            )
            bass_notes.append(note)
        return bass_notes

    def _arrange_rhythm(self, harmony, form):
        """Simple drum pattern"""
        # Placeholder - no drums in this simple example
        return []

    def _apply_articulations(self, arrangement):
        """No articulations in simple example"""
        return arrangement


if __name__ == "__main__":
    # Test the generic arranger
    print("Generic Arranger Base Class")
    print("=" * 50)
    print("\nThis is an abstract base class for all arrangers.")
    print("\nSubclasses must implement:")
    print("  - _arrange_melody()")
    print("  - _arrange_harmony()")
    print("  - _arrange_bass()")
    print("  - _arrange_rhythm()")
    print("  - _apply_articulations()")
    print("\nUniversal methods (provided by base class):")
    print("  - _optimize_voice_leading()")
    print("  - _apply_dynamics()")
    print("  - _apply_humanization()")
    print("\nSee SimpleArranger for a minimal implementation example.")
