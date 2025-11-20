#!/usr/bin/env python3
"""
Big Band Generator API - Complete Integration Layer
====================================================

This module provides the main unified API for professional big band music generation.
It integrates all modules from the 20-agent system into a simple, cohesive interface.

Author: Agent 18 - Integration Architecture Designer
Date: 2025-11-20

Usage:
    # Simple generation
    >>> from api.big_band_api import BigBandGenerator
    >>> generator = BigBandGenerator(style="basie", tempo=140)
    >>> midi_file = generator.generate()
    >>> midi_file.save("basie_swing.mid")

    # Advanced generation with options
    >>> generator = BigBandGenerator(
    ...     style="ellington",
    ...     tempo=120,
    ...     key="Eb",
    ...     form="aaba",
    ...     progression_type="coltrane_changes"
    ... )
    >>> result = generator.generate()
    >>> result.save("ellington_arrangement.mid")
"""

import sys
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    raise ImportError("mido is required. Install with: pip install mido")

# Import style profiles
from .styles import StyleProfile, get_style, list_styles

# Import existing modules (with graceful fallbacks)
try:
    from genres.jazz import (
        JazzNote, JazzChord, BebopMelodyGenerator, SwingTiming, JazzStyle
    )
except ImportError:
    raise ImportError("Jazz module not found. Ensure midi_generator/genres/jazz.py exists")

try:
    from transformation.arrangement_engine import BigBandArranger
    from analysis.midi_analyzer import NoteEvent, ChordEvent
except ImportError:
    raise ImportError("Arrangement engine not found")

try:
    from generators.form_generator import FormGenerator, FormType, MusicalForm, FormSection
except ImportError:
    raise ImportError("Form generator not found")

try:
    from algorithms.rhythm_engine import HumanizationEngine, TimingStyle
    HAS_HUMANIZATION = True
except ImportError:
    HAS_HUMANIZATION = False
    warnings.warn("HumanizationEngine not available - skipping humanization")

# Try to import advanced modules (graceful degradation if not available yet)
try:
    from tools.big_band.generate_big_band_comprehensive import ComprehensiveHarmonyGenerator
    HAS_COMPREHENSIVE_HARMONY = True
except ImportError:
    HAS_COMPREHENSIVE_HARMONY = False

# Future modules (created by other agents) - detect if available
try:
    from genres.jazz_vocabulary import BebopVocabulary
    HAS_BEBOP_VOCABULARY = True
except ImportError:
    HAS_BEBOP_VOCABULARY = False

try:
    from transformation.sax_voicing import SaxSoliVoicing
    HAS_SAX_VOICING = True
except ImportError:
    HAS_SAX_VOICING = False

try:
    from genres.stride_piano import StridePianoGenerator
    HAS_STRIDE_PIANO = True
except ImportError:
    HAS_STRIDE_PIANO = False

try:
    from generators.reharmonization_engine import ReharmonizationEngine
    HAS_REHARMONIZATION = True
except ImportError:
    HAS_REHARMONIZATION = False

try:
    from transformation.articulation_engine import ArticulationEngine
    HAS_ARTICULATION_ENGINE = True
except ImportError:
    HAS_ARTICULATION_ENGINE = False

try:
    from transformation.dynamic_shaping import DynamicShaping
    HAS_DYNAMIC_SHAPING = True
except ImportError:
    HAS_DYNAMIC_SHAPING = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BigBandConfig:
    """Configuration for big band generation"""
    style: str = "basie"
    tempo: int = 140
    key: Union[str, int] = "C"
    form: str = "aaba"
    progression_type: str = "jazz_blues"
    swing_ratio: Optional[float] = None  # None = use style default
    ticks_per_beat: int = 480

    def __post_init__(self):
        """Validate and normalize configuration"""
        # Normalize key to MIDI note number
        if isinstance(self.key, str):
            self.key = self._key_name_to_number(self.key)

    @staticmethod
    def _key_name_to_number(key_name: str) -> int:
        """Convert key name (e.g., 'C', 'Eb', 'F#') to MIDI note number"""
        key_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        key_name = key_name.strip().title()
        if key_name not in key_map:
            raise ValueError(f"Unknown key: {key_name}. Use format like 'C', 'Eb', 'F#', etc.")
        return key_map[key_name]


# ============================================================================
# MAIN API CLASS
# ============================================================================

class BigBandGenerator:
    """
    Main API for professional big band generation

    This class provides a simple interface for generating complete big band
    arrangements in various styles (Basie, Ellington, Thad Jones, etc.).

    Attributes:
        config: BigBandConfig with generation parameters
        style_profile: StyleProfile for the selected style
        form: Generated MusicalForm structure
        progression: Generated chord progression
        arrangement: Generated arrangement (dict of instrument parts)

    Example:
        >>> generator = BigBandGenerator(style="basie", tempo=140)
        >>> midi = generator.generate()
        >>> midi.save("output.mid")
    """

    def __init__(self,
                 style: str = "basie",
                 tempo: int = 140,
                 key: Union[str, int] = "C",
                 form: str = "aaba",
                 progression_type: str = "jazz_blues",
                 swing_ratio: Optional[float] = None):
        """
        Initialize BigBandGenerator

        Args:
            style: Arranging style ("basie", "ellington", "thad_jones")
            tempo: Tempo in BPM (120-200 typical)
            key: Key signature ("C", "Eb", "F#", etc. or MIDI note 0-11)
            form: Musical form ("aaba", "blues")
            progression_type: Harmony type ("jazz_blues", "rhythm_changes", etc.)
            swing_ratio: Swing ratio override (None = use style default)
        """
        self.config = BigBandConfig(
            style=style,
            tempo=tempo,
            key=key,
            form=form,
            progression_type=progression_type,
            swing_ratio=swing_ratio
        )

        # Load style profile
        self.style_profile = get_style(style)

        # Check tempo appropriateness
        self.config.tempo = self.style_profile.get_suggested_tempo(tempo)

        # Override swing ratio if specified
        if swing_ratio is not None:
            self.style_profile.swing_ratio = swing_ratio

        # Initialize generators
        self._init_generators()

        # Generation results (populated by generate())
        self.form: Optional[MusicalForm] = None
        self.progression: Optional[List[JazzChord]] = None
        self.arrangement: Optional[Dict] = None


    def _init_generators(self):
        """Initialize generation modules"""
        self.harmony_gen = self._create_harmony_generator()
        self.melody_gen = BebopMelodyGenerator()

        if HAS_HUMANIZATION:
            self.humanizer = HumanizationEngine(ppqn=self.config.ticks_per_beat)
        else:
            self.humanizer = None


    def _create_harmony_generator(self):
        """Create harmony generator (with fallback)"""
        if HAS_COMPREHENSIVE_HARMONY:
            return ComprehensiveHarmonyGenerator(key=self.config.key)
        else:
            # Fallback to basic harmony
            from genres.jazz import JazzProgressions
            return JazzProgressions()


    def generate(self) -> MidiFile:
        """
        Generate complete big band arrangement

        This is the main method - it orchestrates the entire generation pipeline.

        Returns:
            MidiFile object ready to save

        Pipeline:
            1. Generate form structure
            2. Generate harmony progression
            3. Generate melody
            4. Apply swing timing
            5. Create big band arrangement
            6. Apply articulations (if available)
            7. Apply dynamics (if available)
            8. Humanize (if available)
            9. Export to MIDI
        """
        print(f"\n{'='*80}")
        print(f"BIG BAND GENERATOR - {self.style_profile.style_name} Style")
        print(f"{'='*80}")
        print(f"Tempo: {self.config.tempo} BPM")
        print(f"Key: {self._get_key_name()}")
        print(f"Form: {self.config.form.upper()}")
        print(f"Style: {self.style_profile.style_name} ({self.style_profile.era})")
        print(f"Swing: {self.style_profile.swing_ratio:.2f}")
        print()

        # Step 1: Generate form
        print("✓ Step 1: Generating form structure...")
        self.form = self._generate_form()
        print(f"  {self.form.form_type.value.upper()}: {self.form.total_bars} bars")

        # Step 2: Generate harmony
        print("✓ Step 2: Generating harmony progression...")
        self.progression = self._generate_harmony()
        print(f"  {len(self.progression)} chords")

        # Step 3: Generate melody
        print("✓ Step 3: Generating melody...")
        melody = self._generate_melody()
        print(f"  {len(melody)} notes")

        # Step 4: Apply swing
        print(f"✓ Step 4: Applying swing (ratio: {self.style_profile.swing_ratio})...")
        melody = self._apply_swing(melody)

        # Step 5: Big band arrangement
        print(f"✓ Step 5: Creating big band arrangement...")
        print(f"  Using {self.style_profile.style_name} arranging principles")
        self.arrangement = self._create_arrangement(melody)
        self._print_arrangement_stats()

        # Step 6: Apply articulations (if available)
        if HAS_ARTICULATION_ENGINE:
            print("✓ Step 6: Applying articulations...")
            self.arrangement = self._apply_articulations()
        else:
            print("⊗ Step 6: Articulations (not yet available)")

        # Step 7: Apply dynamics (if available)
        if HAS_DYNAMIC_SHAPING:
            print("✓ Step 7: Applying dynamic shaping...")
            self.arrangement = self._apply_dynamics()
        else:
            print("⊗ Step 7: Dynamic shaping (not yet available)")

        # Step 8: Humanization (if available)
        if self.humanizer:
            print("✓ Step 8: Applying humanization...")
            # Note: Humanization implementation pending
        else:
            print("⊗ Step 8: Humanization (not yet available)")

        # Step 9: Export to MIDI
        print("✓ Step 9: Exporting to MIDI...")
        midi_file = self._export_midi()

        print()
        print(f"{'='*80}")
        print("✅ GENERATION COMPLETE")
        print(f"{'='*80}")

        return midi_file


    def _generate_form(self) -> MusicalForm:
        """Generate musical form structure"""
        if self.config.form.lower() == "aaba":
            sections = [
                FormSection(name="A1", key_relationship=None, length_bars=8),
                FormSection(name="A2", key_relationship=None, length_bars=8),
                FormSection(name="B", key_relationship=None, length_bars=8),
                FormSection(name="A3", key_relationship=None, length_bars=8),
            ]
            form = MusicalForm(
                form_type=FormType.AABA,
                sections=sections,
                tonic_key=self.config.key,
                is_major=True,
                tempo=self.config.tempo,
                time_signature=(4, 4)
            )
        elif self.config.form.lower() == "blues":
            sections = [FormSection(name="Blues", key_relationship=None, length_bars=12)]
            form = MusicalForm(
                form_type=FormType.TWELVE_BAR_BLUES,
                sections=sections,
                tonic_key=self.config.key,
                is_major=True,
                tempo=self.config.tempo,
                time_signature=(4, 4)
            )
        else:
            raise ValueError(f"Unknown form: {self.config.form}. Use 'aaba' or 'blues'")

        return form


    def _generate_harmony(self) -> List[JazzChord]:
        """Generate chord progression"""
        if HAS_COMPREHENSIVE_HARMONY:
            progression, description = self.harmony_gen.generate_progression(
                self.config.progression_type
            )
            print(f"  Type: {description}")
        else:
            # Fallback: basic ii-V-I
            progression = self.harmony_gen.ii_V_I(self.config.key)

        # Expand to match form length
        total_bars = self.form.total_bars
        full_progression = []
        while len(full_progression) < total_bars:
            full_progression.extend(progression)

        return full_progression[:total_bars]


    def _generate_melody(self) -> List[JazzNote]:
        """Generate bebop melody"""
        melody = []

        for i, chord in enumerate(self.progression):
            # Add rests for phrasing (every 4 bars)
            if i > 0 and i % 4 == 3:
                continue

            # Generate phrase
            phrase = self.melody_gen.generate_phrase(
                chord,
                length_beats=4,
                density=0.7
            )

            # Adjust timing
            chord_start = i * 4.0
            for note in phrase:
                note.start_time += chord_start

            melody.extend(phrase)

        return melody


    def _apply_swing(self, melody: List[JazzNote]) -> List[JazzNote]:
        """Apply swing timing"""
        swung = SwingTiming.apply_swing(
            melody,
            swing_ratio=self.style_profile.swing_ratio,
            intensity=self.style_profile.swing_intensity
        )

        # Fix durations to prevent overlap
        return self._fix_durations(swung)


    def _fix_durations(self, notes: List[JazzNote]) -> List[JazzNote]:
        """Fix note durations to prevent overlap"""
        if not notes:
            return []

        fixed = []
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        for i, note in enumerate(sorted_notes):
            new_note = JazzNote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                duration=note.duration,
                articulation=note.articulation,
                swing_offset=getattr(note, 'swing_offset', 0),
                channel=note.channel
            )

            # Prevent overlap with next note
            if i < len(sorted_notes) - 1:
                next_start = sorted_notes[i + 1].start_time
                note_end = new_note.start_time + new_note.duration
                if note_end > next_start:
                    gap = 0.05
                    new_note.duration = max(0.1, next_start - new_note.start_time - gap)

            fixed.append(new_note)

        return fixed


    def _create_arrangement(self, melody: List[JazzNote]) -> Dict:
        """Create big band arrangement"""
        # Convert to NoteEvent/ChordEvent format
        melody_events = self._jazz_to_note_events(melody)
        chord_events = self._jazz_to_chord_events(self.progression)

        # Use BigBandArranger
        arrangement = BigBandArranger.arrange(melody_events, chord_events)

        return arrangement


    def _jazz_to_note_events(self, jazz_notes: List[JazzNote]) -> List[NoteEvent]:
        """Convert JazzNote to NoteEvent"""
        return [
            NoteEvent(
                start_time=jn.start_time,
                duration=jn.duration,
                start_tick=int(jn.start_time * self.config.ticks_per_beat),
                duration_ticks=int(jn.duration * self.config.ticks_per_beat),
                pitch=jn.pitch,
                velocity=jn.velocity,
                channel=jn.channel or 0,
                track_idx=0
            )
            for jn in jazz_notes
        ]


    def _jazz_to_chord_events(self, jazz_chords: List[JazzChord]) -> List[ChordEvent]:
        """Convert JazzChord to ChordEvent"""
        return [
            ChordEvent(
                start_time=i * 4.0,
                duration=4.0,
                root=jc.root,
                quality=jc.quality,
                pitches=[jc.root],
                bass_note=jc.root,
                confidence=1.0
            )
            for i, jc in enumerate(jazz_chords)
        ]


    def _apply_articulations(self) -> Dict:
        """Apply articulations based on style (future)"""
        # Placeholder for Agent 8's articulation engine
        return self.arrangement


    def _apply_dynamics(self) -> Dict:
        """Apply dynamic shaping based on form (future)"""
        # Placeholder for Agent 9's dynamic shaping
        return self.arrangement


    def _export_midi(self) -> MidiFile:
        """Export arrangement to MIDI file"""
        mid = MidiFile(ticks_per_beat=self.config.ticks_per_beat)
        tempo_microseconds = int(60_000_000 / self.config.tempo)

        # Track mapping
        track_mapping = [
            ("Lead Melody", 65, self.arrangement.get('lead', []), 0),
            ("Sax Section", 66, self.arrangement.get('saxes', []), 1),
            ("Brass Section", 56, self.arrangement.get('brass', []), 5),
            ("Piano", 0, self.arrangement.get('piano', []), 14),
            ("Bass", 32, self.arrangement.get('bass', []), 15),
            ("Drums", 0, self.arrangement.get('drums', []), 9),
        ]

        for track_name, program, notes, channel in track_mapping:
            if not notes:
                continue

            track = MidiTrack()
            mid.tracks.append(track)

            track.append(MetaMessage('track_name', name=track_name, time=0))
            track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))

            if channel != 9:  # Not drums
                track.append(Message('program_change', program=program, channel=channel, time=0))

            # Convert to MIDI events
            events = []
            for note in notes:
                note_channel = getattr(note, 'channel', channel)
                events.append({
                    'type': 'note_on',
                    'time': note.start_time,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'channel': note_channel
                })
                events.append({
                    'type': 'note_off',
                    'time': note.start_time + note.duration,
                    'pitch': note.pitch,
                    'channel': note_channel
                })

            events.sort(key=lambda e: e['time'])

            # Write MIDI messages
            last_time = 0.0
            for event in events:
                delta_ticks = int((event['time'] - last_time) * self.config.ticks_per_beat)
                if event['type'] == 'note_on':
                    track.append(Message(
                        'note_on',
                        note=event['pitch'],
                        velocity=event['velocity'],
                        channel=event['channel'],
                        time=delta_ticks
                    ))
                else:
                    track.append(Message(
                        'note_off',
                        note=event['pitch'],
                        velocity=0,
                        channel=event['channel'],
                        time=delta_ticks
                    ))
                last_time = event['time']

            track.append(MetaMessage('end_of_track', time=0))

        return mid


    def _print_arrangement_stats(self):
        """Print arrangement statistics"""
        for section, notes in self.arrangement.items():
            print(f"  {section.title()}: {len(notes)} notes")


    def _get_key_name(self) -> str:
        """Get key name from key number"""
        key_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        return key_names[self.config.key % 12]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def list_available_styles() -> List[str]:
    """List all available arranging styles"""
    return list_styles()


def generate_big_band(style: str = "basie",
                      tempo: int = 140,
                      key: str = "C",
                      output_file: Optional[str] = None) -> MidiFile:
    """
    Quick generation function

    Args:
        style: Arranging style
        tempo: Tempo in BPM
        key: Key signature
        output_file: Optional output file path

    Returns:
        MidiFile object

    Example:
        >>> midi = generate_big_band("basie", 140, "Eb", "basie_swing.mid")
    """
    generator = BigBandGenerator(style=style, tempo=tempo, key=key)
    midi = generator.generate()

    if output_file:
        midi.save(output_file)
        print(f"\n✅ Saved: {output_file}\n")

    return midi


# ============================================================================
# MODULE INFO
# ============================================================================

__version__ = "1.0.0"
__author__ = "Agent 18 - Integration Architecture Designer"
__all__ = [
    'BigBandGenerator',
    'BigBandConfig',
    'generate_big_band',
    'list_available_styles',
]
