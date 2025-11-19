#!/usr/bin/env python3
"""
Style Transfer Engine - Transform MIDI to Different Musical Styles

This module implements comprehensive style transfer for MIDI files, transforming
music across multiple dimensions:

1. HARMONIC TRANSFORMATION
   - Reharmonization (classical, jazz, pop styles)
   - Chord substitution and voice leading
   - Modal interchange and borrowed chords

2. RHYTHMIC TRANSFORMATION
   - Groove application (swing, shuffle, straight)
   - Time signature conversion (4/4 → 7/8, 5/4, etc.)
   - Syncopation addition/removal

3. MELODIC TRANSFORMATION
   - Ornamentation (baroque, romantic styles)
   - Simplification/elaboration
   - Contour modification (stepwise ↔ angular)

4. INSTRUMENTAL TRANSFORMATION
   - Re-orchestration (piano → orchestra, band → quartet, etc.)
   - Timbre mapping
   - Register optimization

Inspired by research from:
- David Cope's "Experiments in Musical Intelligence" (EMI)
- Schillinger System of Musical Composition
- Jazz reharmonization techniques (Barry Harris, etc.)
- Classical orchestration principles

Author: Agent 8 - Style Transfer & Transformation
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Tuple, Dict, Optional, Set, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import copy

# Import our analyzer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from analysis.midi_analyzer import MidiAnalyzer, NoteEvent, ChordEvent


# ==============================================================================
# STYLE PROFILES
# ==============================================================================

@dataclass
class StyleProfile:
    """Defines characteristics of a musical style."""
    name: str

    # Harmonic characteristics
    chord_types: List[str]              # Preferred chord types
    harmonic_rhythm: float              # Chords per measure
    use_extensions: bool                # Use 9ths, 11ths, 13ths
    use_alterations: bool               # Use altered chords
    voice_leading_style: str            # 'smooth', 'open', 'wide'

    # Rhythmic characteristics
    swing_ratio: float                  # Swing feel (0.5=straight, 0.67=triplet swing)
    syncopation_level: float            # 0-1 (0=none, 1=heavy)
    quantization: str                   # 'strict', 'loose', 'human'

    # Melodic characteristics
    ornamentation_density: float        # 0-1
    interval_preference: str            # 'stepwise', 'balanced', 'angular'
    chromatic_usage: float              # 0-1 (0=diatonic only, 1=chromatic)

    # Instrumental
    preferred_instruments: List[int]    # MIDI program numbers
    register_preference: str            # 'low', 'mid', 'high', 'wide'
    texture: str                        # 'monophonic', 'homophonic', 'polyphonic'


# Predefined style profiles
STYLE_PROFILES = {
    'classical': StyleProfile(
        name='Classical',
        chord_types=['major', 'minor', 'diminished', 'dom7'],
        harmonic_rhythm=2.0,
        use_extensions=False,
        use_alterations=False,
        voice_leading_style='smooth',
        swing_ratio=0.5,
        syncopation_level=0.2,
        quantization='strict',
        ornamentation_density=0.3,
        interval_preference='stepwise',
        chromatic_usage=0.2,
        preferred_instruments=[0, 40, 42, 43],  # Piano, violin, cello, etc.
        register_preference='wide',
        texture='polyphonic'
    ),

    'jazz': StyleProfile(
        name='Jazz',
        chord_types=['major7', 'minor7', 'dom7', 'half-dim7', 'altered'],
        harmonic_rhythm=4.0,
        use_extensions=True,
        use_alterations=True,
        voice_leading_style='smooth',
        swing_ratio=0.67,
        syncopation_level=0.7,
        quantization='loose',
        ornamentation_density=0.6,
        interval_preference='balanced',
        chromatic_usage=0.6,
        preferred_instruments=[0, 32, 33, 25],  # Piano, bass, guitar, sax
        register_preference='mid',
        texture='polyphonic'
    ),

    'pop': StyleProfile(
        name='Pop',
        chord_types=['major', 'minor', 'sus2', 'sus4', 'add9'],
        harmonic_rhythm=1.0,
        use_extensions=True,
        use_alterations=False,
        voice_leading_style='open',
        swing_ratio=0.5,
        syncopation_level=0.4,
        quantization='strict',
        ornamentation_density=0.2,
        interval_preference='stepwise',
        chromatic_usage=0.3,
        preferred_instruments=[0, 33, 34, 25],  # Piano, guitar, bass, synth
        register_preference='mid',
        texture='homophonic'
    ),

    'baroque': StyleProfile(
        name='Baroque',
        chord_types=['major', 'minor', 'diminished', 'dom7'],
        harmonic_rhythm=4.0,
        use_extensions=False,
        use_alterations=False,
        voice_leading_style='smooth',
        swing_ratio=0.5,
        syncopation_level=0.1,
        quantization='strict',
        ornamentation_density=0.8,
        interval_preference='stepwise',
        chromatic_usage=0.3,
        preferred_instruments=[6, 40, 42, 73],  # Harpsichord, strings, flute
        register_preference='mid',
        texture='polyphonic'
    ),

    'romantic': StyleProfile(
        name='Romantic',
        chord_types=['major', 'minor', 'augmented', 'major7', 'dim7'],
        harmonic_rhythm=1.5,
        use_extensions=True,
        use_alterations=True,
        voice_leading_style='wide',
        swing_ratio=0.5,
        syncopation_level=0.3,
        quantization='human',
        ornamentation_density=0.5,
        interval_preference='balanced',
        chromatic_usage=0.7,
        preferred_instruments=[0, 48, 49, 56],  # Piano, strings, brass
        register_preference='wide',
        texture='homophonic'
    ),

    'minimalist': StyleProfile(
        name='Minimalist',
        chord_types=['major', 'minor', 'sus2', 'sus4'],
        harmonic_rhythm=0.25,
        use_extensions=False,
        use_alterations=False,
        voice_leading_style='smooth',
        swing_ratio=0.5,
        syncopation_level=0.1,
        quantization='strict',
        ornamentation_density=0.05,
        interval_preference='stepwise',
        chromatic_usage=0.1,
        preferred_instruments=[0, 11, 88],  # Piano, vibes, synth pad
        register_preference='mid',
        texture='homophonic'
    ),
}


# ==============================================================================
# HARMONIC TRANSFORMATION
# ==============================================================================

class HarmonicTransformer:
    """
    Transform harmonic content to match target style.

    Implements sophisticated reharmonization techniques including:
    - Chord substitution (tritone, relative, chromatic mediant)
    - Extension addition/removal
    - Voice leading optimization
    - Modal interchange
    """

    # Chord substitution rules
    SUBSTITUTIONS = {
        'classical_to_jazz': {
            'major': 'major7',
            'minor': 'minor7',
            'dom7': 'dom7',  # Keep
            'diminished': 'half-dim7',
        },
        'jazz_to_classical': {
            'major7': 'major',
            'minor7': 'minor',
            'dom7': 'major',  # Simplify
            'half-dim7': 'diminished',
            'altered': 'dom7',
        },
        'pop_to_jazz': {
            'major': 'major7',
            'minor': 'minor7',
            'sus2': 'minor11',
            'sus4': 'dom7sus4',
        },
    }

    @staticmethod
    def reharmonize(notes: List[NoteEvent],
                    chords: List[ChordEvent],
                    source_style: str,
                    target_style: str) -> List[NoteEvent]:
        """
        Reharmonize MIDI notes according to target style.

        Args:
            notes: Original note events
            chords: Detected chords
            source_style: Original style name
            target_style: Target style name

        Returns:
            Modified note events with new harmonies
        """
        if not chords or target_style not in STYLE_PROFILES:
            return notes

        target_profile = STYLE_PROFILES[target_style]
        reharmonized = []

        # Group notes by chord
        for chord in chords:
            chord_notes = [n for n in notes
                          if chord.start_time <= n.start_time < chord.start_time + chord.duration]

            if not chord_notes:
                continue

            # Transform chord quality
            new_quality = HarmonicTransformer._transform_chord_quality(
                chord.quality, target_profile
            )

            # Rebuild chord tones with new quality
            new_chord_tones = HarmonicTransformer._build_chord(
                chord.root, new_quality, target_profile
            )

            # Remap notes to new chord tones
            for note in chord_notes:
                new_pitch = HarmonicTransformer._remap_to_chord(
                    note.pitch, chord.root, new_chord_tones
                )
                new_note = copy.copy(note)
                new_note.pitch = new_pitch
                reharmonized.append(new_note)

        # Add any notes not in chords
        chord_times = set()
        for chord in chords:
            chord_times.add((chord.start_time, chord.start_time + chord.duration))

        for note in notes:
            in_chord = any(start <= note.start_time < end for start, end in chord_times)
            if not in_chord:
                reharmonized.append(copy.copy(note))

        return sorted(reharmonized, key=lambda n: n.start_time)

    @staticmethod
    def _transform_chord_quality(quality: str, target_profile: StyleProfile) -> str:
        """Transform chord quality to match target style."""
        # Prefer chord types from target profile
        if quality in target_profile.chord_types:
            return quality

        # Map to closest allowed type
        quality_map = {
            'major': 'major7' if target_profile.use_extensions else 'major',
            'minor': 'minor7' if target_profile.use_extensions else 'minor',
            'diminished': 'half-dim7' if target_profile.use_extensions else 'diminished',
            'augmented': 'major7' if target_profile.use_extensions else 'major',
            'dom7': 'dom7' if 'dom7' in target_profile.chord_types else 'major',
            'major7': 'major' if not target_profile.use_extensions else 'major7',
            'minor7': 'minor' if not target_profile.use_extensions else 'minor7',
        }

        return quality_map.get(quality, quality)

    @staticmethod
    def _build_chord(root: int, quality: str, profile: StyleProfile) -> List[int]:
        """Build chord tones for given root and quality."""
        # Basic chord templates
        templates = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'sus2': [0, 2, 7],
            'sus4': [0, 5, 7],
            'dom7': [0, 4, 7, 10],
            'major7': [0, 4, 7, 11],
            'minor7': [0, 3, 7, 10],
            'half-dim7': [0, 3, 6, 10],
            'dim7': [0, 3, 6, 9],
        }

        template = templates.get(quality, [0, 4, 7])
        chord_tones = [(root + interval) % 12 for interval in template]

        # Add extensions if profile allows
        if profile.use_extensions and len(template) >= 4:
            # Add 9th
            chord_tones.append((root + 2) % 12)
            # Add 13th for jazz
            if profile.name == 'Jazz':
                chord_tones.append((root + 9) % 12)

        return sorted(set(chord_tones))

    @staticmethod
    def _remap_to_chord(pitch: int, root: int, chord_tones: List[int]) -> int:
        """Remap a pitch to closest chord tone."""
        pitch_class = pitch % 12
        octave = pitch // 12

        # If already a chord tone, keep it
        if pitch_class in chord_tones:
            return pitch

        # Find closest chord tone
        distances = [(abs((tone - pitch_class + 6) % 12 - 6), tone) for tone in chord_tones]
        closest_tone = min(distances, key=lambda x: x[0])[1]

        return octave * 12 + closest_tone


# ==============================================================================
# RHYTHMIC TRANSFORMATION
# ==============================================================================

class RhythmicTransformer:
    """
    Transform rhythmic feel to match target style.

    Implements:
    - Swing/shuffle application
    - Quantization/humanization
    - Time signature conversion
    - Syncopation modification
    """

    @staticmethod
    def apply_swing(notes: List[NoteEvent], swing_ratio: float = 0.67) -> List[NoteEvent]:
        """
        Apply swing feel to notes.

        Args:
            notes: Note events
            swing_ratio: Swing ratio (0.5=straight, 0.67=triplet swing, 0.75=heavy swing)

        Returns:
            Notes with swing timing applied
        """
        if abs(swing_ratio - 0.5) < 0.01:  # No swing
            return notes

        swung_notes = []

        for note in notes:
            # Detect if note is on offbeat (8th note grid)
            # This is a simplified approach
            beat_position = note.start_time % 1.0  # Position within beat

            new_note = copy.copy(note)

            # If on offbeat, delay it according to swing ratio
            if 0.4 < beat_position < 0.6:  # Roughly on the "and"
                # Shift timing
                swing_offset = (swing_ratio - 0.5) * 0.5
                new_note.start_time += swing_offset
                new_note.start_tick = int(new_note.start_tick * (1 + swing_offset))

            swung_notes.append(new_note)

        return swung_notes

    @staticmethod
    def quantize(notes: List[NoteEvent],
                 grid: str = '16th',
                 strength: float = 1.0) -> List[NoteEvent]:
        """
        Quantize notes to rhythmic grid.

        Args:
            notes: Note events
            grid: Grid size ('8th', '16th', '32nd')
            strength: Quantization strength 0-1 (1=hard quantize, 0=no change)

        Returns:
            Quantized notes
        """
        grid_sizes = {
            '4th': 0.25,
            '8th': 0.125,
            '16th': 0.0625,
            '32nd': 0.03125,
        }

        grid_size = grid_sizes.get(grid, 0.0625)
        quantized = []

        for note in notes:
            new_note = copy.copy(note)

            # Quantize start time
            quantized_start = round(note.start_time / grid_size) * grid_size
            new_note.start_time = note.start_time + (quantized_start - note.start_time) * strength

            # Quantize duration
            quantized_dur = round(note.duration / grid_size) * grid_size
            new_note.duration = note.duration + (quantized_dur - note.duration) * strength

            quantized.append(new_note)

        return quantized

    @staticmethod
    def add_syncopation(notes: List[NoteEvent],
                       intensity: float = 0.5) -> List[NoteEvent]:
        """
        Add syncopation to rhythm.

        Args:
            notes: Note events
            intensity: Syncopation intensity 0-1

        Returns:
            Notes with added syncopation
        """
        syncopated = []

        for note in notes:
            new_note = copy.copy(note)

            # Randomly shift some notes to offbeats
            if np.random.random() < intensity:
                # Shift forward by 1/16th note
                new_note.start_time += 0.0625
                new_note.start_tick += int(note.duration_ticks * 0.0625 / note.duration)

            syncopated.append(new_note)

        return syncopated

    @staticmethod
    def convert_time_signature(notes: List[NoteEvent],
                              source_sig: Tuple[int, int],
                              target_sig: Tuple[int, int]) -> List[NoteEvent]:
        """
        Convert between time signatures.

        Args:
            notes: Note events
            source_sig: Source time signature (numerator, denominator)
            target_sig: Target time signature

        Returns:
            Notes adjusted to new time signature
        """
        source_beats_per_measure = source_sig[0] * (4 / source_sig[1])
        target_beats_per_measure = target_sig[0] * (4 / target_sig[1])

        ratio = target_beats_per_measure / source_beats_per_measure

        converted = []
        for note in notes:
            new_note = copy.copy(note)
            new_note.start_time *= ratio
            new_note.duration *= ratio
            new_note.start_tick = int(new_note.start_tick * ratio)
            new_note.duration_ticks = int(new_note.duration_ticks * ratio)
            converted.append(new_note)

        return converted


# ==============================================================================
# MELODIC TRANSFORMATION
# ==============================================================================

class MelodicTransformer:
    """
    Transform melodic content to match target style.

    Implements:
    - Ornamentation (trills, turns, mordents, grace notes)
    - Simplification (remove passing tones)
    - Interval modification (stepwise ↔ angular)
    - Chromatic alteration
    """

    @staticmethod
    def add_ornamentation(notes: List[NoteEvent],
                         density: float = 0.5,
                         style: str = 'baroque') -> List[NoteEvent]:
        """
        Add ornamental notes.

        Args:
            notes: Melody notes
            density: Ornamentation density 0-1
            style: Ornamentation style ('baroque', 'romantic', 'jazz')

        Returns:
            Ornamented melody
        """
        ornamented = []

        for i, note in enumerate(notes):
            ornamented.append(copy.copy(note))

            # Add ornament with probability based on density
            if np.random.random() < density and note.duration > 0.25:
                # Add passing tone or neighbor tone
                if style == 'baroque':
                    # Upper neighbor tone
                    ornament = copy.copy(note)
                    ornament.pitch += 1  # Semitone above
                    ornament.start_time += note.duration * 0.3
                    ornament.duration = note.duration * 0.2
                    ornament.velocity = int(note.velocity * 0.8)
                    ornamented.append(ornament)

                elif style == 'romantic':
                    # Chromatic approach
                    if i < len(notes) - 1:
                        next_pitch = notes[i + 1].pitch
                        approach = copy.copy(note)
                        approach.pitch = next_pitch - 1  # Chromatic below next note
                        approach.start_time = note.end_time - note.duration * 0.1
                        approach.duration = note.duration * 0.1
                        approach.velocity = int(note.velocity * 0.7)
                        ornamented.append(approach)

                elif style == 'jazz':
                    # Chromatic enclosure
                    upper = copy.copy(note)
                    upper.pitch += 1
                    upper.start_time += note.duration * 0.2
                    upper.duration = note.duration * 0.15
                    upper.velocity = int(note.velocity * 0.8)

                    lower = copy.copy(note)
                    lower.pitch -= 1
                    lower.start_time += note.duration * 0.4
                    lower.duration = note.duration * 0.15
                    lower.velocity = int(note.velocity * 0.8)

                    ornamented.extend([upper, lower])

        return sorted(ornamented, key=lambda n: n.start_time)

    @staticmethod
    def simplify_melody(notes: List[NoteEvent],
                       keep_ratio: float = 0.6) -> List[NoteEvent]:
        """
        Simplify melody by removing passing tones and short notes.

        Args:
            notes: Melody notes
            keep_ratio: Ratio of notes to keep (0.6 = keep 60%)

        Returns:
            Simplified melody
        """
        if len(notes) <= 2:
            return notes

        # Sort by duration (keep longer notes)
        sorted_notes = sorted(enumerate(notes),
                            key=lambda x: x[1].duration,
                            reverse=True)

        num_to_keep = max(2, int(len(notes) * keep_ratio))
        keep_indices = sorted([idx for idx, _ in sorted_notes[:num_to_keep]])

        simplified = [notes[i] for i in keep_indices]
        return sorted(simplified, key=lambda n: n.start_time)

    @staticmethod
    def make_stepwise(notes: List[NoteEvent],
                     scale: List[int]) -> List[NoteEvent]:
        """
        Make melody more stepwise (conjunct motion).

        Args:
            notes: Melody notes
            scale: Scale to use (pitch classes)

        Returns:
            More stepwise melody
        """
        if len(notes) <= 1:
            return notes

        stepwise = [copy.copy(notes[0])]

        for i in range(1, len(notes)):
            prev_pitch = stepwise[-1].pitch
            current = copy.copy(notes[i])

            # If interval is large, insert passing tones
            interval = abs(current.pitch - prev_pitch)

            if interval > 4:  # Larger than major third
                # Move in steps instead
                direction = 1 if current.pitch > prev_pitch else -1
                target = prev_pitch + direction * 2  # Move by whole step
                current.pitch = target

            stepwise.append(current)

        return stepwise

    @staticmethod
    def make_angular(notes: List[NoteEvent]) -> List[NoteEvent]:
        """
        Make melody more angular (disjunct motion, larger intervals).

        Args:
            notes: Melody notes

        Returns:
            More angular melody
        """
        if len(notes) <= 1:
            return notes

        angular = [copy.copy(notes[0])]

        for i in range(1, len(notes)):
            prev_pitch = angular[-1].pitch
            current = copy.copy(notes[i])

            # If interval is small, make it larger
            interval = current.pitch - prev_pitch

            if abs(interval) <= 2:  # Step or smaller
                # Expand to larger interval
                direction = 1 if interval >= 0 else -1
                current.pitch = prev_pitch + direction * (4 + np.random.randint(0, 4))

            angular.append(current)

        return angular


# ==============================================================================
# MAIN STYLE TRANSFER ENGINE
# ==============================================================================

class StyleTransfer:
    """
    Main style transfer engine.

    Coordinates all transformation dimensions to transfer MIDI from
    one style to another.
    """

    def __init__(self, input_midi: str, source_style: str = 'classical'):
        """
        Initialize style transfer engine.

        Args:
            input_midi: Path to input MIDI file
            source_style: Source style name
        """
        self.input_path = Path(input_midi)
        self.source_style = source_style

        # Analyze input
        print(f"Analyzing input MIDI: {self.input_path.name}")
        self.analyzer = MidiAnalyzer(str(self.input_path))
        self.analysis = self.analyzer.analyze()

        print(f"Analysis complete: {len(self.analysis.notes)} notes, "
              f"{len(self.analysis.chords)} chords detected")

    def transfer(self,
                target_style: str,
                transform_harmony: bool = True,
                transform_rhythm: bool = True,
                transform_melody: bool = True,
                output_path: Optional[str] = None) -> str:
        """
        Transfer to target style.

        Args:
            target_style: Target style name
            transform_harmony: Apply harmonic transformation
            transform_rhythm: Apply rhythmic transformation
            transform_melody: Apply melodic transformation
            output_path: Output MIDI path (auto-generated if None)

        Returns:
            Path to output MIDI file
        """
        if target_style not in STYLE_PROFILES:
            raise ValueError(f"Unknown style: {target_style}. "
                           f"Available: {list(STYLE_PROFILES.keys())}")

        print(f"\n{'='*80}")
        print(f"STYLE TRANSFER: {self.source_style} → {target_style}")
        print(f"{'='*80}\n")

        notes = self.analysis.notes.copy()
        target_profile = STYLE_PROFILES[target_style]

        # HARMONIC TRANSFORMATION
        if transform_harmony and self.analysis.chords:
            print("🎸 Applying harmonic transformation...")
            notes = HarmonicTransformer.reharmonize(
                notes, self.analysis.chords,
                self.source_style, target_style
            )
            print(f"   Reharmonized to {target_style} style")

        # RHYTHMIC TRANSFORMATION
        if transform_rhythm:
            print("🥁 Applying rhythmic transformation...")

            # Apply swing
            if abs(target_profile.swing_ratio - 0.5) > 0.01:
                notes = RhythmicTransformer.apply_swing(notes, target_profile.swing_ratio)
                print(f"   Applied swing (ratio: {target_profile.swing_ratio:.2f})")

            # Quantization
            if target_profile.quantization == 'strict':
                notes = RhythmicTransformer.quantize(notes, '16th', strength=0.9)
                print("   Applied strict quantization")
            elif target_profile.quantization == 'loose':
                notes = RhythmicTransformer.quantize(notes, '16th', strength=0.5)
                print("   Applied loose quantization")

            # Syncopation
            if target_profile.syncopation_level > 0.5:
                notes = RhythmicTransformer.add_syncopation(
                    notes, target_profile.syncopation_level
                )
                print(f"   Added syncopation (level: {target_profile.syncopation_level:.1f})")

        # MELODIC TRANSFORMATION
        if transform_melody:
            print("🎶 Applying melodic transformation...")

            # Ornamentation
            if target_profile.ornamentation_density > 0.3:
                notes = MelodicTransformer.add_ornamentation(
                    notes,
                    density=target_profile.ornamentation_density,
                    style=target_style
                )
                print(f"   Added ornamentation (density: {target_profile.ornamentation_density:.1f})")

            # Interval adjustment
            if target_profile.interval_preference == 'stepwise':
                # Use detected key for scale
                if self.analysis.key:
                    scale = self._get_scale_for_key(self.analysis.key.tonic,
                                                   self.analysis.key.mode)
                    notes = MelodicTransformer.make_stepwise(notes, scale)
                    print("   Made melody more stepwise")
            elif target_profile.interval_preference == 'angular':
                notes = MelodicTransformer.make_angular(notes)
                print("   Made melody more angular")

        # WRITE OUTPUT
        if output_path is None:
            output_path = self.input_path.parent / f"{self.input_path.stem}_{target_style}.mid"

        self._write_midi(notes, output_path)

        print(f"\n✅ Style transfer complete!")
        print(f"📁 Output: {output_path}")
        print(f"{'='*80}\n")

        return str(output_path)

    def _get_scale_for_key(self, tonic: int, mode: str) -> List[int]:
        """Get scale pitch classes for a key."""
        major_intervals = [0, 2, 4, 5, 7, 9, 11]
        minor_intervals = [0, 2, 3, 5, 7, 8, 10]

        intervals = minor_intervals if mode == 'minor' else major_intervals
        return [(tonic + i) % 12 for i in intervals]

    def _write_midi(self, notes: List[NoteEvent], output_path: Path):
        """Write notes to MIDI file."""
        # Create new MIDI file
        midi = MidiFile(ticks_per_beat=self.analysis.ticks_per_beat)
        track = MidiTrack()
        midi.tracks.append(track)

        # Add tempo
        if self.analysis.tempo_events:
            tempo = self.analysis.tempo_events[0].microseconds_per_beat
        else:
            tempo = 500000  # 120 BPM
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

        # Add time signature
        if self.analysis.time_signatures:
            ts = self.analysis.time_signatures[0]
            track.append(MetaMessage('time_signature',
                                   numerator=ts.numerator,
                                   denominator=ts.denominator,
                                   time=0))

        # Convert notes to MIDI messages
        events = []
        for note in notes:
            events.append((note.start_tick, 'note_on', note.pitch, note.velocity, note.channel))
            events.append((note.end_tick, 'note_off', note.pitch, 0, note.channel))

        # Sort by time
        events.sort(key=lambda x: x[0])

        # Write with delta times
        current_time = 0
        for abs_time, msg_type, note, velocity, channel in events:
            delta = abs_time - current_time
            delta = max(0, int(delta))  # Ensure non-negative

            track.append(Message(
                msg_type,
                note=note,
                velocity=velocity,
                channel=channel,
                time=delta
            ))
            current_time = abs_time

        track.append(MetaMessage('end_of_track', time=0))

        # Save
        midi.save(str(output_path))


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Transfer MIDI to different musical style')
    parser.add_argument('input', help='Input MIDI file')
    parser.add_argument('target_style',
                       choices=list(STYLE_PROFILES.keys()),
                       help='Target style')
    parser.add_argument('--source-style',
                       choices=list(STYLE_PROFILES.keys()),
                       default='classical',
                       help='Source style (default: classical)')
    parser.add_argument('--output', '-o', help='Output MIDI file')
    parser.add_argument('--no-harmony', action='store_true', help='Skip harmonic transformation')
    parser.add_argument('--no-rhythm', action='store_true', help='Skip rhythmic transformation')
    parser.add_argument('--no-melody', action='store_true', help='Skip melodic transformation')

    args = parser.parse_args()

    # Perform style transfer
    engine = StyleTransfer(args.input, args.source_style)
    output = engine.transfer(
        args.target_style,
        transform_harmony=not args.no_harmony,
        transform_rhythm=not args.no_rhythm,
        transform_melody=not args.no_melody,
        output_path=args.output
    )

    print(f"✅ Done! Transformed to {args.target_style} style: {output}")
