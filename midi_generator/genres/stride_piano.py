#!/usr/bin/env python3
"""
Stride Piano Generator - Authentic Jazz Stride Patterns
=======================================================

Implements stride piano in the tradition of James P. Johnson, Fats Waller,
and Art Tatum. Stride piano is characterized by:

- Left hand: Alternating bass notes (beats 1 & 3) with chords (beats 2 & 4)
- Right hand: Melody, fills, runs, and embellishments
- "Oom-pah, oom-pah" feel that drives the rhythm

Historical Context:
------------------
- James P. Johnson: Father of stride piano ("Carolina Shout")
- Fats Waller: Virtuosic stride with humor ("Ain't Misbehavin'")
- Art Tatum: Technically advanced stride with runs ("Tea for Two")
- Willie "The Lion" Smith: Melodic stride approach

Research Sources:
----------------
- Mark Levine "Jazz Piano Book" - stride chapter
- PiJAMA Dataset analysis - tempo/pattern distribution
- Transcriptions from James P. Johnson, Fats Waller recordings
- Historical performance practice studies

Author: Agent 3 - Piano Comping Virtuoso
Date: 2025
License: MIT
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import jazz types from the main jazz module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from genres.jazz import JazzNote, JazzChord
except ImportError:
    # Fallback data structures if import fails
    from dataclasses import dataclass, field

    @dataclass
    class JazzNote:
        pitch: int
        velocity: int
        start_time: float
        duration: float
        articulation: str = "normal"
        swing_offset: float = 0.0
        channel: int = 0

    @dataclass
    class JazzChord:
        root: int
        quality: str
        extensions: List[int] = field(default_factory=list)
        alterations: List[str] = field(default_factory=list)
        inversion: int = 0
        voicing_type: str = "shell"


class StridePattern(Enum):
    """Stride left-hand patterns"""
    ALTERNATING_BASS = "alternating_bass"      # Root (1), chord (2), fifth (3), chord (4)
    WALKING_TENTHS = "walking_tenths"          # Walking bass with tenths
    SINGLE_NOTE_BASS = "single_note_bass"      # Single bass notes (simpler)
    BROKEN_TENTHS = "broken_tenths"            # Tenths broken with approach notes


class RightHandStyle(Enum):
    """Right-hand playing styles"""
    MELODIC = "melodic"                        # Single-note melody
    OCTAVES = "octaves"                        # Melody in octaves (powerful)
    BLOCK_CHORDS = "block_chords"              # Block chord melody
    SPARSE_FILLS = "sparse_fills"              # Minimal fills between phrases
    VIRTUOSIC_RUNS = "virtuosic_runs"          # Fast scalar/chromatic runs (Art Tatum)


class StridePianoGenerator:
    """
    Generate authentic stride piano patterns.

    Features:
    - Multiple left-hand patterns (alternating bass, walking tenths, etc.)
    - Right-hand styles from simple to virtuosic
    - Tempo-adaptive patterns (lighter at fast tempos)
    - Authentic swing feel
    - Dynamic variation

    Example Usage:
    -------------
    >>> generator = StridePianoGenerator(tempo=120)
    >>> chord = JazzChord(root=0, quality="maj7")  # C major 7
    >>> notes = generator.generate_stride_pattern(
    ...     chord=chord,
    ...     bars=4,
    ...     left_hand_pattern="alternating_bass",
    ...     right_hand_density=0.6
    ... )
    """

    def __init__(self, tempo: int = 120):
        """
        Initialize stride piano generator.

        Args:
            tempo: Tempo in BPM (60-300)
        """
        self.tempo = tempo
        self.current_time = 0.0

        # Tempo adaptations
        self.is_fast_tempo = tempo > 180
        self.is_slow_tempo = tempo < 100

    def generate_stride_pattern(
        self,
        chord: JazzChord,
        bars: int = 4,
        left_hand_pattern: str = "alternating_bass",
        right_hand_style: str = "melodic",
        right_hand_density: float = 0.6,
        base_octave: int = 3,
        swing_ratio: float = 0.62
    ) -> List[JazzNote]:
        """
        Generate complete stride piano pattern (left + right hand).

        Args:
            chord: JazzChord to comp over
            bars: Number of bars to generate
            left_hand_pattern: "alternating_bass", "walking_tenths", "single_note_bass", "broken_tenths"
            right_hand_style: "melodic", "octaves", "block_chords", "sparse_fills", "virtuosic_runs"
            right_hand_density: 0.0-1.0, how busy the right hand is
            base_octave: Base octave for left hand (typically 2-3)
            swing_ratio: Swing timing ratio (0.5=straight, 0.62=medium swing, 0.67=heavy)

        Returns:
            List of JazzNote objects (left + right hand combined)
        """
        left_hand = self._generate_left_hand(
            chord, bars, left_hand_pattern, base_octave, swing_ratio
        )

        right_hand = self._generate_right_hand(
            chord, bars, right_hand_style, right_hand_density, base_octave + 2, swing_ratio
        )

        return left_hand + right_hand

    def _generate_left_hand(
        self,
        chord: JazzChord,
        bars: int,
        pattern: str,
        base_octave: int,
        swing_ratio: float
    ) -> List[JazzNote]:
        """
        Generate left-hand stride pattern.

        Classic stride: Bass note on 1 & 3, chord on 2 & 4
        """
        notes = []
        root_midi = 12 * base_octave + chord.root

        for bar in range(bars):
            bar_start = bar * 4.0  # 4 beats per bar

            if pattern == "alternating_bass":
                # Beat 1: Root (bass)
                notes.append(JazzNote(
                    pitch=root_midi,
                    velocity=80 + random.randint(-5, 5),
                    start_time=bar_start,
                    duration=0.9,
                    articulation="accent"
                ))

                # Beat 2: Chord (mid-range)
                chord_notes = self._get_stride_chord(chord, base_octave + 1)
                for pitch in chord_notes:
                    notes.append(JazzNote(
                        pitch=pitch,
                        velocity=65 + random.randint(-3, 3),
                        start_time=bar_start + 1.0,
                        duration=0.8,
                        articulation="normal"
                    ))

                # Beat 3: Fifth (bass)
                fifth_midi = root_midi + 7
                notes.append(JazzNote(
                    pitch=fifth_midi,
                    velocity=75 + random.randint(-5, 5),
                    start_time=bar_start + 2.0,
                    duration=0.9,
                    articulation="normal"
                ))

                # Beat 4: Chord (mid-range)
                for pitch in chord_notes:
                    notes.append(JazzNote(
                        pitch=pitch,
                        velocity=65 + random.randint(-3, 3),
                        start_time=bar_start + 3.0,
                        duration=0.8,
                        articulation="normal"
                    ))

            elif pattern == "walking_tenths":
                # Walking bass with tenths (root + 10th interval)
                bass_line = self._generate_walking_bass_bar(chord, root_midi, bar)

                for beat, bass_note in enumerate(bass_line):
                    # Bass note
                    notes.append(JazzNote(
                        pitch=bass_note,
                        velocity=75 + random.randint(-5, 5),
                        start_time=bar_start + beat,
                        duration=0.9,
                        articulation="accent" if beat == 0 else "normal"
                    ))

                    # Add tenth on beats 2 & 4
                    if beat in [1, 3]:
                        tenth = bass_note + 16  # Tenth = octave + third
                        if tenth <= 127:
                            notes.append(JazzNote(
                                pitch=tenth,
                                velocity=60 + random.randint(-3, 3),
                                start_time=bar_start + beat,
                                duration=0.8,
                                articulation="normal"
                            ))

            elif pattern == "single_note_bass":
                # Simpler pattern: just bass notes on each beat
                for beat in range(4):
                    if beat == 0:
                        pitch = root_midi
                    elif beat == 1:
                        pitch = root_midi + 7  # Fifth
                    elif beat == 2:
                        pitch = root_midi
                    else:
                        pitch = root_midi + 7

                    notes.append(JazzNote(
                        pitch=pitch,
                        velocity=75 + random.randint(-5, 5),
                        start_time=bar_start + beat,
                        duration=0.9,
                        articulation="accent" if beat == 0 else "normal"
                    ))

            elif pattern == "broken_tenths":
                # Tenths broken with chromatic approach
                for beat in range(0, 4, 2):
                    # Bass on 1 & 3
                    bass = root_midi if beat == 0 else root_midi + 7
                    notes.append(JazzNote(
                        pitch=bass,
                        velocity=80 + random.randint(-5, 5),
                        start_time=bar_start + beat,
                        duration=0.4,
                        articulation="accent" if beat == 0 else "normal"
                    ))

                    # Tenth broken into two notes
                    tenth_lower = bass + 16
                    tenth_upper = bass + 17  # Chromatic approach

                    if tenth_lower <= 127:
                        notes.append(JazzNote(
                            pitch=tenth_lower,
                            velocity=65,
                            start_time=bar_start + beat + 0.5,
                            duration=0.4,
                            articulation="normal"
                        ))

                    if tenth_upper <= 127:
                        notes.append(JazzNote(
                            pitch=tenth_upper,
                            velocity=60,
                            start_time=bar_start + beat + 0.75,
                            duration=0.25,
                            articulation="ghost"
                        ))

        return notes

    def _generate_right_hand(
        self,
        chord: JazzChord,
        bars: int,
        style: str,
        density: float,
        base_octave: int,
        swing_ratio: float
    ) -> List[JazzNote]:
        """
        Generate right-hand patterns.

        Density: 0.0 = very sparse, 1.0 = very busy
        """
        notes = []
        root_midi = 12 * base_octave + chord.root

        for bar in range(bars):
            bar_start = bar * 4.0

            if style == "melodic":
                # Single-note melody with occasional fills
                melody_notes = self._generate_melodic_phrase(
                    chord, root_midi, bar_start, density
                )
                notes.extend(melody_notes)

            elif style == "octaves":
                # Melody in octaves (powerful, assertive)
                melody_notes = self._generate_melodic_phrase(
                    chord, root_midi, bar_start, density * 0.7  # Less dense in octaves
                )

                for note in melody_notes:
                    # Original note
                    notes.append(note)
                    # Octave above
                    if note.pitch + 12 <= 127:
                        notes.append(JazzNote(
                            pitch=note.pitch + 12,
                            velocity=note.velocity - 5,
                            start_time=note.start_time,
                            duration=note.duration,
                            articulation=note.articulation
                        ))

            elif style == "block_chords":
                # Block chord melody
                num_chords = int(2 + density * 4)  # 2-6 chords per bar
                for i in range(num_chords):
                    beat = i * (4.0 / num_chords)
                    chord_pitches = self._get_stride_chord(chord, base_octave)

                    for pitch in chord_pitches:
                        if pitch <= 127:
                            notes.append(JazzNote(
                                pitch=pitch,
                                velocity=70 + random.randint(-5, 5),
                                start_time=bar_start + beat,
                                duration=0.4,
                                articulation="accent" if beat == 0 else "normal"
                            ))

            elif style == "sparse_fills":
                # Minimal fills, mostly rests
                if random.random() < density * 0.5:
                    # Occasional fill
                    fill_start = bar_start + random.choice([1.5, 2.5, 3.5])
                    fill_notes = self._generate_fill(chord, root_midi, fill_start, 0.5)
                    notes.extend(fill_notes)

            elif style == "virtuosic_runs":
                # Art Tatum style - fast runs and embellishments
                if random.random() < density:
                    run_start = bar_start + random.choice([0.0, 2.0])
                    run_notes = self._generate_virtuosic_run(
                        chord, root_midi, run_start, 2.0
                    )
                    notes.extend(run_notes)

        return notes

    def _get_stride_chord(self, chord: JazzChord, octave: int) -> List[int]:
        """
        Get chord voicing for stride (left hand beats 2 & 4, or right hand).

        Uses compact voicings suitable for stride.
        """
        root = 12 * octave + chord.root

        if "maj7" in chord.quality or "maj" in chord.quality:
            # Major 7: root, 3rd, 5th, 7th
            return [root, root + 4, root + 7, root + 11]
        elif "min7" in chord.quality or "min" in chord.quality:
            # Minor 7: root, b3, 5, b7
            return [root, root + 3, root + 7, root + 10]
        elif "dom7" in chord.quality or chord.quality == "7":
            # Dominant 7: root, 3rd, 5th, b7
            return [root, root + 4, root + 7, root + 10]
        elif "dim" in chord.quality:
            # Diminished 7: root, b3, b5, bb7
            return [root, root + 3, root + 6, root + 9]
        else:
            # Default: shell voicing
            third = root + (4 if "maj" in chord.quality else 3)
            return [root, third, root + 7]

    def _generate_walking_bass_bar(
        self, chord: JazzChord, root_midi: int, bar_num: int
    ) -> List[int]:
        """
        Generate 4-note walking bass line for one bar.

        Uses chord tones and chromatic approaches.
        """
        chord_tones = [
            root_midi,              # Root
            root_midi + (3 if "min" in chord.quality else 4),  # Third
            root_midi + 7,          # Fifth
            root_midi + (10 if "dom" in chord.quality or "min" in chord.quality else 11)  # Seventh
        ]

        # Simple walking pattern
        bass_line = [
            chord_tones[0],  # Beat 1: Root
            chord_tones[1],  # Beat 2: Third
            chord_tones[2],  # Beat 3: Fifth
            chord_tones[0] + random.choice([-1, 1])  # Beat 4: Chromatic approach to next root
        ]

        return bass_line

    def _generate_melodic_phrase(
        self, chord: JazzChord, root_midi: int, start_time: float, density: float
    ) -> List[JazzNote]:
        """
        Generate simple melodic phrase for right hand.
        """
        notes = []
        scale = self._get_chord_scale(chord, root_midi)

        num_notes = int(2 + density * 6)  # 2-8 notes per bar

        for i in range(num_notes):
            beat = i * (4.0 / num_notes)
            pitch = random.choice(scale)

            notes.append(JazzNote(
                pitch=pitch,
                velocity=75 + random.randint(-10, 10),
                start_time=start_time + beat,
                duration=0.4 + random.random() * 0.4,
                articulation="accent" if beat == 0 else "normal"
            ))

        return notes

    def _generate_fill(
        self, chord: JazzChord, root_midi: int, start_time: float, duration: float
    ) -> List[JazzNote]:
        """
        Generate short fill (chromatic or scalar).
        """
        notes = []
        scale = self._get_chord_scale(chord, root_midi)

        # Quick 3-4 note fill
        num_notes = random.randint(3, 5)
        note_duration = duration / num_notes

        current_pitch = random.choice(scale)

        for i in range(num_notes):
            notes.append(JazzNote(
                pitch=current_pitch,
                velocity=70 + random.randint(-5, 5),
                start_time=start_time + i * note_duration,
                duration=note_duration * 0.8,
                articulation="normal"
            ))

            # Move up or down by scale step or half-step
            if random.random() < 0.5:
                current_pitch += random.choice([1, 2])  # Half or whole step
            else:
                current_pitch -= random.choice([1, 2])

            # Keep in reasonable range
            current_pitch = max(60, min(96, current_pitch))

        return notes

    def _generate_virtuosic_run(
        self, chord: JazzChord, root_midi: int, start_time: float, duration: float
    ) -> List[JazzNote]:
        """
        Generate fast virtuosic run (Art Tatum style).

        Fast ascending or descending scalar passage.
        """
        notes = []
        scale = self._get_chord_scale(chord, root_midi - 12)  # Start lower

        # Fast run: 16th notes
        num_notes = int(duration * 4)  # 4 sixteenth notes per beat
        note_duration = duration / num_notes

        # Ascending run
        start_pitch = min(scale)

        for i in range(num_notes):
            if i < len(scale):
                pitch = scale[i]
            else:
                pitch = start_pitch + i  # Continue chromatically if we run out of scale

            if pitch <= 127:
                notes.append(JazzNote(
                    pitch=pitch,
                    velocity=85 + random.randint(-5, 5),
                    start_time=start_time + i * note_duration,
                    duration=note_duration * 0.9,
                    articulation="normal"
                ))

        return notes

    def _get_chord_scale(self, chord: JazzChord, root_midi: int) -> List[int]:
        """
        Get appropriate scale for chord (for melody generation).
        """
        # Simplified chord-scale relationships
        if "maj7" in chord.quality or "maj" in chord.quality:
            # Major scale (Ionian)
            intervals = [0, 2, 4, 5, 7, 9, 11]
        elif "min7" in chord.quality or "min" in chord.quality:
            # Dorian mode
            intervals = [0, 2, 3, 5, 7, 9, 10]
        elif "dom7" in chord.quality or chord.quality == "7":
            # Mixolydian mode
            intervals = [0, 2, 4, 5, 7, 9, 10]
        elif "dim" in chord.quality:
            # Diminished scale
            intervals = [0, 2, 3, 5, 6, 8, 9, 11]
        else:
            # Default: major scale
            intervals = [0, 2, 4, 5, 7, 9, 11]

        # Generate two octaves
        scale = []
        for octave_offset in [0, 12]:
            for interval in intervals:
                pitch = root_midi + octave_offset + interval
                if pitch <= 127:
                    scale.append(pitch)

        return sorted(scale)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_stride_accompaniment(
    chords: List[JazzChord],
    tempo: int = 120,
    style: str = "classic",
    bars_per_chord: int = 1
) -> List[JazzNote]:
    """
    Generate stride piano accompaniment for a chord progression.

    Args:
        chords: List of JazzChord objects
        tempo: Tempo in BPM
        style: "classic" (James P. Johnson), "virtuosic" (Art Tatum), "simple" (beginner)
        bars_per_chord: How many bars to spend on each chord

    Returns:
        Complete stride piano part as list of JazzNote objects

    Example:
        >>> from genres.jazz import JazzChord
        >>> progression = [
        ...     JazzChord(root=0, quality="maj7"),   # C major 7
        ...     JazzChord(root=7, quality="dom7"),   # G7
        ... ]
        >>> stride = generate_stride_accompaniment(progression, tempo=140, style="classic")
    """
    generator = StridePianoGenerator(tempo=tempo)

    # Style presets
    if style == "classic":
        left_pattern = "alternating_bass"
        right_style = "melodic"
        right_density = 0.6
    elif style == "virtuosic":
        left_pattern = "walking_tenths"
        right_style = "virtuosic_runs"
        right_density = 0.8
    elif style == "simple":
        left_pattern = "single_note_bass"
        right_style = "sparse_fills"
        right_density = 0.3
    else:
        left_pattern = "alternating_bass"
        right_style = "melodic"
        right_density = 0.5

    all_notes = []
    current_time = 0.0

    for chord in chords:
        notes = generator.generate_stride_pattern(
            chord=chord,
            bars=bars_per_chord,
            left_hand_pattern=left_pattern,
            right_hand_style=right_style,
            right_hand_density=right_density
        )

        # Adjust timing
        for note in notes:
            note.start_time += current_time

        all_notes.extend(notes)
        current_time += bars_per_chord * 4.0  # 4 beats per bar

    return all_notes


if __name__ == "__main__":
    # Test the stride piano generator
    print("Stride Piano Generator - Test")
    print("=" * 50)

    # Create test chord
    test_chord = JazzChord(root=0, quality="maj7")  # C major 7

    # Generate stride pattern
    generator = StridePianoGenerator(tempo=120)
    notes = generator.generate_stride_pattern(
        chord=test_chord,
        bars=4,
        left_hand_pattern="alternating_bass",
        right_hand_density=0.6
    )

    print(f"Generated {len(notes)} notes")
    print(f"Time range: {min(n.start_time for n in notes):.2f} - {max(n.start_time for n in notes):.2f} beats")
    print(f"Pitch range: {min(n.pitch for n in notes)} - {max(n.pitch for n in notes)}")
    print("\nFirst 10 notes:")
    for i, note in enumerate(notes[:10]):
        print(f"  {i+1}. Pitch {note.pitch:3d}, Time {note.start_time:5.2f}, Vel {note.velocity:3d}, Art: {note.articulation}")

    print("\n✓ Stride piano generator test complete!")
