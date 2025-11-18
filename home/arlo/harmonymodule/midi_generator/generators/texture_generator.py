#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Texture and Accompaniment Pattern Generator

Generates various musical textures and accompaniment patterns including:
- Alberti bass and broken chord patterns
- Arpeggiated patterns (up, down, alternating, circular)
- Block chords and rhythmic variations
- Ostinato patterns
- Countermelodies
- Walking bass lines
- Stride piano
- Various homophonic and polyphonic textures

Research References:
- Classical accompaniment patterns (Mozart, Haydn)
- Romantic piano textures (Chopin, Liszt)
- Jazz comping patterns
- Pop/rock accompaniment styles

Author: Claude (Sonnet 4.5)
Created: 2025
"""

from typing import List, Tuple, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import random


class TextureType(Enum):
    """Types of musical texture"""
    MONOPHONIC = "monophonic"  # Single melodic line
    HOMOPHONIC = "homophonic"  # Melody + accompaniment
    POLYPHONIC = "polyphonic"  # Multiple independent lines
    HETEROPHONIC = "heterophonic"  # Variations of same melody


class AccompanimentPattern(Enum):
    """Accompaniment pattern types"""
    BLOCK_CHORDS = "block_chords"
    ALBERTI_BASS = "alberti_bass"
    BROKEN_CHORDS = "broken_chords"
    ARPEGGIATED = "arpeggiated"
    WALTZ = "waltz"
    STRIDE = "stride"
    OSTINATO = "ostinato"
    PEDAL_POINT = "pedal_point"
    COUNTERMELODY = "countermelody"
    WALKING_BASS = "walking_bass"
    REPEATED_CHORDS = "repeated_chords"


@dataclass
class TexturePattern:
    """A generated texture pattern"""
    notes: List[int]  # MIDI note numbers
    durations: List[float]  # Duration in beats
    start_times: List[float]  # Start time in beats
    velocities: List[int]  # MIDI velocities
    pattern_name: str = ""


class TextureGenerator:
    """
    Generate musical textures and accompaniment patterns.

    Can create various accompaniment styles from classical to modern,
    with rhythmic variations and voice leading considerations.
    """

    def __init__(self, beats_per_bar: int = 4, subdivision: int = 4):
        """
        Initialize texture generator.

        Args:
            beats_per_bar: Number of beats per measure
            subdivision: Subdivision per beat (4 = sixteenth notes)
        """
        self.beats_per_bar = beats_per_bar
        self.subdivision = subdivision
        self.beat_duration = 1.0  # Quarter note = 1 beat

    # ========================================================================
    # BROKEN CHORD PATTERNS
    # ========================================================================

    def generate_alberti_bass(
        self,
        chord: List[int],
        num_bars: int = 1,
        velocity: int = 70
    ) -> TexturePattern:
        """
        Generate Alberti bass pattern (low-high-middle-high).

        Classic accompaniment pattern used by Mozart and other Classical composers.

        Args:
            chord: Chord notes (MIDI), typically 3-4 notes
            num_bars: Number of bars to generate
            velocity: MIDI velocity

        Returns:
            TexturePattern with Alberti bass
        """
        if len(chord) < 3:
            raise ValueError("Alberti bass requires at least 3 notes")

        sorted_chord = sorted(chord)
        pattern = []
        durations = []
        start_times = []
        velocities = []

        # Alberti pattern: lowest, highest, middle, highest
        # For 3 notes: 0, 2, 1, 2
        # For 4+ notes: 0, 3, 1, 3 (or 0, 3, 2, 3)

        if len(sorted_chord) == 3:
            alberti_indices = [0, 2, 1, 2]
        else:
            alberti_indices = [0, len(sorted_chord)-1, 1, len(sorted_chord)-1]

        note_duration = self.beat_duration / 4  # Sixteenth notes
        current_time = 0.0

        for bar in range(num_bars):
            for beat in range(self.beats_per_bar):
                for idx in alberti_indices:
                    pattern.append(sorted_chord[idx % len(sorted_chord)])
                    durations.append(note_duration)
                    start_times.append(current_time)
                    velocities.append(velocity)
                    current_time += note_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities,
            pattern_name="Alberti Bass"
        )

    def generate_broken_chords(
        self,
        chord: List[int],
        num_bars: int = 1,
        pattern_type: str = "up",
        velocity: int = 70
    ) -> TexturePattern:
        """
        Generate broken chord patterns.

        Args:
            chord: Chord notes (MIDI)
            num_bars: Number of bars
            pattern_type: "up", "down", "up-down", "random"
            velocity: MIDI velocity

        Returns:
            TexturePattern with broken chords
        """
        sorted_chord = sorted(chord)
        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        note_duration = self.beat_duration / 4  # Sixteenth notes
        current_time = 0.0

        for bar in range(num_bars):
            for beat in range(self.beats_per_bar):
                # Generate pattern for one beat (4 sixteenths)
                beat_notes = self._get_broken_chord_beat(sorted_chord, pattern_type)

                for note in beat_notes:
                    pattern.append(note)
                    durations.append(note_duration)
                    start_times.append(current_time)
                    velocities_list.append(velocity)
                    current_time += note_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name=f"Broken Chords ({pattern_type})"
        )

    def _get_broken_chord_beat(
        self,
        chord: List[int],
        pattern_type: str
    ) -> List[int]:
        """Get notes for one beat of broken chord pattern"""
        if pattern_type == "up":
            return [chord[i % len(chord)] for i in range(4)]
        elif pattern_type == "down":
            return [chord[-(i % len(chord)) - 1] for i in range(4)]
        elif pattern_type == "up-down":
            if random.random() > 0.5:
                return [chord[i % len(chord)] for i in range(4)]
            else:
                return [chord[-(i % len(chord)) - 1] for i in range(4)]
        elif pattern_type == "random":
            return [random.choice(chord) for _ in range(4)]
        else:
            return [chord[i % len(chord)] for i in range(4)]

    def generate_arpeggiated(
        self,
        chord: List[int],
        num_bars: int = 1,
        notes_per_bar: int = 8,
        direction: str = "up",
        velocity: int = 75
    ) -> TexturePattern:
        """
        Generate arpeggiated pattern.

        Args:
            chord: Chord notes (MIDI)
            num_bars: Number of bars
            notes_per_bar: Number of notes per bar
            direction: "up", "down", "alternate", "circular"
            velocity: MIDI velocity

        Returns:
            TexturePattern with arpeggios
        """
        sorted_chord = sorted(chord)
        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        note_duration = self.beats_per_bar / notes_per_bar
        current_time = 0.0
        ascending = True

        for bar in range(num_bars):
            for i in range(notes_per_bar):
                # Select note based on direction
                if direction == "up":
                    note = sorted_chord[i % len(sorted_chord)]
                    if i % len(sorted_chord) == 0 and i > 0:
                        note += 12  # Octave up
                elif direction == "down":
                    idx = len(sorted_chord) - 1 - (i % len(sorted_chord))
                    note = sorted_chord[idx]
                    if i % len(sorted_chord) == 0 and i > 0:
                        note -= 12  # Octave down
                elif direction == "alternate":
                    if ascending:
                        note = sorted_chord[i % len(sorted_chord)]
                        if i % len(sorted_chord) == len(sorted_chord) - 1:
                            ascending = False
                    else:
                        idx = len(sorted_chord) - 1 - (i % len(sorted_chord))
                        note = sorted_chord[idx]
                        if i % len(sorted_chord) == len(sorted_chord) - 1:
                            ascending = True
                else:  # circular
                    note = sorted_chord[i % len(sorted_chord)]

                pattern.append(note)
                durations.append(note_duration)
                start_times.append(current_time)
                velocities_list.append(velocity)
                current_time += note_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name=f"Arpeggio ({direction})"
        )

    # ========================================================================
    # BLOCK CHORD PATTERNS
    # ========================================================================

    def generate_block_chords(
        self,
        chords: List[List[int]],
        rhythm: Optional[List[float]] = None,
        velocity: int = 80
    ) -> TexturePattern:
        """
        Generate block chord pattern.

        Args:
            chords: List of chords (each chord is list of MIDI notes)
            rhythm: Optional rhythm pattern (durations in beats)
            velocity: MIDI velocity

        Returns:
            TexturePattern with block chords
        """
        if rhythm is None:
            # Default: one chord per beat
            rhythm = [1.0] * len(chords)

        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        current_time = 0.0

        for chord, duration in zip(chords, rhythm):
            # Add all notes of chord at same time
            for note in chord:
                pattern.append(note)
                durations.append(duration)
                start_times.append(current_time)
                velocities_list.append(velocity)

            current_time += duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name="Block Chords"
        )

    def generate_repeated_chords(
        self,
        chord: List[int],
        num_bars: int = 1,
        repeats_per_beat: int = 2,
        velocity: int = 75
    ) -> TexturePattern:
        """
        Generate repeated chord pattern (common in pop/rock).

        Args:
            chord: Chord notes (MIDI)
            num_bars: Number of bars
            repeats_per_beat: Number of chord repetitions per beat
            velocity: MIDI velocity

        Returns:
            TexturePattern with repeated chords
        """
        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        note_duration = self.beat_duration / repeats_per_beat
        current_time = 0.0

        for bar in range(num_bars):
            for beat in range(self.beats_per_bar):
                for rep in range(repeats_per_beat):
                    # Add whole chord
                    for note in chord:
                        pattern.append(note)
                        durations.append(note_duration * 0.8)  # Slightly staccato
                        start_times.append(current_time)
                        # Accent on beat
                        vel = velocity + 10 if rep == 0 else velocity
                        velocities_list.append(vel)

                    current_time += note_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name="Repeated Chords"
        )

    # ========================================================================
    # PIANO PATTERNS
    # ========================================================================

    def generate_waltz(
        self,
        chord: List[int],
        bass_note: int,
        num_bars: int = 1,
        velocity: int = 75
    ) -> TexturePattern:
        """
        Generate waltz pattern (bass on 1, chords on 2-3).

        Classic 3/4 time accompaniment.

        Args:
            chord: Chord notes (MIDI)
            bass_note: Bass note (MIDI)
            num_bars: Number of bars
            velocity: MIDI velocity

        Returns:
            TexturePattern with waltz pattern
        """
        if self.beats_per_bar != 3:
            print("Warning: Waltz pattern typically uses 3/4 time")

        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        current_time = 0.0

        for bar in range(num_bars):
            # Beat 1: Bass note
            pattern.append(bass_note)
            durations.append(self.beat_duration)
            start_times.append(current_time)
            velocities_list.append(velocity + 10)  # Accent
            current_time += self.beat_duration

            # Beats 2-3: Chord
            for beat in range(2):
                for note in chord:
                    pattern.append(note)
                    durations.append(self.beat_duration)
                    start_times.append(current_time)
                    velocities_list.append(velocity)

                current_time += self.beat_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name="Waltz"
        )

    def generate_stride_piano(
        self,
        chord: List[int],
        bass_note: int,
        num_bars: int = 1,
        velocity: int = 80
    ) -> TexturePattern:
        """
        Generate stride piano pattern (alternating bass and chords).

        Jazz/ragtime piano style.

        Args:
            chord: Chord notes (MIDI)
            bass_note: Bass note (MIDI)
            num_bars: Number of bars
            velocity: MIDI velocity

        Returns:
            TexturePattern with stride pattern
        """
        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        current_time = 0.0
        beat_duration = self.beat_duration / 2  # Eighth notes

        for bar in range(num_bars):
            for beat in range(self.beats_per_bar):
                # Downbeat: Bass note
                pattern.append(bass_note)
                durations.append(beat_duration)
                start_times.append(current_time)
                velocities_list.append(velocity + 10)
                current_time += beat_duration

                # Upbeat: Chord
                for note in chord:
                    pattern.append(note)
                    durations.append(beat_duration)
                    start_times.append(current_time)
                    velocities_list.append(velocity)

                current_time += beat_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name="Stride Piano"
        )

    # ========================================================================
    # BASS PATTERNS
    # ========================================================================

    def generate_walking_bass(
        self,
        root: int,
        scale: List[int],
        num_bars: int = 1,
        velocity: int = 75
    ) -> TexturePattern:
        """
        Generate walking bass line (stepwise motion, one note per beat).

        Common in jazz and swing.

        Args:
            root: Root note (MIDI)
            scale: Scale notes (intervals from root)
            num_bars: Number of bars
            velocity: MIDI velocity

        Returns:
            TexturePattern with walking bass
        """
        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        # Build scale starting from root
        bass_notes = [root + interval for interval in scale]
        # Extend scale to cover more range
        bass_notes.extend([root + 12 + interval for interval in scale])

        current_time = 0.0
        current_note_idx = 0

        for bar in range(num_bars):
            for beat in range(self.beats_per_bar):
                # Select next note (mostly stepwise)
                if beat == 0:
                    # Start of bar: use root or fifth
                    note = root if random.random() > 0.2 else root + 7
                else:
                    # Stepwise motion
                    step = random.choice([-1, 1, 1, 2])  # More up than down
                    current_note_idx = (current_note_idx + step) % len(bass_notes)
                    note = bass_notes[current_note_idx]

                pattern.append(note)
                durations.append(self.beat_duration * 0.9)  # Slight separation
                start_times.append(current_time)
                # Accent on beat 1 and 3
                vel = velocity + 10 if beat % 2 == 0 else velocity
                velocities_list.append(vel)
                current_time += self.beat_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name="Walking Bass"
        )

    # ========================================================================
    # SPECIAL PATTERNS
    # ========================================================================

    def generate_ostinato(
        self,
        notes: List[int],
        num_repetitions: int = 8,
        note_duration: float = 0.5,
        velocity: int = 70
    ) -> TexturePattern:
        """
        Generate ostinato pattern (repeated melodic/rhythmic pattern).

        Args:
            notes: Ostinato notes (MIDI)
            num_repetitions: Number of times to repeat
            note_duration: Duration of each note (beats)
            velocity: MIDI velocity

        Returns:
            TexturePattern with ostinato
        """
        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        current_time = 0.0

        for rep in range(num_repetitions):
            for note in notes:
                pattern.append(note)
                durations.append(note_duration)
                start_times.append(current_time)
                # Add subtle dynamic variation
                vel = velocity + random.randint(-5, 5)
                velocities_list.append(vel)
                current_time += note_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name="Ostinato"
        )

    def generate_pedal_point(
        self,
        pedal_note: int,
        duration_bars: int = 4,
        velocity: int = 60
    ) -> TexturePattern:
        """
        Generate pedal point (sustained or repeated note).

        Args:
            pedal_note: Pedal note (MIDI)
            duration_bars: Duration in bars
            velocity: MIDI velocity

        Returns:
            TexturePattern with pedal point
        """
        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        current_time = 0.0
        total_duration = self.beats_per_bar * duration_bars

        # Option 1: Long sustained note
        if random.random() > 0.5:
            pattern.append(pedal_note)
            durations.append(total_duration)
            start_times.append(0.0)
            velocities_list.append(velocity)
        else:
            # Option 2: Repeated notes
            num_notes = int(total_duration / 0.5)  # Eighth notes
            for i in range(num_notes):
                pattern.append(pedal_note)
                durations.append(0.5)
                start_times.append(current_time)
                velocities_list.append(velocity)
                current_time += 0.5

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name="Pedal Point"
        )

    def generate_countermelody(
        self,
        main_melody: List[int],
        harmony: List[int],
        velocity: int = 70
    ) -> TexturePattern:
        """
        Generate a countermelody to complement main melody.

        Uses contrary motion and fills rests/long notes.

        Args:
            main_melody: Main melody notes (MIDI)
            harmony: Available harmony notes (chord tones)
            velocity: MIDI velocity

        Returns:
            TexturePattern with countermelody
        """
        pattern = []
        durations = []
        start_times = []
        velocities_list = []

        current_time = 0.0
        note_duration = self.beat_duration

        for i, melody_note in enumerate(main_melody):
            # Generate counter-note
            # Prefer contrary motion
            if i > 0:
                if melody_note > main_melody[i-1]:
                    # Melody going up, counter goes down
                    counter_note = min([h for h in harmony if h < melody_note],
                                     default=harmony[0])
                else:
                    # Melody going down, counter goes up
                    counter_note = max([h for h in harmony if h > melody_note],
                                     default=harmony[-1])
            else:
                counter_note = random.choice(harmony)

            pattern.append(counter_note)
            durations.append(note_duration)
            start_times.append(current_time)
            velocities_list.append(velocity)
            current_time += note_duration

        return TexturePattern(
            notes=pattern,
            durations=durations,
            start_times=start_times,
            velocities=velocities_list,
            pattern_name="Countermelody"
        )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def combine_patterns(
        self,
        patterns: List[TexturePattern]
    ) -> TexturePattern:
        """
        Combine multiple texture patterns.

        Args:
            patterns: List of TexturePatterns to combine

        Returns:
            Combined TexturePattern
        """
        all_notes = []
        all_durations = []
        all_start_times = []
        all_velocities = []

        for pattern in patterns:
            all_notes.extend(pattern.notes)
            all_durations.extend(pattern.durations)
            all_start_times.extend(pattern.start_times)
            all_velocities.extend(pattern.velocities)

        return TexturePattern(
            notes=all_notes,
            durations=all_durations,
            start_times=all_start_times,
            velocities=all_velocities,
            pattern_name="Combined"
        )

    def transpose_pattern(
        self,
        pattern: TexturePattern,
        semitones: int
    ) -> TexturePattern:
        """
        Transpose a pattern by semitones.

        Args:
            pattern: TexturePattern to transpose
            semitones: Number of semitones (positive = up, negative = down)

        Returns:
            Transposed TexturePattern
        """
        transposed_notes = [note + semitones for note in pattern.notes]

        return TexturePattern(
            notes=transposed_notes,
            durations=pattern.durations.copy(),
            start_times=pattern.start_times.copy(),
            velocities=pattern.velocities.copy(),
            pattern_name=f"{pattern.pattern_name} (transposed)"
        )


# ============================================================================
# MAIN (EXAMPLES/TESTS)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TEXTURE GENERATOR - EXAMPLES")
    print("=" * 80)

    generator = TextureGenerator(beats_per_bar=4, subdivision=4)

    # Example chord: C major
    c_major = [60, 64, 67]  # C, E, G
    bass_c = 48

    print("\n1. Alberti Bass Pattern:")
    alberti = generator.generate_alberti_bass(c_major, num_bars=2)
    print(f"   Generated {len(alberti.notes)} notes")
    print(f"   First 8 notes: {alberti.notes[:8]}")

    print("\n2. Arpeggiated Pattern (up):")
    arp = generator.generate_arpeggiated(c_major, num_bars=1, notes_per_bar=8)
    print(f"   Generated {len(arp.notes)} notes")
    print(f"   Notes: {arp.notes}")

    print("\n3. Block Chords:")
    chords = [[60, 64, 67], [62, 65, 69], [64, 67, 71]]  # C, Dm, Em
    block = generator.generate_block_chords(chords)
    print(f"   Generated {len(block.notes)} notes across {len(chords)} chords")

    print("\n4. Waltz Pattern:")
    generator_3_4 = TextureGenerator(beats_per_bar=3, subdivision=4)
    waltz = generator_3_4.generate_waltz(c_major, bass_c, num_bars=2)
    print(f"   Generated {len(waltz.notes)} notes")

    print("\n5. Stride Piano:")
    stride = generator.generate_stride_piano(c_major, bass_c, num_bars=2)
    print(f"   Generated {len(stride.notes)} notes")

    print("\n6. Walking Bass:")
    c_major_scale = [0, 2, 4, 5, 7, 9, 11]  # Intervals
    walking = generator.generate_walking_bass(48, c_major_scale, num_bars=2)
    print(f"   Generated {len(walking.notes)} notes")
    print(f"   First 8 notes: {walking.notes[:8]}")

    print("\n7. Ostinato Pattern:")
    ostinato_notes = [60, 62, 64, 62]  # Simple 4-note pattern
    ostinato = generator.generate_ostinato(ostinato_notes, num_repetitions=4)
    print(f"   Generated {len(ostinato.notes)} notes (4 repetitions)")

    print("\n8. Pedal Point:")
    pedal = generator.generate_pedal_point(48, duration_bars=4)
    print(f"   Generated {len(pedal.notes)} notes")

    print("\n9. Broken Chords (up-down):")
    broken = generator.generate_broken_chords(c_major, num_bars=1, pattern_type="up-down")
    print(f"   Generated {len(broken.notes)} notes")

    print("\n10. Repeated Chords:")
    repeated = generator.generate_repeated_chords(c_major, num_bars=1, repeats_per_beat=2)
    print(f"   Generated {len(repeated.notes)} notes")

    print("\n" + "=" * 80)
    print("Texture generator ready!")
    print("=" * 80)
