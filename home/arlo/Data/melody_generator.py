#!/usr/bin/env python3
"""
Advanced Melody Generator

Creates sophisticated melodies based on key, chord progressions, and musical parameters.
Supports:
- Chord scales and extensions
- Chromatic approach notes
- Passing tones and neighbor tones
- Rhythmic and melodic motifs
- Contour control (direction, range)
- Syncopation and articulation
- Multiple creativity levels
"""

import argparse
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

# Import from melody harmonizer for shared functionality
import sys
sys.path.insert(0, '/home/arlo/Data')
from melody_harmonizer_improved import ChordAnalyzer, ScaleContext, CHORD_LIBRARY


class MelodyGenerator:
    """Advanced melody generation with musical intelligence"""

    # Scale intervals (semitones from root)
    SCALES = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'locrian': [0, 1, 3, 5, 6, 8, 10],
    }

    # Rhythmic patterns (in 480 ticks per quarter note)
    RHYTHMIC_PATTERNS = {
        'simple': [
            [480, 480, 480, 480],  # Quarter notes
            [960, 480, 480],  # Half + two quarters
            [480, 960, 480],  # Quarter + half + quarter
        ],
        'moderate': [
            [240, 240, 480, 480, 480],  # Eighth notes + quarters
            [480, 240, 240, 480, 480],
            [360, 120, 480, 480, 480],  # Dotted eighth + sixteenth
        ],
        'complex': [
            [240, 240, 240, 240, 240, 240, 240, 240],  # All eighths
            [160, 160, 160, 480, 480, 480],  # Triplets + quarters
            [360, 120, 240, 240, 480, 480],  # Mixed rhythms
            [120, 120, 120, 120, 480, 480, 480],  # Sixteenths
        ],
        'syncopated': [
            [240, 480, 240, 480, 480],  # Syncopation
            [480, 240, 480, 240, 480],
            [360, 120, 360, 120, 480, 480],
        ]
    }

    def __init__(
        self,
        key: str = "C major",
        tempo: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        seed: Optional[int] = None
    ):
        """
        Initialize melody generator

        Args:
            key: Musical key (e.g., "C major", "A minor")
            tempo: BPM
            time_signature: Tuple of (numerator, denominator)
            seed: Random seed for reproducibility
        """
        self.parse_key(key)
        self.tempo = tempo
        self.time_signature = time_signature
        self.ticks_per_beat = 480
        self.bar_ticks = self.ticks_per_beat * time_signature[0]

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Melodic state
        self.motifs = []  # Store melodic motifs for repetition
        self.last_direction = 0  # Track melodic direction
        self.phrase_position = 0  # Position within phrase

    def parse_key(self, key: str):
        """Parse key string into root and scale type"""
        parts = key.strip().split()
        if len(parts) == 2:
            root_str, scale_type = parts
        else:
            root_str = parts[0] if parts else "C"
            scale_type = "major"

        # Parse root note
        note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        root_note = note_map.get(root_str[0].upper(), 0)

        # Handle accidentals
        if len(root_str) > 1:
            if 'b' in root_str or '♭' in root_str:
                root_note -= 1
            elif '#' in root_str or '♯' in root_str:
                root_note += 1

        root_note = root_note % 12

        self.key_root = root_note
        self.scale_type = scale_type.lower()
        self.scale_context = ScaleContext(root_note, scale_type)

        # Get scale notes
        scale_intervals = self.SCALES.get(self.scale_type, self.SCALES['major'])
        self.scale_notes = [(self.key_root + interval) % 12 for interval in scale_intervals]

    def get_chord_scale(self, chord_name: str) -> List[int]:
        """
        Get appropriate scale for a chord

        Returns pitch classes (0-11) that work over the chord
        """
        chord_info = ChordAnalyzer.parse_chord(chord_name, use_extensions=True, scale_context=self.scale_context)

        # Combine chord tones and extensions
        available_notes = set()
        for note in chord_info['chord_tones']:
            available_notes.add(note % 12)
        for note in chord_info.get('extensions', []):
            available_notes.add(note % 12)

        # Add scale tones
        for note in self.scale_notes:
            available_notes.add(note)

        return sorted(list(available_notes))

    def get_chord_tones(self, chord_name: str) -> Set[int]:
        """Get chord tones as pitch classes"""
        chord_info = ChordAnalyzer.parse_chord(chord_name, use_extensions=False, scale_context=self.scale_context)
        return set(note % 12 for note in chord_info['chord_tones'])

    def is_chord_tone(self, pitch: int, chord_name: str) -> bool:
        """Check if pitch is a chord tone"""
        return (pitch % 12) in self.get_chord_tones(chord_name)

    def select_note(
        self,
        chord_name: str,
        current_pitch: Optional[int],
        target_direction: float,
        creativity: float,
        chord_tone_weight: float,
        min_pitch: int,
        max_pitch: int,
        previous_notes: List[int],
        bar_position: float,
        is_phrase_end: bool = False
    ) -> int:
        """
        Select next melodic note using advanced logic

        Args:
            chord_name: Current chord
            current_pitch: Previous note (None if first)
            target_direction: Desired direction (-1.0 to 1.0, 0 = neutral)
            creativity: Chromatic/tension level (0.0 to 1.0)
            chord_tone_weight: Preference for chord tones (0.0 to 1.0)
            min_pitch: Minimum MIDI note
            max_pitch: Maximum MIDI note
            previous_notes: Recent note history
            bar_position: Position within bar (0.0 to 1.0)
            is_phrase_end: Whether this is end of phrase

        Returns:
            MIDI note number
        """
        chord_scale = self.get_chord_scale(chord_name)
        chord_tones = self.get_chord_tones(chord_name)

        # Determine if this is a strong beat (emphasize chord tones)
        is_strong_beat = bar_position in [0.0, 0.5]

        # At phrase endings, strongly prefer chord tones (especially root)
        if is_phrase_end:
            chord_tone_weight = min(1.0, chord_tone_weight + 0.3)

        # Build candidate notes
        candidates = []

        if current_pitch is None:
            # First note - prefer chord tones in middle range
            for octave in range(min_pitch // 12, (max_pitch // 12) + 1):
                for pc in chord_scale:
                    note = octave * 12 + pc
                    if min_pitch <= note <= max_pitch:
                        # Weight by how close to middle of range
                        mid_point = (min_pitch + max_pitch) / 2
                        distance_weight = 1.0 - abs(note - mid_point) / (max_pitch - min_pitch)

                        # Prefer chord tones
                        ct_weight = 2.0 if pc in chord_tones else 1.0

                        candidates.append((note, distance_weight * ct_weight))
        else:
            # Subsequent notes

            # Determine interval constraints based on creativity
            if creativity < 0.3:
                max_interval = 5  # Conservative (perfect fourth)
            elif creativity < 0.7:
                max_interval = 7  # Moderate (perfect fifth)
            else:
                max_interval = 12  # Bold (octave)

            # Generate candidates within interval range
            for interval in range(-max_interval, max_interval + 1):
                if interval == 0:
                    continue  # No repeated notes (usually)

                candidate = current_pitch + interval

                # Check range
                if candidate < min_pitch or candidate > max_pitch:
                    continue

                pc = candidate % 12

                # Base weight
                weight = 1.0

                # Penalize large leaps
                leap_penalty = abs(interval) / max_interval
                weight *= (1.0 - leap_penalty * 0.5)

                # Direction preference
                if target_direction > 0 and interval > 0:
                    weight *= (1.0 + target_direction)
                elif target_direction < 0 and interval < 0:
                    weight *= (1.0 + abs(target_direction))

                # Chord tone preference
                if pc in chord_tones:
                    weight *= (1.0 + chord_tone_weight * 2.0)
                    # Extra weight on strong beats
                    if is_strong_beat:
                        weight *= 1.5
                elif pc in self.scale_notes:
                    weight *= (1.0 + chord_tone_weight * 0.5)
                else:
                    # Chromatic note
                    if creativity < 0.5:
                        continue  # Skip chromatic notes at low creativity
                    weight *= creativity  # Scale by creativity

                # Avoid immediate repetition of recent notes
                if len(previous_notes) > 0 and candidate == previous_notes[-1]:
                    weight *= 0.1
                if len(previous_notes) > 1 and candidate in previous_notes[-3:]:
                    weight *= 0.5

                # Approach tones (chromatic or diatonic approach to chord tones)
                if creativity > 0.4:
                    # Check if this is an approach tone
                    target_note = candidate + 1 if interval > 0 else candidate - 1
                    if (target_note % 12) in chord_tones:
                        # This approaches a chord tone
                        weight *= (1.0 + creativity)

                candidates.append((candidate, weight))

        # Handle no candidates (shouldn't happen, but safety)
        if not candidates:
            return current_pitch if current_pitch else 60

        # Weighted random selection
        notes, weights = zip(*candidates)
        weights = np.array(weights)
        weights = weights / weights.sum()

        selected_note = np.random.choice(notes, p=weights)

        return int(selected_note)

    def add_passing_tone(
        self,
        note1: int,
        note2: int,
        chord_name: str,
        creativity: float
    ) -> Optional[int]:
        """
        Add a passing tone between two notes

        Returns the passing tone, or None if not applicable
        """
        if abs(note2 - note1) <= 2:
            return None  # Too close for passing tone

        # Determine if we should add passing tone
        if creativity < 0.3:
            if random.random() > 0.2:
                return None
        elif creativity < 0.7:
            if random.random() > 0.5:
                return None
        else:
            if random.random() > 0.8:
                return None

        # Calculate passing tone
        if note2 > note1:
            # Ascending
            passing = note1 + 1 if creativity > 0.6 else note1 + 2
        else:
            # Descending
            passing = note1 - 1 if creativity > 0.6 else note1 - 2

        # Check if it makes sense (between the two notes)
        if note2 > note1 and passing >= note2:
            return None
        if note2 < note1 and passing <= note2:
            return None

        return passing

    def add_neighbor_tone(
        self,
        note: int,
        chord_name: str,
        creativity: float
    ) -> Optional[int]:
        """
        Add a neighbor tone (upper or lower)

        Returns the neighbor tone, or None if not applicable
        """
        if creativity < 0.5:
            return None

        if random.random() > creativity:
            return None

        # Upper or lower neighbor
        if random.random() < 0.5:
            neighbor = note + 1  # Upper (chromatic)
        else:
            neighbor = note - 1  # Lower (chromatic)

        return neighbor

    def create_motif(
        self,
        chord_name: str,
        length: int,
        min_pitch: int,
        max_pitch: int
    ) -> List[Tuple[int, int]]:
        """
        Create a melodic motif (sequence of intervals)

        Returns list of (interval, duration) tuples
        """
        motif = []
        for i in range(length):
            # Mostly stepwise motion in motifs
            interval = random.choice([-2, -1, 0, 1, 2, 3, 4, 5])
            duration = random.choice([240, 360, 480])
            motif.append((interval, duration))

        return motif

    def apply_motif(
        self,
        motif: List[Tuple[int, int]],
        start_pitch: int,
        min_pitch: int,
        max_pitch: int,
        transposition: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Apply a motif starting from a given pitch

        Returns list of (pitch, duration) tuples
        """
        result = []
        current_pitch = start_pitch + transposition

        for interval, duration in motif:
            current_pitch += interval
            # Clamp to range
            current_pitch = max(min_pitch, min(max_pitch, current_pitch))
            result.append((current_pitch, duration))

        return result

    def generate_rhythm_pattern(
        self,
        bars: int,
        rhythmic_density: str = 'moderate',
        syncopation: float = 0.3
    ) -> List[int]:
        """
        Generate rhythmic pattern for the melody

        Args:
            bars: Number of bars
            rhythmic_density: 'simple', 'moderate', 'complex', 'syncopated'
            syncopation: Amount of syncopation (0.0 to 1.0)

        Returns:
            List of durations in ticks
        """
        total_ticks = bars * self.bar_ticks
        pattern = []
        current_tick = 0

        # Select pattern pool based on density
        if syncopation > 0.6:
            pattern_pool = self.RHYTHMIC_PATTERNS['syncopated'] + self.RHYTHMIC_PATTERNS.get(rhythmic_density, self.RHYTHMIC_PATTERNS['moderate'])
        else:
            pattern_pool = self.RHYTHMIC_PATTERNS.get(rhythmic_density, self.RHYTHMIC_PATTERNS['moderate'])

        while current_tick < total_ticks:
            # Select random pattern
            bar_pattern = random.choice(pattern_pool).copy()

            # Ensure it fits in remaining space
            pattern_sum = sum(bar_pattern)
            remaining = total_ticks - current_tick

            if pattern_sum > remaining:
                # Truncate or adjust
                scale = remaining / pattern_sum
                bar_pattern = [int(d * scale) for d in bar_pattern]

            pattern.extend(bar_pattern)
            current_tick += sum(bar_pattern)

        # Ensure exact length
        total = sum(pattern)
        if total != total_ticks:
            pattern[-1] += (total_ticks - total)

        return pattern

    def generate_melody(
        self,
        chord_progression: Dict[int, str],
        num_bars: int,
        min_pitch: int = 60,
        max_pitch: int = 84,
        creativity: float = 0.5,
        chord_tone_weight: float = 0.7,
        rhythmic_density: str = 'moderate',
        syncopation: float = 0.3,
        direction_bias: float = 0.0,
        use_motifs: bool = True,
        motif_repetition: float = 0.4
    ) -> List[Tuple[int, int, int]]:
        """
        Generate a complete melody

        Args:
            chord_progression: Dict mapping tick positions to chord names
            num_bars: Number of bars to generate
            min_pitch: Minimum MIDI note
            max_pitch: Maximum MIDI note
            creativity: 0.0 (conservative) to 1.0 (adventurous)
            chord_tone_weight: Preference for chord tones (0.0 to 1.0)
            rhythmic_density: 'simple', 'moderate', 'complex', 'syncopated'
            syncopation: Amount of syncopation (0.0 to 1.0)
            direction_bias: Melodic direction (-1.0 descending to 1.0 ascending)
            use_motifs: Whether to create and reuse melodic motifs
            motif_repetition: How often to repeat motifs (0.0 to 1.0)

        Returns:
            List of (tick, duration, pitch) tuples
        """
        print(f"\n{'='*80}")
        print(f"🎵 MELODY GENERATOR")
        print(f"{'='*80}")
        print(f"Key: {self.scale_context}")
        print(f"Bars: {num_bars}")
        print(f"Range: MIDI {min_pitch}-{max_pitch}")
        print(f"Creativity: {creativity:.2f}")
        print(f"Chord tone weight: {chord_tone_weight:.2f}")
        print(f"Rhythmic density: {rhythmic_density}")
        print(f"Syncopation: {syncopation:.2f}")
        print(f"Direction bias: {direction_bias:+.2f}")
        print(f"Use motifs: {use_motifs}")

        # Generate rhythm
        rhythm_pattern = self.generate_rhythm_pattern(num_bars, rhythmic_density, syncopation)
        print(f"\n🥁 Generated rhythm: {len(rhythm_pattern)} notes")

        # Sort chord progression
        chord_changes = sorted(chord_progression.items())

        melody = []
        current_tick = 0
        current_pitch = None
        previous_notes = []
        current_chord = chord_changes[0][1] if chord_changes else "C"
        chord_idx = 0

        # Track phrase structure (every 4 bars is a phrase)
        phrase_length = 4 * self.bar_ticks

        for note_idx, duration in enumerate(rhythm_pattern):
            # Update current chord
            while chord_idx < len(chord_changes) - 1 and current_tick >= chord_changes[chord_idx + 1][0]:
                chord_idx += 1
                current_chord = chord_changes[chord_idx][1]

            # Calculate position within bar and phrase
            bar_position = (current_tick % self.bar_ticks) / self.bar_ticks
            phrase_position = (current_tick % phrase_length) / phrase_length
            is_phrase_end = (current_tick + duration) % phrase_length < duration

            # Determine target direction
            # Alternate between ascending and descending to create contour
            if current_pitch is None:
                target_direction = direction_bias
            else:
                # Add some wave-like motion
                wave = np.sin(phrase_position * np.pi * 2) * 0.5
                target_direction = direction_bias + wave

                # Range correction - if approaching limits, bias away
                if current_pitch >= max_pitch - 5:
                    target_direction = -0.8
                elif current_pitch <= min_pitch + 5:
                    target_direction = 0.8

            # Check if we should use a motif
            use_existing_motif = False
            if use_motifs and len(self.motifs) > 0 and random.random() < motif_repetition:
                use_existing_motif = True

            if use_existing_motif and current_pitch is not None:
                # Apply existing motif
                motif = random.choice(self.motifs)
                transposition = random.choice([0, 2, 4, 5, 7, -2, -4, -5, -7])  # Diatonic transpositions
                motif_notes = self.apply_motif(motif, current_pitch, min_pitch, max_pitch, transposition)

                # Add motif notes to melody
                for pitch, motif_dur in motif_notes:
                    melody.append((current_tick, motif_dur, pitch))
                    current_tick += motif_dur
                    current_pitch = pitch
                    previous_notes.append(pitch)
                    if len(previous_notes) > 8:
                        previous_notes.pop(0)

                # Skip the normal note generation for this duration
                continue

            # Generate note
            pitch = self.select_note(
                current_chord,
                current_pitch,
                target_direction,
                creativity,
                chord_tone_weight,
                min_pitch,
                max_pitch,
                previous_notes,
                bar_position,
                is_phrase_end
            )

            # Occasionally add embellishments
            embellish = False
            if creativity > 0.4 and duration >= 480 and random.random() < creativity * 0.5:
                embellish = True

            if embellish:
                # Split note into main note + embellishment
                main_duration = int(duration * 0.75)
                embel_duration = duration - main_duration

                # Add neighbor or passing tone
                if current_pitch is not None and random.random() < 0.5:
                    neighbor = self.add_neighbor_tone(pitch, current_chord, creativity)
                    if neighbor:
                        melody.append((current_tick, embel_duration, neighbor))
                        melody.append((current_tick + embel_duration, main_duration, pitch))
                    else:
                        melody.append((current_tick, duration, pitch))
                else:
                    melody.append((current_tick, duration, pitch))
            else:
                melody.append((current_tick, duration, pitch))

            # Create new motif occasionally
            if use_motifs and random.random() < 0.1 and len(self.motifs) < 5:
                motif_length = random.randint(3, 5)
                new_motif = self.create_motif(current_chord, motif_length, min_pitch, max_pitch)
                self.motifs.append(new_motif)
                print(f"   🎼 Created motif #{len(self.motifs)} with {motif_length} notes")

            previous_notes.append(pitch)
            if len(previous_notes) > 8:
                previous_notes.pop(0)

            current_pitch = pitch
            current_tick += duration

        print(f"\n✅ Generated {len(melody)} notes")
        print(f"   Motifs created: {len(self.motifs)}")
        print(f"{'='*80}\n")

        return melody

    def save_midi(
        self,
        melody: List[Tuple[int, int, int]],
        output_path: str,
        velocity: int = 80
    ):
        """
        Save melody to MIDI file

        Args:
            melody: List of (tick, duration, pitch) tuples
            output_path: Path to save MIDI file
            velocity: MIDI velocity (0-127)
        """
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        # Add tempo
        tempo_microseconds = int(60_000_000 / self.tempo)
        track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))

        # Add time signature
        track.append(MetaMessage('time_signature',
                                 numerator=self.time_signature[0],
                                 denominator=self.time_signature[1],
                                 time=0))

        # Sort melody by tick
        melody_sorted = sorted(melody, key=lambda x: x[0])

        # Convert to MIDI messages
        events = []
        for tick, duration, pitch in melody_sorted:
            events.append((tick, 'note_on', pitch, velocity))
            events.append((tick + duration, 'note_off', pitch, 0))

        # Sort all events by time
        events.sort(key=lambda x: x[0])

        # Convert to delta times
        current_time = 0
        for event_time, event_type, pitch, vel in events:
            delta = event_time - current_time
            if event_type == 'note_on':
                track.append(Message('note_on', note=pitch, velocity=vel, time=delta))
            else:
                track.append(Message('note_off', note=pitch, velocity=vel, time=delta))
            current_time = event_time

        # Save file
        mid.save(output_path)
        print(f"✅ Saved melody to: {output_path}")


def parse_chord_progression(chord_string: str, ticks_per_bar: int) -> Dict[int, str]:
    """
    Parse chord progression string
    Format: "0:Cm,1920:F7,3840:Bbm" or "Cm,F7,Bbm" (one per bar)
    """
    progression = {}

    if ':' in chord_string:
        # Explicit tick positions
        pairs = chord_string.split(',')
        for pair in pairs:
            tick_str, chord = pair.strip().split(':')
            tick = int(tick_str)
            progression[tick] = chord
    else:
        # One chord per bar
        chords = [c.strip() for c in chord_string.split(',')]
        for i, chord in enumerate(chords):
            tick = i * ticks_per_bar
            progression[tick] = chord

    return progression


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Melody Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple melody in C major
  python melody_generator.py --output melody.mid

  # Jazz melody with chord changes
  python melody_generator.py --key "C major" --chords "Cmaj7,Am7,Dm7,G7" --output jazz.mid

  # Creative, chromatic melody
  python melody_generator.py --key "D minor" --creativity 0.9 --syncopation 0.7 --output creative.mid

  # Descending melodic line
  python melody_generator.py --direction -0.8 --output descending.mid
        """
    )

    parser.add_argument('--output', '-o', required=True, help='Output MIDI file path')
    parser.add_argument('--key', default='C major', help='Musical key (e.g., "C major", "A minor", "Eb major")')
    parser.add_argument('--chords', default='C', help='Chord progression: "C,F,G" or "0:C,1920:F,3840:G"')
    parser.add_argument('--bars', type=int, default=8, help='Number of bars to generate')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo in BPM')

    # Range
    parser.add_argument('--min-note', type=int, default=60, help='Minimum MIDI note (default: 60/C4)')
    parser.add_argument('--max-note', type=int, default=84, help='Maximum MIDI note (default: 84/C6)')

    # Musical parameters
    parser.add_argument('--creativity', type=float, default=0.5,
                       help='Creativity level 0.0-1.0 (chromatic notes, tension)')
    parser.add_argument('--chord-tone-weight', type=float, default=0.7,
                       help='Preference for chord tones 0.0-1.0')
    parser.add_argument('--direction', type=float, default=0.0,
                       help='Melodic direction bias -1.0 (descending) to 1.0 (ascending)')

    # Rhythm
    parser.add_argument('--rhythm', default='moderate',
                       choices=['simple', 'moderate', 'complex', 'syncopated'],
                       help='Rhythmic density')
    parser.add_argument('--syncopation', type=float, default=0.3,
                       help='Syncopation amount 0.0-1.0')

    # Motifs
    parser.add_argument('--use-motifs', action='store_true', default=True,
                       help='Use melodic motifs')
    parser.add_argument('--no-motifs', dest='use_motifs', action='store_false',
                       help='Disable melodic motifs')
    parser.add_argument('--motif-repetition', type=float, default=0.4,
                       help='How often to repeat motifs 0.0-1.0')

    # Other
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--velocity', type=int, default=80, help='MIDI velocity (0-127)')

    args = parser.parse_args()

    # Create generator
    generator = MelodyGenerator(
        key=args.key,
        tempo=args.tempo,
        seed=args.seed
    )

    # Parse chord progression
    chord_progression = parse_chord_progression(args.chords, generator.bar_ticks)

    print(f"Chord progression: {chord_progression}")

    # Generate melody
    melody = generator.generate_melody(
        chord_progression=chord_progression,
        num_bars=args.bars,
        min_pitch=args.min_note,
        max_pitch=args.max_note,
        creativity=args.creativity,
        chord_tone_weight=args.chord_tone_weight,
        rhythmic_density=args.rhythm,
        syncopation=args.syncopation,
        direction_bias=args.direction,
        use_motifs=args.use_motifs,
        motif_repetition=args.motif_repetition
    )

    # Save to MIDI
    generator.save_midi(melody, args.output, velocity=args.velocity)


if __name__ == '__main__':
    main()
