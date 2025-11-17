#!/usr/bin/env python3
"""
Proper Algorithmic Melody Generation

Based on proven jazz pedagogy and algorithmic composition research:
1. Target-note technique (Jamey Aebersold, David Baker)
2. Chord-tone on downbeat principle
3. Passing tones and neighbor tones
4. Proper chord-scale theory (Berklee method)
5. Chromatic enclosures

References:
- David Baker: "How to Play Bebop"
- Jamey Aebersold: Jazz method
- GenJam genetic algorithm (Biles)
- Chord-scale theory (Russell, Berklee)
"""

import argparse
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage


class ProperMelodyGenerator:
    """
    Algorithmic melody generation using target-note technique
    """

    # Proper chord-scale mappings (Berklee method)
    CHORD_SCALES = {
        # Minor chords
        'm': ([0, 2, 3, 5, 7, 9, 10], [0, 3, 7, 10]),      # Dorian: [root, m3, 5, m7]
        'm7': ([0, 2, 3, 5, 7, 9, 10], [0, 3, 7, 10]),
        'm9': ([0, 2, 3, 5, 7, 9, 10], [0, 3, 7, 10]),

        # Dominant chords
        '7': ([0, 2, 4, 5, 7, 9, 10], [0, 4, 7, 10]),      # Mixolydian: [root, M3, 5, m7]
        '9': ([0, 2, 4, 5, 7, 9, 10], [0, 4, 7, 10]),
        '13': ([0, 2, 4, 5, 7, 9, 10], [0, 4, 7, 10]),

        # Major chords
        'maj': ([0, 2, 4, 5, 7, 9, 11], [0, 4, 7]),        # Ionian: [root, M3, 5]
        'maj7': ([0, 2, 4, 5, 7, 9, 11], [0, 4, 7, 11]),
        '6': ([0, 2, 4, 5, 7, 9, 11], [0, 4, 7, 9]),
    }

    def __init__(self, tempo: int = 120, seed: Optional[int] = None):
        self.tempo = tempo
        self.ticks_per_beat = 480
        self.bar_ticks = self.ticks_per_beat * 4  # 4/4 time

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def parse_chord(self, chord_name: str) -> Tuple[int, str]:
        """Parse chord name to root and quality"""
        note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

        # Extract root
        root_letter = chord_name[0].upper()
        root = note_map.get(root_letter, 0)

        # Handle accidentals
        rest = chord_name[1:]
        if rest.startswith('b') or rest.startswith('♭'):
            root -= 1
            rest = rest[1:]
        elif rest.startswith('#') or rest.startswith('♯'):
            root += 1
            rest = rest[1:]

        root = root % 12
        quality = rest if rest else 'maj'

        return root, quality

    def get_chord_info(self, chord_name: str, octave: int = 5) -> Dict:
        """Get scale and chord tones for a chord"""
        root_pc, quality = self.parse_chord(chord_name)

        # Get scale and chord tone intervals
        scale_intervals, chord_intervals = self.CHORD_SCALES.get(
            quality,
            self.CHORD_SCALES['maj']  # Default to major
        )

        # Convert to actual MIDI notes
        base_note = octave * 12 + root_pc
        scale_notes = [base_note + interval for interval in scale_intervals]
        chord_tones = [base_note + interval for interval in chord_intervals]

        # Expand to multiple octaves
        all_scale_notes = []
        all_chord_tones = []

        for oct_offset in [-12, 0, 12]:
            all_scale_notes.extend([n + oct_offset for n in scale_notes])
            all_chord_tones.extend([n + oct_offset for n in chord_tones])

        return {
            'scale_notes': sorted(set(all_scale_notes)),
            'chord_tones': sorted(set(all_chord_tones)),
            'root_pc': root_pc,
            'quality': quality
        }

    def select_target_note(
        self,
        chord_info: Dict,
        previous_note: Optional[int],
        min_pitch: int,
        max_pitch: int,
        prefer_direction: int = 0  # -1, 0, 1
    ) -> int:
        """
        Select a chord tone as target note
        Uses voice-leading principles
        """
        chord_tones = [n for n in chord_info['chord_tones']
                       if min_pitch <= n <= max_pitch]

        if not chord_tones:
            return 60  # Fallback

        if previous_note is None:
            # First note - pick from middle range
            mid = (min_pitch + max_pitch) / 2
            chord_tones.sort(key=lambda n: abs(n - mid))

            # Emphasize characteristic tones
            quality = chord_info['quality']
            if quality in ['m', 'm7', 'm9']:
                # Emphasize minor 3rd
                m3_pc = (chord_info['root_pc'] + 3) % 12
                for note in chord_tones:
                    if note % 12 == m3_pc:
                        return note
            elif quality in ['7', '9', '13']:
                # Emphasize major 3rd (leading tone function)
                M3_pc = (chord_info['root_pc'] + 4) % 12
                for note in chord_tones:
                    if note % 12 == M3_pc:
                        return note

            return chord_tones[0]

        # Subsequent notes - use voice leading with characteristic tone emphasis
        quality = chord_info['quality']

        # Find characteristic tone for this chord
        characteristic_pc = None
        if quality in ['m', 'm7', 'm9']:
            characteristic_pc = (chord_info['root_pc'] + 3) % 12  # Minor 3rd
        elif quality in ['7', '9', '13']:
            characteristic_pc = (chord_info['root_pc'] + 4) % 12  # Major 3rd

        # Look for characteristic tone within reasonable range
        if characteristic_pc is not None:
            characteristic_notes = [n for n in chord_tones if n % 12 == characteristic_pc]
            if characteristic_notes:
                # Find closest characteristic tone
                characteristic_notes.sort(key=lambda n: abs(n - previous_note))
                # Use it if it's within a reasonable interval (prefer within octave)
                if abs(characteristic_notes[0] - previous_note) <= 12:
                    return characteristic_notes[0]

        # Otherwise use voice leading
        chord_tones.sort(key=lambda n: abs(n - previous_note))

        # Apply directional preference
        if prefer_direction != 0:
            directional = [n for n in chord_tones
                          if (n > previous_note and prefer_direction > 0) or
                             (n < previous_note and prefer_direction < 0)]
            if directional:
                return directional[0]

        return chord_tones[0]

    def fill_with_passing_tones(
        self,
        note1: int,
        note2: int,
        chord_info: Dict,
        num_steps: int
    ) -> List[int]:
        """
        Fill between two target notes with passing tones
        Uses scale-wise motion
        """
        if num_steps <= 0:
            return []

        scale_notes = chord_info['scale_notes']

        # Determine direction
        if note2 > note1:
            # Ascending
            available = [n for n in scale_notes if note1 < n < note2]
            available.sort()
        else:
            # Descending
            available = [n for n in scale_notes if note2 < n < note1]
            available.sort(reverse=True)

        if not available:
            # Use chromatic if no scale tones available
            if note2 > note1:
                return list(range(note1 + 1, note2))[:num_steps]
            else:
                return list(range(note1 - 1, note2, -1))[:num_steps]

        # Take evenly spaced notes
        if len(available) <= num_steps:
            return available

        step = len(available) / (num_steps + 1)
        indices = [int(i * step) for i in range(1, num_steps + 1)]
        return [available[i] for i in indices if i < len(available)]

    def add_chromatic_approach(
        self,
        target: int,
        from_below: bool = True
    ) -> int:
        """Add chromatic approach note"""
        if from_below:
            return target - 1
        else:
            return target + 1

    def generate_melody(
        self,
        chord_progression: Dict[int, str],
        num_bars: int,
        min_pitch: int = 60,
        max_pitch: int = 76,
        use_chromatic_approaches: float = 0.3,  # Probability
    ) -> List[Tuple[int, int, int]]:
        """
        Generate melody using target-note technique

        Algorithm:
        1. Place chord tones on strong beats (targets)
        2. Fill with passing tones using scale
        3. Occasionally add chromatic approaches
        """
        print(f"\n{'='*80}")
        print(f"🎵 PROPER ALGORITHMIC MELODY GENERATION")
        print(f"{'='*80}")
        print(f"Method: Target-note technique (David Baker/Jamey Aebersold)")
        print(f"Chord-scale: Berklee method")
        print(f"Range: MIDI {min_pitch}-{max_pitch}")

        melody = []
        chord_changes = sorted(chord_progression.items())

        # Generate structure: target notes on beats 1 and 3
        total_ticks = num_bars * self.bar_ticks
        targets = []  # (tick, chord_name)

        chord_idx = 0
        current_chord = chord_changes[0][1]

        for bar in range(num_bars):
            bar_start = bar * self.bar_ticks

            # Update chord if needed
            while chord_idx < len(chord_changes) - 1 and bar_start >= chord_changes[chord_idx + 1][0]:
                chord_idx += 1
                current_chord = chord_changes[chord_idx][1]

            # Target notes on beats 1 and 3
            targets.append((bar_start, current_chord))  # Beat 1
            targets.append((bar_start + self.ticks_per_beat * 2, current_chord))  # Beat 3

        print(f"Generated {len(targets)} target notes")

        # Select target notes (chord tones)
        target_notes = []
        previous_note = None
        direction_tracker = 0  # Track overall direction

        for i, (tick, chord_name) in enumerate(targets):
            chord_info = self.get_chord_info(chord_name)

            # Alternate direction for contour
            if i % 4 == 0:
                direction_tracker = 0
            elif i % 4 == 1:
                direction_tracker = 1
            elif i % 4 == 2:
                direction_tracker = 0
            else:
                direction_tracker = -1

            target = self.select_target_note(
                chord_info,
                previous_note,
                min_pitch,
                max_pitch,
                direction_tracker
            )

            target_notes.append((tick, target, chord_name))
            previous_note = target

        # Fill between targets with passing tones
        for i in range(len(target_notes)):
            tick, note, chord_name = target_notes[i]
            chord_info = self.get_chord_info(chord_name)

            # Add target note
            if i < len(target_notes) - 1:
                next_tick, next_note, _ = target_notes[i + 1]
                duration = next_tick - tick
            else:
                duration = self.ticks_per_beat  # Half note for last

            # Decide if we use chromatic approach
            use_chromatic = random.random() < use_chromatic_approaches and i < len(target_notes) - 1

            if use_chromatic and i < len(target_notes) - 1:
                # Add chromatic approach before next target
                approach_tick = next_tick - 240  # One eighth note before
                approach_note = self.add_chromatic_approach(next_note, note < next_note)

                # Main note duration
                main_duration = approach_tick - tick
                melody.append((tick, main_duration, note))

                # Approach note
                melody.append((approach_tick, 240, approach_note))
            else:
                # No chromatic approach
                if i < len(target_notes) - 1:
                    # Fill with passing tones
                    next_tick, next_note, next_chord = target_notes[i + 1]
                    interval = abs(next_note - note)

                    if interval > 2:  # More than a whole step
                        # Calculate how many passing tones we can fit
                        available_time = next_tick - tick
                        num_eighths = available_time // 240

                        if num_eighths > 2:
                            num_passing = min(num_eighths - 2, interval - 1)
                            passing = self.fill_with_passing_tones(
                                note,
                                next_note,
                                chord_info,
                                num_passing
                            )

                            # Distribute time
                            total_notes = 1 + len(passing)
                            time_per_note = available_time // total_notes

                            # Add main note
                            melody.append((tick, time_per_note, note))

                            # Add passing tones
                            for j, p_note in enumerate(passing):
                                p_tick = tick + time_per_note * (j + 1)
                                melody.append((p_tick, time_per_note, p_note))
                        else:
                            melody.append((tick, duration, note))
                    else:
                        melody.append((tick, duration, note))
                else:
                    # Last note
                    melody.append((tick, duration, note))

        # Analyze
        melody.sort(key=lambda x: x[0])
        notes = [m[2] for m in melody]
        intervals = [abs(notes[i] - notes[i-1]) for i in range(1, len(notes))]

        if intervals:
            stepwise = sum(1 for i in intervals if i <= 2)
            print(f"\n✅ Generated {len(melody)} notes")
            print(f"   Stepwise motion: {stepwise}/{len(intervals)} ({stepwise/len(intervals)*100:.1f}%)")
            print(f"   Average interval: {sum(intervals)/len(intervals):.2f} semitones")
            print(f"   Max interval: {max(intervals)} semitones")

        print(f"{'='*80}\n")

        return melody

    def save_midi(self, melody: List[Tuple[int, int, int]], output_path: str):
        """Save to MIDI file"""
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        tempo_microseconds = int(60_000_000 / self.tempo)
        track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))

        events = []
        for tick, duration, pitch in melody:
            events.append((tick, 'note_on', pitch, 80))
            events.append((tick + duration, 'note_off', pitch, 0))

        events.sort(key=lambda x: x[0])

        current_time = 0
        for event_time, event_type, pitch, vel in events:
            delta = event_time - current_time
            if event_type == 'note_on':
                track.append(Message('note_on', note=pitch, velocity=vel, time=delta))
            else:
                track.append(Message('note_off', note=pitch, velocity=vel, time=delta))
            current_time = event_time

        mid.save(output_path)
        print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Proper Algorithmic Melody Generator')
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--chords', required=True, help='e.g., "Cm7,G7,Cm7,G7"')
    parser.add_argument('--bars', type=int, default=4)
    parser.add_argument('--min-note', type=int, default=60)
    parser.add_argument('--max-note', type=int, default=76)
    parser.add_argument('--chromatic', type=float, default=0.3, help='Chromatic approach probability')
    parser.add_argument('--tempo', type=int, default=120)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    generator = ProperMelodyGenerator(tempo=args.tempo, seed=args.seed)

    # Parse chords (simple: one per bar)
    chords = [c.strip() for c in args.chords.split(',')]
    chord_progression = {}
    for i, chord in enumerate(chords):
        chord_progression[i * generator.bar_ticks] = chord

    melody = generator.generate_melody(
        chord_progression,
        args.bars,
        args.min_note,
        args.max_note,
        args.chromatic
    )

    generator.save_midi(melody, args.output)


if __name__ == '__main__':
    main()
