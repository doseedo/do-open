#!/usr/bin/env python3
"""
Advanced Melody Generator - Research-Based Implementation

Based on state-of-the-art melodic composition principles:
- Proper chord-scale theory with avoid notes
- 60-75% stepwise motion (research-based)
- Leap resolution (move by step in opposite direction)
- Phrase contour management (single peak, arch shapes)
- Beat-aware chord tone placement
- Proper approach notes and enclosures
"""

import argparse
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

import sys
sys.path.insert(0, '/home/arlo/Data')
from melody_harmonizer_improved import ChordAnalyzer, ScaleContext, CHORD_LIBRARY


class ProperChordScales:
    """
    Proper chord-scale relationships based on jazz theory
    """

    # Mode formulas (semitones from root)
    MODES = {
        'ionian':     [0, 2, 4, 5, 7, 9, 11],      # Major scale
        'dorian':     [0, 2, 3, 5, 7, 9, 10],      # Minor with natural 6
        'phrygian':   [0, 1, 3, 5, 7, 8, 10],      # Minor with b2
        'lydian':     [0, 2, 4, 6, 7, 9, 11],      # Major with #4
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],      # Major with b7 (dominant)
        'aeolian':    [0, 2, 3, 5, 7, 8, 10],      # Natural minor
        'locrian':    [0, 1, 3, 5, 6, 8, 10],      # Half-diminished
        'altered':    [0, 1, 3, 4, 6, 8, 10],      # Altered dominant
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],   # Melodic minor
        'diminished_wh': [0, 2, 3, 5, 6, 8, 9, 11], # Whole-half diminished
    }

    # Chord type to scale mapping
    CHORD_SCALE_MAP = {
        # Major family
        'maj': ('ionian', [3]),      # Avoid 4
        'maj7': ('ionian', [3]),
        '6': ('ionian', [3]),
        'maj9': ('ionian', [3]),
        'maj13': ('lydian', []),     # Lydian for #11

        # Dominant family
        '7': ('mixolydian', [3]),    # Avoid 4
        '9': ('mixolydian', [3]),
        '13': ('mixolydian', [3]),
        '7alt': ('altered', []),
        '7#5': ('altered', []),
        '7b9': ('altered', []),

        # Minor family
        'm': ('dorian', []),         # No avoid notes
        'm7': ('dorian', []),
        'm9': ('dorian', []),
        'm11': ('dorian', []),
        'm6': ('dorian', []),
        'madd9': ('dorian', []),
        'm(maj7)': ('melodic_minor', []),

        # Diminished family
        'm7b5': ('locrian', [1]),    # Avoid natural 2
        'ø7': ('locrian', [1]),
        'dim': ('diminished_wh', []),
        'dim7': ('diminished_wh', []),
        '°7': ('diminished_wh', []),
    }

    @classmethod
    def get_scale_for_chord(cls, chord_name: str, root: int) -> Tuple[List[int], List[int]]:
        """
        Get proper scale and avoid notes for a chord

        Returns:
            (scale_notes, avoid_notes) as pitch classes
        """
        # Parse chord quality
        chord_quality = cls._parse_chord_quality(chord_name)

        # Get scale and avoid notes
        mode_name, avoid_indices = cls.CHORD_SCALE_MAP.get(
            chord_quality,
            ('ionian', [])  # Default to major
        )

        # Build scale from root
        mode_intervals = cls.MODES[mode_name]
        scale_notes = [(root + interval) % 12 for interval in mode_intervals]

        # Build avoid notes list
        avoid_notes = [scale_notes[i] for i in avoid_indices if i < len(scale_notes)]

        return scale_notes, avoid_notes

    @classmethod
    def _parse_chord_quality(cls, chord_name: str) -> str:
        """Extract chord quality from chord name"""
        # Remove root note
        for note in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
            if chord_name.startswith(note):
                quality = chord_name[len(note):]
                # Handle accidentals
                if quality.startswith('b') or quality.startswith('#'):
                    quality = quality[1:]
                return quality.lower() if quality else 'maj'
        return 'maj'


class MelodyGeneratorAdvanced:
    """Advanced melody generation with proper music theory"""

    def __init__(
        self,
        key: str = "C major",
        tempo: int = 120,
        time_signature: Tuple[int, int] = (4, 4),
        seed: Optional[int] = None
    ):
        self.parse_key(key)
        self.tempo = tempo
        self.time_signature = time_signature
        self.ticks_per_beat = 480
        self.bar_ticks = self.ticks_per_beat * time_signature[0]

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Melodic state tracking
        self.recent_notes = deque(maxlen=8)  # Last 8 notes
        self.phrase_notes = []  # All notes in current phrase
        self.last_leap = 0  # Size of last leap
        self.needs_resolution = False  # True after leap > 4

    def parse_key(self, key: str):
        """Parse key string"""
        parts = key.strip().split()
        root_str = parts[0] if parts else "C"
        scale_type = parts[1].lower() if len(parts) > 1 else "major"

        note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        root_note = note_map.get(root_str[0].upper(), 0)

        if len(root_str) > 1:
            if 'b' in root_str or '♭' in root_str:
                root_note -= 1
            elif '#' in root_str or '♯' in root_str:
                root_note += 1

        self.key_root = root_note % 12
        self.scale_type = scale_type

    def get_chord_tones(self, chord_name: str) -> Set[int]:
        """Get chord tones as pitch classes"""
        if chord_name in CHORD_LIBRARY:
            root = CHORD_LIBRARY[chord_name][0] % 12
        else:
            root = 0

        # Parse chord using harmonizer's analyzer
        chord_info = ChordAnalyzer.parse_chord(chord_name, use_extensions=False, scale_context=None)
        return set(note % 12 for note in chord_info['chord_tones'])

    def select_note(
        self,
        chord_name: str,
        current_pitch: Optional[int],
        beat_strength: float,  # 0.0 to 1.0
        stepwise_bias: float,  # How much to prefer stepwise motion
        target_contour: float,  # -1 to +1 (descending to ascending)
        min_pitch: int,
        max_pitch: int,
    ) -> int:
        """
        Select next note using proper music theory

        Args:
            chord_name: Current chord
            current_pitch: Previous note (None if first)
            beat_strength: 1.0 = downbeat, 0.5 = strong beat, 0.0 = weak beat
            stepwise_bias: 0.5-0.8 recommended (forces stepwise motion)
            target_contour: Desired melodic direction
            min_pitch: Minimum MIDI note
            max_pitch: Maximum MIDI note
        """
        # Get proper chord-scale
        if chord_name in CHORD_LIBRARY:
            root = CHORD_LIBRARY[chord_name][0] % 12
        else:
            root = 0

        scale_notes, avoid_notes = ProperChordScales.get_scale_for_chord(chord_name, root)
        chord_tones = self.get_chord_tones(chord_name)

        # Determine max interval based on resolution needs
        if self.needs_resolution:
            max_interval = 2  # MUST resolve by step
        else:
            # Stepwise bias determines max interval
            if stepwise_bias > 0.75:
                max_interval = 2
            elif stepwise_bias > 0.6:
                max_interval = 4
            elif stepwise_bias > 0.4:
                max_interval = 7
            else:
                max_interval = 12

        # First note - prefer chord tones in middle range
        if current_pitch is None:
            candidates = []
            mid_point = (min_pitch + max_pitch) / 2

            for octave in range(min_pitch // 12, (max_pitch // 12) + 1):
                for pc in scale_notes:
                    if pc in avoid_notes:
                        continue
                    note = octave * 12 + pc
                    if min_pitch <= note <= max_pitch:
                        # Prefer chord tones
                        weight = 3.0 if pc in chord_tones else 1.0
                        # Prefer middle range
                        range_weight = 1.0 - abs(note - mid_point) / (max_pitch - min_pitch)
                        candidates.append((note, weight * range_weight))

            if not candidates:
                return 60  # Fallback

            notes, weights = zip(*candidates)
            weights = np.array(weights)
            weights = weights / weights.sum()
            return int(np.random.choice(notes, p=weights))

        # Subsequent notes
        candidates = []

        for interval in range(-max_interval, max_interval + 1):
            if interval == 0:
                continue  # No repeated notes

            candidate = current_pitch + interval

            # Range check
            if candidate < min_pitch or candidate > max_pitch:
                continue

            pc = candidate % 12

            # Skip if not in scale
            if pc not in scale_notes:
                continue

            # Skip avoid notes UNLESS on weak beat with stepwise motion
            if pc in avoid_notes:
                if beat_strength > 0.3 or abs(interval) > 2:
                    continue  # Hard avoid
                # Allow on weak beat if passing

            # Base weight
            weight = 1.0

            # 1. STEPWISE MOTION BIAS (most important)
            abs_interval = abs(interval)
            if abs_interval <= 2:
                weight *= (2.0 + stepwise_bias * 3.0)  # Strong preference
            elif abs_interval <= 4:
                weight *= (1.0 - stepwise_bias * 0.5)  # Reduce small leaps
            else:
                weight *= (0.5 - stepwise_bias * 0.4)  # Heavily penalize leaps

            # 2. LEAP RESOLUTION
            if self.needs_resolution:
                # MUST move opposite direction
                if (self.last_leap > 0 and interval > 0) or (self.last_leap < 0 and interval < 0):
                    continue  # Same direction as leap - skip
                if abs_interval > 2:
                    continue  # Must be stepwise
                weight *= 5.0  # Strongly prefer resolution

            # 3. CHORD TONE PLACEMENT (beat-dependent)
            if pc in chord_tones:
                if beat_strength > 0.8:  # Downbeat
                    weight *= 4.0
                elif beat_strength > 0.5:  # Strong beat
                    weight *= 2.5
                else:  # Weak beat
                    weight *= 1.5
            else:
                if beat_strength > 0.8:  # Downbeat
                    weight *= 0.3  # Strongly avoid non-chord tones
                elif beat_strength > 0.5:  # Strong beat
                    weight *= 0.6
                else:  # Weak beat - OK for passing tones
                    weight *= 1.0

            # 4. CONTOUR GUIDANCE
            if target_contour > 0 and interval > 0:
                weight *= (1.0 + target_contour)
            elif target_contour < 0 and interval < 0:
                weight *= (1.0 + abs(target_contour))

            # 5. AVOID SAWTOOTH PATTERNS
            if len(self.recent_notes) >= 3:
                recent_directions = [
                    1 if self.recent_notes[i] > self.recent_notes[i-1] else -1
                    for i in range(-2, 0)
                ]
                if len(set(recent_directions)) == 1:  # All same direction
                    # Encourage changing direction
                    if (recent_directions[0] > 0 and interval < 0) or \
                       (recent_directions[0] < 0 and interval > 0):
                        weight *= 1.5

            # 6. RANGE CORRECTION
            if candidate >= max_pitch - 3:
                if interval > 0:
                    weight *= 0.2
                else:
                    weight *= 2.0
            elif candidate <= min_pitch + 3:
                if interval < 0:
                    weight *= 0.2
                else:
                    weight *= 2.0

            # 7. AVOID IMMEDIATE REPETITION
            if len(self.recent_notes) > 0:
                if candidate in list(self.recent_notes)[-3:]:
                    weight *= 0.3

            candidates.append((candidate, weight, interval))

        if not candidates:
            # No valid candidates - relax constraints
            return current_pitch

        notes, weights, intervals = zip(*candidates)
        weights = np.array(weights)
        weights = weights / weights.sum()

        selected_idx = np.random.choice(len(notes), p=weights)
        selected_note = int(notes[selected_idx])
        selected_interval = intervals[selected_idx]

        # Update leap tracking
        if abs(selected_interval) > 4:
            self.last_leap = selected_interval
            self.needs_resolution = True
        elif abs(selected_interval) <= 2 and self.needs_resolution:
            # Resolution occurred
            if (self.last_leap > 0 and selected_interval < 0) or \
               (self.last_leap < 0 and selected_interval > 0):
                self.needs_resolution = False

        self.recent_notes.append(selected_note)

        return selected_note

    def generate_melody(
        self,
        chord_progression: Dict[int, str],
        num_bars: int,
        min_pitch: int = 60,
        max_pitch: int = 76,
        stepwise_ratio: float = 0.7,  # Target 70% stepwise motion
        contour_style: str = 'balanced',  # 'ascending', 'descending', 'balanced', 'arch'
    ) -> List[Tuple[int, int, int]]:
        """
        Generate melody with proper music theory

        Args:
            stepwise_ratio: 0.6-0.8 recommended (research-based)
            contour_style: Overall phrase shape
        """
        print(f"\n{'='*80}")
        print(f"🎵 ADVANCED MELODY GENERATOR (Research-Based)")
        print(f"{'='*80}")
        print(f"Key: {self.key_root} {self.scale_type}")
        print(f"Target stepwise motion: {stepwise_ratio*100:.0f}%")
        print(f"Contour style: {contour_style}")
        print(f"Range: MIDI {min_pitch}-{max_pitch}")

        # Generate rhythm (quarter and eighth notes for now)
        total_ticks = num_bars * self.bar_ticks
        rhythm = []
        current_tick = 0

        while current_tick < total_ticks:
            # Mix of quarter and eighth notes
            if random.random() < 0.6:
                duration = 480  # Quarter
            else:
                duration = 240  # Eighth

            remaining = total_ticks - current_tick
            if duration > remaining:
                duration = remaining

            rhythm.append(duration)
            current_tick += duration

        print(f"Generated {len(rhythm)} notes")

        # Sort chord progression
        chord_changes = sorted(chord_progression.items())

        melody = []
        current_tick = 0
        current_pitch = None
        chord_idx = 0
        current_chord = chord_changes[0][1] if chord_changes else "C"

        # Determine contour targets
        phrase_length = 4 * self.bar_ticks

        for note_idx, duration in enumerate(rhythm):
            # Update chord
            while chord_idx < len(chord_changes) - 1 and current_tick >= chord_changes[chord_idx + 1][0]:
                chord_idx += 1
                current_chord = chord_changes[chord_idx][1]

            # Calculate beat strength
            beat_pos = current_tick % self.bar_ticks
            if beat_pos == 0:
                beat_strength = 1.0  # Downbeat
            elif beat_pos == self.ticks_per_beat * 2:  # Beat 3 in 4/4
                beat_strength = 0.6
            elif beat_pos % self.ticks_per_beat == 0:
                beat_strength = 0.5  # Other beats
            else:
                beat_strength = 0.0  # Off-beat

            # Determine contour target
            phrase_pos = (current_tick % phrase_length) / phrase_length

            if contour_style == 'arch':
                # Arch shape - peak in middle
                target_contour = np.sin(phrase_pos * np.pi) * 2 - 1
            elif contour_style == 'ascending':
                target_contour = 0.5
            elif contour_style == 'descending':
                target_contour = -0.5
            else:  # balanced
                # Gentle wave
                target_contour = np.sin(phrase_pos * np.pi * 2) * 0.3

            # Select note
            pitch = self.select_note(
                current_chord,
                current_pitch,
                beat_strength,
                stepwise_ratio,
                target_contour,
                min_pitch,
                max_pitch
            )

            melody.append((current_tick, duration, pitch))
            current_pitch = pitch
            current_tick += duration

        # Analyze generated melody
        intervals = [abs(melody[i][2] - melody[i-1][2]) for i in range(1, len(melody))]
        stepwise = sum(1 for i in intervals if i <= 2)
        stepwise_pct = stepwise / len(intervals) * 100 if intervals else 0

        print(f"\n✅ Generated {len(melody)} notes")
        print(f"   Stepwise motion: {stepwise_pct:.1f}%")
        print(f"   Average interval: {sum(intervals)/len(intervals):.2f} semitones")
        print(f"   Max interval: {max(intervals)} semitones")
        print(f"{'='*80}\n")

        return melody

    def save_midi(self, melody: List[Tuple[int, int, int]], output_path: str, velocity: int = 80):
        """Save melody to MIDI file"""
        mid = MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = MidiTrack()
        mid.tracks.append(track)

        tempo_microseconds = int(60_000_000 / self.tempo)
        track.append(MetaMessage('set_tempo', tempo=tempo_microseconds, time=0))
        track.append(MetaMessage('time_signature',
                                 numerator=self.time_signature[0],
                                 denominator=self.time_signature[1],
                                 time=0))

        events = []
        for tick, duration, pitch in melody:
            events.append((tick, 'note_on', pitch, velocity))
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


def parse_chord_progression(chord_string: str, ticks_per_bar: int) -> Dict[int, str]:
    """Parse chord progression string"""
    progression = {}

    if ':' in chord_string:
        pairs = chord_string.split(',')
        for pair in pairs:
            tick_str, chord = pair.strip().split(':')
            progression[int(tick_str)] = chord
    else:
        chords = [c.strip() for c in chord_string.split(',')]
        for i, chord in enumerate(chords):
            progression[i * ticks_per_bar] = chord

    return progression


def main():
    parser = argparse.ArgumentParser(description='Advanced Melody Generator (Research-Based)')

    parser.add_argument('--output', '-o', required=True, help='Output MIDI file')
    parser.add_argument('--key', default='C major', help='Musical key')
    parser.add_argument('--chords', default='C', help='Chord progression')
    parser.add_argument('--bars', type=int, default=4, help='Number of bars')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo (BPM)')

    parser.add_argument('--min-note', type=int, default=60, help='Min MIDI note')
    parser.add_argument('--max-note', type=int, default=76, help='Max MIDI note (smaller range = more melodic)')

    parser.add_argument('--stepwise', type=float, default=0.7,
                       help='Stepwise motion ratio 0.6-0.8 (research-based)')
    parser.add_argument('--contour', default='balanced',
                       choices=['ascending', 'descending', 'balanced', 'arch'],
                       help='Phrase contour')

    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--velocity', type=int, default=80, help='MIDI velocity')

    args = parser.parse_args()

    generator = MelodyGeneratorAdvanced(
        key=args.key,
        tempo=args.tempo,
        seed=args.seed
    )

    chord_progression = parse_chord_progression(args.chords, generator.bar_ticks)

    melody = generator.generate_melody(
        chord_progression=chord_progression,
        num_bars=args.bars,
        min_pitch=args.min_note,
        max_pitch=args.max_note,
        stepwise_ratio=args.stepwise,
        contour_style=args.contour
    )

    generator.save_midi(melody, args.output, velocity=args.velocity)


if __name__ == '__main__':
    main()
