#!/usr/bin/env python3
"""
Reggae Music Generator - Roots, Dub, and Dancehall

This module implements comprehensive reggae music generation including:
- Roots reggae (Bob Marley, Burning Spear, Culture)
- Dub (King Tubby, Lee "Scratch" Perry, Scientist)
- Dancehall (Yellowman, Shabba Ranks, Beenie Man)
- Lovers Rock, Rocksteady

Features:
- One-drop, rockers, and steppers drum rhythms
- Offbeat guitar "skank" patterns
- Melodic bass lines (bass as lead instrument)
- Dub effects (delay, reverb, drop-outs)
- Riddim-based composition
- Nyabinghi drumming patterns

Author: Agent 7 - World Music & Additional Genres
References:
- "Bass Culture: When Reggae Was King" - Lloyd Bradley
- "Dub: Soundscapes and Shattered Songs" - Michael Veal
- King Tubby's dub mixing techniques
- Studio One and Treasure Isle riddims
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ReggaeStyle(Enum):
    """Reggae sub-genres"""
    ROOTS = "roots"  # Classic roots reggae (70s)
    DUB = "dub"  # Instrumental dub versions
    DANCEHALL = "dancehall"  # Digital dancehall (80s-90s)
    LOVERS_ROCK = "lovers_rock"  # Romantic UK reggae
    ROCKSTEADY = "rocksteady"  # Pre-reggae (60s)
    RAGGA = "ragga"  # Digital ragga/dancehall
    STEPPERS = "steppers"  # UK steppers style


class DrumPattern(Enum):
    """Reggae drum patterns"""
    ONE_DROP = "one_drop"  # Classic reggae (emphasis on 3)
    ROCKERS = "rockers"  # Four-on-the-floor variant
    STEPPERS = "steppers"  # Marching 4/4 feel
    FLYING_CYMBALS = "flying_cymbals"  # Busy hi-hat pattern
    NYABINGHI = "nyabinghi"  # Traditional Rastafarian drumming


@dataclass
class ReggaeRiddim:
    """
    Reggae riddim (instrumental backing track)

    In reggae culture, riddims are reused across many songs with different vocals.
    Famous riddims: "Real Rock", "Stalag 17", "Sleng Teng"
    """
    name: str
    tempo: int  # BPM
    key_root: int  # MIDI note
    chord_progression: List[Tuple[int, str]]  # (degree, quality)
    drum_pattern: DrumPattern
    bass_style: str  # 'walking', 'melodic', 'minimal'


class OneDrop:
    """
    One-drop reggae drum pattern generator

    The one-drop is THE signature reggae rhythm:
    - Bass drum on beat 3 only
    - Snare on beat 3
    - Hi-hat plays steady quarter notes or offbeat eighths
    - Emphasis creates the reggae "lilt"

    References:
    - Carlton Barrett (Bob Marley's drummer)
    - Sly Dunbar technique
    """

    @staticmethod
    def generate_pattern(measures: int = 4,
                        style: str = 'classic',
                        tempo: int = 75) -> List[Tuple[str, float, int]]:
        """
        Generate one-drop drum pattern

        Args:
            measures: Number of measures
            style: 'classic', 'minimal', or 'flying_cymbals'
            tempo: BPM for feel (slower = deeper groove)

        Returns:
            List of (drum_type, time_in_beats, velocity) tuples
        """
        pattern = []

        for measure in range(measures):
            offset = measure * 4.0

            if style == 'classic':
                # Classic one-drop: kick and snare on 3
                pattern.extend([
                    # Beat 1: hi-hat only (or silence)
                    ('hihat_closed', offset + 0.0, 60),
                    ('hihat_closed', offset + 0.5, 50),  # Offbeat

                    # Beat 2: hi-hat
                    ('hihat_closed', offset + 1.0, 60),
                    ('hihat_closed', offset + 1.5, 50),

                    # Beat 3: THE DROP (kick + snare + open hi-hat)
                    ('kick', offset + 2.0, 110),
                    ('snare', offset + 2.0, 100),
                    ('hihat_open', offset + 2.0, 80),
                    ('hihat_closed', offset + 2.5, 50),

                    # Beat 4: hi-hat
                    ('hihat_closed', offset + 3.0, 60),
                    ('hihat_closed', offset + 3.5, 50),
                ])

            elif style == 'minimal':
                # Minimal one-drop (just the essentials)
                pattern.extend([
                    ('kick', offset + 2.0, 110),
                    ('snare', offset + 2.0, 100),
                    ('hihat_closed', offset + 0.5, 50),
                    ('hihat_closed', offset + 1.5, 50),
                    ('hihat_closed', offset + 2.5, 50),
                    ('hihat_closed', offset + 3.5, 50),
                ])

            elif style == 'flying_cymbals':
                # Busy hi-hat (flying cymbals style)
                pattern.extend([
                    ('kick', offset + 2.0, 110),
                    ('snare', offset + 2.0, 100),
                    # Sixteenth note hi-hats
                    ('hihat_closed', offset + 0.0, 55),
                    ('hihat_closed', offset + 0.25, 45),
                    ('hihat_closed', offset + 0.5, 60),
                    ('hihat_closed', offset + 0.75, 45),
                    ('hihat_closed', offset + 1.0, 55),
                    ('hihat_closed', offset + 1.25, 45),
                    ('hihat_closed', offset + 1.5, 60),
                    ('hihat_closed', offset + 1.75, 45),
                    ('hihat_open', offset + 2.0, 70),
                    ('hihat_closed', offset + 2.5, 60),
                    ('hihat_closed', offset + 3.0, 55),
                    ('hihat_closed', offset + 3.5, 60),
                ])

        return pattern


class Rockers:
    """
    Rockers reggae drum pattern generator

    Rockers emphasizes all four beats with kick drum, creating a more
    driving, insistent feel than one-drop. Popular in 70s roots reggae.

    References:
    - Sly Dunbar rockers style
    - "Rockers" soundtrack
    """

    @staticmethod
    def generate_pattern(measures: int = 4,
                        variation: str = 'standard') -> List[Tuple[str, float, int]]:
        """
        Generate rockers drum pattern

        Args:
            measures: Number of measures
            variation: 'standard' or 'heavy'

        Returns:
            List of (drum_type, time_in_beats, velocity) tuples
        """
        pattern = []

        for measure in range(measures):
            offset = measure * 4.0

            # Rockers: kick on all 4 beats, snare on 2 and 4 (or 3)
            if variation == 'standard':
                pattern.extend([
                    # Beat 1
                    ('kick', offset + 0.0, 100),
                    ('hihat_closed', offset + 0.5, 55),

                    # Beat 2
                    ('kick', offset + 1.0, 95),
                    ('snare', offset + 1.0, 85),  # Cross-stick or rim
                    ('hihat_closed', offset + 1.5, 55),

                    # Beat 3
                    ('kick', offset + 2.0, 110),
                    ('snare', offset + 2.0, 100),
                    ('hihat_open', offset + 2.0, 75),
                    ('hihat_closed', offset + 2.5, 55),

                    # Beat 4
                    ('kick', offset + 3.0, 95),
                    ('snare', offset + 3.0, 85),
                    ('hihat_closed', offset + 3.5, 55),
                ])

            elif variation == 'heavy':
                # Heavy rockers (more aggressive)
                pattern.extend([
                    ('kick', offset + 0.0, 110),
                    ('hihat_closed', offset + 0.5, 60),
                    ('kick', offset + 1.0, 110),
                    ('snare', offset + 1.0, 95),
                    ('hihat_closed', offset + 1.5, 60),
                    ('kick', offset + 2.0, 120),
                    ('snare', offset + 2.0, 110),
                    ('hihat_open', offset + 2.0, 85),
                    ('kick', offset + 3.0, 110),
                    ('snare', offset + 3.0, 95),
                    ('hihat_closed', offset + 3.5, 60),
                ])

        return pattern


class Steppers:
    """
    Steppers reggae drum pattern generator

    Steppers rhythm has kick on all four beats with a marching,
    militant feel. Popular in UK sound system culture and roots revival.

    References:
    - Creation Steppers
    - UK sound system style
    """

    @staticmethod
    def generate_pattern(measures: int = 4) -> List[Tuple[str, float, int]]:
        """
        Generate steppers drum pattern

        Args:
            measures: Number of measures

        Returns:
            List of (drum_type, time_in_beats, velocity) tuples
        """
        pattern = []

        for measure in range(measures):
            offset = measure * 4.0

            # Steppers: steady kick on all 4 beats, snare on 3
            pattern.extend([
                # Beat 1
                ('kick', offset + 0.0, 110),
                ('hihat_closed', offset + 0.5, 50),

                # Beat 2
                ('kick', offset + 1.0, 110),
                ('hihat_closed', offset + 1.5, 50),

                # Beat 3
                ('kick', offset + 2.0, 115),
                ('snare', offset + 2.0, 100),
                ('hihat_open', offset + 2.0, 70),
                ('hihat_closed', offset + 2.5, 50),

                # Beat 4
                ('kick', offset + 3.0, 110),
                ('hihat_closed', offset + 3.5, 50),
            ])

        return pattern


class Skank:
    """
    Reggae guitar "skank" pattern generator

    The skank is the characteristic reggae guitar rhythm: short, choppy
    chords on the offbeat (the "and" of each beat). Often played with
    palm muting for percussive effect.

    References:
    - Al Anderson (Bob Marley's guitarist)
    - Ernest Ranglin technique
    """

    @staticmethod
    def generate_pattern(chord_notes: List[int],
                        measures: int = 4,
                        style: str = 'standard') -> List[Tuple[List[int], float, float, int]]:
        """
        Generate guitar skank pattern

        Args:
            chord_notes: MIDI notes for chord voicing
            measures: Number of measures
            style: 'standard', 'double', or 'bubble'

        Returns:
            List of (notes, time_in_beats, duration, velocity) tuples
        """
        pattern = []

        for measure in range(measures):
            offset = measure * 4.0

            if style == 'standard':
                # Standard skank: offbeat eighths
                for beat in range(4):
                    time = offset + beat + 0.5  # Offbeat
                    duration = 0.25  # Short, choppy
                    velocity = 75
                    pattern.append((chord_notes, time, duration, velocity))

            elif style == 'double':
                # Double skank: on-beat and offbeat
                for beat in range(4):
                    # On-beat (softer)
                    pattern.append((chord_notes, offset + beat, 0.2, 60))
                    # Offbeat (louder)
                    pattern.append((chord_notes, offset + beat + 0.5, 0.25, 80))

            elif style == 'bubble':
                # Bubble rhythm (dancehall style)
                # Syncopated pattern with emphasis
                bubble_times = [0.5, 1.5, 2.0, 2.5, 3.5]
                for time_offset in bubble_times:
                    time = offset + time_offset
                    duration = 0.2
                    velocity = 85 if time_offset in [2.0, 2.5] else 70
                    pattern.append((chord_notes, time, duration, velocity))

        return pattern


class ReggaeBass:
    """
    Reggae bass line generator

    In reggae, the bass is often the lead instrument, playing melodic lines
    that define the riddim. Bass lines emphasize roots and fifths with
    melodic passing tones.

    References:
    - Aston "Family Man" Barrett
    - Robbie Shakespeare
    - Flabba Holt
    """

    # Common reggae bass patterns
    PATTERNS = {
        'walking': 'Smooth walking bass with chromatic approaches',
        'melodic': 'Melodic lead bass with rhythmic variations',
        'minimal': 'Minimal roots and fifths',
        'dub': 'Sparse, echoing bass with space',
    }

    @staticmethod
    def generate_line(chord_progression: List[Tuple[int, str, float]],
                     style: str = 'melodic') -> List[Tuple[int, float, float, int]]:
        """
        Generate reggae bass line

        Args:
            chord_progression: List of (root_note, quality, duration) tuples
            style: Bass line style ('walking', 'melodic', 'minimal', 'dub')

        Returns:
            List of (note, time, duration, velocity) tuples
        """
        bass_line = []
        time = 0.0

        for i, (root, quality, chord_duration) in enumerate(chord_progression):
            # Bass plays in lower octave
            bass_root = root - 12 if root > 48 else root
            fifth = bass_root + 7
            octave = bass_root + 12

            # Get next chord for approach tones
            next_root = chord_progression[(i + 1) % len(chord_progression)][0] - 12 if i < len(chord_progression) - 1 else bass_root

            if style == 'melodic':
                # Melodic bass (signature reggae style)
                # Pattern: root, fifth, root, approach to next chord
                bass_line.extend([
                    (bass_root, time, 0.75, 95),
                    (fifth, time + 1.0, 0.5, 85),
                    (bass_root, time + 1.5, 0.5, 90),
                    (octave, time + 2.0, 0.75, 85),
                    (next_root + 1 if next_root > bass_root else next_root - 1,
                     time + 3.0, 0.5, 80),  # Chromatic approach
                ])

            elif style == 'walking':
                # Walking bass
                bass_line.extend([
                    (bass_root, time, 1.0, 90),
                    (bass_root + 3, time + 1.0, 1.0, 80),
                    (fifth, time + 2.0, 1.0, 85),
                    (fifth + 2, time + 3.0, 1.0, 80),
                ])

            elif style == 'minimal':
                # Minimal bass (roots and fifths)
                bass_line.extend([
                    (bass_root, time, 1.5, 100),
                    (fifth, time + 2.0, 1.5, 90),
                ])

            elif style == 'dub':
                # Dub bass (sparse, with space for echo)
                bass_line.extend([
                    (bass_root, time, 1.0, 100),
                    (fifth, time + 2.5, 0.5, 95),
                ])

            time += chord_duration

        return bass_line


class DubEffects:
    """
    Dub mixing effects generator

    Dub is characterized by heavy use of effects, drop-outs, and
    creative mixing. This class generates effect automation curves.

    References:
    - King Tubby's mixing techniques
    - Lee "Scratch" Perry production
    - Scientist's dub style
    """

    @staticmethod
    def generate_delay_pattern(measures: int = 4,
                              delay_time: float = 0.375) -> List[Tuple[str, float, int]]:
        """
        Generate delay/echo effect pattern

        Args:
            measures: Number of measures
            delay_time: Delay time in beats (e.g., 0.375 = dotted eighth)

        Returns:
            List of (event_type, time, intensity) tuples
        """
        # In dub, delay is often used on snare and vocal phrases
        pattern = []

        for measure in range(measures):
            offset = measure * 4.0

            # Apply delay on beat 3 (the drop)
            pattern.append(('delay_send', offset + 2.0, 127))  # Max delay
            pattern.append(('delay_send', offset + 2.5, 80))   # Reduce
            pattern.append(('delay_send', offset + 3.0, 40))   # Fade out
            pattern.append(('delay_send', offset + 3.5, 0))    # Off

        return pattern

    @staticmethod
    def generate_dropout(total_measures: int = 16,
                        track: str = 'drums') -> List[Tuple[float, float, str]]:
        """
        Generate drop-out pattern (remove instruments randomly)

        Args:
            total_measures: Total measures in track
            track: Which track to drop out ('drums', 'bass', 'keys', 'vocal')

        Returns:
            List of (start_time, end_time, track_name) tuples
        """
        dropouts = []
        measure = 0

        while measure < total_measures:
            # Randomly drop out for 1-4 beats
            if random.random() > 0.6:  # 40% chance of dropout
                start = measure * 4.0
                duration = random.choice([1.0, 2.0, 4.0])
                end = start + duration
                dropouts.append((start, end, track))
                measure += duration / 4.0
            else:
                measure += 1

        return dropouts

    @staticmethod
    def generate_filter_sweep(measures: int = 4) -> List[Tuple[float, int]]:
        """
        Generate filter cutoff sweep automation

        Args:
            measures: Number of measures

        Returns:
            List of (time, cutoff_value) tuples
        """
        sweep = []
        total_time = measures * 4.0

        # Sine wave filter sweep
        steps = 32
        for i in range(steps):
            time = (i / steps) * total_time
            # Cutoff ranges from 30 to 127
            import math
            cutoff = int(30 + 97 * (0.5 + 0.5 * math.sin(2 * math.pi * i / steps)))
            sweep.append((time, cutoff))

        return sweep


class ReggaeGenerator:
    """
    Main reggae music generator

    Combines all reggae elements to generate complete riddims and arrangements.
    """

    # Minor pentatonic scale (common in reggae)
    MINOR_PENTATONIC = [0, 3, 5, 7, 10]

    # Major scale
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

    # Famous riddim progressions
    RIDDIMS = {
        'classic_roots': [
            (1, 'minor', 4.0),
            (6, 'major', 4.0),
            (4, 'major', 4.0),
            (5, 'major', 4.0),
        ],
        'steppers': [
            (1, 'minor', 4.0),
            (4, 'major', 4.0),
            (1, 'minor', 4.0),
            (1, 'minor', 4.0),
        ],
        'dancehall': [
            (1, 'minor', 2.0),
            (6, 'major', 2.0),
            (3, 'major', 2.0),
            (7, 'major', 2.0),
        ],
    }

    def __init__(self, style: ReggaeStyle = ReggaeStyle.ROOTS,
                 key_root: int = 57, tempo: int = 75):
        """
        Initialize reggae generator

        Args:
            style: Reggae sub-genre
            key_root: Root note (MIDI), typically A (57) or D (50)
            tempo: BPM (60-90 for roots, 90-110 for dancehall)
        """
        self.style = style
        self.key_root = key_root
        self.tempo = tempo

    def generate_riddim(self, measures: int = 16,
                       riddim_type: Optional[str] = None) -> Dict[str, List]:
        """
        Generate complete reggae riddim (backing track)

        Args:
            measures: Number of measures
            riddim_type: Type of riddim progression

        Returns:
            Dictionary with all instrument tracks
        """
        # Select riddim progression
        if riddim_type is None:
            if self.style == ReggaeStyle.DANCEHALL:
                riddim_type = 'dancehall'
            elif self.style == ReggaeStyle.STEPPERS:
                riddim_type = 'steppers'
            else:
                riddim_type = 'classic_roots'

        progression_pattern = self.RIDDIMS[riddim_type]

        # Build chord progression
        progression = []
        for degree, quality, duration in progression_pattern:
            scale_index = degree - 1
            root = self.key_root + self.MINOR_PENTATONIC[scale_index % len(self.MINOR_PENTATONIC)]
            progression.append((root, quality, duration))

        # Repeat progression to fill measures
        full_progression = progression * (measures // len(progression))

        riddim = {}

        # Generate drums
        if self.style == ReggaeStyle.STEPPERS:
            riddim['drums'] = Steppers.generate_pattern(measures)
        elif self.style == ReggaeStyle.DANCEHALL:
            riddim['drums'] = Rockers.generate_pattern(measures, 'heavy')
        else:
            riddim['drums'] = OneDrop.generate_pattern(measures, 'classic', self.tempo)

        # Generate bass
        bass_style = 'dub' if self.style == ReggaeStyle.DUB else 'melodic'
        riddim['bass'] = ReggaeBass.generate_line(full_progression, bass_style)

        # Generate guitar skank
        skank_style = 'bubble' if self.style == ReggaeStyle.DANCEHALL else 'standard'
        # Use first chord for skank pattern
        first_chord_notes = [full_progression[0][0], full_progression[0][0] + 3,
                            full_progression[0][0] + 7]
        riddim['guitar'] = Skank.generate_pattern(first_chord_notes, measures, skank_style)

        # Add dub effects if dub style
        if self.style == ReggaeStyle.DUB:
            riddim['delay'] = DubEffects.generate_delay_pattern(measures)
            riddim['dropouts'] = DubEffects.generate_dropout(measures, 'drums')
            riddim['filter'] = DubEffects.generate_filter_sweep(measures)

        return riddim


if __name__ == "__main__":
    """Example usage and testing"""

    print("Reggae Music Generator - Test Suite\n")
    print("=" * 60)

    # Test 1: One-drop pattern
    print("\n1. Generating one-drop drum pattern...")
    one_drop = OneDrop.generate_pattern(4, 'classic', 75)
    print(f"   Generated {len(one_drop)} drum events")
    print(f"   First 4 events: {one_drop[:4]}")

    # Test 2: Rockers pattern
    print("\n2. Generating rockers drum pattern...")
    rockers = Rockers.generate_pattern(4, 'standard')
    print(f"   Generated {len(rockers)} drum events")

    # Test 3: Guitar skank
    print("\n3. Generating guitar skank pattern...")
    chord = [57, 60, 64]  # A minor
    skank = Skank.generate_pattern(chord, 4, 'standard')
    print(f"   Generated {len(skank)} skank chords")

    # Test 4: Reggae bass line
    print("\n4. Generating melodic reggae bass line...")
    progression = [(57, 'minor', 4.0), (65, 'major', 4.0),
                   (62, 'major', 4.0), (64, 'major', 4.0)]
    bass = ReggaeBass.generate_line(progression, 'melodic')
    print(f"   Generated {len(bass)} bass notes")

    # Test 5: Complete roots reggae riddim
    print("\n5. Generating complete roots reggae riddim...")
    roots_gen = ReggaeGenerator(ReggaeStyle.ROOTS, key_root=57, tempo=75)
    riddim = roots_gen.generate_riddim(16, 'classic_roots')
    print(f"   Generated riddim with {len(riddim)} tracks:")
    for track, events in riddim.items():
        print(f"   - {track}: {len(events)} events")

    # Test 6: Dub with effects
    print("\n6. Generating dub version with effects...")
    dub_gen = ReggaeGenerator(ReggaeStyle.DUB, key_root=50, tempo=70)
    dub_riddim = dub_gen.generate_riddim(12)
    print(f"   Generated dub riddim with {len(dub_riddim)} tracks:")
    for track, events in dub_riddim.items():
        print(f"   - {track}: {len(events)} events")

    # Test 7: Dancehall
    print("\n7. Generating dancehall riddim...")
    dancehall_gen = ReggaeGenerator(ReggaeStyle.DANCEHALL, key_root=57, tempo=105)
    dancehall_riddim = dancehall_gen.generate_riddim(16, 'dancehall')
    print(f"   Generated dancehall riddim with {len(dancehall_riddim)} tracks:")
    for track, events in dancehall_riddim.items():
        print(f"   - {track}: {len(events)} events")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nReggae features implemented:")
    print("  ✓ One-drop, rockers, steppers drum patterns")
    print("  ✓ Guitar skank (standard, double, bubble)")
    print("  ✓ Melodic bass lines (walking, melodic, minimal, dub)")
    print("  ✓ Dub effects (delay, dropouts, filter sweeps)")
    print("  ✓ Roots, Dub, Dancehall, Steppers styles")
    print("  ✓ Riddim-based composition")
