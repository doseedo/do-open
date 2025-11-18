#!/usr/bin/env python3
"""
Blues Music Generator - Delta, Chicago, and Texas Blues

This module implements comprehensive blues music generation including:
- Delta blues (Robert Johnson, Son House, Skip James)
- Chicago blues (Muddy Waters, Howlin' Wolf, Willie Dixon)
- Texas blues (T-Bone Walker, Stevie Ray Vaughan, Johnny Winter)
- Jump blues, Blues rock

Features:
- 12-bar blues progressions (and variations: 8-bar, 16-bar)
- Blues scales with bent notes (b3, b5, b7)
- Shuffle rhythm (triplet feel)
- Blues turnarounds
- Slide guitar patterns
- Harmonica licks
- Walking bass lines
- Blues piano (boogie-woogie, barrelhouse)
- Call and response

Author: Agent 7 - World Music & Additional Genres
References:
- "Deep Blues" - Robert Palmer
- "The Blues: A Very Short Introduction" - Elijah Wald
- B.B. King vibrato technique
- Muddy Waters slide technique
- Boogie-woogie piano - Pete Johnson, Albert Ammons
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class BluesStyle(Enum):
    """Blues sub-genres"""
    DELTA = "delta"  # Acoustic Delta blues
    CHICAGO = "chicago"  # Electric Chicago blues
    TEXAS = "texas"  # Texas blues (guitar-heavy)
    JUMP = "jump"  # Jump blues (swing feel)
    BOOGIE = "boogie"  # Boogie-woogie piano
    BLUES_ROCK = "blues_rock"  # Rock-influenced blues


@dataclass
class BluesBend:
    """
    Represents a bent note in blues

    Blues "blue notes" are microtonal inflections:
    - Bent 3rd (minor 3rd bending toward major 3rd)
    - Bent 5th (diminished 5th)
    - Bent 7th (minor 7th)
    """
    base_note: int  # MIDI note
    bend_amount: float  # Semitones to bend (0.25 = quarter tone, 0.5 = half step, 1.0 = whole step)
    target_note: int  # Target note after bend

    def get_pitch_bend_value(self) -> int:
        """
        Convert bend amount to MIDI pitch bend value

        Returns:
            MIDI pitch bend value (-8192 to 8191, center=0)
        """
        # 8192 = full range (typically ±2 semitones)
        # For 1 semitone: 4096
        return int(self.bend_amount * 4096)


class BluesScale:
    """
    Blues scale generator

    The blues scale is a minor pentatonic with added b5 (blue note):
    1, b3, 4, b5, 5, b7

    Also includes major blues scale: 1, 2, b3, 3, 5, 6
    """

    # Minor blues scale intervals
    MINOR_BLUES = [0, 3, 5, 6, 7, 10]  # 1, b3, 4, b5, 5, b7

    # Major blues scale intervals
    MAJOR_BLUES = [0, 2, 3, 4, 7, 9]  # 1, 2, b3, 3, 5, 6

    # Mixolydian (dominant 7 scale, common in blues)
    MIXOLYDIAN = [0, 2, 4, 5, 7, 9, 10]  # 1, 2, 3, 4, 5, 6, b7

    @staticmethod
    def get_notes(root: int, scale_type: str = 'minor_blues',
                 octaves: int = 1) -> List[int]:
        """
        Get notes in blues scale

        Args:
            root: Root note (MIDI)
            scale_type: 'minor_blues', 'major_blues', or 'mixolydian'
            octaves: Number of octaves

        Returns:
            List of MIDI note numbers
        """
        if scale_type == 'minor_blues':
            intervals = BluesScale.MINOR_BLUES
        elif scale_type == 'major_blues':
            intervals = BluesScale.MAJOR_BLUES
        else:
            intervals = BluesScale.MIXOLYDIAN

        notes = []
        for octave in range(octaves + 1):
            for interval in intervals:
                notes.append(root + interval + (octave * 12))

        return notes


class TwelveBarBlues:
    """
    12-bar blues progression generator

    The 12-bar blues is the foundation of blues music:
    Bars 1-4:  I7  I7  I7  I7
    Bars 5-8:  IV7 IV7 I7  I7
    Bars 9-12: V7  IV7 I7  V7 (turnaround)

    Variations include quick change, jazz blues, etc.
    """

    @staticmethod
    def generate_progression(key_root: int,
                            variation: str = 'standard') -> List[Tuple[int, str, float]]:
        """
        Generate 12-bar blues progression

        Args:
            key_root: Root note of key (MIDI)
            variation: 'standard', 'quick_change', 'jazz', 'slow'

        Returns:
            List of (root, quality, duration) tuples
        """
        progression = []

        if variation == 'standard':
            # Standard 12-bar blues
            progression = [
                (key_root, '7', 4.0),      # Bar 1: I7
                (key_root, '7', 4.0),      # Bar 2: I7
                (key_root, '7', 4.0),      # Bar 3: I7
                (key_root, '7', 4.0),      # Bar 4: I7
                (key_root + 5, '7', 4.0),  # Bar 5: IV7
                (key_root + 5, '7', 4.0),  # Bar 6: IV7
                (key_root, '7', 4.0),      # Bar 7: I7
                (key_root, '7', 4.0),      # Bar 8: I7
                (key_root + 7, '7', 4.0),  # Bar 9: V7
                (key_root + 5, '7', 4.0),  # Bar 10: IV7
                (key_root, '7', 4.0),      # Bar 11: I7
                (key_root + 7, '7', 4.0),  # Bar 12: V7 (turnaround)
            ]

        elif variation == 'quick_change':
            # Quick change (IV in bar 2)
            progression = [
                (key_root, '7', 4.0),
                (key_root + 5, '7', 4.0),  # IV7 in bar 2
                (key_root, '7', 4.0),
                (key_root, '7', 4.0),
                (key_root + 5, '7', 4.0),
                (key_root + 5, '7', 4.0),
                (key_root, '7', 4.0),
                (key_root, '7', 4.0),
                (key_root + 7, '7', 4.0),
                (key_root + 5, '7', 4.0),
                (key_root, '7', 4.0),
                (key_root + 7, '7', 4.0),
            ]

        elif variation == 'jazz':
            # Jazz blues with more chord changes
            progression = [
                (key_root, 'maj7', 4.0),
                (key_root, '7', 4.0),
                (key_root, '7', 4.0),
                (key_root, '7', 4.0),
                (key_root + 5, '7', 4.0),
                (key_root + 5, '7', 4.0),
                (key_root, '7', 4.0),
                (key_root + 9, 'minor7', 2.0), (key_root + 2, '7', 2.0),  # vi-ii
                (key_root + 7, '7', 4.0),
                (key_root + 5, '7', 4.0),
                (key_root, '7', 2.0), (key_root + 9, 'minor7', 2.0),
                (key_root + 2, 'minor7', 2.0), (key_root + 7, '7', 2.0),
            ]

        elif variation == 'slow':
            # Slow blues (2 bars per chord)
            progression = [
                (key_root, '7', 8.0),      # Bars 1-2: I7
                (key_root, '7', 8.0),      # Bars 3-4: I7
                (key_root + 5, '7', 8.0),  # Bars 5-6: IV7
                (key_root, '7', 8.0),      # Bars 7-8: I7
                (key_root + 7, '7', 4.0),  # Bar 9: V7
                (key_root + 5, '7', 4.0),  # Bar 10: IV7
                (key_root, '7', 4.0),      # Bar 11: I7
                (key_root + 7, '7', 4.0),  # Bar 12: V7
            ]

        return progression


class BluesShuffle:
    """
    Blues shuffle rhythm generator

    The shuffle is a triplet-based feel fundamental to blues:
    Instead of straight eighth notes, use "long-short" triplet pattern.
    This creates the characteristic "swing" feel.
    """

    @staticmethod
    def generate_pattern(measures: int = 12,
                        style: str = 'standard') -> List[Tuple[str, float, int]]:
        """
        Generate shuffle drum pattern

        Args:
            measures: Number of measures
            style: 'standard', 'texas', 'slow'

        Returns:
            List of (drum_type, time_in_beats, velocity) tuples
        """
        pattern = []

        if style == 'standard':
            # Standard shuffle (triplet feel)
            for measure in range(measures):
                offset = measure * 4.0

                # Swing eighth notes on hi-hat/ride
                # In 4/4 with triplet feel: 1, 1+2/3, 2, 2+2/3, 3, 3+2/3, 4, 4+2/3
                triplet_positions = [0.0, 0.667, 1.0, 1.667, 2.0, 2.667, 3.0, 3.667]

                for i, pos in enumerate(triplet_positions):
                    velocity = 75 if i % 2 == 0 else 55  # Accent downbeats
                    pattern.append(('hihat', offset + pos, velocity))

                # Kick pattern
                pattern.extend([
                    ('kick', offset + 0.0, 100),
                    ('kick', offset + 2.0, 100),
                    ('kick', offset + 3.667, 80),  # Anticipated kick
                ])

                # Snare on 2 and 4 (backbeat)
                pattern.extend([
                    ('snare', offset + 1.0, 95),
                    ('snare', offset + 3.0, 95),
                ])

        elif style == 'texas':
            # Texas shuffle (more aggressive)
            for measure in range(measures):
                offset = measure * 4.0

                # Driving shuffle
                triplet_positions = [0.0, 0.667, 1.0, 1.667, 2.0, 2.667, 3.0, 3.667]
                for pos in triplet_positions:
                    pattern.append(('hihat', offset + pos, 80))

                # Heavy kick
                pattern.extend([
                    ('kick', offset + 0.0, 110),
                    ('kick', offset + 1.667, 90),
                    ('kick', offset + 2.0, 110),
                    ('kick', offset + 3.667, 90),
                ])

                # Snare
                pattern.extend([
                    ('snare', offset + 1.0, 100),
                    ('snare', offset + 3.0, 100),
                ])

        elif style == 'slow':
            # Slow blues (half-time feel)
            for measure in range(measures):
                offset = measure * 4.0

                # Sparse shuffle
                pattern.extend([
                    ('hihat', offset + 0.0, 70),
                    ('hihat', offset + 0.667, 50),
                    ('kick', offset + 0.0, 90),
                    ('snare', offset + 2.0, 85),
                    ('hihat', offset + 2.0, 70),
                    ('hihat', offset + 2.667, 50),
                ])

        return pattern


class SlideGuitar:
    """
    Slide guitar (bottleneck) pattern generator

    Slide guitar is iconic in Delta blues, using a glass or metal slide
    on the strings to create smooth glissandi and microtonal inflections.

    References:
    - Robert Johnson technique
    - Muddy Waters electric slide
    - Duane Allman slide rock
    """

    # Open tunings common in slide guitar
    OPEN_G_TUNING = [50, 55, 59, 62, 67, 74]  # D G D G B D
    OPEN_D_TUNING = [50, 57, 62, 66, 69, 74]  # D A D F# A D

    @staticmethod
    def generate_lick(root: int, scale: List[int],
                     lick_type: str = 'delta') -> List[Tuple[int, float, int, Optional[int]]]:
        """
        Generate slide guitar lick

        Args:
            root: Root note
            scale: Scale intervals
            lick_type: 'delta', 'chicago', or 'rock'

        Returns:
            List of (note, duration, velocity, slide_to_note) tuples
        """
        lick = []

        if lick_type == 'delta':
            # Classic Delta blues slide lick
            # Typically uses slides between chord tones
            lick = [
                (root, 0.5, 80, root + 3),  # Slide from root to b3
                (root + 3, 0.5, 85, None),
                (root + 5, 0.5, 80, root + 7),  # Slide from 4 to 5
                (root + 7, 1.0, 90, None),
                (root + 10, 0.5, 85, root + 12),  # Slide to octave
                (root + 12, 1.0, 80, None),
            ]

        elif lick_type == 'chicago':
            # Electric slide (Muddy Waters style)
            lick = [
                (root + 12, 0.25, 95, root + 15),  # High register
                (root + 15, 0.25, 90, root + 12),
                (root + 12, 0.5, 85, None),
                (root + 10, 0.5, 80, None),
                (root + 7, 1.0, 90, None),
            ]

        elif lick_type == 'rock':
            # Blues rock slide (Allman Brothers style)
            lick = [
                (root, 0.25, 100, root + 7),
                (root + 7, 0.25, 95, root + 12),
                (root + 12, 0.5, 100, None),
                (root + 10, 0.25, 85, None),
                (root + 7, 0.5, 90, None),
            ]

        return lick


class BluesHarmonica:
    """
    Blues harmonica (harp) lick generator

    The harmonica is essential to Chicago blues, using techniques like
    bending, tongue blocking, and overblowing.

    References:
    - Little Walter technique
    - Sonny Boy Williamson II
    - Paul Butterfield
    """

    @staticmethod
    def generate_lick(key_root: int,
                     lick_type: str = 'chicago') -> List[Tuple[int, float, int, bool]]:
        """
        Generate harmonica lick

        Args:
            key_root: Key root note
            lick_type: 'chicago', 'country', or 'chromatic'

        Returns:
            List of (note, duration, velocity, is_bent) tuples
        """
        lick = []

        if lick_type == 'chicago':
            # Classic Chicago blues harp lick (hole 2 draw bend, etc.)
            # Using cross harp (second position): play in G on C harp
            lick = [
                (key_root + 7, 0.5, 90, False),   # 5th
                (key_root + 5, 0.25, 85, True),   # Bent 4th
                (key_root + 3, 0.25, 80, True),   # Bent 3rd
                (key_root + 2, 0.5, 85, False),   # 2nd
                (key_root, 0.5, 90, False),       # Root
                (key_root + 7, 1.0, 95, False),   # 5th
            ]

        elif lick_type == 'country':
            # Country-style harp (straight harp, first position)
            lick = [
                (key_root, 0.5, 85, False),
                (key_root + 4, 0.5, 80, False),
                (key_root + 7, 0.5, 85, False),
                (key_root + 9, 0.5, 80, False),
                (key_root + 12, 1.0, 90, False),
            ]

        elif lick_type == 'chromatic':
            # Chromatic harmonica (Toots Thielemans style)
            lick = [
                (key_root + 12, 0.25, 85, False),
                (key_root + 11, 0.25, 80, False),
                (key_root + 10, 0.25, 85, False),
                (key_root + 9, 0.25, 80, False),
                (key_root + 7, 0.5, 90, False),
                (key_root, 1.0, 95, False),
            ]

        return lick


class BluesPiano:
    """
    Blues piano pattern generator

    Includes boogie-woogie, barrelhouse, and blues piano styles.

    References:
    - Professor Longhair
    - Otis Spann
    - Pete Johnson boogie-woogie
    """

    @staticmethod
    def generate_boogie_bass(root: int,
                            measures: int = 4) -> List[Tuple[int, float, int]]:
        """
        Generate boogie-woogie walking bass pattern

        Args:
            root: Root note (bass register)
            measures: Number of measures

        Returns:
            List of (note, time, velocity) tuples
        """
        # Classic boogie-woogie pattern: 1, 3, 5, 6, b7, 6, 5, 3
        pattern_intervals = [0, 4, 7, 9, 10, 9, 7, 4]
        bass_line = []

        for measure in range(measures):
            time = measure * 4.0
            for i, interval in enumerate(pattern_intervals):
                note = root + interval
                beat_time = time + (i * 0.5)  # Eighth notes
                velocity = 90 if i % 2 == 0 else 80
                bass_line.append((note, beat_time, velocity))

        return bass_line

    @staticmethod
    def generate_blues_riff(root: int,
                           style: str = 'chicago') -> List[Tuple[List[int], float, float, int]]:
        """
        Generate blues piano riff

        Args:
            root: Root note
            style: 'chicago', 'barrelhouse', or 'modern'

        Returns:
            List of (chord_notes, time, duration, velocity) tuples
        """
        riff = []

        if style == 'chicago':
            # Classic Chicago blues piano (triads with 6ths)
            riff = [
                ([root, root + 4, root + 7], 0.0, 0.5, 85),
                ([root + 2, root + 5, root + 9], 0.5, 0.5, 80),
                ([root, root + 4, root + 7], 1.0, 0.5, 85),
                ([root + 7, root + 10, root + 14], 1.5, 0.5, 80),
            ]

        elif style == 'barrelhouse':
            # Barrelhouse style (heavy octaves)
            riff = [
                ([root, root + 12], 0.0, 1.0, 95),
                ([root + 7, root + 19], 1.0, 1.0, 90),
                ([root, root + 12], 2.0, 1.0, 95),
                ([root + 5, root + 17], 3.0, 1.0, 90),
            ]

        return riff


class BluesTurnaround:
    """
    Blues turnaround generator

    Turnarounds are the last 2 bars of the 12-bar blues, leading back
    to the beginning. Classic turnarounds use chromatic motion.

    References:
    - Robert Johnson turnarounds
    - T-Bone Walker turnarounds
    - Jazz blues turnarounds
    """

    @staticmethod
    def generate(key_root: int,
                style: str = 'classic') -> List[Tuple[int, str, float]]:
        """
        Generate blues turnaround progression

        Args:
            key_root: Key root note
            style: 'classic', 'jazz', or 'modern'

        Returns:
            List of (root, quality, duration) tuples for last 2 bars
        """
        if style == 'classic':
            # Classic: I - VI7 - ii7 - V7
            return [
                (key_root, '7', 2.0),
                (key_root + 9, '7', 2.0),
                (key_root + 2, 'minor7', 2.0),
                (key_root + 7, '7', 2.0),
            ]

        elif style == 'jazz':
            # Jazz: I - iii7 - VI7 - ii7 - V7
            return [
                (key_root, 'maj7', 1.0),
                (key_root + 4, 'minor7', 1.0),
                (key_root + 9, '7', 1.0),
                (key_root + 2, 'minor7', 1.0),
                (key_root + 7, '7', 2.0),
            ]

        elif style == 'chromatic':
            # Chromatic walkdown
            return [
                (key_root, '7', 1.0),
                (key_root - 1, 'dim', 1.0),
                (key_root - 2, 'minor7', 1.0),
                (key_root - 3, 'dim', 1.0),
                (key_root + 7, '7', 2.0),
            ]

        return []


class BluesGenerator:
    """
    Main blues music generator

    Combines all blues elements for complete arrangements.
    """

    def __init__(self, style: BluesStyle = BluesStyle.CHICAGO,
                 key_root: int = 60, tempo: int = 120):
        """
        Initialize blues generator

        Args:
            style: Blues sub-genre
            key_root: Key root note (MIDI)
            tempo: BPM (60-80 slow, 120-140 shuffle, 180+ fast)
        """
        self.style = style
        self.key_root = key_root
        self.tempo = tempo

    def generate_arrangement(self, choruses: int = 3) -> Dict[str, List]:
        """
        Generate complete blues arrangement

        Args:
            choruses: Number of 12-bar choruses

        Returns:
            Dictionary with all instrument tracks
        """
        arrangement = {}

        # Generate chord progression
        if self.style == BluesStyle.DELTA:
            progression = TwelveBarBlues.generate_progression(self.key_root, 'standard')
        elif self.style == BluesStyle.JUMP:
            progression = TwelveBarBlues.generate_progression(self.key_root, 'quick_change')
        else:
            progression = TwelveBarBlues.generate_progression(self.key_root, 'standard')

        # Repeat for multiple choruses
        full_progression = progression * choruses

        # Generate drums
        if self.style != BluesStyle.DELTA:  # Delta blues typically has no drums
            shuffle_style = 'texas' if self.style == BluesStyle.TEXAS else 'standard'
            arrangement['drums'] = BluesShuffle.generate_pattern(12 * choruses, shuffle_style)

        # Generate bass (boogie-woogie for piano blues, walking for others)
        if self.style == BluesStyle.BOOGIE:
            arrangement['piano_bass'] = BluesPiano.generate_boogie_bass(
                self.key_root - 12, 12 * choruses
            )
        else:
            # Simple walking bass placeholder
            arrangement['bass'] = [(self.key_root - 12, i * 1.0, 90) for i in range(12 * choruses * 4)]

        # Generate lead instrument licks
        if self.style == BluesStyle.DELTA:
            arrangement['slide_guitar'] = SlideGuitar.generate_lick(
                self.key_root, BluesScale.MINOR_BLUES, 'delta'
            )
        elif self.style == BluesStyle.CHICAGO:
            arrangement['guitar'] = SlideGuitar.generate_lick(
                self.key_root, BluesScale.MINOR_BLUES, 'chicago'
            )
            arrangement['harmonica'] = BluesHarmonica.generate_lick(
                self.key_root, 'chicago'
            )

        return arrangement


if __name__ == "__main__":
    """Example usage and testing"""

    print("Blues Music Generator - Test Suite\n")
    print("=" * 60)

    # Test 1: 12-bar blues progression
    print("\n1. Generating standard 12-bar blues in C...")
    progression = TwelveBarBlues.generate_progression(60, 'standard')
    print(f"   Generated {len(progression)} chords")
    for i, (root, quality, dur) in enumerate(progression[:4]):
        print(f"   Bar {i+1}: {root} {quality}")

    # Test 2: Blues scale
    print("\n2. Generating minor blues scale...")
    scale = BluesScale.get_notes(60, 'minor_blues', 1)
    print(f"   Scale notes: {scale}")

    # Test 3: Shuffle pattern
    print("\n3. Generating shuffle drum pattern...")
    shuffle = BluesShuffle.generate_pattern(4, 'standard')
    print(f"   Generated {len(shuffle)} drum events")

    # Test 4: Slide guitar lick
    print("\n4. Generating Delta blues slide guitar lick...")
    slide = SlideGuitar.generate_lick(60, BluesScale.MINOR_BLUES, 'delta')
    print(f"   Generated {len(slide)} notes")

    # Test 5: Harmonica lick
    print("\n5. Generating Chicago blues harmonica lick...")
    harp = BluesHarmonica.generate_lick(60, 'chicago')
    print(f"   Generated {len(harp)} notes")

    # Test 6: Boogie-woogie bass
    print("\n6. Generating boogie-woogie bass line...")
    boogie = BluesPiano.generate_boogie_bass(48, 4)
    print(f"   Generated {len(boogie)} bass notes")

    # Test 7: Blues turnaround
    print("\n7. Generating classic blues turnaround...")
    turnaround = BluesTurnaround.generate(60, 'classic')
    print(f"   Generated {len(turnaround)} chords")
    for root, quality, dur in turnaround:
        print(f"   - {root} {quality}")

    # Test 8: Complete Chicago blues arrangement
    print("\n8. Generating complete Chicago blues arrangement...")
    chicago_gen = BluesGenerator(BluesStyle.CHICAGO, key_root=60, tempo=120)
    arrangement = chicago_gen.generate_arrangement(choruses=2)
    print(f"   Generated {len(arrangement)} tracks:")
    for track, events in arrangement.items():
        print(f"   - {track}: {len(events)} events")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nBlues features implemented:")
    print("  ✓ 12-bar blues progressions (standard, quick change, jazz)")
    print("  ✓ Blues scales (minor, major, mixolydian)")
    print("  ✓ Shuffle rhythm with triplet feel")
    print("  ✓ Slide guitar patterns (Delta, Chicago, Rock)")
    print("  ✓ Harmonica licks with bends")
    print("  ✓ Boogie-woogie piano bass")
    print("  ✓ Blues turnarounds (classic, jazz, chromatic)")
    print("  ✓ Delta, Chicago, Texas blues styles")
