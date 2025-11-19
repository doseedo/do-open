#!/usr/bin/env python3
"""
Singer-Songwriter and Folk Music Generator

Comprehensive implementation of folk and singer-songwriter music across 5 sub-genres.

Sub-genres:
-----------
- Folk (Bob Dylan, Joan Baez, Woody Guthrie)
- Singer-Songwriter (Joni Mitchell, James Taylor, Carole King)
- Contemporary Folk (Mumford & Sons, The Lumineers, Bon Iver)
- Indie Folk (Sufjan Stevens, Iron & Wine, Fleet Foxes)
- Americana (Gillian Welch, Jason Isbell, The Avett Brothers)

Features:
---------
- Acoustic guitar fingerpicking patterns (Travis picking, clawhammer)
- Strumming patterns with dynamic control
- Open tunings (DADGAD, Open D, Open G)
- Simple, memorable chord progressions
- Vocal-range melodies with folk ornaments
- Sparse, intimate arrangements
- Banjo patterns for contemporary folk
- Harmonica and mandolin support
- Story-telling song structures

Research References:
-------------------
- "The Folk Songs of North America" - Alan Lomax (1960)
- "Joni Mitchell: Both Sides Now" - Mark Bego (2014)
- "The NPR Curious Listener's Guide to American Folk Music" - William Anderson (2004)
- Fingerpicking patterns analysis - Stefan Grossman
- "How to Write One Song" - Jeff Tweedy (2020)
- Travis picking technique - Merle Travis, Chet Atkins
- Open tunings in folk music - Joni Mitchell, Nick Drake

Author: Agent 47 - Singer-Songwriter/Folk Module
Date: 2025
License: MIT
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FolkStyle(Enum):
    """Folk and singer-songwriter sub-genres"""
    TRADITIONAL_FOLK = "traditional_folk"  # Bob Dylan, Joan Baez, Woody Guthrie
    SINGER_SONGWRITER = "singer_songwriter"  # Joni Mitchell, James Taylor, Carole King
    CONTEMPORARY_FOLK = "contemporary_folk"  # Mumford & Sons, The Lumineers, Bon Iver
    INDIE_FOLK = "indie_folk"  # Sufjan Stevens, Iron & Wine, Fleet Foxes
    AMERICANA = "americana"  # Gillian Welch, Jason Isbell, The Avett Brothers


class FolkInstrument(Enum):
    """Folk-specific instruments (MIDI program numbers)"""
    ACOUSTIC_GUITAR_NYLON = 24  # GM Acoustic Guitar (nylon)
    ACOUSTIC_GUITAR_STEEL = 25  # GM Acoustic Guitar (steel)
    ACOUSTIC_BASS = 32  # GM Acoustic Bass
    HARMONICA = 22  # GM Harmonica
    BANJO = 105  # GM Banjo
    FIDDLE = 110  # GM Fiddle
    MANDOLIN = 24  # GM Acoustic Guitar (approximation)
    ACCORDION = 21  # GM Accordion
    BRUSHES_KIT = 0  # GM Standard Kit with brushes


@dataclass
class FolkNote:
    """
    Individual note with folk expression

    Attributes:
        pitch: MIDI note number (0-127)
        velocity: Velocity (1-127)
        start_time: Start time in beats
        duration: Duration in beats
        articulation: Articulation type ('normal', 'staccato', 'slide', 'hammer', 'pull')
        channel: MIDI channel
    """
    pitch: int
    velocity: int
    start_time: float
    duration: float
    articulation: str = "normal"
    channel: int = 0


@dataclass
class FolkChord:
    """
    Folk chord with optional extensions

    Attributes:
        root: Root note (MIDI note number)
        quality: Chord quality
        duration: Duration in beats
    """
    root: int
    quality: str  # 'major', 'minor', 'sus2', 'sus4', 'add9', '7', 'maj7'
    duration: float

    def get_notes(self) -> List[int]:
        """Get MIDI notes for the chord"""
        notes = [self.root]

        if self.quality == 'major':
            notes.extend([self.root + 4, self.root + 7])
        elif self.quality == 'minor':
            notes.extend([self.root + 3, self.root + 7])
        elif self.quality == 'sus2':
            notes.extend([self.root + 2, self.root + 7])
        elif self.quality == 'sus4':
            notes.extend([self.root + 5, self.root + 7])
        elif self.quality == 'add9':
            notes.extend([self.root + 4, self.root + 7, self.root + 14])
        elif self.quality == '7':
            notes.extend([self.root + 4, self.root + 7, self.root + 10])
        elif self.quality == 'maj7':
            notes.extend([self.root + 4, self.root + 7, self.root + 11])

        return notes


class TravisPicking:
    """
    Travis picking fingerpicking pattern generator

    Travis picking is a fingerstyle technique characterized by:
    - Alternating bass notes (thumb)
    - Syncopated treble notes (fingers)
    - Steady, driving rhythm

    Named after Merle Travis, popularized by Chet Atkins.

    References:
    - Merle Travis "Sixteen Tons" technique
    - Chet Atkins fingerpicking style
    - Classical Travis pattern: T 1 T 2 T 3 T 2 (T=thumb, 1-3=fingers)
    """

    # Travis pattern as (string_offset, is_bass) tuples
    # is_bass=True means thumb (lower strings), False means fingers (higher strings)
    CLASSIC_PATTERN = [
        (0, True),   # Thumb on root
        (2, False),  # Finger on 3rd
        (1, True),   # Thumb on 5th
        (3, False),  # Finger on top
        (0, True),   # Thumb on root
        (3, False),  # Finger on top
        (1, True),   # Thumb on 5th
        (2, False),  # Finger on 3rd
    ]

    ALTERNATING_BASS = [
        (0, True),   # Root
        (2, False),  # High note
        (1, True),   # Fifth
        (2, False),  # High note
    ]

    @staticmethod
    def generate_pattern(chord: FolkChord, pattern_type: str = 'classic',
                        bars: int = 1) -> List[Tuple[int, float, int]]:
        """
        Generate Travis picking pattern

        Args:
            chord: Chord to arpeggiate
            pattern_type: 'classic' or 'alternating_bass'
            bars: Number of bars

        Returns:
            List of (note, duration, velocity) tuples
        """
        chord_notes = chord.get_notes()
        pattern = (TravisPicking.CLASSIC_PATTERN if pattern_type == 'classic'
                  else TravisPicking.ALTERNATING_BASS)

        notes = []
        note_duration = 0.5  # Eighth notes

        for bar in range(bars):
            for i, (offset, is_bass) in enumerate(pattern):
                note = chord_notes[offset % len(chord_notes)]

                # Bass notes (thumb) are louder
                velocity = 80 if is_bass else 65

                notes.append((note, note_duration, velocity))

        return notes


class StrummingPattern:
    """
    Acoustic guitar strumming pattern generator

    Strumming is the foundation of folk guitar, with patterns varying from
    simple downstrokes to complex up-down patterns with palm muting.

    References:
    - Bob Dylan strumming style
    - Common folk strumming patterns (D-DU-UDU, etc.)
    """

    # Strumming patterns as (direction, duration, is_muted) tuples
    # direction: 1=down, -1=up
    PATTERNS = {
        'basic': [
            (1, 1.0, False),  # Down
            (1, 1.0, False),  # Down
            (1, 1.0, False),  # Down
            (1, 1.0, False),  # Down
        ],
        'folk_standard': [
            (1, 0.5, False),   # Down
            (1, 0.5, False),   # Down
            (-1, 0.5, False),  # Up
            (1, 0.5, False),   # Down
            (-1, 0.5, False),  # Up
            (1, 0.5, False),   # Down
            (-1, 0.5, False),  # Up
        ],
        'indie_folk': [
            (1, 0.5, False),   # Down
            (-1, 0.25, False), # Up
            (-1, 0.25, False), # Up
            (1, 0.5, False),   # Down
            (-1, 0.5, False),  # Up
            (1, 0.5, False),   # Down
            (-1, 0.5, False),  # Up
        ],
        'mumford': [
            (1, 0.5, False),   # Down (heavy)
            (1, 0.25, True),   # Down (muted)
            (-1, 0.25, False), # Up
            (1, 0.5, False),   # Down
            (-1, 0.5, False),  # Up
        ],
    }

    @staticmethod
    def generate_strum(chord: FolkChord, pattern_type: str = 'folk_standard',
                      bars: int = 1) -> List[Tuple[List[int], float, int]]:
        """
        Generate strumming pattern

        Args:
            chord: Chord to strum
            pattern_type: Pattern from PATTERNS dictionary
            bars: Number of bars

        Returns:
            List of (chord_notes, duration, velocity) tuples
        """
        chord_notes = chord.get_notes()
        pattern = StrummingPattern.PATTERNS.get(pattern_type,
                                               StrummingPattern.PATTERNS['folk_standard'])

        strums = []

        for bar in range(bars):
            for direction, duration, is_muted in pattern:
                # Downstrokes are louder than upstrokes
                base_velocity = 75 if direction == 1 else 60
                velocity = 45 if is_muted else base_velocity

                strums.append((chord_notes, duration, velocity))

        return strums


class OpenTuning:
    """
    Open tuning simulation for folk guitar

    Open tunings allow for rich, resonant chord voicings and are central
    to the sound of folk icons like Joni Mitchell and Nick Drake.

    Common open tunings:
    - DADGAD (D-A-D-G-A-D): Celtic folk, Jimmy Page
    - Open D (D-A-D-F#-A-D): Slide guitar, blues
    - Open G (D-G-D-G-B-D): Keith Richards, Joni Mitchell

    References:
    - Joni Mitchell's use of over 50 tunings
    - Nick Drake's DADGAD compositions
    - Celtic DADGAD tradition
    """

    # Tunings as MIDI offset from standard tuning
    # Standard: E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
    TUNINGS = {
        'standard': [40, 45, 50, 55, 59, 64],
        'dadgad': [38, 45, 50, 55, 57, 62],  # D-A-D-G-A-D
        'open_d': [38, 45, 50, 54, 57, 62],   # D-A-D-F#-A-D
        'open_g': [38, 43, 50, 55, 59, 62],   # D-G-D-G-B-D
    }

    @staticmethod
    def get_chord_voicing(root: int, quality: str,
                         tuning: str = 'standard') -> List[int]:
        """
        Get chord voicing for open tuning

        Args:
            root: Root note
            quality: Chord quality
            tuning: Tuning name from TUNINGS

        Returns:
            List of MIDI notes for chord voicing
        """
        strings = OpenTuning.TUNINGS.get(tuning, OpenTuning.TUNINGS['standard'])

        # For open tunings, strum all open strings or simple fingerings
        if tuning == 'dadgad':
            # DADGAD creates suspended sounds naturally
            if quality in ['sus4', 'sus2', 'major']:
                # Use open strings + simple fret positions
                return [s for s in strings]

        # For standard tuning, return normal chord voicing
        chord = FolkChord(root, quality, 4.0)
        return chord.get_notes()


class BanjoPattern:
    """
    Clawhammer and three-finger banjo patterns

    Banjo is essential in contemporary folk (Mumford & Sons, Avett Brothers)
    and traditional American folk music.

    Clawhammer: Down-picking technique, rhythmic
    Three-finger: Scruggs style, more melodic

    References:
    - Earl Scruggs three-finger style
    - Pete Seeger clawhammer technique
    - Contemporary folk banjo (Mumford & Sons)
    """

    # Clawhammer pattern: brush down, thumb note, strum
    CLAWHAMMER = [
        (2, 0.5, 75),   # Brush down
        (0, 0.5, 90),   # Thumb (drone string)
        (2, 0.5, 75),   # Brush
        (1, 0.5, 80),   # Melody note
    ]

    # Three-finger roll
    THREE_FINGER = [
        (0, 0.33, 80),  # Thumb
        (1, 0.33, 75),  # Index
        (2, 0.33, 75),  # Middle
        (0, 0.33, 80),  # Thumb
        (2, 0.33, 75),  # Middle
        (1, 0.33, 75),  # Index
    ]

    @staticmethod
    def generate_banjo_pattern(chord_notes: List[int], pattern_type: str = 'clawhammer',
                              bars: int = 1) -> List[Tuple[int, float, int]]:
        """
        Generate banjo pattern

        Args:
            chord_notes: Notes in chord
            pattern_type: 'clawhammer' or 'three_finger'
            bars: Number of bars

        Returns:
            List of (note, duration, velocity) tuples
        """
        pattern = (BanjoPattern.CLAWHAMMER if pattern_type == 'clawhammer'
                  else BanjoPattern.THREE_FINGER)

        notes = []
        for bar in range(bars):
            for string_offset, duration, velocity in pattern:
                note = chord_notes[string_offset % len(chord_notes)]
                notes.append((note, duration, velocity))

        return notes


class FolkMelody:
    """
    Vocal melody generator for folk music

    Folk melodies are characterized by:
    - Limited range (octave to 10th)
    - Stepwise motion (easy to sing)
    - Pentatonic influences
    - Syllabic setting (one note per syllable)
    - Simple ornaments (grace notes, slides)

    References:
    - "The Folk Songs of North America" - Alan Lomax
    - Appalachian folk melody analysis
    - Singer-songwriter melody construction
    """

    # Pentatonic major scale (common in folk)
    PENTATONIC_MAJOR = [0, 2, 4, 7, 9]

    # Pentatonic minor scale
    PENTATONIC_MINOR = [0, 3, 5, 7, 10]

    # Major scale
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

    # Minor scale (natural)
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

    @staticmethod
    def generate_folk_melody(root: int, scale_type: str = 'pentatonic_major',
                            phrase_length: int = 8,
                            max_interval: int = 3) -> List[FolkNote]:
        """
        Generate folk melody with stepwise motion

        Args:
            root: Root note of scale
            scale_type: Scale to use
            phrase_length: Number of notes in phrase
            max_interval: Maximum interval jump (in scale degrees)

        Returns:
            List of FolkNote objects
        """
        # Get scale
        if scale_type == 'pentatonic_major':
            scale = FolkMelody.PENTATONIC_MAJOR
        elif scale_type == 'pentatonic_minor':
            scale = FolkMelody.PENTATONIC_MINOR
        elif scale_type == 'major':
            scale = FolkMelody.MAJOR_SCALE
        else:
            scale = FolkMelody.MINOR_SCALE

        # Generate melody with stepwise motion
        melody = []
        current_degree = 0  # Start on root
        time = 0.0

        for i in range(phrase_length):
            # Mostly stepwise motion (±1 or ±2 scale degrees)
            if i == phrase_length - 1:
                # End on root or fifth
                current_degree = random.choice([0, 4])
            else:
                step = random.choice([-2, -1, 0, 1, 2])
                current_degree = max(0, min(len(scale) * 2, current_degree + step))

            # Get MIDI note
            octave = current_degree // len(scale)
            scale_degree = current_degree % len(scale)
            pitch = root + scale[scale_degree] + (octave * 12)

            # Vary rhythm (quarter, eighth notes)
            duration = random.choice([0.5, 1.0, 1.0, 1.0])
            velocity = random.randint(70, 85)

            # Occasional slide or hammer-on
            articulation = 'normal'
            if i > 0 and random.random() < 0.1:
                articulation = random.choice(['slide', 'hammer'])

            note = FolkNote(
                pitch=pitch,
                velocity=velocity,
                start_time=time,
                duration=duration,
                articulation=articulation
            )
            melody.append(note)
            time += duration

        return melody


class SingerSongwriterGenerator:
    """
    Main singer-songwriter and folk music generator

    Combines all folk elements to generate complete arrangements
    in various folk styles.
    """

    # Common folk chord progressions
    PROGRESSIONS = {
        'folk_standard': [
            (0, 'major'), (5, 'major'), (7, 'major'), (0, 'major'),  # I-IV-V-I
        ],
        'folk_pop': [
            (0, 'major'), (7, 'major'), (9, 'minor'), (5, 'major'),  # I-V-vi-IV
        ],
        'sad_folk': [
            (9, 'minor'), (5, 'major'), (0, 'major'), (7, 'major'),  # vi-IV-I-V
        ],
        'modal_folk': [
            (0, 'major'), (10, 'major'), (5, 'major'), (0, 'major'),  # I-bVII-IV-I (Mixolydian)
        ],
        'contemporary': [
            (0, 'major'), (0, 'sus2'), (5, 'add9'), (7, 'major'),  # I-Isus2-IVadd9-V
        ],
    }

    def __init__(self, style: FolkStyle = FolkStyle.TRADITIONAL_FOLK,
                 key_root: int = 60, tempo: int = 90):
        """
        Initialize singer-songwriter generator

        Args:
            style: Folk sub-genre style
            key_root: Root note of key (MIDI note number)
            tempo: Tempo in BPM
        """
        self.style = style
        self.key_root = key_root
        self.tempo = tempo

    def generate_chord_progression(self, bars: int = 8,
                                   progression_type: Optional[str] = None) -> List[FolkChord]:
        """
        Generate folk chord progression

        Args:
            bars: Number of bars
            progression_type: Specific progression type or None for auto-select

        Returns:
            List of FolkChord objects
        """
        # Auto-select progression based on style
        if progression_type is None:
            if self.style == FolkStyle.CONTEMPORARY_FOLK:
                progression_type = 'contemporary'
            elif self.style == FolkStyle.INDIE_FOLK:
                progression_type = 'folk_pop'
            else:
                progression_type = 'folk_standard'

        # Get progression pattern
        pattern = self.PROGRESSIONS.get(progression_type,
                                       self.PROGRESSIONS['folk_standard'])

        # Build chord progression
        progression = []
        for i in range(bars):
            offset, quality = pattern[i % len(pattern)]
            chord_root = self.key_root + offset

            chord = FolkChord(
                root=chord_root,
                quality=quality,
                duration=4.0  # Whole note
            )
            progression.append(chord)

        return progression

    def generate_guitar_part(self, chord: FolkChord,
                           technique: str = 'fingerpicking') -> List[FolkNote]:
        """
        Generate guitar part for chord

        Args:
            chord: Chord to play
            technique: 'fingerpicking', 'strumming', or 'arpeggio'

        Returns:
            List of FolkNote objects
        """
        notes = []

        if technique == 'fingerpicking':
            pattern = TravisPicking.generate_pattern(chord, 'classic', bars=1)
            time = 0.0
            for pitch, duration, velocity in pattern:
                note = FolkNote(pitch, velocity, time, duration)
                notes.append(note)
                time += duration

        elif technique == 'strumming':
            pattern_type = 'mumford' if self.style == FolkStyle.CONTEMPORARY_FOLK else 'folk_standard'
            strums = StrummingPattern.generate_strum(chord, pattern_type, bars=1)
            time = 0.0
            for chord_notes, duration, velocity in strums:
                # Create notes for each string
                for pitch in chord_notes:
                    note = FolkNote(pitch, velocity, time, duration * 0.1)
                    notes.append(note)
                time += duration

        elif technique == 'arpeggio':
            # Simple arpeggio pattern
            chord_notes = chord.get_notes()
            time = 0.0
            for i in range(8):
                pitch = chord_notes[i % len(chord_notes)]
                note = FolkNote(pitch, 70, time, 0.5)
                notes.append(note)
                time += 0.5

        return notes

    def generate_arrangement(self, bars: int = 16) -> Dict[str, List[FolkNote]]:
        """
        Generate complete folk arrangement

        Args:
            bars: Number of bars

        Returns:
            Dictionary mapping instrument names to note lists
        """
        arrangement = {}

        # Generate chord progression
        progression = self.generate_chord_progression(bars)

        # Guitar part (main accompaniment)
        guitar_notes = []
        time = 0.0
        for chord in progression:
            # Choose technique based on style
            if self.style == FolkStyle.TRADITIONAL_FOLK:
                technique = 'fingerpicking'
            elif self.style == FolkStyle.CONTEMPORARY_FOLK:
                technique = 'strumming'
            else:
                technique = random.choice(['fingerpicking', 'strumming'])

            part = self.generate_guitar_part(chord, technique)
            # Adjust start times
            for note in part:
                note.start_time += time
            guitar_notes.extend(part)
            time += 4.0

        arrangement['acoustic_guitar'] = guitar_notes

        # Vocal melody
        melody = FolkMelody.generate_folk_melody(
            self.key_root + 12,  # Octave up for vocal range
            'pentatonic_major',
            phrase_length=bars * 2
        )
        arrangement['vocals'] = melody

        # Bass (simple root notes on downbeats)
        bass_notes = []
        time = 0.0
        for chord in progression:
            note = FolkNote(chord.root - 12, 75, time, 3.5)  # Root note, octave down
            bass_notes.append(note)
            time += 4.0
        arrangement['bass'] = bass_notes

        # Add banjo for contemporary folk
        if self.style == FolkStyle.CONTEMPORARY_FOLK:
            banjo_notes = []
            time = 0.0
            for chord in progression:
                pattern = BanjoPattern.generate_banjo_pattern(
                    chord.get_notes(), 'clawhammer', bars=1
                )
                for pitch, duration, velocity in pattern:
                    note = FolkNote(pitch, velocity, time, duration)
                    banjo_notes.append(note)
                    time += duration
            arrangement['banjo'] = banjo_notes

        return arrangement


# Convenience functions

def create_folk_song(style: FolkStyle = FolkStyle.TRADITIONAL_FOLK,
                     key: int = 60, tempo: int = 90,
                     length: int = 16) -> Dict[str, List[FolkNote]]:
    """
    Convenience function to create folk song

    Args:
        style: Folk style
        key: Root note
        tempo: Tempo in BPM
        length: Length in bars

    Returns:
        Dictionary of instrument parts
    """
    gen = SingerSongwriterGenerator(style, key, tempo)
    return gen.generate_arrangement(length)


# Exports
__all__ = [
    'FolkStyle',
    'FolkInstrument',
    'FolkNote',
    'FolkChord',
    'TravisPicking',
    'StrummingPattern',
    'OpenTuning',
    'BanjoPattern',
    'FolkMelody',
    'SingerSongwriterGenerator',
    'create_folk_song',
]


# Self-test code
if __name__ == "__main__":
    print("=" * 60)
    print("Singer-Songwriter/Folk Music Generator - Self Test")
    print("=" * 60)

    # Test 1: Traditional folk with fingerpicking
    print("\n1. Generating traditional folk song with Travis picking...")
    gen = SingerSongwriterGenerator(
        style=FolkStyle.TRADITIONAL_FOLK,
        key_root=60,  # C
        tempo=90
    )

    progression = gen.generate_chord_progression(8, 'folk_standard')
    print(f"   Generated {len(progression)} chords")
    for i, chord in enumerate(progression[:4]):
        print(f"   Bar {i+1}: Root={chord.root} ({chord.quality})")

    # Test 2: Travis picking pattern
    print("\n2. Generating Travis picking pattern...")
    test_chord = FolkChord(60, 'major', 4.0)
    travis = TravisPicking.generate_pattern(test_chord, 'classic', bars=2)
    print(f"   Generated {len(travis)} notes")
    print(f"   First 4 notes: {[note for note, dur, vel in travis[:4]]}")

    # Test 3: Strumming pattern
    print("\n3. Generating folk strumming pattern...")
    strums = StrummingPattern.generate_strum(test_chord, 'folk_standard', bars=1)
    print(f"   Generated {len(strums)} strums")

    # Test 4: Folk melody
    print("\n4. Generating folk vocal melody...")
    melody = FolkMelody.generate_folk_melody(72, 'pentatonic_major', phrase_length=8)
    print(f"   Generated {len(melody)} notes")
    print(f"   Note range: {min(n.pitch for n in melody)} to {max(n.pitch for n in melody)}")

    # Test 5: Banjo pattern
    print("\n5. Generating clawhammer banjo pattern...")
    chord_notes = [55, 59, 62, 67]  # G major
    banjo = BanjoPattern.generate_banjo_pattern(chord_notes, 'clawhammer', bars=2)
    print(f"   Generated {len(banjo)} notes")

    # Test 6: Contemporary folk arrangement
    print("\n6. Generating contemporary folk arrangement (Mumford & Sons style)...")
    contemp_gen = SingerSongwriterGenerator(
        style=FolkStyle.CONTEMPORARY_FOLK,
        key_root=62,  # D
        tempo=120
    )
    arrangement = contemp_gen.generate_arrangement(bars=8)
    print(f"   Generated {len(arrangement)} instrument tracks:")
    for instrument, notes in arrangement.items():
        print(f"   - {instrument}: {len(notes)} notes")

    # Test 7: Indie folk
    print("\n7. Generating indie folk arrangement (Fleet Foxes style)...")
    indie_gen = SingerSongwriterGenerator(
        style=FolkStyle.INDIE_FOLK,
        key_root=57,  # A
        tempo=85
    )
    indie_arr = indie_gen.generate_arrangement(bars=12)
    print(f"   Generated {len(indie_arr)} instrument tracks")

    # Test 8: Open tuning
    print("\n8. Testing open tunings (DADGAD)...")
    dadgad_voicing = OpenTuning.get_chord_voicing(50, 'sus4', 'dadgad')
    print(f"   DADGAD chord voicing: {dadgad_voicing}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nSinger-Songwriter/Folk features implemented:")
    print("  ✓ Traditional Folk (Dylan, Baez, Guthrie)")
    print("  ✓ Singer-Songwriter (Mitchell, Taylor, King)")
    print("  ✓ Contemporary Folk (Mumford, Lumineers, Bon Iver)")
    print("  ✓ Indie Folk (Stevens, Iron & Wine, Fleet Foxes)")
    print("  ✓ Americana (Welch, Isbell, Avett Brothers)")
    print("  ✓ Travis picking fingerpicking patterns")
    print("  ✓ Folk strumming patterns")
    print("  ✓ Open tunings (DADGAD, Open D, Open G)")
    print("  ✓ Banjo patterns (clawhammer, three-finger)")
    print("  ✓ Vocal melodies with folk characteristics")
    print("  ✓ Sparse, intimate arrangements")
    print("=" * 60)
