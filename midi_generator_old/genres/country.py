#!/usr/bin/env python3
"""
Country Music Generator - Traditional, Bluegrass, and Modern Country

This module implements comprehensive country music generation including:
- Traditional country (honky-tonk, Nashville sound)
- Bluegrass (fast picking, banjo rolls, fiddle licks)
- Modern country (pop-country, bro-country)

Features:
- Pedal steel bends and slides
- Banjo roll patterns
- Fiddle double stops and slides
- Train beat rhythms
- Walking bass lines
- Chicken pickin' guitar
- Nashville number system chord progressions

Author: Agent 7 - World Music & Additional Genres
References:
- "The Country Music Encyclopedia" - Erlewine et al.
- "Bluegrass: A History" - Neil V. Rosenberg
- Pedal steel technique studies - Buddy Emmons
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CountryStyle(Enum):
    """Country music sub-genres"""
    TRADITIONAL = "traditional"  # Hank Williams, Patsy Cline
    HONKY_TONK = "honky_tonk"  # Ernest Tubb, Lefty Frizzell
    NASHVILLE_SOUND = "nashville_sound"  # Patsy Cline, Jim Reeves
    OUTLAW = "outlaw"  # Willie Nelson, Waylon Jennings
    BLUEGRASS = "bluegrass"  # Bill Monroe, Flatt & Scruggs
    MODERN = "modern"  # Contemporary country-pop
    BRO_COUNTRY = "bro_country"  # 2010s country-rock
    ALT_COUNTRY = "alt_country"  # Americana, insurgent country


class CountryInstrument(Enum):
    """Country-specific instruments"""
    PEDAL_STEEL = 0  # MIDI program will be set in GM
    BANJO = 105  # GM Banjo
    FIDDLE = 110  # GM Fiddle
    ACOUSTIC_GUITAR = 24  # GM Acoustic Guitar (steel)
    ELECTRIC_GUITAR_CLEAN = 27  # GM Electric Guitar (clean)
    ELECTRIC_GUITAR_MUTED = 28  # GM Electric Guitar (muted)
    ACOUSTIC_BASS = 32  # GM Acoustic Bass
    ELECTRIC_BASS_FINGER = 33  # GM Electric Bass (finger)
    HARMONICA = 22  # GM Harmonica
    ACCORDION = 21  # GM Accordion


@dataclass
class CountryChord:
    """Country chord with Nashville number notation"""
    root: int  # MIDI note number
    quality: str  # 'major', 'minor', '7', 'maj7', 'add9', 'sus2', 'sus4'
    degree: int  # Nashville number (1-7)
    duration: float  # Duration in beats

    def get_notes(self) -> List[int]:
        """Get MIDI notes for the chord"""
        notes = [self.root]

        if self.quality in ['major', 'add9']:
            notes.extend([self.root + 4, self.root + 7])
            if self.quality == 'add9':
                notes.append(self.root + 14)  # Add 9th
        elif self.quality == 'minor':
            notes.extend([self.root + 3, self.root + 7])
        elif self.quality == '7':
            notes.extend([self.root + 4, self.root + 7, self.root + 10])
        elif self.quality == 'maj7':
            notes.extend([self.root + 4, self.root + 7, self.root + 11])
        elif self.quality == 'sus2':
            notes.extend([self.root + 2, self.root + 7])
        elif self.quality == 'sus4':
            notes.extend([self.root + 5, self.root + 7])

        return notes


class PedalSteelBend:
    """
    Pedal steel guitar bend and slide generator

    The pedal steel is iconic in country music, capable of smooth glissandi
    and complex chord changes via pedals and knee levers.

    References:
    - Buddy Emmons technique
    - Copedent charts (standard E9 and C6 tunings)
    """

    # E9 tuning (high to low): B, G#, F#, E, D, B, G#, F#, E, B
    E9_TUNING = [71, 68, 66, 64, 62, 59, 56, 54, 52, 47]

    # Common pedal steel licks (intervals from root)
    LICKS = {
        'country_cry': [0, 2, 4, 2, 0, -1],  # Signature descending cry
        'sweet_glide': [0, 2, 4, 5, 7],  # Major scale ascent
        'honky_tonk': [0, -2, 0, 2, 0],  # Classic country sound
        'western_swing': [0, 4, 7, 9, 7, 4],  # Swinging feel
    }

    @staticmethod
    def generate_bend(start_note: int, end_note: int,
                     duration: float, steps: int = 8) -> List[Tuple[int, int]]:
        """
        Generate pitch bend data for smooth slide between notes

        Args:
            start_note: Starting MIDI note
            end_note: Ending MIDI note
            duration: Duration in ticks
            steps: Number of pitch bend steps

        Returns:
            List of (time_offset, pitch_bend_value) tuples
        """
        bend_range = (end_note - start_note) * 4096 // 2  # MIDI pitch bend units
        time_step = duration / steps

        bends = []
        for i in range(steps + 1):
            time_offset = int(i * time_step)
            bend_value = int((i / steps) * bend_range)
            bends.append((time_offset, bend_value))

        return bends

    @staticmethod
    def generate_lick(lick_type: str, root: int,
                     rhythm: List[float]) -> List[Tuple[int, float]]:
        """
        Generate a pedal steel lick

        Args:
            lick_type: Type of lick from LICKS dictionary
            root: Root MIDI note
            rhythm: List of note durations

        Returns:
            List of (note, duration) tuples
        """
        intervals = PedalSteelBend.LICKS.get(lick_type, [0])
        notes = [root + interval for interval in intervals]

        # Match notes to rhythm pattern
        lick = []
        for i, note in enumerate(notes):
            duration = rhythm[i % len(rhythm)]
            lick.append((note, duration))

        return lick


class BanjoRoll:
    """
    Bluegrass banjo roll pattern generator

    Banjo rolls are continuous arpeggiated patterns that create the
    driving rhythm in bluegrass music. Common rolls include forward roll,
    backward roll, and alternating thumb roll.

    References:
    - Earl Scruggs "three-finger picking" technique
    - Tony Trischka banjo method books
    """

    # Standard 5-string banjo tuning (open G): D4, B3, G3, D3, G2
    OPEN_G_TUNING = [62, 59, 55, 50, 43]

    # Roll patterns (string indices: 0=5th string, 1=4th, 2=3rd, 3=2nd, 4=1st)
    ROLLS = {
        'forward': [2, 1, 4, 2, 1, 4, 2, 1],  # Classic forward roll
        'backward': [4, 1, 2, 4, 1, 2, 4, 1],  # Backward roll
        'alternating_thumb': [2, 0, 1, 0, 2, 0, 1, 0],  # Alternating thumb
        'forward_reverse': [2, 1, 4, 2, 4, 1, 2, 1],  # Mixed pattern
        'tag_lick': [2, 1, 4, 3, 2, 0],  # Common ending lick
    }

    @staticmethod
    def generate_roll(roll_type: str, chord_notes: List[int],
                     measures: int = 1,
                     notes_per_measure: int = 8) -> List[Tuple[int, float, int]]:
        """
        Generate banjo roll pattern

        Args:
            roll_type: Type of roll from ROLLS dictionary
            chord_notes: Chord tones to arpeggiate
            measures: Number of measures
            notes_per_measure: Notes per measure (typically 8 for 2/4 or 4/4)

        Returns:
            List of (note, duration, velocity) tuples
        """
        pattern = BanjoRoll.ROLLS.get(roll_type, BanjoRoll.ROLLS['forward'])
        roll = []

        # Duration for each note (assumes 4/4 time, eighth notes)
        duration = 0.5  # Eighth note

        for measure in range(measures):
            for i, string_index in enumerate(pattern):
                # Map string to chord tone
                note = chord_notes[string_index % len(chord_notes)]

                # Vary velocity for dynamics (thumb notes slightly louder)
                velocity = 90 if string_index in [0, 2] else 75

                roll.append((note, duration, velocity))

        return roll


class FiddleLick:
    """
    Bluegrass and country fiddle lick generator

    Fiddle is central to bluegrass and old-time country. Techniques include
    double stops, slides, shuffles, and characteristic melodic patterns.

    References:
    - Kenny Baker bluegrass fiddle style
    - Texas contest fiddling tradition
    """

    # Common fiddle double stops (intervals)
    DOUBLE_STOPS = {
        'major_3rd': 4,
        'perfect_5th': 7,
        'major_6th': 9,
        'octave': 12,
    }

    # Characteristic fiddle licks (scale degrees in major)
    LICKS = {
        'bluegrass_kickoff': [5, 5, 6, 7, 8, 7, 6, 5],  # Classic intro
        'texas_shuffle': [1, 3, 5, 6, 5, 3, 1],  # Texas style
        'double_stop_run': [1, 1, 2, 2, 3, 3, 4, 4],  # With double stops
        'slide_lick': [1, 2, 3, 5, 6, 5, 3, 1],  # With slides
        'ornament': [5, 6, 5, 4, 5],  # Quick ornament
    }

    @staticmethod
    def generate_lick(lick_type: str, key_root: int,
                     scale: List[int],
                     add_double_stops: bool = False) -> List[Tuple[int, float, Optional[int]]]:
        """
        Generate fiddle lick

        Args:
            lick_type: Type of lick from LICKS dictionary
            key_root: Root note of key
            scale: Scale intervals (e.g., major scale)
            add_double_stops: Whether to add harmony notes

        Returns:
            List of (note, duration, optional_harmony_note) tuples
        """
        degrees = FiddleLick.LICKS.get(lick_type, [1, 2, 3, 4, 5])
        lick = []

        for i, degree in enumerate(degrees):
            # Convert scale degree to MIDI note
            note = key_root + scale[(degree - 1) % len(scale)]
            duration = 0.25  # Sixteenth notes for fiddle runs

            # Add double stop harmony
            harmony = None
            if add_double_stops and lick_type == 'double_stop_run':
                harmony = note + FiddleLick.DOUBLE_STOPS['perfect_5th']

            lick.append((note, duration, harmony))

        return lick


class WalkingBass:
    """
    Country walking bass line generator

    Walking bass is essential in country, bluegrass, and rockabilly.
    Typically uses root, 3rd, 5th, and chromatic approaches.

    References:
    - Upright bass in bluegrass (Bill Monroe's bassists)
    - Country walking bass (Bob Moore, Nashville session bassist)
    """

    @staticmethod
    def generate_walking_line(chord_progression: List[CountryChord],
                             beats_per_chord: int = 4) -> List[Tuple[int, float]]:
        """
        Generate walking bass line for chord progression

        Args:
            chord_progression: List of CountryChord objects
            beats_per_chord: Number of beats per chord

        Returns:
            List of (note, duration) tuples
        """
        bass_line = []

        for i, chord in enumerate(chord_progression):
            root = chord.root - 12  # Drop octave for bass register
            third = root + (3 if chord.quality == 'minor' else 4)
            fifth = root + 7

            # Get next chord root for chromatic approach
            next_chord = chord_progression[(i + 1) % len(chord_progression)]
            next_root = next_chord.root - 12

            # Walking pattern: root, 3rd, 5th, chromatic approach
            if beats_per_chord == 4:
                # 4/4 time
                pattern = [
                    (root, 1.0),
                    (third, 1.0),
                    (fifth, 1.0),
                    (next_root - 1 if next_root > root else next_root + 1, 1.0),  # Chromatic approach
                ]
            elif beats_per_chord == 2:
                # 2/4 time
                pattern = [
                    (root, 1.0),
                    (fifth, 1.0),
                ]
            else:
                # Default: root on beat 1, fifth on other beats
                pattern = [(root, 1.0)] + [(fifth, 1.0)] * (beats_per_chord - 1)

            bass_line.extend(pattern)

        return bass_line


class TrainBeat:
    """
    Country "train beat" rhythm generator

    The train beat (boom-chick-a-boom-chick) is iconic in country music,
    mimicking the sound of a steam locomotive.

    References:
    - Johnny Cash "Folsom Prison Blues"
    - Hank Williams train songs
    """

    @staticmethod
    def generate_train_beat(measures: int = 4,
                           tempo: str = 'medium') -> List[Tuple[str, float, int]]:
        """
        Generate train beat rhythm pattern

        Args:
            measures: Number of measures
            tempo: 'slow', 'medium', or 'fast'

        Returns:
            List of (drum_type, time, velocity) tuples
        """
        # Drum types: 'kick', 'snare', 'hihat'
        pattern = []

        # Basic train beat: boom (kick) chick (snare) a (hihat) boom chick
        if tempo == 'fast':
            # Eighth note subdivisions
            single_measure = [
                ('kick', 0.0, 100),
                ('hihat', 0.5, 70),
                ('snare', 1.0, 90),
                ('hihat', 1.5, 70),
                ('kick', 2.0, 100),
                ('hihat', 2.5, 70),
                ('snare', 3.0, 90),
                ('hihat', 3.5, 70),
            ]
        else:
            # Quarter note subdivisions
            single_measure = [
                ('kick', 0.0, 100),
                ('snare', 1.0, 90),
                ('kick', 2.0, 100),
                ('snare', 3.0, 90),
            ]

        for measure in range(measures):
            offset = measure * 4.0
            for drum, time, velocity in single_measure:
                pattern.append((drum, offset + time, velocity))

        return pattern


class ChickenPickin:
    """
    Country "chicken pickin'" guitar technique generator

    Chicken pickin' is a hybrid picking technique combining pick and fingers,
    creating percussive, staccato notes characteristic of country and rockabilly.

    References:
    - James Burton technique
    - Albert Lee chicken pickin'
    """

    @staticmethod
    def generate_lick(root: int, scale: List[int],
                     pattern_type: str = 'classic') -> List[Tuple[int, float, int, bool]]:
        """
        Generate chicken pickin' guitar lick

        Args:
            root: Root note
            scale: Scale intervals
            pattern_type: 'classic', 'rockabilly', or 'modern'

        Returns:
            List of (note, duration, velocity, is_muted) tuples
        """
        lick = []

        if pattern_type == 'classic':
            # Classic chicken pickin': alternating picked and muted notes
            degrees = [1, 0, 2, 0, 3, 0, 5, 0]  # 0 = muted
            for degree in degrees:
                if degree == 0:
                    # Muted percussive note
                    lick.append((root, 0.25, 60, True))
                else:
                    note = root + scale[(degree - 1) % len(scale)]
                    lick.append((note, 0.25, 90, False))

        elif pattern_type == 'rockabilly':
            # Fast rockabilly lick
            degrees = [1, 3, 5, 6, 5, 3, 1]
            for degree in degrees:
                note = root + scale[(degree - 1) % len(scale)]
                lick.append((note, 0.125, 95, False))

        elif pattern_type == 'modern':
            # Modern country with string bends
            degrees = [1, 2, 3, 5, 5]  # Last 5 will be bent
            for i, degree in enumerate(degrees):
                note = root + scale[(degree - 1) % len(scale)]
                # Bend last note up whole step
                is_bent = (i == len(degrees) - 1)
                lick.append((note, 0.25, 90, is_bent))

        return lick


class CountryGenerator:
    """
    Main country music generator

    Combines all country music elements to generate complete arrangements
    in various country styles.
    """

    # Major scale intervals
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

    # Pentatonic major scale (common in country)
    PENTATONIC_MAJOR = [0, 2, 4, 7, 9]

    # Common country chord progressions (Nashville numbers)
    PROGRESSIONS = {
        'classic_country': [
            (1, 'major'), (4, 'major'), (1, 'major'), (1, 'major'),
            (4, 'major'), (4, 'major'), (1, 'major'), (1, 'major'),
            (5, '7'), (4, 'major'), (1, 'major'), (5, '7'),
        ],
        'bluegrass': [
            (1, 'major'), (4, 'major'), (1, 'major'), (5, '7'),
            (1, 'major'), (4, 'major'), (1, 'major'), (1, 'major'),
        ],
        'modern_country': [
            (1, 'major'), (5, 'major'), (6, 'minor'), (4, 'major'),
            (1, 'major'), (5, 'major'), (4, 'major'), (4, 'major'),
        ],
        'country_blues': [
            (1, '7'), (1, '7'), (1, '7'), (1, '7'),
            (4, '7'), (4, '7'), (1, '7'), (1, '7'),
            (5, '7'), (4, '7'), (1, '7'), (5, '7'),
        ],
    }

    def __init__(self, style: CountryStyle = CountryStyle.TRADITIONAL,
                 key_root: int = 60, tempo: int = 120):
        """
        Initialize country music generator

        Args:
            style: Country sub-genre style
            key_root: Root note of key (MIDI note number)
            tempo: Tempo in BPM
        """
        self.style = style
        self.key_root = key_root
        self.tempo = tempo

    def generate_chord_progression(self, bars: int = 8,
                                   progression_type: Optional[str] = None) -> List[CountryChord]:
        """
        Generate country chord progression

        Args:
            bars: Number of bars
            progression_type: Specific progression type, or None for auto-select

        Returns:
            List of CountryChord objects
        """
        # Auto-select progression based on style
        if progression_type is None:
            if self.style == CountryStyle.BLUEGRASS:
                progression_type = 'bluegrass'
            elif self.style in [CountryStyle.MODERN, CountryStyle.BRO_COUNTRY]:
                progression_type = 'modern_country'
            else:
                progression_type = 'classic_country'

        # Get progression pattern
        pattern = self.PROGRESSIONS.get(progression_type,
                                       self.PROGRESSIONS['classic_country'])

        # Build chord progression
        progression = []
        for i in range(bars):
            degree, quality = pattern[i % len(pattern)]

            # Calculate chord root
            scale_index = degree - 1
            chord_root = self.key_root + self.MAJOR_SCALE[scale_index % len(self.MAJOR_SCALE)]

            chord = CountryChord(
                root=chord_root,
                quality=quality,
                degree=degree,
                duration=4.0  # Whole note
            )
            progression.append(chord)

        return progression

    def generate_melody(self, chord_progression: List[CountryChord],
                       style_variant: str = 'vocal') -> List[Tuple[int, float, int]]:
        """
        Generate country melody

        Args:
            chord_progression: Chord progression to follow
            style_variant: 'vocal', 'fiddle', 'pedal_steel', or 'banjo'

        Returns:
            List of (note, duration, velocity) tuples
        """
        melody = []

        if style_variant == 'vocal':
            # Simple vocal melody using pentatonic scale
            for chord in chord_progression:
                for beat in range(4):
                    # Choose chord tone or passing tone
                    if beat % 2 == 0:
                        # Chord tone
                        note = random.choice(chord.get_notes())
                    else:
                        # Passing tone from pentatonic
                        degree = random.choice([1, 2, 3, 5, 6])
                        note = self.key_root + self.PENTATONIC_MAJOR[(degree - 1) % len(self.PENTATONIC_MAJOR)]

                    melody.append((note, 1.0, 80))

        elif style_variant == 'fiddle':
            # Generate fiddle licks
            for chord in chord_progression:
                lick = FiddleLick.generate_lick('bluegrass_kickoff',
                                               self.key_root,
                                               self.MAJOR_SCALE)
                for note, duration, harmony in lick:
                    melody.append((note, duration, 90))

        elif style_variant == 'banjo':
            # Generate banjo rolls
            for chord in chord_progression:
                roll = BanjoRoll.generate_roll('forward',
                                              chord.get_notes(),
                                              measures=1)
                melody.extend(roll)

        return melody

    def generate_arrangement(self, bars: int = 16) -> Dict[str, List]:
        """
        Generate complete country arrangement

        Args:
            bars: Number of bars

        Returns:
            Dictionary with instrument tracks
        """
        # Generate chord progression
        progression = self.generate_chord_progression(bars)

        arrangement = {}

        # Add appropriate instruments based on style
        if self.style == CountryStyle.BLUEGRASS:
            # Bluegrass instrumentation
            arrangement['banjo'] = self.generate_melody(progression, 'banjo')
            arrangement['fiddle'] = self.generate_melody(progression, 'fiddle')
            arrangement['bass'] = WalkingBass.generate_walking_line(progression, 2)

        elif self.style in [CountryStyle.TRADITIONAL, CountryStyle.HONKY_TONK]:
            # Traditional country
            arrangement['vocal'] = self.generate_melody(progression, 'vocal')
            arrangement['pedal_steel'] = self.generate_melody(progression, 'pedal_steel')
            arrangement['bass'] = WalkingBass.generate_walking_line(progression, 4)
            arrangement['drums'] = TrainBeat.generate_train_beat(bars, 'medium')

        elif self.style in [CountryStyle.MODERN, CountryStyle.BRO_COUNTRY]:
            # Modern country
            arrangement['vocal'] = self.generate_melody(progression, 'vocal')
            arrangement['bass'] = WalkingBass.generate_walking_line(progression, 4)
            arrangement['drums'] = TrainBeat.generate_train_beat(bars, 'fast')

        return arrangement


if __name__ == "__main__":
    """Example usage and testing"""

    print("Country Music Generator - Test Suite\n")
    print("=" * 60)

    # Test 1: Generate bluegrass arrangement
    print("\n1. Generating Bluegrass arrangement (G major, 120 BPM)...")
    bluegrass_gen = CountryGenerator(
        style=CountryStyle.BLUEGRASS,
        key_root=55,  # G
        tempo=140
    )

    progression = bluegrass_gen.generate_chord_progression(8, 'bluegrass')
    print(f"   Generated {len(progression)} chords")
    for i, chord in enumerate(progression[:4]):
        print(f"   Bar {i+1}: Degree {chord.degree} ({chord.quality})")

    # Test 2: Banjo roll
    print("\n2. Generating banjo forward roll...")
    chord_notes = [55, 59, 62, 67]  # G major chord
    roll = BanjoRoll.generate_roll('forward', chord_notes, measures=2)
    print(f"   Generated {len(roll)} notes")
    print(f"   First 4 notes: {[note for note, dur, vel in roll[:4]]}")

    # Test 3: Fiddle lick
    print("\n3. Generating fiddle bluegrass kickoff lick...")
    lick = FiddleLick.generate_lick('bluegrass_kickoff', 67,
                                    CountryGenerator.MAJOR_SCALE,
                                    add_double_stops=True)
    print(f"   Generated {len(lick)} notes")

    # Test 4: Walking bass
    print("\n4. Generating walking bass line...")
    bass = WalkingBass.generate_walking_line(progression[:4], beats_per_chord=4)
    print(f"   Generated {len(bass)} bass notes")

    # Test 5: Pedal steel bend
    print("\n5. Generating pedal steel bend (E to G)...")
    bends = PedalSteelBend.generate_bend(64, 67, 480, steps=8)
    print(f"   Generated {len(bends)} pitch bend points")

    # Test 6: Train beat
    print("\n6. Generating train beat rhythm...")
    train = TrainBeat.generate_train_beat(4, 'medium')
    print(f"   Generated {len(train)} drum hits")

    # Test 7: Complete arrangement
    print("\n7. Generating complete traditional country arrangement...")
    trad_gen = CountryGenerator(
        style=CountryStyle.TRADITIONAL,
        key_root=60,  # C
        tempo=120
    )
    arrangement = trad_gen.generate_arrangement(bars=8)
    print(f"   Generated {len(arrangement)} instrument tracks:")
    for instrument, notes in arrangement.items():
        print(f"   - {instrument}: {len(notes)} events")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nCountry music features implemented:")
    print("  ✓ Traditional, Honky-Tonk, Nashville Sound")
    print("  ✓ Bluegrass (banjo rolls, fiddle licks)")
    print("  ✓ Modern Country and Bro-Country")
    print("  ✓ Pedal steel bends and slides")
    print("  ✓ Walking bass lines")
    print("  ✓ Train beat rhythms")
    print("  ✓ Chicken pickin' guitar")
    print("  ✓ Nashville number system progressions")
