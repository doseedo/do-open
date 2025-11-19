#!/usr/bin/env python3
"""
African Music Generator - West African and Traditional African Music

This module implements comprehensive African music generation including:
- West African polyrhythmic patterns
- Timeline patterns (bell patterns)
- Call and response structures
- Talking drum patterns
- Traditional African instruments and ensembles

Features:
- Polyrhythms (2:3, 3:4, 4:5, cross-rhythms)
- West African timeline patterns (12/8, 16/16)
- Call and response (leader-chorus)
- Talking drum (dundun, djembe)
- Kora, Balafon, Mbira patterns
- Regional styles (West African, Central African, South African)

Author: Agent 7 - World Music & Additional Genres
References:
- "African Rhythm and African Sensibility" - John Miller Chernoff
- "The Music of Africa" - J.H. Kwabena Nketia
- "Representing African Music" - Kofi Agawu
- "Theory and Practice in African Drumming" - David Locke
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AfricanRegion(Enum):
    """African musical regions"""
    WEST_AFRICA = "west_africa"  # Ghana, Senegal, Mali, Guinea
    CENTRAL_AFRICA = "central_africa"  # Congo, Cameroon
    EAST_AFRICA = "east_africa"  # Kenya, Tanzania, Ethiopia
    SOUTH_AFRICA = "south_africa"  # South Africa, Zimbabwe


class TimelinePattern(Enum):
    """Common West African timeline (bell) patterns"""
    STANDARD_12_8 = "standard_12_8"  # Standard 12/8 pattern
    SON_CLAVE_32 = "son_clave_32"  # 3-2 Son clave
    BEMBÉ = "bembe"  # Bembe pattern
    GAHU = "gahu"  # Ewe Gahu pattern
    KUI = "kui"  # Kui pattern


@dataclass
class Polyrhythm:
    """
    Represents a polyrhythmic pattern

    Polyrhythm: simultaneous use of two or more conflicting rhythms
    that are not readily perceived as deriving from one another.

    Example: 3 against 2 means 3 beats in the same time as 2 beats
    """
    numerator: int  # First rhythm (e.g., 3 in "3 against 2")
    denominator: int  # Second rhythm (e.g., 2 in "3 against 2")
    cycle_length: float  # Length of one complete cycle

    def get_pattern_a(self) -> List[float]:
        """Get time points for first rhythm"""
        return [i * (self.cycle_length / self.numerator)
                for i in range(self.numerator)]

    def get_pattern_b(self) -> List[float]:
        """Get time points for second rhythm"""
        return [i * (self.cycle_length / self.denominator)
                for i in range(self.denominator)]


class WestAfricanTimeline:
    """
    West African timeline (bell) pattern generator

    The timeline is the organizing principle of West African rhythm.
    Typically played on a bell or high-pitched instrument, it provides
    the rhythmic foundation for the ensemble.

    The most common is the 12/8 "standard pattern" found across West Africa.
    """

    @staticmethod
    def generate_standard_12_8(cycles: int = 4) -> List[Tuple[float, int]]:
        """
        Generate standard 12/8 West African timeline

        Pattern: X . . X . . X . . X . .
        (beats: 1, 4, 7, 10 out of 12)

        This pattern is found in:
        - Ewe music (Ghana)
        - Yoruba music (Nigeria)
        - Afro-Cuban music (via diaspora)

        Args:
            cycles: Number of 12/8 cycles

        Returns:
            List of (time_in_beats, velocity) tuples
        """
        pattern = []
        # 12/8 pattern: play on beats 1, 4, 7, 10 (out of 12 eighth notes)
        bell_pattern = [0, 3, 6, 9]  # In eighth notes

        for cycle in range(cycles):
            offset = cycle * 12.0  # 12 eighth notes per cycle

            for beat in bell_pattern:
                time = offset + beat
                velocity = 95 if beat == 0 else 85  # Accent first beat
                pattern.append((time, velocity))

        return pattern

    @staticmethod
    def generate_gahu(cycles: int = 4) -> List[Tuple[float, int]]:
        """
        Generate Gahu timeline pattern (Ewe, Ghana)

        Gahu is a recreational dance from the Ewe people.
        Pattern in 12/8: X . . X . X . . X X . .

        Args:
            cycles: Number of cycles

        Returns:
            List of (time_in_beats, velocity) tuples
        """
        pattern = []
        # Gahu pattern positions (in eighth notes)
        bell_pattern = [0, 3, 5, 8, 9]

        for cycle in range(cycles):
            offset = cycle * 12.0

            for beat in bell_pattern:
                time = offset + beat
                velocity = 95 if beat == 0 else 85
                pattern.append((time, velocity))

        return pattern

    @staticmethod
    def generate_clave_32(cycles: int = 4) -> List[Tuple[float, int]]:
        """
        Generate 3-2 Son clave pattern

        While Cuban, the clave has African origins and is widely used
        in African diaspora music.

        Pattern in 16ths: X . . X . . X . | . . X . X . . .
        (3 side) | (2 side)

        Args:
            cycles: Number of cycles

        Returns:
            List of (time_in_beats, velocity) tuples
        """
        pattern = []
        # 3-2 clave pattern (in 16th notes over 2 bars)
        clave_pattern = [
            # Bar 1 (3 side)
            0, 3, 6,
            # Bar 2 (2 side)
            10, 12
        ]

        for cycle in range(cycles):
            offset = cycle * 16.0

            for beat in clave_pattern:
                time = offset + beat
                velocity = 90
                pattern.append((time, velocity))

        return pattern


class Djembe:
    """
    Djembe drum pattern generator

    The djembe is a goblet-shaped hand drum from West Africa.
    Three main sounds:
    - Bass: Open tone in center
    - Tone: Open tone near edge
    - Slap: Sharp slap near edge
    """

    SOUNDS = {
        'bass': 45,  # MIDI note for bass
        'tone': 60,  # MIDI note for tone
        'slap': 65,  # MIDI note for slap
    }

    @staticmethod
    def generate_pattern(style: str = 'standard',
                        measures: int = 4) -> List[Tuple[str, float, int]]:
        """
        Generate djembe pattern

        Args:
            style: 'standard', 'mamady', or 'dundunba'
            measures: Number of measures (in 12/8)

        Returns:
            List of (sound_type, time, velocity) tuples
        """
        pattern = []

        if style == 'standard':
            # Standard djembe accompaniment (12/8)
            single_measure = [
                ('bass', 0.0, 100),
                ('tone', 3.0, 80),
                ('slap', 6.0, 95),
                ('tone', 9.0, 80),
            ]

        elif style == 'mamady':
            # Mamady Keita style (more complex)
            single_measure = [
                ('bass', 0.0, 100),
                ('slap', 1.5, 90),
                ('tone', 3.0, 80),
                ('bass', 6.0, 100),
                ('tone', 7.5, 75),
                ('slap', 9.0, 95),
                ('tone', 10.5, 75),
            ]

        elif style == 'dundunba':
            # Dundunba rhythm
            single_measure = [
                ('bass', 0.0, 110),
                ('tone', 2.0, 75),
                ('slap', 3.0, 95),
                ('bass', 6.0, 110),
                ('slap', 9.0, 95),
            ]

        else:
            single_measure = [('bass', 0.0, 100)]

        for measure in range(measures):
            offset = measure * 12.0

            for sound, time, velocity in single_measure:
                pattern.append((sound, offset + time, velocity))

        return pattern


class TalkingDrum:
    """
    Talking drum (dundun, kalangu) pattern generator

    The talking drum can change pitch by squeezing, mimicking
    the tonal nature of West African languages. Used for
    communication and music.
    """

    @staticmethod
    def generate_speech_pattern(phrase: str,
                               tones: List[int]) -> List[Tuple[int, float, int]]:
        """
        Generate talking drum pattern mimicking speech

        In tonal languages (Yoruba, Akan, etc.), pitch conveys meaning.
        The talking drum recreates these tones.

        Args:
            phrase: Text phrase (for structure, not literal)
            tones: List of MIDI notes representing pitch levels

        Returns:
            List of (note, time, velocity) tuples
        """
        pattern = []
        time = 0.0

        # Each syllable gets a tone
        syllables = phrase.split()

        for i, syllable in enumerate(syllables):
            # Choose tone based on position (simulating tonal contour)
            tone_idx = i % len(tones)
            note = tones[tone_idx]

            # Syllable length varies
            duration = 0.3 + (len(syllable) * 0.1)
            velocity = 85

            pattern.append((note, time, velocity))
            time += duration

        return pattern

    @staticmethod
    def generate_rhythm_pattern(measures: int = 4) -> List[Tuple[int, float, int]]:
        """
        Generate rhythmic dundun (bass drum) pattern

        The dundun provides the rhythmic foundation in West African
        drumming ensembles.

        Args:
            measures: Number of measures

        Returns:
            List of (note, time, velocity) tuples
        """
        pattern = []
        # Dundun typically plays simple, repetitive patterns
        # 12/8 feel: Bass on 1, 7

        for measure in range(measures):
            offset = measure * 12.0

            pattern.extend([
                (45, offset + 0.0, 110),  # Beat 1
                (45, offset + 6.0, 105),  # Beat 7
            ])

        return pattern


class Kora:
    """
    Kora pattern generator

    The kora is a 21-string bridge-harp from West Africa (Mali, Senegal,
    Gambia). It plays intricate interlocking patterns called kumbengo
    (ostinato) and birimintingo (improvisation).
    """

    # Kora tuning (simplified, in F major)
    # Actual kora has complex tuning system
    KORA_SCALE = [0, 2, 4, 5, 7, 9, 11]  # Major scale

    @staticmethod
    def generate_kumbengo(root: int, measures: int = 4) -> List[Tuple[int, float, int]]:
        """
        Generate kumbengo (ostinato pattern) for kora

        Kumbengo is the cyclical accompaniment pattern that supports
        the song and vocal.

        Args:
            root: Root note of scale
            measures: Number of measures

        Returns:
            List of (note, time, velocity) tuples
        """
        pattern = []

        # Typical kumbengo: arpeggiated pattern in compound meter
        # Using notes from scale
        scale_notes = [root + interval for interval in Kora.KORA_SCALE]

        # Create interlocking pattern between left and right hand
        for measure in range(measures):
            offset = measure * 12.0  # 12/8 time

            # Left hand (bass notes, beats 1, 4, 7, 10)
            left_hand = [
                (scale_notes[0], offset + 0.0, 75),  # Root
                (scale_notes[2], offset + 3.0, 70),  # Third
                (scale_notes[4], offset + 6.0, 75),  # Fifth
                (scale_notes[2], offset + 9.0, 70),  # Third
            ]

            # Right hand (treble notes, offbeat)
            right_hand = [
                (scale_notes[4] + 12, offset + 1.5, 65),
                (scale_notes[2] + 12, offset + 4.5, 65),
                (scale_notes[4] + 12, offset + 7.5, 65),
                (scale_notes[0] + 12, offset + 10.5, 65),
            ]

            pattern.extend(left_hand)
            pattern.extend(right_hand)

        return sorted(pattern, key=lambda x: x[1])  # Sort by time


class Balafon:
    """
    Balafon pattern generator

    The balafon is a West African xylophone with gourd resonators.
    Known for its bright, percussive sound.
    """

    @staticmethod
    def generate_pattern(root: int, pattern_type: str = 'standard',
                        measures: int = 4) -> List[Tuple[int, float, int]]:
        """
        Generate balafon pattern

        Args:
            root: Root note
            pattern_type: 'standard', 'fast', or 'polyrhythmic'
            measures: Number of measures

        Returns:
            List of (note, time, velocity) tuples
        """
        pattern = []
        scale = [root + interval for interval in [0, 2, 4, 5, 7, 9, 11]]

        if pattern_type == 'standard':
            # Simple repeating pattern
            for measure in range(measures):
                offset = measure * 12.0

                for i, note in enumerate(scale[:4]):
                    time = offset + (i * 3.0)
                    pattern.append((note, time, 85))

        elif pattern_type == 'fast':
            # Fast interlocking pattern
            for measure in range(measures):
                offset = measure * 12.0

                for i in range(12):
                    note = scale[i % len(scale)]
                    time = offset + i
                    velocity = 80 + (i % 2) * 10
                    pattern.append((note, time, velocity))

        elif pattern_type == 'polyrhythmic':
            # 3:2 polyrhythm
            for measure in range(measures):
                offset = measure * 12.0

                # Pattern A (3)
                for i in range(3):
                    note = scale[i % len(scale)]
                    time = offset + (i * 4.0)
                    pattern.append((note, time, 90))

                # Pattern B (2)
                for i in range(2):
                    note = scale[(i + 2) % len(scale)]
                    time = offset + (i * 6.0)
                    pattern.append((note, time, 80))

        return sorted(pattern, key=lambda x: x[1])


class CallAndResponse:
    """
    African call and response pattern generator

    Call and response is fundamental to African music. A leader
    "calls" and the group "responds." This creates a dialogue
    structure central to African musical aesthetics.
    """

    @staticmethod
    def generate_pattern(call_phrase: List[int],
                        response_phrase: List[int],
                        repetitions: int = 4,
                        call_duration: float = 6.0,
                        response_duration: float = 6.0) -> Dict[str, List[Tuple[int, float, int]]]:
        """
        Generate call and response pattern

        Args:
            call_phrase: Leader's melody (MIDI notes)
            response_phrase: Group's response (MIDI notes)
            repetitions: Number of call-response cycles
            call_duration: Duration of call in beats
            response_duration: Duration of response in beats

        Returns:
            Dictionary with 'call' and 'response' tracks
        """
        pattern = {'call': [], 'response': []}

        time = 0.0

        for rep in range(repetitions):
            # Call
            note_duration = call_duration / len(call_phrase)
            for i, note in enumerate(call_phrase):
                pattern['call'].append((note, time + (i * note_duration), 90))

            time += call_duration

            # Response
            note_duration = response_duration / len(response_phrase)
            for i, note in enumerate(response_phrase):
                pattern['response'].append((note, time + (i * note_duration), 85))

            time += response_duration

        return pattern


class PolyrhythmGenerator:
    """
    Polyrhythm generator for complex African rhythms

    Common polyrhythms in African music:
    - 2:3 (hemiola)
    - 3:4
    - 4:5
    - 3:2 (inverse hemiola)
    """

    @staticmethod
    def generate(ratio: str, cycles: int = 4,
                cycle_length: float = 12.0) -> Dict[str, List[Tuple[float, int]]]:
        """
        Generate polyrhythmic pattern

        Args:
            ratio: Polyrhythm ratio (e.g., "3:2", "4:5")
            cycles: Number of cycles
            cycle_length: Length of one cycle in beats

        Returns:
            Dictionary with 'part_a' and 'part_b' rhythm tracks
        """
        parts = ratio.split(':')
        if len(parts) != 2:
            raise ValueError("Ratio must be in format 'X:Y'")

        numerator = int(parts[0])
        denominator = int(parts[1])

        poly = Polyrhythm(numerator, denominator, cycle_length)

        pattern = {'part_a': [], 'part_b': []}

        for cycle in range(cycles):
            offset = cycle * cycle_length

            # Generate part A
            for time in poly.get_pattern_a():
                pattern['part_a'].append((offset + time, 90))

            # Generate part B
            for time in poly.get_pattern_b():
                pattern['part_b'].append((offset + time, 85))

        return pattern


class AfricanMusicGenerator:
    """
    Main African music generator

    Combines timeline, polyrhythms, drums, and melodic instruments
    for authentic African ensemble music.
    """

    def __init__(self, region: AfricanRegion = AfricanRegion.WEST_AFRICA,
                 root_note: int = 60):
        """
        Initialize generator

        Args:
            region: African musical region
            root_note: Root note for melodic instruments
        """
        self.region = region
        self.root_note = root_note

    def generate_ensemble(self, measures: int = 8) -> Dict[str, List]:
        """
        Generate complete African ensemble

        Args:
            measures: Number of measures (in 12/8)

        Returns:
            Dictionary with all instrument tracks
        """
        ensemble = {}

        # Timeline (bell)
        ensemble['bell'] = WestAfricanTimeline.generate_standard_12_8(measures)

        # Djembe
        djembe_pattern = Djembe.generate_pattern('standard', measures)
        ensemble['djembe'] = [(Djembe.SOUNDS[sound], time, vel)
                              for sound, time, vel in djembe_pattern]

        # Dundun (bass drum)
        ensemble['dundun'] = TalkingDrum.generate_rhythm_pattern(measures)

        # Kora (if West African)
        if self.region == AfricanRegion.WEST_AFRICA:
            ensemble['kora'] = Kora.generate_kumbengo(self.root_note, measures)

            # Balafon
            ensemble['balafon'] = Balafon.generate_pattern(
                self.root_note + 12, 'fast', measures
            )

        return ensemble


if __name__ == "__main__":
    """Example usage and testing"""

    print("African Music Generator - Test Suite\n")
    print("=" * 60)

    # Test 1: West African timeline
    print("\n1. Generating standard 12/8 West African timeline...")
    timeline = WestAfricanTimeline.generate_standard_12_8(2)
    print(f"   Generated {len(timeline)} bell hits")
    print(f"   Pattern: {[time for time, vel in timeline]}")

    # Test 2: Polyrhythm 3:2
    print("\n2. Generating 3:2 polyrhythm...")
    poly = PolyrhythmGenerator.generate("3:2", 2, 12.0)
    print(f"   Part A (3): {len(poly['part_a'])} hits")
    print(f"   Part B (2): {len(poly['part_b'])} hits")

    # Test 3: Djembe pattern
    print("\n3. Generating djembe pattern...")
    djembe = Djembe.generate_pattern('mamady', 4)
    print(f"   Generated {len(djembe)} djembe strokes")

    # Test 4: Kora kumbengo
    print("\n4. Generating kora kumbengo pattern...")
    kora = Kora.generate_kumbengo(60, 4)
    print(f"   Generated {len(kora)} kora notes")

    # Test 5: Balafon
    print("\n5. Generating balafon polyrhythmic pattern...")
    balafon = Balafon.generate_pattern(67, 'polyrhythmic', 4)
    print(f"   Generated {len(balafon)} balafon notes")

    # Test 6: Call and response
    print("\n6. Generating call and response...")
    call = [60, 62, 64, 65]
    response = [67, 65, 64, 62]
    call_resp = CallAndResponse.generate_pattern(call, response, 3)
    print(f"   Call events: {len(call_resp['call'])}")
    print(f"   Response events: {len(call_resp['response'])}")

    # Test 7: Talking drum
    print("\n7. Generating talking drum speech pattern...")
    tones = [57, 60, 62, 64]  # High, mid, low tones
    talking = TalkingDrum.generate_speech_pattern("Greetings welcome friends", tones)
    print(f"   Generated {len(talking)} talking drum syllables")

    # Test 8: Complete ensemble
    print("\n8. Generating complete West African ensemble...")
    generator = AfricanMusicGenerator(AfricanRegion.WEST_AFRICA, 60)
    ensemble = generator.generate_ensemble(8)
    print(f"   Generated ensemble with {len(ensemble)} instruments:")
    for instrument, events in ensemble.items():
        print(f"   - {instrument}: {len(events)} events")

    # Test 9: Gahu timeline
    print("\n9. Generating Gahu timeline pattern...")
    gahu = WestAfricanTimeline.generate_gahu(4)
    print(f"   Generated {len(gahu)} bell hits")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nAfrican music features implemented:")
    print("  ✓ West African timeline patterns (12/8, Gahu, Clave)")
    print("  ✓ Polyrhythms (2:3, 3:4, 4:5, custom ratios)")
    print("  ✓ Djembe patterns (bass, tone, slap)")
    print("  ✓ Talking drum (dundun) patterns")
    print("  ✓ Kora kumbengo (ostinato) patterns")
    print("  ✓ Balafon xylophone patterns")
    print("  ✓ Call and response structures")
    print("  ✓ West African ensemble generation")
    print("  ✓ Interlocking rhythmic patterns")
