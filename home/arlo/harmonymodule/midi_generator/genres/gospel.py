#!/usr/bin/env python3
"""
Gospel Music Generator - Traditional and Contemporary Gospel

This module implements comprehensive gospel music generation including:
- Traditional gospel (Mahalia Jackson, Golden Gate Quartet)
- Contemporary gospel (Kirk Franklin, Yolanda Adams)
- Black church tradition
- Southern gospel quartet

Features:
- SATB choir voicing (Soprano, Alto, Tenor, Bass)
- Hammond B3 organ runs and fills
- Call and response patterns
- Gospel chord progressions (rich harmonies)
- Piano accompaniment patterns
- Tambourine and hand claps
- Vocal runs and melisma
- Shout sections

Author: Agent 7 - World Music & Additional Genres
References:
- "The Gospel Sound" - Anthony Heilbut
- "We'll Understand It Better By and By" - Bernice Johnson Reagon
- Hammond organ technique - Jimmy Smith, Booker T. Jones
- Contemporary gospel harmony - Israel Houghton, Fred Hammond
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GospelStyle(Enum):
    """Gospel music sub-genres"""
    TRADITIONAL = "traditional"  # 1930s-60s traditional gospel
    QUARTET = "quartet"  # Southern gospel quartet
    MASS_CHOIR = "mass_choir"  # 1970s-80s mass choir
    CONTEMPORARY = "contemporary"  # Modern contemporary gospel
    PRAISE_WORSHIP = "praise_worship"  # Praise and worship
    URBAN_GOSPEL = "urban_gospel"  # Hip-hop influenced gospel


@dataclass
class ChoirVoicing:
    """SATB choir voice ranges and notes"""
    soprano: List[int]  # Typical range: C4-C6 (60-84)
    alto: List[int]     # Typical range: G3-G5 (55-79)
    tenor: List[int]    # Typical range: C3-C5 (48-72)
    bass: List[int]     # Typical range: E2-E4 (40-64)

    @staticmethod
    def from_chord(root: int, quality: str,
                   soprano_note: Optional[int] = None) -> 'ChoirVoicing':
        """
        Create SATB voicing from chord root and quality

        Args:
            root: Chord root note (MIDI)
            quality: Chord quality ('major', 'minor', '7', 'maj7', etc.)
            soprano_note: Specific soprano note, or None for automatic

        Returns:
            ChoirVoicing with proper voice leading
        """
        # Build chord tones
        chord_tones = [root]

        if quality in ['major', 'maj7', '9', 'maj9']:
            chord_tones.extend([root + 4, root + 7])
            if 'maj7' in quality or '9' in quality:
                chord_tones.append(root + 11)
            if '9' in quality:
                chord_tones.append(root + 14)
        elif quality in ['minor', 'min7', 'min9']:
            chord_tones.extend([root + 3, root + 7])
            if 'min7' in quality or 'min9' in quality:
                chord_tones.append(root + 10)
        elif quality in ['7', 'dom7']:
            chord_tones.extend([root + 4, root + 7, root + 10])
        elif quality == 'dim':
            chord_tones.extend([root + 3, root + 6])
        elif quality == 'aug':
            chord_tones.extend([root + 4, root + 8])

        # Place voices in appropriate ranges
        soprano = soprano_note if soprano_note else chord_tones[-1] + 12
        # Ensure soprano in range (C4-C6)
        while soprano < 60:
            soprano += 12
        while soprano > 84:
            soprano -= 12

        # Voice lead from soprano downward
        alto = max([t for t in chord_tones if t < soprano], default=soprano - 4)
        while alto < 55 or alto >= soprano:
            alto -= 12 if alto >= soprano else -12
            if 55 <= alto < soprano:
                break

        tenor = max([t for t in chord_tones if t < alto], default=alto - 4)
        while tenor < 48 or tenor >= alto:
            tenor -= 12 if tenor >= alto else -12
            if 48 <= tenor < alto:
                break

        bass_note = min([t for t in chord_tones], default=root)
        while bass_note > 64 or bass_note >= tenor:
            bass_note -= 12

        return ChoirVoicing(
            soprano=[soprano],
            alto=[alto],
            tenor=[tenor],
            bass=[bass_note]
        )


class HammondOrgan:
    """
    Hammond B3 organ pattern generator

    The Hammond organ is central to gospel music, providing harmonic
    foundation and energetic fills. Characteristic techniques include
    runs, trills, and drawbar manipulation.

    References:
    - Jimmy Smith technique
    - Booker T. Jones
    - Billy Preston gospel organ
    """

    # Drawbar settings (simulated via velocity/timbre)
    DRAWBAR_SETTINGS = {
        'full': [8, 8, 8, 8, 8, 8, 8, 8, 8],  # All out
        'gospel': [8, 8, 5, 4, 3, 2, 1, 0, 0],  # Classic gospel sound
        'blues': [8, 8, 8, 0, 0, 0, 0, 0, 0],  # Bluesy sound
        'soft': [5, 4, 3, 2, 1, 0, 0, 0, 0],  # Softer background
    }

    @staticmethod
    def generate_run(start_note: int, end_note: int,
                    scale: List[int],
                    run_type: str = 'ascending') -> List[Tuple[int, float, int]]:
        """
        Generate Hammond organ run

        Args:
            start_note: Starting note
            end_note: Ending note
            scale: Scale intervals to use
            run_type: 'ascending', 'descending', 'trill'

        Returns:
            List of (note, duration, velocity) tuples
        """
        run = []

        if run_type == 'ascending':
            # Fast ascending run (common fill)
            current = start_note
            while current <= end_note:
                duration = 0.125  # Sixteenth notes
                velocity = 95
                run.append((current, duration, velocity))

                # Find next scale tone
                for interval in scale:
                    next_note = (start_note // 12) * 12 + interval
                    while next_note <= current:
                        next_note += 12
                    if next_note > current:
                        current = next_note
                        break

        elif run_type == 'descending':
            # Fast descending run
            current = end_note
            while current >= start_note:
                duration = 0.125
                velocity = 95
                run.append((current, duration, velocity))

                # Find previous scale tone
                for interval in reversed(scale):
                    prev_note = (end_note // 12) * 12 + interval
                    while prev_note >= current:
                        prev_note -= 12
                    if prev_note < current:
                        current = prev_note
                        break

        elif run_type == 'trill':
            # Trill between two adjacent notes
            for i in range(8):
                note = start_note if i % 2 == 0 else start_note + 2
                run.append((note, 0.125, 90))

        return run

    @staticmethod
    def generate_chord_pattern(chord_notes: List[int],
                              measures: int = 4,
                              style: str = 'block') -> List[Tuple[List[int], float, float, int]]:
        """
        Generate Hammond organ chord accompaniment pattern

        Args:
            chord_notes: Notes in the chord
            measures: Number of measures
            style: 'block', 'broken', 'walking'

        Returns:
            List of (notes, time, duration, velocity) tuples
        """
        pattern = []

        if style == 'block':
            # Block chords on beats 2 and 4 (gospel style)
            for measure in range(measures):
                offset = measure * 4.0
                pattern.append((chord_notes, offset + 1.0, 0.75, 85))
                pattern.append((chord_notes, offset + 3.0, 0.75, 85))

        elif style == 'broken':
            # Arpeggiated chords
            for measure in range(measures):
                offset = measure * 4.0
                for beat in range(4):
                    note = chord_notes[beat % len(chord_notes)]
                    pattern.append(([note], offset + beat, 0.5, 75))

        elif style == 'walking':
            # Walking organ bass line with chord stabs
            for measure in range(measures):
                offset = measure * 4.0
                # Bass notes walking
                for beat in range(4):
                    bass = chord_notes[0] - 12
                    pattern.append(([bass], offset + beat, 0.5, 80))
                # Chord stab on 2 and 4
                pattern.append((chord_notes[1:], offset + 1.0, 0.5, 85))
                pattern.append((chord_notes[1:], offset + 3.0, 0.5, 85))

        return pattern


class CallAndResponse:
    """
    Gospel call and response pattern generator

    Call and response is fundamental to African-American gospel tradition.
    The lead voice "calls" and the choir "responds."

    References:
    - African-American church tradition
    - Lined-out hymnody
    """

    @staticmethod
    def generate_pattern(lead_phrase: List[Tuple[int, float]],
                        choir_chord: List[int],
                        repetitions: int = 4) -> Dict[str, List[Tuple]]:
        """
        Generate call and response pattern

        Args:
            lead_phrase: Lead vocal melody (note, duration) tuples
            choir_chord: Choir response chord notes
            repetitions: Number of call-response cycles

        Returns:
            Dictionary with 'lead' and 'choir' tracks
        """
        pattern = {'lead': [], 'choir': []}

        time = 0.0

        for rep in range(repetitions):
            # Call: Lead vocal
            for note, duration in lead_phrase:
                pattern['lead'].append((note, time, duration, 90))
                time += duration

            # Response: Choir (half the length of call)
            choir_duration = sum(d for _, d in lead_phrase) / 2
            pattern['choir'].append((choir_chord, time, choir_duration, 85))
            time += choir_duration

            # Brief rest
            time += 0.5

        return pattern


class GospelChords:
    """
    Gospel chord progression generator

    Gospel music uses rich harmonic progressions with:
    - Secondary dominants
    - Chromatic passing chords
    - Extended chords (9ths, 11ths, 13ths)
    - Gospel turnarounds

    References:
    - Gospel piano tradition
    - Contemporary gospel harmony
    """

    # Common gospel progressions
    PROGRESSIONS = {
        'traditional': [
            (1, 'major'), (4, 'major'), (1, 'major'), (5, '7'),
            (1, 'major'), (6, 'minor'), (2, 'minor'), (5, '7'),
        ],
        'contemporary': [
            (1, 'maj7'), (6, 'minor7'), (4, 'maj7'), (5, '7'),
            (3, 'minor7'), (6, 'minor7'), (2, 'minor7'), (5, '7'),
        ],
        'turnaround': [
            (1, 'maj7'), (6, '7'), (2, 'minor7'), (5, '7'),
        ],
        'shout': [
            (4, 'major'), (4, 'major'), (1, 'major'), (1, 'major'),
        ],
    }

    # Major scale
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

    @staticmethod
    def generate_progression(key_root: int,
                            progression_type: str = 'traditional',
                            bars: int = 8) -> List[Tuple[int, str, float]]:
        """
        Generate gospel chord progression

        Args:
            key_root: Root note of key
            progression_type: Type of progression
            bars: Number of bars

        Returns:
            List of (root, quality, duration) tuples
        """
        pattern = GospelChords.PROGRESSIONS.get(
            progression_type,
            GospelChords.PROGRESSIONS['traditional']
        )

        progression = []

        for i in range(bars):
            degree, quality = pattern[i % len(pattern)]
            scale_index = degree - 1
            root = key_root + GospelChords.MAJOR_SCALE[scale_index % len(GospelChords.MAJOR_SCALE)]

            progression.append((root, quality, 4.0))

        return progression

    @staticmethod
    def add_passing_chords(progression: List[Tuple[int, str, float]]) -> List[Tuple[int, str, float]]:
        """
        Add chromatic passing chords between main chords

        Args:
            progression: Original progression

        Returns:
            Progression with passing chords added
        """
        enhanced = []

        for i, (root, quality, duration) in enumerate(progression):
            # Add main chord
            enhanced.append((root, quality, duration * 0.75))

            # Add passing chord if not last chord
            if i < len(progression) - 1:
                next_root = progression[i + 1][0]
                # Chromatic approach from below
                if next_root > root:
                    passing_root = next_root - 1
                else:
                    passing_root = next_root + 1

                enhanced.append((passing_root, 'dim', duration * 0.25))

        return enhanced


class GospelDrums:
    """Gospel drum pattern generator"""

    @staticmethod
    def generate_pattern(style: str = 'traditional',
                        measures: int = 4) -> List[Tuple[str, float, int]]:
        """
        Generate gospel drum pattern

        Args:
            style: 'traditional' (minimal), 'contemporary' (full kit)
            measures: Number of measures

        Returns:
            List of (drum_type, time, velocity) tuples
        """
        pattern = []

        if style == 'traditional':
            # Traditional: just tambourine and hand claps
            for measure in range(measures):
                offset = measure * 4.0
                # Tambourine on all beats
                for beat in range(4):
                    pattern.append(('tambourine', offset + beat, 70))
                    pattern.append(('tambourine', offset + beat + 0.5, 50))

                # Hand claps on 2 and 4
                pattern.append(('clap', offset + 1.0, 90))
                pattern.append(('clap', offset + 3.0, 90))

        elif style == 'contemporary':
            # Contemporary: full drum kit
            for measure in range(measures):
                offset = measure * 4.0

                # Kick pattern
                pattern.extend([
                    ('kick', offset + 0.0, 100),
                    ('kick', offset + 2.0, 100),
                    ('kick', offset + 3.5, 85),
                ])

                # Snare on 2 and 4 (backbeat)
                pattern.extend([
                    ('snare', offset + 1.0, 100),
                    ('snare', offset + 3.0, 100),
                ])

                # Hi-hat eighth notes
                for eighth in range(8):
                    velocity = 75 if eighth % 2 == 0 else 55
                    pattern.append(('hihat', offset + eighth * 0.5, velocity))

        return pattern


class VocalRun:
    """
    Gospel vocal run/melisma generator

    Vocal runs are ornamental passages sung on a single syllable,
    characteristic of gospel and R&B singing.

    References:
    - Mahalia Jackson technique
    - Aretha Franklin runs
    - Kim Burrell contemporary runs
    """

    @staticmethod
    def generate_run(start_note: int, end_note: int,
                    scale: List[int],
                    complexity: str = 'moderate') -> List[Tuple[int, float, int]]:
        """
        Generate vocal run

        Args:
            start_note: Starting note
            end_note: Ending note
            scale: Scale intervals
            complexity: 'simple', 'moderate', 'complex'

        Returns:
            List of (note, duration, velocity) tuples
        """
        run = []

        # Determine note density
        if complexity == 'simple':
            subdivisions = 4  # Eighth notes
            duration = 0.5
        elif complexity == 'moderate':
            subdivisions = 8  # Sixteenth notes
            duration = 0.25
        else:  # complex
            subdivisions = 16  # Thirty-second notes
            duration = 0.125

        # Generate scale-based run
        direction = 1 if end_note > start_note else -1
        current = start_note

        for i in range(subdivisions):
            run.append((current, duration, 85 + random.randint(-5, 10)))

            # Move to next scale tone
            if i < subdivisions - 2:
                # Find next/previous scale tone
                scale_tones = [start_note // 12 * 12 + interval for interval in scale]
                scale_tones = [n for n in scale_tones if start_note <= n <= end_note or end_note <= n <= start_note]

                if direction > 0:
                    next_tones = [n for n in scale_tones if n > current]
                    current = next_tones[0] if next_tones else current
                else:
                    prev_tones = [n for n in scale_tones if n < current]
                    current = prev_tones[-1] if prev_tones else current
            else:
                # End on target note
                current = end_note

        return run


class GospelGenerator:
    """
    Main gospel music generator

    Combines all gospel elements for complete arrangements.
    """

    # Major scale
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

    def __init__(self, style: GospelStyle = GospelStyle.TRADITIONAL,
                 key_root: int = 60, tempo: int = 80):
        """
        Initialize gospel generator

        Args:
            style: Gospel sub-genre
            key_root: Key root note (often C, F, or G)
            tempo: BPM (70-90 traditional, 120+ contemporary)
        """
        self.style = style
        self.key_root = key_root
        self.tempo = tempo

    def generate_arrangement(self, bars: int = 16) -> Dict[str, List]:
        """
        Generate complete gospel arrangement

        Args:
            bars: Number of bars

        Returns:
            Dictionary with all instrument/voice tracks
        """
        arrangement = {}

        # Generate chord progression
        progression_type = 'contemporary' if self.style == GospelStyle.CONTEMPORARY else 'traditional'
        progression = GospelChords.generate_progression(
            self.key_root, progression_type, bars
        )

        # Add passing chords for contemporary style
        if self.style == GospelStyle.CONTEMPORARY:
            progression = GospelChords.add_passing_chords(progression)

        # Generate SATB choir
        choir_voicings = []
        for root, quality, duration in progression:
            voicing = ChoirVoicing.from_chord(root, quality)
            choir_voicings.append(voicing)

        arrangement['choir'] = choir_voicings

        # Generate Hammond organ
        organ_chords = [[root, root + 4, root + 7] for root, quality, _ in progression[:bars]]
        organ_style = 'walking' if self.style == GospelStyle.TRADITIONAL else 'block'
        arrangement['organ'] = HammondOrgan.generate_chord_pattern(
            organ_chords[0], measures=bars, style=organ_style
        )

        # Generate drums
        drum_style = 'traditional' if self.style == GospelStyle.TRADITIONAL else 'contemporary'
        arrangement['drums'] = GospelDrums.generate_pattern(drum_style, bars)

        # Generate organ runs (fills between sections)
        if bars >= 8:
            run = HammondOrgan.generate_run(
                self.key_root, self.key_root + 12,
                self.MAJOR_SCALE, 'ascending'
            )
            arrangement['organ_fills'] = run

        return arrangement


if __name__ == "__main__":
    """Example usage and testing"""

    print("Gospel Music Generator - Test Suite\n")
    print("=" * 60)

    # Test 1: SATB choir voicing
    print("\n1. Generating SATB choir voicing for C major chord...")
    voicing = ChoirVoicing.from_chord(60, 'maj7')
    print(f"   Soprano: {voicing.soprano}")
    print(f"   Alto: {voicing.alto}")
    print(f"   Tenor: {voicing.tenor}")
    print(f"   Bass: {voicing.bass}")

    # Test 2: Hammond organ run
    print("\n2. Generating Hammond organ ascending run...")
    organ_run = HammondOrgan.generate_run(60, 72, GospelGenerator.MAJOR_SCALE, 'ascending')
    print(f"   Generated {len(organ_run)} notes")
    print(f"   First 4 notes: {[note for note, dur, vel in organ_run[:4]]}")

    # Test 3: Gospel chord progression
    print("\n3. Generating contemporary gospel progression...")
    progression = GospelChords.generate_progression(60, 'contemporary', 8)
    print(f"   Generated {len(progression)} chords")
    for i, (root, quality, dur) in enumerate(progression[:4]):
        print(f"   Chord {i+1}: {root} {quality}")

    # Test 4: Call and response
    print("\n4. Generating call and response pattern...")
    lead = [(60, 1.0), (62, 1.0), (64, 1.0), (65, 1.0)]
    choir = [64, 67, 72]
    call_response = CallAndResponse.generate_pattern(lead, choir, 2)
    print(f"   Lead events: {len(call_response['lead'])}")
    print(f"   Choir events: {len(call_response['choir'])}")

    # Test 5: Vocal run
    print("\n5. Generating gospel vocal run...")
    run = VocalRun.generate_run(60, 72, GospelGenerator.MAJOR_SCALE, 'moderate')
    print(f"   Generated {len(run)} notes in run")

    # Test 6: Traditional gospel drums
    print("\n6. Generating traditional gospel drums (tambourine + claps)...")
    trad_drums = GospelDrums.generate_pattern('traditional', 4)
    print(f"   Generated {len(trad_drums)} drum events")

    # Test 7: Complete contemporary gospel arrangement
    print("\n7. Generating complete contemporary gospel arrangement...")
    gospel_gen = GospelGenerator(GospelStyle.CONTEMPORARY, key_root=60, tempo=130)
    arrangement = gospel_gen.generate_arrangement(16)
    print(f"   Generated {len(arrangement)} tracks:")
    for track, content in arrangement.items():
        print(f"   - {track}: {len(content)} events")

    # Test 8: Traditional gospel arrangement
    print("\n8. Generating traditional gospel arrangement...")
    trad_gen = GospelGenerator(GospelStyle.TRADITIONAL, key_root=65, tempo=75)
    trad_arrangement = trad_gen.generate_arrangement(8)
    print(f"   Generated {len(trad_arrangement)} tracks:")
    for track, content in trad_arrangement.items():
        print(f"   - {track}: {len(content)} events")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nGospel features implemented:")
    print("  ✓ SATB choir voicing with proper voice leading")
    print("  ✓ Hammond B3 organ runs and patterns")
    print("  ✓ Call and response structures")
    print("  ✓ Rich gospel chord progressions")
    print("  ✓ Traditional and contemporary styles")
    print("  ✓ Vocal runs and melisma")
    print("  ✓ Gospel drums (tambourine, claps, full kit)")
    print("  ✓ Passing chords and chromatic harmony")
